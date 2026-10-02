"""Run the live-model reviewer quality corpus and emit machine-readable metrics."""

from __future__ import annotations

import argparse
import difflib
import json
import os
from pathlib import Path

import yaml

from evals.judge import judge_finding
from roborak.analysis.reviewer import Reviewer
from roborak.context.chunker import ChunkStrategy
from roborak.context.diff import parse_diff
from roborak.core.config import Config
from roborak.core.models import ChangeSet, Finding
from roborak.core.severity import Evidence, Severity
from roborak.core.verdict import blocking_findings
from roborak.llm.client import LLMClient
from roborak.render.markdown import Form, finding_markdown

ROOT = Path(__file__).parent


def synthetic_diff(path: str, before: str, after: str) -> str:
    body = "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )
    return f"diff --git a/{path} b/{path}\n{body}\n"


def score(rows: list[dict[str, object]]) -> dict[str, float | int]:
    defects = [row for row in rows if row["expected_category"]]
    matched = [row for row in defects if row["matched"]]

    # The evidence policy is a trade, so both halves are measured together: a run
    # that stops blocking on guesses by also refusing to block on real defects has
    # not improved anything.
    controls = [row for row in rows if row.get("expect_blocker") is False]
    provable = [row for row in rows if row.get("expect_blocker") is True]

    # The controls are *meant* to draw a nonblocking finding, so they are not
    # false positives; only the cases expected to stay silent are.
    clean = [
        row
        for row in rows
        if not row["expected_category"] and row.get("expect_blocker") is not False
    ]

    # A finding is only as good as what the reader can act on, so the representative
    # cases that carry a judge verdict are scored on the rubric too -- the mean pass
    # rate across every graded clarity/support check. A judge-tagged case that matched
    # always records a ``judge`` entry: a dict when the judge answered, ``None`` when
    # it could not. ``judge_completion`` keeps a run where every judge call failed from
    # passing by default -- an unassessed review is not a clean one.
    attempts = [row for row in rows if "judge" in row]
    graded = [verdict for row in attempts if isinstance(verdict := row["judge"], dict)]
    checks = [bool(passed) for verdict in graded for passed in verdict.values()]

    return {
        "cases": len(rows),
        "judge_attempts": len(attempts),
        "judged": len(graded),
        "judge_completion": len(graded) / len(attempts) if attempts else 1.0,
        "finding_quality": sum(checks) / len(checks) if checks else 1.0,
        "recall": len(matched) / len(defects) if defects else 1.0,
        "clean_false_positive_rate": (
            sum(bool(row["findings"]) for row in clean) / len(clean) if clean else 0.0
        ),
        "unproven_blocker_rate": (
            sum(bool(row["blockers"]) for row in controls) / len(controls) if controls else 0.0
        ),
        "blocker_recall": (
            sum(bool(row["matched_blocker"]) for row in provable) / len(provable)
            if provable
            else 1.0
        ),
        "anchor_accuracy": (
            sum(bool(row["exact_anchor"]) for row in matched) / len(matched) if matched else 0.0
        ),
        "parse_success": sum(not row["errors"] for row in rows) / len(rows) if rows else 1.0,
        "tokens": sum(int(row["tokens"]) for row in rows),
    }


def compare_chunking(
    baseline_rows: list[dict[str, object]], semantic_rows: list[dict[str, object]]
) -> dict[str, object]:
    """Compare the new planner with the retained directory/language baseline."""
    baseline = score(baseline_rows)
    semantic = score(semantic_rows)
    return {
        "baseline": baseline,
        "semantic": semantic,
        "recall_delta": float(semantic["recall"]) - float(baseline["recall"]),
        "clean_false_positive_rate_delta": float(semantic["clean_false_positive_rate"])
        - float(baseline["clean_false_positive_rate"]),
    }


def _diff_text(case: dict[str, object]) -> str:
    raw_files = case.get("files")
    if isinstance(raw_files, list):
        return "".join(
            synthetic_diff(str(file["path"]), str(file["before"]), str(file["after"]))
            for file in raw_files
            if isinstance(file, dict)
        )
    return synthetic_diff(str(case["path"]), str(case["before"]), str(case["after"]))


def _changeset(case: dict[str, object]) -> ChangeSet:
    return ChangeSet(files=parse_diff(_diff_text(case)), title=str(case["id"]))


def _evaluate(
    case: dict[str, object],
    config: Config,
    *,
    strategy: ChunkStrategy = "semantic",
    judge: LLMClient | None = None,
) -> dict[str, object]:
    result = Reviewer(
        config=config,
        repo=ROOT.parent,
        llm=LLMClient(config.llm),
        chunk_strategy=strategy,
    ).review(_changeset(case))
    expected = case.get("expected_category")
    expected_file = str(case.get("expected_file") or "")
    line = int(case.get("expected_line") or 0)
    candidates = [
        finding
        for finding in result.findings
        if finding.category.value == expected
        and (not expected_file or finding.file == expected_file)
    ]
    blockers = blocking_findings(result, Severity.MAJOR)
    near = [finding for finding in candidates if abs(finding.start_line - line) <= 3]
    row: dict[str, object] = {
        "id": case["id"],
        "expected_category": expected,
        "expect_blocker": case.get("expect_blocker"),
        "findings": len(result.findings),
        "blockers": len(blockers),
        "matched": bool(near),
        "matched_blocker": any(any(blocker is finding for blocker in blockers) for finding in near),
        "exact_anchor": any(finding.start_line == line for finding in candidates),
        "errors": result.errors,
        "tokens": result.tokens_used,
    }
    # Only representative cases carry ``judge: true``; grade the finding the reader
    # would actually see, not the model's raw candidate, so the rubric judges the
    # rendered prose. An unmatched case has nothing to grade -- recall already
    # records the miss.
    if judge is not None and case.get("judge") and near:
        row["judge"] = _judge_row(judge, diff=_diff_text(case), finding=near[0])
    return row


def _judge_row(judge: LLMClient, *, diff: str, finding: Finding) -> dict[str, bool] | None:
    return judge_finding(
        judge,
        diff=diff,
        rendered_finding=finding_markdown(finding, form=Form.PUBLISHED),
        evidence_unverified=finding.evidence is Evidence.UNVERIFIED,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("ROBORAK_EVAL_MODEL"))
    parser.add_argument("--judge-model", default=os.getenv("ROBORAK_EVAL_JUDGE_MODEL"))
    parser.add_argument("--output", type=Path, default=ROOT / "eval-results.json")
    args = parser.parse_args()

    config = Config()
    if args.model:
        config.llm.model = args.model
    config.output.walkthrough = False
    config.static.enabled = False

    # The judge grades prose, so it runs deterministically on its own config: a
    # separate model when one is given, the review model otherwise, always at
    # temperature 0 to keep the clarity metric from drifting run to run.
    judge_config = config.llm.model_copy(deep=True)
    if args.judge_model:
        judge_config.model = args.judge_model
    judge_config.temperature = 0.0
    judge = LLMClient(judge_config)

    cases = yaml.safe_load((ROOT / "cases.yaml").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []

    for case in cases:
        rows.append(_evaluate(case, config, judge=judge))

    metrics = score(rows)
    chunking_cases = yaml.safe_load((ROOT / "chunking_cases.yaml").read_text(encoding="utf-8"))
    baseline_rows: list[dict[str, object]] = []
    semantic_rows: list[dict[str, object]] = []
    for case in chunking_cases:
        case_config = config.model_copy(deep=True)
        case_config.llm.context_budget = int(case.get("context_budget") or 80)
        baseline_rows.append(_evaluate(case, case_config, strategy="directory"))
        semantic_rows.append(_evaluate(case, case_config, strategy="semantic"))
    chunking = compare_chunking(baseline_rows, semantic_rows)
    args.output.write_text(
        json.dumps(
            {
                "model": config.model,
                "metrics": metrics,
                "cases": rows,
                "chunking_comparison": chunking,
                "chunking_cases": {
                    "baseline": baseline_rows,
                    "semantic": semantic_rows,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"metrics": metrics, "chunking_comparison": chunking}, indent=2))
    baseline_metrics = chunking["baseline"]
    semantic_metrics = chunking["semantic"]
    assert isinstance(baseline_metrics, dict) and isinstance(semantic_metrics, dict)
    return int(
        metrics["recall"] < 0.80
        or metrics["clean_false_positive_rate"] > 0.10
        or metrics["unproven_blocker_rate"] > 0.10
        or metrics["blocker_recall"] < 0.80
        or metrics["anchor_accuracy"] < 0.95
        or metrics["parse_success"] < 0.99
        # A judge that never answered cannot vouch for the prose. If the run tried to
        # grade representative cases, most of those attempts must have come back, and
        # what came back must clear the quality bar -- a skipped judge is a failed
        # gate, not a free pass.
        or (int(metrics["judge_attempts"]) > 0 and metrics["judge_completion"] < 0.80)
        or (int(metrics["judged"]) > 0 and metrics["finding_quality"] < 0.80)
        or float(semantic_metrics["recall"]) < float(baseline_metrics["recall"])
        or float(semantic_metrics["clean_false_positive_rate"])
        > float(baseline_metrics["clean_false_positive_rate"])
    )


if __name__ == "__main__":
    raise SystemExit(main())
