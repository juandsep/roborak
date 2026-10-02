from evals.judge import parse_judge_reply
from evals.run import compare_chunking, score


def test_eval_metrics_are_computed_from_case_outcomes():
    rows = [
        {
            "expected_category": "bug",
            "matched": True,
            "exact_anchor": True,
            "findings": 1,
            "blockers": 1,
            "errors": [],
            "tokens": 10,
        },
        {
            "expected_category": None,
            "matched": False,
            "exact_anchor": False,
            "findings": 0,
            "blockers": 0,
            "errors": [],
            "tokens": 5,
        },
    ]
    metrics = score(rows)
    assert metrics["recall"] == 1.0
    assert metrics["clean_false_positive_rate"] == 0.0
    assert metrics["anchor_accuracy"] == 1.0
    assert metrics["parse_success"] == 1.0
    assert metrics["tokens"] == 15


def _row(*, expect_blocker: bool, blockers: int) -> dict[str, object]:
    return {
        "expected_category": "bug" if expect_blocker else None,
        "expect_blocker": expect_blocker,
        "matched": expect_blocker,
        "matched_blocker": expect_blocker and bool(blockers),
        "exact_anchor": expect_blocker,
        "findings": blockers,
        "blockers": blockers,
        "errors": [],
        "tokens": 1,
    }


def test_the_evidence_metrics_measure_both_halves_of_the_trade():
    """Blocking on nothing scores perfectly on one metric and fails the other."""
    metrics = score(
        [
            _row(expect_blocker=False, blockers=1),
            _row(expect_blocker=False, blockers=0),
            _row(expect_blocker=True, blockers=1),
            _row(expect_blocker=True, blockers=0),
        ]
    )
    assert metrics["unproven_blocker_rate"] == 0.5
    assert metrics["blocker_recall"] == 0.5


def test_blocker_recall_needs_the_blocker_to_be_the_expected_defect():
    """An unrelated major finding cannot stand in for the defect the case tests."""
    row = _row(expect_blocker=True, blockers=1)
    row["matched_blocker"] = False
    assert score([row])["blocker_recall"] == 0.0


def test_nonblocking_controls_are_not_counted_as_clean_false_positives():
    """The controls are meant to draw a finding; only silence-expected cases aren't."""
    metrics = score([_row(expect_blocker=False, blockers=0) | {"findings": 1}])
    assert metrics["clean_false_positive_rate"] == 0.0


def test_rows_without_a_blocker_label_are_left_out_of_both_metrics():
    """The 30 original cases predate the policy and must not skew it."""
    metrics = score(
        [
            {
                "expected_category": "bug",
                "matched": True,
                "exact_anchor": True,
                "findings": 1,
                "blockers": 1,
                "errors": [],
                "tokens": 1,
            }
        ]
    )
    assert metrics["unproven_blocker_rate"] == 0.0
    assert metrics["blocker_recall"] == 1.0


def _judged(**verdict: bool) -> dict[str, object]:
    return {
        "expected_category": "bug",
        "matched": True,
        "exact_anchor": True,
        "findings": 1,
        "blockers": 0,
        "errors": [],
        "tokens": 1,
        "judge": verdict,
    }


def test_finding_quality_is_the_mean_pass_rate_over_judged_checks():
    """A perfect verdict and one with a single failed check average across checks."""
    metrics = score(
        [
            _judged(states_trigger=True, states_consequence=True, states_fix=True, faithful=True),
            _judged(states_trigger=True, states_consequence=True, states_fix=True, faithful=False),
        ]
    )
    assert metrics["judged"] == 2
    assert metrics["finding_quality"] == 0.875


def test_rows_without_a_judge_verdict_do_not_affect_finding_quality():
    """Ungraded rows (no judge, or a judge that could not answer) are left out."""
    ungraded = {
        "expected_category": "bug",
        "matched": True,
        "exact_anchor": True,
        "findings": 1,
        "blockers": 0,
        "errors": [],
        "tokens": 1,
    }
    unanswered = ungraded | {"judge": None}
    metrics = score([ungraded, unanswered])
    assert metrics["judged"] == 0
    assert metrics["finding_quality"] == 1.0


def test_chunking_comparison_reports_recall_and_false_positive_deltas():
    defect = _row(expect_blocker=True, blockers=1)
    missed = defect | {"matched": False, "matched_blocker": False, "blockers": 0}
    clean = {
        "expected_category": None,
        "matched": False,
        "exact_anchor": False,
        "findings": 0,
        "blockers": 0,
        "errors": [],
        "tokens": 1,
    }
    noisy = clean | {"findings": 1}

    comparison = compare_chunking([missed, noisy], [defect, clean])

    assert comparison["recall_delta"] == 1.0
    assert comparison["clean_false_positive_rate_delta"] == -1.0


def test_judge_reply_parses_a_full_boolean_verdict():
    verdict = parse_judge_reply(
        "states_trigger: true\nstates_consequence: true\nstates_fix: false\nfaithful: true\n"
    )
    assert verdict == {
        "states_trigger": True,
        "states_consequence": True,
        "states_fix": False,
        "faithful": True,
    }


def test_judge_reply_is_untrusted_when_a_field_is_missing_or_unparseable():
    """A half-answered or malformed reply graded nothing -- it must not pass by default."""
    missing = parse_judge_reply("states_trigger: true\nstates_consequence: true\n")
    assert missing is None
    non_boolean = parse_judge_reply(
        "states_trigger: maybe\nstates_consequence: true\nstates_fix: true\nfaithful: true\n"
    )
    assert non_boolean is None
    assert parse_judge_reply("not: [valid") is None
    assert parse_judge_reply("just a sentence") is None
