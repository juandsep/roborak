"""An LLM rubric that grades a rendered finding for trigger, consequence and fix.

Eval-only, deliberately outside ``src/``: this judge prompt never ships with the
reviewer. The reviewer-quality corpus already checks that a finding lands on the
right category and line; this adds the second half of issue #95 -- whether the
finding the reader actually sees *says* what breaks, what it costs, and how to fix
it, and does so without claiming a reproduction it never ran.

A judge that cannot answer (an unparseable reply, a model failure) returns ``None``
rather than a pass. An inconclusive check is never recorded as a passing one, the
same bar the reviewer's own investigation stage holds itself to.
"""

from __future__ import annotations

import yaml

from roborak.llm.client import LLMClient, LLMError

JUDGE_SYSTEM = """\
You grade one code-review finding for clarity and factual support. You are given the
diff that was reviewed and the finding as the author will read it. Judge only what the
finding says against what the diff shows; do not review the code yourself.

Answer each question with true or false:

- states_trigger: does the finding name the concrete input or condition that provokes
  the problem, not merely that a problem exists?
- states_consequence: does it name the observable wrong behaviour -- what goes wrong
  for a real input or at runtime?
- states_fix: does it give a practical fix direction, or name the specific thing to
  verify? A vague "be careful" does not count.
- faithful: is every concrete claim supported by the diff, and -- when the finding is
  labelled unverified -- does it read as reasoning rather than a reproduction it
  actually ran?

Respond with YAML only, no prose and no code fences:

states_trigger: <true|false>
states_consequence: <true|false>
states_fix: <true|false>
faithful: <true|false>
"""

JUDGE_FIELDS = ("states_trigger", "states_consequence", "states_fix", "faithful")


def build_judge_prompt(*, diff: str, rendered_finding: str, evidence_unverified: bool) -> str:
    """The user message pairing the reviewed diff with the finding under grading."""
    label = (
        "The finding is labelled UNVERIFIED (reasoning only)."
        if evidence_unverified
        else "The finding claims verified evidence."
    )
    return f"{label}\n\n# Diff\n\n{diff}\n\n# Finding\n\n{rendered_finding}\n"


def parse_judge_reply(text: str) -> dict[str, bool] | None:
    """Coerce a judge reply into booleans, or ``None`` when it cannot be trusted.

    A missing or non-boolean field fails the whole verdict: a judge that answered
    only half the rubric has not graded the finding, and guessing the rest would be
    the recorded-a-pass-we-did-not-make mistake this module exists to avoid.
    """
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(loaded, dict):
        return None
    verdict: dict[str, bool] = {}
    for field in JUDGE_FIELDS:
        value = loaded.get(field)
        if not isinstance(value, bool):
            return None
        verdict[field] = value
    return verdict


def judge_finding(
    llm: LLMClient, *, diff: str, rendered_finding: str, evidence_unverified: bool
) -> dict[str, bool] | None:
    """Grade one rendered finding, or ``None`` if the judge could not answer."""
    user = build_judge_prompt(
        diff=diff,
        rendered_finding=rendered_finding,
        evidence_unverified=evidence_unverified,
    )
    try:
        response = llm.complete(JUDGE_SYSTEM, user)
    except LLMError:
        return None
    return parse_judge_reply(response.text)
