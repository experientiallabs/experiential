"""Gold-assertion judging: did the agent's run satisfy the task's success conditions?

Closed-loop eval scores *task success*, not per-step fidelity (docs/reference/closed_loop.md).
Success is defined by the task's `gold` assertions — semantic post-conditions checked against the
transcript by an LLM judge, so they are robust to wording (exact-match rules systematically
under-report success on semantically-correct answers). The verdict is always scored against the FULL
gold list: a truncated judge reply that omits assertions cannot report success.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.providers.base import Message, Provider

GOLD_JUDGE_MARKER = "grade whether an agent completed a task"

GOLD_JUDGE_SYSTEM = """You grade whether an agent completed a task. You are given the task, the
agent's final answer, a transcript of what it did, and a list of GOLD assertions defining success.
For each assertion, decide whether the transcript+answer show it is satisfied. Judge by MEANING, not
wording: an assertion holds if the evidence shows the described outcome, even if phrased otherwise.
Do not give credit for merely attempting or claiming success without supporting evidence in the
transcript.

Respond with ONLY a JSON object, no prose:
{"assertions": [{"assertion": "<verbatim>", "passed": <bool>, "why": "<short reason>"}],
 "passed": <bool: true iff every assertion passed>}"""


class AssertionResult(BaseModel):
    assertion: str
    passed: bool
    why: str = ""


class GoldVerdict(BaseModel):
    """The judge's verdict on one run against its gold assertions."""

    passed: bool = False  # every assertion satisfied
    fraction: float = 0.0  # fraction of assertions satisfied (partial-credit signal)
    assertions: list[AssertionResult] = Field(default_factory=list)
    rationale: str = ""

    @classmethod
    def trivially_passed(cls) -> GoldVerdict:
        """A run with no gold assertions cannot fail its (empty) spec; treat as passed."""
        return cls(passed=True, fraction=1.0, rationale="no gold assertions")


class _RawVerdict(BaseModel):
    assertions: list[AssertionResult] = Field(default_factory=list)
    passed: bool = False


class GoldJudge:
    """LLM judge that checks a run transcript against a task's gold assertions."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    def score(self, instruction: str, answer: str, transcript: str, gold: list[str]) -> GoldVerdict:
        if not gold:
            return GoldVerdict.trivially_passed()
        user = _build_prompt(instruction, answer, transcript, gold)
        completion = self._provider.complete(
            GOLD_JUDGE_SYSTEM,
            [Message(role="user", content=user)],
            temperature=0.0,
            max_tokens=1024,
        )
        return _parse(completion.text, gold)


def _build_prompt(instruction: str, answer: str, transcript: str, gold: list[str]) -> str:
    assertions = "\n".join(f"- {g}" for g in gold)
    return (
        f"TASK:\n{instruction}\n\n"
        f"AGENT FINAL ANSWER:\n{answer or '(none)'}\n\n"
        f"TRANSCRIPT:\n{transcript or '(empty)'}\n\n"
        f"GOLD ASSERTIONS (all must hold for success):\n{assertions}\n"
    )


def _parse(text: str, gold: list[str]) -> GoldVerdict:
    """Parse the judge reply; fall back to a failed verdict if it is unusable."""
    raw = extract_json_object(text)
    if raw is not None:
        try:
            parsed = _RawVerdict.model_validate_json(raw)
        except ValidationError:
            parsed = None
        if parsed is not None and parsed.assertions:
            # Score against ALL required gold assertions, matched BY TEXT to what the judge echoed
            # back (the prompt demands verbatim echoes). A truncated reply that omits an assertion,
            # or one that duplicates a passing assertion to pad the count, cannot report success:
            # every unmatched gold assertion counts as failed. Fail-closed by construction.
            passed_texts = {a.assertion.strip() for a in parsed.assertions if a.passed}
            total = len(gold)
            n_pass = sum(1 for g in gold if g.strip() in passed_texts)
            return GoldVerdict(
                passed=parsed.passed and n_pass == total,
                fraction=n_pass / total if total else 1.0,
                assertions=parsed.assertions,
                rationale=f"{n_pass}/{total} assertions satisfied",
            )
    return GoldVerdict(
        passed=False,
        fraction=0.0,
        rationale=f"unparseable judge response; treated as failure. raw: {text.strip()[:200]}",
    )
