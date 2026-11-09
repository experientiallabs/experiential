"""Decide whether a replayed tool observation conveys the same outcome as the recorded one.

A reconstructed start replays a source trace's recorded prefix through the simulated world.
The simulator writes plausible tool output, never the recorded bytes: listings carry other
dates and owners, tracebacks are abridged, version strings shortened. Byte equality refused
starts that were right in substance, so a step is matched on substance, in order:

1. exact: equal observations and equal error outcome;
2. exit status: when both observations state an exit code they must be equal;
3. error outcome: the recorded and replayed error flags must agree;
4. normalised output: whitespace collapsed; paths' volatile parts, hex ids, timestamps,
   durations, PIDs and numbers masked; the token sets of the salient lines compared by Jaccard
   similarity. At or above :data:`PASS_SIMILARITY` the step passes, below
   :data:`JUDGE_SIMILARITY` it fails.

What normalisation cannot decide (a similarity between the two bounds, or an exit code stated on
only one side) is left ``undecided`` for an open-weight judge to settle, one step at a time.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from exp.common.core.artifacts import ContractModel, JsonValue

PASS_SIMILARITY = 0.6
"""Token-set Jaccard similarity of salient lines at or above which a step passes."""

JUDGE_SIMILARITY = 0.35
"""Similarity below which a step fails; between the bounds the judge decides."""

MatchRule = Literal["exact", "normalised", "judge"]
Decision = Literal["pass", "fail", "undecided"]

_EXIT = re.compile(
    r"(?:exited with (?:exit )?code|exit code|exit status|returned non-zero exit status)"
    r"[\s:=]*(-?\d+)",
    re.IGNORECASE,
)
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
_ISO = r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"
_UNITS = r"(?:ms|us|µs|ns|s|sec|secs|seconds|minutes|min)"
_MASKS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(_ISO), "<time>"),
    (re.compile(rf"\b{_MONTH}\s+\d{{1,2}}\s+(?:\d{{2}}:\d{{2}}|\d{{4}})\b"), "<time>"),
    (re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"), "<time>"),
    (re.compile(rf"\b\d+(?:\.\d+)?\s?{_UNITS}\b"), "<duration>"),
    (re.compile(r"\bpid[\s:=]*\d+\b", re.IGNORECASE), "pid <pid>"),
    (re.compile(r"\b(?:0x)?[0-9a-f]{7,}\b", re.IGNORECASE), "<hex>"),
    (re.compile(r"(?<=/)[^/\s]*\d[^/\s]*(?=/)"), "<seg>"),
    (re.compile(r"\b\d+(?:\.\d+)+\b"), "<num>"),
    (re.compile(r"\b\d+\b"), "<n>"),
)
_TOKEN = re.compile(r"<[a-z]+>|[A-Za-z_][A-Za-z0-9_.\-]*")


class StepMatch(ContractModel):
    """How a replayed step compares with its recording.

    Attributes:
        decision: ``pass``, ``fail``, or ``undecided`` when only a judge can settle it.
        rule: The rule that passed the step, when one did.
        similarity: Token-set Jaccard similarity of the normalised salient lines.
        recorded_exit: Exit code the recorded observation states, if any.
        replayed_exit: Exit code the replayed observation states, if any.
        reason: Why the step passed, failed or is undecided.
    """

    decision: Decision
    rule: MatchRule | None = None
    similarity: float | None = None
    recorded_exit: int | None = None
    replayed_exit: int | None = None
    reason: str


def observation_text(value: JsonValue) -> str:
    """The observation as text: strings as they are, anything else as sorted JSON."""
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def exit_code(text: str) -> int | None:
    """The last exit code an observation states, or None when it states none."""
    found = _EXIT.findall(text)
    return int(found[-1]) if found else None


def salient_tokens(text: str) -> frozenset[str]:
    """The token set of an observation's non-empty lines after masking volatile values."""
    tokens: set[str] = set()
    for line in text.splitlines():
        line = " ".join(line.split())
        if not line:
            continue
        for pattern, mask in _MASKS:
            line = pattern.sub(mask, line)
        tokens.update(token.lower() for token in _TOKEN.findall(line))
    return frozenset(tokens)


def similarity(recorded: str, replayed: str) -> float:
    """Token-set Jaccard similarity of two observations' normalised salient lines."""
    left, right = salient_tokens(recorded), salient_tokens(replayed)
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def match_step(
    recorded: JsonValue, recorded_error: bool, replayed: JsonValue, replayed_error: bool
) -> StepMatch:
    """Compare one replayed step with its recording on substance.

    Args:
        recorded: The recorded observation.
        recorded_error: Whether the recorded call failed.
        replayed: The replayed observation.
        replayed_error: Whether the replayed call failed.

    Returns:
        The decision, the rule that passed it and the scores.
    """
    if recorded == replayed and recorded_error == replayed_error:
        return StepMatch(decision="pass", rule="exact", similarity=1.0, reason="identical")
    left, right = observation_text(recorded), observation_text(replayed)
    codes = {"recorded_exit": exit_code(left), "replayed_exit": exit_code(right)}
    score = round(similarity(left, right), 4)
    known = (codes["recorded_exit"] is not None) + (codes["replayed_exit"] is not None)
    if known == 2 and codes["recorded_exit"] != codes["replayed_exit"]:
        return StepMatch(decision="fail", similarity=score, reason="the exit codes differ", **codes)
    if recorded_error != replayed_error:
        if known == 1:
            return StepMatch(
                decision="undecided",
                similarity=score,
                reason="the error outcomes differ and only one side states an exit code",
                **codes,
            )
        return StepMatch(
            decision="fail", similarity=score, reason="one call failed and the other did not",
            **codes,
        )  # fmt: skip
    if score < JUDGE_SIMILARITY:
        return StepMatch(
            decision="fail", similarity=score, reason="the outputs do not agree", **codes
        )
    if known == 1:
        return StepMatch(
            decision="undecided", similarity=score,
            reason="only one side states an exit code", **codes,
        )  # fmt: skip
    if score >= PASS_SIMILARITY:
        return StepMatch(
            decision="pass", rule="normalised", similarity=score,
            reason="same outcome; the outputs agree after normalisation", **codes,
        )  # fmt: skip
    return StepMatch(
        decision="undecided", similarity=score,
        reason="the outputs partly agree after normalisation", **codes,
    )  # fmt: skip
