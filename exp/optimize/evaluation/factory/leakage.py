"""Answer values a task requires, and whether public actor context already states them.

These are pure functions over saved evidence, shared by every world kind. A produced value is a
substantive scalar that the solution control's final state or answer contains, or that a final
verifier equality rule requires, and that the initial world state does not already contain. Values
copied from the world are obtainable by acting on it, so they are never answer leaks. A value that
every failing control's final state also contains (an output file name, a status every attempt
writes) does not distinguish a correct result, so it is not an answer value either. Values the
instruction legitimately asks for (an address to write, a status to set) are declared by the
operator in ``specified_values``; they stay visible in reports instead of being guessed away.
"""

import json
import re
from collections.abc import Iterator, Sequence

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue
from exp.common.models import ModelMessage
from exp.common.tasks import ToolSchema
from exp.simulation.evaluation.tasks import TaskVerifier

_DIGEST = re.compile(r"[0-9a-f]{32,}")
_SPACE = re.compile(r"\s+")


class ValidityRule(ContractModel):
    """Registered thresholds for answer-value extraction and closed-book recall.

    Attributes:
        minimum_text_characters: Shorter strings are too common to count as answer values.
        minimum_number_digits: Numbers with fewer digits are too common to count.
        specified_values: Values the instruction itself requests; never counted as leaked
            or as closed-book recall. They are reported, so an overbroad list stays visible.
        value_recall: Fraction of produced values a closed-book attempt must reproduce to count
            as solving the task without its world. Applies only when produced values exist.
    """

    minimum_text_characters: int = Field(default=3, ge=1, le=64, strict=True)
    minimum_number_digits: int = Field(default=3, ge=1, le=32, strict=True)
    specified_values: tuple[str | int | float, ...] = Field(default=(), max_length=256)
    value_recall: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)


def scalar_leaves(value: JsonValue) -> Iterator[str | int | float]:
    """Yield every string and number in a JSON value, skipping booleans and nulls."""
    if isinstance(value, dict):
        for item in value.values():
            yield from scalar_leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from scalar_leaves(item)
    elif isinstance(value, str) or (isinstance(value, int | float) and not isinstance(value, bool)):
        yield value


def canonical(value: str | int | float) -> str:
    """Normalize one scalar: casefolded single-spaced text, or a plain decimal number."""
    if isinstance(value, str):
        return _SPACE.sub(" ", value).strip().casefold()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _substantive(value: str | int | float, rule: ValidityRule) -> bool:
    """Keep distinctive values; short text, few-digit numbers and opaque digests are excluded."""
    text = canonical(value)
    if isinstance(value, str):
        return len(text) >= rule.minimum_text_characters and not _DIGEST.fullmatch(text)
    return sum(character.isdigit() for character in text) >= rule.minimum_number_digits


def _answer_rules(verifier: TaskVerifier) -> Iterator[str | int | float]:
    """Yield literals of equality rules over the final state or final answer."""
    for rule in verifier.rules:
        if rule.operator == "equals" and rule.path.startswith(("/final_state", "/final_answer")):
            yield from scalar_leaves(rule.value)


def _result_values(observation: JsonObject) -> set[str]:
    """Canonical scalars of an observation's final state and final answer."""
    return {
        canonical(value)
        for value in (
            *scalar_leaves(observation.get("final_state")),
            *scalar_leaves(observation.get("final_answer")),
        )
    }


def produced_values(
    verifier: TaskVerifier,
    solution: JsonObject,
    failing: Sequence[JsonObject],
    rule: ValidityRule,
) -> tuple[str, ...]:
    """Return the sorted canonical values a correct result must add beyond the initial world.

    Args:
        verifier: Candidate verifier; its final-state equality literals are required values.
        solution: Saved native observation of the passing solution control.
        failing: Saved observations of the failing controls; values all of them share do not
            distinguish a correct result.
        rule: Registered thresholds and operator-declared specified values.

    Returns:
        Canonical produced values, excluding initial-world, shared-failure and specified values.
    """
    initial = {canonical(value) for value in scalar_leaves(solution.get("initial_state"))}
    specified = {canonical(value) for value in rule.specified_values}
    shared = set.intersection(*map(_result_values, failing)) if failing else set()
    found = {
        canonical(value)
        for value in (
            *scalar_leaves(solution.get("final_state")),
            *scalar_leaves(solution.get("final_answer")),
            *_answer_rules(verifier),
        )
        if _substantive(value, rule)
    }
    return tuple(sorted(found - initial - shared - specified))


def public_text(messages: Sequence[ModelMessage], tools: Sequence[ToolSchema]) -> str:
    """Render everything the actor can read before acting: messages and tool schemas."""
    parts = [message.content or "" for message in messages]
    parts.extend(
        json.dumps(message.assistant_action.model_dump(mode="json"), sort_keys=True)
        for message in messages
        if message.assistant_action is not None
    )
    parts.extend(json.dumps(tool.model_dump(mode="json"), sort_keys=True) for tool in tools)
    return "\n".join(parts)


def _pattern(value: str) -> re.Pattern[str]:
    """Match a canonical value as a whole token, including thousands-separated numbers."""
    forms = [re.escape(value)]
    if re.fullmatch(r"-?\d{4,}", value):
        forms.append(re.escape(f"{int(value):,}"))
    return re.compile(r"(?<![\w.])(?:" + "|".join(forms) + r")(?![\w]|\.\d)")


def mentioned(values: Sequence[str], text: str) -> tuple[str, ...]:
    """Return the canonical values that appear as whole tokens in normalized text."""
    normalized = _SPACE.sub(" ", text).casefold()
    return tuple(value for value in values if _pattern(value).search(normalized))
