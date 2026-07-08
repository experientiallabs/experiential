"""NumericJudge: deterministic scoring for structured, measurement-style observations.

The semantic judge (`RubricJudge`) grades text equivalence; corpora whose observations are
measurements (robot poses, memory footprints, latencies) need exact structural comparison instead.
The judge flattens both observations' JSON content into leaf fields, scores numeric leaves by
relative error and every other leaf (bools, strings, nulls) by exact match, and reports the mean —
with per-field scores in `JudgeResult.dimensions` so callers can also threshold individual fields
(e.g. "did it predict the OOM?").
"""

from __future__ import annotations

import json
import math

from pydantic import JsonValue

from wmh.core.parsing import extract_json_object
from wmh.core.types import Observation, Step
from wmh.optimize.judge import JudgeResult

# A flattened JSON leaf: numbers score by relative error, everything else by exact match.
Leaf = float | bool | str | None


class NumericJudge:
    """Relative-error judge over the leaf fields of JSON observations.

    Each numeric leaf shared by both observations scores
    `max(0, 1 - |predicted - actual| / (|actual| + eps))`, clamped to [0, 1]. Booleans, strings,
    and nulls must match exactly — including type: a boolean predicted as a number (or a string
    as a number, etc.) scores 0, so a prediction can't pass by getting only the numbers right
    while flipping a status field. Fields present in only one observation score 0 (a missing or
    hallucinated field is wrong, not ignorable). A numeric field whose actual value is exactly 0
    must be predicted exactly (relative error is undefined at zero). Content that isn't a JSON
    object (or an empty one) falls back to exact match so the judge never silently passes garbage.
    """

    def __init__(self, *, tolerance: float = 0.0) -> None:
        # `tolerance` is a relative-error floor under which a numeric field counts as exact
        # (score 1.0), e.g. 0.05 treats predictions within 5% as correct. 0.0 keeps the raw
        # proportional score.
        if tolerance < 0.0:
            raise ValueError(f"tolerance must be >= 0, got {tolerance}")
        self._tolerance = tolerance

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        if predicted.is_error != actual.is_error:
            return JudgeResult(
                score=0.0,
                critique=(
                    f"error-status mismatch: predicted is_error={predicted.is_error}, "
                    f"actual is_error={actual.is_error}"
                ),
            )
        predicted_fields = _leaf_fields(predicted.content)
        actual_fields = _leaf_fields(actual.content)
        if predicted_fields is None or actual_fields is None:
            return _exact_match_result(predicted, actual, reason="non-JSON content")
        if not actual_fields and not predicted_fields:
            return _exact_match_result(predicted, actual, reason="empty JSON objects")

        dimensions: dict[str, float] = {}
        misses: list[str] = []
        for field, actual_value in actual_fields.items():
            if field not in predicted_fields:
                dimensions[field] = 0.0
                misses.append(f"{field}: missing from prediction")
                continue
            dimensions[field] = self._field_score(predicted_fields[field], actual_value)
            if dimensions[field] < 1.0:
                misses.append(
                    f"{field}: predicted {predicted_fields[field]!r} vs actual {actual_value!r}"
                )
        for field in predicted_fields:
            if field not in actual_fields:
                dimensions[field] = 0.0
                misses.append(f"{field}: fabricated (absent from actual)")

        mean = sum(dimensions.values()) / len(dimensions)
        critique = "all fields match" if not misses else "; ".join(misses[:5])
        return JudgeResult(score=mean, critique=critique, dimensions=dimensions)

    def _field_score(self, predicted: Leaf, actual: Leaf) -> float:
        # Non-numeric leaves (and cross-type pairs) compare exactly. Booleans are checked before
        # numbers because `True == 1` in Python — a flag predicted as a number is the wrong type.
        if isinstance(actual, bool) or isinstance(predicted, bool):
            if isinstance(actual, bool) != isinstance(predicted, bool):
                return 0.0
            return 1.0 if predicted == actual else 0.0
        if (
            isinstance(actual, str)
            or isinstance(predicted, str)
            or actual is None
            or predicted is None
        ):
            if type(predicted) is not type(actual):
                return 0.0
            return 1.0 if predicted == actual else 0.0
        if not (math.isfinite(predicted) and math.isfinite(actual)):
            # json.loads accepts NaN/Infinity; a correctly-predicted non-finite sentinel counts.
            if math.isnan(predicted) and math.isnan(actual):
                return 1.0
            return 1.0 if predicted == actual else 0.0
        relative_error = abs(predicted - actual) / (abs(actual) + 1e-12)
        if relative_error <= self._tolerance:
            return 1.0
        return max(0.0, 1.0 - relative_error)


def _exact_match_result(predicted: Observation, actual: Observation, *, reason: str) -> JudgeResult:
    exact = predicted.content.strip() == actual.content.strip()
    return JudgeResult(
        score=1.0 if exact else 0.0,
        critique=f"{reason}; scored by exact match" + ("" if exact else " (contents differ)"),
    )


def _leaf_fields(content: str) -> dict[str, Leaf] | None:
    """Flatten every leaf of a JSON object into dotted-path fields.

    Tolerates JSON wrapped in code fences or surrounding prose (same leniency as the LLM judges,
    via `extract_json_object`). Returns None when no JSON object can be found (caller falls back
    to exact match). Note: paths are synthesized with `.` and `[i]`, so a literal key like
    `"a.b"` shares a path with nested `a.b` — exotic, but don't feed the judge keys that contain
    dots or brackets.
    """
    raw = extract_json_object(content)
    if raw is None:
        return None
    try:
        parsed: JsonValue = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    fields: dict[str, Leaf] = {}
    _collect(parsed, prefix="", into=fields)
    return fields


def _collect(value: JsonValue, *, prefix: str, into: dict[str, Leaf]) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        into[prefix] = value
    elif isinstance(value, (int, float)):
        into[prefix] = float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            _collect(child, prefix=f"{prefix}.{key}" if prefix else key, into=into)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _collect(child, prefix=f"{prefix}[{i}]", into=into)
