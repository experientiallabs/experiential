"""The LLM judge that scores a predicted observation against the real one.

The judge is both GEPA's fitness signal and the open-loop fidelity scorer: it returns a scalar
score *and* a natural-language critique, and the critique is what GEPA reflects on to mutate the
prompt. There is exactly one judge (`RubricJudge`); it scores five fidelity dimensions and reports
their factuality-weighted mean as the headline score, so optimization hill-climbs the same metric
evaluation reports.

Judge failures are not world-model failures: a reply that cannot be parsed into the five
dimensions is retried once and, if still bad, flagged with `JudgeResult.valid=False` so callers
(`wmh.engine.replay`) can exclude it from fidelity aggregates instead of recording a spurious 0.
"""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.core.types import JsonObject, Observation, Step
from wmh.providers.base import Message, Provider

# A stable substring of JUDGE_SYSTEM used to recognize judge calls when attributing run cost
# (wmh.tracking.metered). Kept as a named constant so the prompt and the classifier never drift:
# JUDGE_SYSTEM must always contain JUDGE_MARKER (pinned by judge_test.py).
JUDGE_MARKER = "grade a world model"

# Bumped whenever scoring semantics change (prompt rules, weights, validity, truncation), so
# persisted eval results record which judge produced them and cross-version rows are never
# silently compared. "rubric-v2": factuality-weighted headline + validity + middle truncation.
JUDGE_VERSION = "rubric-v2"

# The five fidelity dimensions, scored independently in 0..1. Modeled on Qwen-AgentWorld
# (arXiv 2606.24597) "AgentWorldBench" rubric.
RUBRIC_DIMENSIONS = ("format", "factuality", "consistency", "realism", "quality")

# The same five names as a type, for callers that select one dimension (e.g. score_prompt's
# `score_dimension`). Keep in sync with RUBRIC_DIMENSIONS above.
RubricDimension = Literal["format", "factuality", "consistency", "realism", "quality"]

# Headline weights. Factuality dominates because it *is* the definition of functional
# equivalence (same outcome, errors, and salient data the agent would act on); quality is the
# judge's holistic usability verdict; format/consistency/realism are form diagnostics that are
# high for almost any well-shaped emission, so an unweighted mean lets them mask total factual
# failure (measured on the judge-quality meta-eval: wrong-facts predictions averaged 0.52-0.66
# under the unweighted mean with factuality <= 0.1). Any reply with all dimensions equal scores
# identically under both aggregations, so uniformly-judged steps are unaffected.
RUBRIC_WEIGHTS: dict[str, float] = {
    "factuality": 0.5,
    "quality": 0.2,
    "format": 0.1,
    "consistency": 0.1,
    "realism": 0.1,
}

# Observation content longer than head+tail is truncated in the middle before judging: real
# corpora contain observations up to ~190 KB (terminal-tasks), which would dominate judge cost,
# and a divergence hidden in the tail must stay visible (head-only truncation would hide it).
# `content_length` always reports the full untruncated length.
OBSERVATION_HEAD_CHARS = 6000
OBSERVATION_TAIL_CHARS = 6000
_OMITTED_MARKER = "\n[... {omitted} characters omitted ...]\n"

JUDGE_SYSTEM = """You grade a world model that simulates an environment for an AI agent. You
see the agent's action, the ACTUAL observation the real environment returned, and a PREDICTED
observation the world model generated. Score the prediction's fidelity to the actual observation on
five independent dimensions, each from 0.0 to 1.0:

- format: same shape/structure/encoding the environment uses (JSON shape, field names, exit status).
- factuality: conveys the same outcome, errors, and SALIENT data the agent would act on.
- consistency: coherent with the action and the environment's established behavior.
- realism: looks like a real response this environment would emit (not an explanation or apology).
- quality: overall, how usable this prediction is as a stand-in for the real observation.

Each observation is a JSON object with fields:
- is_error: the environment's error flag for that observation.
- content: the observation text. Very long content is truncated in the middle, marked by
  "[... N characters omitted ...]"; judge what is visible and do not treat the marker as content.
- content_length: the exact FULL length of content in characters (even when truncated).
- content_sha256: present only when content was truncated — the hash of the FULL untruncated
  content. If the two hashes are EQUAL the full contents are identical even where omitted; if
  they DIFFER while the visible text matches, the divergence hides in the omitted middle — the
  prediction is NOT a verified match there. For deterministic content that hidden divergence is
  an unverifiable factual gap: factuality must not exceed 0.5 and quality must not exceed 0.4
  (an output whose middle cannot be trusted is not usable as a stand-in).
- empty_content / empty_sentinel: set when content is the empty string ("<EMPTY_PREDICTION>" /
  "<EMPTY_ACTUAL_OBSERVATION>"). The sentinel is a label, not content.

Edge rules:
- If the predicted content is empty but the actual is non-empty, the prediction omitted the
  environment's response entirely: factuality and quality are 0.0, and format/realism must be
  scored near 0.0 too — producing nothing is not a well-formatted response. Do not infer or fill
  in the missing content from the action.
- If both contents are empty, the prediction matches exactly: score all dimensions 1.0 (empty
  output is a real, common environment response, e.g. redirected or silent commands).
- A flipped outcome — predicted is_error disagrees with actual, or success text where the real
  environment failed (or vice versa) — is a factuality failure no matter how plausible the
  content looks.

CONTENT TYPE matters — infer it from the action and observation:
- DETERMINISTIC / computed content (a file's contents via `cat`, a command's computed stdout, a
  lookup of state that exists) MUST match the actual values to score high on factuality — wrong
  computed values are unambiguously wrong even if well-formatted.
- VOLATILE / incidental content (PIDs, timestamps, random ids, ordering of unordered output) should
  be judged on plausibility and format only — a different-but-plausible value is fine.

Respond with ONLY a JSON object, no prose:
{"format": <0..1>, "factuality": <0..1>, "consistency": <0..1>, "realism": <0..1>,
 "quality": <0..1>, "critique": "<one or two sentences: what matched, what diverged>"}"""


class JudgeResult(BaseModel):
    score: float  # 0..1 headline fidelity: weighted mean of the rubric dimensions
    critique: str  # natural-language feedback; feeds GEPA reflection
    dimensions: dict[str, float] = Field(default_factory=dict)  # per-dimension scores (0..1)
    # False when the judge itself failed (unparseable/incomplete reply after a retry). Fidelity
    # aggregates must exclude invalid results — a judge failure says nothing about the prediction.
    valid: bool = True


@runtime_checkable
class Judge(Protocol):
    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult: ...


def _build_judge_prompt(predicted: Observation, actual: Observation, context: Step) -> str:
    action = context.action
    action_desc = action.name or action.content or "(none)"
    actual_payload = _observation_payload(actual, empty_sentinel="<EMPTY_ACTUAL_OBSERVATION>")
    predicted_payload = _observation_payload(predicted, empty_sentinel="<EMPTY_PREDICTION>")
    return (
        f"AGENT ACTION ({action.kind.value}): {action_desc}\n"
        f"ACTION ARGUMENTS: {json.dumps(action.arguments, sort_keys=True, default=str)}\n\n"
        "ACTUAL OBSERVATION JSON:\n"
        f"{json.dumps(actual_payload, ensure_ascii=False, sort_keys=True)}\n\n"
        "PREDICTED OBSERVATION JSON:\n"
        f"{json.dumps(predicted_payload, ensure_ascii=False, sort_keys=True)}\n"
    )


def _observation_payload(observation: Observation, *, empty_sentinel: str) -> JsonObject:
    truncated = len(observation.content) > OBSERVATION_HEAD_CHARS + OBSERVATION_TAIL_CHARS
    payload: JsonObject = {
        "is_error": observation.is_error,
        "content_length": len(observation.content),
        "content": _truncate_middle(observation.content),
        "empty_content": observation.content == "",
        "empty_sentinel": empty_sentinel if observation.content == "" else None,
    }
    if truncated:
        # Without this, two equal-length observations diverging only in the omitted middle are
        # byte-identical to the judge; the hash is the remaining tell (see JUDGE_SYSTEM).
        payload["content_sha256"] = sha256(observation.content.encode("utf-8")).hexdigest()
    return payload


def _truncate_middle(content: str) -> str:
    """Keep the head and tail of oversized content, marking how much was omitted in between."""
    limit = OBSERVATION_HEAD_CHARS + OBSERVATION_TAIL_CHARS
    if len(content) <= limit:
        return content
    omitted = len(content) - limit
    return (
        content[:OBSERVATION_HEAD_CHARS]
        + _OMITTED_MARKER.format(omitted=omitted)
        + content[-OBSERVATION_TAIL_CHARS:]
    )


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


class _RawRubric(BaseModel):
    """Typed view of the judge's JSON; every dimension is required or the reply is unusable."""

    format: float
    factuality: float
    consistency: float
    realism: float
    quality: float
    critique: str = ""


class RubricJudge:
    """The reference-grounded 5-dimension fidelity judge.

    Scores the five `RUBRIC_DIMENSIONS` and reports their `RUBRIC_WEIGHTS`-weighted mean as the
    headline `score`, with the per-dimension breakdown in `JudgeResult.dimensions`. A reply that
    cannot be parsed into all five dimensions is retried once; if it is still unusable the result
    is flagged `valid=False` rather than scored 0.0, so judge failures never masquerade as
    world-model failures.
    """

    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        user = _build_judge_prompt(predicted, actual, context)
        result = self._ask(user)
        if not result.valid:
            # At temperature 0 an identical re-ask reproduces the same malformed reply, so the
            # one retry states what was invalid about the first one.
            dims = ", ".join(f'"{name}"' for name in RUBRIC_DIMENSIONS)
            result = self._ask(
                f"{user}\n\nYour previous reply was invalid: {result.critique}\n"
                f"Respond with ONLY the JSON object, containing all five dimension keys "
                f'({dims}) and "critique".'
            )
        return result

    def _ask(self, user: str) -> JudgeResult:
        completion = self._provider.complete(
            JUDGE_SYSTEM,
            [Message(role="user", content=user)],
            temperature=0.0,
            max_tokens=1024,
        )
        return _parse_rubric(completion.text)


def _parse_rubric(text: str) -> JudgeResult:
    """Parse the judge's reply into a JudgeResult (headline = weighted mean of dimensions).

    Accepts a bare JSON object, JSON inside a ```json fence, or JSON embedded in surrounding
    prose. A reply that is not JSON, is missing a dimension, or scores a dimension far outside
    0..1 (scale confusion, e.g. answering 85 on a 0-100 scale) yields `valid=False`; values with
    minor float slop (within [-0.5, 1.5]) are clamped into range.
    """
    raw = extract_json_object(text)
    if raw is None:
        return _invalid(f"Unparseable judge reply (no JSON object). Raw: {text.strip()[:200]}")
    try:
        parsed = _RawRubric.model_validate_json(raw)
    except ValidationError as exc:
        missing = [
            str(error["loc"][0])
            for error in exc.errors()
            if error["type"] == "missing" and error["loc"]
        ]
        if missing:
            return _invalid(
                f"Judge reply missing dimension {missing[0]!r}. Raw: {text.strip()[:200]}"
            )
        return _invalid(f"Unparseable judge reply (bad JSON types). Raw: {text.strip()[:200]}")
    dims: dict[str, float] = {}
    for name in RUBRIC_DIMENSIONS:
        value = float(getattr(parsed, name))
        if not -0.5 <= value <= 1.5:
            return _invalid(
                f"Judge dimension {name!r} out of range ({value}). Raw: {text.strip()[:200]}"
            )
        dims[name] = _clamp(value)
    score = sum(RUBRIC_WEIGHTS[name] * dims[name] for name in RUBRIC_DIMENSIONS)
    return JudgeResult(score=score, critique=parsed.critique.strip(), dimensions=dims)


def _invalid(critique: str) -> JudgeResult:
    return JudgeResult(score=0.0, critique=critique, valid=False)
