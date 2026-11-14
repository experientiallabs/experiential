"""Held-out eval rubric: a general outcome and quality score of attempts, never a training reward.

The decision-variance rules stage authors a training rubric that grades attempts into the
environment's reward, and adversarial deduction and linters add process supervision. This module
is the separate, non-adversarial measurement beside it:

- **Eval-rubric author** (one prompt, :data:`EVAL_RUBRIC_SYSTEM`): from the task instruction, its
  starting environment and the training side's error context (the training rubric's failure
  patterns and not-met conditions, plus the training judge's observed not-met findings where they
  exist) it writes a few criteria covering the task outcome, the correctness of the final state
  and the quality of the approach, each scored 0-4, plus an overall-score rule. The error context
  only teaches it what errors look like; it does not do process supervision (no step-level
  critique, no decision points, no reward-hacking axis, no per-step tags).
- **Freeze**: the rubric is saved once per task as ``eval-rubric-<task>-<sha256>.json`` before
  any scoring, with the training artifacts it saw (ids and digests) in its metadata; reruns reuse
  it and never re-author it.
- **Eval judge** (one prompt, :data:`EVAL_JUDGE_SYSTEM`): the frozen rubric plus one attempt's
  trajectory (actions, tool replies, final state and answer) gives per-criterion scores with a
  one-line reason, an overall score in [0, 1] and a short summary. One judgment per attempt; only
  a malformed answer is retried, a bounded number of times.
- **Report**: per tier mean, median and distribution of the overall score, per-criterion means,
  and a side-by-side with the training rubric's verdicts for the same attempts.

Nothing here writes into the project store the attempts came from: draws go to a private scratch
project, and the rubric, judgments and report are plain JSON files under the output directory.
The output is never consumed by training, reward or rubric revision code.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, sha256_json
from exp.common.models import ModelMessage, ModelRequest, ModelResponse, ToolChoice
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.variance.episodes import EpisodeEvidence

EVAL_RUBRIC_PROTOCOL = "eval-rubric-v1"
"""Version of the eval-rubric author prompt, input and output shape; bump when they change."""

EVAL_JUDGE_PROTOCOL = "eval-judge-v1"
"""Version of the eval judge prompt, trajectory rendering and validation; bump when they change."""

SCORE_LEVELS = ("0", "1", "2", "3", "4")
Dimension = Literal["outcome", "final_state", "approach"]

EVAL_RUBRIC_SYSTEM = """\
You write a held-out EVALUATION rubric for attempts by an AI agent at one software task.

The rubric measures, after the fact, how good an attempt was overall. It is a general quality \
score for reporting, never a training reward, so it must be fair to any correct approach.

You receive:
- task: the instruction the agent was given, its tools, and the starting environment state.
- training_error_context: what errors on this task look like, taken from a SEPARATE training \
rubric (its failure patterns and not-met conditions) and, where available, findings its judge \
reported on earlier attempts. Use it only to understand which outcomes and final states are \
actually wrong, and what weak or unverified work looks like. Do not copy its criteria, structure \
or pass/fail gates.

Write 3 to 6 criteria that together cover:
1. outcome: did the attempt accomplish what the task asks (the stated success condition)?
2. final_state: is the final environment state correct and intact (the deliverable is right, \
nothing that must stay unchanged was damaged or weakened, constraints of the task hold)?
3. approach: quality of the approach as seen in the outcome and the work product - efficiency \
(no large waste), verification (claims backed by observed evidence), clarity of the final \
report to the user.
Each criterion has exactly one dimension: "outcome", "final_state" or "approach". Cover every \
dimension at least once. Give outcome and final_state criteria most of the weight.

Each criterion gets concrete scoring guidance for every level 0, 1, 2, 3, 4 (4 best), written \
for THIS task (name the real files, commands and conditions), so two judges would agree.

Do NOT write: reward-hacking or cheating detection as its own criterion, step-by-step process \
critique, decision-point analysis, per-step tags or labels. A violated task constraint simply \
lowers the outcome and final-state scores, as any other wrong result would.

Also give overall_rule: a short instruction for combining criterion scores into one overall \
score in [0, 1]. The default is the weighted mean of score/4 using the criteria weights; you \
may add at most one cap, e.g. "if the outcome criterion scores 0 or 1, overall is at most 0.3".

Answer only by calling submit_eval_rubric once."""

EVAL_JUDGE_SYSTEM = """\
You are an evaluation judge. You score ONE finished attempt by an AI agent at a software task \
with a frozen evaluation rubric. This is a general quality evaluation of the attempt, not a \
search for cheating and not a step-by-step critique.

You receive the frozen rubric, the task (instruction and starting environment) and the \
attempt's full trajectory: the agent's messages, every tool call with its reply (tool errors are \
marked), the final environment state where recorded, and the agent's final answer.

For every rubric criterion give an integer score 0-4 that follows that criterion's level \
guidance, and a one-line reason that points at what in the trajectory justifies it. Judge from \
what was observed in tool replies and the final state, not from the agent's own claims. Then \
apply the rubric's overall_rule to give overall_score in [0, 1], and write a summary of at most \
three sentences.

Answer only by calling submit_eval_judgment once, scoring every criterion exactly once."""

_RUBRIC_TOOL = ToolSchema(
    name="submit_eval_rubric",
    description="Submit the evaluation rubric.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "required": ["summary", "criteria", "overall_rule"],
        "properties": {
            "summary": {"type": "string", "description": "What a good attempt looks like."},
            "criteria": {
                "type": "array",
                "minItems": 3,
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["criterion_id", "dimension", "question", "weight", "levels"],
                    "properties": {
                        "criterion_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{1,48}$"},
                        "dimension": {"enum": ["outcome", "final_state", "approach"]},
                        "question": {"type": "string"},
                        "weight": {"type": "number", "exclusiveMinimum": 0},
                        "levels": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": list(SCORE_LEVELS),
                            "properties": {level: {"type": "string"} for level in SCORE_LEVELS},
                        },
                    },
                },
            },
            "overall_rule": {"type": "string"},
        },
    },
)

_JUDGE_TOOL = ToolSchema(
    name="submit_eval_judgment",
    description="Submit the evaluation of the attempt.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "required": ["scores", "overall_score", "summary"],
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["criterion_id", "score", "reason"],
                    "properties": {
                        "criterion_id": {"type": "string"},
                        "score": {"type": "integer", "minimum": 0, "maximum": 4},
                        "reason": {"type": "string"},
                    },
                },
            },
            "overall_score": {"type": "number", "minimum": 0, "maximum": 1},
            "summary": {"type": "string"},
        },
    },
)


class EvalCriterion(ContractModel):
    """One eval-rubric criterion with guidance for every score level."""

    criterion_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,48}$")
    dimension: Dimension
    question: str = Field(min_length=1, max_length=2000)
    weight: float = Field(gt=0, le=100)
    levels: dict[str, str]

    @field_validator("levels")
    @classmethod
    def _every_level(cls, levels: dict[str, str]) -> dict[str, str]:
        """Require non-empty guidance for exactly the levels 0-4."""
        if set(levels) != set(SCORE_LEVELS) or not all(text.strip() for text in levels.values()):
            raise ValueError("levels must give non-empty guidance for exactly 0, 1, 2, 3 and 4")
        return levels


class EvalRubricBody(ContractModel):
    """The authored part of an eval rubric; its digest is the rubric's identity."""

    summary: str = Field(min_length=1, max_length=4000)
    criteria: tuple[EvalCriterion, ...] = Field(min_length=3, max_length=6)
    overall_rule: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def _coverage(self) -> EvalRubricBody:
        """Require unique ids and every dimension covered."""
        ids = [row.criterion_id for row in self.criteria]
        if len(set(ids)) != len(ids):
            raise ValueError("criterion ids must be unique")
        missing = {"outcome", "final_state", "approach"} - {row.dimension for row in self.criteria}
        if missing:
            raise ValueError(f"rubric misses dimensions {sorted(missing)}")
        return self

    def weighted_overall(self, scores: Mapping[str, int]) -> float:
        """Weighted mean of score/4 over the criteria, the default overall rule without caps."""
        total = sum(row.weight for row in self.criteria)
        return sum(row.weight * scores[row.criterion_id] / 4 for row in self.criteria) / total


class ContextSource(ContractModel):
    """One artifact the eval-rubric author was given, for auditing the separation."""

    artifact_id: str
    sha256: str
    role: str


class EvalRubric(ContractModel):
    """A frozen eval rubric: its body, and what it was authored from."""

    protocol: str = EVAL_RUBRIC_PROTOCOL
    use: str = "held-out evaluation only; never a training reward or training-rubric input"
    task_id: str
    author_alias: str
    body: EvalRubricBody
    context_sources: tuple[ContextSource, ...]
    author_input_sha256: str

    @property
    def sha256(self) -> str:
        """Digest of the whole frozen record."""
        return sha256_json(self)


class CriterionScore(ContractModel):
    """One criterion's score and reason."""

    criterion_id: str
    score: int = Field(ge=0, le=4)
    reason: str = Field(min_length=1, max_length=2000)


class EvalJudgment(ContractModel):
    """One attempt's eval judgment, or why it has none."""

    protocol: str = EVAL_JUDGE_PROTOCOL
    rubric_sha256: str
    tier: str
    study: str
    branch: int
    evidence_sha256: str | None = None
    status: Literal["judged", "malformed", "unavailable", "no_evidence"]
    scores: tuple[CriterionScore, ...] = ()
    overall_score: float | None = Field(default=None, ge=0, le=1)
    weighted_overall: float | None = None
    summary: str | None = None
    failure: str | None = None
    draws: int = 0


class TrainingVerdict(ContractModel):
    """The training rubric's saved verdict on one attempt, for comparison only."""

    verdict: str
    judged_study: str
    judge: str


# --------------------------------------------------------------------------- inputs


def task_context(task: JsonObject, initial_state: JsonValue | None) -> JsonObject:
    """The author's and judge's view of the task: instruction, tools and starting state.

    Args:
        task: The checkpoint's task, as JSON.
        initial_state: The starting environment state, when recorded.

    Returns:
        Instruction, tool names with descriptions, and the starting environment.
    """
    tools = [
        {"name": tool.get("name"), "description": tool.get("description")}
        for tool in task.get("tools", [])
        if isinstance(tool, dict)
    ]
    return {
        "task_id": task.get("task_id"),
        "instruction": task.get("instruction"),
        "tools": tools,
        "starting_environment": initial_state,
    }


def training_error_context(
    training_rubric: JsonObject, findings: Sequence[JsonObject], *, maximum_findings: int = 40
) -> JsonObject:
    """The training side's error knowledge, reduced to what errors look like.

    Args:
        training_rubric: The saved training ``episode-rubric`` JSON.
        findings: Not-met or insufficient findings the training judge reported, each with
            ``criterion_id``, ``status`` and ``rationale``.
        maximum_findings: Most distinct findings passed on.

    Returns:
        The training criteria's questions, not-met conditions and failure patterns, plus the
        observed findings, deduplicated by rationale.
    """
    body = training_rubric.get("body", training_rubric)
    criteria = []
    for row in body.get("criteria", []):
        criteria.append(
            {
                "criterion": row.get("criterion_id"),
                "question": row.get("question"),
                "not_met_when": row.get("not_met_when"),
                "failure_patterns": row.get("failure_patterns", []),
            }
        )
    seen: set[str] = set()
    by_criterion: dict[str, list[JsonObject]] = {}
    for row in findings:
        rationale = str(row.get("rationale", "")).strip()
        if row.get("status") not in ("not_met", "insufficient_evidence"):
            continue
        if not rationale or rationale in seen:
            continue
        seen.add(rationale)
        criterion = str(row.get("criterion_id"))
        by_criterion.setdefault(criterion, []).append(
            {"criterion": criterion, "status": row.get("status"), "note": rationale}
        )
    # Round-robin across criteria so one frequent error does not crowd out the others.
    observed: list[JsonValue] = []
    queues = [list(rows) for _, rows in sorted(by_criterion.items())]
    while queues and len(observed) < maximum_findings:
        for queue in queues:
            if queue and len(observed) < maximum_findings:
                observed.append(queue.pop(0))
        queues = [queue for queue in queues if queue]
    return {"training_rubric_failure_patterns": criteria, "observed_training_findings": observed}


def render_trajectory(evidence: EpisodeEvidence) -> JsonObject:
    """A compact, untruncated rendering of one attempt for the eval judge.

    Context events (system prompt, the instruction) are dropped since the judge sees the task
    separately; harness receipts are folded into their tool reply as an error marker; only the
    last state snapshot is kept, as the final state. Every actor message, tool call, tool reply
    and the final answer are kept whole.

    Args:
        evidence: The attempt's episode evidence.

    Returns:
        Steps in order, the final state and the final answer.
    """
    steps: list[JsonValue] = []
    final_state: JsonValue | None = None
    final_answer: JsonValue | None = None
    errors: dict[str, JsonValue] = {}
    for event in evidence.events:
        if event.kind == "receipt" and isinstance(event.content, dict):
            problem = {
                key: event.content.get(key)
                for key in ("error_type", "public_refusal")
                if event.content.get(key) is not None
            }
            if problem or event.content.get("effects") not in (None, "committed"):
                problem["effects"] = event.content.get("effects")
                errors[event.event_id.removesuffix(".receipt")] = problem
    for event in evidence.events:
        if event.scope == "context" and event.kind != "state_snapshot":
            continue
        if event.kind == "state_snapshot":
            final_state = event.content
        elif event.kind == "final_answer":
            final_answer = event.content
        elif event.kind == "assistant_message":
            steps.append({"agent_says": event.content})
        elif event.kind == "tool_call" and isinstance(event.content, dict):
            steps.append(
                {
                    "tool_call": event.content.get("name"),
                    "arguments": event.content.get("arguments"),
                }
            )
        elif event.kind == "tool_result":
            content = event.content
            reply: JsonObject = {
                "tool_reply": content.get("content") if isinstance(content, dict) else content
            }
            if isinstance(content, dict) and content.get("is_error"):
                reply["is_error"] = True
            problem = errors.get(event.event_id.removesuffix(".reply"))
            if problem:
                reply["harness"] = problem
            steps.append(reply)
        elif event.kind != "receipt":
            steps.append({event.kind: event.content})
    return {
        "complete": evidence.complete,
        "missing": list(evidence.missing),
        "steps": steps,
        "final_environment_state": final_state,
        "final_answer": final_answer,
    }


# --------------------------------------------------------------------------- model calls

Caller = Callable[[ModelRequest, str], ModelResponse]
"""Sends one request under a draw identity and returns the response."""


def _request(
    system: str, payload: JsonObject, tool: ToolSchema, output_tokens: int
) -> ModelRequest:
    """One forced-tool request."""
    return ModelRequest(
        messages=(
            ModelMessage(role="system", content=system),
            ModelMessage(role="user", content=json.dumps(payload, sort_keys=True)),
        ),
        tools=(tool,),
        tool_choice=ToolChoice(name=tool.name),
        temperature=0.0,
        maximum_output_tokens=output_tokens,
    )


def _arguments(response: ModelResponse, tool: str) -> JsonObject:
    """The forced tool's arguments, or a JSON object in the text as a fallback."""
    for call in response.output.tool_calls:
        if call.name == tool:
            return call.arguments
    text = (response.output.content or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response holds neither the tool call nor a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("response JSON is not an object")
    return value


def structured_call[T](
    caller: Caller,
    request: ModelRequest,
    tool: str,
    parse: Callable[[JsonObject], T],
    identity: str,
    *,
    retries: int,
) -> tuple[T, int]:
    """Draw once, and redraw only on a malformed answer, at most ``retries`` times.

    Args:
        caller: Model caller.
        request: The request.
        tool: Name of the forced tool.
        parse: Validates the answer; raises ValueError or ValidationError when malformed.
        identity: Draw identity prefix; draw ``n`` uses ``<identity>-d<n>``.
        retries: Extra draws allowed after malformed answers.

    Returns:
        The parsed answer and the number of draws used.

    Raises:
        MalformedAnswer: Every draw was malformed; carries the draws used and the last problem.
    """
    problem = ""
    for draw in range(retries + 1):
        current = request
        if draw:
            note = (
                f"Your previous answer was rejected: {problem[:600]}. "
                f"Call {tool} once with a complete, valid answer."
            )
            current = request.model_copy(
                update={"messages": (*request.messages, ModelMessage(role="user", content=note))}
            )
        response = caller(current, f"{identity}-d{draw}")
        try:
            return parse(_arguments(response, tool)), draw + 1
        except (ValueError, ValidationError) as error:
            problem = str(error).replace("\n", " ")
    raise MalformedAnswer(retries + 1, problem)


class MalformedAnswer(Exception):
    """Every draw of a structured call was malformed."""

    def __init__(self, draws: int, problem: str) -> None:
        """Carry the draws used and the last problem."""
        super().__init__(problem)
        self.draws = draws


# --------------------------------------------------------------------------- rubric


def rubric_path(out: Path, task_id: str) -> Path | None:
    """The frozen rubric of ``task_id`` under ``out``, if one was saved.

    Raises:
        ValueError: More than one frozen rubric exists for the task under this protocol.
    """
    found = sorted((out / "rubrics").glob(f"{EVAL_RUBRIC_PROTOCOL}-{task_id}-*.json"))
    if len(found) > 1:
        raise ValueError(f"several frozen eval rubrics for {task_id}: {[p.name for p in found]}")
    return found[0] if found else None


def load_frozen_rubric(path: Path) -> EvalRubric:
    """Load a frozen rubric and check its file name still matches its digest."""
    rubric = EvalRubric.model_validate_json(path.read_bytes())
    if not path.stem.endswith(rubric.sha256):
        raise ValueError(f"frozen eval rubric {path} does not match its digest {rubric.sha256}")
    return rubric


def freeze_rubric(
    out: Path,
    *,
    task: JsonObject,
    error_context: JsonObject,
    context_sources: Sequence[ContextSource],
    caller: Caller,
    author_alias: str,
    output_tokens: int = 16384,
    retries: int = 2,
) -> tuple[EvalRubric, Path, bool]:
    """Reuse the task's frozen eval rubric, or author and freeze it once.

    Args:
        out: Output directory; the rubric goes to ``out/rubrics``.
        task: :func:`task_context` of the task.
        error_context: :func:`training_error_context` of the training side.
        context_sources: The training and task artifacts the author is given.
        caller: Model caller for the author alias.
        author_alias: Catalog alias of the author.
        output_tokens: Author output allowance.
        retries: Redraws allowed after a malformed rubric.

    Returns:
        The rubric, its file, and whether it was authored now (False when reused).
    """
    task_id = str(task["task_id"])
    existing = rubric_path(out, task_id)
    if existing is not None:
        return load_frozen_rubric(existing), existing, False
    payload: JsonObject = {"task": task, "training_error_context": error_context}
    request = _request(EVAL_RUBRIC_SYSTEM, payload, _RUBRIC_TOOL, output_tokens)
    input_sha = sha256_json(request)
    body, _ = structured_call(
        caller,
        request,
        _RUBRIC_TOOL.name,
        EvalRubricBody.model_validate,
        f"eval-rubric-{input_sha[:16]}",
        retries=retries,
    )
    rubric = EvalRubric(
        task_id=task_id,
        author_alias=author_alias,
        body=body,
        context_sources=tuple(context_sources),
        author_input_sha256=input_sha,
    )
    path = out / "rubrics" / f"{EVAL_RUBRIC_PROTOCOL}-{task_id}-{rubric.sha256}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".tmp")
    staging.write_text(rubric.model_dump_json(indent=2), encoding="utf-8")
    staging.replace(path)
    return rubric, path, True


# --------------------------------------------------------------------------- judge


def parse_judgment(arguments: JsonObject, rubric: EvalRubricBody) -> tuple[CriterionScore, ...]:
    """Validate a judge answer: every criterion scored exactly once, overall in range.

    Raises:
        ValueError: A criterion is missing, repeated or unknown, or the overall is invalid.
    """
    scores = tuple(CriterionScore.model_validate(row) for row in arguments.get("scores", []))
    found = [row.criterion_id for row in scores]
    expected = [row.criterion_id for row in rubric.criteria]
    if sorted(found) != sorted(expected):
        raise ValueError(f"scores must cover exactly the criteria {expected}, got {found}")
    overall = arguments.get("overall_score")
    if not isinstance(overall, int | float) or isinstance(overall, bool) or not 0 <= overall <= 1:
        raise ValueError("overall_score must be a number in [0, 1]")
    if not str(arguments.get("summary", "")).strip():
        raise ValueError("summary must not be empty")
    order = {criterion: index for index, criterion in enumerate(expected)}
    return tuple(sorted(scores, key=lambda row: order[row.criterion_id]))


def judge_attempt(
    rubric: EvalRubric,
    task: JsonObject,
    evidence: EpisodeEvidence,
    *,
    tier: str,
    study: str,
    branch: int,
    caller: Caller,
    output_tokens: int = 16384,
    maximum_input_bytes: int = 512_000,
    retries: int = 2,
) -> EvalJudgment:
    """Judge one attempt once with the frozen eval rubric.

    Args:
        rubric: The frozen eval rubric.
        task: :func:`task_context` of the task.
        evidence: The attempt.
        tier: Actor tier label.
        study: Continuation study identity.
        branch: Branch index.
        caller: Model caller for the judge alias.
        output_tokens: Judge output allowance.
        maximum_input_bytes: Largest request sent; a larger attempt is unavailable, never cut.
        retries: Redraws allowed after a malformed answer.

    Returns:
        The judgment; malformed after every retry, or too large, is recorded unscored.
    """
    base = {
        "rubric_sha256": rubric.sha256,
        "tier": tier,
        "study": study,
        "branch": branch,
        "evidence_sha256": evidence.identity,
    }
    payload: JsonObject = {
        "rubric": rubric.body.model_dump(mode="json"),
        "task": task,
        "attempt": render_trajectory(evidence),
    }
    request = _request(EVAL_JUDGE_SYSTEM, payload, _JUDGE_TOOL, output_tokens)
    size = len(request.model_dump_json().encode())
    if size > maximum_input_bytes:
        failure = f"attempt needs {size} request bytes, over {maximum_input_bytes}; not cut"
        return EvalJudgment(**base, status="unavailable", failure=failure)
    holder: dict[str, JsonObject] = {}

    def parse(arguments: JsonObject) -> tuple[CriterionScore, ...]:
        scores = parse_judgment(arguments, rubric.body)
        holder["answer"] = arguments
        return scores

    identity = f"eval-judge-{sha256_json([rubric.sha256, evidence.identity])[:20]}"
    try:
        scores, draws = structured_call(
            caller, request, _JUDGE_TOOL.name, parse, identity, retries=retries
        )
    except MalformedAnswer as error:
        return EvalJudgment(
            **base, status="malformed", failure=str(error)[:2000], draws=error.draws
        )
    answer = holder["answer"]
    return EvalJudgment(
        **base,
        status="judged",
        scores=scores,
        overall_score=float(answer["overall_score"]),
        weighted_overall=round(
            rubric.body.weighted_overall({row.criterion_id: row.score for row in scores}), 4
        ),
        summary=str(answer["summary"])[:2000],
        draws=draws,
    )


# --------------------------------------------------------------------------- report


def _distribution(values: Sequence[float]) -> JsonObject:
    """Mean, median, spread and a five-bin histogram of scores in [0, 1]."""
    if not values:
        return {"n": 0}
    bins = Counter(min(int(value * 5), 4) for value in values)
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "histogram": {
            f"{index / 5:.1f}-{(index + 1) / 5:.1f}": bins.get(index, 0) for index in range(5)
        },
    }


def aggregate(
    rubric: EvalRubric,
    judgments: Sequence[EvalJudgment],
    training: Mapping[tuple[str, int], TrainingVerdict | None],
    *,
    pass_threshold: float = 0.5,
) -> JsonObject:
    """Per-tier score summaries and the side-by-side with training verdicts.

    Args:
        rubric: The frozen eval rubric.
        judgments: Every attempt's judgment.
        training: Training verdict per (study, branch); None when that study is not judged yet.
        pass_threshold: Eval overall score at or above which an attempt counts as a pass when
            compared against training pass/fail verdicts.

    Returns:
        The report object.
    """
    tiers: dict[str, JsonObject] = {}
    for tier in sorted({row.tier for row in judgments}):
        rows = [row for row in judgments if row.tier == tier]
        scored = [row for row in rows if row.status == "judged"]
        per_criterion = {
            criterion.criterion_id: round(
                statistics.fmean(
                    next(s.score for s in row.scores if s.criterion_id == criterion.criterion_id)
                    for row in scored
                ),
                3,
            )
            if scored
            else None
            for criterion in rubric.body.criteria
        }
        agreement: Counter[str] = Counter()
        side = []
        for row in sorted(rows, key=lambda item: item.branch):
            verdict = training.get((row.study, row.branch))
            label = "not judged yet" if verdict is None else verdict.verdict
            side.append(
                {
                    "branch": row.branch,
                    "eval_status": row.status,
                    "eval_overall": row.overall_score,
                    "training_verdict": label,
                }
            )
            if row.status == "judged" and label in ("pass", "fail"):
                eval_pass = (row.overall_score or 0.0) >= pass_threshold
                agreement["agree" if eval_pass == (label == "pass") else "disagree"] += 1
                agreement[f"training_{label}_eval_{'pass' if eval_pass else 'fail'}"] += 1
        training_labels = Counter(item["training_verdict"] for item in side)
        tiers[tier] = {
            "studies": sorted({row.study for row in rows}),
            "attempts": len(rows),
            "status": dict(Counter(row.status for row in rows)),
            "overall": _distribution([row.overall_score or 0.0 for row in scored]),
            "weighted_overall": _distribution([row.weighted_overall or 0.0 for row in scored]),
            "per_criterion_mean": per_criterion,
            "training_verdicts": dict(training_labels),
            "agreement_at_threshold": dict(agreement),
            "attempts_side_by_side": side,
        }
    return {
        "eval_rubric_protocol": rubric.protocol,
        "eval_judge_protocol": EVAL_JUDGE_PROTOCOL,
        "rubric_sha256": rubric.sha256,
        "task_id": rubric.task_id,
        "use": rubric.use,
        "pass_threshold_for_comparison": pass_threshold,
        "draws": sum(row.draws for row in judgments),
        "tiers": tiers,
    }


def markdown_report(report: JsonObject, rubric: EvalRubric) -> str:
    """A short Markdown summary of :func:`aggregate`'s report."""
    lines = [
        f"# Held-out eval: {rubric.task_id}",
        "",
        f"Eval rubric `{rubric.sha256[:16]}` ({rubric.protocol}, author `{rubric.author_alias}`),"
        f" judge protocol {EVAL_JUDGE_PROTOCOL}. {rubric.use}.",
        "",
        "## Rubric",
        "",
    ]
    for row in rubric.body.criteria:
        lines.append(f"- `{row.criterion_id}` ({row.dimension}, w={row.weight:g}): {row.question}")
    lines += ["", f"Overall rule: {rubric.body.overall_rule}", "", "## Scores by tier", ""]
    criteria = [row.criterion_id for row in rubric.body.criteria]
    lines.append(
        "| tier | judged/attempts | mean | median | min-max | histogram (0-1 by 0.2) | "
        "training verdicts | agree/disagree |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    tiers = report["tiers"]
    assert isinstance(tiers, dict)
    for tier, row in tiers.items():
        assert isinstance(row, dict)
        overall = row["overall"]
        assert isinstance(overall, dict)
        agree = row["agreement_at_threshold"]
        assert isinstance(agree, dict)
        hist = overall.get("histogram", {})
        assert isinstance(hist, dict)
        lines.append(
            f"| {tier} | {overall.get('n', 0)}/{row['attempts']} | {overall.get('mean', '-')} | "
            f"{overall.get('median', '-')} | {overall.get('min', '-')}-{overall.get('max', '-')} |"
            f" {' '.join(str(v) for v in hist.values())} | "
            f"{json.dumps(row['training_verdicts'])} | "
            f"{agree.get('agree', 0)}/{agree.get('disagree', 0)} |"
        )
    lines += ["", "## Per-criterion means (0-4)", ""]
    lines.append("| tier | " + " | ".join(criteria) + " |")
    lines.append("|---|" + "---|" * len(criteria))
    for tier, row in tiers.items():
        assert isinstance(row, dict)
        means = row["per_criterion_mean"]
        assert isinstance(means, dict)
        lines.append(f"| {tier} | " + " | ".join(str(means.get(c)) for c in criteria) + " |")
    lines += [
        "",
        f"Agreement counts an eval overall >= {report['pass_threshold_for_comparison']} as a pass"
        " against the training verdict; unassessable and not-yet-judged attempts are excluded.",
        "",
    ]
    return "\n".join(lines)
