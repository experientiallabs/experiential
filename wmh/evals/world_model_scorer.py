"""Closed-loop world-model projection into immutable WMH harness scores.

The world-model sibling of :class:`wmh.evals.harbor.scorer.HarborScorer`: the real harness
process runs (in-process by default, in E2B sandboxes for pi-node candidates) while every
environment response is simulated by the world model, and a :class:`GoldJudge` grades each
rollout transcript against the task's gold assertions. Rollouts reuse the merged lane's
closed-loop machinery (`wmh.evals.closed_loop.evaluate_closed_loop`, the same composition
`wmh.harness.create._score` uses) rather than reimplementing them, and the result is projected
into the evaluator-neutral `HarnessScore` evidence that `score_harness` verifies.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, Protocol
from uuid import uuid4

from wmh.core.types import JsonObject
from wmh.engine.world_model import WorldModel
from wmh.evals.closed_loop import ClosedLoopReport, RolloutEvidence, evaluate_closed_loop
from wmh.evals.gold import GoldJudge, GoldVerdict
from wmh.evals.tasks import TaskSpec
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_sandbox import resolve_e2b_template
from wmh.harness.runtime import (
    DEFAULT_EVAL_EPISODE_TIMEOUT_S,
    HarnessSearchCancelled,
    Runtime,
    RuntimeCancelled,
    TokenUsage,
    validate_episode_timeout_s,
)
from wmh.harness.scoring import (
    EvaluationArtifact,
    HarnessScore,
    HarnessScoreReport,
    ScoreCell,
    ScoreContext,
    ScoreRequest,
)
from wmh.providers.base import Provider

_WORLD_MODEL_SCORER_VERSION = "1"
# Cell summaries index into the full artifact; keep them far below MAX_CELL_SUMMARY_CHARS.
_SUMMARY_RATIONALE_CHARS = 1_000


class ClosedLoopEvaluate(Protocol):
    """The exact rollout seam `evaluate_closed_loop` satisfies.

    The scorer owns the runtime and the projection; the seam owns rollouts and judging. Tests
    inject a fake here so no provider, world model, or sandbox is exercised.
    """

    def __call__(
        self,
        tasks: list[TaskSpec],
        world_model: WorldModel,
        agent_provider: Provider,
        judge: GoldJudge,
        *,
        label: str,
        k: int,
        concurrency: int,
        runtime: Runtime,
        should_cancel: Callable[[], bool] | None,
    ) -> ClosedLoopReport: ...


class InMemoryArtifactReader:
    """Serve evaluator-owned artifact bytes from an in-memory `path -> bytes` map."""

    def __init__(self, contents: Mapping[str, bytes]) -> None:
        self._contents = dict(contents)

    def read_bytes(self, path: str) -> bytes:
        """Return one artifact's exact bytes; unknown paths are a hard error."""
        content = self._contents.get(path)
        if content is None:
            raise KeyError(f"unknown artifact path {path!r}")
        return content


class WorldModelScorer:
    """Evaluate exact harness candidates closed-loop against a simulated environment.

    Each `score` call builds the candidate's runtime (`candidate.runtime(agent_provider)` for
    local execution, or the E2B pi-node runtime backed by a per-call sandbox pool for
    `harness_backend='e2b'`), runs `request.attempts` passes per requested task with the world
    model answering every environment action, and judges each rollout with the configured
    `GoldJudge`. Every (task, attempt) cell carries exactly one JSON transcript artifact at
    `rollouts/task-{index:04d}/attempt-{n}.json`, where `index` is the task's 1-based position
    in `request.task_ids` (task ids may contain characters that are unsafe in artifact paths,
    so they are never embedded in the path).

    `model_identity` is the caller-supplied world-model identity (typically the model name plus
    a build fingerprint) committed into the evaluator digest; `judge_identity` names the judge
    the same way and defaults to the judge's class name only, so callers who care about judge
    replayability should pass the judge model identity explicitly.

    `last_worker_usage` is per-score-call state: it is reset when `score` starts and afterwards
    holds that call's aggregated self-metered worker token spend (`None` when the runtime does
    not self-meter, exactly as `ClosedLoopReport.worker_usage` reports it). On cancellation it
    holds the partial spend carried by the cancelled wave. Callers metering spend must read it
    after every call; a later call overwrites it.
    """

    def __init__(
        self,
        *,
        world_model: WorldModel,
        tasks: Sequence[TaskSpec],
        agent_provider: Provider,
        judge: GoldJudge,
        model_identity: JsonObject,
        judge_identity: JsonObject | None = None,
        harness_backend: Literal["local", "e2b"] = "local",
        e2b_template: str | None = None,
        eval_concurrency: int,
        episode_timeout_s: float = DEFAULT_EVAL_EPISODE_TIMEOUT_S,
        should_cancel: Callable[[], bool] | None = None,
        evaluate: ClosedLoopEvaluate | None = None,
    ) -> None:
        task_list = [TaskSpec.model_validate(task.model_dump(mode="python")) for task in tasks]
        if not task_list:
            raise ValueError("tasks must be nonempty")
        ids = [task.task_id for task in task_list]
        duplicates = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate task_id(s): {duplicates}")
        if harness_backend not in ("local", "e2b"):
            raise ValueError("harness_backend must be local or e2b")
        if harness_backend == "local" and e2b_template is not None:
            raise ValueError("e2b_template requires harness_backend='e2b'")
        if (
            isinstance(eval_concurrency, bool)
            or not isinstance(eval_concurrency, int)
            or eval_concurrency < 0
        ):
            raise ValueError("eval_concurrency must be an integer >= 0 (0 runs every cell at once)")
        episode_timeout_s = validate_episode_timeout_s(episode_timeout_s)
        if harness_backend == "local" and episode_timeout_s != DEFAULT_EVAL_EPISODE_TIMEOUT_S:
            raise ValueError("episode_timeout_s requires harness_backend='e2b'")
        self._world_model = world_model
        self._tasks = tuple(task_list)
        self._tasks_by_id = {task.task_id: task for task in task_list}
        self._agent_provider = agent_provider
        self._judge = judge
        self._model_identity = _frozen_identity(model_identity, field="model_identity")
        self._judge_identity = _frozen_identity(
            judge_identity if judge_identity is not None else {"class": type(judge).__name__},
            field="judge_identity",
        )
        self._harness_backend: Literal["local", "e2b"] = harness_backend
        if harness_backend == "e2b":
            # Resolve once so the execution digest and the per-call pool agree even when the
            # template comes from $WMH_E2B_TEMPLATE; "" pins "resolved to nothing" (the pool's
            # own resolution treats "" as an explicit no-template, never an env fallback).
            effective_template = resolve_e2b_template(e2b_template)
            self._e2b_template = effective_template if effective_template is not None else ""
        else:
            self._e2b_template = None
        self._eval_concurrency = eval_concurrency
        self._episode_timeout_s = episode_timeout_s
        self._should_cancel = should_cancel
        self._evaluate: ClosedLoopEvaluate = (
            evaluate if evaluate is not None else evaluate_closed_loop
        )
        # Per-score-call worker spend (see the class docstring).
        self.last_worker_usage: TokenUsage | None = None

    def context(self) -> ScoreContext:
        """Build the frozen content identities this scorer evaluates under."""
        task_payload = {
            "tasks": [
                {
                    "task_id": task.task_id,
                    "instruction": task.instruction,
                    "gold": list(task.gold),
                }
                for task in sorted(self._tasks, key=lambda task: task.task_id)
            ]
        }
        evaluator_payload = {
            "scorer_version": _WORLD_MODEL_SCORER_VERSION,
            "world_model": self._model_identity,
            "judge": self._judge_identity,
        }
        execution_payload = {
            "harness_backend": self._harness_backend,
            "e2b_template": self._e2b_template,
            "eval_concurrency": self._eval_concurrency,
            "episode_timeout_s": self._episode_timeout_s,
            "agent_provider": self._agent_provider.config.model_dump(mode="json"),
        }
        return ScoreContext(
            task_set_digest=_digest_json(task_payload),
            evaluator_digest=_digest_json(evaluator_payload),
            execution_config_digest=_digest_json(execution_payload),
        )

    def request(self, *, attempts: int) -> ScoreRequest:
        """Build the full-suite score request (every configured task, in configured order)."""
        return ScoreRequest(
            context=self.context(),
            task_ids=tuple(task.task_id for task in self._tasks),
            attempts=attempts,
        )

    def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScore:
        """Run and project one candidate, raising unless every requested cell is scoreable.

        `request.task_ids` may be any ordered subset of the configured tasks (screening lanes
        score failure clusters); ids this scorer does not serve, or a request minted under a
        different configuration, raise before any rollout spend.
        """
        _check_cancelled(self._should_cancel)
        if request.context != self.context():
            raise ValueError("score request context differs from this scorer's configuration")
        unknown = sorted(set(request.task_ids) - set(self._tasks_by_id))
        if unknown:
            raise ValueError(f"score request names task(s) this scorer does not serve: {unknown}")
        ordered_tasks = [self._tasks_by_id[task_id] for task_id in request.task_ids]
        self.last_worker_usage = None
        report = self._evaluate_candidate(candidate, ordered_tasks, request)
        self.last_worker_usage = report.worker_usage
        cells, manifests, contents = self._project(report, ordered_tasks, request)
        score_report = HarnessScoreReport(
            source_run_id=f"wm-{candidate.doc_hash[:12]}-{uuid4().hex[:12]}",
            candidate_doc_hash=candidate.doc_hash,
            request=request,
            cells=tuple(cells),
            artifacts=tuple(manifests),
        )
        return HarnessScore(report=score_report, artifacts=InMemoryArtifactReader(contents))

    def _evaluate_candidate(
        self,
        candidate: HarnessDoc,
        tasks: list[TaskSpec],
        request: ScoreRequest,
    ) -> ClosedLoopReport:
        """Build the candidate's runtime for the configured backend and run the rollout seam."""
        if self._harness_backend == "local":
            if self._eval_concurrency != 1 and candidate.runtime_kind() == "pi-node":
                # Local pi runtimes are single-episode resources (one runner port/workdir or one
                # RunnerLink channel): parallel cells would collide. Checked per-candidate
                # because runtime kind is a document surface (create.py precedent).
                raise ValueError(
                    "pi-node harnesses run one episode at a time under harness_backend='local' "
                    "(single runner port/channel); use eval_concurrency=1 or "
                    "harness_backend='e2b'"
                )
            return self._run(candidate, tasks, request, candidate.runtime(self._agent_provider))
        if candidate.runtime_kind() != "pi-node":
            raise ValueError(
                "harness_backend='e2b' runs the pi-node harness process in sandboxes; "
                f"candidate runtime kind is {candidate.runtime_kind()!r}, which already runs "
                "in-process; use harness_backend='local'"
            )
        # Lazy: the e2b backend is an optional extra; local scoring must import none of it.
        from wmh.harness.pi_e2b import E2BSandboxPool

        # v1 pool lifecycle: one pool per score call, closed before the score returns, so no
        # sandbox lease outlives the evaluation that owns it.
        pool = E2BSandboxPool(
            template=self._e2b_template,
            episode_timeout_s=self._episode_timeout_s,
        )
        try:
            runtime = candidate.runtime(
                self._agent_provider,
                backend="e2b",
                e2b_pool=pool,
                episode_timeout_s=self._episode_timeout_s,
                should_cancel=self._should_cancel,
            )
            return self._run(candidate, tasks, request, runtime)
        finally:
            pool.close()

    def _run(
        self,
        candidate: HarnessDoc,
        tasks: list[TaskSpec],
        request: ScoreRequest,
        runtime: Runtime,
    ) -> ClosedLoopReport:
        """Run k=attempts closed-loop passes per task, preserving cancelled-wave spend."""
        try:
            return self._evaluate(
                list(tasks),
                self._world_model,
                self._agent_provider,
                self._judge,
                label=candidate.name,
                k=request.attempts,
                concurrency=self._eval_concurrency,
                runtime=runtime,
                should_cancel=self._should_cancel,
            )
        except RuntimeCancelled as exc:
            # Cancelled cells are not scoreable outcomes: convert at the scorer boundary
            # (create.py precedent) and keep the partial wave's spend meterable.
            self.last_worker_usage = exc.worker_usage
            raise HarnessSearchCancelled(
                "harness score cancelled", worker_usage=exc.worker_usage
            ) from exc

    def _project(
        self,
        report: ClosedLoopReport,
        tasks: list[TaskSpec],
        request: ScoreRequest,
    ) -> tuple[list[ScoreCell], list[EvaluationArtifact], dict[str, bytes]]:
        """Map the closed-loop report onto the exact requested cell matrix, fail-closed."""
        extra = sorted(set(report.per_task) - set(request.task_ids))
        if extra:
            raise ValueError(f"closed-loop report contains unrequested task(s): {extra}")
        cells: list[ScoreCell] = []
        manifests: list[EvaluationArtifact] = []
        contents: dict[str, bytes] = {}
        for index, task in enumerate(tasks, 1):
            outcome = report.per_task.get(task.task_id)
            if outcome is None:
                raise ValueError(f"closed-loop report is missing task {task.task_id!r}")
            if (
                outcome.passes != request.attempts
                or len(outcome.verdicts) != request.attempts
                or len(outcome.attempts) != request.attempts
            ):
                raise ValueError(
                    f"closed-loop report for task {task.task_id!r} does not carry exactly "
                    f"{request.attempts} judged attempt(s)"
                )
            for attempt in range(1, request.attempts + 1):
                verdict = outcome.verdicts[attempt - 1]
                evidence = outcome.attempts[attempt - 1]
                path = f"rollouts/task-{index:04d}/attempt-{attempt}.json"
                content = _artifact_content(task, verdict, evidence)
                manifests.append(
                    EvaluationArtifact.from_text(
                        path=path, content=content, media_type="application/json"
                    )
                )
                contents[path] = content.encode("utf-8")
                cells.append(
                    ScoreCell(
                        task_id=task.task_id,
                        attempt=attempt,
                        score=verdict.fraction,
                        passed=verdict.passed,
                        summary=_cell_summary(verdict, evidence),
                        artifact_paths=(path,),
                    )
                )
        return cells, manifests, contents


def _artifact_content(task: TaskSpec, verdict: GoldVerdict, evidence: RolloutEvidence) -> str:
    """Serialize one judged rollout as deterministic JSON evidence."""
    payload = {
        "schema_version": 1,
        "task_id": task.task_id,
        "instruction": task.instruction,
        "gold": list(task.gold),
        "verdict": {
            "passed": verdict.passed,
            "fraction": verdict.fraction,
            "rationale": verdict.rationale,
            "assertions": [
                {"assertion": item.assertion, "passed": item.passed, "why": item.why}
                for item in verdict.assertions
            ],
        },
        "rollout": {
            "answer": evidence.answer,
            "transcript": evidence.transcript,
            "stop_reason": evidence.stop_reason.value,
            "turns": evidence.turns,
        },
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _cell_summary(verdict: GoldVerdict, evidence: RolloutEvidence) -> str:
    """One short diagnostic line per cell; the full evidence lives in the artifact."""
    rationale = verdict.rationale
    if len(rationale) > _SUMMARY_RATIONALE_CHARS:
        rationale = rationale[:_SUMMARY_RATIONALE_CHARS] + " ..."
    return (
        f"passed={verdict.passed} fraction={verdict.fraction:.3f} "
        f"stop={evidence.stop_reason.value} turns={evidence.turns}: {rationale}"
    )


def _frozen_identity(value: JsonObject, *, field: str) -> JsonObject:
    """Snapshot one caller-supplied identity object, proving it is durable JSON."""
    if not value:
        raise ValueError(f"{field} must be a nonempty JSON object")
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be JSON-serializable") from error
    frozen = json.loads(encoded)
    if not isinstance(frozen, dict):
        raise ValueError(f"{field} must be a JSON object")
    return frozen


def _digest_json(value: object) -> str:
    """Digest one JSON-native payload deterministically (sorted keys, exact separators)."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _check_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise HarnessSearchCancelled("harness score cancelled")
