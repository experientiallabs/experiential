"""Hermetic WorldModelScorer tests: fake providers and a stubbed rollout seam, no network.

Two layers, mirroring the closed-loop test idioms: the round-trip test drives the REAL
`evaluate_closed_loop` machinery with one `RoleProvider` playing agent, world model, and judge
(the `closed_loop_test` pattern), while the projection tests inject a `FakeEvaluate` seam so the
scorer's request validation, cell mapping, and artifact evidence are pinned without any rollout.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Literal

import pytest

from wmh.core.types import JsonObject
from wmh.engine.world_model import WorldModel
from wmh.evals.closed_loop import ClosedLoopReport, RolloutEvidence, TaskOutcome
from wmh.evals.gold import AssertionResult, GoldJudge, GoldVerdict
from wmh.evals.tasks import TaskSpec
from wmh.evals.world_model_scorer import (
    ClosedLoopEvaluate,
    InMemoryArtifactReader,
    WorldModelScorer,
)
from wmh.harness.doc import RUNTIME_KIND_ID, TOOL_POLICY_ID, HarnessDoc, Surface, SurfaceKind
from wmh.harness.runtime import (
    DEFAULT_EVAL_EPISODE_TIMEOUT_S,
    Runtime,
    StopReason,
    TokenUsage,
)
from wmh.harness.scoring import HarnessScore, ScoreRequest, score_harness
from wmh.providers.base import Completion, Message, Provider, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


class RoleProvider:
    """Plays agent, world model, and gold judge, keyed off the system prompt."""

    def __init__(self, *, judge_passes: bool = True, model: str = "m") -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model=model)
        self._judge_passes = judge_passes

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        if "grade whether an agent completed a task" in system:
            passed = "true" if self._judge_passes else "false"
            return Completion(
                text='{"assertions": [{"assertion": "did it", "passed": '
                + passed
                + ', "why": "x"}], "passed": '
                + passed
                + "}"
            )
        if system.startswith("You are a capable command-line agent"):
            return Completion(
                text='{"tool": "submit", "arguments": {"answer": "the answer is 42"}}'
            )
        # world model
        return Completion(text='{"output": "ok", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201 - test fake never calls it
        raise NotImplementedError


def _wm(provider: RoleProvider) -> WorldModel:
    return WorldModel(provider, EmbeddingRetriever(HashingEmbedder(dim=16)))


def _tasks() -> list[TaskSpec]:
    return [
        TaskSpec(task_id="q1", instruction="answer it", gold=["did it"]),
        TaskSpec(task_id="q2", instruction="answer it again", gold=["did it"]),
    ]


def _seam_tasks() -> list[TaskSpec]:
    return [
        TaskSpec(task_id="a", instruction="do a", gold=["g"]),
        TaskSpec(task_id="b", instruction="do b", gold=["g"]),
        TaskSpec(task_id="c", instruction="do c", gold=["g"]),
    ]


def _make_scorer(
    *,
    provider: RoleProvider | None = None,
    tasks: list[TaskSpec] | None = None,
    model_identity: JsonObject | None = None,
    judge_identity: JsonObject | None = None,
    harness_backend: Literal["local", "e2b"] = "local",
    e2b_template: str | None = None,
    eval_concurrency: int = 1,
    episode_timeout_s: float = DEFAULT_EVAL_EPISODE_TIMEOUT_S,
    should_cancel: Callable[[], bool] | None = None,
    evaluate: ClosedLoopEvaluate | None = None,
) -> WorldModelScorer:
    resolved = provider if provider is not None else RoleProvider()
    return WorldModelScorer(
        world_model=_wm(resolved),
        tasks=tasks if tasks is not None else _tasks(),
        agent_provider=resolved,
        judge=GoldJudge(resolved),
        model_identity=(
            model_identity
            if model_identity is not None
            else {"world_model": "wm-test", "build": "fingerprint-1"}
        ),
        judge_identity=judge_identity,
        harness_backend=harness_backend,
        e2b_template=e2b_template,
        eval_concurrency=eval_concurrency,
        episode_timeout_s=episode_timeout_s,
        should_cancel=should_cancel,
        evaluate=evaluate,
    )


class FakeEvaluate:
    """Rollout seam stub: records every call and fabricates a `ClosedLoopReport` per plan."""

    def __init__(
        self,
        *,
        fractions: dict[str, list[float]] | None = None,
        worker_usage: TokenUsage | None = None,
        drop_last_verdict: bool = False,
        wrong_passes: bool = False,
        extra_task_id: str | None = None,
    ) -> None:
        self.fractions = fractions
        self.worker_usage = worker_usage
        self.drop_last_verdict = drop_last_verdict
        self.wrong_passes = wrong_passes
        self.extra_task_id = extra_task_id
        self.calls: list[dict[str, object]] = []

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
    ) -> ClosedLoopReport:
        self.calls.append(
            {
                "task_ids": [task.task_id for task in tasks],
                "label": label,
                "k": k,
                "concurrency": concurrency,
                "runtime": runtime,
            }
        )
        per_task = {task.task_id: self._outcome(task, k) for task in tasks}
        if self.extra_task_id is not None:
            extra = TaskSpec(task_id=self.extra_task_id, instruction="extra", gold=["g"])
            per_task[extra.task_id] = self._outcome(extra, k)
        return ClosedLoopReport(
            label=label,
            success_rate=0.0,
            mean_fraction=0.0,
            k=k,
            per_task=per_task,
            worker_usage=self.worker_usage,
        )

    def _outcome(self, task: TaskSpec, k: int) -> TaskOutcome:
        fracs = (self.fractions or {}).get(task.task_id, [1.0] * k)
        verdicts = [
            GoldVerdict(
                passed=fraction == 1.0,
                fraction=fraction,
                assertions=[AssertionResult(assertion="g", passed=fraction == 1.0, why="w")],
                rationale=f"{task.task_id} attempt {attempt} rationale",
            )
            for attempt, fraction in enumerate(fracs, 1)
        ]
        if self.drop_last_verdict:
            verdicts = verdicts[:-1]
        attempts = [
            RolloutEvidence(
                answer=f"answer-{task.task_id}-{attempt}",
                transcript=f"[1] tool_call: bash cmd-{task.task_id}-{attempt}\n    -> ok",
                stop_reason=StopReason.SUBMITTED,
                turns=attempt,
            )
            for attempt in range(1, len(fracs) + 1)
        ]
        return TaskOutcome(
            task_id=task.task_id,
            success_rate=sum(1.0 for verdict in verdicts if verdict.passed) / max(len(fracs), 1),
            mean_fraction=sum(verdict.fraction for verdict in verdicts) / max(len(fracs), 1),
            passes=k - 1 if self.wrong_passes else k,
            verdicts=verdicts,
            attempts=attempts,
        )


def _pi_node_doc() -> HarnessDoc:
    return HarnessDoc(
        name="pi",
        surfaces=[
            Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p"),
            Surface(id=TOOL_POLICY_ID, kind=SurfaceKind.TOOL_POLICY, content="bash\nsubmit"),
            Surface(id=RUNTIME_KIND_ID, kind=SurfaceKind.PARAM, content="pi-node"),
        ],
    )


# -- digests and requests ---------------------------------------------------------------------


def test_context_and_request_are_deterministic_across_instances() -> None:
    first = _make_scorer()
    second = _make_scorer()
    assert first.context() == second.context()
    assert first.request(attempts=2) == second.request(attempts=2)
    assert first.request(attempts=2).task_ids == ("q1", "q2")


def test_task_set_digest_ignores_task_order_but_request_preserves_it() -> None:
    forward = _make_scorer(tasks=_tasks())
    reversed_tasks = _make_scorer(tasks=list(reversed(_tasks())))
    assert forward.context() == reversed_tasks.context()
    assert reversed_tasks.request(attempts=1).task_ids == ("q2", "q1")


def test_digests_isolate_tasks_evaluator_and_execution() -> None:
    base = _make_scorer().context()

    changed_tasks = _make_scorer(
        tasks=[
            TaskSpec(task_id="q1", instruction="a DIFFERENT instruction", gold=["did it"]),
            TaskSpec(task_id="q2", instruction="answer it again", gold=["did it"]),
        ]
    ).context()
    assert changed_tasks.task_set_digest != base.task_set_digest
    assert changed_tasks.evaluator_digest == base.evaluator_digest
    assert changed_tasks.execution_config_digest == base.execution_config_digest

    changed_model = _make_scorer(model_identity={"world_model": "wm-test", "build": "v2"}).context()
    assert changed_model.evaluator_digest != base.evaluator_digest
    assert changed_model.task_set_digest == base.task_set_digest
    assert changed_model.execution_config_digest == base.execution_config_digest

    changed_judge = _make_scorer(judge_identity={"judge_model": "other"}).context()
    assert changed_judge.evaluator_digest != base.evaluator_digest
    assert changed_judge.task_set_digest == base.task_set_digest
    assert changed_judge.execution_config_digest == base.execution_config_digest

    for variant in (
        _make_scorer(eval_concurrency=4),
        _make_scorer(provider=RoleProvider(model="other-model")),
        _make_scorer(harness_backend="e2b", e2b_template="tpl", eval_concurrency=0),
        _make_scorer(
            harness_backend="e2b",
            e2b_template="tpl",
            eval_concurrency=0,
            episode_timeout_s=120.0,
        ),
    ):
        context = variant.context()
        assert context.execution_config_digest != base.execution_config_digest
        assert context.task_set_digest == base.task_set_digest
        assert context.evaluator_digest == base.evaluator_digest


# -- end to end through the real closed-loop machinery -----------------------------------------


def test_score_round_trips_through_score_harness_with_real_rollouts() -> None:
    scorer = _make_scorer()
    candidate = HarnessDoc.baseline("candidate")
    request = scorer.request(attempts=2)

    scored = score_harness(scorer, candidate, request=request)

    assert scored.report.candidate_doc_hash == candidate.doc_hash
    assert scored.report.request == request
    assert [
        (cell.task_id, cell.attempt, cell.score, cell.passed) for cell in scored.report.cells
    ] == [
        ("q1", 1, 1.0, True),
        ("q1", 2, 1.0, True),
        ("q2", 1, 1.0, True),
        ("q2", 2, 1.0, True),
    ]
    assert scored.report.score == 1.0
    assert scored.report.pass_rate == 1.0
    assert [artifact.path for artifact in scored.report.artifacts] == [
        "rollouts/task-0001/attempt-1.json",
        "rollouts/task-0001/attempt-2.json",
        "rollouts/task-0002/attempt-1.json",
        "rollouts/task-0002/attempt-2.json",
    ]
    assert all(artifact.media_type == "application/json" for artifact in scored.report.artifacts)
    # Every cell references exactly its own transcript artifact and no orphans exist
    # (HarnessScoreReport validation), so the mapping below is total.
    by_cell = {(cell.task_id, cell.attempt): cell.artifact_paths for cell in scored.report.cells}
    assert by_cell[("q2", 2)] == ("rollouts/task-0002/attempt-2.json",)

    payload = json.loads(scored.artifacts.read_bytes("rollouts/task-0001/attempt-1.json").decode())
    assert payload["task_id"] == "q1"
    assert payload["instruction"] == "answer it"
    assert payload["gold"] == ["did it"]
    assert payload["verdict"]["passed"] is True
    assert payload["verdict"]["fraction"] == 1.0
    assert payload["verdict"]["assertions"] == [{"assertion": "did it", "passed": True, "why": "x"}]
    assert "submit" in payload["rollout"]["transcript"]
    assert payload["rollout"]["answer"] == "the answer is 42"
    assert payload["rollout"]["stop_reason"] == "submitted"
    # The fixed AgentRuntime is provider-wrapped, so nothing self-meters worker tokens.
    assert scorer.last_worker_usage is None


def test_failed_judgement_maps_to_zero_scores() -> None:
    scorer = _make_scorer(provider=RoleProvider(judge_passes=False))
    scored = score_harness(scorer, HarnessDoc.baseline(), request=scorer.request(attempts=1))
    assert scored.report.score == 0.0
    assert all(not cell.passed for cell in scored.report.cells)


# -- cell mapping through the stubbed seam ------------------------------------------------------


def test_cells_map_fraction_passed_and_one_indexed_attempts() -> None:
    fake = FakeEvaluate(fractions={"a": [1.0, 0.5], "b": [0.0, 1.0], "c": [1.0, 1.0]})
    scorer = _make_scorer(tasks=_seam_tasks(), evaluate=fake)
    candidate = HarnessDoc.baseline("candidate")
    request = scorer.request(attempts=2)

    scored = score_harness(scorer, candidate, request=request)

    assert [
        (cell.task_id, cell.attempt, cell.score, cell.passed) for cell in scored.report.cells
    ] == [
        ("a", 1, 1.0, True),
        ("a", 2, 0.5, False),
        ("b", 1, 0.0, False),
        ("b", 2, 1.0, True),
        ("c", 1, 1.0, True),
        ("c", 2, 1.0, True),
    ]
    assert scored.report.score == pytest.approx(4.5 / 6)
    half_credit = next(
        cell for cell in scored.report.cells if (cell.task_id, cell.attempt) == ("a", 2)
    )
    assert "fraction=0.500" in half_credit.summary
    assert "a attempt 2 rationale" in half_credit.summary
    assert len(half_credit.summary) < 16_000
    assert len(fake.calls) == 1
    assert fake.calls[0]["k"] == 2
    assert fake.calls[0]["concurrency"] == 1
    assert fake.calls[0]["label"] == "candidate"


def test_subset_and_reordered_task_ids_score_only_requested_tasks() -> None:
    fake = FakeEvaluate(fractions={"c": [1.0], "a": [0.0]})
    scorer = _make_scorer(tasks=_seam_tasks(), evaluate=fake)
    request = ScoreRequest(context=scorer.context(), task_ids=("c", "a"), attempts=1)

    scored = score_harness(scorer, HarnessDoc.baseline(), request=request)

    assert fake.calls[0]["task_ids"] == ["c", "a"]  # request order, not configured order
    assert [(cell.task_id, cell.score) for cell in scored.report.cells] == [
        ("a", 0.0),
        ("c", 1.0),
    ]
    # Artifact indices follow the request's 1-based positions: task-0001 is "c".
    assert [artifact.path for artifact in scored.report.artifacts] == [
        "rollouts/task-0001/attempt-1.json",
        "rollouts/task-0002/attempt-1.json",
    ]
    first = json.loads(scored.artifacts.read_bytes("rollouts/task-0001/attempt-1.json").decode())
    second = json.loads(scored.artifacts.read_bytes("rollouts/task-0002/attempt-1.json").decode())
    assert first["task_id"] == "c"
    assert second["task_id"] == "a"


def test_artifact_contents_are_deterministic_json_evidence() -> None:
    fake = FakeEvaluate(fractions={"a": [0.5]})
    scorer = _make_scorer(tasks=_seam_tasks()[:1], evaluate=fake)
    scored = score_harness(scorer, HarnessDoc.baseline(), request=scorer.request(attempts=1))

    raw = scored.artifacts.read_bytes("rollouts/task-0001/attempt-1.json")
    payload = json.loads(raw.decode())
    assert payload == {
        "schema_version": 1,
        "task_id": "a",
        "instruction": "do a",
        "gold": ["g"],
        "verdict": {
            "passed": False,
            "fraction": 0.5,
            "rationale": "a attempt 1 rationale",
            "assertions": [{"assertion": "g", "passed": False, "why": "w"}],
        },
        "rollout": {
            "answer": "answer-a-1",
            "transcript": "[1] tool_call: bash cmd-a-1\n    -> ok",
            "stop_reason": "submitted",
            "turns": 1,
        },
    }
    # Deterministic serialization: sorted keys, exact separators.
    assert raw.decode() == json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


# -- request and report validation --------------------------------------------------------------


def test_unknown_task_ids_raise_before_any_rollout() -> None:
    fake = FakeEvaluate()
    scorer = _make_scorer(tasks=_seam_tasks(), evaluate=fake)
    request = ScoreRequest(context=scorer.context(), task_ids=("a", "zz"), attempts=1)
    with pytest.raises(ValueError, match="does not serve"):
        scorer.score(HarnessDoc.baseline(), request=request)
    assert fake.calls == []


def test_stale_request_from_another_configuration_raises_before_any_rollout() -> None:
    fake = FakeEvaluate()
    scorer = _make_scorer(tasks=_seam_tasks(), evaluate=fake)
    other = _make_scorer(
        tasks=_seam_tasks(),
        model_identity={"world_model": "different", "build": "v9"},
    )
    with pytest.raises(ValueError, match="context differs"):
        scorer.score(HarnessDoc.baseline(), request=other.request(attempts=1))
    assert fake.calls == []


@pytest.mark.parametrize("defect", ["drop_last_verdict", "wrong_passes", "extra_task_id"])
def test_malformed_closed_loop_reports_are_rejected(defect: str) -> None:
    fake = FakeEvaluate(
        drop_last_verdict=defect == "drop_last_verdict",
        wrong_passes=defect == "wrong_passes",
        extra_task_id="zz" if defect == "extra_task_id" else None,
    )
    scorer = _make_scorer(tasks=_seam_tasks(), evaluate=fake)
    with pytest.raises(ValueError, match="closed-loop report"):
        scorer.score(HarnessDoc.baseline(), request=scorer.request(attempts=2))


def test_score_harness_rejects_a_tampered_candidate_hash() -> None:
    class TamperingScorer:
        """Delegates to a real scorer, then rewrites the report's candidate identity."""

        def __init__(self, inner: WorldModelScorer) -> None:
            self._inner = inner

        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScore:
            scored = self._inner.score(candidate, request=request)
            # doc_hash covers surfaces, not the display name, so the tampered identity must
            # come from a document whose surfaces genuinely differ.
            other = HarnessDoc(
                name="other",
                surfaces=[
                    Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="another prompt")
                ],
            )
            tampered = scored.report.model_copy(update={"candidate_doc_hash": other.doc_hash})
            return HarnessScore(report=tampered, artifacts=scored.artifacts)

    inner = _make_scorer(tasks=_seam_tasks(), evaluate=FakeEvaluate())
    scorer = TamperingScorer(inner)
    candidate = HarnessDoc.baseline("candidate")
    request = inner.request(attempts=1)
    with pytest.raises(ValueError, match="does not match the harness"):
        score_harness(scorer, candidate, request=request)


def test_local_pi_node_candidates_require_sequential_cells() -> None:
    fake = FakeEvaluate()
    scorer = _make_scorer(tasks=_seam_tasks(), eval_concurrency=2, evaluate=fake)
    with pytest.raises(ValueError, match="one episode at a time"):
        scorer.score(_pi_node_doc(), request=scorer.request(attempts=1))
    assert fake.calls == []


def test_e2b_backend_rejects_in_process_candidates_before_any_sandbox() -> None:
    fake = FakeEvaluate()
    scorer = _make_scorer(
        harness_backend="e2b",
        e2b_template="tpl",
        eval_concurrency=0,
        tasks=_seam_tasks(),
        evaluate=fake,
    )
    with pytest.raises(ValueError, match="already runs in-process"):
        scorer.score(HarnessDoc.baseline(), request=scorer.request(attempts=1))
    assert fake.calls == []


# -- worker usage --------------------------------------------------------------------------------


def test_worker_usage_is_per_score_call_state() -> None:
    fake = FakeEvaluate(worker_usage=TokenUsage(input_tokens=10, output_tokens=3, calls=2))
    scorer = _make_scorer(tasks=_seam_tasks(), evaluate=fake)
    request = scorer.request(attempts=1)

    scorer.score(HarnessDoc.baseline(), request=request)
    assert scorer.last_worker_usage == TokenUsage(input_tokens=10, output_tokens=3, calls=2)

    fake.worker_usage = None  # a later call overwrites: no stale spend survives
    scorer.score(HarnessDoc.baseline(), request=request)
    assert scorer.last_worker_usage is None


# -- constructor validation ----------------------------------------------------------------------


def test_constructor_rejects_bad_configuration() -> None:
    with pytest.raises(ValueError, match="tasks must be nonempty"):
        _make_scorer(tasks=[])
    with pytest.raises(ValueError, match="duplicate task_id"):
        _make_scorer(tasks=[_tasks()[0], _tasks()[0]])
    with pytest.raises(ValueError, match="e2b_template requires"):
        _make_scorer(e2b_template="tpl")
    with pytest.raises(ValueError, match="episode_timeout_s requires"):
        _make_scorer(episode_timeout_s=10.0)
    with pytest.raises(ValueError, match="finite positive"):
        _make_scorer(
            harness_backend="e2b", e2b_template="tpl", eval_concurrency=0, episode_timeout_s=-1.0
        )
    with pytest.raises(ValueError, match="eval_concurrency"):
        _make_scorer(eval_concurrency=-1)
    with pytest.raises(ValueError, match="eval_concurrency"):
        _make_scorer(eval_concurrency=True)
    with pytest.raises(ValueError, match="model_identity"):
        _make_scorer(model_identity={})


def test_scorer_snapshots_tasks_against_caller_mutation() -> None:
    tasks = _tasks()
    scorer = _make_scorer(tasks=tasks)
    before = scorer.context()
    tasks[0].instruction = "mutated after construction"
    assert scorer.context() == before


def test_in_memory_reader_serves_exact_bytes_and_rejects_unknown_paths() -> None:
    reader = InMemoryArtifactReader({"rollouts/x.json": b"{}"})
    assert reader.read_bytes("rollouts/x.json") == b"{}"
    with pytest.raises(KeyError, match="unknown artifact path"):
        reader.read_bytes("rollouts/missing.json")
