"""Deterministic end-to-end tests for atomic text world-model simulation."""

import hashlib
import json
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from wmo.common.core.artifacts import (
    SECRET_REDACTION_PLACEHOLDER,
    ArtifactInput,
    assert_secret_free,
    canonical_json_bytes,
)
from wmo.common.evaluations import EvaluationCell, EvaluationPlan
from wmo.common.evaluations.build_test import _snapshot, _store
from wmo.common.models import (
    AssistantAction,
    CompletionCostReservation,
    Embedding,
    EmbeddingCostReservation,
    ModelCapabilities,
    ModelClient,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    RoutedCandidateSnapshot,
    ToolCall,
    Usage,
    completion_cost_reservation,
)
from wmo.common.project import ArtifactStore, artifact_input
from wmo.common.rollouts import (
    UNKNOWN_DISPATCH_RESERVED_COST_KEY,
    RolloutArtifact,
    SimulationCellBinding,
    SimulationMode,
    StopReason,
)
from wmo.common.tasks import TaskCase, TaskSet, ToolSchema
from wmo.runtime.agents import AgentEpisode, AgentRuntime
from wmo.runtime.environments import EnvironmentSession
from wmo.runtime.models import ResolvedModel
from wmo.runtime.models.providers.transport import ProviderTransportError
from wmo.simulation.engines.text.bindings import (
    binding_digest,
    lease_id_for_binding,
    rollout_id_for_binding,
)
from wmo.simulation.engines.text.leases import TextCellLeaseStore
from wmo.simulation.engines.text.simulator import (
    SimulationConfigurationError,
    SimulationContentionError,
    SimulationResumeError,
    WorldModelSimulator,
)
from wmo.simulation.engines.text.spec_persistence import persist_canonical_specification
from wmo.simulation.retrieval import (
    RAGEmbedderBinding,
    RAGLineageBinding,
    RAGMatch,
    RAGQuery,
    TraceRAGRetriever,
    load_fit_rag_retriever,
    persist_trace_rag,
)
from wmo.simulation.retrieval.retrieval_test import _persist_traces
from wmo.simulation.retrieval.transitions import render_rag_key
from wmo.simulation.specs import (
    CandidateCompletionReservation,
    SimulationSpec,
    WorldModelSettings,
    persist_simulation_completion_contract,
    simulation_spec_digest,
)
from wmo.simulation.world_model import GroundedWorldModel, GroundedWorldModelArtifact
from wmo.simulation.world_model.artifact import (
    GROUNDED_WORLD_MODEL_PROMPT_VERSION,
    grounded_world_model_prompt_sha256,
)

_TIME = datetime(2026, 8, 12, tzinfo=UTC)


class _ScriptedClient:
    def __init__(self, responses: list[ModelResponse], *, delay_seconds: float = 0.0) -> None:
        self._responses = list(responses)
        self._delay_seconds = delay_seconds
        self._lock = threading.Lock()
        self.requests: list[ModelRequest] = []
        self.active_calls = 0
        self.maximum_active_calls = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        with self._lock:
            self.requests.append(request)
            self.active_calls += 1
            self.maximum_active_calls = max(self.maximum_active_calls, self.active_calls)
        try:
            if self._delay_seconds:
                time.sleep(self._delay_seconds)
            with self._lock:
                return self._responses.pop(0)
        finally:
            with self._lock:
                self.active_calls -= 1


class _TimeoutClient:
    """Provider seam that records dispatch and then fails without authoritative economics."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        raise TimeoutError("provider outcome is unknown")


class _FlakyOnceClient:
    """Raise one exhausted transport failure, then delegate to scripted responses."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        """Store the answers served after the single scripted transport failure.

        Args:
            responses: Responses returned in order once the transport recovers.
        """
        self._responses = list(responses)
        self.requests: list[ModelRequest] = []
        self._failed = False

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Fail the first dispatch at the transport level and answer afterwards.

        Args:
            request: Candidate request emitted by the recording boundary.

        Returns:
            The next scripted response after the transport recovers.

        Raises:
            ProviderTransportError: The first dispatch, mimicking exhausted bounded retries.
        """
        self.requests.append(request)
        if not self._failed:
            self._failed = True
            raise ProviderTransportError("connection reset by provider")
        return self._responses.pop(0)


class _CountingEmbedder:
    """Return stable vectors while recording every embedding dispatch."""

    def __init__(self) -> None:
        """Initialize an empty ordered dispatch log."""
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Record one batch and return equal deterministic unit vectors.

        Args:
            texts: Canonical RAG texts submitted in one embedding operation.

        Returns:
            One fixed two-dimensional unit vector per input text.
        """
        self.calls.append(tuple(texts))
        return tuple(Embedding(values=(1.0, 0.0)) for _ in texts)


@dataclass
class _FitRetriever:
    """Small read-only fit retriever used to isolate text-simulator tests."""

    rag_input: ArtifactInput
    maximum_attempts: int = 2
    input_usd_per_million_tokens: float = 0.001
    embedder: ModelSnapshot | None = None
    index: SimpleNamespace = field(init=False)
    queries: list[RAGQuery] = field(init=False, default_factory=list)
    estimate_calls: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Bind the configured immutable fit scope to the fake index.

        Returns:
            None after initializing the read-only index metadata.
        """
        self.index = SimpleNamespace(
            embedder=self.embedder or _snapshot("embedder-a"),
            included_partitions=("fit",),
            included_lineage_ids=(
                "lineage-task-a",
                "lineage-task-b",
                "lineage-task-c",
                "lineage-task-d",
            ),
            fit_lineage_ids=(
                "lineage-task-a",
                "lineage-task-b",
                "lineage-task-c",
                "lineage-task-d",
            ),
        )

    def estimate_query_economics(
        self,
        query: RAGQuery,
        reservation: EmbeddingCostReservation,
    ) -> OperationEconomics:
        """Record one pre-dispatch estimate using the production upper bound.

        Args:
            query: Canonical retrieval query being estimated.
            reservation: Immutable embedding price and retry reservation.

        Returns:
            Retry-inclusive conservative query economics.
        """
        self.estimate_calls += 1
        key_text = render_rag_key(
            task=query.task,
            initial_context=query.initial_context,
            action=query.action,
        )
        reserved_tokens = len(key_text.encode("utf-8")) * reservation.maximum_attempts
        return OperationEconomics(
            cost_usd=NumericMeasurement(
                value=(reserved_tokens * reservation.input_usd_per_million_tokens / 1_000_000),
                provenance="estimated",
            )
        )

    def retrieve(self, query: RAGQuery) -> tuple[RAGMatch, ...]:
        """Record one retrieval without returning fixture examples.

        Args:
            query: Canonical retrieval query dispatched by the simulator.

        Returns:
            Empty deterministic result set.
        """
        self.queries.append(query)
        return ()


class _OneTurnAgent:
    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        del environment
        response = model.complete(
            ModelRequest(messages=(ModelMessage(role="user", content=task.instruction),))
        )
        return AgentEpisode(stop_reason=StopReason.COMPLETED, final_action=response.output)


class _ToolAttemptAgent:
    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        del task, model
        environment.execute(ToolCall(call_id="call-a", name="unexpected_tool", arguments={}))
        raise AssertionError("text-only environment must reject the attempted tool call")


def _response(
    content: str,
    *,
    snapshot: ModelSnapshot,
    cost: float | None = 0.10,
    finish_reason: ModelFinishReason = ModelFinishReason.COMPLETED,
) -> ModelResponse:
    return ModelResponse(
        output=AssistantAction(content=content),
        model=snapshot,
        economics=OperationEconomics(
            usage=Usage(input_tokens=8, output_tokens=4),
            cost_usd=(
                NumericMeasurement(value=cost, provenance="observed") if cost is not None else None
            ),
        ),
        finish_reason=finish_reason,
    )


def _task(task_id: str, *, tools: tuple[ToolSchema, ...] = ()) -> TaskCase:
    return TaskCase(
        task_id=task_id,
        lineage_group_id=f"lineage-{task_id}",
        partition="fit",
        instruction=f"Resolve {task_id} politely.",
        initial_context={"customer": "Ada"},
        tools=tools,
        workload_weight=1.0,
        source_trace_ids=(f"trace-{task_id}",),
    )


def _plan(cells: tuple[EvaluationCell, ...]) -> EvaluationPlan:
    candidate = RoutedCandidateSnapshot(alias="candidate-a", model=_snapshot("candidate-a"))
    return EvaluationPlan(
        schema_version=2,
        created_at=_TIME,
        code_revision="test-revision",
        plan_id="evaluation-plan",
        task_set_id="task-set",
        candidate_snapshots=(candidate,),
        pricing_snapshot_id="pricing-1",
        pricing_snapshot_sha256="d" * 64,
        cells=cells,
    )


def _cell(cell_id: str, task_id: str) -> EvaluationCell:
    return EvaluationCell(
        cell_id=cell_id,
        task_id=task_id,
        candidate_alias="candidate-a",
        repeat=0,
        purpose="fit",
        execution="simulate",
    )


def _persist_plan(store: ArtifactStore, plan: EvaluationPlan) -> ArtifactInput:
    manifest = store.write_json(
        artifact_id=plan.plan_id,
        artifact_type="evaluation-plan",
        envelope=plan,
        files={"evaluation-plan.json": plan},
    )
    return artifact_input(manifest)


def _persist_task_set(store: ArtifactStore, tasks: dict[str, TaskCase]) -> ArtifactInput:
    """Persist the immutable full task set required by text simulation identity checks."""
    ordered = tuple(tasks[task_id] for task_id in sorted(tasks))
    payload = b"\n".join(canonical_json_bytes(task) for task in ordered) + b"\n"
    task_set = TaskSet(
        schema_version=1,
        created_at=_TIME,
        code_revision="test-revision",
        task_set_id="task-set",
        task_ids=tuple(task.task_id for task in ordered),
        tasks_path="tasks.jsonl",
        tasks_sha256=hashlib.sha256(payload).hexdigest(),
    )
    manifest = store.write(
        artifact_id=task_set.task_set_id,
        artifact_type="task-set",
        envelope=task_set,
        files={"task-set.json": canonical_json_bytes(task_set), "tasks.jsonl": payload},
    )
    return artifact_input(manifest)


def _fit_rag_input() -> ArtifactInput:
    """Return the canonical fit-only RAG pointer used by simulator fixtures.

    Returns:
        Stable fit-only RAG manifest pointer.
    """
    return ArtifactInput(artifact_id="fit-rag", sha256="f" * 64)


def _grounded_world_model_input() -> ArtifactInput:
    """Return the immutable grounded world-model pointer used by simulator fixtures.

    Returns:
        Stable completed world-model manifest pointer.
    """
    return ArtifactInput(artifact_id="grounded-world-model", sha256="e" * 64)


def _grounded_world_model(
    world_client: ModelClient,
    retriever: TraceRAGRetriever,
    *,
    artifact_input: ArtifactInput | None = None,
) -> GroundedWorldModel:
    """Bind a fixture provider client to the exact fake fit retriever.

    Args:
        world_client: Provider seam that executes grounded requests.
        retriever: Exact fit-only retriever shared with the simulator.
        artifact_input: Optional exact persisted world-model pointer.

    Returns:
        Artifact-bound fixture executor.
    """
    serving_input = ArtifactInput(artifact_id="serving-rag", sha256="c" * 64)
    return GroundedWorldModel(
        artifact_input=artifact_input or _grounded_world_model_input(),
        artifact=GroundedWorldModelArtifact(
            schema_version=1,
            created_at=_TIME,
            inputs=(serving_input,),
            code_revision="test-revision",
            world_model_id="grounded-world-model",
            serving_rag=serving_input,
            model_alias="world-model-a",
            model=_snapshot("world-model-a"),
            prompt_version=GROUNDED_WORLD_MODEL_PROMPT_VERSION,
            prompt_sha256=grounded_world_model_prompt_sha256(),
            top_k=8,
        ),
        retriever=retriever,
        client=world_client,
    )


def _query_embedding(
    *,
    price: float = 0.001,
    maximum_attempts: int = 2,
    maximum_input_tokens: int = 10_000,
) -> EmbeddingCostReservation:
    """Build a bounded query-embedding reservation for simulator fixtures.

    Args:
        price: Catalog input price in USD per million tokens.
        maximum_attempts: Maximum provider attempts reserved per query.
        maximum_input_tokens: Maximum query input admitted for dispatch.

    Returns:
        Immutable reservation pinned to the fixture embedder.
    """
    return EmbeddingCostReservation(
        model=_snapshot("embedder-a"),
        input_usd_per_million_tokens=price,
        maximum_attempts=maximum_attempts,
        maximum_input_tokens=maximum_input_tokens,
    )


def _completion_reservation(alias: str) -> CompletionCostReservation:
    """Build a complete one-attempt request reservation for one fixture model.

    Args:
        alias: Exact candidate or world-model alias.

    Returns:
        Conservative full-context completion request reservation.
    """
    return completion_cost_reservation(
        model=_snapshot(alias),
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=2,
        cached_input_usd_per_million_tokens=0.5,
        cache_write_usd_per_million_tokens=1.5,
        maximum_attempts=1,
        maximum_input_tokens=80_000,
        maximum_output_tokens=16_000,
    )


def _persist_completion_contract(store: ArtifactStore) -> ArtifactInput:
    """Persist the complete candidate and world reservation fixture.

    Args:
        store: Project-local artifact store.

    Returns:
        Exact completion-contract manifest input.
    """
    _contract, contract_input = persist_simulation_completion_contract(
        store,
        inputs=(),
        candidate_requests=(
            CandidateCompletionReservation(
                candidate_alias="candidate-a",
                request=_completion_reservation("candidate-a"),
            ),
            CandidateCompletionReservation(
                candidate_alias="candidate-b",
                request=_completion_reservation("candidate-b"),
            ),
        ),
        world_model_alias="world-model-a",
        world_model_request=_completion_reservation("world-model-a"),
        maximum_attempts=1,
        created_at=_TIME,
        code_revision="test-revision",
    )
    return contract_input


def _resolved(
    alias: str,
    client: ModelClient,
    *,
    context_window: int = 100_000,
) -> ResolvedModel:
    """Build one resolved completion model for simulator tests.

    Args:
        alias: Catalog alias and model identifier used by the fixture.
        client: Provider-neutral client injected into the resolved model.
        context_window: Declared maximum input context.

    Returns:
        Resolved model with complete completion capabilities and pricing.
    """
    return ResolvedModel(
        alias=alias,
        snapshot=_snapshot(alias),
        capabilities=ModelCapabilities(
            supports_completions=True,
            context_window_tokens=context_window,
            maximum_output_tokens=16_000,
            input_cost_per_million_tokens_usd=1,
            output_cost_per_million_tokens_usd=2,
            cached_input_cost_per_million_tokens_usd=0.5,
            cache_write_cost_per_million_tokens_usd=1.5,
        ),
        client=client,
        embedding_client=None,
    )


def _spec(
    plan_input: ArtifactInput,
    task_set_input: ArtifactInput,
    cells: tuple[str, ...],
    *,
    fit_rag_input: ArtifactInput | None = None,
    completion_contract_input: ArtifactInput | None = None,
    query_embedding: EmbeddingCostReservation | None = None,
    **updates: object,
) -> SimulationSpec:
    """Build a finite text-simulation specification over immutable inputs.

    Args:
        plan_input: Exact persisted evaluation-plan pointer.
        task_set_input: Exact persisted task-set pointer.
        cells: Ordered evaluation-cell identities selected for execution.
        fit_rag_input: Optional explicit fit-only RAG pointer.
        query_embedding: Optional explicit query-embedding reservation.
        completion_contract_input: Optional exact completion reservation artifact.
        **updates: Additional specification fields overriding fixture defaults.

    Returns:
        Validated finite-cost world-model specification.
    """
    rag_input = fit_rag_input or _fit_rag_input()
    grounded_input = _grounded_world_model_input()
    inputs = [plan_input, task_set_input, rag_input, grounded_input]
    if completion_contract_input is not None:
        inputs.append(completion_contract_input)
    values: dict[str, object] = {
        "schema_version": 1,
        "created_at": _TIME,
        "inputs": tuple(
            sorted(
                inputs,
                key=lambda item: item.artifact_id,
            )
        ),
        "code_revision": "test-revision",
        "simulation_id": "simulation-a",
        "evaluation_plan_id": "evaluation-plan",
        "cell_ids": cells,
        "agent_id": "agent-a",
        "mode": SimulationMode.WORLD_MODEL,
        "world_model": WorldModelSettings(
            world_model_alias="world-model-a",
            grounded_world_model_input=grounded_input,
            prompt_version="text-world-model-v1",
            query_embedding=query_embedding or _query_embedding(),
        ),
        "seed": 11,
        "maximum_steps": 2,
        "maximum_cost_usd": 10.0,
    }
    values.update(updates)
    return SimulationSpec.model_validate(values)


def _simulator(
    store: ArtifactStore,
    plan: EvaluationPlan,
    plan_input: ArtifactInput,
    task_set_input: ArtifactInput,
    candidate_client: ModelClient,
    world_client: ModelClient,
    *,
    candidate_context_window: int = 100_000,
    agent_factory: Callable[[], AgentRuntime] = _OneTurnAgent,
    fit_retriever: TraceRAGRetriever | _FitRetriever | None = None,
    fit_rag_input: ArtifactInput | None = None,
    completion_contract_input: ArtifactInput | None = None,
) -> WorldModelSimulator:
    """Bind deterministic clients and an exact fit retriever to the simulator.

    Args:
        store: Immutable artifact store receiving simulator evidence.
        plan: Frozen evaluation plan selected for execution.
        plan_input: Exact persisted plan pointer.
        task_set_input: Exact persisted task-set pointer.
        candidate_client: Candidate model client used by the agent.
        world_client: Text-world-model client used for environment transitions.
        candidate_context_window: Candidate request context-window ceiling.
        agent_factory: Factory creating one isolated agent runtime per episode.
        fit_retriever: Optional exact read-only fit retriever.
        fit_rag_input: Optional explicit fit-only RAG pointer.
        completion_contract_input: Optional exact completion reservation artifact.

    Returns:
        Fully bound text-world-model simulator.
    """
    retriever = fit_retriever or _FitRetriever(_fit_rag_input())
    rag_input = fit_rag_input or retriever.rag_input
    typed_retriever = cast(TraceRAGRetriever, retriever)
    return WorldModelSimulator(
        store=store,
        evaluation_plan=plan,
        evaluation_plan_input=plan_input,
        task_set_input=task_set_input,
        fit_rag_input=rag_input,
        fit_retriever=typed_retriever,
        candidate_models={
            "candidate-a": _resolved(
                "candidate-a",
                candidate_client,
                context_window=candidate_context_window,
            )
        },
        world_models={"world-model-a": _resolved("world-model-a", world_client)},
        grounded_world_models={
            "world-model-a": _grounded_world_model(world_client, typed_retriever)
        },
        agent_factory=agent_factory,
        completion_contract_input=completion_contract_input,
        clock=lambda: _TIME,
        monotonic=lambda: 1.0,
    )


def test_text_simulation_persists_separate_economics_and_resumes_without_duplicate_calls(
    tmp_path: Path,
) -> None:
    """Persist separate economics and replay an immutable rollout without calls.

    Args:
        tmp_path: Isolated project root for immutable simulator artifacts.
    """
    cell = _cell("cell-a", "task-a")
    plan = _plan((cell,))
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    tasks = {"task-a": _task("task-a")}
    task_set_input = _persist_task_set(store, tasks)
    candidate_client = _ScriptedClient(
        [_response("I can help.", snapshot=_snapshot("candidate-a"), cost=0.2)]
    )
    world_client = _ScriptedClient(
        [
            _response(
                '{"message":"Thanks.","terminal":true}',
                snapshot=_snapshot("world-model-a"),
                cost=0.8,
            )
        ]
    )
    retriever = _FitRetriever(_fit_rag_input())
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
        fit_retriever=retriever,
    )
    spec = _spec(plan_input, task_set_input, ("cell-a",))

    artifact_set = simulator.run(spec)
    rollout_id = artifact_set.artifact_ids[0]
    rollout = simulator._load_rollout(rollout_id)
    resumed = simulator.run(spec.model_copy(update={"created_at": _TIME + timedelta(hours=1)}))

    assert rollout.candidate_economics.cost_usd == NumericMeasurement(
        value=0.2,
        provenance="observed",
    )
    assert rollout.world_model_economics is not None
    assert rollout.world_model_economics.cost_usd == NumericMeasurement(
        value=0.8,
        provenance="observed",
    )
    assert rollout.retrieval_economics is not None
    query = retriever.queries[0]
    query_text = render_rag_key(
        task=query.task,
        initial_context=query.initial_context,
        action=query.action,
    )
    assert rollout.retrieval_economics.cost_usd == NumericMeasurement(
        value=len(query_text.encode("utf-8")) * 2 * 0.001 / 1_000_000,
        provenance="estimated",
    )
    assert retriever.estimate_calls == 1
    assert len(retriever.queries) == 1
    assert retriever.queries[0].excluded_lineage_ids == ("lineage-task-a",)
    assert rollout.simulation_spec_sha256 == simulation_spec_digest(spec)
    assert len(rollout.spans) == 2
    assert len(candidate_client.requests) == 1
    assert len(world_client.requests) == 1
    assert retriever.estimate_calls == 1
    assert len(retriever.queries) == 1
    assert resumed.artifact_ids == artifact_set.artifact_ids
    persisted_spec = SimulationSpec.model_validate_json(
        store.read_bytes(spec.simulation_id, "simulation-spec.json")
    )
    assert persisted_spec.created_at == spec.created_at
    assert persisted_spec == spec
    assert persisted_spec != spec.model_copy(update={"created_at": _TIME + timedelta(hours=1)})

    assert spec.world_model is not None
    drifted_specs = (
        spec.model_copy(update={"evaluation_plan_id": "different-plan"}),
        spec.model_copy(update={"cell_ids": ("cell-b",)}),
        spec.model_copy(
            update={
                "world_model": spec.world_model.model_copy(update={"prompt_version": "changed"})
            }
        ),
        spec.model_copy(update={"maximum_steps": 3}),
        spec.model_copy(update={"maximum_concurrency": 2}),
        spec.model_copy(update={"maximum_cost_usd": 9.0}),
        spec.model_copy(update={"code_revision": "changed-revision"}),
    )
    for drifted in drifted_specs:
        with pytest.raises((SimulationConfigurationError, SimulationResumeError)):
            simulator.run(drifted)
    assert len(candidate_client.requests) == 1
    assert len(world_client.requests) == 1
    assert retriever.estimate_calls == 1
    assert len(retriever.queries) == 1


def test_persisted_rollout_redacts_generated_secrets_and_records_audit_count(
    tmp_path: Path,
) -> None:
    """Redact credential-shaped simulated output at persistence and record the count.

    Args:
        tmp_path: Isolated project root for immutable simulator artifacts.
    """
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    cell = _cell("cell-a", "task-a")
    plan = _plan((cell,))
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(store, {"task-a": _task("task-a")})
    candidate_client = _ScriptedClient(
        [_response(f"export KEY={secret}", snapshot=_snapshot("candidate-a"), cost=0.2)]
    )
    world_client = _ScriptedClient(
        [
            _response(
                '{"message":"Done.","terminal":true}',
                snapshot=_snapshot("world-model-a"),
                cost=0.8,
            )
        ]
    )
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
        fit_retriever=_FitRetriever(_fit_rag_input()),
    )
    spec = _spec(plan_input, task_set_input, ("cell-a",))

    artifact_set = simulator.run(spec)
    rollout = simulator._load_rollout(artifact_set.artifact_ids[0])

    assert rollout.secret_redaction_count >= 1
    assert secret not in rollout.model_dump_json()
    assert rollout.final_output is not None
    assert rollout.final_output.content == f"export KEY={SECRET_REDACTION_PLACEHOLDER}"
    assert_secret_free(rollout)


def test_persisted_fit_rag_grounds_active_simulation_and_replay_has_zero_dispatch(
    tmp_path: Path,
) -> None:
    """Ground the prompt in frozen real traces and replay without dispatch.

    Args:
        tmp_path: Isolated project root for immutable simulator artifacts.
    """
    cell = _cell("cell-a", "task-a")
    plan = _plan((cell,))
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(store, {"task-a": _task("task-a")})
    trace_input, traces = _persist_traces(store, count=2)
    embedding_client = _CountingEmbedder()
    embedding = RAGEmbedderBinding(
        client=embedding_client,
        snapshot=_snapshot("embedder-a"),
        maximum_attempts=2,
        input_usd_per_million_tokens=0.001,
    )
    persisted = persist_trace_rag(
        store,
        (trace_input,),
        (
            RAGLineageBinding(
                trace_id=traces[0].trace_id,
                lineage_id="lineage-task-a",
                partition="fit",
            ),
            RAGLineageBinding(
                trace_id=traces[1].trace_id,
                lineage_id="lineage-other",
                partition="fit",
            ),
        ),
        created_at=_TIME,
        code_revision="test-revision",
        embedder=embedding,
    )
    fit_rag_input = artifact_input(persisted.manifest)
    retriever = load_fit_rag_retriever(store, fit_rag_input, embedder=embedding)
    candidate_client = _ScriptedClient(
        [_response("I can help.", snapshot=_snapshot("candidate-a"), cost=0.2)]
    )
    world_client = _ScriptedClient(
        [
            _response(
                '{"message":"Thanks.","terminal":true}',
                snapshot=_snapshot("world-model-a"),
                cost=0.8,
            )
        ]
    )
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
        fit_retriever=retriever,
        fit_rag_input=fit_rag_input,
    )
    reservation = _query_embedding()
    spec = _spec(
        plan_input,
        task_set_input,
        ("cell-a",),
        fit_rag_input=fit_rag_input,
        query_embedding=reservation,
    )
    rag_before = store.read_bytes(persisted.index.rag_id, "rag-index.json")

    first = simulator.run(spec)
    dispatches = (
        len(embedding_client.calls),
        len(candidate_client.requests),
        len(world_client.requests),
    )
    replay = simulator.run(spec)

    evidence = world_client.requests[0].messages[1].content
    assert evidence is not None
    grounded = json.loads(evidence)["grounded_examples"]
    assert [item["transition_id"] for item in grounded] == [
        transition.transition_id
        for transition in persisted.transitions
        if transition.lineage_id == "lineage-other"
    ]
    assert dispatches == (2, 1, 1)
    assert (
        len(embedding_client.calls),
        len(candidate_client.requests),
        len(world_client.requests),
    ) == dispatches
    assert replay == first
    assert store.read_bytes(persisted.index.rag_id, "rag-index.json") == rag_before


def test_query_reservation_exceeding_remaining_budget_blocks_every_dispatch(
    tmp_path: Path,
) -> None:
    """Reject a query reservation that exceeds the remaining cell budget.

    Args:
        tmp_path: Isolated project root used to verify zero provider dispatch.
    """
    cell = _cell("cell-a", "task-a")
    plan = _plan((cell,))
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(store, {"task-a": _task("task-a")})
    candidate_client = _ScriptedClient([])
    world_client = _ScriptedClient([])
    retriever = _FitRetriever(_fit_rag_input(), input_usd_per_million_tokens=100.0)
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
        fit_retriever=retriever,
    )
    settings = WorldModelSettings(
        world_model_alias="world-model-a",
        grounded_world_model_input=_grounded_world_model_input(),
        prompt_version="text-world-model-v1",
        query_embedding=_query_embedding(price=100.0),
    )

    artifact_set = simulator.run(
        _spec(
            plan_input,
            task_set_input,
            ("cell-a",),
            world_model=settings,
            maximum_cost_usd=0.1,
        )
    )
    rollout = simulator._load_rollout(artifact_set.artifact_ids[0])

    assert rollout.stop_reason == StopReason.MAXIMUM_COST
    assert rollout.failure is not None
    assert rollout.failure.details["phase"] == "query_embedding_reservation"
    assert candidate_client.requests == []
    assert world_client.requests == []
    assert retriever.estimate_calls == 0
    assert retriever.queries == []


def test_full_episode_reservation_blocks_candidate_retrieval_and_world_dispatch(
    tmp_path: Path,
) -> None:
    """Reserve every possible turn before the first candidate or retrieval call.

    Args:
        tmp_path: Isolated project root used to verify zero provider dispatch.
    """
    cell = _cell("cell-a", "task-a")
    plan = _plan((cell,))
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(store, {"task-a": _task("task-a")})
    candidate_client = _ScriptedClient([])
    world_client = _ScriptedClient([])
    retriever = _FitRetriever(_fit_rag_input())
    _contract, completion_input = persist_simulation_completion_contract(
        store,
        inputs=(),
        candidate_requests=(
            CandidateCompletionReservation(
                candidate_alias="candidate-a",
                request=_completion_reservation("candidate-a"),
            ),
            CandidateCompletionReservation(
                candidate_alias="candidate-b",
                request=_completion_reservation("candidate-b"),
            ),
        ),
        world_model_alias="world-model-a",
        world_model_request=_completion_reservation("world-model-a"),
        maximum_attempts=1,
        created_at=_TIME,
        code_revision="test-revision",
    )
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
        fit_retriever=retriever,
        completion_contract_input=completion_input,
    )
    settings = WorldModelSettings(
        world_model_alias="world-model-a",
        grounded_world_model_input=_grounded_world_model_input(),
        prompt_version="text-world-model-v1",
        query_embedding=_query_embedding(),
    )

    artifact_set = simulator.run(
        _spec(
            plan_input,
            task_set_input,
            ("cell-a",),
            world_model=settings,
            completion_contract_input=completion_input,
            maximum_cost_usd=0.1,
        )
    )
    rollout = simulator._load_rollout(artifact_set.artifact_ids[0])

    assert rollout.stop_reason == StopReason.MAXIMUM_COST
    assert rollout.failure is not None
    assert rollout.failure.details["phase"] == "episode_provider_reservation"
    assert candidate_client.requests == []
    assert world_client.requests == []
    assert retriever.estimate_calls == 0
    assert retriever.queries == []


@pytest.mark.parametrize(
    ("price", "maximum_attempts", "message"),
    (
        (0.002, 2, "price reservation"),
        (0.001, 3, "retry reservation"),
    ),
)
def test_query_embedding_catalog_drift_blocks_every_dispatch(
    tmp_path: Path,
    price: float,
    maximum_attempts: int,
    message: str,
) -> None:
    """Reject catalog drift before artifacts or provider dispatch.

    Args:
        tmp_path: Isolated project root used to verify zero provider dispatch.
        price: Active catalog price supplied by the retriever fixture.
        maximum_attempts: Active retry bound supplied by the retriever fixture.
        message: Expected validation-error fragment for the drift case.
    """
    cell = _cell("cell-a", "task-a")
    plan = _plan((cell,))
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(store, {"task-a": _task("task-a")})
    candidate_client = _ScriptedClient([])
    world_client = _ScriptedClient([])
    retriever = _FitRetriever(_fit_rag_input())
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
        fit_retriever=retriever,
    )
    settings = WorldModelSettings(
        world_model_alias="world-model-a",
        grounded_world_model_input=_grounded_world_model_input(),
        prompt_version="text-world-model-v1",
        query_embedding=_query_embedding(
            price=price,
            maximum_attempts=maximum_attempts,
        ),
    )
    artifacts_before = store.list_ids()

    with pytest.raises(SimulationConfigurationError, match=message):
        simulator.run(_spec(plan_input, task_set_input, ("cell-a",), world_model=settings))

    assert store.list_ids() == artifacts_before
    assert candidate_client.requests == []
    assert world_client.requests == []
    assert retriever.estimate_calls == 0
    assert retriever.queries == []


def test_text_simulation_records_tool_tasks_and_context_overflow_as_failed_cells(
    tmp_path: Path,
) -> None:
    """Neither a declared tool nor an overflowing request reaches a remote provider silently."""
    tool = ToolSchema(
        name="lookup",
        description="Lookup an account.",
        input_schema={"type": "object"},
    )
    cells = (_cell("cell-a", "task-a"), _cell("cell-b", "task-b"))
    plan = _plan(cells)
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    tasks = {"task-a": _task("task-a", tools=(tool,)), "task-b": _task("task-b")}
    task_set_input = _persist_task_set(store, tasks)
    candidate_client = _ScriptedClient([])
    world_client = _ScriptedClient([])
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
        candidate_context_window=16_000,
    )
    spec = _spec(plan_input, task_set_input, ("cell-a", "cell-b"))

    artifact_set = simulator.run(spec)
    tool_rollout = simulator._load_rollout(artifact_set.artifact_ids[0])
    overflow_rollout = simulator._load_rollout(artifact_set.artifact_ids[1])

    assert tool_rollout.failure is not None
    assert tool_rollout.failure.code.value == "unsupported"
    assert overflow_rollout.stop_reason == StopReason.CONTEXT_OVERFLOW
    assert overflow_rollout.failure is not None
    assert overflow_rollout.failure.code.value == "context_overflow"
    assert candidate_client.requests == []
    assert world_client.requests == []


def test_text_simulation_normalizes_agent_tool_attempts_to_unsupported_cells(
    tmp_path: Path,
) -> None:
    """An agent cannot bypass task tool declarations to get execution in a text-only episode."""
    cell = _cell("cell-a", "task-a")
    plan = _plan((cell,))
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    tasks = {"task-a": _task("task-a")}
    task_set_input = _persist_task_set(store, tasks)
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        _ScriptedClient([]),
        _ScriptedClient([]),
        agent_factory=_ToolAttemptAgent,
    )

    artifact_set = simulator.run(_spec(plan_input, task_set_input, ("cell-a",)))
    rollout = simulator._load_rollout(artifact_set.artifact_ids[0])

    assert rollout.failure is not None
    assert rollout.failure.code.value == "unsupported"
    assert rollout.failure.attribution is not None
    assert rollout.failure.attribution.value == "tool"


def test_text_simulation_observes_length_stop_and_stops_spend_admission(tmp_path: Path) -> None:
    """A length finish is durable evidence, then later selected cells become budget failures."""
    cells = (_cell("cell-a", "task-a"), _cell("cell-b", "task-b"))
    plan = _plan(cells)
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    tasks = {"task-a": _task("task-a"), "task-b": _task("task-b")}
    task_set_input = _persist_task_set(store, tasks)
    candidate_client = _ScriptedClient(
        [
            _response(
                "unfinished",
                snapshot=_snapshot("candidate-a"),
                cost=0.6,
                finish_reason=ModelFinishReason.LENGTH,
            )
        ]
    )
    world_client = _ScriptedClient([])
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
    )
    spec = _spec(plan_input, task_set_input, ("cell-a", "cell-b"), maximum_cost_usd=0.5)

    artifact_set = simulator.run(spec)
    length_rollout = simulator._load_rollout(artifact_set.artifact_ids[0])
    budget_rollout = simulator._load_rollout(artifact_set.artifact_ids[1])

    assert length_rollout.stop_reason == StopReason.LENGTH
    assert budget_rollout.stop_reason == StopReason.MAXIMUM_COST
    assert budget_rollout.failure is not None
    assert budget_rollout.failure.code.value == "budget"
    assert len(candidate_client.requests) == 1
    assert world_client.requests == []


def test_text_simulation_does_not_treat_unpriced_provider_calls_as_zero_spend(
    tmp_path: Path,
) -> None:
    """Block retrieval and later paid cells after unknown candidate spend.

    Args:
        tmp_path: Isolated project root for failure evidence.
    """
    cells = (_cell("cell-a", "task-a"), _cell("cell-b", "task-b"))
    plan = _plan(cells)
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    tasks = {"task-a": _task("task-a"), "task-b": _task("task-b")}
    task_set_input = _persist_task_set(store, tasks)
    candidate_client = _ScriptedClient(
        [_response("I can help.", snapshot=_snapshot("candidate-a"), cost=None)]
    )
    world_client = _ScriptedClient(
        [
            _response(
                '{"message":"done","terminal":true}',
                snapshot=_snapshot("world-model-a"),
                cost=0.1,
            )
        ]
    )
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
    )
    spec = _spec(plan_input, task_set_input, ("cell-a", "cell-b"), maximum_cost_usd=1.0)

    artifact_set = simulator.run(spec)
    second_rollout = simulator._load_rollout(artifact_set.artifact_ids[1])

    assert second_rollout.stop_reason == StopReason.MAXIMUM_COST
    assert len(candidate_client.requests) == 1
    assert world_client.requests == []


@pytest.mark.parametrize("invalid_role", ["candidate", "world_model"])
def test_invalid_production_usage_charges_reservation_and_admits_later_paid_cells(
    tmp_path: Path,
    invalid_role: str,
) -> None:
    """Persist a worst-case reservation for unknown spend without blocking later cells.

    Args:
        tmp_path: Isolated project root for durable failure evidence.
        invalid_role: Candidate or world-model response whose usage cannot be priced.
    """
    cells = (_cell("cell-a", "task-a"), _cell("cell-b", "task-b"))
    plan = _plan(cells)
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(
        store,
        {"task-a": _task("task-a"), "task-b": _task("task-b")},
    )
    missing_usage = _response(
        "I can help.", snapshot=_snapshot("candidate-a"), cost=None
    ).model_copy(update={"economics": OperationEconomics()})
    valid_candidate = _response("I can help.", snapshot=_snapshot("candidate-a"), cost=None)
    valid_world = _response(
        '{"message":"done","terminal":true}',
        snapshot=_snapshot("world-model-a"),
        cost=None,
    )
    invalid_world = valid_world.model_copy(
        update={
            "economics": OperationEconomics(
                usage=Usage(input_tokens=8, output_tokens=4, cached_input_tokens=9)
            )
        }
    )
    candidate_client = _ScriptedClient(
        [missing_usage, valid_candidate]
        if invalid_role == "candidate"
        else [valid_candidate, valid_candidate]
    )
    world_client = _ScriptedClient(
        [valid_world] if invalid_role == "candidate" else [invalid_world, valid_world]
    )
    completion_input = _persist_completion_contract(store)
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
        completion_contract_input=completion_input,
    )
    spec = _spec(
        plan_input,
        task_set_input,
        ("cell-a", "cell-b"),
        completion_contract_input=completion_input,
        maximum_cost_usd=1.0,
    )

    artifact_set = simulator.run(spec)
    first = simulator._load_rollout(artifact_set.artifact_ids[0])
    second = simulator._load_rollout(artifact_set.artifact_ids[1])
    calls = (len(candidate_client.requests), len(world_client.requests))
    replay = simulator.run(spec)

    assert first.stop_reason == StopReason.FAILURE
    assert first.failure is not None
    assert first.failure.details["provider_dispatch_unknown_spend"] is True
    assert first.failure.retryable is False
    reserved = first.failure.details[UNKNOWN_DISPATCH_RESERVED_COST_KEY]
    assert isinstance(reserved, float) and reserved > 0
    assert second.stop_reason == StopReason.COMPLETED
    assert replay == artifact_set
    assert (len(candidate_client.requests), len(world_client.requests)) == calls


def test_resume_reexecutes_retryable_transport_failure_as_new_immutable_attempt(
    tmp_path: Path,
) -> None:
    """A persisted transport failure is superseded on resume by a fresh-budget attempt.

    Args:
        tmp_path: Isolated project root for durable failure and retry evidence.
    """
    cells = (_cell("cell-a", "task-a"),)
    plan = _plan(cells)
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(store, {"task-a": _task("task-a")})
    candidate_client = _FlakyOnceClient(
        [_response("I can help.", snapshot=_snapshot("candidate-a"), cost=None)]
    )
    world_client = _ScriptedClient(
        [
            _response(
                '{"message":"done","terminal":true}',
                snapshot=_snapshot("world-model-a"),
                cost=None,
            )
        ]
    )
    completion_input = _persist_completion_contract(store)
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
        completion_contract_input=completion_input,
    )
    spec = _spec(
        plan_input,
        task_set_input,
        ("cell-a",),
        completion_contract_input=completion_input,
        maximum_cost_usd=1.0,
    )

    first_set = simulator.run(spec)
    first = simulator._load_rollout(first_set.artifact_ids[0])
    resumed_set = simulator.run(spec)
    second = simulator._load_rollout(resumed_set.artifact_ids[0])
    replay = simulator.run(spec)

    assert first.stop_reason == StopReason.FAILURE
    assert first.failure is not None
    assert first.failure.retryable is True
    assert first.failure.exception_type == "ProviderTransportError"
    assert first.failure.details["provider_dispatch_unknown_spend"] is True
    reserved = first.failure.details[UNKNOWN_DISPATCH_RESERVED_COST_KEY]
    assert isinstance(reserved, float) and reserved > 0
    assert first.retry_attempt == 0
    assert second.retry_attempt == 1
    assert second.rollout_id != first.rollout_id
    assert second.stop_reason == StopReason.COMPLETED
    assert simulator._load_rollout(first.artifact_id) == first
    assert replay == resumed_set
    assert len(candidate_client.requests) == 2


def test_prior_attempt_reservation_charges_the_ceiling_before_retry(tmp_path: Path) -> None:
    """A superseded unknown-spend attempt keeps its worst-case charge on retry admission.

    Args:
        tmp_path: Isolated project root for durable reservation evidence.
    """
    cells = (_cell("cell-a", "task-a"),)
    plan = _plan(cells)
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(store, {"task-a": _task("task-a")})
    candidate_client = _FlakyOnceClient(
        [_response("I can help.", snapshot=_snapshot("candidate-a"), cost=None)]
    )
    world_client = _ScriptedClient([])
    completion_input = _persist_completion_contract(store)
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
        completion_contract_input=completion_input,
    )
    call_reservation = _completion_reservation("candidate-a").estimated_maximum_call_cost_usd
    episode_reservation = 2 * (2 * call_reservation + 0.000001)
    spec = _spec(
        plan_input,
        task_set_input,
        ("cell-a",),
        completion_contract_input=completion_input,
        maximum_cost_usd=episode_reservation + 0.4 * call_reservation,
    )

    first_set = simulator.run(spec)
    first = simulator._load_rollout(first_set.artifact_ids[0])
    resumed_set = simulator.run(spec)
    second = simulator._load_rollout(resumed_set.artifact_ids[0])
    replay = simulator.run(spec)

    assert first.stop_reason == StopReason.FAILURE
    assert first.failure is not None
    assert first.failure.retryable is True
    assert first.failure.details[UNKNOWN_DISPATCH_RESERVED_COST_KEY] == pytest.approx(
        call_reservation
    )
    assert second.retry_attempt == 1
    assert second.stop_reason == StopReason.MAXIMUM_COST
    assert replay == resumed_set
    assert len(candidate_client.requests) == 1
    assert world_client.requests == []


def test_finite_budget_provider_timeout_poisons_later_paid_admission(tmp_path: Path) -> None:
    """A dispatched timeout has unknown spend, so no second paid cell may be sent."""
    cells = (_cell("cell-a", "task-a"), _cell("cell-b", "task-b"))
    plan = _plan(cells)
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(
        store,
        {"task-a": _task("task-a"), "task-b": _task("task-b")},
    )
    candidate_client = _TimeoutClient()
    world_client = _ScriptedClient([])
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
    )

    artifact_set = simulator.run(
        _spec(plan_input, task_set_input, ("cell-a", "cell-b"), maximum_cost_usd=0.01)
    )
    first = simulator._load_rollout(artifact_set.artifact_ids[0])
    second = simulator._load_rollout(artifact_set.artifact_ids[1])

    assert len(candidate_client.requests) == 1
    assert world_client.requests == []
    assert first.stop_reason == StopReason.FAILURE
    assert first.failure is not None
    assert first.failure.details["provider_dispatch_unknown_spend"] is True
    assert first.candidate_economics.cost_usd is None
    assert second.stop_reason == StopReason.MAXIMUM_COST
    assert second.failure is not None
    assert second.failure.details["observed_spend_usd"] is None


def test_stale_transition_blocks_paid_admission_until_unknown_spend_rollout_persists(
    tmp_path: Path,
) -> None:
    """A stale tombstone is a budget barrier while its durable rollout is still pending."""
    cells = (_cell("cell-a", "task-a"), _cell("cell-b", "task-b"))
    plan = _plan(cells)
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(
        store,
        {"task-a": _task("task-a"), "task-b": _task("task-b")},
    )
    candidate_client = _ScriptedClient([])
    world_client = _ScriptedClient([])
    recovery = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
    )
    contender = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
    )
    spec = _spec(plan_input, task_set_input, ("cell-a", "cell-b"), maximum_cost_usd=1.0)
    selected, world_model, grounded_world_model = recovery._validate_spec_and_bindings(spec)
    canonical_spec, spec_input = persist_canonical_specification(store, spec)
    resolution, resolution_input, bindings = recovery._persist_resolution(
        canonical_spec, spec_input, selected, world_model, grounded_world_model
    )
    first_binding = bindings["cell-a"]
    holder = TextCellLeaseStore(store.project_directory, clock=lambda: _TIME)
    holder.acquire(
        lease_id=lease_id_for_binding(resolution, first_binding),
        resolution_id=resolution.resolution_id,
        simulation_id=spec.simulation_id,
        rollout_id=rollout_id_for_binding(first_binding),
        binding_sha256=binding_digest(first_binding),
        maximum_cost_usd=1.0,
        rollout_completed=lambda _rollout_id: False,
        observed_spend_usd=lambda: 0.0,
    )
    recovery._leases = TextCellLeaseStore(
        store.project_directory,
        clock=lambda: _TIME.replace(hour=1),
        owner_alive=lambda _pid: False,
    )
    elapsed = [0.0]
    contender._leases = TextCellLeaseStore(
        store.project_directory,
        clock=lambda: _TIME.replace(hour=1),
        owner_alive=lambda _pid: False,
        sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
        monotonic=lambda: elapsed[0],
        wait_timeout_seconds=0.05,
    )
    stale_persist_started = threading.Event()
    allow_stale_persist = threading.Event()
    persist = recovery._persist_rollout

    def pause_stale_persist(
        rollout: RolloutArtifact,
        cell: EvaluationCell,
        binding: SimulationCellBinding,
        resolved_input: ArtifactInput,
        *,
        attempt: int = 0,
    ) -> RolloutArtifact:
        if rollout.failure is not None and rollout.failure.details.get("phase") == (
            "paid_cell_stale_lease"
        ):
            stale_persist_started.set()
            assert allow_stale_persist.wait(timeout=5)
        return persist(rollout, cell, binding, resolved_input, attempt=attempt)

    recovery.__dict__["_persist_rollout"] = pause_stale_persist
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(recovery.run, spec)
        try:
            assert stale_persist_started.wait(timeout=5)
            with pytest.raises(SimulationContentionError, match="contended; retry"):
                contender._execute_and_persist_cell(
                    spec,
                    selected[1],
                    world_model,
                    grounded_world_model,
                    spec_input,
                    resolution,
                    resolution_input,
                    bindings,
                )
            assert candidate_client.requests == []
            assert world_client.requests == []
        finally:
            allow_stale_persist.set()
        artifact_set = future.result(timeout=5)

    second = recovery._load_rollout(artifact_set.artifact_ids[1])
    assert second.stop_reason == StopReason.MAXIMUM_COST
    assert candidate_client.requests == []
    assert world_client.requests == []


def test_text_simulation_serializes_finite_cost_admission(tmp_path: Path) -> None:
    """Serialize cells so later admission uses reconciled provider spend.

    Args:
        tmp_path: Isolated project root for durable admission leases.
    """
    cells = tuple(_cell(f"cell-{letter}", f"task-{letter}") for letter in "abcd")
    plan = _plan(cells)
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    tasks = {
        task.task_id: task
        for task in (_task("task-a"), _task("task-b"), _task("task-c"), _task("task-d"))
    }
    task_set_input = _persist_task_set(store, tasks)
    candidate_client = _ScriptedClient(
        [_response(f"candidate {index}", snapshot=_snapshot("candidate-a")) for index in range(4)],
        delay_seconds=0.03,
    )
    world_client = _ScriptedClient(
        [
            _response(
                '{"message":"done","terminal":true}',
                snapshot=_snapshot("world-model-a"),
            )
            for _ in range(4)
        ]
    )
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
    )
    spec = _spec(
        plan_input,
        task_set_input,
        tuple(cell.cell_id for cell in cells),
        maximum_concurrency=2,
    )

    artifact_set = simulator.run(spec)

    assert len(artifact_set.artifact_ids) == 4
    assert candidate_client.maximum_active_calls == 1
    assert world_client.maximum_active_calls == 1


def test_text_simulation_continues_after_agent_completion_until_world_terminal(
    tmp_path: Path,
) -> None:
    """A one-turn agent cannot turn a nonterminal world response into a completed rollout."""
    cell = _cell("cell-a", "task-a")
    plan = _plan((cell,))
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(store, {"task-a": _task("task-a")})
    candidate_client = _ScriptedClient(
        [
            _response("first answer", snapshot=_snapshot("candidate-a")),
            _response("second answer", snapshot=_snapshot("candidate-a")),
        ]
    )
    world_client = _ScriptedClient(
        [
            _response(
                '{"message":"Please continue.","terminal":false}',
                snapshot=_snapshot("world-model-a"),
            ),
            _response(
                '{"message":"done","terminal":true}',
                snapshot=_snapshot("world-model-a"),
            ),
        ]
    )
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
    )

    artifact_set = simulator.run(_spec(plan_input, task_set_input, ("cell-a",)))
    rollout = simulator._load_rollout(artifact_set.artifact_ids[0])

    assert rollout.stop_reason == StopReason.COMPLETED
    assert len(candidate_client.requests) == 2
    assert len(world_client.requests) == 2
    assert candidate_client.requests[1].messages[-2].assistant_action is not None
    assert candidate_client.requests[1].messages[-2].assistant_action.content == "first answer"
    assert candidate_client.requests[1].messages[-1].content == "Please continue."


def test_text_simulation_cross_runner_claim_prevents_duplicate_paid_calls(tmp_path: Path) -> None:
    """Two concurrent same-spec runners share one durable paid-cell claim and rollout."""
    cell = _cell("cell-a", "task-a")
    plan = _plan((cell,))
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(store, {"task-a": _task("task-a")})
    candidate_client = _ScriptedClient(
        [_response("answer", snapshot=_snapshot("candidate-a"))], delay_seconds=0.08
    )
    world_client = _ScriptedClient(
        [
            _response(
                '{"message":"done","terminal":true}',
                snapshot=_snapshot("world-model-a"),
            )
        ]
    )
    spec = _spec(plan_input, task_set_input, ("cell-a",))
    first = _simulator(store, plan, plan_input, task_set_input, candidate_client, world_client)
    second = _simulator(store, plan, plan_input, task_set_input, candidate_client, world_client)

    with ThreadPoolExecutor(max_workers=2) as executor:
        artifact_sets = tuple(executor.map(lambda simulator: simulator.run(spec), (first, second)))

    assert artifact_sets[0].artifact_ids == artifact_sets[1].artifact_ids
    assert len(candidate_client.requests) == 1
    assert len(world_client.requests) == 1


def test_text_simulation_live_hung_claim_times_out_without_calls_or_result_artifact(
    tmp_path: Path,
) -> None:
    """A live owner yields retryable contention without provider work or permanent cell output."""
    cell = _cell("cell-a", "task-a")
    plan = _plan((cell,))
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(store, {"task-a": _task("task-a")})
    candidate_client = _ScriptedClient([])
    world_client = _ScriptedClient([])
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
    )
    spec = _spec(plan_input, task_set_input, ("cell-a",), maximum_cost_usd=1.0)
    cells, world_model, grounded_world_model = simulator._validate_spec_and_bindings(spec)
    canonical_spec, spec_input = persist_canonical_specification(store, spec)
    resolution, resolution_input, bindings = simulator._persist_resolution(
        canonical_spec, spec_input, cells, world_model, grounded_world_model
    )
    binding = bindings[cell.cell_id]
    rollout_id = rollout_id_for_binding(binding)
    holder = TextCellLeaseStore(store.project_directory, clock=lambda: _TIME)
    holder.acquire(
        lease_id=lease_id_for_binding(resolution, binding),
        resolution_id=resolution.resolution_id,
        simulation_id=spec.simulation_id,
        rollout_id=rollout_id,
        binding_sha256=binding_digest(binding),
        maximum_cost_usd=spec.maximum_cost_usd,
        rollout_completed=lambda _rollout_id: False,
        observed_spend_usd=lambda: 0.0,
    )
    elapsed = [0.0]
    simulator._leases = TextCellLeaseStore(
        store.project_directory,
        clock=lambda: _TIME,
        sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
        monotonic=lambda: elapsed[0],
        wait_timeout_seconds=0.05,
    )
    artifacts_before = store.list_ids()

    with pytest.raises(SimulationContentionError, match="contended; retry"):
        simulator.run(spec)

    assert elapsed[0] == pytest.approx(0.05)
    assert candidate_client.requests == []
    assert world_client.requests == []
    assert store.list_ids() == artifacts_before
    assert rollout_id not in store.list_ids()


def test_two_finite_budget_runners_complete_each_cell_exactly_once(tmp_path: Path) -> None:
    """Cross-runner followers recompute spend and share both under-budget cell artifacts."""
    cells = (_cell("cell-a", "task-a"), _cell("cell-b", "task-b"))
    plan = _plan(cells)
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(
        store,
        {"task-a": _task("task-a"), "task-b": _task("task-b")},
    )
    candidate_client = _ScriptedClient(
        [
            _response("answer a", snapshot=_snapshot("candidate-a"), cost=0.1),
            _response("answer b", snapshot=_snapshot("candidate-a"), cost=0.1),
        ],
        delay_seconds=0.05,
    )
    world_client = _ScriptedClient(
        [
            _response(
                '{"message":"done","terminal":true}',
                snapshot=_snapshot("world-model-a"),
                cost=0.1,
            )
            for _cell_index in range(2)
        ]
    )
    spec = _spec(
        plan_input,
        task_set_input,
        ("cell-a", "cell-b"),
        maximum_cost_usd=0.5,
    )
    runners = (
        _simulator(store, plan, plan_input, task_set_input, candidate_client, world_client),
        _simulator(store, plan, plan_input, task_set_input, candidate_client, world_client),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        artifact_sets = tuple(executor.map(lambda runner: runner.run(spec), runners))

    assert artifact_sets[0].artifact_ids == artifact_sets[1].artifact_ids
    assert len(artifact_sets[0].artifact_ids) == 2
    assert len(candidate_client.requests) == 2
    assert len(world_client.requests) == 2
    rollouts = tuple(runners[0]._load_rollout(item) for item in artifact_sets[0].artifact_ids)
    assert all(rollout.stop_reason == StopReason.COMPLETED for rollout in rollouts)
