"""Run simulator-only actor councils and read-only trace analysis through maintained owners."""

import inspect
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Lock
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.evaluations.reconstruction import (
    EvaluationReferences,
    JudgeConfig,
    evaluate,
    mechanical_checks,
)
from exp.common.models import ModelCatalog, load_model_catalog
from exp.common.progress import ProgressHook, report
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.council.fidelity import load_tool_fidelity
from exp.optimize.evaluation.council.memory import CouncilMemory, read_council_memory
from exp.optimize.evaluation.fidelity.decision_access import decision_caller
from exp.optimize.evaluation.fidelity.retrieval_binding import (
    bind_index_embedder,
    index_embedder_alias,
    require_unindexed_held_out_task,
)
from exp.optimize.evaluation.fidelity.runner import FidelityRetrieval
from exp.runtime.agents.factory import ActorHarness, create_actor_runtime
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.structured_completion import (
    CompletionPlan,
    ModelCompletion,
    StructuredCompletion,
)
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.runtime.sessions.store import SessionStore
from exp.simulation.actor import run_actor_workflow
from exp.simulation.actor_context import (
    ActorContextFragment,
    ContextRecordingClient,
    with_actor_context,
)
from exp.simulation.evaluation.analysis import (
    ANALYSIS_PROMPT,
    CouncilTrace,
    TraceAnalysis,
    aggregate_analyses,
    analyze_traces,
)
from exp.simulation.evaluation.counterfactuals import RecordedEpisode, matched_native_replies
from exp.simulation.evaluation.tasks import load_evaluation_task
from exp.simulation.retrieval.jev import JEVCueSelector, JEVSelector
from exp.simulation.world_model.checkpoints import load_world_checkpoint
from exp.simulation.world_model.durable import SessionPolicy
from exp.simulation.world_model.experience_memory import (
    consumed_task_lineages,
    memory_head,
    reserve_task_lineage,
)
from exp.simulation.world_model.grounded_session import open_grounded_session
from exp.simulation.world_model.runtime import (
    load_grounded_world_model,
    load_grounded_world_model_artifact,
)

# Identity of the code that decides this artifact's behavior, in place of its source text.
# Bump it when behavior changes. exp/tests/code_protocols_test.py fails when the covered
# source changes until it is bumped or, for a pure refactor, its digest is updated.
COUNCIL_HARNESS_PROTOCOL = "council-harness-v6"


class CouncilActor(ContractModel):
    """A configured actor with an explicitly operator-declared capability label.

    Attributes:
        alias: Catalog model alias, unique within the council.
        tier: Descriptive label, never treated as a proven capability ranking.
    """

    alias: str = Field(min_length=1)
    tier: str = Field(min_length=1)


class TraceCouncilSpec(ContractModel):
    """Frozen simulator-only workload; there is no native-environment connection.

    Attributes:
        run_id: Fresh execution identity; identical completed runs replay without calls.
        world_model: Existing reconstructed simulator and its immutable knowledge/RAG.
        tasks: Public actor tasks; evaluator reference answers are prohibited.
        actors: At least two distinct configured actor aliases.
        analysts: Distinct configured read-only trace analysts.
        judge_alias: Maintained dual evaluator's configured judge.
        judge: Frozen five-axis rubric and prompt.
        judge_output_protocol: Explicit JSON text or named-tool response framing for judging.
        retrieval: Existing JEV binding when required by the simulator artifact.
        memory: Optional frozen checkpoint, excluding every actor task lineage.
        task_bundle: Optional private task verifier and shared immutable candidate checkpoint.
        actor_context: Explicit source-annotated instructions delivered only to actors.
        analyst_fidelity_plans: Immutable held-out fidelity runs, visible only to analysts.
        analyst_native_episodes: Immutable archived observations, used only after execution.
        deadline: Shared deadline for actors, simulation, judging and analysis.
        maximum_physical_attempts: Global finite cap, including transport retries.
        maximum_concurrency: Bounded independent actor/task workers; one preserves serial execution.
        maximum_actor_turns: Maintained adapter model-call limit for every cell.
        actor_harness: Same adapter and pinned Pi installation, when selected, for every actor.
        maximum_tool_operations: Durable simulator operation cap for every cell.
        actor_output_tokens: Per-call actor allowance.
        simulator_output_tokens: Per-call simulator allowance.
        judge_output_tokens: Per-call judge allowance.
        analyst_output_tokens: Per-call trace analysis allowance.
        request_reservation_usd: Accounting hold, not a claim of known hosted prices.
        output_protocol: Explicit simulator transition format.
    """

    run_id: str = Field(min_length=1)
    world_model: ArtifactInput
    tasks: tuple[TaskCase, ...] = Field(min_length=1)
    actors: tuple[CouncilActor, ...] = Field(min_length=2)
    analysts: tuple[str, ...] = Field(min_length=1)
    judge_alias: str = Field(min_length=1)
    judge: JudgeConfig
    judge_output_protocol: Literal["json", "tool"] = Field(
        default="tool", exclude_if=lambda value: value == "tool"
    )
    retrieval: FidelityRetrieval | None = None
    memory: CouncilMemory | None = None
    task_bundle: ArtifactInput | None = Field(default=None, exclude_if=lambda value: value is None)
    actor_context: tuple[ActorContextFragment, ...] = ()
    analyst_fidelity_plans: tuple[ArtifactInput, ...] = ()
    analyst_native_episodes: tuple[ArtifactInput, ...] = ()
    deadline: AwareDatetime
    maximum_physical_attempts: int = Field(gt=0, strict=True)
    maximum_concurrency: int = Field(
        default=1, ge=1, le=32, strict=True, exclude_if=lambda value: value == 1
    )
    maximum_actor_turns: int = Field(default=4, gt=0, strict=True)
    actor_harness: ActorHarness = Field(default_factory=ActorHarness)
    maximum_tool_operations: int = Field(default=8, gt=0, strict=True)
    actor_output_tokens: int = Field(default=4096, gt=0, strict=True)
    simulator_output_tokens: int = Field(default=12288, gt=0, strict=True)
    judge_output_tokens: int = Field(default=16384, gt=0, strict=True)
    analyst_output_tokens: int = Field(default=8192, gt=0, strict=True)
    request_reservation_usd: float = Field(default=1, gt=0, allow_inf_nan=False)
    output_protocol: Literal["text", "json-tool"] = "json-tool"

    @model_validator(mode="after")
    def identities(self) -> "TraceCouncilSpec":
        """Require independent cells and keep evaluator references out of every actor input."""
        if len({a.alias for a in self.actors}) != len(self.actors):
            raise ValueError("council actor aliases must be distinct")
        if len(set(self.analysts)) != len(self.analysts):
            raise ValueError("council analyst aliases must be distinct")
        if len({t.task_id for t in self.tasks}) != len(self.tasks):
            raise ValueError("council task IDs must be distinct")
        if any(t.reference_answer is not None for t in self.tasks):
            raise ValueError("simulator-only council tasks cannot include reference answers")
        if len({fragment.fragment_id for fragment in self.actor_context}) != len(
            self.actor_context
        ):
            raise ValueError("actor context fragment identities must be distinct")
        if self.memory is not None and self.retrieval is None:
            raise ValueError("frozen actor memory requires a registered JEV selector")
        return self


class PreparedCouncil(ContractModel):
    """Credential-free immutable council registration.

    Attributes:
        id: Stable run identity.
        spec: Frozen input and resource configuration.
        catalog_sha256: Exact credential-reference catalog identity.
        code_revision: Producing source identity.
        definition: Read-only privileged definition supplied only to analysts.
    """

    id: str
    spec: TraceCouncilSpec
    catalog_sha256: str
    code_revision: str
    definition: JsonObject


def validate_council_models(
    catalog: ModelCatalog, spec: TraceCouncilSpec, *, simulator_alias: str
) -> None:
    """Check every completion role before any preceding fidelity work is purchased.

    Args:
        catalog: Credential-free model catalog frozen by the caller.
        spec: Actor, simulator, analyst and judge limits to validate.
        simulator_alias: Serving alias of the selected simulator artifact.

    Raises:
        ValueError: Any role lacks the required interface, capacity or model identity.
    """
    limits = [(a.alias, spec.actor_output_tokens, True) for a in spec.actors]
    limits += [(alias, spec.analyst_output_tokens, True) for alias in spec.analysts]
    limits += [
        (simulator_alias, spec.simulator_output_tokens, True),
        (spec.judge_alias, spec.judge_output_tokens, spec.judge_output_protocol == "tool"),
    ]
    for alias, output, needs_tools in limits:
        record = catalog.models.get(alias)
        if (
            record is None
            or record.capabilities is None
            or (needs_tools and record.capabilities.supports_tools is not True)
        ):
            raise ValueError(f"model {alias!r} needs declared tool capabilities")
        if catalog.connections[record.connection].provider != "openai-compatible":
            raise ValueError(f"model {alias!r} needs an OpenAI-compatible completion connection")
        caps = record.capabilities
        if caps.context_window_tokens is None or (
            caps.maximum_output_tokens is not None and output > caps.maximum_output_tokens
        ):
            raise ValueError(f"model {alias!r} needs compatible context/output capacity")
    if catalog.models[spec.judge_alias].model != spec.judge.model_identity:
        raise ValueError("judge alias must match the frozen rubric model identity")


def prepare_trace_council(
    project: ProjectStore, spec: TraceCouncilSpec, *, code_revision: str
) -> PreparedCouncil:
    """Freeze the exact simulator, verifier and harness before resolving any credentials.

    Args:
        project: Existing reconstructed simulation project.
        spec: Complete simulator-only council configuration.
        code_revision: Source revision or source fingerprint for this run.

    Returns:
        Persisted registration, or its identical existing instance.
    """
    identity = stable_id("trace-council", {"run_id": spec.run_id})
    if project.artifacts.exists(identity):
        saved = PreparedCouncil.model_validate_json(
            project.artifacts.read_bytes(identity, "plan.json")
        )
        if saved.spec != spec or saved.code_revision != code_revision:
            raise ValueError("council registration changed; use its saved inputs or a fresh run ID")
        if spec.memory is not None:
            context = read_council_memory(
                project.artifacts,
                spec.memory,
                spec.world_model,
                tuple(t.lineage_group_id for t in spec.tasks),
            )
            if saved.definition.get("memory") != {
                "snapshot": context.snapshot.model_dump(mode="json"),
                "context_sha256": sha256_json(context),
            }:
                raise ValueError("frozen actor memory changed since registration")
            _reserve_actor_tasks(project, spec, code_revision)
        return saved
    artifact = load_grounded_world_model_artifact(project.artifacts, spec.world_model)
    spec.actor_harness.preflight()
    task_definition = None
    if spec.task_bundle is not None:
        task_definition = load_evaluation_task(project.artifacts, spec.task_bundle)
        checkpoint = load_world_checkpoint(project.artifacts, task_definition.checkpoint)
        if (
            task_definition.world_model != spec.world_model
            or len(spec.tasks) != 1
            or spec.tasks[0] != task_definition.task
            or checkpoint.world_model != spec.world_model
            or checkpoint.public_context != spec.tasks[0].initial_context
            or spec.actor_context != task_definition.context_fragments
        ):
            raise ValueError("council task, world, context or checkpoint differs from its bundle")
    catalog = load_model_catalog(project.model_catalog_path)
    index_embedder_alias(project, catalog, artifact)
    for task in spec.tasks:
        require_unindexed_held_out_task(project, artifact, task)
    validate_council_models(catalog, spec, simulator_alias=artifact.model_alias)
    bound = artifact.selection_policy_sha256
    if bound is not None and (spec.retrieval is None or spec.retrieval.jev.identity != bound):
        raise ValueError("simulator requires its registered JEV selection policy")
    if spec.retrieval is not None:
        alias = spec.retrieval.jev_alias
        if (
            alias not in catalog.models
            or catalog.connections[catalog.models[alias].connection].provider != "typesafe"
        ):
            raise ValueError("JEV requires a configured native decision alias")
    tool_fidelity = load_tool_fidelity(
        project,
        spec.analyst_fidelity_plans,
        spec.world_model,
        tuple(task.lineage_group_id for task in spec.tasks),
    )
    for reference in spec.analyst_native_episodes:
        _native_episode(project, reference)
    verifier_source = inspect.getsource(mechanical_checks)
    definition: JsonObject = {
        "analysis_protocol_sha256": sha256_json(
            {"prompt": ANALYSIS_PROMPT, "schema": TraceAnalysis.model_json_schema()}
        ),
        "tool_fidelity": tool_fidelity,
        "native_sources": [ref.model_dump(mode="json") for ref in spec.analyst_native_episodes],
        "simulator": artifact.model_dump(mode="json"),
        "tasks": [t.model_dump(mode="json") for t in spec.tasks],
        "verifier": {
            "judge": spec.judge.model_dump(mode="json"),
            "mechanical_check_source": verifier_source,
            "mechanical_check_sha256": sha256_json(verifier_source),
            "reference_capability": "description_only",
            "limitation": (
                "Judge used description-only evidence. No live native environment. "
                "Matched archived replies and held-out fidelity are analyst-only inputs."
            ),
        },
        "harness": {
            "agent": (
                "exp.runtime.agents.chat.ChatAgentRuntime"
                if spec.actor_harness.kind == "chat"
                else "exp.runtime.agents.pi.PiAgentRuntime"
            ),
            "protocol": COUNCIL_HARNESS_PROTOCOL,
            "adapter": spec.actor_harness.model_dump(mode="json"),
            "installation_scope": (
                "declared entrypoint and dependency pins; no complete transitive dependency "
                "or deployed filesystem/network containment claim"
            ),
            "capture": "exp.simulation.actor.run_actor_workflow",
            "session": "exp.simulation.world_model.durable.DurableWorldModelSession",
            "maximum_actor_turns": spec.maximum_actor_turns,
            "maximum_tool_operations": spec.maximum_tool_operations,
            "actor_output_tokens": spec.actor_output_tokens,
            "simulator_output_tokens": spec.simulator_output_tokens,
            "output_protocol": spec.output_protocol,
            "state_policy": "fresh world per actor/task; carried candidate state; no native rescue",
            "knowledge_policy": "frozen simulator artifact; no learning, promotion or deployment",
            "tier_labels": "operator declarations; not externally established capability rankings",
        },
    }
    if task_definition is not None:
        definition["task_verifier"] = task_definition.verifier.model_dump(mode="json")
        definition["task_intent"] = task_definition.intent
        definition["task_limitations"] = list(task_definition.known_limits)
    memory_inputs: tuple[ArtifactInput, ...] = ()
    if spec.memory is not None:
        context = read_council_memory(
            project.artifacts,
            spec.memory,
            spec.world_model,
            tuple(t.lineage_group_id for t in spec.tasks),
        )
        definition["memory"] = {
            "snapshot": context.snapshot.model_dump(mode="json"),
            "context_sha256": sha256_json(context),
        }
        memory_inputs = (context.snapshot,)
    prepared = PreparedCouncil(
        id=identity,
        spec=spec,
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        code_revision=code_revision,
        definition=definition,
    )
    envelope = ArtifactEnvelope(
        schema_version=1,
        created_at=datetime.now(UTC),
        inputs=tuple(
            sorted(
                (
                    spec.world_model,
                    *((spec.task_bundle,) if spec.task_bundle is not None else ()),
                    *memory_inputs,
                    *spec.analyst_fidelity_plans,
                    *spec.analyst_native_episodes,
                ),
                key=lambda value: value.artifact_id,
            )
        ),
        code_revision=code_revision,
    )
    _reserve_actor_tasks(project, spec, code_revision)
    project.artifacts.write_json(
        artifact_id=identity,
        artifact_type="trace-council-plan",
        envelope=envelope,
        files={"plan.json": prepared.model_dump(mode="json")},
    )
    return prepared


def _reserve_actor_tasks(project: ProjectStore, spec: TraceCouncilSpec, revision: str) -> None:
    """Refuse already learned task lineages and reserve all validated actor tasks."""
    if spec.memory is None:
        return
    tasks = {task.lineage_group_id for task in spec.tasks}
    consumed = consumed_task_lineages(
        project.artifacts, memory_head(project.artifacts, spec.memory.lineage)
    )
    if tasks & consumed:
        raise ValueError("actor evaluation task lineage was already consumed by memory")
    for task_lineage in sorted(tasks):
        reserve_task_lineage(
            project.artifacts,
            spec.memory.lineage,
            task_lineage,
            created_at=datetime.now(UTC),
            code_revision=revision,
        )


def saved_trace_council(project: ProjectStore, prepared: PreparedCouncil) -> JsonObject | None:
    """Read a completed immutable report without provider or environment access."""
    identity = prepared.id + "-report"
    if not project.artifacts.exists(identity):
        return None
    return json.loads(project.artifacts.read_bytes(identity, "report.json"))


def execute_trace_council(
    project: ProjectStore,
    prepared: PreparedCouncil,
    *,
    clock: Callable[[], datetime] | None = None,
    progress: ProgressHook | None = None,
) -> JsonObject:
    """Run the maintained actor, simulator and evaluator, then analyze all saved traces.

    Args:
        project: Project owning all immutable artifacts and session stores.
        prepared: Previously frozen council registration.
        clock: Aware clock shared with provider and durable session owners.
        progress: Existing terminal progress callback.

    Returns:
        Findings, explicit failures and resource counts; never an acceptance decision.
    """
    now = clock or (lambda: datetime.now(UTC))
    saved = PreparedCouncil.model_validate_json(
        project.artifacts.read_bytes(prepared.id, "plan.json")
    )
    if saved != prepared:
        raise ValueError("execution differs from the saved council registration")
    with CoordinationLock(project.artifacts.project_directory / (prepared.id + ".lock"), timeout=0):
        previous = saved_trace_council(project, prepared)
        if previous is not None:
            return previous
        if project.artifacts.exists(prepared.id + "-execution"):
            raise ValueError(
                "incomplete council; inspect saved attempts before authorizing another run"
            )
        spec = prepared.spec
        spec.actor_harness.preflight()
        if prepared.definition.get("analysis_protocol_sha256") != sha256_json(
            {"prompt": ANALYSIS_PROMPT, "schema": TraceAnalysis.model_json_schema()}
        ):
            raise ValueError("analyst protocol changed; prepare a fresh council run")
        if now() >= spec.deadline:
            raise ValueError("council deadline expired before dispatch")
        catalog = load_model_catalog(project.model_catalog_path)
        if sha256_json(catalog.model_dump(mode="json")) != prepared.catalog_sha256:
            raise ValueError("model catalog changed; register a fresh council")
        directory = project.artifacts.project_directory
        fingerprint = sha256_json(prepared.model_dump(mode="json"))
        ledger = DispatchLedger(
            directory / (prepared.id + "-physical.sqlite"),
            spec.maximum_physical_attempts,
            fingerprint,
        )
        transport = AdmittedJsonTransport(
            HttpxJsonTransport(),
            ledger=ledger,
            artifacts=project.artifacts,
            code_revision=prepared.code_revision,
            deadline=spec.deadline,
            clock=now,
        )
        registry = RuntimeModelCatalog(catalog, transport_factory=lambda: transport)
        progress_lock = Lock()
        receipt_lock = Lock()

        def emit(message: str) -> None:
            """Serialize terminal updates while independent cells execute concurrently."""
            with progress_lock:
                report(progress, message)

        budget = RequestBudget(project, identity=fingerprint, maximum_cost_usd=None)

        def completion(
            alias: str,
            output: int,
            *,
            structured: bool,
            runtime_registry: RuntimeModelCatalog | None = None,
        ) -> StructuredCompletion:
            """Reuse the maintained request, physical-attempt and accounting owners."""
            resolved = (runtime_registry or registry).resolve(alias)
            return StructuredCompletion(
                client=resolved.client,
                plan=CompletionPlan(
                    model=resolved.snapshot,
                    capabilities=resolved.capabilities,
                    deadline=spec.deadline,
                    output_tokens=output,
                    maximum_cost_per_call=spec.request_reservation_usd,
                    code_revision=prepared.code_revision,
                    output_protocol="tool" if structured else "json",
                ),
                budget=budget,
                artifacts=project.artifacts,
                clock=now,
            )

        artifact = load_grounded_world_model_artifact(project.artifacts, spec.world_model)
        checkpoint = None
        if spec.task_bundle is not None:
            task_definition = load_evaluation_task(project.artifacts, spec.task_bundle)
            checkpoint = load_world_checkpoint(project.artifacts, task_definition.checkpoint)
        memory_context = None
        if spec.memory is not None:
            memory_context = read_council_memory(
                project.artifacts,
                spec.memory,
                spec.world_model,
                tuple(t.lineage_group_id for t in spec.tasks),
            )
            if prepared.definition.get("memory") != {
                "snapshot": memory_context.snapshot.model_dump(mode="json"),
                "context_sha256": sha256_json(memory_context),
            }:
                raise ValueError("frozen actor memory changed since registration")

        def record(receipt: JsonObject) -> None:
            """Publish duplicate JEV receipts once across concurrent cells."""
            identity = stable_id("jev-receipt", {"run": prepared.id, "receipt": receipt})
            with receipt_lock:
                if not project.artifacts.exists(identity):
                    _write(
                        project, prepared, identity, "jev-receipt", "receipt.json", receipt, now()
                    )

        _write(
            project,
            prepared,
            prepared.id + "-execution",
            "trace-council-execution",
            "execution.json",
            {"plan_sha256": fingerprint, "native_environment_accessed": False},
            now(),
        )

        def run_cell(item: tuple[TaskCase, CouncilActor]) -> CouncilTrace:
            """Own one sequential world and its clients; retain every failed planned cell."""
            task, actor = item
            cell = stable_id(
                "council-cell", {"run": prepared.id, "task": task.task_id, "actor": actor.alias}
            )
            emit(f"Actor {actor.alias}: {task.task_id}")
            trace = CouncilTrace(
                cell_id=cell, actor_alias=actor.alias, tier=actor.tier, task_id=task.task_id
            )
            context_client: ContextRecordingClient | None = None
            try:
                cell_registry = RuntimeModelCatalog(catalog, transport_factory=lambda: transport)
                selector, memory_selector = None, None
                if spec.retrieval is not None:
                    decide = decision_caller(
                        cell_registry.resolve(spec.retrieval.jev_alias),
                        transport,
                        spec.deadline,
                        now,
                    )
                    if artifact.selection_policy_sha256 is not None:
                        selector = JEVSelector(
                            policy=spec.retrieval.jev, decide=decide, record=record
                        )
                    if memory_context is not None:
                        memory_selector = JEVCueSelector(
                            policy=spec.retrieval.jev,
                            top_k=spec.retrieval.memory_top_k,
                            decide=decide,
                            record=record,
                        )
                resolved = cell_registry.resolve(artifact.model_alias)
                runtime = load_grounded_world_model(
                    project.artifacts,
                    spec.world_model.artifact_id,
                    client=resolved.client,
                    capabilities=resolved.capabilities,
                    embedder=bind_index_embedder(
                        cell_registry,
                        index_embedder_alias(project, catalog, artifact),
                        artifact,
                        project,
                        budget,
                    ),
                    evidence_selector=selector,
                )
                simulator = completion(
                    artifact.model_alias,
                    spec.simulator_output_tokens,
                    structured=False,
                    runtime_registry=cell_registry,
                )
                judge = completion(
                    spec.judge_alias,
                    spec.judge_output_tokens,
                    structured=spec.judge_output_protocol == "tool",
                    runtime_registry=cell_registry,
                )
                public = task.model_copy(update={"task_id": cell})
                session = open_grounded_session(
                    runtime=runtime,
                    task=public,
                    store=SessionStore(directory / (cell + ".sqlite"), clock=now),
                    artifacts=project.artifacts,
                    session_id=cell,
                    policy=SessionPolicy(
                        deadline=spec.deadline,
                        maximum_operations=spec.maximum_tool_operations,
                        inference_identity=fingerprint,
                        code_revision=prepared.code_revision,
                    ),
                    complete=ModelCompletion(simulator),
                    lock_path=directory / (cell + ".lock"),
                    maximum_output_tokens=spec.simulator_output_tokens,
                    output_protocol=spec.output_protocol,
                    research_selection=memory_selector,
                    research_prior=None if memory_context is None else memory_context.records,
                    research_prior_identity=None
                    if memory_context is None
                    else memory_context.identity,
                    research_overview=() if memory_context is None else memory_context.overview,
                    research_prior_sources=None
                    if memory_context is None
                    else memory_context.sources,
                    research_overview_sources=()
                    if memory_context is None
                    else memory_context.overview_sources,
                    clock=now,
                    checkpoint=checkpoint,
                )
                actor_task = with_actor_context(public, spec.actor_context)
                context_client = ContextRecordingClient(
                    completion(
                        actor.alias,
                        spec.actor_output_tokens,
                        structured=False,
                        runtime_registry=cell_registry,
                    ).as_model_client(cell),
                    actor_task,
                )
                workflow = run_actor_workflow(
                    session,
                    actor_task,
                    agent=create_actor_runtime(
                        spec.actor_harness,
                        maximum_model_calls=spec.maximum_actor_turns,
                        maximum_output_tokens=spec.actor_output_tokens,
                    ),
                    model=context_client,
                    artifacts=project.artifacts,
                    clock=now,
                    code_revision=prepared.code_revision,
                )
                trace = trace.model_copy(
                    update={"workflow": workflow, "error_type": workflow.error_type}
                )
                emit(f"Judge {actor.alias}: {task.task_id}")
                evaluation = evaluate(
                    workflow.evidence,
                    EvaluationReferences(
                        capability="description_only",
                        description=(
                            "Simulator-only evaluation. Actual environment and "
                            "independent native outcomes unavailable."
                        ),
                    ),
                    judge=spec.judge,
                    complete=judge,
                    artifacts=project.artifacts,
                    artifact_id=cell + "-evaluation",
                    clock=now,
                    code_revision=prepared.code_revision,
                )
                trace = trace.model_copy(update={"evaluation": evaluation})
            except Exception as error:  # noqa: BLE001 - retain every planned failed cell.
                trace = trace.model_copy(update={"error_type": type(error).__name__})
            trace = trace.model_copy(
                update={
                    "context_manifest": {} if context_client is None else context_client.manifest()
                }
            )
            _write(
                project,
                prepared,
                cell + "-trace",
                "council-trace",
                "trace.json",
                trace.model_dump(mode="json"),
                now(),
            )
            return trace

        cells = [(task, actor) for task in spec.tasks for actor in spec.actors]
        if spec.maximum_concurrency == 1:
            traces = [run_cell(cell) for cell in cells]
        else:
            with ThreadPoolExecutor(max_workers=min(spec.maximum_concurrency, len(cells))) as pool:
                traces = list(pool.map(run_cell, cells))
        analysis_definition = dict(prepared.definition)
        episodes = tuple(_native_episode(project, ref) for ref in spec.analyst_native_episodes)
        tasks = {task.task_id: task for task in spec.tasks}
        analysis_definition["counterfactuals"] = [
            row
            for trace in traces
            if trace.workflow is not None
            for row in matched_native_replies(
                tasks[trace.task_id], trace.workflow.evidence.steps, episodes, cell_id=trace.cell_id
            )
        ]
        _write(
            project,
            prepared,
            prepared.id + "-analysis-evidence",
            "council-analysis-evidence",
            "evidence.json",
            analysis_definition,
            now(),
        )
        analyses: dict[str, TraceAnalysis] = {}
        errors: dict[str, str] = {}
        for alias in spec.analysts:
            emit(f"Trace analysis: {alias}")
            identity = stable_id("council-analysis", {"run": prepared.id, "analyst": alias})
            try:
                analysis = analyze_traces(
                    analysis_definition,
                    traces,
                    complete=completion(alias, spec.analyst_output_tokens, structured=True),
                    logical_id=identity,
                )
                analyses[alias] = analysis
                _write(
                    project,
                    prepared,
                    identity,
                    "trace-analysis",
                    "analysis.json",
                    analysis.model_dump(mode="json"),
                    now(),
                )
            except Exception as error:  # noqa: BLE001 - unavailable analyses are explicit.
                errors[alias] = type(error).__name__
        result: JsonObject = {
            **aggregate_analyses(analyses),
            "run_id": spec.run_id,
            "definition_sha256": sha256_json(prepared.definition),
            "analysis_evidence_sha256": sha256_json(analysis_definition),
            "analysis_evidence": prepared.id + "-analysis-evidence",
            "planned_actor_cells": len(spec.tasks) * len(spec.actors),
            "maximum_concurrency": spec.maximum_concurrency,
            "traces": [t.model_dump(mode="json") for t in traces],
            "analyses": {alias: value.model_dump(mode="json") for alias, value in analyses.items()},
            "analysis_failures": errors,
            "physical_attempts": ledger.used(),
            "maximum_physical_attempts": spec.maximum_physical_attempts,
            "cost_status": "provider receipts retained; unknown charges remain unknown",
        }
        _write(
            project,
            prepared,
            prepared.id + "-report",
            "trace-council-report",
            "report.json",
            result,
            now(),
            inputs=tuple(
                artifact_input(project.artifacts.read(identity).manifest)
                for attempt in ledger.attempt_ids()
                for suffix in ("intent", "response", "failure", "rejected", "not-dispatched")
                if project.artifacts.exists(
                    identity := "provider-attempt-" + attempt + "-" + suffix
                )
            ),
        )
        return result


def _write(
    project: ProjectStore,
    prepared: PreparedCouncil,
    identity: str,
    kind: str,
    path: str,
    payload: JsonObject,
    now: datetime,
    *,
    inputs: tuple[ArtifactInput, ...] = (),
) -> None:
    """Write immutable council evidence with its exact registration and attempt receipts."""
    project.artifacts.write_json(
        artifact_id=identity,
        artifact_type=kind,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=now,
            code_revision=prepared.code_revision,
            inputs=tuple(
                sorted(
                    {artifact_input(project.artifacts.read(prepared.id).manifest), *inputs},
                    key=lambda ref: ref.artifact_id,
                )
            ),
        ),
        files={path: payload},
    )


def _native_episode(project: ProjectStore, reference: ArtifactInput) -> RecordedEpisode:
    """Read a manifest-verified archive, never an environment or an actor input."""
    manifest = project.artifacts.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != "archived-native-episode":
        raise ValueError("native archive manifest differs; supply the exact archived episode")
    episode = RecordedEpisode.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "episode.json")
    )
    if episode.episode_id != reference.artifact_id:
        raise ValueError("native episode identity differs from its source artifact")
    return episode
