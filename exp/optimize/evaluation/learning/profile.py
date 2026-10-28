"""Actor evaluation reads the build's learned simulator memory, frozen at its latest version.

``exp build`` learns every trace into one memory (see ``trace_learning``). Actor evaluation
pins that memory's latest version when it is prepared and delivers it, read-only, to every
simulated world-model turn: no publication, no feedback, no online research. Recall uses the
learning plan's alias and policy, and the overview is the version's pinned overview.

Evaluation tasks are mined from the same traces the memory learned, so each task reads memory
with its own task lineage hidden (and the lineage of every source trace it was mined from), the
same exclusion learning applies; lineages reserved for evaluation stay withheld, and a held-out
task reads only memory of known provenance. The pinned memory is part of the prepared
evaluation, so it enters the plan identity: saved results replay only for identical memory.
"""

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import AwareDatetime, Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelCatalog, ModelSnapshot
from exp.common.project import ArtifactAlreadyExistsError, ProjectStore, artifact_input
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.fidelity.memory import refuse_ineligible_memory
from exp.optimize.evaluation.fidelity.runner import FidelityRetrieval
from exp.optimize.evaluation.learning.traces import Scope, learning_plan, recorded_plan
from exp.optimize.router.evaluation.build import completed_project_build
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.call_evidence import canonical_call
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.transport import (
    HttpxJsonTransport,
    JsonHttpTransport,
    ProviderTransportError,
    RetryPolicy,
    classify_unrepeatable_retry,
    run_with_retry,
)
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.retrieval.build_inputs import load_completed_build_rag_lineage_bindings
from exp.simulation.retrieval.jev import JEVCueSelector
from exp.simulation.world_model.artifact import GroundedWorldModelArtifact
from exp.simulation.world_model.experience_memory import (
    Eligibility,
    memory_head,
    read_memory_version,
    require_memory_scope,
    reserved_task_lineages,
)
from exp.simulation.world_model.frozen_memory import FrozenMemory
from exp.simulation.world_model.overview_pin import pinned_summaries

NO_MEMORY_NOTE = (
    "Simulator memory: none (the simulator has no learned memory; add a native decision "
    "model to the catalog and run exp build again to learn one)"
)
_DECISION = "evaluation-recall-decision"


class EvaluationMemory(ContractModel):
    """The learned memory an actor evaluation reads, pinned in the evaluation's identity.

    Attributes:
        lineage: Memory lineage the build learned into.
        version: Frozen version every simulated turn reads; evaluation never publishes.
        world_model: Simulator the memory was learned on, the build's world model.
        recall: The learning plan's recall alias and policy.
        recall_model: Exact native decision model behind the recall alias.
        overview_lines: Line budget of the overview delivered with every turn.
    """

    lineage: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    version: int = Field(ge=1, strict=True)
    world_model: ArtifactInput
    recall: FidelityRetrieval
    recall_model: ModelSnapshot
    overview_lines: int = Field(default=32, ge=1, strict=True)

    def line(self) -> str:
        """One line naming the memory version, its lineage and the recall model."""
        return (
            f"Simulator memory: v{self.version} ({self.lineage}, "
            f"recall {self.recall.jev_alias}, top {self.recall.memory_top_k})"
        )


def learned_evaluation_memory(
    project: ProjectStore, catalog: ModelCatalog, scope: Scope = "all"
) -> EvaluationMemory | None:
    """Pin the build's learned memory at its latest version, or None when nothing was learned.

    Args:
        project: Project with a completed build.
        catalog: Credential-free catalog naming the recall model.
        scope: ``all`` pins the memory ``exp build`` learned from every trace; ``fit`` pins
            the fit-only memory ``exp eval --audit`` learned, which never consumed a held-out
            lineage and is read over the fit-only index.

    Returns:
        The memory every simulated turn reads, or None without learned memory.

    Raises:
        ValueError: The memory's recall alias is no longer configured.
    """
    if not recorded_plan(project, scope):
        return None
    plan = learning_plan(project, None, scope)
    refuse_ineligible_memory(project, plan.lineage)
    head = memory_head(project.artifacts, plan.lineage)
    if head is None:
        return None
    alias = plan.recall.jev_alias
    if alias not in catalog.models:
        raise ValueError(
            f"the simulator's memory recalls with {alias!r}, which is not configured; add "
            "that decision model back to the catalog, or run exp build again"
        )
    return EvaluationMemory(
        lineage=plan.lineage,
        version=head.version,
        world_model=plan.world_model,
        recall=plan.recall,
        recall_model=RuntimeModelCatalog(catalog, environment={}).snapshot(alias)[0],
    )


def bind_evaluation_memory(
    project: ProjectStore,
    memory: EvaluationMemory,
    *,
    catalog: RuntimeModelCatalog,
    artifact: GroundedWorldModelArtifact,
    world_model: ArtifactInput,
    created_at: datetime,
    code_revision: str,
    decision_transport: JsonHttpTransport | None = None,
    research_history: bool = True,
) -> FrozenMemory:
    """Bind the pinned version read-only, after checking it still matches the simulator.

    Args:
        project: Project owning the memory and the build.
        memory: Memory pinned when the evaluation was prepared.
        catalog: Runtime catalog resolving the recall model's credentials.
        artifact: The evaluated simulator's grounded world model.
        world_model: That simulator's manifest reference.
        created_at: Evaluation time, recorded only by the version's first overview pin.
        code_revision: Recording implementation identity.

    Returns:
        Frozen memory handing each task its eligible view.

    Raises:
        ValueError: The memory, simulator or recall model differs from the prepared one.
    """
    if memory.world_model != world_model:
        raise ValueError("evaluation memory belongs to a different simulator; prepare again")
    refuse_ineligible_memory(project, memory.lineage, checkpoint=memory.version)
    version = read_memory_version(project.artifacts, memory.lineage, memory.version)
    selector = JEVCueSelector(
        policy=memory.recall.jev,
        top_k=memory.recall.memory_top_k,
        maximum_input_bytes=memory.recall.memory_input_bytes,
        decide=_recorded_decisions(project, memory, catalog, code_revision, decision_transport),
        record=_selection_recorder(project, memory, code_revision),
    )
    require_memory_scope(
        version,
        world_model=world_model,
        artifact=artifact,
        selection_policy_sha256=selector.identity,
    )
    build = completed_project_build(project)
    lineage_of = {
        row.trace_id: row.lineage_id
        for row in load_completed_build_rag_lineage_bindings(project.artifacts, build)
    }
    reserved = set(reserved_task_lineages(project.artifacts, memory.lineage))

    def eligibility(task: TaskCase) -> Eligibility:
        """Hide the task's lineage and its source traces' lineages; withhold reservations."""
        own = {lineage_of[t] for t in task.source_trace_ids if t in lineage_of}
        return Eligibility(
            reserved_task_lineages=tuple(sorted((reserved | own) - {task.lineage_group_id})),
            hidden_task_lineage=task.lineage_group_id,
            strict=task.partition == "held_out",
        )

    return FrozenMemory(
        project.artifacts,
        version,
        selection=selector,
        eligibility=eligibility,
        overview_lines=memory.overview_lines,
        pinned=pinned_summaries(
            project.artifacts, version, created_at=created_at, code_revision=code_revision
        ),
        record=_selection_recorder(project, memory, code_revision),
        research_history=research_history,
    )


class RecallRecovery(ContractModel):
    """Explicit authority for one replacement of a lost read-only recall decision.

    Attributes:
        intent: Exact original request whose external outcome remains unknown.
        deadline: Absolute cutoff for the single new physical attempt.
        rationale: Operator acknowledgement of possible duplicate billing and selection bias.
    """

    intent: ArtifactInput
    deadline: AwareDatetime
    rationale: str = Field(min_length=1)


def recover_recall_decision(
    project: ProjectStore,
    memory: EvaluationMemory,
    request: DecisionRequest,
    *,
    recovery: RecallRecovery,
    catalog: RuntimeModelCatalog,
    code_revision: str,
    transport: JsonHttpTransport | None = None,
) -> JsonObject:
    """Explicitly recover one lost recall, preserving its identity and ambiguity receipts.

    A completed replacement replays. An interrupted replacement is never dispatched again
    by this permit. This is a development re-attempt, not a new independent measurement.
    """
    return _recorded_decisions(
        project, memory, catalog, code_revision, transport, recovery=recovery
    )(request)


def _recorded_decisions(
    project: ProjectStore,
    memory: EvaluationMemory,
    catalog: RuntimeModelCatalog,
    code_revision: str,
    transport: JsonHttpTransport | None = None,
    *,
    recovery: RecallRecovery | None = None,
) -> Callable[[DecisionRequest], JsonObject]:
    """Native recall decisions, each saved once and replayed for an identical request.

    Recall is bound to the complete pinned memory, model, request and code revision.
    Saved outcomes replay; retained intents prevent repeating interrupted purchases.
    """
    if catalog.snapshot(memory.recall.jev_alias)[0] != memory.recall_model:
        raise ValueError(f"recall model {memory.recall.jev_alias!r} changed; prepare again")
    client = catalog.resolve(memory.recall.jev_alias).client
    profile_of = getattr(client, "gateway_wire_profile", None)
    profile = None if profile_of is None else profile_of()
    if profile is None or profile.dialect != "typesafe_systemone" or not profile.decisions_url:
        raise ValueError("the recall alias does not expose the native decision wire")
    url = profile.decisions_url
    transport = transport if transport is not None else HttpxJsonTransport()
    retry_policy = RetryPolicy()
    request_timeout = 120.0
    lock_timeout = retry_policy.maximum_attempts * (
        request_timeout + retry_policy.maximum_delay_seconds
    )
    if isinstance(transport, AdmittedJsonTransport):
        transport = transport.priced_as(memory.recall.jev_alias)

    def post(request: DecisionRequest, active_transport: JsonHttpTransport) -> JsonObject:
        """One decision attempt over a wire that defines no idempotency key.

        Only a trusted certified non-dispatch refusal may retry; an HTTP status alone
        does not prove non-execution. Unknown outcomes are never resent.
        """
        response = active_transport.post(
            url,
            headers=profile.headers,
            payload=request.provider_body(profile.model_id),
            timeout_seconds=request_timeout,
        )
        if response.status_code != 200 or not isinstance(response.body, dict):
            raise ProviderTransportError(
                f"recall decision returned HTTP {response.status_code}",
                status_code=response.status_code,
                known_unbilled=response.known_unbilled,
            )
        return response.body

    def decide(request: DecisionRequest) -> JsonObject:
        """Replay a saved decision, or make and save it."""
        request_sha256 = sha256_json(request.model_dump(mode="json"))
        identity = stable_id(
            _DECISION,
            {"memory": sha256_json(memory), "request": request_sha256, "code": code_revision},
        )
        store = project.artifacts
        with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=lock_timeout):
            if not store.exists(identity):
                intent = identity + "-intent"
                recovering = store.exists(intent)
                if recovering and recovery is None:
                    raise ValueError(
                        "recall decision has an unknown outcome; use recover_recall_decision "
                        "with its exact intent and an explicit single-attempt permit"
                    )
                if recovery is not None:
                    if (
                        not recovering
                        or artifact_input(store.read(intent).manifest) != recovery.intent
                    ):
                        raise ValueError(
                            "recall recovery requires the exact unknown original intent"
                        )
                    if datetime.now(UTC) >= recovery.deadline:
                        raise ValueError("recall recovery deadline expired before dispatch")
                    original = json.loads(store.read_bytes(intent, "request.json"))
                    if original != {
                        "memory": memory.model_dump(mode="json"),
                        "request_sha256": request_sha256,
                        "request": request.model_dump(mode="json"),
                    }:
                        raise ValueError("recall recovery request differs from its original intent")
                    if store.exists(identity + "-recovery"):
                        raise ValueError(
                            "recall recovery outcome is unknown; no second replacement"
                        )
                envelope = ArtifactEnvelope(
                    schema_version=1,
                    created_at=datetime.now(UTC),
                    code_revision=code_revision,
                    inputs=(memory.world_model,),
                )
                if not recovering:
                    store.write_json(
                        artifact_id=intent,
                        artifact_type="evaluation-recall-intent",
                        envelope=envelope,
                        files={
                            "request.json": {
                                "memory": memory.model_dump(mode="json"),
                                "request_sha256": request_sha256,
                                "request": request.model_dump(mode="json"),
                            }
                        },
                    )
                started = time.monotonic()
                dependencies = (artifact_input(store.read(intent).manifest),)
                if recovery is None:
                    body = run_with_retry(
                        lambda: post(request, transport),
                        policy=retry_policy,
                        classify=classify_unrepeatable_retry,
                    )
                else:
                    recovery_manifest = store.write_json(
                        artifact_id=identity + "-recovery",
                        artifact_type="evaluation-recall-recovery",
                        envelope=envelope.model_copy(update={"inputs": dependencies}),
                        files={
                            "recovery.json": {
                                "permit": recovery.model_dump(mode="json"),
                                "measurement": (
                                    "development re-attempt; original outcome remains unknown"
                                ),
                                "maximum_new_physical_attempts": 1,
                            }
                        },
                    )
                    recovery_ref = artifact_input(recovery_manifest)
                    admitted = AdmittedJsonTransport(
                        transport,
                        ledger=DispatchLedger(
                            store.project_directory / (identity + "-recovery.sqlite"),
                            1,
                            sha256_json(recovery),
                        ),
                        artifacts=store,
                        code_revision=code_revision,
                        deadline=recovery.deadline,
                    ).priced_as(memory.recall.jev_alias)
                    with canonical_call(recovery_ref):
                        body = post(request, admitted)
                    dependencies += (recovery_ref,)

                store.write_json(
                    artifact_id=identity,
                    artifact_type=_DECISION,
                    envelope=envelope.model_copy(update={"inputs": dependencies}),
                    files={
                        "decision.json": {
                            "request_sha256": request_sha256,
                            "model_id": profile.model_id,
                            "elapsed_seconds": time.monotonic() - started,
                            "response": body,
                        }
                    },
                )

        saved = json.loads(project.artifacts.read_bytes(identity, "decision.json"))
        return saved["response"]

    return decide


def _selection_recorder(
    project: ProjectStore, memory: EvaluationMemory, code_revision: str
) -> Callable[[JsonObject], None]:
    """Retain bounded recall coverage, including a selection that makes no provider call."""

    def record(receipt: JsonObject) -> None:
        """Save each complete selection receipt once under its frozen memory identity."""
        contents: JsonObject = {
            "memory": memory.model_dump(mode="json"),
            "selection": receipt,
        }
        identity = stable_id("evaluation-recall-selection", contents)
        if project.artifacts.exists(identity):
            return
        try:
            project.artifacts.write_json(
                artifact_id=identity,
                artifact_type="evaluation-recall-selection",
                envelope=ArtifactEnvelope(
                    schema_version=1,
                    created_at=datetime.now(UTC),
                    code_revision=code_revision,
                    inputs=(memory.world_model,),
                ),
                files={"selection.json": contents},
            )
        except ArtifactAlreadyExistsError:
            pass  # Another evaluation saved this identical receipt first.

    return record
