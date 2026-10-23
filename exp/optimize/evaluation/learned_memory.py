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

from filelock import FileLock
from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.models import ModelCatalog, ModelSnapshot
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.fidelity import FidelityRetrieval
from exp.optimize.evaluation.trace_learning import learning_plan, recorded_plan
from exp.optimize.router.evaluation.build import completed_project_build
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.transport import (
    HttpxJsonTransport,
    ProviderTransportError,
    RetryPolicy,
    run_with_retry,
)
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
    project: ProjectStore, catalog: ModelCatalog
) -> EvaluationMemory | None:
    """Pin the build's learned memory at its latest version, or None when nothing was learned.

    Args:
        project: Project with a completed build.
        catalog: Credential-free catalog naming the recall model.

    Returns:
        The memory every simulated turn reads, or None without learned memory.

    Raises:
        ValueError: The memory's recall alias is no longer configured.
    """
    if not recorded_plan(project, "all"):
        return None
    plan = learning_plan(project, None, "all")
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
    version = read_memory_version(project.artifacts, memory.lineage, memory.version)
    selector = JEVCueSelector(
        policy=memory.recall.jev,
        top_k=memory.recall.memory_top_k,
        decide=_recorded_decisions(project, memory, catalog, code_revision),
        record=lambda receipt: None,
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
    )


def _recorded_decisions(
    project: ProjectStore,
    memory: EvaluationMemory,
    catalog: RuntimeModelCatalog,
    code_revision: str,
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
    transport = HttpxJsonTransport()
    retry_policy = RetryPolicy()
    request_timeout = 120.0
    lock_timeout = retry_policy.maximum_attempts * (
        request_timeout + retry_policy.maximum_delay_seconds
    )

    def post(request: DecisionRequest) -> JsonObject:
        """One decision attempt; transport failures are retried, refusals are not."""
        response = transport.post(
            url,
            headers=profile.headers,
            payload=request.provider_body(profile.model_id),
            timeout_seconds=request_timeout,
        )
        if response.status_code != 200 or not isinstance(response.body, dict):
            raise ProviderTransportError(
                f"recall decision returned HTTP {response.status_code}",
                status_code=response.status_code,
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
        with FileLock(store.project_directory / (identity + ".lock"), timeout=lock_timeout):
            if not store.exists(identity):
                intent = identity + "-intent"
                if store.exists(intent):
                    raise ValueError(
                        "recall decision has an unknown outcome; "
                        f"inspect retained intent {intent}; "
                        "automatic replay cannot purchase it again"
                    )
                envelope = ArtifactEnvelope(
                    schema_version=1,
                    created_at=datetime.now(UTC),
                    code_revision=code_revision,
                    inputs=(memory.world_model,),
                )
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
                body = run_with_retry(lambda: post(request), policy=retry_policy)
                store.write_json(
                    artifact_id=identity,
                    artifact_type=_DECISION,
                    envelope=envelope.model_copy(
                        update={"inputs": (artifact_input(store.read(intent).manifest),)}
                    ),
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
