"""Native downstream-stage quotes bound to the frozen pipeline and original prelude."""

from __future__ import annotations

from typing import Literal

from exp.common.core.artifacts import ArtifactInput, ContractModel, Sha256, sha256_json
from exp.common.project import artifact_input
from exp.optimize.evaluation.variance.bootstrap import resolve_bootstrap
from exp.optimize.evaluation.variance.council_prelude import _save
from exp.optimize.evaluation.variance.council_prelude_world import _qualified
from exp.optimize.evaluation.variance.pipeline import DEFAULT_EXECUTORS, load_stage_record
from exp.optimize.evaluation.variance.pipeline_spec import (
    STAGE_TYPE,
    STAGES,
    CostLine,
    StageContext,
    StageRecord,
    stage_identity,
)

INVENTORY_TYPE = "native-pipeline-downstream-inventory"


class DownstreamStageInventory(ContractModel):
    """Exact native whole-stage upper quote and conservative later-stage reservation.

    This is an inventory, never authority to execute a generated job or a qualified task.

    Attributes:
        qualification: Exact prelude under the original parent policy.
        stage: Native downstream stage being admitted.
        identity: Actual stage owner, derived through the entire native stage graph.
        producer_revision: Exact reviewed executor.
        contract_sha256: Complete prelude and frozen pipeline identity.
        spec_sha256: Resolved native pipeline settings.
        upstream: Every committed native earlier-stage owner, in graph order.
        costs: Unchanged current native stage planner's physical upper bounds.
        future_upper_costs: Later native stage upper bounds reserved conservatively, not a
            logical minimum and not a free allowance for future work.
    """

    qualification: ArtifactInput
    stage: Literal["families", "records"]
    identity: str
    producer_revision: str
    contract_sha256: Sha256
    spec_sha256: Sha256
    upstream: tuple[ArtifactInput, ...]
    costs: tuple[CostLine, ...]
    future_upper_costs: tuple[CostLine, ...]

    @property
    def maximum_attempts(self) -> int:
        """Return the native physical current-stage upper bound."""
        return sum(row.attempts for row in self.costs)

    @property
    def maximum_known_usd(self) -> float:
        """Keep known tariff liabilities separate from unpriced attempts."""
        return sum(row.usd for row in self.costs if row.usd is not None)

    @property
    def reserved_future_known_usd(self) -> float:
        """Retain known tariff liability of the later native upper quote."""
        return sum(row.usd for row in self.future_upper_costs if row.usd is not None)

    @property
    def reserved_future_attempts(self) -> int:
        """Reserve the native later-stage upper quote without calling it a minimum."""
        return sum(row.attempts for row in self.future_upper_costs)


def downstream_inventory(
    context: StageContext,
    qualification: ArtifactInput,
) -> DownstreamStageInventory:
    """Recompute the actual native quote from exact saved upstream stage owners.

    Raises:
        ValueError: Source, settings, graph, stage, quote or earlier-stage provenance differs.
    """
    project, revision, spec = context.project, context.code_revision, context.spec
    qualified = _qualified(project, qualification, revision)
    expected = resolve_bootstrap(project, qualified.contract.pipeline)
    plan = context.plan
    if plan is None:
        raise ValueError("downstream admission requires the native pipeline stage plan")
    if spec != expected or plan.stage not in ("families", "records"):
        raise ValueError("downstream stage differs from its frozen native pipeline")
    stage = plan.stage
    chain: list[str] = []
    owners: list[ArtifactInput] = []
    earlier = STAGES[: STAGES.index(stage)]
    if set(context.upstream) != set(earlier):
        raise ValueError("downstream stage requires every exact native upstream owner")
    for name in STAGES:
        executor = DEFAULT_EXECUTORS[name]
        identity = stage_identity(name, spec, tuple(chain), revision, executor.keys(project, spec))
        if name == stage:
            if context.identity != identity or plan.identity != identity:
                raise ValueError("downstream stage owner differs from its native graph identity")
            break
        record = load_stage_record(project, identity)
        if record is None or record != context.upstream[name] or record.stage != name:
            raise ValueError("downstream stage lost a committed native upstream owner")
        stored = project.artifacts.read(identity + "-stage")
        reference = artifact_input(stored.manifest)
        if (
            stored.manifest.artifact_type != STAGE_TYPE
            or stored.manifest.code_revision != revision
            or record.identity != identity
            or record.upstream != tuple(owners)
            or stored.manifest.inputs
            != tuple(sorted({*owners, *record.references}, key=lambda ref: ref.artifact_id))
        ):
            raise ValueError("downstream stage changed native upstream ancestry or producer")
        owners.append(reference)
        chain.append(identity)
    costs, _ = DEFAULT_EXECUTORS[stage].plan(project, spec, revision)
    if not plan.saved and plan.costs != costs:
        raise ValueError("downstream stage changed its native physical quote")
    future = (
        () if stage == "records" else DEFAULT_EXECUTORS["records"].plan(project, spec, revision)[0]
    )
    return DownstreamStageInventory(
        qualification=qualification,
        stage=stage,
        identity=context.identity,
        producer_revision=revision,
        contract_sha256=sha256_json(qualified.contract),
        spec_sha256=sha256_json(spec),
        upstream=tuple(owners),
        costs=costs,
        future_upper_costs=future,
    )


def save_downstream_inventory(context: StageContext, qualification: ArtifactInput) -> ArtifactInput:
    """Persist a genuine native stage quote without opening any paid authority."""
    inventory = downstream_inventory(context, qualification)
    identity = f"pipeline-downstream-{sha256_json(inventory)[:32]}"
    return _save(
        context.project,
        identity,
        INVENTORY_TYPE,
        "inventory.json",
        inventory,
        (qualification, *inventory.upstream),
        context.code_revision,
    )


def committed_downstream_stage(
    context: StageContext,
    inventory: DownstreamStageInventory,
    owner: ArtifactInput,
) -> StageRecord:
    """Read the real native stage record before a caller can settle its paid envelope."""
    expected = downstream_inventory(context, inventory.qualification)
    if expected != inventory:
        raise ValueError("downstream completion changed its exact native inventory")
    stored = context.project.artifacts.read(owner.artifact_id)
    record = StageRecord.model_validate_json(stored.payloads["record.json"])
    if (
        artifact_input(stored.manifest) != owner
        or owner.artifact_id != context.identity + "-stage"
        or stored.manifest.artifact_type != STAGE_TYPE
        or stored.manifest.code_revision != context.code_revision
        or record.stage != inventory.stage
        or record.identity != inventory.identity
        or record.upstream != inventory.upstream
        or stored.manifest.inputs
        != tuple(sorted({*inventory.upstream, *record.references}, key=lambda ref: ref.artifact_id))
    ):
        raise ValueError("downstream completion is not its actual committed native stage owner")
    return record
