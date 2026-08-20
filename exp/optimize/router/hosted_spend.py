"""Stage reconciliation for the hosted router provider-spend ledger."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

from exp.common.core.artifacts import ArtifactInput, stable_id
from exp.common.core.money import reserve_usd
from exp.common.evaluations.evidence import (
    read_evaluation_plan,
    read_judgment,
    read_rollout,
)
from exp.common.models import BillingSource, ModelSnapshot
from exp.common.project import (
    ProjectRouterPolicyArtifacts,
    ProjectRouterReportArtifacts,
    ProjectStage,
    ProjectStore,
)
from exp.common.rollouts import RolloutArtifact
from exp.common.rollouts.otel import RolloutEventKind
from exp.optimize.router.automatic.service import (
    AutomaticRouterArtifacts,
    AutomaticRouterPreflight,
)
from exp.optimize.router.composition import RouterPolicyLock
from exp.optimize.router.spend import (
    ProviderSpendComponent,
    ProviderSpendEntry,
    ProviderSpendLedger,
    ProviderSpendStatus,
    load_provider_spend_ledger,
    not_incurred_entry,
    persist_provider_spend_ledger,
    spend_entry_from_economics,
)


def optimization_entries(
    project: ProjectStore,
    lock: RouterPolicyLock,
    preflight: AutomaticRouterPreflight,
    artifacts: AutomaticRouterArtifacts,
    prior_entries: tuple[ProviderSpendEntry, ...],
    *,
    purpose: str,
) -> tuple[ProviderSpendEntry, ...]:
    """Collect immutable provider economics for fit-only or complete plan scope."""
    plan, _plan_input = read_evaluation_plan(project.artifacts, lock.plan.artifact_id)
    allowed_cells = {
        cell.cell_id for cell in plan.cells if purpose == "all" or cell.purpose == purpose
    }
    rollout_ids = {
        str(cell.observed_rollout_id)
        for cell in plan.cells
        if cell.cell_id in allowed_cells and cell.observed_rollout_id is not None
    }
    entries: list[ProviderSpendEntry] = list(prior_entries)
    entries.append(
        ProviderSpendEntry(
            operation_id=stable_id(
                "provider-spend-operation",
                {
                    "embedding_set_id": artifacts.router_embeddings.embedding_set_id,
                    "component": "router_embedding",
                },
            ),
            component=ProviderSpendComponent.ROUTER_EMBEDDING,
            billing_source=preflight.embedder.billing_source,
            status=ProviderSpendStatus.RESERVED,
            operation_count=1,
            amount_usd=reserve_usd(preflight.router_embedding_reservation.estimated_cost_usd),
            evidence=artifacts.router_embedding_input,
        )
    )
    for artifact_id in project.artifacts.list_ids():
        stored = project.artifacts.read(artifact_id)
        if stored.manifest.artifact_type != "rollout":
            continue
        rollout, rollout_input = read_rollout(project.artifacts, artifact_id)
        if rollout.evidence_source != "world_model" or rollout.cell_id not in allowed_cells:
            continue
        rollout_ids.add(rollout.rollout_id)
        entries.extend(_rollout_spend_entries(rollout, rollout_input))
    for artifact_id in project.artifacts.list_ids():
        stored = project.artifacts.read(artifact_id)
        if stored.manifest.artifact_type != "judgment":
            continue
        judgment, judgment_input = read_judgment(project.artifacts, artifact_id)
        if judgment.rollout_id not in rollout_ids:
            continue
        if judgment.judge_economics is None:
            raise ValueError("completed hosted judgment omits provider economics")
        entries.append(
            spend_entry_from_economics(
                operation_id=stable_id(
                    "provider-spend-operation",
                    {"judgment_id": judgment.judgment_id, "component": "judge"},
                ),
                component=ProviderSpendComponent.JUDGE,
                billing_source=judgment.judge_model.billing_source,
                economics=judgment.judge_economics,
                evidence=judgment_input,
            )
        )
    return complete_component_entries(
        tuple(entries),
        provider_spend_source_pairs(
            candidates=tuple(item.model for item in preflight.candidates),
            world_model=preflight.world_model,
            judge=preflight.judge_model,
            embedder=preflight.embedder,
        ),
    )


def stage_ledger(
    project: ProjectStore,
    *,
    selected: ProjectRouterPolicyArtifacts | ProjectRouterReportArtifacts | None,
    stage: ProjectStage,
    attempt_id: str,
    attempt_authority_sha256: str,
    ceiling_usd: Decimal,
    entries: tuple[ProviderSpendEntry, ...],
    stage_outputs: tuple[ArtifactInput, ...],
    created_at: datetime,
    code_revision: str,
) -> tuple[ProviderSpendLedger, ArtifactInput]:
    """Load an exact selected stage ledger or persist its first completed selection."""
    if selected is not None:
        ledger = load_provider_spend_ledger(project.artifacts, selected.spend_ledger)
        if (
            ledger.stage != stage
            or ledger.attempt_id != attempt_id
            or ledger.attempt_authority_sha256 != attempt_authority_sha256
            or ledger.ceiling_usd != ceiling_usd
            or ledger.entries != entries
            or ledger.stage_outputs
            != tuple(sorted(stage_outputs, key=lambda item: item.artifact_id))
            or ledger.outcome != "completed"
        ):
            raise ValueError("selected hosted stage ledger differs from exact replay")
        return ledger, selected.spend_ledger
    return persist_provider_spend_ledger(
        project.artifacts,
        project_id=project.paths.project_id,
        stage=stage,
        attempt_id=attempt_id,
        attempt_authority_sha256=attempt_authority_sha256,
        ceiling_usd=ceiling_usd,
        entries=entries,
        stage_outputs=stage_outputs,
        outcome="completed",
        created_at=created_at,
        code_revision=code_revision,
    )


def incurred_entries(
    entries: Sequence[ProviderSpendEntry],
) -> tuple[ProviderSpendEntry, ...]:
    """Drop stage-local not-incurred sentinels before adding later provider operations."""
    return tuple(item for item in entries if item.status != ProviderSpendStatus.NOT_INCURRED)


def complete_component_entries(
    entries: Sequence[ProviderSpendEntry],
    expected_pairs: Sequence[tuple[ProviderSpendComponent, BillingSource]],
) -> tuple[ProviderSpendEntry, ...]:
    """Add explicit zero-call records for every absent component and billing source."""
    result = list(entries)
    present = {(item.component, item.billing_source) for item in result}
    expected = set(expected_pairs)
    if {component for component, _source in expected} != set(ProviderSpendComponent):
        raise ValueError("provider spend source plan must represent every provider component")
    if not present.issubset(expected):
        raise ValueError("provider spend entry names a source outside the frozen source plan")
    result.extend(
        not_incurred_entry(component, billing_source)
        for component, billing_source in sorted(
            expected - present,
            key=lambda item: (item[0].value, item[1].value),
        )
    )
    return tuple(sorted(result, key=lambda item: item.operation_id))


def provider_spend_source_pairs(
    *,
    candidates: Sequence[ModelSnapshot],
    world_model: ModelSnapshot,
    judge: ModelSnapshot,
    embedder: ModelSnapshot,
) -> tuple[tuple[ProviderSpendComponent, BillingSource], ...]:
    """Return the complete alias-free billing-source plan for hosted provider work.

    Args:
        candidates: Frozen candidate model snapshots.
        world_model: Frozen grounded simulator model snapshot.
        judge: Frozen provisional judge model snapshot.
        embedder: Frozen retrieval and router embedder snapshot.

    Returns:
        Canonically sorted component and billing-source pairs, including explicit unused
        ``other_provider`` sources.
    """
    candidate_sources = {item.billing_source for item in candidates}
    all_sources = candidate_sources | {
        world_model.billing_source,
        judge.billing_source,
        embedder.billing_source,
    }
    pairs = {(ProviderSpendComponent.CANDIDATE, source) for source in candidate_sources}
    pairs.update(
        {
            (ProviderSpendComponent.WORLD_MODEL, world_model.billing_source),
            (ProviderSpendComponent.JUDGE, judge.billing_source),
            (ProviderSpendComponent.RETRIEVAL_EMBEDDING, embedder.billing_source),
            (ProviderSpendComponent.ROUTER_EMBEDDING, embedder.billing_source),
        }
    )
    pairs.update((ProviderSpendComponent.OTHER_PROVIDER, source) for source in all_sources)
    return tuple(sorted(pairs, key=lambda item: (item[0].value, item[1].value)))


def _rollout_spend_entries(
    rollout: RolloutArtifact,
    evidence: ArtifactInput,
) -> tuple[ProviderSpendEntry, ...]:
    """Convert one simulated rollout's dispatched component economics into ledger entries."""
    counts = {
        ProviderSpendComponent.CANDIDATE: sum(
            span.kind == RolloutEventKind.AGENT_MODEL_CALL for span in rollout.spans
        ),
        ProviderSpendComponent.WORLD_MODEL: sum(
            span.kind == RolloutEventKind.SIMULATOR_WORLD_MODEL_CALL for span in rollout.spans
        ),
    }
    counts[ProviderSpendComponent.RETRIEVAL_EMBEDDING] = counts[ProviderSpendComponent.WORLD_MODEL]
    economics = {
        ProviderSpendComponent.CANDIDATE: rollout.candidate_economics,
        ProviderSpendComponent.WORLD_MODEL: rollout.world_model_economics,
        ProviderSpendComponent.RETRIEVAL_EMBEDDING: rollout.retrieval_economics,
    }
    if (
        rollout.candidate is None
        or rollout.world_model is None
        or rollout.simulation_binding is None
    ):
        raise ValueError("hosted world-model rollout omits exact model billing identities")
    billing_sources = {
        ProviderSpendComponent.CANDIDATE: rollout.candidate.billing_source,
        ProviderSpendComponent.WORLD_MODEL: rollout.world_model.billing_source,
        ProviderSpendComponent.RETRIEVAL_EMBEDDING: (
            rollout.simulation_binding.query_embedding.model.billing_source
        ),
    }
    result = []
    for component, value in economics.items():
        count = counts.get(
            component,
            1 if value is not None and value.cost_usd is not None else 0,
        )
        if count <= 0:
            continue
        if value is None:
            raise ValueError(f"completed rollout omits {component.value} economics")
        result.append(
            spend_entry_from_economics(
                operation_id=stable_id(
                    "provider-spend-operation",
                    {"rollout_id": rollout.rollout_id, "component": component.value},
                ),
                component=component,
                billing_source=billing_sources[component],
                economics=value,
                operation_count=count,
                evidence=evidence,
            )
        )
    return tuple(result)
