"""Durable bundle publication and typed events for hosted router stages."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from exp.common.core.artifacts import ArtifactInput, FailureCode, stable_id
from exp.common.project import (
    ExportedProjectBundle,
    ProjectStage,
    ProjectStageEvent,
    ProjectStageEventKind,
    ProjectStageFailure,
    ProjectStore,
    export_project_bundle,
)
from exp.optimize.router.attempt_authority import (
    HostedAttemptAuthority,
    HostedAttemptAuthorityStore,
    HostedStageCommit,
)
from exp.optimize.router.hosted_verification import verify_hosted_project
from exp.optimize.router.spend import ProviderSpendLedger

HostedEventSink = Callable[[ProjectStageEvent], None]


@dataclass(frozen=True)
class HostedStageBundle:
    """One newly completed durable stage and its verified portable Project bundle."""

    stage: ProjectStage
    bundle: ExportedProjectBundle


def export_stage_bundle(
    project: ProjectStore,
    bundles: list[HostedStageBundle],
    *,
    bundle_directory: Path,
    attempt_id: str,
    code_revision: str,
    stage: ProjectStage,
) -> HostedStageBundle:
    """Export one locally verified stage bundle without claiming external completion."""
    verify_hosted_project(project)
    destination = bundle_directory / f"{attempt_id}-{stage.value}.exp.zip"
    completed = HostedStageBundle(
        stage=stage,
        bundle=export_project_bundle(
            project,
            destination,
            producer_revision=code_revision,
        ),
    )
    bundles.append(completed)
    return completed


def commit_completed_stage(
    attempt_store: HostedAttemptAuthorityStore,
    authority: HostedAttemptAuthority,
    project: ProjectStore,
    stage_bundle: HostedStageBundle,
    events: list[ProjectStageEvent],
    event_sink: HostedEventSink | None,
    *,
    occurred_at: datetime,
    outputs: Sequence[ArtifactInput],
    spend_ledger: ProviderSpendLedger,
    spend_ledger_input: ArtifactInput,
) -> None:
    """Commit external bundle and spend evidence before emitting completion."""
    bundle = stage_bundle.bundle
    attempt_store.commit_stage(
        HostedStageCommit(
            project_id=project.paths.project_id,
            attempt_id=authority.attempt_id,
            authority_sha256=authority.authority_sha256,
            stage=stage_bundle.stage,
            bundle_sha256=bundle.sha256,
            bundle_size_bytes=bundle.size_bytes,
            spend_ledger=spend_ledger_input,
            spend_total_usd=spend_ledger.total_usd,
        ),
        bundle,
        spend_ledger,
    )
    emit_hosted_event(
        events,
        event_sink,
        project,
        authority.attempt_id,
        occurred_at,
        stage_bundle.stage,
        "completed",
        outputs=tuple(outputs),
    )


def emit_hosted_event(
    events: list[ProjectStageEvent],
    sink: HostedEventSink | None,
    project: ProjectStore,
    attempt_id: str,
    occurred_at: datetime,
    stage: ProjectStage,
    kind: str,
    *,
    outputs: tuple[ArtifactInput, ...] = (),
) -> None:
    """Create one ordered customer-safe typed stage event and forward it to the sink."""
    sequence = len(events)
    event_kind = ProjectStageEventKind(kind)
    ordered = tuple(sorted(outputs, key=lambda item: item.artifact_id))
    event = ProjectStageEvent(
        event_id=stable_id(
            "project-stage-event",
            {
                "project_id": project.paths.project_id,
                "attempt_id": attempt_id,
                "sequence": sequence,
                "stage": stage.value,
                "kind": event_kind.value,
            },
        ),
        project_id=project.paths.project_id,
        attempt_id=attempt_id,
        sequence=sequence,
        occurred_at=occurred_at,
        stage=stage,
        kind=event_kind,
        outputs=ordered,
        failure=(
            ProjectStageFailure(
                code=FailureCode.PROVIDER,
                retryable=False,
                detail_code="ambiguous-provider-operation",
            )
            if event_kind == ProjectStageEventKind.FAILED
            else None
        ),
    )
    events.append(event)
    if sink is not None:
        sink(event)
