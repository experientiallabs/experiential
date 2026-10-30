"""Operator retirement of world checkpoints and reclamation of the session state they held.

Capturing a world checkpoint pins its source session's revisions under the checkpoint's artifact
identity (:mod:`exp.simulation.world_model.checkpoints`). That pin is released by one rule only:
the operator retires the checkpoint with ``retire_checkpoint``, and the retirement is refused
while any saved artifact (a study, campaign, task, package, fork or other result) still names
the checkpoint in its inputs or its data. Nothing releases a pin automatically.

``collect_state_garbage`` then reclaims, in every session database of the project, the nodes no
session head and no remaining pin reaches. Its dry run writes nothing: it reports what would be
reclaimed and, per pin, which owner holds which revisions, how many nodes only that pin keeps,
and whether the owner can be retired. The immutable event logs are never reclaimed, so every
revision stays replayable.
"""

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, ContractModel, stable_id
from exp.common.project import ArtifactStore, ProjectStore, artifact_input
from exp.runtime.sessions.retention import (
    CollectionPreview,
    CollectionReport,
    RetentionPin,
)
from exp.runtime.sessions.schema import is_session_database
from exp.runtime.sessions.store import SessionStore
from exp.simulation.world_model.checkpoints import load_world_checkpoint

RETIREMENT_TYPE = "world-checkpoint-retirement"
CHECKPOINT_TYPE = "world-checkpoint"
_LISTED_REFERRERS = 5

type OwnerStatus = Literal["referenced", "retirable", "retired", "not_a_checkpoint"]


class CheckpointRetirement(ContractModel):
    """The ``retire-checkpoint`` payload: one checkpoint the operator no longer needs.

    Attributes:
        checkpoint: Exact saved world checkpoint whose history pins should be released.
        reason: Why the checkpoint is retired, kept in the retirement record.
    """

    checkpoint: ArtifactInput
    reason: str = Field(min_length=1, max_length=500)


class StateCollection(ContractModel):
    """The ``collect-garbage`` payload.

    Attributes:
        batch_size: Node rows examined per write transaction while sweeping each database.
        vacuum: Also rewrite each database that reclaimed nodes, so the freed pages return to
            the file system; without it SQLite reuses them for later writes.
    """

    batch_size: int = Field(default=500, ge=1, le=100_000)
    vacuum: bool = False


class ReleasedPin(ContractModel):
    """One pin a retirement released, or would release in a dry run.

    Attributes:
        store: Session database path relative to the project directory.
        pin: The owner and revision range.
    """

    store: str
    pin: RetentionPin


class RetirementDecision(ContractModel):
    """The outcome of one retirement.

    Attributes:
        checkpoint: Retired checkpoint identity.
        record: Saved retirement record; None in a dry run.
        released: Pins released now, or that a recorded retirement would release.
    """

    checkpoint: str
    record: ArtifactInput | None
    released: tuple[ReleasedPin, ...]


class HeldPin(ContractModel):
    """One pin, what it alone keeps, and whether its owner can be retired.

    Attributes:
        store: Session database path relative to the project directory.
        pin: The owner and revision range.
        exclusive_nodes: Nodes that only this pin keeps; releasing it makes them collectable.
        exclusive_bytes: Stored bytes of those nodes.
        owner: ``referenced`` (saved artifacts name the checkpoint), ``retirable`` (nothing
            does; ``retire-checkpoint`` would release it), ``retired`` (a retirement is recorded;
            re-run it to release this pin) or ``not_a_checkpoint`` (never released here).
        referrers: Up to five saved artifacts that reference the owner.
        referrer_count: Number of saved artifacts that reference the owner.
    """

    store: str
    pin: RetentionPin
    exclusive_nodes: int = Field(ge=0)
    exclusive_bytes: int = Field(ge=0)
    owner: OwnerStatus
    referrers: tuple[str, ...]
    referrer_count: int = Field(ge=0)


class StoreCollection(ContractModel):
    """One session database before collection and, outside a dry run, what was reclaimed.

    Attributes:
        store: Session database path relative to the project directory.
        preview: Reclaimable and retained nodes before collection.
        report: Collection result; None in a dry run.
        vacuumed_bytes: Bytes the vacuum returned to the file system; None when not vacuumed.
    """

    store: str
    preview: CollectionPreview
    report: CollectionReport | None
    vacuumed_bytes: int | None = None


class CollectionInventory(ContractModel):
    """Every session database of a project, every pin, and what was or would be reclaimed.

    Attributes:
        dry_run: Whether nothing was written.
        stores: One entry per session database.
        pins: Every pin with its owner status.
    """

    dry_run: bool
    stores: tuple[StoreCollection, ...]
    pins: tuple[HeldPin, ...]

    @property
    def reclaimable_nodes(self) -> int:
        """Nodes reclaimable before this run, across all databases."""
        return sum(row.preview.reclaimable_nodes for row in self.stores)

    @property
    def reclaimable_bytes(self) -> int:
        """Stored bytes reclaimable before this run, across all databases."""
        return sum(row.preview.reclaimable_bytes for row in self.stores)


def session_databases(project: ProjectStore) -> tuple[Path, ...]:
    """Return every session database under the project directory, without opening any.

    Args:
        project: Project whose directory holds the session stores.

    Returns:
        Paths in sorted order; dispatch ledgers and other SQLite files are excluded.
    """
    root = project.artifacts.project_directory
    if not root.exists():
        return ()
    return tuple(path for path in sorted(root.rglob("*.sqlite")) if is_session_database(path))


def retire_checkpoint(
    project: ProjectStore,
    spec: CheckpointRetirement,
    *,
    created_at: datetime,
    code_revision: str,
    dry_run: bool,
) -> RetirementDecision:
    """Retire one unreferenced world checkpoint and release its session-history pins.

    The retirement record is saved before any pin is released, so an interrupted retirement
    resumes by running it again.

    Args:
        project: Project owning the checkpoint and its session databases.
        spec: Checkpoint and reason.
        created_at: Aware record time.
        code_revision: Producing source identity.
        dry_run: Check and report the pins it would release, writing nothing.

    Returns:
        The record and the released pins.

    Raises:
        ValueError: The reference is not a saved world checkpoint, or saved artifacts still
            reference it.
    """
    store = project.artifacts
    checkpoint = load_world_checkpoint(store, spec.checkpoint).artifact_id
    referrers = checkpoint_referrers(store, {checkpoint})[checkpoint]
    if referrers:
        listed = ", ".join(referrers[:_LISTED_REFERRERS])
        raise ValueError(
            f"checkpoint {checkpoint} is still referenced by {len(referrers)} saved "
            f"artifact(s) ({listed}); its session history stays pinned while they exist. "
            "Retire only checkpoints no saved study, campaign or package uses"
        )
    held = tuple(
        ReleasedPin(store=_relative(project, path), pin=pin)
        for path in session_databases(project)
        for pin in SessionStore(path).pins()
        if pin.owner_id == checkpoint
    )
    if dry_run:
        return RetirementDecision(checkpoint=checkpoint, record=None, released=held)
    identity = stable_id(RETIREMENT_TYPE, {"checkpoint": spec.checkpoint.model_dump(mode="json")})
    if not store.exists(identity):
        store.write_json(
            artifact_id=identity,
            artifact_type=RETIREMENT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=created_at,
                code_revision=code_revision,
                inputs=(spec.checkpoint,),
            ),
            files={"retirement.json": spec},
        )
    record = artifact_input(store.read(identity).manifest)
    released = tuple(
        row
        for row in held
        if SessionStore(project.artifacts.project_directory / row.store).release_pin(
            row.pin.owner_id, row.pin.session_id
        )
    )
    return RetirementDecision(checkpoint=checkpoint, record=record, released=released)


def collect_state_garbage(
    project: ProjectStore, spec: StateCollection, *, dry_run: bool
) -> CollectionInventory:
    """Report every pin and reclaim unowned session state in every session database.

    Args:
        project: Project whose session databases are collected.
        spec: Sweep batch size and whether to vacuum afterwards.
        dry_run: Report what would be reclaimed and which pins hold what; write nothing.

    Returns:
        Per-database previews and results, and every pin with its owner status.
    """
    previews = []
    for path in session_databases(project):
        session_store = SessionStore(path)
        previews.append((path, session_store, session_store.preview_collection()))
    owners = {holding.pin.owner_id for _, _, preview in previews for holding in preview.pins}
    statuses = _owner_statuses(project.artifacts, owners)
    referrers = checkpoint_referrers(project.artifacts, owners)
    pins = tuple(
        HeldPin(
            store=_relative(project, path),
            pin=holding.pin,
            exclusive_nodes=holding.exclusive_nodes,
            exclusive_bytes=holding.exclusive_bytes,
            owner=_status(statuses[holding.pin.owner_id], referrers[holding.pin.owner_id]),
            referrers=tuple(referrers[holding.pin.owner_id][:_LISTED_REFERRERS]),
            referrer_count=len(referrers[holding.pin.owner_id]),
        )
        for path, _, preview in previews
        for holding in preview.pins
    )
    stores = []
    for path, session_store, preview in previews:
        report = None if dry_run else session_store.collect_garbage(batch_size=spec.batch_size)
        vacuumed = (
            session_store.vacuum()
            if report is not None and spec.vacuum and report.reclaimed_nodes
            else None
        )
        stores.append(
            StoreCollection(
                store=_relative(project, path),
                preview=preview,
                report=report,
                vacuumed_bytes=vacuumed,
            )
        )
    return CollectionInventory(dry_run=dry_run, stores=tuple(stores), pins=pins)


def checkpoint_referrers(artifacts: ArtifactStore, targets: Iterable[str]) -> dict[str, list[str]]:
    """Return, for each target identity, the saved artifacts that reference it.

    An artifact references a target when its manifest lists the target as an input or any of
    its data files contains the target's identity. The target itself and retirement records
    do not count. The data check over-approximates on purpose: a reference held only inside
    a payload still keeps the checkpoint pinned.

    Args:
        artifacts: Project artifact store.
        targets: Artifact identities to look for.

    Returns:
        Sorted referrer identities per target, empty when nothing references it.
    """
    wanted = set(targets)
    found: dict[str, list[str]] = {target: [] for target in wanted}
    if not wanted:
        return found
    needles = {target: target.encode() for target in wanted}
    for identity in artifacts.list_ids():
        stored = artifacts.read(identity)
        if stored.manifest.artifact_type == RETIREMENT_TYPE:
            continue
        inputs = {row.artifact_id for row in stored.manifest.inputs}
        for target, needle in needles.items():
            if identity != target and (
                target in inputs or any(needle in payload for payload in stored.payloads.values())
            ):
                found[target].append(identity)
    return {target: sorted(rows) for target, rows in found.items()}


def _owner_statuses(artifacts: ArtifactStore, owners: Iterable[str]) -> dict[str, OwnerStatus]:
    """Classify each pin owner as a retired, retirable or unknown checkpoint, before references."""
    retired = set()
    for identity in artifacts.list_ids(artifact_type=RETIREMENT_TYPE):
        retired.update(row.artifact_id for row in artifacts.read(identity).manifest.inputs)
    checkpoints = set(artifacts.list_ids(artifact_type=CHECKPOINT_TYPE))
    return {
        owner: "retired"
        if owner in retired
        else "retirable"
        if owner in checkpoints
        else "not_a_checkpoint"
        for owner in owners
    }


def _status(status: OwnerStatus, referrers: list[str]) -> OwnerStatus:
    """A checkpoint that saved artifacts reference stays pinned, whatever its record says."""
    return "referenced" if referrers and status != "not_a_checkpoint" else status


def _relative(project: ProjectStore, path: Path) -> str:
    """Name a session database relative to the project directory."""
    return path.relative_to(project.artifacts.project_directory).as_posix()
