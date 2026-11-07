"""Frozen, human-approved refined prompts: the only part of the Astra refinement the runtime reads.

An Astra refinement (:mod:`.astra_prompt_loop`, offline only) ends with a candidate prompt and its
held-out parity. A human then approves or declines it (:func:`approve` / :func:`decline`), a move
of the refinement's state machine. Approval freezes the prompt as an immutable, versioned artifact
(``astra-frozen-prompt``); a later approval of the same target is the next version and supersedes
it.

The runtime reads approved prompts only:

- :func:`refined_role_prompt` gives the shared role runner the approved text of a role's
  instruction or stance (matched by the role runner's label and the exact code text it replaced,
  so a changed code default stops matching instead of being silently overridden). The draw's
  identity keys on the text, so an approved prompt is a new draw.
- :func:`refined_identity` is folded into every decision-variance stage identity once any prompt
  is approved, so approving or re-approving a prompt re-keys the stages as usual; with nothing
  approved it is empty and every identity is unchanged.

This module imports no model client and holds no Astra settings: a frozen prompt is text plus
digests and provenance, and the role runs it on its own open-weight model.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.project import ProjectStore, artifact_input
from exp.common.state_machine import TransitionJournal
from exp.optimize.evaluation.variance.astra_prompt_contracts import RECORD_TYPE, RefinementRecord
from exp.optimize.evaluation.variance.astra_prompt_states import (
    REFINEMENT_MACHINE,
    RefinementState,
)
from exp.optimize.evaluation.variance.prompt_targets import Part, target

FROZEN_TYPE = "astra-frozen-prompt"
"""Artifact type of one approved, frozen prompt version."""


class FrozenPrompt(ContractModel):
    """One approved prompt version.

    Attributes:
        target_id: The target.
        role: The pipeline role.
        runner_role: The shared role runner's label, when the role runs there.
        part: What part of the role's prompt it is.
        variant: The stance or variant name, for an expanded attribute.
        version: Version within the target, from 1.
        base_text: The exact code text it replaces.
        base_sha256: Digest of ``base_text``.
        text: The approved text.
        sha256: Digest of ``text``.
        record: The refinement record it came from.
        heldout_parity: Its held-out parity at approval.
        agreement: Astra/reviewer held-out agreement at approval.
        approved_by: Who approved it.
        note: The reviewer's note.
        approved_at: When.
    """

    target_id: str
    role: str
    runner_role: str | None
    part: Part
    variant: str | None
    version: int = Field(ge=1)
    base_text: str
    base_sha256: Sha256
    text: str
    sha256: Sha256
    record: str
    heldout_parity: float | None
    agreement: float | None
    approved_by: str = Field(min_length=1)
    note: str = Field(default="", max_length=4000)
    approved_at: datetime


def journal_of(project: ProjectStore) -> TransitionJournal:
    """The refinement journal (loop moves and human reviews)."""
    return TransitionJournal(project.paths.runtime_directory / "astra-prompts" / "journal.jsonl")


def frozen_prompts(project: ProjectStore) -> tuple[FrozenPrompt, ...]:
    """Every approved version, oldest first."""
    store = project.artifacts
    rows = [
        FrozenPrompt.model_validate_json(store.read_bytes(artifact_id, "frozen.json"))
        for artifact_id in store.list_ids(artifact_type=FROZEN_TYPE)
    ]
    return tuple(sorted(rows, key=lambda row: (row.target_id, row.version)))


def approved_prompts(project: ProjectStore) -> dict[str, FrozenPrompt]:
    """The current (latest) approved version of every target."""
    current: dict[str, FrozenPrompt] = {}
    for row in frozen_prompts(project):
        current[row.target_id] = row
    return current


def refined_identity(project: ProjectStore) -> JsonObject:
    """Identity content of the approved prompts; empty when nothing is approved."""
    rows = approved_prompts(project).values()
    if not rows:
        return {}
    return {
        "refined_prompts": sha256_json(
            sorted([row.target_id, row.version, row.sha256] for row in rows)
        )
    }


def refined_role_prompt(
    project: ProjectStore, role: str, instruction: str, stance: tuple[str, str]
) -> tuple[str, tuple[str, str]]:
    """Return the instruction and stance a role runs, with every approved text applied.

    A stance target applies when its variant names the stance and its code text is the stance
    text; any other part applies when its code text occurs exactly once in the instruction.
    """
    name, text = stance
    for row in approved_prompts(project).values():
        if row.runner_role != role or sha256_json(row.base_text) != row.base_sha256:
            continue
        if row.part == "stance":
            if row.variant == name and text == row.base_text:
                text = row.text
        elif instruction.count(row.base_text) == 1:
            instruction = instruction.replace(row.base_text, row.text, 1)
    return instruction, (name, text)


def _record(project: ProjectStore, record_id: str) -> RefinementRecord:
    """Read a saved refinement record."""
    if not project.artifacts.exists(record_id):
        raise ValueError(f"no saved refinement record {record_id}")
    stored = project.artifacts.read(record_id)
    if stored.manifest.artifact_type != RECORD_TYPE:
        raise ValueError(f"{record_id} is not a refinement record")
    return RefinementRecord.model_validate_json(stored.payloads["record.json"])


def _review_state(project: ProjectStore, record: RefinementRecord) -> RefinementState:
    """The record's current state: its terminal loop state, or a recorded review."""
    state = RefinementState(record.state)
    for row in journal_of(project).rows():
        if (
            row["subject"] == record.identity
            and not row["refused"]
            and row["to"]
            in (
                RefinementState.APPROVED.value,
                RefinementState.DECLINED.value,
            )
        ):
            state = RefinementState(str(row["to"]))
    return state


def _review(project: ProjectStore, record: RefinementRecord, event: RefinementState) -> None:
    """Apply one checked, journaled review move."""
    REFINEMENT_MACHINE.move(
        record.identity,
        _review_state(project, record),
        event.value,
        journal_of(project),
        target=event,
    )


def approve(
    project: ProjectStore, record_id: str, *, by: str, note: str = "", code_revision: str
) -> FrozenPrompt:
    """Approve a candidate and freeze it as the target's next version.

    Raises:
        IllegalTransition: The record is not a reviewable candidate (budget refused, or already
            reviewed); the refused move is journaled.
        ValueError: The code text changed since the refinement, so the candidate no longer
            replaces what the role runs.
    """
    record = _record(project, record_id)
    found = target(record.target_id)
    if found.sha256 != record.base_sha256:
        raise ValueError(f"{record.target_id}'s code text changed since its refinement; rerun it")
    _review(project, record, RefinementState.APPROVED)
    version = 1 + sum(1 for row in frozen_prompts(project) if row.target_id == record.target_id)
    final = record.final
    frozen = FrozenPrompt(
        target_id=record.target_id,
        role=found.role,
        runner_role=found.runner_role,
        part=found.part,
        variant=found.variant,
        version=version,
        base_text=found.text,
        base_sha256=found.sha256,
        text=record.prompt,
        sha256=record.prompt_sha256,
        record=record_id,
        heldout_parity=None if final is None else final.heldout_parity,
        agreement=None if final is None else final.agreement,
        approved_by=by,
        note=note,
        approved_at=datetime.now(UTC),
    )
    stored = project.artifacts.read(record_id)
    project.artifacts.write_json(
        artifact_id=stable_id(FROZEN_TYPE, {"target": record.target_id, "version": version}),
        artifact_type=FROZEN_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=(artifact_input(stored.manifest),),
        ),
        files={"frozen.json": frozen},
    )
    return frozen


def decline(project: ProjectStore, record_id: str) -> None:
    """Decline a candidate; nothing is frozen.

    Raises:
        IllegalTransition: The record is not a reviewable candidate; the refusal is journaled.
    """
    _review(project, _record(project, record_id), RefinementState.DECLINED)
