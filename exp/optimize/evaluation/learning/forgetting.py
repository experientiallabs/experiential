"""Operator forgetting of a stale memory summary, after an eligibility check.

Forgetting invalidates one compressed block summary and its complete ancestors through the
memory owner (:func:`exp.simulation.world_model.experience_memory.forget_summary`). It never
erases a memory line, the source evidence or retrieval, and it never changes a published
checkpoint: the record is pinned to the current head and takes effect in the next version,
whose maintenance rebuilds the blocks from their eligible lines while retired hypotheses stay
excluded. Readers pinned to earlier versions, including every audit receipt, keep their exact
view.

A forget is refused while a registered trajectory-count study still has to learn checkpoints
of the memory: those checkpoints must come from learning alone, so an operator edit would
change what the study measures.
"""

from datetime import datetime

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.common.project import ProjectStore
from exp.optimize.evaluation.audit.receipts import audit_receipts
from exp.optimize.evaluation.learning.sample_count import (
    REGISTRATION_TYPE,
    SampleCountRegistration,
)
from exp.optimize.evaluation.learning.traces import learning_plan
from exp.simulation.world_model.experience_memory import forget_summary, memory_head
from exp.simulation.world_model.memory_tree import Block, containing_blocks

SCOPE = (
    "only compressed summaries are forgotten; memory lines, retired-hypothesis exclusion, source "
    "evidence and retrieval are unchanged, and published checkpoints keep their exact view"
)


class SummaryForget(ContractModel):
    """An operator decision to forget one stale block summary of a memory lineage.

    Attributes:
        lineage: Memory lineage; omitted, the one recorded learning plan's memory that has
            published (refused when both the all-traces and fit-only memories have).
        block: The block as memory views print it, ``#first-last`` (for example ``#0-3``).
        reason: Why the summary is stale; kept with the record.
    """

    lineage: str | None = Field(default=None, min_length=1)
    block: str = Field(pattern=r"^#\d+-\d+$")
    reason: str = Field(min_length=1, max_length=2000)

    @property
    def bounds(self) -> Block:
        """The block as a half-open ``(lo, hi)`` line range."""
        first, _, last = self.block.removeprefix("#").partition("-")
        return int(first), int(last) + 1


class ForgetDecision(ContractModel):
    """The eligibility outcome and the saved forget record.

    Attributes:
        lineage: Memory lineage the forget applies to.
        block: Forgotten block, ``#first-last``.
        reason: Operator reason kept with the record.
        head_version: Published version the record is pinned to; it stays unchanged.
        takes_effect_at: First version whose readers no longer see the forgotten summaries.
        forgets: The block and every complete ancestor whose generation is raised.
        pinned_audits: Audit receipts reading checkpoints of this lineage; each reads a
            published version, which keeps its exact view.
        record: Forget record identity; None on a dry run of a new forget.
        scope: What forgetting does and does not change.
    """

    lineage: str
    block: str
    reason: str
    head_version: int = Field(ge=1)
    takes_effect_at: int = Field(ge=2)
    forgets: tuple[str, ...] = Field(min_length=1)
    pinned_audits: tuple[str, ...] = ()
    record: str | None = None
    scope: str = SCOPE


def forget_memory_summary(
    project: ProjectStore,
    request: SummaryForget,
    *,
    created_at: datetime,
    code_revision: str,
    dry_run: bool = False,
) -> ForgetDecision:
    """Check that a summary can be forgotten, then record the forget against the head.

    Args:
        project: Project owning the memory.
        request: Lineage, block and reason.
        created_at: Aware record time.
        code_revision: Recording implementation identity.
        dry_run: Check eligibility without recording.

    Returns:
        The decision with the record identity, or without it on a dry run.

    Raises:
        ValueError: The memory has no published version, a registered trajectory-count study
            still learns checkpoints of it, the block is not a complete aligned block, it has
            no built summary to forget, or it was forgotten against this head for another
            reason.
    """
    store = project.artifacts
    lineage = request.lineage or learning_plan(project).lineage
    head = memory_head(store, lineage)
    if head is None:
        raise ValueError(f"memory {lineage!r} has no published version to forget from")
    _require_no_open_study(project, lineage, head.version)
    lo, hi = request.bounds
    identity = forget_summary(
        store,
        lineage,
        (lo, hi),
        reason=request.reason,
        created_at=created_at,
        code_revision=code_revision,
        dry_run=dry_run,
    )
    blocks = [
        block
        for block in containing_blocks(head.log_length or 0, [lo])
        if block[0] <= lo and hi <= block[1]
    ]
    return ForgetDecision(
        lineage=lineage,
        block=request.block,
        reason=request.reason,
        head_version=head.version,
        takes_effect_at=head.version + 1,
        forgets=tuple(f"#{first}-{end - 1}" for first, end in blocks),
        pinned_audits=tuple(
            receipt.receipt_id
            for receipt in audit_receipts(project)
            if receipt.checkpoint is not None and receipt.checkpoint.lineage == lineage
        ),
        record=identity if store.exists(identity) else None,
    )


def _require_no_open_study(project: ProjectStore, lineage: str, version: int) -> None:
    """Refuse while a registered trajectory-count grid reaches past the published versions.

    Raises:
        ValueError: A registration on this memory still has a grid count to learn.
    """
    for identity in project.artifacts.list_ids(artifact_type=REGISTRATION_TYPE):
        registration = SampleCountRegistration.model_validate_json(
            project.artifacts.read_bytes(identity, "registration.json")
        )
        if registration.memory_lineage == lineage and registration.grid[-1] > version:
            raise ValueError(
                f"sample-count registration {identity} still learns checkpoints of memory "
                f"{lineage!r} up to {registration.grid[-1]} trajectories (published: {version}); "
                "its checkpoints must come from learning alone. Forget after learning passes "
                "the registered grid"
            )
