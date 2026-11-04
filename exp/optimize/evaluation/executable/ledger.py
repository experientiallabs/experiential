"""The harness-owned, append-only ledger of one executable Harbor trial.

Every trial records, in one sequence, the initial evaluator projection, each tool request and its
observed reply or failure, each customer-facing message the actor sends through the bridge, and
the final submission. The ledger is what the episode judge reads.
"""

from datetime import datetime
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.project import ProjectStore

LedgerKind = Literal["start", "tool_call", "tool_reply", "tool_failure", "message", "submission"]
"""What one trial ledger entry records."""


class TrialLedgerEntry(ContractModel):
    """One harness-recorded entry of a trial's append-only ledger.

    Attributes:
        sequence: Logical position in the ledger, starting at 0 with the ``start`` entry.
        kind: ``start`` holds the initial evaluator projection; ``tool_call`` the exact request;
            ``tool_reply`` the saved step (observation, receipt and resulting projection);
            ``tool_failure`` a dispatch or observation failure; ``message`` customer-facing
            text the actor sent; ``submission`` the final answer and final projection.
        content: The recorded payload.
        record: Backing trial record artifact, or None when the ledger entry is the record.
        recorded_at: Wall-clock time the harness appended the entry.
        reconciled: Whether the entry was appended on restart from an already saved record,
            because the process stopped between that record and its ledger entry.
    """

    sequence: int = Field(ge=0)
    kind: LedgerKind
    content: JsonObject
    record: str | None = None
    recorded_at: datetime
    reconciled: bool = False


def trial_ledger(project: ProjectStore, identity: str) -> tuple[TrialLedgerEntry, ...]:
    """Read a trial's contiguous ledger without opening its runtime.

    Args:
        project: Project holding the trial records.
        identity: Durable trial identity (``ExecutableHarborTrial.identity``).

    Returns:
        Every saved entry in sequence order.

    Raises:
        ValueError: An entry's recorded sequence differs from its position.
    """
    entries: list[TrialLedgerEntry] = []
    while project.artifacts.exists(f"{identity}-ledger-{len(entries)}"):
        entry = TrialLedgerEntry.model_validate_json(
            project.artifacts.read_bytes(f"{identity}-ledger-{len(entries)}", "record.json")
        )
        if entry.sequence != len(entries):
            raise ValueError("trial ledger entry differs from its position")
        entries.append(entry)
    return tuple(entries)
