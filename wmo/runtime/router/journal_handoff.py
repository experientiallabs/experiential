"""Atomic local work handoff bound to one exact runtime-journal prefix."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from wmo.common.core.artifacts import Sha256, canonical_json_bytes
from wmo.common.core.locks import file_write_lock
from wmo.runtime.router.journal import (
    RuntimeInteractionJournal,
    RuntimeJournalError,
    RuntimeJournalEvent,
)


def commit_runtime_journal_prefix[ResultT](
    journal: RuntimeInteractionJournal,
    *,
    last_ordinal: int,
    prefix_sha256: Sha256,
    commit: Callable[[], ResultT],
) -> ResultT:
    """Commit one local handoff only while an exact journal prefix remains current.

    The callback runs under the journal's append lock and must only perform a bounded local
    durable write. Provider access, credential reads, and other unbounded work do not belong
    inside this boundary.

    Args:
        journal: Validated project journal whose append lock defines the handoff boundary.
        last_ordinal: Exact final event ordinal approved by the caller.
        prefix_sha256: Canonical newline-framed digest of every approved event.
        commit: Bounded local callback that durably accepts work bound to the prefix.

    Returns:
        The callback result after the exact prefix is verified and accepted.

    Raises:
        RuntimeJournalError: The journal changed before the local handoff committed.
    """
    with file_write_lock(journal.path, what="the routed-interaction journal"):
        events = journal._read_unlocked()  # noqa: SLF001
        if (
            not events
            or events[-1].ordinal != last_ordinal
            or _events_sha256(events) != prefix_sha256
        ):
            raise RuntimeJournalError(
                "runtime journal changed before the approved prefix was accepted"
            )
        return commit()


def _events_sha256(events: tuple[RuntimeJournalEvent, ...]) -> str:
    """Return the canonical newline-framed digest of validated journal events.

    Args:
        events: Ordered journal events read under the append lock.

    Returns:
        SHA-256 digest matching the runtime snapshot prefix contract.
    """
    payload = b"\n".join(canonical_json_bytes(event) for event in events)
    if payload:
        payload += b"\n"
    return hashlib.sha256(payload, usedforsecurity=False).hexdigest()
