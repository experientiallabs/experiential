"""Shared error contract for routed interaction journals."""

from __future__ import annotations


class RuntimeJournalError(ValueError):
    """The runtime journal is corrupt or an attempted transition is invalid."""
