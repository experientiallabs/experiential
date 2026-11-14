"""Operator deadline extension for frozen runtime and job deadlines (dispatch only)."""

from __future__ import annotations

import os
from datetime import datetime


def dispatch_deadline(frozen: datetime) -> datetime:
    """The frozen deadline, or a later operator extension (EXP_RUNTIME_DEADLINE, ISO time).

    Frozen runtimes, jobs and family simulators pin a deadline into saved definitions; once it
    passes, every call is refused before dispatch. The extension only lets new calls dispatch;
    it never enters a saved definition or any identity.
    """
    extension = os.environ.get("EXP_RUNTIME_DEADLINE")
    if not extension:
        return frozen
    return max(frozen, datetime.fromisoformat(extension.replace("Z", "+00:00")))
