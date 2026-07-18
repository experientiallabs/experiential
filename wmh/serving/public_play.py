"""Rate-limiting and a hard spend ceiling for anonymous ("public") play.

The public catalog lets logged-out visitors step a world model live. That spends real provider
tokens with no account behind it, so the public play surface runs through this limiter: a global
ceiling on total steps (which bounds total spend deterministically, since each step costs at most
one model call), plus a per-session step cap and a cap on how many sessions may be opened. When a
ceiling is hit the caller gets a typed refusal that the serving layer turns into HTTP 429.

The limiter is process-local and in-memory: it protects a single public-play server instance. It
never tracks user identity (there is none); the ceilings are the whole defense.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


class PublicPlayLimitError(Exception):
    """A public-play ceiling was reached; the serving layer maps this to HTTP 429."""


@dataclass(frozen=True)
class PublicPlayLimits:
    """Ceilings for the public play surface. Total spend is bounded by `max_total_steps`."""

    max_total_steps: int = 500
    max_steps_per_session: int = 20
    max_sessions: int = 200


class PublicPlayLimiter:
    """Thread-safe ceilings shared across every public-play session on one server."""

    def __init__(self, limits: PublicPlayLimits | None = None) -> None:
        self._limits = limits or PublicPlayLimits()
        self._total_steps = 0
        self._sessions_opened = 0
        self._session_steps: dict[str, int] = {}
        self._session_model: dict[str, str] = {}
        self._lock = threading.Lock()

    def open_session(self, model_name: str, session_id: str) -> None:
        """Reserve a new public session, or refuse when the session ceiling is reached."""
        with self._lock:
            if self._sessions_opened >= self._limits.max_sessions:
                raise PublicPlayLimitError(
                    "The public demo is at capacity right now. Log in to keep playing."
                )
            self._sessions_opened += 1
            self._session_steps[session_id] = 0
            self._session_model[session_id] = model_name

    def model_for_session(self, session_id: str) -> str | None:
        """The model a public session belongs to, or None if the session is unknown here."""
        with self._lock:
            return self._session_model.get(session_id)

    def charge_step(self, session_id: str) -> None:
        """Count one step against the global and per-session ceilings, or refuse at the limit.

        Charged BEFORE the model call so a refused step never spends anything.
        """
        with self._lock:
            if session_id not in self._session_steps:
                raise PublicPlayLimitError("Unknown or expired public session.")
            if self._total_steps >= self._limits.max_total_steps:
                raise PublicPlayLimitError(
                    "The public demo has reached its usage limit. Log in to keep playing."
                )
            if self._session_steps[session_id] >= self._limits.max_steps_per_session:
                raise PublicPlayLimitError(
                    "This session has reached the public step limit. Log in to keep playing."
                )
            self._total_steps += 1
            self._session_steps[session_id] += 1

    def snapshot(self) -> dict[str, int]:
        """Current counters, for observability and tests."""
        with self._lock:
            return {
                "total_steps": self._total_steps,
                "sessions_opened": self._sessions_opened,
                "max_total_steps": self._limits.max_total_steps,
            }
