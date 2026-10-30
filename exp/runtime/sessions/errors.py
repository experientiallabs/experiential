"""Typed refusals of the durable session store and its incremental state owner."""

from __future__ import annotations


class SessionStoreError(RuntimeError):
    """Base class for refused session-store operations."""


class SessionNotFoundError(SessionStoreError):
    """The requested session does not exist in this database."""


class SessionExistsError(SessionStoreError):
    """A session with this identifier already exists."""


class VersionConflictError(SessionStoreError):
    """The expected session version does not match the stored version."""


class StepConflictError(SessionStoreError):
    """A step ID was reused with a different request digest."""


class StepInFlightError(SessionStoreError):
    """Another step on the same session is open, so this step must wait."""


class StepAmbiguousError(SessionStoreError):
    """A step reached the provider without a persisted outcome and is blocked for review."""


class StepRejectedError(SessionStoreError):
    """The step was closed without a state change. Its stored reason is in the message."""


class SessionBindingMismatchError(SessionStoreError):
    """The request binding differs from the binding pinned at session creation."""


class ProviderBudgetExhaustedError(SessionStoreError):
    """The session has used its provider-call budget, so no further dispatch may start."""


class StepDeadlineExceededError(SessionStoreError):
    """A returned proposal was retained but cannot commit after its session deadline."""


class SessionDrainingError(SessionStoreError):
    """The session is draining and refuses new steps."""


class ClaimLostError(SessionStoreError):
    """The claim is not the open claim for its step."""


class StateReferenceError(SessionStoreError):
    """A state reference names another store, an unknown revision or a different root."""


class StateUnavailableError(SessionStoreError):
    """A revision's state was reclaimed or its stored nodes are missing or corrupt."""


class RetentionConflictError(SessionStoreError):
    """A retention owner already holds a different range of the same session."""
