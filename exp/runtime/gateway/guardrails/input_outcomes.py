"""Request-owned, content-free input inspection evidence for durable accounting."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from threading import Lock
from typing import Literal
from weakref import ReferenceType, ref

from pydantic import BaseModel, ConfigDict

from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.guardrails.classification_window import (
    ClassificationWindow,
    classification_window,
    message_text,
)
from exp.runtime.gateway.guardrails.contracts import (
    GuardrailAction,
    GuardrailCheck,
    GuardrailOutcome,
    GuardrailPolicy,
)


class InputCheckOutcome(BaseModel):
    """One frozen check's execution result, never a grant of retrospective authority.

    Attributes:
        policy_id: Authored policy identity.
        policy_revision: Frozen detector and policy revision.
        check_id: Authored input check identity.
        stage: Input scope only; output coverage cannot be inferred.
        mode: Whether the original policy could affect delivery.
        action: The original check's authored action.
        protected: Whether the policy was owned by the operator.
        outcome: Actual bounded inspection result, or not_started before execution.
        recovery_reason: Explicit missing-coverage reason, otherwise None.
        latest_user_window_sha256: Digest of the 30,000-character latest-user head/tail
            window, only when all admitted subjects agree; never a digest of output.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_id: str
    policy_revision: str
    check_id: str
    stage: Literal["input"] = "input"
    mode: Literal["enforce", "observe"]
    action: GuardrailAction
    protected: bool
    outcome: Literal["allow", "block", "missed", "unsupported", "not_started"]
    recovery_reason: Literal["deadline_expired", "classifier_error", "result_unavailable"] | None
    latest_user_window_sha256: str | None = None


class InputOutcomes:
    """Collect execution evidence synchronously, without an asynchronous telemetry sink.

    Attributes:
        _results: Outcomes by policy/check, containing no subjects or classifier payloads.
        _started: Checks that actually entered execution.
        _windows: At most two bounded input-window hashes per check; disagreement
            makes the evidence unusable for a single-window retrospective review.
        _lock: Serializes classifier completion, native expiry and finalization.
        _frozen: Immutable terminal snapshot, or None while execution is open.
    """

    def __init__(self) -> None:
        """Begin without execution or durable-completion claims."""
        self._results: dict[tuple[str, str], GuardrailOutcome] = {}
        self._started: set[tuple[str, str]] = set()
        self._windows: dict[tuple[str, str], set[str]] = {}
        self._subjects: list[tuple[ReferenceType[GatewayRequest], str]] = []
        self._lock = Lock()
        self._frozen: tuple[InputCheckOutcome, ...] | None = None

    @contextmanager
    def collect(self) -> Iterator[None]:
        """Bind only this execution coroutine, including its awaited check decisions."""
        token = _CURRENT.set(self)
        try:
            yield
        finally:
            _CURRENT.reset(token)

    def start(self, policy: GuardrailPolicy, request: GatewayRequest | None = None) -> None:
        """Remember actual admission of a check without retaining its subject."""
        if request is not None:
            self.admit(policy, request)
        with self._lock:
            if self._frozen is None:
                self._started.update(
                    (policy.policy_id, check.check_id) for check in policy.input_checks
                )

    def admit(self, policy: GuardrailPolicy, request: GatewayRequest) -> None:
        """Remember an input window without claiming that its queued checks executed."""
        with self._lock:
            if self._frozen is None:
                # Weak identity caches only immutable text projections, without
                # retaining prompts or confusing recycled object IDs.
                digest = next(
                    (value for subject, value in self._subjects if subject() is request), None
                )
                if digest is None:
                    digest, immutable = _window_digest(request)
                    if immutable:
                        self._subjects = [*self._subjects[-1:], (ref(request), digest)]
                for check in policy.input_checks:
                    windows = self._windows.setdefault((policy.policy_id, check.check_id), set())
                    if len(windows) < 2:
                        windows.add(digest)

    def record(
        self, policy: GuardrailPolicy, check: GuardrailCheck, result: GuardrailOutcome
    ) -> None:
        """Keep unsupported subjects unrecoverable and misses sticky until terminalization."""
        key = (policy.policy_id, check.check_id)
        with self._lock:
            if self._frozen is not None:
                return
            previous = self._results.get(key)
            if (
                result is GuardrailOutcome.UNSUPPORTED
                or previous is None
                or previous is GuardrailOutcome.ALLOW
                or (
                    previous is GuardrailOutcome.FLAGGED
                    and check.action in (GuardrailAction.ALLOW, GuardrailAction.MODIFY)
                )
            ):
                self._results[key] = result

    def expire(self, policy: GuardrailPolicy) -> None:
        """Native expiry wins over an uncommitted classifier result."""
        with self._lock:
            if self._frozen is None:
                for check in policy.input_checks:
                    key = (policy.policy_id, check.check_id)
                    if (
                        key in self._started
                        and self._results.get(key) is not GuardrailOutcome.UNSUPPORTED
                    ):
                        self._results[key] = GuardrailOutcome.TIMEOUT

    def freeze(
        self, policies: tuple[GuardrailPolicy, ...], *, blocked: bool
    ) -> tuple[InputCheckOutcome, ...]:
        """Seal the evidence once; accounting retries receive exactly the same tuple."""
        with self._lock:
            if self._frozen is None:
                self._frozen = tuple(
                    self._outcome(policy, check, blocked=blocked)
                    for policy in policies
                    for check in policy.input_checks
                )
            return self._frozen

    def _outcome(
        self, policy: GuardrailPolicy, check: GuardrailCheck, *, blocked: bool
    ) -> InputCheckOutcome:
        """Project execution facts conservatively while the collection lock is held."""
        key = (policy.policy_id, check.check_id)
        result = self._results.get(key)
        windows = self._windows.get(key, set())
        outcome: Literal["allow", "block", "missed", "unsupported", "not_started"]
        reason: Literal["deadline_expired", "classifier_error", "result_unavailable"] | None = None
        if key not in self._started or policy.mode != "enforce":
            outcome = "not_started"
        elif result is GuardrailOutcome.ALLOW or (
            result is GuardrailOutcome.FLAGGED and check.action is GuardrailAction.ALLOW
        ):
            outcome = "allow"
        elif result is GuardrailOutcome.UNSUPPORTED:
            outcome = "unsupported"
        elif blocked:
            outcome = "block"
        else:
            outcome = "missed"
            reason = (
                "deadline_expired"
                if result is GuardrailOutcome.TIMEOUT
                else "classifier_error"
                if result is GuardrailOutcome.UNAVAILABLE
                else "result_unavailable"
            )
        return InputCheckOutcome(
            policy_id=policy.policy_id,
            policy_revision=policy.revision,
            check_id=check.check_id,
            mode=policy.mode,
            action=check.action,
            protected=policy.protected,
            outcome=outcome,
            recovery_reason=reason,
            latest_user_window_sha256=next(iter(windows)) if len(windows) == 1 else None,
        )


_CURRENT: ContextVar[InputOutcomes | None] = ContextVar("input_guardrail_outcomes", default=None)
_LATEST_USER_WINDOW = ClassificationWindow(
    system_chars=0, latest_user_chars=30_000, prior_messages=0
)


def latest_user_window_digest(request: GatewayRequest) -> str:
    """Hash the exact latest-user classifier window, excluding unrelated text."""
    return _window_digest(request)[0]


def _window_digest(request: GatewayRequest) -> tuple[str, bool]:
    """Return the digest and whether its selected text can safely use identity caching."""
    latest = None
    immutable = True
    for message in reversed(request.messages):
        if message.role != "user":
            continue
        # Frozen Pydantic models can still carry caller-owned mutable dictionaries.
        # A blank mutable suffix can also change which earlier message is selected.
        if message.provider_native_item is not None or (
            message.content is None
            and (
                message.provider_anthropic_block is not None
                or message.provider_anthropic_blocks is not None
            )
        ):
            immutable = False
        if message_text(message).strip():
            latest = message
            break
    subject = request.model_copy(update={"messages": () if latest is None else (latest,)})
    payload = classification_window(subject, _LATEST_USER_WINDOW).encode("utf-8")
    digest = sha256()
    # hashlib releases the GIL above 2047 bytes; avoid a handoff for this bounded window.
    for offset in range(0, len(payload), 2047):
        digest.update(payload[offset : offset + 2047])
    return digest.hexdigest(), immutable


def record_input_outcome(
    policy: GuardrailPolicy, check: GuardrailCheck | None, outcome: GuardrailOutcome
) -> None:
    """Collect a bounded input result only inside its request-owned execution context."""
    current = _CURRENT.get()
    if current is not None:
        for candidate in policy.input_checks if check is None else (check,):
            if candidate.stage.value == "input":
                current.record(policy, candidate, outcome)
