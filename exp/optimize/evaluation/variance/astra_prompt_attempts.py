"""One refinement draw's attempts: a transient or refused attempt is a receipt, never a verdict.

The shared author-response owner saves an intent before dispatch and a response after it, and
refuses to redraw an intent without a response. A Codex call can fail at the transport level
(timeout, rate limit, non-zero exit) or be refused because the event stream showed a tool; both
leave an intent without a response. :data:`~.astra_prompt_states.DRAW_MACHINE` makes the
lifecycle explicit, as the environment review's draw machine does:

- attempt ``n`` of a draw has the identity ``<draw>`` (``n = 0``) or ``<draw>-retry-<n>``;
- ``pending`` -> ``answered`` when the response is saved; it replays with no call from then on;
- ``pending`` -> ``transient_failed`` (transport-level, by the shared retry classifier or an
  exhausted attempt budget) or ``refused`` (a Codex tool event): a ``<draw>-failed-<n>`` receipt
  is saved, :class:`TransientDraw` is raised, and nothing is used as an answer;
- the next run moves it back to ``pending`` (``retried``) under attempt ``n + 1``.

Other failures are not retried here; the caller records them as the draw's failure.
"""

from __future__ import annotations

from datetime import UTC, datetime

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, ContractModel
from exp.common.models import ModelRequest, ModelResponse
from exp.common.project import ProjectStore
from exp.optimize.evaluation.factory.authoring import AuthorSettings, author_response
from exp.optimize.evaluation.variance.astra_prompt_states import DRAW_MACHINE, DrawState
from exp.optimize.evaluation.variance.refined_prompts import journal_of
from exp.runtime.models.providers.transport import ProviderTransportError, classify_retry
from exp.runtime.sessions.dispatch_ledger import DispatchCapExhaustedError

FAILED_TYPE = "astra-prompt-draw-failure"
"""Artifact type of one failed attempt's receipt."""

REFUSED_CODE = "codex_tool_use_refused"
"""Error code of a Codex output refused for a tool event."""


class TransientDraw(RuntimeError):
    """A draw's attempt failed transiently or was refused; the draw stays pending for a rerun.

    Attributes:
        identity: The draw identity.
        kind: ``transient_failed`` or ``refused``.
        reason: Why.
    """

    def __init__(self, identity: str, kind: str, reason: str) -> None:
        """Name the draw and why."""
        super().__init__(f"draw {identity[:40]} {kind} ({reason}); it is retried on the next run")
        self.identity, self.kind, self.reason = identity, kind, reason


class FailedAttempt(ContractModel):
    """The receipt of one failed attempt.

    Attributes:
        attempt: The attempt identity.
        kind: ``transient_failed`` or ``refused``.
        reason: The classification and message.
    """

    attempt: str
    kind: str
    reason: str


def failure_kind(error: BaseException) -> tuple[str, str] | None:
    """``(kind, reason)`` of a transient or refused failure anywhere in the cause chain."""
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, ProviderTransportError) and current.error_code == REFUSED_CODE:
            return DrawState.REFUSED.value, str(current)[:300]
        if isinstance(current, DispatchCapExhaustedError):
            return DrawState.TRANSIENT_FAILED.value, "attempt_budget_exhausted"
        if isinstance(current, Exception):
            classification = classify_retry(current)
            if classification.retryable:
                return DrawState.TRANSIENT_FAILED.value, classification.reason
        current = current.__cause__ or current.__context__
    return None


def _receipts(project: ProjectStore, identity: str) -> list[FailedAttempt]:
    """Every saved failed-attempt receipt of a draw, in attempt order."""
    found: list[FailedAttempt] = []
    while project.artifacts.exists(f"{identity}-failed-{len(found)}"):
        raw = project.artifacts.read_bytes(f"{identity}-failed-{len(found)}", "failed.json")
        found.append(FailedAttempt.model_validate_json(raw))
    return found


def attempt_draw(
    project: ProjectStore,
    settings: AuthorSettings,
    request: ModelRequest,
    *,
    identity: str,
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
) -> tuple[ArtifactInput, ModelResponse]:
    """Replay the draw's answered attempt, or make its next attempt.

    Raises:
        TransientDraw: The attempt failed transiently or was refused; a receipt was saved.
    """
    failed = _receipts(project, identity)
    attempt = identity if not failed else f"{identity}-retry-{len(failed)}"
    if project.artifacts.exists(f"{attempt}-response"):
        return author_response(
            project, settings, request, identity=attempt, sources=sources,
            code_revision=code_revision,
        )  # fmt: skip
    journal = journal_of(project)
    if failed:
        DRAW_MACHINE.move(
            identity, DrawState(failed[-1].kind), "retried", journal, target=DrawState.PENDING
        )
    else:
        DRAW_MACHINE.move(identity, None, DrawState.PENDING.value, journal)
    try:
        result = author_response(
            project, settings, request, identity=attempt, sources=sources,
            code_revision=code_revision,
        )  # fmt: skip
    except Exception as error:
        found = failure_kind(error)
        if found is None:
            raise
        kind, reason = found
        DRAW_MACHINE.move(identity, DrawState.PENDING, kind, journal, target=DrawState(kind))
        project.artifacts.write_json(
            artifact_id=f"{identity}-failed-{len(failed)}",
            artifact_type=FAILED_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=(),
            ),  # fmt: skip
            files={"failed.json": FailedAttempt(attempt=attempt, kind=kind, reason=reason)},
        )
        raise TransientDraw(identity, kind, reason) from error
    DRAW_MACHINE.move(identity, DrawState.PENDING, "answered", journal, target=DrawState.ANSWERED)
    return result
