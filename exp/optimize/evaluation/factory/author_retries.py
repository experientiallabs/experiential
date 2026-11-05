"""Explicit author retries and finite execution windows preserve original authority."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import AwareDatetime, Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
)
from exp.common.project import ProjectStore, artifact_input
from exp.runtime.models.spend_ceiling import OperationSpendCeiling, current_spend_ceiling
from exp.runtime.sessions.dispatch_ledger import DispatchLedger

if TYPE_CHECKING:
    from exp.optimize.evaluation.factory.authoring import AuthorSettings


AUTHORIZATION_TYPE = "author-retry-authorization"
RESOLUTION_TYPE = "author-retry-resolution"


class AuthorRetryAuthorization(ContractModel):
    """One explicit retry lifetime, bound to original evidence and operation limits.

    Attributes:
        authorization_id: Caller-supplied identity of the explicit retry instruction.
        parent_intent: Exact failed author intent, retained unchanged.
        parent_code_revision: Historical producer of that original intent, never relabeled.
        terminal_responses: Every original physical attempt's definitive 503 response.
        settings_sha256: Canonical digest of the original saved settings.
        request_sha256: Canonical digest of the original logical request.
        effective_request_sha256: Digest of its explicit reasoning request, when present.
        sources: Original disclosed evidence references, in their original order.
        operation_ledger_sha256: Identity of the original resolved operation ledger path.
        operation_ceiling_usd: Original dollar ceiling, never enlarged by this grant.
        operation_unpriced_attempts: Original operation-wide unpriced ceiling.
        maximum_physical_attempts: Child lifetime, at most the original draw allowance.
        deadline: Original absolute author deadline, retained unchanged.
        execution_window: Explicit operational renewal, or None for the original deadline.
        effective_retry_deadline: Separate finite execution cutoff authorized for this child.
        child_identity: Deterministic child author identity owned by this registration.
    """

    authorization_id: str = Field(min_length=1, max_length=128)
    parent_intent: ArtifactInput
    parent_code_revision: str = Field(min_length=1)
    terminal_responses: tuple[ArtifactInput, ...]
    settings_sha256: Sha256
    request_sha256: Sha256
    effective_request_sha256: Sha256 | None
    sources: tuple[ArtifactInput, ...]
    operation_ledger_sha256: Sha256
    operation_ceiling_usd: float = Field(ge=0, allow_inf_nan=False)
    operation_unpriced_attempts: int = Field(ge=0, le=100_000)
    maximum_physical_attempts: int = Field(ge=1, le=10, strict=True)
    deadline: AwareDatetime
    execution_window: ArtifactInput | None = None
    effective_retry_deadline: AwareDatetime
    child_identity: str

    def require_operation(self) -> None:
        """Refuse fresh child calls outside the original operation and unchanged ceilings."""
        operation = current_spend_ceiling()
        if operation is None or operation_identity(operation) != self.operation_ledger_sha256:
            raise ValueError("author retry requires its original active operation ceiling")
        operation.require_accounted_attempts(
            tuple(
                ref.artifact_id.removeprefix("provider-attempt-").removesuffix("-response")
                for ref in self.terminal_responses
            )
        )
        summary = operation.summary()
        if (summary.ceiling_usd, summary.unpriced_attempt_ceiling) != (
            self.operation_ceiling_usd,
            self.operation_unpriced_attempts,
        ):
            raise ValueError("author retry operation ceilings changed since authorization")
        if datetime.now(UTC) >= self.effective_retry_deadline:
            reason = (
                "original deadline"
                if self.execution_window is None
                else "execution window deadline"
            )
            raise ValueError(f"author retry {reason} expired before dispatch")


def operation_identity(operation: OperationSpendCeiling) -> Sha256:
    """Identify an existing operation through its public canonical ledger path."""
    return sha256_json({"ledger_path": str(operation.path.resolve())})


def retry_resolution_id(identity: str) -> str:
    """Name one logical author's immutable child resolution without changing its identity."""
    return "author-retry-resolution-" + sha256_json({"parent_identity": identity})[:32]


def _authorization_id(parent: ArtifactInput) -> str:
    """Give each failed parent exactly one child registration, including on duplicate resumes."""
    return "author-retry-authorization-" + sha256_json(parent)[:32]


def _parent_values(project: ProjectStore, parent: ArtifactInput) -> tuple[JsonObject, JsonObject]:
    """Read exact original settings and request only from a verified failed author intent."""
    store = project.artifacts
    manifest = store.read(parent.artifact_id).manifest
    if (
        artifact_input(manifest) != parent
        or manifest.artifact_type != "author-intent"
        or not parent.artifact_id.endswith("-intent")
    ):
        raise ValueError("author retry requires its exact original author intent")
    identity = parent.artifact_id.removesuffix("-intent")
    if store.exists(identity + "-response") or store.exists(identity + "-rejection"):
        raise ValueError("author retry cannot replace a response or known author rejection")
    return (
        json.loads(store.read_bytes(parent.artifact_id, "spec.json")),
        json.loads(store.read_bytes(parent.artifact_id, "request.json")),
    )


def _effective_digest(project: ProjectStore, parent: ArtifactInput) -> Sha256 | None:
    """Bind an effective reasoning request when the original intent saved one."""
    manifest = project.artifacts.read(parent.artifact_id).manifest
    if "effective-request.json" not in manifest.files:
        return None
    return sha256_json(
        json.loads(project.artifacts.read_bytes(parent.artifact_id, "effective-request.json"))
    )


def _parent_limits(settings: JsonObject) -> tuple[int, datetime]:
    """Parse the original finite attempts and absolute deadline at the saved-JSON boundary."""
    cap, deadline = settings.get("maximum_physical_attempts"), settings.get("deadline")
    if not isinstance(cap, int) or isinstance(cap, bool) or not 1 <= cap <= 10:
        raise ValueError("author retry parent has no valid physical-attempt allowance")
    if not isinstance(deadline, str):
        raise ValueError("author retry parent has no original deadline")
    parsed = datetime.fromisoformat(deadline)
    if parsed.tzinfo is None:
        raise ValueError("author retry original deadline must be timezone aware")
    return cap, parsed


def _terminal_responses(
    project: ProjectStore, parent: ArtifactInput, settings: JsonObject
) -> tuple[ArtifactInput, ...]:
    """Require every exhausted ledger claim to have one retained, parent-bound terminal 503."""
    store = project.artifacts
    identity = parent.artifact_id.removesuffix("-intent")
    cap, _ = _parent_limits(settings)
    path = store.project_directory / (identity + ".sqlite")
    if not path.is_file():
        raise ValueError("author retry parent dispatch ledger is missing")
    ledger = DispatchLedger(path, cap, sha256_json(settings))
    attempts = ledger.attempt_ids()
    if len(attempts) != cap:
        raise ValueError("author retry requires the exhausted original attempt lifetime")
    responses: list[ArtifactInput] = []
    wire_digest: str | None = None
    for sequence, attempt in enumerate(attempts, 1):
        base = "provider-attempt-" + attempt
        intent = store.read(base + "-intent").manifest
        sent = json.loads(store.read_bytes(intent.artifact_id, "receipt.json"))
        if (
            intent.artifact_type != "provider-physical-attempt"
            or intent.inputs != (parent,)
            or sent.get("sequence") != sequence
            or sent.get("canonical_request") != parent.model_dump(mode="json")
            or not isinstance(sent.get("request_sha256"), str)
        ):
            raise ValueError("author retry physical intent lost its original parent")
        if wire_digest is not None and sent["request_sha256"] != wire_digest:
            raise ValueError("author retry original physical requests differ")
        wire_digest = sent["request_sha256"]
        if any(store.exists(base + suffix) for suffix in ("-failure", "-rejected", "-late")):
            raise ValueError("author retry cannot retry an ambiguous physical outcome")
        response = store.read(base + "-response").manifest
        received = json.loads(store.read_bytes(response.artifact_id, "receipt.json"))
        if (
            response.artifact_type != "provider-physical-attempt"
            or response.inputs != (artifact_input(intent),)
            or received.get("sequence") != sequence
            or received.get("http_status") != 503
            or received.get("known_unbilled") is not False
            or received.get("returned_after_deadline") is not False
        ):
            raise ValueError("author retry requires every definitive charged-or-unpriced HTTP 503")
        responses.append(artifact_input(response))
    return tuple(responses)


def register_author_retry(
    project: ProjectStore,
    parent_intent: ArtifactInput,
    *,
    operation: OperationSpendCeiling,
    authorization_id: str,
    code_revision: str,
    maximum_physical_attempts: int | None = None,
    execution_window: ArtifactInput | None = None,
) -> ArtifactInput:
    """Register one explicitly requested child under the caller's exclusive operation lock.

    This performs no provider call and never changes original receipts or limits. Every child
    physical dispatch still reserves against the same ambient operation. Repeating the same
    registration is idempotent; different instructions cannot replenish this parent's child.
    """
    store = project.artifacts
    settings, request = _parent_values(project, parent_intent)
    terminal = _terminal_responses(project, parent_intent, settings)
    parent = store.read(parent_intent.artifact_id).manifest
    cap, deadline = _parent_limits(settings)
    attempts = cap if maximum_physical_attempts is None else maximum_physical_attempts
    if attempts > cap:
        raise ValueError("author retry child allowance exceeds the original draw")
    summary = operation.summary()
    effective_deadline = deadline
    if execution_window is not None:
        window = load_author_execution_window(project, execution_window)
        if window.settings_sha256 != sha256_json(settings) or window.original_deadline != deadline:
            raise ValueError("author retry execution window differs from its original settings")
        window.require_operation()
        effective_deadline = window.execution_deadline
    authorization = AuthorRetryAuthorization(
        authorization_id=authorization_id,
        parent_intent=parent_intent,
        parent_code_revision=parent.code_revision,
        terminal_responses=terminal,
        settings_sha256=sha256_json(settings),
        request_sha256=sha256_json(request),
        effective_request_sha256=_effective_digest(project, parent_intent),
        sources=parent.inputs,
        operation_ledger_sha256=operation_identity(operation),
        operation_ceiling_usd=summary.ceiling_usd,
        operation_unpriced_attempts=summary.unpriced_attempt_ceiling,
        maximum_physical_attempts=attempts,
        deadline=deadline,
        execution_window=execution_window,
        effective_retry_deadline=effective_deadline,
        child_identity="author-retry-"
        + sha256_json(
            {"parent": parent_intent.model_dump(mode="json"), "authorization_id": authorization_id}
        )[:32],
    )
    identity = _authorization_id(parent_intent)
    if store.exists(identity):
        if (
            load_author_retry(project, artifact_input(store.read(identity).manifest))
            != authorization
        ):
            raise ValueError("author retry already has a different immutable authorization")
        return artifact_input(store.read(identity).manifest)
    authorization.require_operation()
    manifest = store.write_json(
        artifact_id=identity,
        artifact_type=AUTHORIZATION_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(
                sorted(
                    (parent_intent, *terminal, *((execution_window,) if execution_window else ())),
                    key=lambda ref: ref.artifact_id,
                )
            ),
        ),
        files={"authorization.json": authorization},
    )
    return artifact_input(manifest)


def load_author_retry(project: ProjectStore, reference: ArtifactInput) -> AuthorRetryAuthorization:
    """Verify the complete immutable authorization and its still-retained original failures."""
    store = project.artifacts
    manifest = store.read(reference.artifact_id).manifest
    authorization = AuthorRetryAuthorization.model_validate_json(
        store.read_bytes(reference.artifact_id, "authorization.json")
    )
    settings, request = _parent_values(project, authorization.parent_intent)
    parent = store.read(authorization.parent_intent.artifact_id).manifest
    cap, deadline = _parent_limits(settings)
    effective_deadline = deadline
    if authorization.execution_window is not None:
        window = load_author_execution_window(project, authorization.execution_window)
        if (
            window.settings_sha256 != sha256_json(settings)
            or window.original_deadline != deadline
            or window.operation_ledger_sha256 != authorization.operation_ledger_sha256
            or window.operation_ceiling_usd != authorization.operation_ceiling_usd
            or window.operation_unpriced_attempts != authorization.operation_unpriced_attempts
        ):
            raise ValueError("author retry execution window changed its original authority")
        effective_deadline = window.execution_deadline
    if (
        artifact_input(manifest) != reference
        or manifest.artifact_type != AUTHORIZATION_TYPE
        or reference.artifact_id != _authorization_id(authorization.parent_intent)
        or manifest.inputs
        != tuple(
            sorted(
                (
                    authorization.parent_intent,
                    *authorization.terminal_responses,
                    *((authorization.execution_window,) if authorization.execution_window else ()),
                ),
                key=lambda ref: ref.artifact_id,
            )
        )
        or authorization.parent_code_revision != parent.code_revision
        or authorization.settings_sha256 != sha256_json(settings)
        or authorization.request_sha256 != sha256_json(request)
        or authorization.effective_request_sha256
        != _effective_digest(project, authorization.parent_intent)
        or authorization.sources != parent.inputs
        or authorization.deadline != deadline
        or authorization.effective_retry_deadline != effective_deadline
        or authorization.maximum_physical_attempts > cap
        or authorization.terminal_responses
        != _terminal_responses(project, authorization.parent_intent, settings)
        or authorization.child_identity
        != "author-retry-"
        + sha256_json(
            {
                "parent": authorization.parent_intent.model_dump(mode="json"),
                "authorization_id": authorization.authorization_id,
            }
        )[:32]
    ):
        raise ValueError("author retry authorization differs from its original immutable evidence")
    return authorization


class AuthorRetryResolution(ContractModel):
    """The actual saved child result selected by an explicit retry authorization.

    Attributes:
        retry_registration: Exact immutable retry registration.
        response: Actual child author response, never relabeled as the failed parent's response.
    """

    retry_registration: ArtifactInput
    response: ArtifactInput


class AuthorExecutionWindow(ContractModel):
    """One explicit finite operational renewal without changing original author semantics.

    Attributes:
        window_id: Identity of the operator's finite retry watchdog renewal.
        settings_sha256: Canonical digest of the exact original author settings.
        original_deadline: Original saved cutoff, retained as provenance.
        original_maximum_physical_attempts: Original attempt ceiling, never enlarged.
        execution_deadline: Separate finite cutoff for fresh dispatch under this window.
        operation_ledger_sha256: Original canonical operation ledger identity.
        operation_ceiling_usd: Original dollar limit.
        operation_unpriced_attempts: Original unpriced-attempt limit.
        sources: Exact original spec, plan or approval provenance supplied by the owner.
    """

    window_id: str = Field(min_length=1, max_length=128)
    settings_sha256: Sha256
    original_deadline: AwareDatetime
    original_maximum_physical_attempts: int = Field(ge=1, le=10, strict=True)
    execution_deadline: AwareDatetime
    operation_ledger_sha256: Sha256
    operation_ceiling_usd: float = Field(ge=0, allow_inf_nan=False)
    operation_unpriced_attempts: int = Field(ge=0, le=100_000)
    sources: tuple[ArtifactInput, ...]

    def require_settings(self, settings: AuthorSettings) -> None:
        """Bind original settings, permitting only an explicitly smaller child attempt cap."""
        original = settings.model_dump(mode="json")
        if settings.maximum_physical_attempts > self.original_maximum_physical_attempts:
            raise ValueError("author execution window cannot enlarge the original attempt cap")
        original["maximum_physical_attempts"] = self.original_maximum_physical_attempts
        if (
            sha256_json(original) != self.settings_sha256
            or settings.deadline != self.original_deadline
        ):
            raise ValueError("author execution window differs from original author settings")

    def require_operation(self) -> None:
        """Require the unchanged original ambient operation before any fresh dispatch."""
        operation = current_spend_ceiling()
        if operation is None or operation_identity(operation) != self.operation_ledger_sha256:
            raise ValueError("author execution window requires its original active operation")
        summary = operation.summary()
        if (summary.ceiling_usd, summary.unpriced_attempt_ceiling) != (
            self.operation_ceiling_usd,
            self.operation_unpriced_attempts,
        ):
            raise ValueError("author execution window operation ceilings changed")
        if datetime.now(UTC) >= self.execution_deadline:
            raise ValueError("author execution window deadline expired before dispatch")


def register_author_execution_window(
    project: ProjectStore,
    settings: AuthorSettings,
    *,
    operation: OperationSpendCeiling,
    window_id: str,
    execution_deadline: datetime,
    code_revision: str,
    sources: tuple[ArtifactInput, ...] = (),
) -> ArtifactInput:
    """Register one explicit watchdog renewal of at most six hours under original limits.

    The caller owns the exclusive preparation operation lease. This never updates an original
    author intent, model settings, physical ledger, or operation ceiling. Sources bind the
    orchestration owner's separately retained original spec, plan and approval evidence.
    """
    store = project.artifacts
    sources = tuple(sorted(set(sources), key=lambda ref: ref.artifact_id))
    for source in sources:
        if artifact_input(store.read(source.artifact_id).manifest) != source:
            raise ValueError("author execution window source changed")
    summary = operation.summary()
    window = AuthorExecutionWindow(
        window_id=window_id,
        settings_sha256=sha256_json(settings),
        original_deadline=settings.deadline,
        original_maximum_physical_attempts=settings.maximum_physical_attempts,
        execution_deadline=execution_deadline,
        operation_ledger_sha256=operation_identity(operation),
        operation_ceiling_usd=summary.ceiling_usd,
        operation_unpriced_attempts=summary.unpriced_attempt_ceiling,
        sources=sources,
    )
    identity = (
        "author-execution-window-"
        + sha256_json(
            {
                "window_id": window_id,
                "settings": window.settings_sha256,
                "operation": window.operation_ledger_sha256,
            }
        )[:32]
    )
    if store.exists(identity):
        reference = artifact_input(store.read(identity).manifest)
        if load_author_execution_window(project, reference) != window:
            raise ValueError("author execution window already has different immutable limits")
        return reference
    now = datetime.now(UTC)
    if window.execution_deadline <= now or window.execution_deadline > now + timedelta(hours=6):
        raise ValueError("author execution window must be future and at most six hours")
    window.require_operation()
    return artifact_input(
        store.write_json(
            artifact_id=identity,
            artifact_type="author-execution-window",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=now,
                code_revision=code_revision,
                inputs=sources,
            ),
            files={"window.json": window},
        )
    )


def load_author_execution_window(
    project: ProjectStore, reference: ArtifactInput
) -> AuthorExecutionWindow:
    """Verify a saved operational renewal offline without granting fresh dispatch by itself."""
    store = project.artifacts
    manifest = store.read(reference.artifact_id).manifest
    window = AuthorExecutionWindow.model_validate_json(
        store.read_bytes(reference.artifact_id, "window.json")
    )
    expected = (
        "author-execution-window-"
        + sha256_json(
            {
                "window_id": window.window_id,
                "settings": window.settings_sha256,
                "operation": window.operation_ledger_sha256,
            }
        )[:32]
    )
    if (
        artifact_input(manifest) != reference
        or manifest.artifact_type != "author-execution-window"
        or reference.artifact_id != expected
        or manifest.inputs != window.sources
        or window.execution_deadline <= manifest.created_at
        or window.execution_deadline > manifest.created_at + timedelta(hours=6)
    ):
        raise ValueError("author execution window lost its original immutable evidence")
    return window


def author_execution_deadline(
    project: ProjectStore, settings: AuthorSettings, reference: ArtifactInput
) -> datetime:
    """Resolve a fresh dispatch's explicit operational cutoff under the original authority."""
    window = load_author_execution_window(project, reference)
    window.require_settings(settings)
    window.require_operation()
    return window.execution_deadline
