"""One bounded author draw with durable intent, exact response replay and no implicit redraw."""

import json
from copy import deepcopy
from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
)
from exp.common.models import (
    ModelCatalog,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    ReasoningEffort,
    load_model_catalog,
)
from exp.common.project import ProjectStore, artifact_input
from exp.runtime.models.call_evidence import canonical_call
from exp.runtime.models.providers.admitted_transport import (
    AdmittedJsonTransport,
    require_admitted_transport,
)
from exp.runtime.models.providers.errors import ProviderRetryableResponseError
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.reasoning_view import reasoning_catalog
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.sessions.dispatch_ledger import DispatchLedger


class AuthorSettings(ContractModel):
    """Frozen model and finite allowance for one logical author response.

    Attributes:
        model_alias: Configured author alias.
        model: Exact model snapshot.
        catalog_sha256: Credential-free catalog digest.
        output_tokens: Maximum response tokens.
        reasoning_effort: Per-call override, or None to preserve the request/catalog default.
        maximum_input_bytes: Whole serialized request ceiling, without truncation.
        maximum_physical_attempts: Transport attempt ceiling for this single draw.
        deadline: Latest provider dispatch/response boundary.
    """

    model_alias: str = Field(min_length=1)
    model: ModelSnapshot
    catalog_sha256: Sha256
    output_tokens: int = Field(default=4096, ge=1, le=8192)
    reasoning_effort: ReasoningEffort | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    maximum_input_bytes: int = Field(default=128000, ge=512, le=512000)
    maximum_physical_attempts: int = Field(default=3, ge=1, le=10)
    deadline: AwareDatetime


class _ReasoningProfile(ContractModel):
    """Exact per-draw reasoning mode, distinct from the caller's logical request.

    Attributes:
        model: Unchanged provider model binding.
        catalog_sha256: Original catalog binding, before the per-call view.
        requested_effort: Explicit frozen author override.
        supports_reasoning: Original model's declared reasoning capability.
        effective_effort: Override sent on reasoning-capable routes; None means already off.
        logical_request_sha256: Caller request retained for existing producer replay.
        effective_request_sha256: Actual normalized request supplied to the provider client.
    """

    model: ModelSnapshot
    catalog_sha256: Sha256
    requested_effort: ReasoningEffort
    supports_reasoning: bool
    effective_effort: ReasoningEffort | None
    logical_request_sha256: Sha256
    effective_request_sha256: Sha256


def _reasoning_profile(
    spec: AuthorSettings,
    request: ModelRequest,
    effective: ModelRequest,
    *,
    supports_reasoning: bool,
) -> _ReasoningProfile:
    """Bind the dispatched profile so offline replay needs no catalog or credential access."""
    if spec.reasoning_effort is None:
        raise ValueError("a saved reasoning profile requires an explicit author override")
    if not supports_reasoning and spec.reasoning_effort != "none":
        raise ValueError("a non-reasoning author cannot carry an enabled reasoning profile")
    return _ReasoningProfile(
        model=spec.model,
        catalog_sha256=spec.catalog_sha256,
        requested_effort=spec.reasoning_effort,
        supports_reasoning=supports_reasoning,
        effective_effort=spec.reasoning_effort if supports_reasoning else None,
        logical_request_sha256=sha256_json(request),
        effective_request_sha256=sha256_json(effective),
    )


def preflight_author(project: ProjectStore, spec: AuthorSettings, request: ModelRequest) -> None:
    """Check exact catalog, typed output capability and conservative byte-based capacity."""
    _prepare_author(project, spec, request)


def _effective_request(spec: AuthorSettings, request: ModelRequest) -> ModelRequest:
    """Bind an explicit author effort without changing the caller's logical request."""
    if spec.reasoning_effort is None:
        return request
    if request.reasoning_effort not in (None, spec.reasoning_effort):
        raise ValueError("author request differs from its frozen reasoning effort")
    return request.model_copy(update={"reasoning_effort": spec.reasoning_effort})


def _prepare_author(
    project: ProjectStore, spec: AuthorSettings, request: ModelRequest
) -> ModelCatalog:
    """Validate the effective request and return its credential-free per-call catalog view."""
    request = _effective_request(spec, request)
    if len(request.model_dump_json().encode()) > spec.maximum_input_bytes:
        raise ValueError("author context exceeds its frozen byte allowance")
    if request.maximum_output_tokens != spec.output_tokens:
        raise ValueError("author request differs from its frozen output allowance")
    catalog = load_model_catalog(project.model_catalog_path)
    if sha256_json(catalog.model_dump(mode="json")) != spec.catalog_sha256:
        raise ValueError("author catalog changed since registration")
    model, capability = RuntimeModelCatalog(catalog, environment={}).snapshot(spec.model_alias)
    if model != spec.model or capability.supports_tools is not True:
        raise ValueError("author model must match and support typed tool output")
    if (
        capability.maximum_output_tokens is not None
        and spec.output_tokens > capability.maximum_output_tokens
    ):
        raise ValueError("author output exceeds model capacity")
    size = len(request.model_dump_json().encode()) + 4 * len(request.messages)
    if (
        capability.context_window_tokens is not None
        and size + spec.output_tokens > capability.context_window_tokens
    ):
        raise ValueError("author complete input and output exceed model capacity")
    require_admitted_transport(catalog, spec.model_alias, consumer="author")
    return reasoning_catalog(catalog, spec.model_alias, request.reasoning_effort)


class _AuthorRejection(ContractModel):
    """A known completed author draw that spent its output allowance without usable output.

    Attributes:
        reason: Terminal rejection classification, distinct from an unknown dispatch.
        output_tokens: Frozen output allowance of this draw.
        response_body: Actual provider body when attached by its owner, otherwise missing.
        response_sha256: Exact body digest, or None when the body was unavailable.
    """

    reason: Literal["output_allowance_exhausted"] = "output_allowance_exhausted"
    output_tokens: int = Field(ge=1)
    response_body: JsonObject | None = Field(default=None, repr=False)
    response_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def _bind_response(self) -> Self:
        """Keep missing evidence explicit and bind only an actual saved provider body."""
        expected = None if self.response_body is None else sha256_json(self.response_body)
        if self.response_sha256 != expected:
            raise ValueError("author rejection response body differs from its saved digest")
        return self

    def error(self) -> ProviderRetryableResponseError:
        """Restore the same known budget-stop class without fabricating a model response."""
        return ProviderRetryableResponseError(
            "author exhausted its frozen output allowance; "
            "saved known rejection, no implicit redraw",
            hit_length_limit=True,
            response_body=deepcopy(self.response_body),
        )


def _verify_author_intent(
    project: ProjectStore,
    spec: AuthorSettings,
    request: ModelRequest,
    effective: ModelRequest,
    sources: tuple[ArtifactInput, ...],
    intent_id: str,
) -> ArtifactInput:
    """Verify the exact original settings, source disclosure and dispatched profile offline."""
    store = project.artifacts
    intent = store.read(intent_id).manifest
    if (
        intent.artifact_type != "author-intent"
        or intent.inputs != sources
        or json.loads(store.read_bytes(intent_id, "spec.json")) != spec.model_dump(mode="json")
        or ModelRequest.model_validate_json(store.read_bytes(intent_id, "request.json")) != request
        or (
            spec.reasoning_effort is not None
            and ModelRequest.model_validate_json(
                store.read_bytes(intent_id, "effective-request.json")
            )
            != effective
        )
    ):
        raise ValueError("author response lost its original request or provenance")
    if spec.reasoning_effort is not None:
        profile = _ReasoningProfile.model_validate_json(
            store.read_bytes(intent_id, "reasoning-profile.json")
        )
        if profile != _reasoning_profile(
            spec, request, effective, supports_reasoning=profile.supports_reasoning
        ):
            raise ValueError("author response lost its original reasoning profile")
    return artifact_input(intent)


def author_response(
    project: ProjectStore,
    spec: AuthorSettings,
    request: ModelRequest,
    *,
    identity: str,
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
) -> tuple[ArtifactInput, ModelResponse]:
    """Save one draw under the caller's exclusive operation lease; replay requires exact pins.

    The caller holds its operation file lock through publication. A durable intent without a
    response never buys another draw. A known output-allowance rejection has its own immutable
    receipt and replays as the same known budget stop. Other malformed output stays available.
    Logical requests remain unchanged for producer replay; explicit reasoning overrides add the
    exact dispatched request and its catalog-bound profile to the same immutable intent.
    """
    store = project.artifacts
    effective = _effective_request(spec, request)
    sources = tuple(sorted(set(sources), key=lambda ref: ref.artifact_id))
    for source in sources:
        if artifact_input(store.read(source.artifact_id).manifest) != source:
            raise ValueError("author source changed; pin the saved immutable reference")
    intent_id, response_id = identity + "-intent", identity + "-response"
    rejection_id = identity + "-rejection"
    if store.exists(response_id) and store.exists(rejection_id):
        raise ValueError("author identity has conflicting saved outcomes")
    if store.exists(rejection_id):
        intent_ref = _verify_author_intent(project, spec, request, effective, sources, intent_id)
        rejected = store.read(rejection_id).manifest
        rejection = _AuthorRejection.model_validate_json(
            store.read_bytes(rejection_id, "rejection.json")
        )
        if (
            rejected.artifact_type != "author-rejection"
            or rejected.inputs != (intent_ref,)
            or rejection.output_tokens != spec.output_tokens
        ):
            raise ValueError("author rejection lost its original allowance or provenance")
        raise rejection.error()
    if not store.exists(response_id):
        if store.exists(intent_id):
            raise ValueError("author dispatch has an unknown outcome; no implicit redraw")
        catalog = _prepare_author(project, spec, request)
        if datetime.now(UTC) >= spec.deadline:
            raise ValueError("author deadline expired before dispatch")
        request_files: dict[str, ContractModel] = {"spec.json": spec, "request.json": request}
        if spec.reasoning_effort is not None:
            request_files["effective-request.json"] = effective
            _, capability = RuntimeModelCatalog(catalog, environment={}).snapshot(spec.model_alias)
            request_files["reasoning-profile.json"] = _reasoning_profile(
                spec, request, effective, supports_reasoning=capability.supports_reasoning
            )
        intent = store.write_json(
            artifact_id=intent_id,
            artifact_type="author-intent",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=sources,
            ),
            files=request_files,
        )
        transport = AdmittedJsonTransport(
            HttpxJsonTransport(),
            artifacts=store,
            code_revision=code_revision,
            deadline=spec.deadline,
            ledger=DispatchLedger(
                store.project_directory / (identity + ".sqlite"),
                spec.maximum_physical_attempts,
                sha256_json(spec),
            ),
        )
        registry = RuntimeModelCatalog(catalog, transport_factory=lambda: transport)
        try:
            with canonical_call(artifact_input(intent)):
                response = registry.resolve(spec.model_alias).client.complete(effective)
        except ProviderRetryableResponseError as error:
            if not error.hit_length_limit:
                raise
            rejection = _AuthorRejection(
                output_tokens=spec.output_tokens,
                response_body=deepcopy(error.response_body),
                response_sha256=(
                    None if error.response_body is None else sha256_json(error.response_body)
                ),
            )
            store.write_json(
                artifact_id=rejection_id,
                artifact_type="author-rejection",
                envelope=ArtifactEnvelope(
                    schema_version=1,
                    created_at=datetime.now(UTC),
                    code_revision=code_revision,
                    inputs=(artifact_input(intent),),
                ),
                files={"rejection.json": rejection},
            )
            raise rejection.error() from error
        store.write_json(
            artifact_id=response_id,
            artifact_type="author-response",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=(artifact_input(intent),),
            ),
            files={"response.json": response},
        )
    intent_ref = _verify_author_intent(project, spec, request, effective, sources, intent_id)
    response_manifest = store.read(response_id).manifest
    if response_manifest.inputs != (intent_ref,):
        raise ValueError("author response lost its original request or provenance")
    return artifact_input(response_manifest), ModelResponse.model_validate_json(
        store.read_bytes(response_id, "response.json")
    )
