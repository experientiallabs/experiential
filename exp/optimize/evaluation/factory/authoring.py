"""One bounded author draw with durable intent, exact response replay and no implicit redraw."""

import json
from datetime import UTC, datetime

from pydantic import AwareDatetime, Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    Sha256,
    sha256_json,
)
from exp.common.models import ModelRequest, ModelResponse, ModelSnapshot, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.runtime.models.providers.admitted_transport import (
    AdmittedJsonTransport,
    require_admitted_transport,
)
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.sessions.dispatch_ledger import DispatchLedger


class AuthorSettings(ContractModel):
    """Frozen model and finite allowance for one logical author response.

    Attributes:
        model_alias: Configured author alias.
        model: Exact model snapshot.
        catalog_sha256: Credential-free catalog digest.
        output_tokens: Maximum response tokens.
        maximum_input_bytes: Whole serialized request ceiling, without truncation.
        maximum_physical_attempts: Transport attempt ceiling for this single draw.
        deadline: Latest provider dispatch/response boundary.
    """

    model_alias: str = Field(min_length=1)
    model: ModelSnapshot
    catalog_sha256: Sha256
    output_tokens: int = Field(default=4096, ge=1, le=8192)
    maximum_input_bytes: int = Field(default=128000, ge=512, le=512000)
    maximum_physical_attempts: int = Field(default=3, ge=1, le=10)
    deadline: AwareDatetime


def preflight_author(project: ProjectStore, spec: AuthorSettings, request: ModelRequest) -> None:
    """Check exact catalog, typed output capability and conservative byte-based capacity."""
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
    response never buys another draw. Saved malformed output stays available for inspection.
    """
    store = project.artifacts
    sources = tuple(sorted(set(sources), key=lambda ref: ref.artifact_id))
    for source in sources:
        if artifact_input(store.read(source.artifact_id).manifest) != source:
            raise ValueError("author source changed; pin the saved immutable reference")
    intent_id, response_id = identity + "-intent", identity + "-response"
    if not store.exists(response_id):
        if store.exists(intent_id):
            raise ValueError("author dispatch has an unknown outcome; no implicit redraw")
        preflight_author(project, spec, request)
        if datetime.now(UTC) >= spec.deadline:
            raise ValueError("author deadline expired before dispatch")
        intent = store.write_json(
            artifact_id=intent_id,
            artifact_type="author-intent",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=sources,
            ),
            files={"spec.json": spec, "request.json": request},
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
        registry = RuntimeModelCatalog(
            load_model_catalog(project.model_catalog_path), transport_factory=lambda: transport
        )
        response = registry.resolve(spec.model_alias).client.complete(request)
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
    intent, response_manifest = store.read(intent_id).manifest, store.read(response_id).manifest
    if (
        intent.inputs != sources
        or json.loads(store.read_bytes(intent_id, "spec.json")) != spec.model_dump(mode="json")
        or ModelRequest.model_validate_json(store.read_bytes(intent_id, "request.json")) != request
        or response_manifest.inputs != (artifact_input(intent),)
    ):
        raise ValueError("author response lost its original request or provenance")
    return artifact_input(response_manifest), ModelResponse.model_validate_json(
        store.read_bytes(response_id, "response.json")
    )
