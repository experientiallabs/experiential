"""Bounded, replayable judge and analyst requests for paired executable evidence."""

import json
from datetime import UTC, datetime
from typing import Literal

from jsonschema import Draft202012Validator
from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    redact_secret_text,
    sha256_json,
)
from exp.common.core.json_schema import expand_local_references
from exp.common.models import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    ToolChoice,
    load_model_catalog,
)
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import ToolSchema
from exp.runtime.models.call_evidence import call_consumer, canonical_call
from exp.runtime.models.providers.admitted_transport import (
    AdmittedJsonTransport,
    require_admitted_transport,
)
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.sessions.dispatch_ledger import DispatchCapExhaustedError, DispatchLedger


class ReviewModel(ContractModel):
    """One pinned reviewer, without authority over execution or training admission.

    Attributes:
        alias: Explicit model catalog entry.
        model: Exact model and capability identity.
        output_tokens: Per-call output allowance.
    """

    alias: str = Field(min_length=1)
    model: ModelSnapshot
    output_tokens: int = Field(default=4096, ge=1, le=65536)


class ReviewUnavailable(RuntimeError):
    """A saved failed or uncertain review cannot be silently redrawn."""


def preflight_reviewer(
    project: ProjectStore, reviewer: ReviewModel, catalog_sha256: Sha256
) -> None:
    """Verify a reviewer without credentials, provider calls or environment execution."""
    catalog = load_model_catalog(project.model_catalog_path)
    if sha256_json(catalog.model_dump(mode="json")) != catalog_sha256:
        raise ValueError("equivalence catalog changed; freeze a new study")
    snapshot, caps = RuntimeModelCatalog(catalog, environment={}).snapshot(reviewer.alias)
    if snapshot != reviewer.model or caps.supports_tools is not True:
        raise ValueError("equivalence reviewer requires its pinned tool-capable model")
    require_admitted_transport(catalog, reviewer.alias, consumer="equivalence reviewer")
    if caps.context_window_tokens is None or (
        caps.maximum_output_tokens is not None
        and reviewer.output_tokens > caps.maximum_output_tokens
    ):
        raise ValueError(
            "equivalence reviewer needs sufficient declared context and output capacity"
        )


def review_inputs(project: ProjectStore, identity: str) -> tuple[ArtifactInput, ...]:
    """Return exact durable evidence for this review, including failed dispatches."""
    return tuple(
        artifact_input(project.artifacts.read(identity + suffix).manifest)
        for suffix in ("-intent", "-response", "-failure")
        if project.artifacts.exists(identity + suffix)
    )


def review_call(
    project: ProjectStore,
    reviewer: ReviewModel,
    body: JsonObject,
    *,
    identity: str,
    panel_id: str,
    catalog_sha256: Sha256,
    maximum_attempts: int,
    maximum_call_attempts: int,
    deadline: datetime,
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
    allow_dispatch: bool,
    consumer: Literal["analyst", "unknown"],
) -> JsonObject:
    """Replay raw results before deadlines or credentials; never redraw an unknown call.

    Args:
        project: Evidence and catalog owner.
        reviewer: Exact selected judge or analyst.
        body: Complete messages and local response schema; no truncation is permitted.
        identity: Unique role and case slot within the panel.
        panel_id: Shared review budget identity and caller-held panel lease.
        catalog_sha256: Frozen catalog digest.
        maximum_attempts: Shared physical attempt ceiling across all reviewers.
        maximum_call_attempts: This review's predeclared share, including transport retries.
        deadline: Dispatch deadline, not a deadline on reading saved results.
        sources: Exact measured evidence used to construct this request.
        code_revision: Pinned implementation.
        allow_dispatch: False for verification-only replay.
        consumer: Diagnostic role recorded with the physical request.

    Returns:
        Validated structured object from one immutable response.
    """
    if not 1 <= maximum_call_attempts <= maximum_attempts:
        raise ValueError("per-review allowance must fit the shared physical ceiling")
    schema, messages = body.get("response_schema"), body.get("messages")
    if not isinstance(schema, dict) or not isinstance(messages, list):
        raise ValueError("equivalence review requires messages and an explicit response schema")
    schema = expand_local_references(schema)
    request = ModelRequest(
        messages=tuple(ModelMessage.model_validate(row) for row in messages),
        tools=(
            ToolSchema(
                name="submit_proposal",
                description="Submit evidence-based review.",
                input_schema=schema,
            ),
        ),
        tool_choice=ToolChoice(name="submit_proposal"),
        maximum_output_tokens=reviewer.output_tokens,
    )
    store = project.artifacts
    envelope = ArtifactEnvelope(
        schema_version=1,
        created_at=datetime.now(UTC),
        code_revision=code_revision,
        inputs=tuple(sorted(set(sources), key=lambda ref: ref.artifact_id)),
    )
    intent_id, response_id, failure_id = (
        identity + suffix for suffix in ("-intent", "-response", "-failure")
    )
    if store.exists(intent_id):
        intent = store.read(intent_id).manifest
        if (
            intent.artifact_type != "equivalence-review-intent"
            or intent.code_revision != code_revision
            or intent.inputs != envelope.inputs
            or ModelRequest.model_validate_json(store.read_bytes(intent_id, "request.json"))
            != request
            or ReviewModel.model_validate_json(store.read_bytes(intent_id, "reviewer.json"))
            != reviewer
        ):
            raise ValueError("equivalence review no longer reproduces its registered request")
    elif allow_dispatch:
        # Intent also binds requests refused locally; all subsequent attempts replay that outcome.
        store.write_json(
            artifact_id=intent_id,
            artifact_type="equivalence-review-intent",
            envelope=envelope,
            files={"request.json": request, "reviewer.json": reviewer},
        )
        phase = "catalog_and_capacity"
        try:
            preflight_reviewer(project, reviewer, catalog_sha256)
            if datetime.now(UTC) >= deadline:
                raise TimeoutError("equivalence review dispatch deadline expired")
            catalog = load_model_catalog(project.model_catalog_path)
            _, caps = RuntimeModelCatalog(catalog, environment={}).snapshot(reviewer.alias)
            if len(request.model_dump_json().encode()) + reviewer.output_tokens + 1024 > (
                caps.context_window_tokens or 0
            ):
                raise ValueError("complete equivalence evidence exceeds reviewer context capacity")
            phase = "transport_setup"
            attempts = 0

            def require_allowance() -> None:
                """Bound retries inside this call without spending other reviewers' slots.

                The caller holds the panel lease, and an existing intent is never dispatched
                again. The shared durable ledger still accounts for every physical claim.
                """
                nonlocal attempts
                if attempts >= maximum_call_attempts:
                    raise DispatchCapExhaustedError(
                        f"per-review allowance {maximum_call_attempts} is spent; "
                        "remaining slots belong to other planned reviews"
                    )
                attempts += 1

            transport = AdmittedJsonTransport(
                HttpxJsonTransport(),
                artifacts=store,
                code_revision=code_revision,
                deadline=deadline,
                before_claim=require_allowance,
                ledger=DispatchLedger(
                    store.project_directory / (panel_id + "-reviews.sqlite"),
                    maximum_attempts,
                    sha256_json({"panel": panel_id}),
                ),
            )
            registry = RuntimeModelCatalog(catalog, transport_factory=lambda: transport)
            intent_ref = artifact_input(store.read(intent_id).manifest)
            phase = "provider_dispatch"
            with call_consumer(consumer), canonical_call(intent_ref):
                response = registry.resolve(reviewer.alias).client.complete(request)
            phase = "response_persistence"
            store.write_json(
                artifact_id=response_id,
                artifact_type="equivalence-review-response",
                envelope=envelope.model_copy(update={"inputs": (intent_ref,)}),
                files={"response.json": response},
            )
        except (ValueError, OSError, RuntimeError) as error:
            store.write_json(
                artifact_id=failure_id,
                artifact_type="equivalence-review-failure",
                envelope=envelope.model_copy(
                    update={"inputs": (artifact_input(store.read(intent_id).manifest),)}
                ),
                files={
                    "failure.json": {
                        "error_type": type(error).__name__,
                        "phase": phase,
                        "reason": redact_secret_text(str(error))[0][:2000],
                        "outcome": "review unavailable; no redraw",
                    }
                },
            )
    else:
        raise ValueError("saved equivalence report lacks its review intent")
    if not store.exists(response_id):
        detail = "outcome unknown; inspect its intent and physical receipts"
        if store.exists(failure_id):
            failure = json.loads(store.read_bytes(failure_id, "failure.json"))
            detail = redact_secret_text(
                f"{failure['phase']}: {failure['error_type']}: {failure['reason']}"
            )[0][:2000]
        raise ReviewUnavailable("equivalence review has no saved response; " + detail)
    manifest = store.read(response_id).manifest
    if manifest.artifact_type != "equivalence-review-response" or manifest.inputs != (
        artifact_input(store.read(intent_id).manifest),
    ):
        raise ValueError("review response lost its exact request binding")
    response = ModelResponse.model_validate_json(store.read_bytes(response_id, "response.json"))
    if response.model != reviewer.model or response.finish_reason.value != "completed":
        raise ValueError("equivalence reviewer returned an incomplete or different-model response")
    calls = response.output.tool_calls
    if len(calls) != 1 or calls[0].name != "submit_proposal":
        raise ValueError("equivalence reviewer must submit one typed proposal")
    if not Draft202012Validator(schema).is_valid(calls[0].arguments):
        raise ValueError("equivalence review does not satisfy its frozen response schema")
    return calls[0].arguments
