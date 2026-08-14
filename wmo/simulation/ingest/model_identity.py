"""Versioned provenance for model identities extracted from normalized telemetry."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Literal

from pydantic import field_validator

from wmo.common.core.artifacts import ContractModel, JsonObject, Sha256
from wmo.common.models import ConnectionConfig, ModelSnapshot
from wmo.common.traces import Trace

IdentityComponentProvenance = Literal["declared", "inferred", "unspecified"]

CAPABILITIES_DIGEST_ATTRIBUTE = "wmo.model.capabilities_sha256"
CONNECTION_DIGEST_ATTRIBUTE = "wmo.model.connection_sha256"


class TraceModelIdentityEvidence(ContractModel):
    """Provenance for the model identity retained on one normalized model span."""

    trace_id: str
    span_id: str
    model: ModelSnapshot
    capabilities: IdentityComponentProvenance
    connection: IdentityComponentProvenance


class TraceModelIdentityEvidenceSet(ContractModel):
    """Complete deterministic model-identity provenance for one trace dataset."""

    schema_version: Literal[1] = 1
    records: tuple[TraceModelIdentityEvidence, ...]

    @field_validator("records")
    @classmethod
    def _require_sorted_unique_records(
        cls,
        value: tuple[TraceModelIdentityEvidence, ...],
    ) -> tuple[TraceModelIdentityEvidence, ...]:
        """Require one deterministically ordered record per model span.

        Args:
            value: Model-span provenance records.

        Returns:
            The unchanged validated tuple.

        Raises:
            ValueError: Record keys repeat or are not sorted.
        """
        keys = tuple((item.trace_id, item.span_id) for item in value)
        if len(set(keys)) != len(keys):
            raise ValueError("model identity evidence must not repeat trace and span IDs")
        if keys != tuple(sorted(keys)):
            raise ValueError("model identity evidence must be sorted by trace and span ID")
        return value


def normalized_connection_sha256(
    attributes: JsonObject,
    provider: str,
    *,
    error_type: type[ValueError],
) -> Sha256:
    """Return declared endpoint evidence or an explicitly inferred standard endpoint.

    Args:
        attributes: Source telemetry attributes or properties.
        provider: Exact provider name retained by normalization.
        error_type: Source-specific validation exception type.

    Returns:
        Declared connection digest or the provider standard-endpoint digest.

    Raises:
        ValueError: The declared extension value is not a SHA-256 digest.
    """
    declared = attributes.get(CONNECTION_DIGEST_ATTRIBUTE)
    if declared is not None:
        if not isinstance(declared, str) or not re.fullmatch(r"[0-9a-f]{64}", declared):
            raise error_type(f"{CONNECTION_DIGEST_ATTRIBUTE} must be a SHA-256 digest")
        return declared
    return ConnectionConfig(provider=provider).identity_sha256()


def normalized_capabilities_sha256(
    attributes: JsonObject,
    provider: str,
    model_id: str,
    revision: str | None,
    *,
    error_type: type[ValueError],
) -> Sha256:
    """Return declared capability evidence or an explicitly inferred model digest.

    Args:
        attributes: Source telemetry attributes or properties.
        provider: Exact provider name retained by normalization.
        model_id: Exact provider model identifier retained by normalization.
        revision: Exact optional model revision retained by normalization.
        error_type: Source-specific validation exception type.

    Returns:
        Declared capability digest or the deterministic telemetry fallback.

    Raises:
        ValueError: The declared extension value is not a SHA-256 digest.
    """
    declared = attributes.get(CAPABILITIES_DIGEST_ATTRIBUTE)
    if declared is not None:
        if not isinstance(declared, str) or not re.fullmatch(r"[0-9a-f]{64}", declared):
            raise error_type(f"{CAPABILITIES_DIGEST_ATTRIBUTE} must be a SHA-256 digest")
        return declared
    return hashlib.sha256(f"{provider}\0{model_id}\0{revision or ''}".encode()).hexdigest()


def normalized_model_identity_evidence(
    traces: Sequence[Trace],
) -> tuple[TraceModelIdentityEvidence, ...]:
    """Classify digest components retained by an OTLP or PostHog normalizer.

    Model provider, model ID, and revision are copied directly from the source telemetry.
    Capabilities and connection digests are declared only when their WMO extension attributes
    were present. Otherwise their normalized fallback values are explicitly inferred.

    Args:
        traces: Traces produced by a telemetry-aware normalizer.

    Returns:
        Complete ordered provenance for every normalized model span.
    """
    records = []
    for trace in traces:
        for span in trace.spans:
            if span.model is None:
                continue
            records.append(
                TraceModelIdentityEvidence(
                    trace_id=trace.trace_id,
                    span_id=span.span_id,
                    model=span.model,
                    capabilities=(
                        "declared"
                        if CAPABILITIES_DIGEST_ATTRIBUTE in span.attributes
                        else "inferred"
                    ),
                    connection=(
                        "declared" if CONNECTION_DIGEST_ATTRIBUTE in span.attributes else "inferred"
                    ),
                )
            )
    return tuple(sorted(records, key=lambda item: (item.trace_id, item.span_id)))


def complete_model_identity_evidence(
    traces: Sequence[Trace],
    supplied: Sequence[TraceModelIdentityEvidence] | None,
) -> TraceModelIdentityEvidenceSet:
    """Validate supplied normalization evidence or classify direct records as unspecified.

    Args:
        traces: Exact normalized trace records being persisted.
        supplied: Telemetry-normalizer evidence, or ``None`` for direct/programmatic records.

    Returns:
        Complete ordered evidence for every span carrying a model snapshot.

    Raises:
        ValueError: Supplied evidence is incomplete, duplicated, extra, or differs from a span.
    """
    expected = {
        (trace.trace_id, span.span_id): span.model
        for trace in traces
        for span in trace.spans
        if span.model is not None
    }
    if supplied is None:
        records = tuple(
            TraceModelIdentityEvidence(
                trace_id=trace_id,
                span_id=span_id,
                model=model,
                capabilities="unspecified",
                connection="unspecified",
            )
            for (trace_id, span_id), model in sorted(expected.items())
        )
    else:
        records = tuple(sorted(supplied, key=lambda item: (item.trace_id, item.span_id)))
    payload = TraceModelIdentityEvidenceSet(records=records)
    actual = {(item.trace_id, item.span_id): item.model for item in payload.records}
    if actual != expected:
        raise ValueError(
            "model identity evidence must cover every model span exactly and preserve its snapshot"
        )
    return payload


def require_model_identity_evidence_matches_traces(
    traces: Sequence[Trace],
    payload: TraceModelIdentityEvidenceSet,
) -> None:
    """Verify persisted identity evidence against exact normalized model spans.

    Args:
        traces: Recursively verified trace records.
        payload: Parsed versioned identity evidence.

    Raises:
        ValueError: Coverage, uniqueness, or snapshot identity differs.
    """
    complete_model_identity_evidence(traces, payload.records)
    spans = {
        (trace.trace_id, span.span_id): span
        for trace in traces
        for span in trace.spans
        if span.model is not None
    }
    for item in payload.records:
        span = spans[(item.trace_id, item.span_id)]
        _require_component_provenance(
            span.attributes,
            CAPABILITIES_DIGEST_ATTRIBUTE,
            item.capabilities,
            item.model.capabilities_sha256,
            normalized_capabilities_sha256(
                {},
                item.model.provider,
                item.model.model_id,
                item.model.revision,
                error_type=ValueError,
            ),
        )
        _require_component_provenance(
            span.attributes,
            CONNECTION_DIGEST_ATTRIBUTE,
            item.connection,
            item.model.connection_sha256,
            ConnectionConfig(provider=item.model.provider).identity_sha256(),
        )


def _require_component_provenance(
    attributes: JsonObject,
    key: str,
    provenance: IdentityComponentProvenance,
    recorded_digest: Sha256,
    inferred_digest: Sha256,
) -> None:
    """Verify one declared or inferred digest against canonical span attributes.

    Args:
        attributes: Exact normalized span attributes.
        key: WMO digest extension key.
        provenance: Persisted declared, inferred, or unspecified classification.
        recorded_digest: Digest retained in the span's model snapshot.
        inferred_digest: Deterministic normalizer fallback digest.

    Raises:
        ValueError: Declared or inferred provenance disagrees with the canonical span.
    """
    attribute = attributes.get(key)
    if provenance == "declared":
        if attribute != recorded_digest:
            raise ValueError(f"declared {key} is absent or differs from its model snapshot")
    elif provenance == "inferred" and (attribute is not None or recorded_digest != inferred_digest):
        raise ValueError(f"inferred {key} differs from the canonical telemetry fallback")
