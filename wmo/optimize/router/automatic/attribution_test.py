"""Tests for immutable real-overlap candidate attribution."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from wmo.common.core.artifacts import JsonObject, SourceIdentity
from wmo.common.models import ConnectionConfig, ModelSnapshot, RoutedCandidateSnapshot
from wmo.common.tasks import TaskCase
from wmo.common.traces import Trace, TraceSource, TraceSpan
from wmo.optimize.router.automatic.attribution import (
    RouterAttributionError,
    resolve_router_observed_attributions,
)
from wmo.simulation.ingest.model_identity import (
    CAPABILITIES_DIGEST_ATTRIBUTE,
    IdentityComponentProvenance,
    TraceModelIdentityEvidence,
    TraceModelIdentityEvidenceSet,
    normalized_capabilities_sha256,
)

_TIME = datetime(2026, 8, 14, tzinfo=UTC)


def test_unique_inferred_identity_maps_without_relabeling_fallback_digests() -> None:
    """Provider/model/revision uniqueness admits inferred telemetry as inferred only."""
    recorded = _fallback_model("openai", "gpt-test", None)
    selected = recorded.model_copy(
        update={"capabilities_sha256": "a" * 64, "connection_sha256": "b" * 64}
    )
    trace = _trace((recorded,))

    records = resolve_router_observed_attributions(
        (_task(trace),),
        (trace,),
        _evidence(trace, capabilities="inferred", connection="inferred"),
        _candidates(selected, _model("anthropic", "other", None, "c", "d")),
    )

    assert records[0].candidate_alias == "candidate-a"
    assert records[0].match_kind == "inferred_unique"


def test_accidental_inferred_digest_equality_remains_inferred() -> None:
    """Matching fallback bytes never upgrades inferred telemetry to declared exact evidence."""
    recorded = _fallback_model("openai", "gpt-test", None)
    trace = _trace((recorded,))

    record = resolve_router_observed_attributions(
        (_task(trace),),
        (trace,),
        _evidence(trace, capabilities="inferred", connection="inferred"),
        _candidates(recorded, _model("anthropic", "other", None, "c", "d")),
    )[0]

    assert record.candidate_model == recorded
    assert record.match_kind == "inferred_unique"


def test_mixed_declared_component_is_a_hard_constraint() -> None:
    """A declared capability digest cannot fall back through a unique base identity."""
    recorded = _fallback_model("openai", "gpt-test", None).model_copy(
        update={"capabilities_sha256": "a" * 64}
    )
    trace = _trace(
        (recorded,),
        attributes=({CAPABILITIES_DIGEST_ATTRIBUTE: "a" * 64},),
    )
    wrong = recorded.model_copy(
        update={"capabilities_sha256": "b" * 64, "connection_sha256": "c" * 64}
    )

    records = resolve_router_observed_attributions(
        (_task(trace),),
        (trace,),
        _evidence(trace, capabilities="declared", connection="inferred"),
        _candidates(wrong, _model("anthropic", "other", None, "d", "e")),
    )

    assert records == ()


def test_unique_inferred_mapping_omits_ambiguous_aliases_and_revision_drift() -> None:
    """Ambiguous or revision-drifted traces remain optional fit evidence."""
    recorded = _fallback_model("openai", "gpt-test", "2026-08")
    trace = _trace((recorded,))
    evidence = _evidence(trace, capabilities="inferred", connection="inferred")

    assert (
        resolve_router_observed_attributions(
            (_task(trace),),
            (trace,),
            evidence,
            _candidates(
                _model("openai", "gpt-test", "2026-08", "a", "b"),
                _model("openai", "gpt-test", "2026-08", "c", "d"),
            ),
        )
        == ()
    )
    assert (
        resolve_router_observed_attributions(
            (_task(trace),),
            (trace,),
            evidence,
            _candidates(
                _model("openai", "gpt-test", None, "a", "b"),
                _model("anthropic", "other", None, "c", "d"),
            ),
        )
        == ()
    )


def test_unspecified_and_missing_identity_require_full_snapshot_equality() -> None:
    """Direct and unspecified records never receive provider/model-only inference."""
    recorded = _model("openai", "gpt-test", None, "a", "b")
    trace = _trace((recorded,))
    candidates = _candidates(recorded, _model("anthropic", "other", None, "c", "d"))

    unspecified = resolve_router_observed_attributions(
        (_task(trace),),
        (trace,),
        _evidence(trace, capabilities="unspecified", connection="unspecified"),
        candidates,
    )[0]
    missing = resolve_router_observed_attributions(
        (_task(trace),),
        (trace,),
        None,
        candidates,
    )[0]

    assert unspecified.match_kind == "strict_snapshot"
    assert missing.match_kind == "strict_snapshot"

    changed = recorded.model_copy(update={"connection_sha256": "e" * 64})
    assert (
        resolve_router_observed_attributions(
            (_task(trace),),
            (trace,),
            None,
            _candidates(changed, _model("anthropic", "other", None, "c", "d")),
        )
        == ()
    )


def test_multi_span_trace_requires_one_candidate_alias() -> None:
    """Multiple model spans may agree on one alias but can never cross aliases."""
    first = _fallback_model("openai", "gpt-a", None)
    second = _fallback_model("openai", "gpt-b", None)
    same = _trace((first, first))
    cross = _trace((first, second))

    same_record = resolve_router_observed_attributions(
        (_task(same),),
        (same,),
        _evidence(same, capabilities="inferred", connection="inferred"),
        _candidates(first, second),
    )[0]
    assert len(same_record.spans) == 2

    assert (
        resolve_router_observed_attributions(
            (_task(cross),),
            (cross,),
            _evidence(cross, capabilities="inferred", connection="inferred"),
            _candidates(first, second),
        )
        == ()
    )


def test_trace_summary_reports_inference_when_any_span_requires_it() -> None:
    """Mixed exact and inferred spans retain inference in the trace-level summary."""
    recorded = _fallback_model("openai", "gpt-a", None)
    trace = _trace((recorded, recorded))
    evidence = TraceModelIdentityEvidenceSet(
        records=(
            TraceModelIdentityEvidence(
                trace_id=trace.trace_id,
                span_id=trace.spans[0].span_id,
                model=recorded,
                capabilities="unspecified",
                connection="unspecified",
            ),
            TraceModelIdentityEvidence(
                trace_id=trace.trace_id,
                span_id=trace.spans[1].span_id,
                model=recorded,
                capabilities="inferred",
                connection="inferred",
            ),
        )
    )

    record = resolve_router_observed_attributions(
        (_task(trace),),
        (trace,),
        evidence,
        _candidates(recorded, _fallback_model("anthropic", "other", None)),
    )[0]

    assert record.match_kind == "inferred_unique"


def test_trace_without_model_is_omitted_but_inconsistent_evidence_is_typed() -> None:
    """Unattributable traces are optional while corrupt evidence remains an error."""
    trace = _trace((None,))
    candidates = _candidates(
        _model("openai", "gpt-a", None, "a", "b"),
        _model("openai", "gpt-b", None, "c", "d"),
    )
    assert (
        resolve_router_observed_attributions(
            (_task(trace),),
            (trace,),
            TraceModelIdentityEvidenceSet(records=()),
            candidates,
        )
        == ()
    )

    modeled = _trace((candidates[0].model,))
    with pytest.raises(RouterAttributionError, match="evidence is inconsistent"):
        resolve_router_observed_attributions(
            (_task(modeled),),
            (modeled,),
            TraceModelIdentityEvidenceSet(records=()),
            candidates,
        )


def _fallback_model(provider: str, model_id: str, revision: str | None) -> ModelSnapshot:
    """Return the exact fallback snapshot produced by telemetry normalization.

    Args:
        provider: Source provider name.
        model_id: Source provider model identifier.
        revision: Exact optional source revision.

    Returns:
        Inferred capability and standard-connection snapshot.
    """
    return ModelSnapshot(
        provider=provider,
        model_id=model_id,
        revision=revision,
        capabilities_sha256=normalized_capabilities_sha256(
            {}, provider, model_id, revision, error_type=ValueError
        ),
        connection_sha256=ConnectionConfig(provider=provider).identity_sha256(),
    )


def _model(
    provider: str,
    model_id: str,
    revision: str | None,
    capabilities_prefix: str,
    connection_prefix: str,
) -> ModelSnapshot:
    """Return one deterministic selected model snapshot.

    Args:
        provider: Provider name.
        model_id: Provider model identifier.
        revision: Exact optional revision.
        capabilities_prefix: Character repeated into the capability digest.
        connection_prefix: Character repeated into the connection digest.

    Returns:
        Exact test model snapshot.
    """
    return ModelSnapshot(
        provider=provider,
        model_id=model_id,
        revision=revision,
        capabilities_sha256=capabilities_prefix * 64,
        connection_sha256=connection_prefix * 64,
    )


def _candidates(
    first: ModelSnapshot,
    second: ModelSnapshot,
) -> tuple[RoutedCandidateSnapshot, RoutedCandidateSnapshot]:
    """Return two explicitly selected candidate aliases.

    Args:
        first: Candidate A model snapshot.
        second: Candidate B model snapshot.

    Returns:
        Ordered selected candidates.
    """
    return (
        RoutedCandidateSnapshot(alias="candidate-a", model=first),
        RoutedCandidateSnapshot(alias="candidate-b", model=second),
    )


def _trace(
    models: tuple[ModelSnapshot | None, ...],
    *,
    attributes: tuple[JsonObject, ...] | None = None,
) -> Trace:
    """Return one fit-source trace with the requested model spans.

    Args:
        models: Optional model snapshot for each span.
        attributes: Optional exact span attribute dictionaries.

    Returns:
        Canonical trace fixture.
    """
    span_attributes: tuple[JsonObject, ...] = attributes or tuple({} for _item in models)
    return Trace(
        trace_id="trace-1",
        task="Resolve the support request",
        spans=tuple(
            TraceSpan(
                span_id=f"span-{index}",
                name="agent.model_call",
                started_at=_TIME + timedelta(seconds=index),
                ended_at=_TIME + timedelta(seconds=index + 1),
                attributes=span_attributes[index],
                model=model,
            )
            for index, model in enumerate(models)
        ),
        source=TraceSource(
            identity=SourceIdentity(kind="otlp", source_id="fixture", sha256="f" * 64),
            semantic_convention_version="1.37.0",
        ),
    )


def _task(trace: Trace) -> TaskCase:
    """Return one fit task bound to a trace fixture.

    Args:
        trace: Source trace fixture.

    Returns:
        Canonical fit task.
    """
    return TaskCase(
        task_id="task-1",
        lineage_group_id="lineage-1",
        partition="fit",
        instruction=trace.task,
        workload_weight=1.0,
        source_trace_ids=(trace.trace_id,),
    )


def _evidence(
    trace: Trace,
    *,
    capabilities: IdentityComponentProvenance,
    connection: IdentityComponentProvenance,
) -> TraceModelIdentityEvidenceSet:
    """Return complete model-span evidence with one shared provenance classification.

    Args:
        trace: Exact source trace.
        capabilities: Capability provenance literal.
        connection: Connection provenance literal.

    Returns:
        Complete ordered identity evidence.
    """
    return TraceModelIdentityEvidenceSet(
        records=tuple(
            TraceModelIdentityEvidence(
                trace_id=trace.trace_id,
                span_id=span.span_id,
                model=span.model,
                capabilities=capabilities,
                connection=connection,
            )
            for span in trace.spans
            if span.model is not None
        )
    )
