"""Tests for the pluggable optimizer interface (result types + artifact refs)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest
from pydantic import ValidationError

from wmo.optimize.base import (
    ArtifactProvenance,
    ArtifactRef,
    OptimizeMetrics,
    Optimizer,
    OptimizeResult,
    ResumableOptimizer,
)
from wmo.optimize.gepa import GEPAOptimizer

if TYPE_CHECKING:
    from wmo.optimize.judge import Judge
    from wmo.providers.base import Provider


def test_optimize_result_defaults_are_prompt_only() -> None:
    result = OptimizeResult(prompt="serve this")
    assert result.prompt == "serve this"
    assert result.artifacts == []
    assert result.metrics == OptimizeMetrics()


def test_optimize_metrics_fresh_recheck_fields_default_to_none_not_zero() -> None:
    # A "no fresh comparison happened" build (GEPA never ran, or its search-time winner was
    # already base) must be distinguishable from a real measured tie: None, never a 0.0 that a
    # reader could mistake for "GEPA ran and found no difference".
    metrics = OptimizeMetrics()
    assert metrics.base_fresh is None
    assert metrics.best_fresh is None
    assert metrics.fresh_delta is None
    # Round-trips through JSON as `null`, not omitted or coerced to 0.0.
    dumped = json.loads(metrics.model_dump_json())
    assert dumped["base_fresh"] is None
    assert dumped["best_fresh"] is None
    assert dumped["fresh_delta"] is None


def test_artifact_refs_cover_every_optimizer_family() -> None:
    # prompt (GEPA), routing_policy (routing), model_weights + adapter (future distillation):
    # the result type must carry each so a training-type optimizer FITS without schema changes.
    refs = [
        ArtifactRef(kind="prompt"),
        ArtifactRef(kind="routing_policy", path=".wmo/models/tau/policy.json"),
        ArtifactRef(kind="model_weights", path="s3://bucket/ckpt-30b"),
        ArtifactRef(kind="adapter", path=".wmo/adapters/lora-1", metadata={"rank": 32}),
    ]
    result = OptimizeResult(prompt="", artifacts=refs)
    assert [a.kind for a in result.artifacts] == [
        "prompt",
        "routing_policy",
        "model_weights",
        "adapter",
    ]


def test_artifact_ref_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        ArtifactRef.model_validate({"kind": "vibes"})


def test_gepa_optimizer_satisfies_the_protocol() -> None:
    provider = cast("Provider", None)  # protocol check needs no live backend
    judge = cast("Judge", None)
    assert isinstance(GEPAOptimizer(provider, judge), Optimizer)


def test_routing_policies_are_never_exportable() -> None:
    # The ownership boundary: customer-owned checkpoints leave the platform, routing policies
    # never do - even when a caller explicitly claims otherwise.
    policy = ArtifactRef(kind="routing_policy", path="policy.json", exportable=True)
    assert policy.exportable is False
    weights = ArtifactRef(kind="model_weights", path="s3://bucket/ckpt")
    assert weights.exportable is True

    # Assignment re-validates, so the boundary cannot be flipped after construction.
    policy.exportable = True
    assert policy.exportable is False
    # Switching an exportable artifact's kind pulls its exportability with it.
    weights.kind = "routing_policy"
    assert weights.exportable is False
    # model_copy(update=...) bypasses validation by pydantic design (documented on ArtifactRef);
    # anything that crosses a serialization boundary is re-normalized.
    leaked = policy.model_copy(update={"exportable": True})
    assert leaked.exportable is True
    assert ArtifactRef.model_validate_json(leaked.model_dump_json()).exportable is False


def test_exportable_checkpoint_carries_provenance() -> None:
    ref = ArtifactRef(
        kind="model_weights",
        path="s3://bucket/ckpt-30b",
        provenance=ArtifactProvenance(
            optimizer="distill",
            base_model="qwen3-30b",
            trace_count=1200,
            trace_sha256="ab" * 32,
            created_at="2026-07-25T00:00:00Z",
            config={"seed": 42},
        ),
    )
    assert ref.provenance is not None
    assert ref.provenance.trace_count == 1200
    # Round-trips through JSON so the platform can persist ownership records verbatim.
    again = ArtifactRef.model_validate_json(ref.model_dump_json())
    assert again.provenance == ref.provenance


def test_resumable_optimizer_is_a_runtime_checkable_shape() -> None:
    class WarmStart:
        def resume(self, prior: ArtifactRef, new_traces: list, budget: int) -> OptimizeResult:
            return OptimizeResult(prompt="resumed")

    class ColdOnly:
        def optimize(self) -> None: ...

    assert isinstance(WarmStart(), ResumableOptimizer)
    assert not isinstance(ColdOnly(), ResumableOptimizer)
