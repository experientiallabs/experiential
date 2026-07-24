"""Tests for the pluggable optimizer interface (result types + artifact refs)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from pydantic import ValidationError

from wmh.optimize.base import ArtifactRef, OptimizeMetrics, Optimizer, OptimizeResult
from wmh.optimize.gepa import GEPAOptimizer

if TYPE_CHECKING:
    from wmh.optimize.judge import Judge
    from wmh.providers.base import Provider


def test_optimize_result_defaults_are_prompt_only() -> None:
    result = OptimizeResult(prompt="serve this")
    assert result.prompt == "serve this"
    assert result.artifacts == []
    assert result.metrics == OptimizeMetrics()


def test_artifact_refs_cover_every_optimizer_family() -> None:
    # prompt (GEPA), routing_policy (routing), model_weights + adapter (future distillation):
    # the result type must carry each so a training-type optimizer FITS without schema changes.
    refs = [
        ArtifactRef(kind="prompt"),
        ArtifactRef(kind="routing_policy", path=".wmh/models/tau/policy.json"),
        ArtifactRef(kind="model_weights", path="s3://bucket/ckpt-30b"),
        ArtifactRef(kind="adapter", path=".wmh/adapters/lora-1", metadata={"rank": 32}),
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
