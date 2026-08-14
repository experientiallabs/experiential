"""Request-time capability eligibility for the frozen router runtime."""

from typing import cast

import pytest

from wmo.common.core.artifacts import sha256_json
from wmo.common.models import (
    AssistantAction,
    ModelCapabilities,
    ModelMessage,
    ModelRequest,
    ModelSnapshot,
    RoutedCandidateSnapshot,
    ToolCall,
)
from wmo.runtime.models import ResolvedModel, RuntimeModelCatalog
from wmo.runtime.router import RouterModelCapabilityError
from wmo.runtime.router.runtime import RouterRuntime
from wmo.runtime.router.runtime_test import _Catalog, _fixture, _request, _runtime


def test_selected_model_must_prove_tool_capability_before_dispatch() -> None:
    """A tool-bearing request never reaches an incapable selected model."""
    runtime, client = _runtime(candidate_tools=False)

    with pytest.raises(RouterModelCapabilityError, match="does not support tool calls"):
        runtime.complete(_request(tool_name="read"), episode_id="tool-episode")

    assert client.complete_calls == 0


def test_selected_model_must_prove_output_capacity_before_dispatch() -> None:
    """An explicit output limit never reaches a model with unknown capacity."""
    runtime, client = _runtime()
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="route me"),),
        maximum_output_tokens=100,
    )

    with pytest.raises(RouterModelCapabilityError, match="output-token capacity"):
        runtime.complete(request, episode_id="capacity-episode")

    assert client.complete_calls == 0


def test_replayed_tool_history_requires_tool_capability() -> None:
    """A tool result cannot bypass eligibility by omitting current tool definitions."""
    runtime, client = _runtime(candidate_tools=False)
    request = ModelRequest(
        messages=(
            ModelMessage(role="user", content="read it"),
            ModelMessage(
                role="assistant",
                assistant_action=AssistantAction(
                    tool_calls=(ToolCall(call_id="call-a", name="read", arguments={"path": "a"}),)
                ),
            ),
            ModelMessage(role="tool", content="done", tool_call_id="call-a"),
        )
    )

    with pytest.raises(RouterModelCapabilityError, match="does not support tool calls"):
        runtime.complete(request, episode_id="history-episode")

    assert client.complete_calls == 0


def test_selection_uses_an_eligible_frozen_candidate_before_dispatch() -> None:
    """An incapable preferred alias does not mask another eligible frozen candidate."""
    policy, manifest, bank, snapshots, client = _fixture()
    capabilities = {
        "cheap": ModelCapabilities(),
        "baseline": ModelCapabilities(supports_tools=True),
        "embedder": ModelCapabilities(supports_embeddings=True),
    }
    snapshots = {
        alias: snapshot.model_copy(update={"capabilities_sha256": sha256_json(capabilities[alias])})
        for alias, snapshot in snapshots.items()
    }
    policy = policy.model_copy(
        update={
            "candidates": tuple(
                RoutedCandidateSnapshot(alias=item.alias, model=snapshots[item.alias])
                for item in policy.candidates
            ),
            "embedder": snapshots[policy.embedder_alias],
        }
    )

    class _MixedCatalog(_Catalog):
        def snapshot(self, alias: str) -> tuple[ModelSnapshot, ModelCapabilities]:
            return snapshots[alias], capabilities[alias]

        def resolve(self, alias: str) -> ResolvedModel:
            snapshot, capability = self.snapshot(alias)
            return ResolvedModel(
                alias,
                snapshot,
                capability,
                client,
                client if capability.supports_embeddings else None,
            )

    runtime = RouterRuntime(
        policy,
        manifest,
        bank,
        cast(RuntimeModelCatalog, _MixedCatalog(snapshots, client)),
        pricing_snapshot_id="pricing-a",
        pricing_snapshot_sha256="a" * 64,
        pricing_candidate_aliases=bank.candidate_aliases,
    )

    result = runtime.complete(_request(tool_name="read"), episode_id="eligible")

    assert result.decision.selected_alias == "baseline"
    assert result.decision.fallback_reason == "capability_eligibility"
    assert client.complete_calls == 1


def test_capability_fallback_replaces_the_sticky_episode_model() -> None:
    """A later capability fallback remains sticky for subsequent ordinary turns."""
    policy, manifest, bank, snapshots, client = _fixture()
    capabilities = {
        "cheap": ModelCapabilities(),
        "baseline": ModelCapabilities(supports_tools=True),
        "embedder": ModelCapabilities(supports_embeddings=True),
    }
    snapshots = {
        alias: snapshot.model_copy(update={"capabilities_sha256": sha256_json(capabilities[alias])})
        for alias, snapshot in snapshots.items()
    }
    policy = policy.model_copy(
        update={
            "candidates": tuple(
                RoutedCandidateSnapshot(alias=item.alias, model=snapshots[item.alias])
                for item in policy.candidates
            ),
            "embedder": snapshots[policy.embedder_alias],
        }
    )

    class _MixedCatalog(_Catalog):
        def snapshot(self, alias: str) -> tuple[ModelSnapshot, ModelCapabilities]:
            return snapshots[alias], capabilities[alias]

        def resolve(self, alias: str) -> ResolvedModel:
            snapshot, capability = self.snapshot(alias)
            return ResolvedModel(
                alias,
                snapshot,
                capability,
                client,
                client if capability.supports_embeddings else None,
            )

    runtime = RouterRuntime(
        policy,
        manifest,
        bank,
        cast(RuntimeModelCatalog, _MixedCatalog(snapshots, client)),
        pricing_snapshot_id="pricing-a",
        pricing_snapshot_sha256="a" * 64,
        pricing_candidate_aliases=bank.candidate_aliases,
    )

    first = runtime.select(_request(), episode_id="eligible-sticky")
    fallback = runtime.select(_request(tool_name="read"), episode_id="eligible-sticky")
    later = runtime.select(
        ModelRequest(messages=(ModelMessage(role="user", content="ordinary later turn"),)),
        episode_id="eligible-sticky",
    )

    assert first.selected_alias == "cheap"
    assert fallback.selected_alias == "baseline"
    assert fallback.fallback_reason == "capability_eligibility"
    assert later.selected_alias == "baseline"
