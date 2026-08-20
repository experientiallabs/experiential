"""Request-time capability eligibility for the frozen router runtime."""

from typing import cast

import pytest

from exp.common.models import (
    AssistantAction,
    ModelCapabilities,
    ModelMessage,
    ModelRequest,
    ModelSnapshot,
    RoutedCandidateSnapshot,
    ToolCall,
)
from exp.runtime.models import CatalogRoleName, ResolvedModel, RuntimeModelCatalog
from exp.runtime.router import RouterModelCapabilityError
from exp.runtime.router.runtime import RouterRuntime
from exp.runtime.router.runtime_test import _Catalog, _fixture, _request, _runtime


def test_selected_model_must_prove_tool_capability_before_dispatch() -> None:
    """A tool-bearing request never reaches an incapable selected model."""
    runtime, client = _runtime(candidate_tools=False)

    with pytest.raises(RouterModelCapabilityError, match="does not support tool calls"):
        runtime.complete(_request(tool_name="read"), episode_id="tool-episode")

    assert client.complete_calls == 0


def test_selected_model_output_capacity_blocks_only_an_explicit_smaller_limit() -> None:
    """Unknown output capacity dispatches while an explicit smaller limit fails closed."""
    runtime, client = _runtime()
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="route me"),),
        maximum_output_tokens=100,
    )

    runtime.complete(request, episode_id="capacity-episode")

    assert client.complete_calls == 1

    policy, manifest, bank, snapshots, bounded_client = _fixture()
    capabilities = ModelCapabilities(supports_tools=True, maximum_output_tokens=50)
    snapshots = {
        alias: (
            snapshot
            if alias == policy.embedder_alias
            else snapshot.model_copy(update={"capabilities_sha256": capabilities.identity_sha256()})
        )
        for alias, snapshot in snapshots.items()
    }
    policy = policy.model_copy(
        update={
            "candidates": tuple(
                RoutedCandidateSnapshot(alias=item.alias, model=snapshots[item.alias])
                for item in policy.candidates
            ),
        }
    )

    class _BoundedCatalog(_Catalog):
        """Catalog whose candidate aliases declare an explicit small output limit."""

        def snapshot(self, alias: str) -> tuple[ModelSnapshot, ModelCapabilities]:
            """Return the frozen model and explicitly bounded candidate capabilities."""
            if alias == "embedder":
                return super().snapshot(alias)
            return snapshots[alias], capabilities

        def resolve(self, alias: str, *, role: CatalogRoleName | None = None) -> ResolvedModel:
            """Resolve an alias to the shared test client with bounded capabilities."""
            del role
            snapshot, bounded = self.snapshot(alias)
            return ResolvedModel(alias, snapshot, bounded, self.client, self.client)

    bounded_runtime = RouterRuntime(
        policy,
        manifest,
        bank,
        cast(RuntimeModelCatalog, _BoundedCatalog(snapshots, bounded_client)),
        pricing_snapshot_id="pricing-a",
        pricing_snapshot_sha256="a" * 64,
        pricing_candidate_aliases=bank.candidate_aliases,
    )

    with pytest.raises(RouterModelCapabilityError, match="output-token capacity"):
        bounded_runtime.complete(request, episode_id="bounded-capacity-episode")

    assert bounded_client.complete_calls == 0


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


def test_capability_fallback_replaces_the_sticky_episode_model() -> None:
    """Prove a capability fallback replaces the episode's sticky model.

    An ordinary turn first selects the cheap model, a tool turn falls back to the eligible model,
    and both a later turn and an exact replay of the first request remain on that replacement.
    A fresh episode then dispatches through complete() with the same capability-fallback
    evidence and exactly one client call.
    """
    policy, manifest, bank, snapshots, client = _fixture()
    capabilities = {
        "cheap": ModelCapabilities(supports_tools=False),
        "baseline": ModelCapabilities(supports_tools=True),
        "embedder": ModelCapabilities(supports_embeddings=True, supports_tools=False),
    }
    snapshots = {
        alias: snapshot.model_copy(
            update={"capabilities_sha256": capabilities[alias].identity_sha256()}
        )
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
            """Return the frozen model and capabilities for an alias."""
            return snapshots[alias], capabilities[alias]

        def resolve(self, alias: str, *, role: CatalogRoleName | None = None) -> ResolvedModel:
            del role
            """Resolve an alias to the shared test client and frozen metadata.

            Args:
                alias: Frozen candidate alias to resolve.

            Returns:
                Resolved model with capability-appropriate embedding support.
            """
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
    replayed_first = runtime.select(_request(), episode_id="eligible-sticky")

    assert first.selected_alias == "cheap"
    assert fallback.selected_alias == "baseline"
    assert fallback.fallback_reason == "capability_eligibility"
    assert later.selected_alias == "baseline"
    assert replayed_first.selected_alias == "baseline"

    result = runtime.complete(_request(tool_name="read"), episode_id="eligible")

    assert result.decision.selected_alias == "baseline"
    assert result.decision.fallback_reason == "capability_eligibility"
    assert client.complete_calls == 1
