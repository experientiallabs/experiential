"""Request-time capability eligibility for the frozen router runtime."""

from typing import cast

from wmo.common.models import (
    ModelCapabilities,
    ModelMessage,
    ModelRequest,
    ModelSnapshot,
    RoutedCandidateSnapshot,
)
from wmo.runtime.models import CatalogRoleName, ResolvedModel, RuntimeModelCatalog
from wmo.runtime.router.runtime import RouterRuntime
from wmo.runtime.router.runtime_test import _Catalog, _fixture, _request, _select


def test_capability_fallback_replaces_the_sticky_episode_model() -> None:
    """Prove a capability fallback replaces the episode's sticky model.

    An ordinary turn first selects the cheap model, a tool turn falls back to the eligible model,
    and both a later turn and an exact replay of the first request remain on that replacement.
    A fresh episode then selects with the same capability-fallback evidence.
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

    first = _select(runtime, _request(), episode_id="eligible-sticky")
    fallback = _select(runtime, _request(tool_name="read"), episode_id="eligible-sticky")
    later = _select(
        runtime,
        ModelRequest(messages=(ModelMessage(role="user", content="ordinary later turn"),)),
        episode_id="eligible-sticky",
    )
    replayed_first = _select(runtime, _request(), episode_id="eligible-sticky")

    assert first.selected_alias == "cheap"
    assert fallback.selected_alias == "baseline"
    assert fallback.fallback_reason == "capability_eligibility"
    assert later.selected_alias == "baseline"
    assert replayed_first.selected_alias == "baseline"

    fresh = _select(runtime, _request(tool_name="read"), episode_id="eligible")

    assert fresh.selected_alias == "baseline"
    assert fresh.fallback_reason == "capability_eligibility"
    assert client.complete_calls == 0


def test_unknown_capability_declarations_stay_selectable() -> None:
    """Unknown tool support and output capacity never exclude a selected model.

    Only an explicit ``supports_tools=False`` or an explicitly smaller declared
    output limit rules a candidate out of request-time eligibility.
    """
    policy, manifest, bank, snapshots, client = _fixture()
    runtime = RouterRuntime(
        policy,
        manifest,
        bank,
        cast(RuntimeModelCatalog, _Catalog(snapshots, client)),
        pricing_snapshot_id="pricing-a",
        pricing_snapshot_sha256="a" * 64,
        pricing_candidate_aliases=bank.candidate_aliases,
    )

    request = _request(tool_name="read").model_copy(update={"maximum_output_tokens": 100})
    decision = _select(runtime, request, episode_id="permissive-episode")

    assert decision.selected_alias == "cheap"
    assert decision.fallback_reason is None
