"""Focused restart and provider-idempotency tests for ``RouterRuntime``."""

from __future__ import annotations

from typing import cast

import pytest

from wmo.common.models import ModelRequest, ModelResponse
from wmo.runtime.models import RuntimeModelCatalog
from wmo.runtime.router.runtime import RouterRuntime
from wmo.runtime.router.runtime_test import _Catalog, _Client, _fixture, _request, _runtime

_DIGEST = "a" * 64


class _IdempotentClient(_Client):
    """Completion fake implementing the explicit provider idempotency capability."""

    def __init__(self) -> None:
        super().__init__()
        self.idempotency_keys: list[str] = []

    def complete_idempotent(self, request: ModelRequest, *, idempotency_key: str) -> ModelResponse:
        self.idempotency_keys.append(idempotency_key)
        return self.complete(request)


def test_complete_reinstalls_one_canonical_prior_decision_after_restart() -> None:
    """A verified durable decision can be consumed after the in-memory cache is lost."""
    first, _client = _runtime()
    request = _request(tool_name="read")
    decision = first.select(request, episode_id="episode-a")
    restarted, restarted_client = _runtime()

    routed = restarted.complete(request, episode_id="episode-a", decision=decision)

    assert routed.decision == decision
    assert restarted_client.embed_calls == 0
    assert restarted_client.requests == [request]


def test_provider_idempotency_key_uses_only_the_explicit_capability() -> None:
    """Capable clients receive the key while ordinary model clients keep their existing seam."""
    policy, manifest, bank, snapshots, _client = _fixture()
    capable = _IdempotentClient()
    runtime = RouterRuntime(
        policy,
        manifest,
        bank,
        cast(RuntimeModelCatalog, _Catalog(snapshots, capable)),
        pricing_snapshot_id="pricing-a",
        pricing_snapshot_sha256=_DIGEST,
        pricing_candidate_aliases=bank.candidate_aliases,
    )
    request = _request()
    decision = runtime.select(request, episode_id="episode-a")

    runtime.complete(
        request,
        episode_id="episode-a",
        decision=decision,
        provider_idempotency_key="caller-key-1",
    )

    assert capable.idempotency_keys == ["caller-key-1"]
    with pytest.raises(ValueError, match="visible ASCII"):
        runtime.complete(
            request,
            episode_id="episode-a",
            decision=decision,
            provider_idempotency_key="bad key",
        )
