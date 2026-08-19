"""Adversarial online router runtime and endpoint tests."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from wmo.common.core.artifacts import canonical_json_bytes, sha256_json
from wmo.common.evaluations import (
    EvaluationCell,
    EvaluationDatasetManifest,
    EvaluationPlan,
    EvaluationProtocol,
)
from wmo.common.models import (
    AssistantAction,
    BillingSource,
    CandidateTokenPrice,
    Embedding,
    ModelCapabilities,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
    PricingSnapshot,
    RoutedCandidateSnapshot,
    ToolCall,
    Usage,
)
from wmo.common.project import ArtifactStore, ProjectPaths, artifact_input
from wmo.common.routing import KnnGuard, KnnRouterPolicy
from wmo.common.routing.bank import (
    CandidateEvidenceCount,
    KnnBankManifest,
    KnnEvidenceBank,
    bank_bytes,
)
from wmo.common.routing.features import ROUTER_FEATURE_SCHEMA_SHA256
from wmo.common.tasks import TaskCase, TaskSet
from wmo.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    ProjectTarget,
)
from wmo.runtime.gateway.routing import RouterProjectTargetResolver, gateway_model_request
from wmo.runtime.models import CatalogRoleName, ResolvedModel, RuntimeModelCatalog
from wmo.runtime.models.providers.async_transport import ProviderDeadlineExceeded
from wmo.runtime.router import RouterRuntime, RouterRuntimeIntegrityError
from wmo.runtime.router.economics import (
    RoutedProviderComponent,
    RoutedSpendDisposition,
)
from wmo.runtime.router.runtime_support import sticky_decision

_TIME = datetime(2026, 8, 12, tzinfo=UTC)
_DIGEST = "a" * 64


class _Client:
    """Deterministic embedding and completion client with captured structured requests."""

    def __init__(self) -> None:
        self.embedding_values: tuple[Embedding, ...] | Exception = (Embedding(values=(1.0, 0.0)),)
        self.embed_calls = 0
        self.complete_calls = 0
        self.completion_error: Exception | None = None
        self.requests: list[ModelRequest] = []

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        del texts
        self.embed_calls += 1
        if isinstance(self.embedding_values, Exception):
            raise self.embedding_values
        return self.embedding_values

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.complete_calls += 1
        if self.completion_error is not None:
            error, self.completion_error = self.completion_error, None
            raise error
        self.requests.append(request)
        return ModelResponse(
            output=AssistantAction(
                tool_calls=(ToolCall(call_id="call-out", name="write", arguments={"x": 1}),)
            ),
            model=_snapshot("cheap"),
            economics=OperationEconomics(
                usage=Usage(input_tokens=7, output_tokens=3, cached_input_tokens=2)
            ),
            finish_reason=ModelFinishReason.COMPLETED,
        )


class _Catalog:
    """Exact static identity catalog with no environment or provider access."""

    def __init__(
        self,
        snapshots: dict[str, ModelSnapshot],
        client: _Client,
        *,
        candidate_tools: bool = True,
    ) -> None:
        self.snapshots = snapshots
        self.client = client
        self.candidate_tools = candidate_tools
        self.resolve_calls: list[str] = []

    def snapshot(self, alias: str) -> tuple[ModelSnapshot, ModelCapabilities]:
        return self.snapshots[alias], ModelCapabilities(
            supports_tools=self.candidate_tools and alias != "embedder",
            supports_embeddings=alias == "embedder",
        )

    def resolve(self, alias: str, *, role: CatalogRoleName | None = None) -> ResolvedModel:
        del role
        self.resolve_calls.append(alias)
        snapshot, capabilities = self.snapshot(alias)
        return ResolvedModel(
            alias,
            snapshot,
            capabilities,
            self.client,
            self.client if capabilities.supports_embeddings else None,
        )


@pytest.mark.parametrize(
    "bad",
    [
        RuntimeError("embed failed"),
        (),
        (Embedding(values=(1.0, 0.0)), Embedding(values=(1.0, 0.0))),
        (Embedding(values=(1.0, 0.0, 0.0)),),
        (cast(Embedding, SimpleNamespace(values=(float("nan"), 0.0))),),
        (cast(Embedding, SimpleNamespace(values=(0.0, 0.0))),),
    ],
)
def test_request_embedding_failures_always_fall_back_with_evidence(bad: object) -> None:
    """Count, dimension, NaN, zero-vector, and embed exceptions never escape selection."""
    runtime, client = _runtime()
    client.embedding_values = cast(tuple[Embedding, ...] | Exception, bad)

    decision = runtime.select(_request(), episode_id="episode-a")

    assert decision.selected_alias == runtime.policy.baseline_alias
    assert decision.fallback_reason == "embedding_error"
    assert decision.neighbor_count == decision.paired_count == 0
    operation = runtime.selection_operation(_request(), episode_id="episode-a", decision=decision)
    assert operation.component == RoutedProviderComponent.ROUTER_EMBEDDING
    assert operation.disposition == RoutedSpendDisposition.RESERVED_AMBIGUOUS
    assert operation.operation_count == 1


def test_episode_stickiness_uses_caller_identity_and_tools_affect_request_hash() -> None:
    """An episode stays on its first alias while separate IDs and tool schemas remain distinct."""
    runtime, client = _runtime()
    first = runtime.select(_request(tool_name="read"), episode_id="episode-a")
    client.embedding_values = ()
    next_turn = runtime.select(_request(tool_name="write"), episode_id="episode-a")
    sticky = runtime.select(_request(tool_name="read"), episode_id="episode-a")
    separate = runtime.select(_request(tool_name="read"), episode_id="episode-b")

    assert first.selected_alias == next_turn.selected_alias == sticky.selected_alias == "cheap"
    assert sticky == first
    assert sticky.decision_id == first.decision_id
    assert next_turn.decision_id != first.decision_id
    assert next_turn.request_sha256 != first.request_sha256
    assert separate.selected_alias == "baseline"
    assert separate.episode_id_sha256 != first.episode_id_sha256
    assert client.embed_calls == 2
    assert set(runtime._episode_decisions) == {  # noqa: SLF001 - privacy regression
        hashlib.sha256(b"episode-a").hexdigest(),
        hashlib.sha256(b"episode-b").hexdigest(),
    }
    assert "episode-a" not in first.model_dump_json()


def test_artifact_mutation_and_pricing_or_alias_drift_block_activation() -> None:
    """Artifact integrity rejects while request-time problems conservatively fall back."""
    policy, manifest, bank, snapshots, client = _fixture()
    catalog = cast(RuntimeModelCatalog, _Catalog(snapshots, client))
    with pytest.raises(RouterRuntimeIntegrityError, match="pricing snapshot"):
        RouterRuntime(
            policy,
            manifest,
            bank,
            catalog,
            pricing_snapshot_id="pricing-b",
            pricing_snapshot_sha256=_DIGEST,
            pricing_candidate_aliases=bank.candidate_aliases,
        )
    drifted = dict(snapshots)
    drifted["cheap"] = snapshots["cheap"].model_copy(update={"connection_sha256": "f" * 64})
    with pytest.raises(RouterRuntimeIntegrityError, match="cannot resolve policy pins"):
        RouterRuntime(
            policy,
            manifest,
            bank,
            cast(RuntimeModelCatalog, _Catalog(drifted, client)),
            pricing_snapshot_id="pricing-a",
            pricing_snapshot_sha256=_DIGEST,
            pricing_candidate_aliases=bank.candidate_aliases,
        )
    runtime = RouterRuntime(
        policy,
        manifest,
        bank,
        catalog,
        pricing_snapshot_id="pricing-a",
        pricing_snapshot_sha256=_DIGEST,
        pricing_candidate_aliases=bank.candidate_aliases,
    )
    bank.scores.setflags(write=True)
    bank.scores[0, 0] = 0.0
    assert runtime.select(_request(), episode_id="episode-a").selected_alias == "cheap"
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        runtime.bank.scores.setflags(write=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("bank_artifact_id", "bank-b", "bank artifact"),
        ("bank_sha256", "f" * 64, "bank digest"),
        ("fit_evaluation_id", "fit-b", "fit evaluation"),
        ("evaluation_plan_id", "plan-b", "evaluation plan"),
        ("evaluation_plan_sha256", "f" * 64, "evaluation plan digest"),
        ("task_set_id", "tasks-b", "task set"),
        ("task_set_sha256", "f" * 64, "task-set digest"),
        ("evaluation_protocols_sha256", "f" * 64, "evaluation protocol scope"),
        ("embedder_alias", "embedder-b", "embedder alias"),
        ("feature_extractor_id", "request-visible-v3", "feature extractor"),
        ("feature_schema_sha256", "f" * 64, "feature schema"),
        ("pricing_snapshot_id", "pricing-b", "pricing snapshot"),
        ("pricing_snapshot_sha256", "f" * 64, "pricing snapshot digest"),
        ("candidate_aliases", ("cheap", "baseline"), "candidate aliases"),
        ("task_ids", tuple(reversed(tuple(f"task-{index}" for index in range(8)))), "bank rows"),
    ],
)
def test_final_fit_pins_reject_activation_before_embedding(
    field: str, value: object, message: str
) -> None:
    """Every final W10 policy/bank identity mismatch blocks runtime activation."""
    policy, manifest, bank, snapshots, client = _fixture()
    drifted = manifest.model_copy(update={field: value})

    with pytest.raises(RouterRuntimeIntegrityError, match=message):
        RouterRuntime(
            policy,
            drifted,
            bank,
            cast(RuntimeModelCatalog, _Catalog(snapshots, client)),
            pricing_snapshot_id="pricing-a",
            pricing_snapshot_sha256=_DIGEST,
            pricing_candidate_aliases=bank.candidate_aliases,
        )
    assert client.embed_calls == 0
    assert client.requests == []


def test_pricing_digest_and_candidate_order_reject_before_embedding() -> None:
    """Runtime pricing must match the policy manifest and exact ordered candidate scope."""
    policy, manifest, bank, snapshots, client = _fixture()
    catalog = cast(RuntimeModelCatalog, _Catalog(snapshots, client))
    with pytest.raises(RouterRuntimeIntegrityError, match="pricing manifest"):
        RouterRuntime(
            policy,
            manifest,
            bank,
            catalog,
            pricing_snapshot_id="pricing-a",
            pricing_snapshot_sha256="f" * 64,
            pricing_candidate_aliases=bank.candidate_aliases,
        )
    with pytest.raises(RouterRuntimeIntegrityError, match="pricing candidates"):
        RouterRuntime(
            policy,
            manifest,
            bank,
            catalog,
            pricing_snapshot_id="pricing-a",
            pricing_snapshot_sha256=_DIGEST,
            pricing_candidate_aliases=tuple(reversed(bank.candidate_aliases)),
        )
    assert client.embed_calls == 0


def test_decision_integrity_error_is_not_converted_to_embedding_fallback() -> None:
    """A post-activation policy mutation remains fatal instead of silently selecting baseline."""
    runtime, client = _runtime()
    object.__setattr__(runtime.policy.guard, "uncertainty_multiplier", 0.0)

    with pytest.raises(ValueError, match="finite and positive"):
        runtime.select(_request(), episode_id="episode-a")
    assert client.embed_calls == 1


def test_store_backed_load_verifies_artifacts_and_normalizes_failures(tmp_path: Path) -> None:
    """The public loader composes exact pricing, bank, policy, and catalog artifacts."""
    store, policy, catalog, client = _persist_runtime_fixture(tmp_path)

    runtime = RouterRuntime.load(
        store,
        policy.policy_id,
        catalog,
        pricing_snapshot_id=policy.pricing_snapshot_id,
    )
    assert runtime.select(_request(), episode_id="episode-a").selected_alias == "cheap"
    with pytest.raises(RouterRuntimeIntegrityError, match="router policy missing-policy"):
        RouterRuntime.load(
            store,
            "missing-policy",
            catalog,
            pricing_snapshot_id=policy.pricing_snapshot_id,
        )
    with pytest.raises(RouterRuntimeIntegrityError, match="router policy pricing-a"):
        RouterRuntime.load(
            store,
            "pricing-a",
            catalog,
            pricing_snapshot_id=policy.pricing_snapshot_id,
        )
    assert client.embed_calls == 1


@pytest.mark.parametrize(
    "seal_mutation",
    [
        "omit-policy:fit-a",
        "omit-policy:bank-a",
        "omit-bank:fit-a",
        "omit-bank:plan-a",
        "omit-bank:tasks-a",
        "omit-bank:pricing-a",
        "wrong-bank:plan-a",
        "extra-bank",
    ],
)
def test_store_load_requires_exact_canonical_artifact_input_chain(
    tmp_path: Path, seal_mutation: str
) -> None:
    """Missing, extra, or wrong-digest W10 inputs reject before model resolution."""
    store, policy, catalog, client = _persist_runtime_fixture(tmp_path, seal_mutation=seal_mutation)
    concrete = cast(_Catalog, catalog)

    with pytest.raises(RouterRuntimeIntegrityError, match="router policy"):
        RouterRuntime.load(
            store,
            policy.policy_id,
            catalog,
            pricing_snapshot_id=policy.pricing_snapshot_id,
        )
    assert concrete.resolve_calls == []
    assert client.embed_calls == 0


@pytest.mark.parametrize("drift_alias", ["baseline", "embedder"])
def test_resolved_catalog_identity_cannot_diverge_from_snapshot_pin(drift_alias: str) -> None:
    """Activation verifies the actual resolved client identity, not a prior snapshot lookup."""
    policy, manifest, bank, snapshots, client = _fixture()

    class _SplitCatalog(_Catalog):
        def resolve(self, alias: str, *, role: CatalogRoleName | None = None) -> ResolvedModel:
            del role
            resolved = super().resolve(alias)
            if alias == drift_alias:
                return ResolvedModel(
                    alias,
                    resolved.snapshot.model_copy(update={"connection_sha256": "f" * 64}),
                    resolved.capabilities,
                    resolved.client,
                    resolved.embedding_client,
                )
            return resolved

    with pytest.raises(RouterRuntimeIntegrityError, match="cannot resolve policy pins"):
        RouterRuntime(
            policy,
            manifest,
            bank,
            cast(RuntimeModelCatalog, _SplitCatalog(snapshots, client)),
            pricing_snapshot_id=policy.pricing_snapshot_id,
            pricing_snapshot_sha256=policy.pricing_snapshot_sha256,
            pricing_candidate_aliases=manifest.candidate_aliases,
        )
    assert client.embed_calls == 0


def test_catalog_resolution_failure_is_normalized_to_integrity_error() -> None:
    """Catalog lookup failures never leak a provider-specific activation exception."""
    policy, manifest, bank, snapshots, client = _fixture()

    class _BrokenCatalog(_Catalog):
        def resolve(self, alias: str, *, role: CatalogRoleName | None = None) -> ResolvedModel:
            del role
            raise RuntimeError(f"cannot resolve {alias}")

    with pytest.raises(RouterRuntimeIntegrityError, match="cannot resolve policy pins"):
        RouterRuntime(
            policy,
            manifest,
            bank,
            cast(RuntimeModelCatalog, _BrokenCatalog(snapshots, client)),
            pricing_snapshot_id=policy.pricing_snapshot_id,
            pricing_snapshot_sha256=policy.pricing_snapshot_sha256,
            pricing_candidate_aliases=manifest.candidate_aliases,
        )


def test_embedder_candidate_alias_overlap_requires_the_same_frozen_identity() -> None:
    """An embedder alias cannot hide a differently pinned routed-candidate snapshot.

    The regression rejects activation before the runtime can dispatch any provider call.
    """
    policy, manifest, bank, snapshots, client = _fixture()
    different = policy.model_copy(
        update={
            "embedder_alias": "baseline",
            "embedder": snapshots["embedder"],
        }
    )
    with pytest.raises(RouterRuntimeIntegrityError, match="embedder alias overlaps"):
        RouterRuntime(
            different,
            manifest.model_copy(
                update={"embedder_alias": "baseline", "embedder": snapshots["embedder"]}
            ),
            bank,
            cast(RuntimeModelCatalog, _Catalog(snapshots, client)),
            pricing_snapshot_id=policy.pricing_snapshot_id,
            pricing_snapshot_sha256=policy.pricing_snapshot_sha256,
            pricing_candidate_aliases=manifest.candidate_aliases,
        )
    assert client.embed_calls == 0

    embedding_capabilities = ModelCapabilities(supports_tools=True, supports_embeddings=True)
    overlapping_snapshot = snapshots["baseline"].model_copy(
        update={"capabilities_sha256": embedding_capabilities.identity_sha256()}
    )
    overlapping_candidates = tuple(
        candidate.model_copy(update={"model": overlapping_snapshot})
        if candidate.alias == "baseline"
        else candidate
        for candidate in policy.candidates
    )
    same = policy.model_copy(
        update={
            "candidates": overlapping_candidates,
            "embedder_alias": "baseline",
            "embedder": overlapping_snapshot,
        }
    )
    same_manifest = manifest.model_copy(
        update={
            "embedder_alias": "baseline",
            "embedder": overlapping_snapshot,
        }
    )

    class _OverlapCatalog(_Catalog):
        def snapshot(self, alias: str) -> tuple[ModelSnapshot, ModelCapabilities]:
            snapshot = self.snapshots[alias]
            return snapshot, ModelCapabilities(
                supports_tools=alias in {"baseline", "cheap"},
                supports_embeddings=alias == "baseline",
            )

    runtime = RouterRuntime(
        same,
        same_manifest,
        bank,
        cast(
            RuntimeModelCatalog,
            _OverlapCatalog({**snapshots, "baseline": overlapping_snapshot}, client),
        ),
        pricing_snapshot_id=policy.pricing_snapshot_id,
        pricing_snapshot_sha256=policy.pricing_snapshot_sha256,
        pricing_candidate_aliases=manifest.candidate_aliases,
    )
    assert runtime.policy.embedder == runtime.policy.candidates[0].model


def test_decision_sink_failure_never_publishes_an_unrecorded_cache_entry() -> None:
    """A recorder failure retries selection and recording instead of bypassing evidence."""
    policy, manifest, bank, snapshots, client = _fixture()
    attempts = 0
    recorded = []

    def sink(decision: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("sink unavailable")
        recorded.append(decision)

    runtime = RouterRuntime(
        policy,
        manifest,
        bank,
        cast(RuntimeModelCatalog, _Catalog(snapshots, client)),
        pricing_snapshot_id=policy.pricing_snapshot_id,
        pricing_snapshot_sha256=policy.pricing_snapshot_sha256,
        pricing_candidate_aliases=manifest.candidate_aliases,
        decision_sink=sink,
    )
    with pytest.raises(RuntimeError, match="sink unavailable"):
        runtime.select(_request(), episode_id="episode-a")
    decision = runtime.select(_request(), episode_id="episode-a")

    assert attempts == 2
    assert recorded == [decision]
    assert client.embed_calls == 2


def test_concurrent_first_selection_embeds_and_records_exactly_once() -> None:
    """Two simultaneous first requests share one deterministic cached episode decision."""
    policy, manifest, bank, snapshots, client = _fixture()
    recorded = []
    runtime = RouterRuntime(
        policy,
        manifest,
        bank,
        cast(RuntimeModelCatalog, _Catalog(snapshots, client)),
        pricing_snapshot_id=policy.pricing_snapshot_id,
        pricing_snapshot_sha256=policy.pricing_snapshot_sha256,
        pricing_candidate_aliases=manifest.candidate_aliases,
        decision_sink=recorded.append,
    )
    start = threading.Barrier(3)
    decisions: list[object] = []

    def select() -> None:
        start.wait()
        decisions.append(runtime.select(_request(), episode_id="episode-a"))

    threads = (threading.Thread(target=select), threading.Thread(target=select))
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()

    assert len(decisions) == 2
    assert decisions[0] == decisions[1]
    assert client.embed_calls == 1
    assert recorded == [decisions[0]]


def test_distinct_first_selections_embed_outside_the_shared_cache_lock() -> None:
    """Independent episodes can embed concurrently without serializing on cache state."""
    runtime, client = _runtime()
    entered = threading.Barrier(3)
    decisions: list[object] = []

    def concurrent_embed(texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Wait until both embedding calls are concurrently in flight."""
        del texts
        client.embed_calls += 1
        entered.wait(timeout=1)
        return (Embedding(values=(1.0, 0.0)),)

    client.__dict__["embed"] = concurrent_embed

    def select(episode_id: str) -> None:
        """Select one independent episode on a worker thread."""
        decisions.append(runtime.select(_request(), episode_id=episode_id))

    threads = (
        threading.Thread(target=select, args=("episode-a",)),
        threading.Thread(target=select, args=("episode-b",)),
    )
    for thread in threads:
        thread.start()
    entered.wait(timeout=1)
    for thread in threads:
        thread.join(timeout=1)

    assert all(not thread.is_alive() for thread in threads)
    assert len(decisions) == 2
    assert client.embed_calls == 2


def test_decision_cache_ttl_and_capacity_bound_all_process_local_state() -> None:
    """Least-recent decisions evict at capacity and expired episodes reselect."""
    policy, manifest, bank, snapshots, client = _fixture()
    now = [10.0]
    runtime = RouterRuntime(
        policy,
        manifest,
        bank,
        cast(RuntimeModelCatalog, _Catalog(snapshots, client)),
        pricing_snapshot_id=policy.pricing_snapshot_id,
        pricing_snapshot_sha256=policy.pricing_snapshot_sha256,
        pricing_candidate_aliases=manifest.candidate_aliases,
        decision_capacity=1,
        decision_ttl_seconds=5,
        clock=lambda: now[0],
    )

    runtime.select(_request(), episode_id="episode-a")
    runtime.select(_request(), episode_id="episode-b")
    assert len(runtime._episode_decisions) == 1  # noqa: SLF001 - bounded-state regression
    assert len(runtime._request_decisions) == 1  # noqa: SLF001 - bounded-state regression
    assert len(runtime._request_embedding_economics) == 1  # noqa: SLF001
    assert len(runtime._request_embedding_dispositions) == 1  # noqa: SLF001
    now[0] = 16.0
    runtime.select(_request(), episode_id="episode-b")

    assert client.embed_calls == 3
    assert len(runtime._episode_decisions) == 1  # noqa: SLF001 - bounded-state regression
    assert len(runtime._request_decisions) == 1  # noqa: SLF001 - bounded-state regression
    assert set(runtime._request_embedding_economics) == set(  # noqa: SLF001
        runtime._request_decisions  # noqa: SLF001
    )
    assert set(runtime._request_embedding_dispositions) == set(  # noqa: SLF001
        runtime._request_decisions  # noqa: SLF001
    )


def test_prepared_selections_keep_physical_evidence_local_until_atomic_retain() -> None:
    """Concurrent-equivalent bundles cannot collide or publish before explicit retain."""
    runtime, client = _runtime()
    request = _request()
    first = runtime._select_unretained(request, episode_id="episode-a")  # noqa: SLF001
    client.embedding_values = RuntimeError("embed failed")
    second = runtime._select_unretained(request, episode_id="episode-a")  # noqa: SLF001

    assert first is not second
    assert first.disposition == RoutedSpendDisposition.LOCALLY_PRICED
    assert second.disposition == RoutedSpendDisposition.RESERVED_AMBIGUOUS
    assert runtime._episode_decisions == {}  # noqa: SLF001 - publication boundary
    assert runtime._request_decisions == {}  # noqa: SLF001 - publication boundary

    retained = runtime._retain_prepared_selection(  # noqa: SLF001
        request,
        episode_id="episode-a",
        prepared=first,
    )
    first_operation = runtime.selection_operation(
        request,
        episode_id="episode-a",
        decision=retained,
    )
    reconciled = runtime._retain_prepared_selection(  # noqa: SLF001
        request,
        episode_id="episode-a",
        prepared=second,
    )
    second_operation = runtime.selection_operation(
        request,
        episode_id="episode-a",
        decision=reconciled,
    )

    assert reconciled == retained
    assert first_operation.disposition == RoutedSpendDisposition.LOCALLY_PRICED
    assert second_operation.disposition == RoutedSpendDisposition.RESERVED_AMBIGUOUS
    assert client.embed_calls == 2


def test_project_resolver_retains_failed_embedding_evidence_without_reembedding() -> None:
    """A failed physical embed binds ambiguous accounting through gateway selection."""
    runtime, client = _runtime()
    client.embedding_values = RuntimeError("embed failed")
    target = ProjectTarget(
        project_ref="project-one",
        activation_ref="activation-one",
        catalog_sha256=_DIGEST,
    )
    resolver = RouterProjectTargetResolver(
        {("project-one", "activation-one"): runtime},
        {("project-one", "activation-one", _DIGEST, "baseline"): "exact-baseline"},
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="route"),),
    )
    namespace = ("org", "identity", "revision", "episode")

    selection = asyncio.run(
        resolver.select(
            target=target,
            request=request,
            episode_namespace=namespace,
            deadline_monotonic=__import__("time").monotonic() + 1,
        )
    )

    assert selection.selected_alias == "baseline"
    assert client.embed_calls == 1
    assert len(runtime._request_decisions) == 1  # noqa: SLF001 - accounting regression
    decision = next(iter(runtime._request_decisions.values()))  # noqa: SLF001
    operation = runtime.selection_operation(
        gateway_model_request(request),
        episode_id="\x1f".join(namespace),
        decision=decision,
    )
    assert operation.disposition == RoutedSpendDisposition.RESERVED_AMBIGUOUS
    assert operation.operation_count == 1


def test_timed_out_project_selection_cannot_publish_late_sticky_state() -> None:
    """A detached blocking embed remains unretained after its request deadline expires."""
    runtime, client = _runtime()
    recorded: list[object] = []
    runtime._decision_sink = recorded.append  # noqa: SLF001 - deadline publication regression
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    def blocking_embed(texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Hold selection beyond the gateway deadline, then report completion."""
        del texts
        entered.set()
        release.wait(timeout=1)
        completed.set()
        return (Embedding(values=(1.0, 0.0)),)

    client.__dict__["embed"] = blocking_embed
    target = ProjectTarget(
        project_ref="project-one",
        activation_ref="activation-one",
        catalog_sha256=_DIGEST,
    )
    resolver = RouterProjectTargetResolver(
        {("project-one", "activation-one"): runtime},
        {("project-one", "activation-one", _DIGEST, "cheap"): "exact-cheap"},
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="route"),),
    )

    async def scenario() -> None:
        """Expire the wrapper while the unretained worker remains blocked."""
        selection = asyncio.create_task(
            resolver.select(
                target=target,
                request=request,
                episode_namespace=("org", "identity", "revision", "episode"),
                deadline_monotonic=__import__("time").monotonic() + 0.01,
            )
        )
        await asyncio.to_thread(entered.wait, 1)
        with pytest.raises(ProviderDeadlineExceeded):
            await selection
        release.set()
        await asyncio.to_thread(completed.wait, 1)
        await asyncio.sleep(0)

    asyncio.run(scenario())

    assert runtime._episode_decisions == {}  # noqa: SLF001 - deadline isolation regression
    assert runtime._request_decisions == {}  # noqa: SLF001 - deadline isolation regression
    assert recorded == []


def test_cached_selection_and_completion_reuse_sealed_activation_without_rehashing() -> None:
    """Forged decisions reject while exact retries reuse immutable activation state."""
    runtime, client = _runtime()
    request = _request()
    decision = runtime.select(request, episode_id="episode-a")
    forged = decision.model_copy(update={"decision_id": "forged-decision"})
    with pytest.raises(ValueError, match="exact cached"):
        runtime.complete(request, episode_id="episode-a", decision=forged)
    assert client.requests == []

    runtime.complete(request, episode_id="episode-a", decision=decision)
    runtime.complete(request, episode_id="episode-a", decision=decision)
    assert len(client.requests) == 2

    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        runtime.bank.scores.setflags(write=True)
    assert runtime.select(request, episode_id="episode-a") == decision


def test_sticky_decision_identity_hashes_all_retained_evidence() -> None:
    """Unequal first-turn evidence cannot collide for the same later request and episode."""
    runtime, _client = _runtime()
    first = runtime.select(_request(tool_name="read"), episode_id="episode-a")
    changed = first.model_copy(
        update={
            "neighbor_count": first.neighbor_count + 1,
            "paired_count": first.paired_count + 1,
            "best_similarity": 0.5,
        }
    )
    request_sha256 = "f" * 64

    sticky = sticky_decision(first, request_sha256)
    changed_sticky = sticky_decision(changed, request_sha256)

    assert sticky != changed_sticky
    assert sticky.decision_id != changed_sticky.decision_id


def test_complete_validates_one_prior_decision_without_selecting_again() -> None:
    """Python completion validates and reuses the exact cached supplied decision."""
    policy, manifest, bank, snapshots, client = _fixture()
    recorded = []
    runtime = RouterRuntime(
        policy,
        manifest,
        bank,
        cast(RuntimeModelCatalog, _Catalog(snapshots, client)),
        pricing_snapshot_id="pricing-a",
        pricing_snapshot_sha256=_DIGEST,
        pricing_candidate_aliases=bank.candidate_aliases,
        decision_sink=recorded.append,
    )
    request = _request(tool_name="read")
    decision = runtime.select(request, episode_id="episode-a")
    routed = runtime.complete(request, episode_id="episode-a", decision=decision)

    assert routed.decision is decision
    assert len(recorded) == 1
    assert client.requests == [request]
    with pytest.raises(ValueError, match="does not match"):
        runtime.complete(_request(tool_name="write"), episode_id="episode-a", decision=decision)


def test_routed_completion_reconciles_alias_free_embedding_and_candidate_economics() -> None:
    """Return typed request economics without placing the selected private alias in that record."""
    policy, manifest, bank, snapshots, client = _fixture()
    embedding_capabilities = ModelCapabilities(
        supports_embeddings=True,
        input_cost_per_million_tokens_usd=0.1,
    )
    embedder = snapshots["embedder"].model_copy(
        update={
            "billing_source": BillingSource.HOST_MANAGED,
            "capabilities_sha256": embedding_capabilities.identity_sha256(),
        }
    )
    policy = policy.model_copy(update={"embedder": embedder})
    manifest = manifest.model_copy(update={"embedder": embedder})

    class _PricedCatalog(_Catalog):
        def snapshot(self, alias: str) -> tuple[ModelSnapshot, ModelCapabilities]:
            if alias == "embedder":
                return embedder, embedding_capabilities
            return super().snapshot(alias)

    prices = tuple(
        CandidateTokenPrice(
            candidate_alias=alias,
            input_usd_per_million_tokens=1.0,
            output_usd_per_million_tokens=2.0,
            cached_input_usd_per_million_tokens=0.5,
            cache_write_usd_per_million_tokens=1.5,
        )
        for alias in bank.candidate_aliases
    )
    runtime = RouterRuntime(
        policy,
        manifest,
        bank,
        cast(RuntimeModelCatalog, _PricedCatalog(snapshots, client)),
        pricing_snapshot_id="pricing-a",
        pricing_snapshot_sha256=_DIGEST,
        pricing_candidate_aliases=bank.candidate_aliases,
        pricing_candidate_prices=prices,
    )
    request = _request(tool_name="read")
    decision = runtime.select(request, episode_id="episode-a")
    routed = runtime.complete(request, episode_id="episode-a", decision=decision)

    economics = routed.economics
    embedding_economics = economics.router_embedding.economics
    candidate_economics = economics.selected_candidate.economics
    assert economics.router_embedding.billing_source == BillingSource.HOST_MANAGED
    assert economics.selected_candidate.billing_source == BillingSource.CUSTOMER_MANAGED
    assert economics.operation_count == 2
    assert [item.disposition for item in economics.operations] == [
        RoutedSpendDisposition.LOCALLY_PRICED,
        RoutedSpendDisposition.LOCALLY_PRICED,
    ]
    assert embedding_economics.usage is not None
    assert embedding_economics.usage.input_tokens > 0
    assert embedding_economics.cost_usd is not None
    assert candidate_economics.usage == Usage(
        input_tokens=7,
        output_tokens=3,
        cached_input_tokens=2,
    )
    assert candidate_economics.cost_usd is not None
    assert candidate_economics.cost_usd.value == pytest.approx(14.5 / 1_000_000)
    assert tuple(item.billing_source for item in economics.by_billing_source) == (
        BillingSource.CUSTOMER_MANAGED,
        BillingSource.HOST_MANAGED,
    )
    assert economics.total.cost_usd is not None
    assert economics.total.cost_usd.value == pytest.approx(
        embedding_economics.cost_usd.value + candidate_economics.cost_usd.value
    )
    serialized = economics.model_dump_json()
    assert decision.selected_alias not in serialized
    assert "model_id" not in serialized


def _request(*, tool_name: str | None = None) -> ModelRequest:
    from wmo.common.tasks import ToolSchema

    tools = (
        (ToolSchema(name=tool_name, description="fixture", input_schema={"type": "object"}),)
        if tool_name is not None
        else ()
    )
    return ModelRequest(messages=(ModelMessage(role="user", content="route me"),), tools=tools)


def _runtime(*, candidate_tools: bool = True) -> tuple[RouterRuntime, _Client]:
    policy, manifest, bank, snapshots, client = _fixture(candidate_tools=candidate_tools)
    return (
        RouterRuntime(
            policy,
            manifest,
            bank,
            cast(
                RuntimeModelCatalog,
                _Catalog(snapshots, client, candidate_tools=candidate_tools),
            ),
            pricing_snapshot_id="pricing-a",
            pricing_snapshot_sha256=_DIGEST,
            pricing_candidate_aliases=bank.candidate_aliases,
        ),
        client,
    )


def _fixture(
    *, candidate_tools: bool = True
) -> tuple[
    KnnRouterPolicy,
    KnnBankManifest,
    KnnEvidenceBank,
    dict[str, ModelSnapshot],
    _Client,
]:
    snapshots = {
        alias: _snapshot(alias, candidate_tools=candidate_tools)
        for alias in ("baseline", "cheap", "embedder")
    }
    client = _Client()
    bank = KnnEvidenceBank(
        task_ids=tuple(f"task-{index}" for index in range(8)),
        candidate_aliases=("baseline", "cheap"),
        embeddings=np.asarray(((1.0, 0.0),) * 8, dtype=np.float32),
        scores=np.asarray(((0.4, 1.0),) * 8, dtype=np.float32),
        candidate_costs=np.asarray(((0.5, 0.1),) * 8, dtype=np.float64),
        score_counts=np.ones((8, 2), dtype=np.int32),
        cost_counts=np.ones((8, 2), dtype=np.int32),
        workload_weights=np.ones(8, dtype=np.float64),
        novelty_floor=0.5,
    )
    digest = hashlib.sha256(bank_bytes(bank)).hexdigest()
    manifest = KnnBankManifest(
        schema_version=1,
        created_at=_TIME,
        code_revision="test",
        bank_artifact_id="bank-a",
        fit_evaluation_id="fit-a",
        evaluation_plan_id="plan-a",
        evaluation_plan_sha256=_DIGEST,
        task_set_id="tasks-a",
        task_set_sha256=_DIGEST,
        task_ids=bank.task_ids,
        candidate_aliases=bank.candidate_aliases,
        evaluation_protocols_sha256=_DIGEST,
        embedder_alias="embedder",
        embedder=snapshots["embedder"],
        feature_extractor_id="request-visible-v2",
        feature_schema_sha256=ROUTER_FEATURE_SCHEMA_SHA256,
        pricing_snapshot_id="pricing-a",
        pricing_snapshot_sha256=_DIGEST,
        bank_sha256=digest,
        embedding_dimension=2,
        novelty_floor=0.5,
        evidence_counts=tuple(
            CandidateEvidenceCount(candidate_alias=alias, scored_task_count=8, costed_task_count=8)
            for alias in bank.candidate_aliases
        ),
    )
    policy = KnnRouterPolicy(
        schema_version=1,
        created_at=_TIME,
        code_revision="test",
        policy_id="policy-a",
        baseline_alias="baseline",
        candidates=tuple(
            RoutedCandidateSnapshot(alias=alias, model=snapshots[alias])
            for alias in bank.candidate_aliases
        ),
        embedder_alias="embedder",
        embedder=snapshots["embedder"],
        feature_extractor_id=manifest.feature_extractor_id,
        feature_schema_sha256=manifest.feature_schema_sha256,
        pricing_snapshot_id="pricing-a",
        pricing_snapshot_sha256=_DIGEST,
        bank_artifact_id="bank-a",
        bank_sha256=digest,
        guard=KnnGuard(
            maximum_neighbors=8,
            minimum_paired_observations=8,
            relative_similarity_threshold=0.95,
            uncertainty_multiplier=0.5,
            quality_tolerance=0,
        ),
        fit_evaluation_id="fit-a",
        evaluation_plan_id=manifest.evaluation_plan_id,
        evaluation_plan_sha256=manifest.evaluation_plan_sha256,
        task_set_id=manifest.task_set_id,
        task_set_sha256=manifest.task_set_sha256,
        evaluation_protocols_sha256=manifest.evaluation_protocols_sha256,
        judgment_status="provisional",
    )
    return policy, manifest, bank, snapshots, client


def _snapshot(alias: str, *, candidate_tools: bool = True) -> ModelSnapshot:
    """Create one deterministic frozen model snapshot for router fixtures.

    Args:
        alias: Stable local model alias.
        candidate_tools: Whether non-embedder aliases advertise tool support.

    Returns:
        Fixture snapshot using the runtime identity digest contract.
    """
    capabilities = ModelCapabilities(
        supports_tools=candidate_tools and alias != "embedder",
        supports_embeddings=alias == "embedder",
    )
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="test",
        model_id=alias,
        revision="fixture",
        capabilities_sha256=capabilities.identity_sha256(),
        connection_sha256="b" * 64,
    )


def _persist_runtime_fixture(
    root: Path,
    *,
    seal_mutation: str | None = None,
) -> tuple[ArtifactStore, KnnRouterPolicy, RuntimeModelCatalog, _Client]:
    """Persist the exact policy dependency chain consumed by ``RouterRuntime.load``."""
    policy, manifest, bank, snapshots, client = _fixture()
    store = ArtifactStore(ProjectPaths(root=root, project_id="project-a"))
    pricing = PricingSnapshot(
        schema_version=1,
        created_at=_TIME,
        code_revision="test",
        pricing_snapshot_id="pricing-a",
        candidate_prices=tuple(
            CandidateTokenPrice(
                candidate_alias=alias,
                input_usd_per_million_tokens=1.0,
                output_usd_per_million_tokens=2.0,
            )
            for alias in bank.candidate_aliases
        ),
    )
    pricing_manifest = store.write_json(
        artifact_id=pricing.pricing_snapshot_id,
        artifact_type="pricing-snapshot",
        envelope=pricing,
        files={"pricing.json": pricing},
    )
    pricing_input = artifact_input(pricing_manifest)
    tasks = tuple(
        TaskCase(
            task_id=task_id,
            lineage_group_id=f"lineage-{index}",
            partition="fit",
            instruction=f"route task {index}",
            workload_weight=1.0,
            source_trace_ids=(f"trace-{index}",),
        )
        for index, task_id in enumerate(bank.task_ids)
    )
    task_payload = b"\n".join(canonical_json_bytes(task) for task in tasks) + b"\n"
    task_set = TaskSet(
        schema_version=1,
        created_at=_TIME,
        code_revision="test",
        task_set_id="tasks-a",
        task_ids=bank.task_ids,
        tasks_path="tasks.jsonl",
        tasks_sha256=hashlib.sha256(task_payload).hexdigest(),
    )
    task_manifest = store.write(
        artifact_id=task_set.task_set_id,
        artifact_type="task-set",
        envelope=task_set,
        files={"task-set.json": canonical_json_bytes(task_set), "tasks.jsonl": task_payload},
    )
    task_input = artifact_input(task_manifest)
    plan = EvaluationPlan(
        schema_version=2,
        created_at=_TIME,
        inputs=tuple(
            sorted(
                (pricing_input, task_input),
                key=lambda item: item.artifact_id,
            )
        ),
        code_revision="test",
        plan_id="plan-a",
        task_set_id=task_set.task_set_id,
        candidate_snapshots=policy.candidates,
        pricing_snapshot_id=pricing.pricing_snapshot_id,
        pricing_snapshot_sha256=pricing_input.sha256,
        cells=tuple(
            EvaluationCell(
                cell_id=f"cell-{task.task_id}-{alias}",
                task_id=task.task_id,
                candidate_alias=alias,
                repeat=0,
                purpose="fit",
                execution="observed",
                observed_rollout_id=f"rollout-{task.task_id}-{alias}",
            )
            for task in tasks
            for alias in bank.candidate_aliases
        ),
    )
    plan_manifest = store.write_json(
        artifact_id=plan.plan_id,
        artifact_type="evaluation-plan",
        envelope=plan,
        files={"plan.json": plan},
    )
    plan_input = artifact_input(plan_manifest)
    protocol = EvaluationProtocol(
        protocol_id="protocol-a",
        evidence_source="production",
        agent_id="agent-a",
        simulator_id="production-a",
        rubric_id="rubric-a",
        judge_calibration_id="calibration-a",
        pricing_snapshot_id=pricing.pricing_snapshot_id,
    )
    evaluation_inputs = tuple(
        sorted(
            (plan_input, pricing_input, task_input),
            key=lambda item: item.artifact_id,
        )
    )
    rows_payload = b""
    evaluation_manifest = EvaluationDatasetManifest(
        schema_version=1,
        created_at=_TIME,
        inputs=evaluation_inputs,
        code_revision="test",
        evaluation_id="fit-a",
        evaluation_plan_id=plan.plan_id,
        evaluation_plan_sha256=plan_input.sha256,
        task_set_id=task_set.task_set_id,
        fit_task_ids=bank.task_ids,
        held_out_task_ids=(),
        candidate_snapshots=policy.candidates,
        protocols=(protocol,),
        rows_path="rows.jsonl",
        rows_sha256=hashlib.sha256(rows_payload).hexdigest(),
    )
    evaluation_record = store.write(
        artifact_id=evaluation_manifest.evaluation_id,
        artifact_type="evaluation",
        envelope=evaluation_manifest,
        files={
            "evaluation.json": canonical_json_bytes(evaluation_manifest),
            "rows.jsonl": rows_payload,
        },
    )
    evaluation_input = artifact_input(evaluation_record)
    protocol_scope = sha256_json([protocol.model_dump(mode="json")])
    bank_inputs = tuple(
        sorted(
            (evaluation_input, plan_input, pricing_input, task_input),
            key=lambda item: item.artifact_id,
        )
    )
    if seal_mutation and seal_mutation.startswith("omit-bank:"):
        omitted = seal_mutation.partition(":")[2]
        bank_inputs = tuple(item for item in bank_inputs if item.artifact_id != omitted)
    elif seal_mutation and seal_mutation.startswith("wrong-bank:"):
        changed = seal_mutation.partition(":")[2]
        bank_inputs = tuple(
            item.model_copy(update={"sha256": "f" * 64}) if item.artifact_id == changed else item
            for item in bank_inputs
        )
    elif seal_mutation == "extra-bank":
        extra = PricingSnapshot(
            schema_version=1,
            created_at=_TIME,
            code_revision="test",
            pricing_snapshot_id="pricing-extra",
            candidate_prices=pricing.candidate_prices,
        )
        extra_record = store.write_json(
            artifact_id=extra.pricing_snapshot_id,
            artifact_type="pricing-snapshot",
            envelope=extra,
            files={"pricing.json": extra},
        )
        bank_inputs = tuple(
            sorted((*bank_inputs, artifact_input(extra_record)), key=lambda item: item.artifact_id)
        )
    manifest = manifest.model_copy(
        update={
            "inputs": bank_inputs,
            "evaluation_plan_sha256": plan_input.sha256,
            "task_set_sha256": task_input.sha256,
            "evaluation_protocols_sha256": protocol_scope,
            "pricing_snapshot_sha256": pricing_input.sha256,
        }
    )
    bank_manifest = store.write(
        artifact_id=manifest.bank_artifact_id,
        artifact_type="knn-bank",
        envelope=manifest,
        files={"bank.json": manifest.model_dump_json().encode(), "bank.npz": bank_bytes(bank)},
    )
    policy_inputs = tuple(
        sorted(
            (evaluation_input, artifact_input(bank_manifest)),
            key=lambda item: item.artifact_id,
        )
    )
    if seal_mutation and seal_mutation.startswith("omit-policy:"):
        omitted = seal_mutation.partition(":")[2]
        policy_inputs = tuple(item for item in policy_inputs if item.artifact_id != omitted)
    policy = policy.model_copy(
        update={
            "inputs": policy_inputs,
            "evaluation_plan_sha256": plan_input.sha256,
            "task_set_sha256": task_input.sha256,
            "evaluation_protocols_sha256": protocol_scope,
            "pricing_snapshot_sha256": pricing_input.sha256,
        }
    )
    store.write_json(
        artifact_id=policy.policy_id,
        artifact_type="router-policy",
        envelope=policy,
        files={"policy.json": policy},
    )
    return store, policy, cast(RuntimeModelCatalog, _Catalog(snapshots, client)), client
