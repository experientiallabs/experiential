"""Online selection and invocation from one immutable guarded router policy."""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from wmo.common.core.artifacts import (
    ArtifactId,
    ContractModel,
    Sha256,
    envelope_matches_manifest,
    sha256_json,
)
from wmo.common.models import (
    CandidateTokenPrice,
    IdempotentModelClient,
    ModelAlias,
    ModelRequest,
    ModelResponse,
    OperationEconomics,
)
from wmo.common.project import ArtifactCorruptionError, ArtifactStore
from wmo.common.routing import (
    KnnRouterPolicy,
    RouterFeatureExtractor,
    RoutingDecision,
)
from wmo.common.routing.bank import KnnBankManifest, KnnEvidenceBank, bank_bytes, load_knn_bank
from wmo.common.routing.decision import policy_content_sha256, select_from_bank
from wmo.runtime.models import ResolvedModel, RuntimeModelCatalog
from wmo.runtime.router.economics import (
    BillingSourceEconomics,
    RoutedCompletionEconomics,
    RoutedProviderComponent,
    RoutedProviderOperation,
    RoutedSpendDisposition,
    routed_completion_economics,
    zero_operation_economics,
)
from wmo.runtime.router.runtime_support import (
    candidate_completion_economics as _candidate_completion_economics,
)
from wmo.runtime.router.runtime_support import (
    candidate_reservation_economics as _candidate_reservation_economics,
)
from wmo.runtime.router.runtime_support import (
    candidate_success_disposition as _candidate_success_disposition,
)
from wmo.runtime.router.runtime_support import (
    decision_content_id as _decision_content_id,
)
from wmo.runtime.router.runtime_support import eligible_decision as _eligible_decision
from wmo.runtime.router.runtime_support import (
    embedding_economics as _embedding_economics,
)
from wmo.runtime.router.runtime_support import fallback_decision as _fallback_decision
from wmo.runtime.router.runtime_support import (
    requires_tool_protocol as _requires_tool_protocol,
)
from wmo.runtime.router.runtime_support import (
    runtime_provider_operation as _runtime_provider_operation,
)
from wmo.runtime.router.runtime_support import (
    sealed_bank as _sealed_bank,
)
from wmo.runtime.router.runtime_support import sticky_decision as _sticky_decision
from wmo.runtime.router.runtime_support import (
    validate_idempotency_key as _validate_idempotency_key,
)


class RouterRuntimeIntegrityError(ValueError):
    """Frozen policy, bank, catalog, pricing, or feature identity cannot be activated."""


class RouterEpisodeConflictError(ValueError):
    """A caller reused an episode identity with different request-visible inputs."""


class RouterModelCapabilityError(ValueError):
    """The selected frozen model cannot preserve the requested OpenAI capability."""


class RoutedModelResponse(ContractModel):
    """One exact routing decision and the response produced by its selected model."""

    decision: RoutingDecision
    response: ModelResponse
    economics: RoutedCompletionEconomics


DecisionSink = Callable[[RoutingDecision], None]


@dataclass(frozen=True)
class _PreparedSelection:
    """One unretained decision and its exact physical embedding evidence."""

    decision: RoutingDecision
    request_key: tuple[Sha256, Sha256]
    economics: OperationEconomics
    disposition: RoutedSpendDisposition


class RouterRuntime:
    """Select and call pinned model aliases without importing offline optimizer code."""

    def __init__(
        self,
        policy: KnnRouterPolicy,
        manifest: KnnBankManifest,
        bank: KnnEvidenceBank,
        catalog: RuntimeModelCatalog,
        *,
        pricing_snapshot_id: ArtifactId,
        pricing_snapshot_sha256: Sha256,
        pricing_candidate_aliases: tuple[ModelAlias, ...],
        pricing_candidate_prices: tuple[CandidateTokenPrice, ...] | None = None,
        decision_sink: DecisionSink | None = None,
        decision_capacity: int = 4_096,
        decision_ttl_seconds: float = 24 * 60 * 60,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if decision_capacity < 1 or decision_ttl_seconds <= 0:
            raise ValueError("router decision bounds must be positive")
        self.policy = policy
        self.manifest = manifest
        self.bank = _sealed_bank(bank)
        self.catalog = catalog
        self._extractor = RouterFeatureExtractor()
        self._decision_sink = decision_sink
        self._episode_decisions: OrderedDict[str, RoutingDecision] = OrderedDict()
        self._request_decisions: OrderedDict[tuple[Sha256, Sha256], RoutingDecision] = OrderedDict()
        self._request_embedding_economics: dict[tuple[Sha256, Sha256], OperationEconomics] = {}
        self._request_embedding_dispositions: dict[
            tuple[Sha256, Sha256], RoutedSpendDisposition
        ] = {}
        self._episode_expiry: dict[str, float] = {}
        self._request_expiry: dict[tuple[Sha256, Sha256], float] = {}
        self._selection_inflight: dict[tuple[Sha256, Sha256], threading.Event] = {}
        self._decision_capacity = decision_capacity
        self._decision_ttl_seconds = decision_ttl_seconds
        self._clock = clock
        self._episode_lock = threading.Lock()
        self._resolved: dict[str, ResolvedModel] = {}
        self._expected_models = {
            candidate.alias: candidate.model for candidate in policy.candidates
        }
        overlapping_embedder = self._expected_models.get(policy.embedder_alias)
        if overlapping_embedder is not None and overlapping_embedder != policy.embedder:
            raise RouterRuntimeIntegrityError(
                "router embedder alias overlaps a candidate with a different frozen identity"
            )
        self._expected_models[policy.embedder_alias] = policy.embedder
        self._bank_sha256 = hashlib.sha256(bank_bytes(self.bank)).hexdigest()
        self._require_activation_identity(
            pricing_snapshot_id, pricing_snapshot_sha256, pricing_candidate_aliases
        )
        self._candidate_prices = {
            item.candidate_alias: item for item in pricing_candidate_prices or ()
        }
        if self._candidate_prices and tuple(self._candidate_prices) != pricing_candidate_aliases:
            raise RouterRuntimeIntegrityError(
                "runtime candidate prices differ from fit-time candidate order"
            )
        try:
            for candidate in policy.candidates:
                self._resolve(candidate.alias)
            embedder = self._resolve(policy.embedder_alias)
        except Exception as exc:  # noqa: BLE001 - normalize catalog/provider construction errors
            raise RouterRuntimeIntegrityError(
                "runtime model catalog cannot resolve policy pins"
            ) from exc
        if embedder.embedding_client is None or not embedder.capabilities.supports_embeddings:
            raise RouterRuntimeIntegrityError("frozen router embedder lacks embedding capability")
        self._embedder = embedder.embedding_client
        self._embedder_billing_source = embedder.snapshot.billing_source
        self._embedder_input_price = embedder.capabilities.input_cost_per_million_tokens_usd

    @property
    def records_decisions(self) -> bool:
        """Return whether selections are sent to an injected decision recorder."""
        return self._decision_sink is not None

    @classmethod
    def load(
        cls,
        store: ArtifactStore,
        policy_id: ArtifactId,
        catalog: RuntimeModelCatalog,
        *,
        pricing_snapshot_id: ArtifactId,
        decision_sink: DecisionSink | None = None,
    ) -> RouterRuntime:
        """Load verified policy, bank, and pricing artifacts before activating runtime.

        Args:
            store: Project-local immutable artifact store.
            policy_id: Frozen router-policy identity.
            catalog: Runtime catalog resolving pinned aliases.
            pricing_snapshot_id: Canonical pricing artifact expected by the policy.
            decision_sink: Optional immutable decision recorder.

        Returns:
            Activated runtime with verified immutable dependencies.

        Raises:
            RouterRuntimeIntegrityError: Any artifact or runtime identity is invalid.
        """
        from wmo.common.models import load_pricing_snapshot
        from wmo.common.project import artifact_input

        try:
            from wmo.common.evaluations import load_evaluation_dataset
            from wmo.common.evaluations.evidence import read_evaluation_plan
            from wmo.common.tasks import load_task_set

            pricing, pricing_sha256 = load_pricing_snapshot(store, pricing_snapshot_id)
            pricing_input = artifact_input(store.read(pricing_snapshot_id).manifest)
            stored = store.read(policy_id)
            if stored.manifest.artifact_type != "router-policy":
                raise ValueError(f"artifact {policy_id} is not a router policy")
            policy = KnnRouterPolicy.model_validate_json(store.read_bytes(policy_id, "policy.json"))
            if policy.policy_id != policy_id:
                raise ValueError("router policy ID differs from its artifact")
            if not envelope_matches_manifest(policy, stored.manifest):
                raise ValueError("router policy payload differs from its artifact manifest")
            manifest, bank = load_knn_bank(
                store, policy.bank_artifact_id, expected_sha256=policy.bank_sha256
            )
            bank_input = artifact_input(store.read(policy.bank_artifact_id).manifest)
            evaluation = load_evaluation_dataset(store, policy.fit_evaluation_id)
            evaluation_input = artifact_input(store.read(policy.fit_evaluation_id).manifest)
            plan, plan_input = read_evaluation_plan(store, policy.evaluation_plan_id)
            load_task_set(store, policy.task_set_id)
            task_input = artifact_input(store.read(policy.task_set_id).manifest)
            expected_policy_inputs = tuple(
                sorted((evaluation_input, bank_input), key=lambda item: item.artifact_id)
            )
            expected_bank_inputs = tuple(
                sorted(
                    (evaluation_input, plan_input, task_input, pricing_input),
                    key=lambda item: item.artifact_id,
                )
            )
            protocol_scope_sha256 = sha256_json(
                [item.model_dump(mode="json") for item in evaluation.manifest.protocols]
            )
            if policy.inputs != expected_policy_inputs:
                raise ValueError("router policy inputs differ from the canonical fit lock")
            if manifest.inputs != expected_bank_inputs:
                raise ValueError("router bank inputs differ from the canonical fit evidence")
            evaluation_checks = (
                (evaluation.manifest.evaluation_plan_id, policy.evaluation_plan_id),
                (evaluation.manifest.evaluation_plan_sha256, policy.evaluation_plan_sha256),
                (evaluation.manifest.task_set_id, policy.task_set_id),
                (evaluation.manifest.candidate_snapshots, policy.candidates),
                (protocol_scope_sha256, policy.evaluation_protocols_sha256),
                (plan_input.sha256, policy.evaluation_plan_sha256),
                (task_input.sha256, policy.task_set_sha256),
                (evaluation.manifest.fit_task_ids, manifest.task_ids),
                (evaluation.manifest.held_out_task_ids, ()),
                (plan.task_set_id, policy.task_set_id),
                (plan.candidate_snapshots, policy.candidates),
                (plan.pricing_snapshot_id, policy.pricing_snapshot_id),
                (plan.pricing_snapshot_sha256, policy.pricing_snapshot_sha256),
            )
            if any(actual != expected for actual, expected in evaluation_checks):
                raise ValueError("router fit evidence differs from the frozen policy scope")
            required_evaluation_inputs = {
                plan_input,
                task_input,
                pricing_input,
            }
            if not required_evaluation_inputs.issubset(evaluation.manifest.inputs):
                raise ValueError("router evaluation omits a frozen scope input")
            if task_input not in plan.inputs or pricing_input not in plan.inputs:
                raise ValueError("router evaluation plan omits task or pricing scope")
        except (ArtifactCorruptionError, ValueError) as exc:
            raise RouterRuntimeIntegrityError(f"router policy {policy_id} is invalid") from exc
        return cls(
            policy,
            manifest,
            bank,
            catalog,
            pricing_snapshot_id=pricing_snapshot_id,
            pricing_snapshot_sha256=pricing_sha256,
            pricing_candidate_aliases=tuple(
                item.candidate_alias for item in pricing.candidate_prices
            ),
            pricing_candidate_prices=pricing.candidate_prices,
            decision_sink=decision_sink,
        )

    def select(
        self,
        request: ModelRequest,
        *,
        episode_id: str | None = None,
        retain: bool = True,
    ) -> RoutingDecision:
        """Return one sticky guarded decision with conservative request-time fallback.

        Args:
            request: Provider-neutral request visible before candidate execution.
            episode_id: Optional caller-owned identity for sticky whole-episode routing.
            retain: Whether to publish the decision into process-local sticky state.

        Returns:
            Exact immutable decision selected or cached for the episode.

        Raises:
            ValueError: The episode identity is malformed.
            RouterRuntimeIntegrityError: Frozen selection state is invalid.
        """
        if episode_id is not None and (not episode_id.strip() or len(episode_id) > 512):
            raise ValueError("episode_id must be 1 to 512 non-blank characters")
        identity = episode_id or f"request-{uuid.uuid4().hex}"
        identity_sha256 = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        feature = self._extractor.from_request(request)
        request_sha256 = hashlib.sha256(feature.encode("utf-8")).hexdigest()
        request_key = (identity_sha256, request_sha256)
        if not retain:
            return self._select_unretained(request, episode_id=identity).decision
        while True:
            with self._episode_lock:
                self._expire_decisions()
                existing = self._request_decisions.get(request_key)
                if existing is not None:
                    self._request_decisions.move_to_end(request_key)
                    return existing
                episode_decision = self._episode_decisions.get(identity_sha256)
                if episode_decision is not None:
                    decision = _sticky_decision(episode_decision, request_sha256)
                    return self._publish_decision(
                        request=request,
                        decision=decision,
                        request_key=request_key,
                        identity=identity,
                        embedding_economics=zero_operation_economics(),
                        embedding_disposition=RoutedSpendDisposition.DEFINITELY_NOT_INCURRED,
                    )
                waiter = self._selection_inflight.get(request_key)
                if waiter is None:
                    if len(self._selection_inflight) >= self._decision_capacity:
                        raise RouterRuntimeIntegrityError(
                            "router selection admission capacity is exhausted"
                        )
                    waiter = threading.Event()
                    self._selection_inflight[request_key] = waiter
                    break
            waiter.wait()
        try:
            prepared = self._select_unretained(request, episode_id=identity)
            proposed = prepared.decision
            with self._episode_lock:
                self._expire_decisions()
                existing = self._request_decisions.get(request_key)
                if existing is not None:
                    return existing
                episode_decision = self._episode_decisions.get(identity_sha256)
                decision = (
                    _sticky_decision(episode_decision, request_sha256)
                    if episode_decision is not None
                    else proposed
                )
                return self._publish_decision(
                    request=request,
                    decision=decision,
                    request_key=request_key,
                    identity=identity,
                    embedding_economics=prepared.economics,
                    embedding_disposition=prepared.disposition,
                )
        finally:
            with self._episode_lock:
                completed = self._selection_inflight.pop(request_key, None)
                if completed is not None:
                    completed.set()

    def _select_unretained(
        self,
        request: ModelRequest,
        *,
        episode_id: str,
    ) -> _PreparedSelection:
        """Compute a selection bundle without mutating sticky or recorder state.

        Args:
            request: Exact request visible to learned selection.
            episode_id: Stable caller-owned sticky identity.

        Returns:
            Opaque decision and exact physical embedding evidence.

        Raises:
            ValueError: The episode identity is invalid.
        """
        if not episode_id.strip() or len(episode_id) > 512:
            raise ValueError("episode_id must be 1 to 512 non-blank characters")
        identity_sha256 = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()
        feature = self._extractor.from_request(request)
        request_sha256 = hashlib.sha256(feature.encode("utf-8")).hexdigest()
        decision, economics, disposition = self._embedded_decision(
            feature=feature,
            request_sha256=request_sha256,
            identity=episode_id,
        )
        decision = _eligible_decision(
            request,
            decision,
            request_sha256,
            episode_id,
            policy=self.policy,
            bank=self.bank,
            resolve=self._resolve,
        )
        return _PreparedSelection(
            decision=decision,
            request_key=(identity_sha256, request_sha256),
            economics=economics,
            disposition=disposition,
        )

    def _reuse_sticky_selection(
        self,
        request: ModelRequest,
        *,
        episode_id: str,
    ) -> RoutingDecision | None:
        """Reuse one retained episode decision without provider work.

        Args:
            request: Provider-neutral request visible before execution.
            episode_id: Stable caller-owned episode identity.

        Returns:
            One cached or sticky decision, or ``None`` when selection needs an embedding.
        """
        if not episode_id.strip() or len(episode_id) > 512:
            raise ValueError("episode_id must be 1 to 512 non-blank characters")
        identity_sha256 = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()
        feature = self._extractor.from_request(request)
        request_sha256 = hashlib.sha256(feature.encode("utf-8")).hexdigest()
        request_key = (identity_sha256, request_sha256)
        with self._episode_lock:
            self._expire_decisions()
            existing = self._request_decisions.get(request_key)
            if existing is not None:
                self._request_decisions.move_to_end(request_key)
                return existing
            episode_decision = self._episode_decisions.get(identity_sha256)
            if episode_decision is None:
                return None
            decision = _sticky_decision(episode_decision, request_sha256)
            return self._publish_decision(
                request=request,
                decision=decision,
                request_key=request_key,
                identity=episode_id,
                embedding_economics=zero_operation_economics(),
                embedding_disposition=RoutedSpendDisposition.DEFINITELY_NOT_INCURRED,
            )

    def _retain_prepared_selection(
        self,
        request: ModelRequest,
        *,
        episode_id: str,
        prepared: _PreparedSelection,
    ) -> RoutingDecision:
        """Atomically publish one worker-owned prepared selection bundle.

        Args:
            request: Exact request used to produce the bundle.
            episode_id: Stable caller-owned sticky identity.
            prepared: Opaque decision and physical embedding evidence.

        Returns:
            Exact retained decision after reconciling concurrent episode state.

        Raises:
            ValueError: The bundle does not bind this request and episode.
        """
        if not episode_id.strip() or len(episode_id) > 512:
            raise ValueError("episode_id must be 1 to 512 non-blank characters")
        identity_sha256 = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()
        feature = self._extractor.from_request(request)
        request_sha256 = hashlib.sha256(feature.encode("utf-8")).hexdigest()
        decision = prepared.decision
        if (
            prepared.request_key != (identity_sha256, request_sha256)
            or decision.policy_id != self.policy.policy_id
            or decision.policy_sha256 != policy_content_sha256(self.policy)
            or decision.request_sha256 != request_sha256
            or decision.episode_id_sha256 != identity_sha256
            or decision.decision_id != _decision_content_id(decision)
        ):
            raise ValueError("unretained routing decision does not match this request")
        request_key = (identity_sha256, request_sha256)
        with self._episode_lock:
            self._expire_decisions()
            existing = self._request_decisions.get(request_key)
            if existing is not None:
                self._request_decisions.move_to_end(request_key)
                self._request_expiry[request_key] = self._expiry()
                self._request_embedding_economics[request_key] = prepared.economics
                self._request_embedding_dispositions[request_key] = prepared.disposition
                return existing
            episode_decision = self._episode_decisions.get(identity_sha256)
            selected = (
                _sticky_decision(episode_decision, request_sha256)
                if episode_decision is not None
                else decision
            )
            return self._publish_decision(
                request=request,
                decision=selected,
                request_key=request_key,
                identity=episode_id,
                embedding_economics=prepared.economics,
                embedding_disposition=prepared.disposition,
            )

    def embedding_reservation(self, request: ModelRequest) -> BillingSourceEconomics:
        """Return the source-attributed reservation before router embedding dispatch.

        Args:
            request: Provider-neutral request whose exact router feature will be embedded.

        Returns:
            Alias-free conservative embedding economics and immutable billing source.
        """
        feature = self._extractor.from_request(request)
        return BillingSourceEconomics(
            billing_source=self._embedder_billing_source,
            economics=_embedding_economics(
                feature,
                input_usd_per_million_tokens=self._embedder_input_price,
            ),
        )

    def selection_operation(
        self,
        request: ModelRequest,
        *,
        episode_id: str,
        decision: RoutingDecision,
    ) -> RoutedProviderOperation:
        """Return the exact settled embedding disposition for one completed selection.

        Args:
            request: Provider-neutral request used for selection.
            episode_id: Journal-derived sticky routing lineage.
            decision: Exact decision returned by ``select``.

        Returns:
            Alias-free operation evidence suitable for durable rebinding by the journal.

        Raises:
            ValueError: The decision was not produced for this request and lineage.
        """
        feature = self._extractor.from_request(request)
        request_sha256 = hashlib.sha256(feature.encode("utf-8")).hexdigest()
        episode_sha256 = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()
        request_key = (episode_sha256, request_sha256)
        with self._episode_lock:
            if self._request_decisions.get(request_key) != decision:
                raise ValueError("routing decision does not match the completed selection")
            economics = self._request_embedding_economics[request_key]
            disposition = self._request_embedding_dispositions[request_key]
        return _runtime_provider_operation(
            decision,
            operation_ordinal=1,
            component=RoutedProviderComponent.ROUTER_EMBEDDING,
            billing_source=self._embedder_billing_source,
            disposition=disposition,
            economics=economics,
        )

    def candidate_reservation(
        self,
        request: ModelRequest,
        *,
        episode_id: str,
        decision: RoutingDecision,
    ) -> BillingSourceEconomics:
        """Validate a pinned candidate and price its request before provider dispatch.

        Args:
            request: Provider-neutral request to complete.
            episode_id: Journal-derived sticky routing lineage.
            decision: Exact accepted routing decision.

        Returns:
            Alias-free conservative candidate reservation and immutable billing source.
        """
        resolved, _, _ = self._prepare_completion(request, episode_id=episode_id, decision=decision)
        return BillingSourceEconomics(
            billing_source=resolved.snapshot.billing_source,
            economics=_candidate_reservation_economics(
                request,
                resolved,
                self._candidate_prices.get(decision.selected_alias),
            ),
        )

    def complete(
        self,
        request: ModelRequest,
        *,
        episode_id: str | None = None,
        decision: RoutingDecision | None = None,
        provider_idempotency_key: str | None = None,
    ) -> RoutedModelResponse:
        """Validate one exact cached decision and call its pinned model client.

        Args:
            request: Provider-neutral request to route and complete.
            episode_id: Optional caller-owned identity for sticky routing.
            decision: Optional prior decision to validate and consume without reselection.
            provider_idempotency_key: Optional validated key forwarded only when the selected
                client implements the explicit idempotency capability.

        Returns:
            Exact routing decision beside the selected model response.

        Raises:
            ValueError: A supplied decision does not bind this request, episode, or policy.
        """
        selected = decision or self.select(request, episode_id=episode_id)
        if provider_idempotency_key is not None:
            _validate_idempotency_key(provider_idempotency_key)
        resolved, embedding_economics, embedding_disposition = self._prepare_completion(
            request,
            episode_id=episode_id,
            decision=selected,
        )
        client = resolved.client
        if provider_idempotency_key is not None and isinstance(client, IdempotentModelClient):
            response = client.complete_idempotent(request, idempotency_key=provider_idempotency_key)
        else:
            response = client.complete(request)
        price = self._candidate_prices.get(selected.selected_alias)
        candidate_economics = _candidate_completion_economics(response.economics, price)
        operations = (
            _runtime_provider_operation(
                selected,
                operation_ordinal=1,
                component=RoutedProviderComponent.ROUTER_EMBEDDING,
                billing_source=self._embedder_billing_source,
                disposition=embedding_disposition,
                economics=embedding_economics,
            ),
            _runtime_provider_operation(
                selected,
                operation_ordinal=2,
                component=RoutedProviderComponent.SELECTED_CANDIDATE,
                billing_source=resolved.snapshot.billing_source,
                disposition=_candidate_success_disposition(response.economics, candidate_economics),
                economics=candidate_economics,
            ),
        )
        return RoutedModelResponse(
            decision=selected,
            response=response,
            economics=routed_completion_economics(operations),
        )

    def _prepare_completion(
        self,
        request: ModelRequest,
        *,
        episode_id: str | None,
        decision: RoutingDecision,
    ) -> tuple[ResolvedModel, OperationEconomics, RoutedSpendDisposition]:
        """Validate request pins and capability before candidate provider dispatch."""
        selected = decision
        feature = self._extractor.from_request(request)
        request_sha256 = hashlib.sha256(feature.encode("utf-8")).hexdigest()
        expected_episode_sha256 = (
            hashlib.sha256(episode_id.encode("utf-8")).hexdigest()
            if episode_id is not None
            else selected.episode_id_sha256
        )
        if (
            selected.policy_id != self.policy.policy_id
            or selected.policy_sha256 != policy_content_sha256(self.policy)
            or selected.request_sha256 != request_sha256
            or selected.episode_id_sha256 != expected_episode_sha256
            or selected.selected_alias not in {item.alias for item in self.policy.candidates}
        ):
            raise ValueError("routing decision does not match this policy, request, or episode")
        self._require_bank_integrity()
        with self._episode_lock:
            self._expire_decisions()
            request_key = (expected_episode_sha256, request_sha256)
            cached = self._request_decisions.get(request_key)
            episode_decision = self._episode_decisions.get(expected_episode_sha256)
            if cached is None:
                if selected.decision_id != _decision_content_id(selected):
                    raise ValueError(
                        "routing decision does not match this policy, request, or episode"
                    )
                if (
                    episode_decision is not None
                    and episode_decision.selected_alias != selected.selected_alias
                ):
                    raise ValueError("routing decision conflicts with the cached episode model")
                if episode_decision is None:
                    self._insert_episode(expected_episode_sha256, selected)
                self._insert_request(
                    request_key,
                    selected,
                    zero_operation_economics(),
                    RoutedSpendDisposition.DEFINITELY_NOT_INCURRED,
                )
                cached = selected
            if cached != selected:
                raise ValueError("routing decision is not the exact cached episode decision")
            embedding_economics = self._request_embedding_economics.get(
                request_key, zero_operation_economics()
            )
            embedding_disposition = self._request_embedding_dispositions.get(
                request_key, RoutedSpendDisposition.DEFINITELY_NOT_INCURRED
            )
        resolved = self._resolve(selected.selected_alias)
        if _requires_tool_protocol(request) and not resolved.capabilities.supports_tools:
            raise RouterModelCapabilityError(
                f"routed model alias {selected.selected_alias!r} does not support tool calls"
            )
        if request.maximum_output_tokens is not None and (
            resolved.capabilities.maximum_output_tokens is None
            or request.maximum_output_tokens > resolved.capabilities.maximum_output_tokens
        ):
            raise RouterModelCapabilityError(
                f"routed model alias {selected.selected_alias!r} cannot prove the requested "
                "output-token capacity"
            )
        return resolved, embedding_economics, embedding_disposition

    def _require_activation_identity(
        self,
        pricing_snapshot_id: ArtifactId,
        pricing_snapshot_sha256: Sha256,
        pricing_candidate_aliases: tuple[ModelAlias, ...],
    ) -> None:
        if pricing_snapshot_id != self.policy.pricing_snapshot_id:
            raise RouterRuntimeIntegrityError("runtime pricing snapshot differs from fit time")
        if pricing_snapshot_sha256 != self.policy.pricing_snapshot_sha256:
            raise RouterRuntimeIntegrityError("runtime pricing manifest differs from fit time")
        if (
            self._extractor.extractor_id != self.policy.feature_extractor_id
            or self._extractor.schema_sha256 != self.policy.feature_schema_sha256
        ):
            raise RouterRuntimeIntegrityError("runtime router feature implementation has drifted")
        policy_aliases = tuple(candidate.alias for candidate in self.policy.candidates)
        if pricing_candidate_aliases != policy_aliases:
            raise RouterRuntimeIntegrityError("runtime pricing candidates differ from fit time")
        checks = (
            (self.policy.bank_artifact_id, self.manifest.bank_artifact_id, "bank artifact"),
            (self.policy.bank_sha256, self.manifest.bank_sha256, "bank digest"),
            (self.policy.fit_evaluation_id, self.manifest.fit_evaluation_id, "fit evaluation"),
            (
                self.policy.evaluation_plan_id,
                self.manifest.evaluation_plan_id,
                "evaluation plan",
            ),
            (
                self.policy.evaluation_plan_sha256,
                self.manifest.evaluation_plan_sha256,
                "evaluation plan digest",
            ),
            (self.policy.task_set_id, self.manifest.task_set_id, "task set"),
            (self.policy.task_set_sha256, self.manifest.task_set_sha256, "task-set digest"),
            (
                self.policy.evaluation_protocols_sha256,
                self.manifest.evaluation_protocols_sha256,
                "evaluation protocol scope",
            ),
            (self.policy.embedder_alias, self.manifest.embedder_alias, "embedder alias"),
            (self.policy.embedder, self.manifest.embedder, "embedder snapshot"),
            (
                self.policy.feature_extractor_id,
                self.manifest.feature_extractor_id,
                "feature extractor",
            ),
            (
                self.policy.feature_schema_sha256,
                self.manifest.feature_schema_sha256,
                "feature schema",
            ),
            (
                self.policy.pricing_snapshot_id,
                self.manifest.pricing_snapshot_id,
                "pricing snapshot",
            ),
            (
                self.policy.pricing_snapshot_sha256,
                self.manifest.pricing_snapshot_sha256,
                "pricing snapshot digest",
            ),
            (policy_aliases, self.manifest.candidate_aliases, "candidate aliases"),
            (self.manifest.candidate_aliases, self.bank.candidate_aliases, "bank columns"),
            (self.manifest.task_ids, self.bank.task_ids, "bank rows"),
        )
        for expected_value, actual_value, label in checks:
            if expected_value != actual_value:
                raise RouterRuntimeIntegrityError(f"router {label} has drifted from fit time")
        self._require_bank_integrity()

    def _require_bank_integrity(self) -> None:
        current = hashlib.sha256(bank_bytes(self.bank)).hexdigest()
        if current != self._bank_sha256 or current != self.policy.bank_sha256:
            raise RouterRuntimeIntegrityError("router bank content has mutated from fit time")

    def _resolve(self, alias: str) -> ResolvedModel:
        """Resolve and cache one model only after its frozen identity verifies.

        Args:
            alias: Candidate alias pinned by the frozen router policy.

        Returns:
            The verified runtime model binding for ``alias``.

        Raises:
            RouterRuntimeIntegrityError: The runtime binding differs from its fit-time identity.
        """
        resolved = self._resolved.get(alias)
        if resolved is None:
            resolved = self.catalog.resolve(alias, role="candidate")
            expected = self._expected_models.get(alias)
            if (
                expected is None
                or resolved.alias != alias
                or resolved.snapshot != expected
                or resolved.capabilities.identity_sha256() != expected.capabilities_sha256
            ):
                raise RouterRuntimeIntegrityError(
                    f"resolved runtime alias {alias!r} differs from its frozen identity"
                )
            self._resolved[alias] = resolved
        return resolved

    def _embedded_decision(
        self,
        *,
        feature: str,
        request_sha256: Sha256,
        identity: str,
    ) -> tuple[RoutingDecision, OperationEconomics, RoutedSpendDisposition]:
        """Select outside the shared lock and retain exact embedding accounting."""
        economics = _embedding_economics(
            feature,
            input_usd_per_million_tokens=self._embedder_input_price,
        )
        try:
            embedded = self._embedder.embed((feature,))
            if len(embedded) != 1:
                raise ValueError("embedder returned a non-singleton result")
            vector = np.asarray(embedded[0].values, dtype=np.float64)
            if (
                vector.shape != (self.manifest.embedding_dimension,)
                or not np.all(np.isfinite(vector))
                or float(np.linalg.norm(vector)) == 0
            ):
                raise ValueError("embedder returned an invalid router vector")
        except Exception:  # noqa: BLE001 - request-time embedding failures fall back
            return (
                _fallback_decision(self.policy, request_sha256, identity, "embedding_error"),
                economics,
                RoutedSpendDisposition.RESERVED_AMBIGUOUS,
            )
        return (
            select_from_bank(
                self.policy,
                self.manifest,
                self.bank,
                vector,
                request_sha256=request_sha256,
                episode_id=identity,
            ),
            economics,
            RoutedSpendDisposition.LOCALLY_PRICED,
        )

    def _publish_decision(
        self,
        *,
        request: ModelRequest,
        decision: RoutingDecision,
        request_key: tuple[Sha256, Sha256],
        identity: str,
        embedding_economics: OperationEconomics,
        embedding_disposition: RoutedSpendDisposition,
    ) -> RoutingDecision:
        """Validate, record, and retain one decision while the caller holds the lock."""
        identity_sha256, request_sha256 = request_key
        episode_decision = self._episode_decisions.get(identity_sha256)
        decision = _eligible_decision(
            request,
            decision,
            request_sha256,
            identity,
            policy=self.policy,
            bank=self.bank,
            resolve=self._resolve,
        )
        self._record(decision)
        if episode_decision is None or decision.selected_alias != episode_decision.selected_alias:
            if episode_decision is not None:
                self._remove_episode_requests(identity_sha256)
            self._insert_episode(identity_sha256, decision)
        else:
            self._episode_decisions.move_to_end(identity_sha256)
            self._episode_expiry[identity_sha256] = self._expiry()
        self._insert_request(
            request_key,
            decision,
            embedding_economics,
            embedding_disposition,
        )
        return decision

    def _expiry(self) -> float:
        """Return one deadline for newly retained process-local decisions."""
        return self._clock() + self._decision_ttl_seconds

    def _insert_episode(self, key: str, decision: RoutingDecision) -> None:
        """Insert one episode decision with TTL and bounded least-recent eviction."""
        self._episode_decisions[key] = decision
        self._episode_decisions.move_to_end(key)
        self._episode_expiry[key] = self._expiry()
        while len(self._episode_decisions) > self._decision_capacity:
            stale_key, _ = self._episode_decisions.popitem(last=False)
            self._episode_expiry.pop(stale_key, None)
            self._remove_episode_requests(stale_key)

    def _insert_request(
        self,
        key: tuple[Sha256, Sha256],
        decision: RoutingDecision,
        embedding_economics: OperationEconomics,
        embedding_disposition: RoutedSpendDisposition,
    ) -> None:
        """Insert one request decision with TTL and bounded least-recent eviction."""
        self._request_decisions[key] = decision
        self._request_embedding_economics[key] = embedding_economics
        self._request_embedding_dispositions[key] = embedding_disposition
        self._request_decisions.move_to_end(key)
        self._request_expiry[key] = self._expiry()
        while len(self._request_decisions) > self._decision_capacity:
            stale_key, _ = self._request_decisions.popitem(last=False)
            self._request_expiry.pop(stale_key, None)
            self._request_embedding_economics.pop(stale_key, None)
            self._request_embedding_dispositions.pop(stale_key, None)

    def _remove_episode_requests(self, identity_sha256: Sha256) -> None:
        """Remove retained request variants belonging to one episode identity."""
        stale_keys = tuple(key for key in self._request_decisions if key[0] == identity_sha256)
        for stale_key in stale_keys:
            self._request_decisions.pop(stale_key, None)
            self._request_expiry.pop(stale_key, None)
            self._request_embedding_economics.pop(stale_key, None)
            self._request_embedding_dispositions.pop(stale_key, None)

    def _expire_decisions(self) -> None:
        """Discard expired process-local decisions while the caller holds the lock."""
        now = self._clock()
        expired_episodes = tuple(
            key for key, expires_at in self._episode_expiry.items() if expires_at <= now
        )
        for key in expired_episodes:
            self._episode_decisions.pop(key, None)
            self._episode_expiry.pop(key, None)
            self._remove_episode_requests(key)
        expired_requests = tuple(
            key for key, expires_at in self._request_expiry.items() if expires_at <= now
        )
        for key in expired_requests:
            self._request_decisions.pop(key, None)
            self._request_expiry.pop(key, None)
            self._request_embedding_economics.pop(key, None)
            self._request_embedding_dispositions.pop(key, None)

    def _record(self, decision: RoutingDecision) -> RoutingDecision:
        if self._decision_sink is not None:
            self._decision_sink(decision)
        return decision
