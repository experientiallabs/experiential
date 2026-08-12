"""Online selection and invocation from one immutable guarded router policy."""

from __future__ import annotations

import hashlib
import threading
import uuid
from collections.abc import Callable

import numpy as np

from wmo.common.core.artifacts import ArtifactId, ContractModel, stable_id
from wmo.common.models import ModelRequest, ModelResponse
from wmo.common.project import ArtifactStore
from wmo.common.routing import KnnRouterPolicy, RouterFeatureExtractor, RoutingDecision
from wmo.common.routing.bank import KnnBankManifest, KnnEvidenceBank, bank_bytes, load_knn_bank
from wmo.common.routing.decision import policy_content_sha256, select_from_bank
from wmo.runtime.models import ResolvedModel, RuntimeModelCatalog


class RouterRuntimeIntegrityError(ValueError):
    """Frozen policy, bank, catalog, pricing, or feature identity cannot be activated."""


class RouterEpisodeConflictError(ValueError):
    """A caller reused an episode identity with different request-visible inputs."""


class RoutedModelResponse(ContractModel):
    """One exact routing decision and the response produced by its selected model."""

    decision: RoutingDecision
    response: ModelResponse


DecisionSink = Callable[[RoutingDecision], None]


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
        pricing_snapshot_sha256: str | None = None,
        decision_sink: DecisionSink | None = None,
    ) -> None:
        self.policy = policy
        self.manifest = manifest
        self.bank = bank
        self.catalog = catalog
        self._extractor = RouterFeatureExtractor()
        self._decision_sink = decision_sink
        self._episode_decisions: dict[str, RoutingDecision] = {}
        self._episode_lock = threading.Lock()
        self._resolved: dict[str, ResolvedModel] = {}
        self._bank_sha256 = hashlib.sha256(bank_bytes(bank)).hexdigest()
        self._require_activation_identity(pricing_snapshot_id, pricing_snapshot_sha256)
        for candidate in policy.candidates:
            self._resolve(candidate.alias)
        embedder = self._resolve(policy.embedder_alias)
        if embedder.embedding_client is None or not embedder.capabilities.supports_embeddings:
            raise RouterRuntimeIntegrityError("frozen router embedder lacks embedding capability")
        self._embedder = embedder.embedding_client

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

        try:
            _pricing, pricing_sha256 = load_pricing_snapshot(store, pricing_snapshot_id)
        except ValueError as exc:
            raise RouterRuntimeIntegrityError("runtime pricing identity is invalid") from exc
        stored = store.read(policy_id)
        if stored.manifest.artifact_type != "router-policy":
            raise RouterRuntimeIntegrityError(f"artifact {policy_id} is not a router policy")
        try:
            policy = KnnRouterPolicy.model_validate_json(store.read_bytes(policy_id, "policy.json"))
            if policy.policy_id != policy_id:
                raise ValueError("router policy ID differs from its artifact")
            manifest, bank = load_knn_bank(
                store, policy.bank_artifact_id, expected_sha256=policy.bank_sha256
            )
        except ValueError as exc:
            raise RouterRuntimeIntegrityError(f"router policy {policy_id} is invalid") from exc
        return cls(
            policy,
            manifest,
            bank,
            catalog,
            pricing_snapshot_id=pricing_snapshot_id,
            pricing_snapshot_sha256=pricing_sha256,
            decision_sink=decision_sink,
        )

    def select(self, request: ModelRequest, *, episode_id: str | None = None) -> RoutingDecision:
        """Return one sticky guarded decision with conservative request-time fallback.

        Args:
            request: Provider-neutral request visible before candidate execution.
            episode_id: Optional caller-owned identity for sticky whole-episode routing.

        Returns:
            Exact immutable decision selected or cached for the episode.

        Raises:
            ValueError: The episode identity is malformed.
            RouterEpisodeConflictError: An episode identity is reused for another request.
            RouterRuntimeIntegrityError: The immutable evidence bank has mutated.
        """
        if episode_id is not None and (not episode_id.strip() or len(episode_id) > 512):
            raise ValueError("episode_id must be 1 to 512 non-blank characters")
        identity = episode_id or f"request-{uuid.uuid4().hex}"
        identity_sha256 = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        feature = self._extractor.from_request(request)
        request_sha256 = hashlib.sha256(feature.encode("utf-8")).hexdigest()
        with self._episode_lock:
            existing = self._episode_decisions.get(identity_sha256)
        if existing is not None:
            if existing.request_sha256 != request_sha256:
                raise RouterEpisodeConflictError(
                    "episode ID was reused with a different request-visible hash"
                )
            return existing
        self._require_bank_integrity()
        try:
            embedded = self._embedder.embed((feature,))
            if len(embedded) != 1:
                raise ValueError("embedder returned a non-singleton result")
            vector = np.asarray(embedded[0].values, dtype=np.float64)
            decision = select_from_bank(
                self.policy,
                self.manifest,
                self.bank,
                vector,
                request_sha256=request_sha256,
                episode_id=identity,
            )
        except Exception:  # noqa: BLE001 - all request-time embed/vector failures fall back
            decision = self._fallback_decision(request_sha256, identity, "embedding_error")
        with self._episode_lock:
            existing = self._episode_decisions.setdefault(identity_sha256, decision)
        if existing.request_sha256 != request_sha256:
            raise RouterEpisodeConflictError(
                "episode ID was reused with a different request-visible hash"
            )
        decision = existing
        return self._record(decision)

    def complete(
        self,
        request: ModelRequest,
        *,
        episode_id: str | None = None,
        decision: RoutingDecision | None = None,
    ) -> RoutedModelResponse:
        """Consume one validated decision exactly once and call its pinned model client.

        Args:
            request: Provider-neutral request to route and complete.
            episode_id: Optional caller-owned identity for sticky routing.
            decision: Optional prior decision to validate and consume without reselection.

        Returns:
            Exact routing decision beside the selected model response.

        Raises:
            ValueError: A supplied decision does not bind this request, episode, or policy.
            RouterEpisodeConflictError: An episode identity is reused for another request.
        """
        selected = decision or self.select(request, episode_id=episode_id)
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
        response = self._resolve(selected.selected_alias).client.complete(request)
        return RoutedModelResponse(decision=selected, response=response)

    def _require_activation_identity(
        self, pricing_snapshot_id: ArtifactId, pricing_snapshot_sha256: str | None
    ) -> None:
        if pricing_snapshot_id != self.policy.pricing_snapshot_id:
            raise RouterRuntimeIntegrityError("runtime pricing snapshot differs from fit time")
        if (
            pricing_snapshot_sha256 is not None
            and pricing_snapshot_sha256 != self.policy.pricing_snapshot_sha256
        ):
            raise RouterRuntimeIntegrityError("runtime pricing manifest differs from fit time")
        if (
            self._extractor.extractor_id != self.policy.feature_extractor_id
            or self._extractor.schema_sha256 != self.policy.feature_schema_sha256
        ):
            raise RouterRuntimeIntegrityError("runtime router feature implementation has drifted")
        policy_aliases = tuple(candidate.alias for candidate in self.policy.candidates)
        checks = (
            (self.policy.bank_artifact_id, self.manifest.bank_artifact_id, "bank artifact"),
            (self.policy.bank_sha256, self.manifest.bank_sha256, "bank digest"),
            (self.policy.fit_evaluation_id, self.manifest.fit_evaluation_id, "fit evaluation"),
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
            (policy_aliases, self.manifest.candidate_aliases, "candidate aliases"),
            (self.manifest.candidate_aliases, self.bank.candidate_aliases, "bank columns"),
            (self.manifest.task_ids, self.bank.task_ids, "bank rows"),
        )
        for expected_value, actual_value, label in checks:
            if expected_value != actual_value:
                raise RouterRuntimeIntegrityError(f"router {label} has drifted from fit time")
        expected = {candidate.alias: candidate.model for candidate in self.policy.candidates}
        expected[self.policy.embedder_alias] = self.policy.embedder
        for alias, snapshot in expected.items():
            actual, _capabilities = self.catalog.snapshot(alias)
            if actual != snapshot:
                raise RouterRuntimeIntegrityError(
                    f"runtime alias {alias!r} identity or connection digest has drifted"
                )
        self._require_bank_integrity()

    def _require_bank_integrity(self) -> None:
        current = hashlib.sha256(bank_bytes(self.bank)).hexdigest()
        if current != self._bank_sha256 or current != self.policy.bank_sha256:
            raise RouterRuntimeIntegrityError("router bank content has mutated from fit time")

    def _resolve(self, alias: str) -> ResolvedModel:
        resolved = self._resolved.get(alias)
        if resolved is None:
            resolved = self.catalog.resolve(alias)
            self._resolved[alias] = resolved
        return resolved

    def _fallback_decision(
        self,
        request_sha256: str,
        episode_id: str,
        reason: str,
        *,
        selected_alias: str | None = None,
    ) -> RoutingDecision:
        alias = selected_alias or self.policy.baseline_alias
        episode_id_sha256 = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()
        material = {
            "policy_id": self.policy.policy_id,
            "request_sha256": request_sha256,
            "episode_id_sha256": episode_id_sha256,
            "selected_alias": alias,
            "fallback_reason": reason,
        }
        return RoutingDecision(
            decision_id=stable_id("routing-decision", material),
            policy_id=self.policy.policy_id,
            policy_sha256=policy_content_sha256(self.policy),
            request_sha256=request_sha256,
            episode_id_sha256=episode_id_sha256,
            selected_alias=alias,
            baseline_alias=self.policy.baseline_alias,
            neighbor_count=0,
            paired_count=0,
            fallback_reason=reason,
        )

    def _record(self, decision: RoutingDecision) -> RoutingDecision:
        if self._decision_sink is not None:
            self._decision_sink(decision)
        return decision
