"""Online selection and invocation from one immutable guarded router policy."""

from __future__ import annotations

import hashlib
import threading
import uuid
from collections.abc import Callable

import numpy as np

from wmo.common.core.artifacts import ArtifactId, ContractModel, Sha256, sha256_json, stable_id
from wmo.common.models import ModelAlias, ModelRequest, ModelResponse
from wmo.common.project import ArtifactCorruptionError, ArtifactStore
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
        pricing_snapshot_sha256: Sha256,
        pricing_candidate_aliases: tuple[ModelAlias, ...],
        decision_sink: DecisionSink | None = None,
    ) -> None:
        self.policy = policy
        self.manifest = manifest
        self.bank = bank
        self.catalog = catalog
        self._extractor = RouterFeatureExtractor()
        self._decision_sink = decision_sink
        self._episode_decisions: dict[str, RoutingDecision] = {}
        self._request_decisions: dict[tuple[Sha256, Sha256], RoutingDecision] = {}
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
        self._bank_sha256 = hashlib.sha256(bank_bytes(bank)).hexdigest()
        self._require_activation_identity(
            pricing_snapshot_id, pricing_snapshot_sha256, pricing_candidate_aliases
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
            from wmo.common.evaluations.evidence import (
                read_evaluation_plan,
                read_fidelity_report,
            )
            from wmo.common.tasks import load_task_set

            pricing, pricing_sha256 = load_pricing_snapshot(store, pricing_snapshot_id)
            pricing_input = artifact_input(store.read(pricing_snapshot_id).manifest)
            stored = store.read(policy_id)
            if stored.manifest.artifact_type != "router-policy":
                raise ValueError(f"artifact {policy_id} is not a router policy")
            policy = KnnRouterPolicy.model_validate_json(store.read_bytes(policy_id, "policy.json"))
            if policy.policy_id != policy_id:
                raise ValueError("router policy ID differs from its artifact")
            if (
                policy.schema_version,
                policy.created_at,
                policy.inputs,
                policy.code_revision,
                policy.source,
            ) != (
                stored.manifest.schema_version,
                stored.manifest.created_at,
                stored.manifest.inputs,
                stored.manifest.code_revision,
                stored.manifest.source,
            ):
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
            report_inputs = tuple(
                read_fidelity_report(store, report_id)[1]
                for report_id in policy.fidelity_report_ids
            )
            expected_policy_inputs = tuple(
                sorted((evaluation_input, bank_input), key=lambda item: item.artifact_id)
            )
            expected_bank_inputs = tuple(
                sorted(
                    (evaluation_input, plan_input, task_input, pricing_input, *report_inputs),
                    key=lambda item: item.artifact_id,
                )
            )
            protocol_scope_sha256 = sha256_json(
                {
                    "protocols": [
                        item.model_dump(mode="json") for item in evaluation.manifest.protocols
                    ],
                    "fidelity_report_ids": list(evaluation.manifest.fidelity_report_ids),
                }
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
                (evaluation.manifest.fidelity_report_ids, policy.fidelity_report_ids),
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
                *report_inputs,
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
            RouterRuntimeIntegrityError: The immutable evidence bank has mutated.
        """
        if episode_id is not None and (not episode_id.strip() or len(episode_id) > 512):
            raise ValueError("episode_id must be 1 to 512 non-blank characters")
        identity = episode_id or f"request-{uuid.uuid4().hex}"
        identity_sha256 = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        feature = self._extractor.from_request(request)
        request_sha256 = hashlib.sha256(feature.encode("utf-8")).hexdigest()
        with self._episode_lock:
            self._require_bank_integrity()
            request_key = (identity_sha256, request_sha256)
            existing = self._request_decisions.get(request_key)
            if existing is not None:
                return existing
            episode_decision = self._episode_decisions.get(identity_sha256)
            if episode_decision is not None:
                decision = self._sticky_decision(episode_decision, request_sha256)
            else:
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
                    decision = self._fallback_decision(request_sha256, identity, "embedding_error")
                else:
                    decision = select_from_bank(
                        self.policy,
                        self.manifest,
                        self.bank,
                        vector,
                        request_sha256=request_sha256,
                        episode_id=identity,
                    )
            self._record(decision)
            if episode_decision is None:
                self._episode_decisions[identity_sha256] = decision
            self._request_decisions[request_key] = decision
            return decision

    def complete(
        self,
        request: ModelRequest,
        *,
        episode_id: str | None = None,
        decision: RoutingDecision | None = None,
    ) -> RoutedModelResponse:
        """Validate one exact cached decision and call its pinned model client.

        Args:
            request: Provider-neutral request to route and complete.
            episode_id: Optional caller-owned identity for sticky routing.
            decision: Optional prior decision to validate and consume without reselection.

        Returns:
            Exact routing decision beside the selected model response.

        Raises:
            ValueError: A supplied decision does not bind this request, episode, or policy.
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
        self._require_bank_integrity()
        with self._episode_lock:
            cached = self._request_decisions.get((expected_episode_sha256, request_sha256))
            if cached != selected:
                raise ValueError("routing decision is not the exact cached episode decision")
        response = self._resolve(selected.selected_alias).client.complete(request)
        return RoutedModelResponse(decision=selected, response=response)

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
            (
                self.policy.fidelity_report_ids,
                self.manifest.fidelity_report_ids,
                "fidelity scope",
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
        resolved = self._resolved.get(alias)
        if resolved is None:
            resolved = self.catalog.resolve(alias)
            expected = self._expected_models.get(alias)
            if (
                expected is None
                or resolved.alias != alias
                or resolved.snapshot != expected
                or sha256_json(resolved.capabilities) != expected.capabilities_sha256
            ):
                raise RouterRuntimeIntegrityError(
                    f"resolved runtime alias {alias!r} differs from its frozen identity"
                )
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

    def _sticky_decision(
        self, episode_decision: RoutingDecision, request_sha256: Sha256
    ) -> RoutingDecision:
        """Bind a later episode turn to the original selected alias and evidence."""
        material = episode_decision.model_copy(update={"request_sha256": request_sha256})
        identity_material = material.model_dump(mode="json")
        del identity_material["decision_id"]
        return material.model_copy(
            update={"decision_id": stable_id("routing-decision", identity_material)}
        )

    def _record(self, decision: RoutingDecision) -> RoutingDecision:
        if self._decision_sink is not None:
            self._decision_sink(decision)
        return decision
