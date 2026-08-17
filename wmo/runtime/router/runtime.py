"""Online selection and invocation from one immutable guarded router policy."""

from __future__ import annotations

import hashlib
import threading
import uuid
from collections.abc import Callable

import numpy as np
from pydantic import Field, model_validator

from wmo.common.core.artifacts import (
    ArtifactId,
    ContractModel,
    Sha256,
    envelope_matches_manifest,
    sha256_json,
    stable_id,
)
from wmo.common.models import (
    CandidateTokenPrice,
    IdempotentModelClient,
    ModelAlias,
    ModelRequest,
    ModelResponse,
    NumericMeasurement,
    OperationEconomics,
    Usage,
    combine_economics,
)
from wmo.common.project import ArtifactCorruptionError, ArtifactStore
from wmo.common.routing import (
    KnnRouterPolicy,
    RouterFeatureExtractor,
    RoutingDecision,
    router_feature_token_upper_bound,
)
from wmo.common.routing.bank import KnnBankManifest, KnnEvidenceBank, bank_bytes, load_knn_bank
from wmo.common.routing.decision import policy_content_sha256, select_from_bank
from wmo.runtime.models import ResolvedModel, RuntimeModelCatalog


class RouterRuntimeIntegrityError(ValueError):
    """Frozen policy, bank, catalog, pricing, or feature identity cannot be activated."""


class RouterEpisodeConflictError(ValueError):
    """A caller reused an episode identity with different request-visible inputs."""


class RouterModelCapabilityError(ValueError):
    """The selected frozen model cannot preserve the requested OpenAI capability."""


class RoutedCompletionEconomics(ContractModel):
    """Alias-free economics for routing and the selected candidate completion."""

    router_embedding: OperationEconomics = Field(default_factory=OperationEconomics)
    selected_candidate: OperationEconomics = Field(default_factory=OperationEconomics)
    total: OperationEconomics = Field(default_factory=OperationEconomics)

    @model_validator(mode="after")
    def _require_complete_total(self) -> RoutedCompletionEconomics:
        """Require the total to be the strict sum of both customer-visible components."""
        expected = combine_economics((self.router_embedding, self.selected_candidate))
        if self.total != expected:
            raise ValueError("routed completion total differs from its component economics")
        return self


class RoutedModelResponse(ContractModel):
    """One exact routing decision and the response produced by its selected model."""

    decision: RoutingDecision
    response: ModelResponse
    economics: RoutedCompletionEconomics = Field(default_factory=RoutedCompletionEconomics)


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
        pricing_candidate_prices: tuple[CandidateTokenPrice, ...] | None = None,
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
        self._request_embedding_economics: dict[tuple[Sha256, Sha256], OperationEconomics] = {}
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
            embedding_economics = _zero_operation_economics()
            if episode_decision is not None:
                decision = self._sticky_decision(episode_decision, request_sha256)
            else:
                try:
                    embedded = self._embedder.embed((feature,))
                    embedding_economics = _embedding_economics(
                        feature,
                        input_usd_per_million_tokens=self._embedder_input_price,
                    )
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
            decision = self._eligible_decision(request, decision, request_sha256, identity)
            self._record(decision)
            if (
                episode_decision is None
                or decision.selected_alias != episode_decision.selected_alias
            ):
                if episode_decision is not None:
                    stale_request_keys = tuple(
                        key for key in self._request_decisions if key[0] == identity_sha256
                    )
                    for stale_key in stale_request_keys:
                        del self._request_decisions[stale_key]
                        self._request_embedding_economics.pop(stale_key, None)
                self._episode_decisions[identity_sha256] = decision
            self._request_decisions[request_key] = decision
            self._request_embedding_economics[request_key] = embedding_economics
            return decision

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
        if provider_idempotency_key is not None:
            _validate_idempotency_key(provider_idempotency_key)
        self._require_bank_integrity()
        with self._episode_lock:
            request_key = (expected_episode_sha256, request_sha256)
            cached = self._request_decisions.get(request_key)
            episode_decision = self._episode_decisions.get(expected_episode_sha256)
            if cached is None and decision is not None:
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
                    self._episode_decisions[expected_episode_sha256] = selected
                self._request_decisions[request_key] = selected
                self._request_embedding_economics[request_key] = _zero_operation_economics()
                cached = selected
            if cached != selected:
                raise ValueError("routing decision is not the exact cached episode decision")
            embedding_economics = self._request_embedding_economics.get(
                request_key, _zero_operation_economics()
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
        client = resolved.client
        if provider_idempotency_key is not None and isinstance(client, IdempotentModelClient):
            response = client.complete_idempotent(request, idempotency_key=provider_idempotency_key)
        else:
            response = client.complete(request)
        candidate_economics = _candidate_completion_economics(
            response.economics,
            self._candidate_prices.get(selected.selected_alias),
        )
        return RoutedModelResponse(
            decision=selected,
            response=response,
            economics=RoutedCompletionEconomics(
                router_embedding=embedding_economics,
                selected_candidate=candidate_economics,
                total=combine_economics((embedding_economics, candidate_economics)),
            ),
        )

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
            resolved = self.catalog.resolve(alias)
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

    def _eligible_decision(
        self,
        request: ModelRequest,
        decision: RoutingDecision,
        request_sha256: Sha256,
        episode_id: str,
    ) -> RoutingDecision:
        """Use a frozen eligible candidate when the original selection cannot serve the request.

        Args:
            request: Provider-neutral request whose capabilities must be supported.
            decision: Original guarded routing decision.
            request_sha256: Frozen feature identity for the request.
            episode_id: Stable caller identity used by fallback decisions.

        Returns:
            The original decision when eligible, otherwise a guarded fallback decision.
        """
        if _supports_request(self._resolve(decision.selected_alias), request):
            return decision
        eligible = tuple(
            candidate.alias
            for candidate in self.policy.candidates
            if _supports_request(self._resolve(candidate.alias), request)
        )
        if not eligible:
            return decision
        alias = (
            self.policy.baseline_alias
            if self.policy.baseline_alias in eligible
            else min(
                eligible,
                key=lambda item: (
                    self.bank.complete_weighted_cost(item) is None,
                    self.bank.complete_weighted_cost(item) or 0.0,
                    item,
                ),
            )
        )
        return self._fallback_decision(
            request_sha256,
            episode_id,
            "capability_eligibility",
            selected_alias=alias,
        )

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
        provisional = RoutingDecision(
            decision_id="routing-decision-provisional",
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
        return provisional.model_copy(update={"decision_id": _decision_content_id(provisional)})

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


def _zero_operation_economics() -> OperationEconomics:
    """Return explicit alias-free evidence that no provider operation was incurred."""
    return OperationEconomics(
        usage=Usage(input_tokens=0, output_tokens=0),
        cost_usd=NumericMeasurement(value=0.0, provenance="estimated"),
    )


def _embedding_economics(
    feature: str,
    *,
    input_usd_per_million_tokens: float | None,
) -> OperationEconomics:
    """Estimate one successful online router embedding from its request-visible feature.

    Args:
        feature: Exact provider input rendered by the router feature extractor.
        input_usd_per_million_tokens: Active explicit embedding price when declared.

    Returns:
        Conservative usage and locally priced cost without a model alias.
    """
    tokens = router_feature_token_upper_bound(feature)
    cost = (
        None
        if input_usd_per_million_tokens is None
        else NumericMeasurement(
            value=tokens * input_usd_per_million_tokens / 1_000_000,
            provenance="estimated",
        )
    )
    return OperationEconomics(
        usage=Usage(input_tokens=tokens, output_tokens=0),
        cost_usd=cost,
    )


def _candidate_completion_economics(
    economics: OperationEconomics,
    price: CandidateTokenPrice | None,
) -> OperationEconomics:
    """Retain measured candidate cost or locally price its observed token usage.

    Args:
        economics: Provider response economics.
        price: Fit-time selected-candidate price, kept private from the result.

    Returns:
        Reconciled alias-free candidate economics.

    Raises:
        ValueError: Provider cache counters exceed the reported input total.
    """
    usage = economics.usage
    if economics.cost_usd is not None or usage is None or price is None:
        return economics
    cached = usage.cached_input_tokens
    written = usage.cache_write_input_tokens
    if cached is not None and cached > usage.input_tokens:
        raise ValueError("candidate cached input exceeds total input usage")
    if written is not None and written > usage.input_tokens:
        raise ValueError("candidate cache-write input exceeds total input usage")
    if cached is not None and written is not None and cached + written > usage.input_tokens:
        raise ValueError("candidate cache counters overlap beyond total input usage")
    input_cost = _candidate_input_cost_usd(price, usage)
    output_cost = usage.output_tokens * price.output_usd_per_million_tokens / 1_000_000
    return economics.model_copy(
        update={
            "cost_usd": NumericMeasurement(
                value=input_cost + output_cost,
                provenance="estimated",
            )
        }
    )


def _candidate_input_cost_usd(price: CandidateTokenPrice, usage: Usage) -> float:
    """Conservatively price mutually exclusive ordinary, cached, and cache-write input.

    Args:
        price: Frozen candidate price units.
        usage: Provider-reported input and cache token counts.

    Returns:
        Locally priced input cost in USD.
    """
    base = price.input_usd_per_million_tokens
    cached_price = price.cached_input_usd_per_million_tokens
    write_price = price.cache_write_usd_per_million_tokens
    cached = usage.cached_input_tokens
    written = usage.cache_write_input_tokens
    if cached is not None and written is not None:
        ordinary = usage.input_tokens - cached - written
        total = (
            ordinary * base
            + cached * (cached_price if cached_price is not None else base)
            + written * (write_price if write_price is not None else base)
        )
    elif cached is not None:
        ordinary_price = max(base, write_price if write_price is not None else base)
        total = (
            cached * (cached_price if cached_price is not None else base)
            + (usage.input_tokens - cached) * ordinary_price
        )
    elif written is not None:
        ordinary_price = max(base, cached_price if cached_price is not None else base)
        total = (
            written * (write_price if write_price is not None else base)
            + (usage.input_tokens - written) * ordinary_price
        )
    else:
        total = usage.input_tokens * max(
            base,
            cached_price if cached_price is not None else base,
            write_price if write_price is not None else base,
        )
    return total / 1_000_000


def _requires_tool_protocol(request: ModelRequest) -> bool:
    """Return whether preserving this request requires structured tool support."""
    return bool(
        request.tools
        or request.tool_choice is not None
        or any(
            message.role == "tool"
            or (message.assistant_action is not None and bool(message.assistant_action.tool_calls))
            for message in request.messages
        )
    )


def _decision_content_id(decision: RoutingDecision) -> str:
    """Return the canonical content identity for a routing decision."""
    material = decision.model_dump(mode="json")
    del material["decision_id"]
    return stable_id("routing-decision", material)


def _validate_idempotency_key(value: str) -> None:
    """Reject keys that cannot safely cross an HTTP provider boundary."""
    if not value or len(value) > 512 or value.strip() != value:
        raise ValueError("idempotency key must be 1 to 512 non-blank characters")
    if any(ord(character) < 33 or ord(character) > 126 for character in value):
        raise ValueError("idempotency key must contain only visible ASCII characters")


def _supports_request(resolved: ResolvedModel, request: ModelRequest) -> bool:
    """Check whether a resolved model proves every requested protocol capability.

    Args:
        resolved: Frozen runtime model and its declared capabilities.
        request: Provider-neutral request to evaluate.

    Returns:
        True when tool and output-token requirements are both proven.
    """
    if _requires_tool_protocol(request) and not resolved.capabilities.supports_tools:
        return False
    requested = request.maximum_output_tokens
    available = resolved.capabilities.maximum_output_tokens
    return requested is None or (available is not None and requested <= available)
