"""Tag-rules routing: a decision model's category flags choose the serving candidate.

A tag-rules router policy names its candidate aliases, pinned to their model
identities like every router policy, one decision-model classifier (the
``decision_model`` document classification guardrails use), ordered rules and a
default candidate. Before dispatch the classifier reads its window of the request
once; the first rule whose ``when_all`` categories all reached their thresholds
and whose ``when_none`` categories all stayed below them names the candidate.
No match routes to the default, and so does a classifier that fails, times out,
or finds no text, so routing never fails a request. Like learned routing, the
first decision of an episode serves its later turns.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ContractModel,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.models import ModelAlias, RoutedCandidateSnapshot
from exp.common.routing.policy import RoutingDecision
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.guardrails.bounded import run_on_native_loop
from exp.runtime.gateway.guardrails.classification import (
    DecisionModelAdapterDocument,
    DecisionModelClassifier,
)
from exp.runtime.gateway.guardrails.classification_window import classification_window
from exp.runtime.gateway.guardrails.contracts import (
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    ScoredVerdict,
)
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.async_transport import RequestDeadline

TAG_ROUTER_ROUTE_REASON: Final = "tag_router"
NO_RULE_MATCHED: Final = "no_rule_matched"
CLASSIFIER_UNAVAILABLE: Final = "classifier_unavailable"
NO_CLASSIFIABLE_TEXT: Final = "no_classifiable_text"
_MAX_RULES: Final = 64
_PENDING_DECISION_ID: Final = "routing-decision-pending"


class TagRouterIntegrityError(ValueError):
    """A tag-rules policy cannot activate against the runtime model catalog."""


class TagRule(ContractModel):
    """One ordered routing rule over a decision model's category flags.

    Attributes:
        when_all: Categories that must all reach their thresholds.
        when_none: Categories that must all stay below their thresholds, none by default.
        route: Candidate alias that serves a matching request.
    """

    when_all: tuple[ArtifactId, ...] = Field(min_length=1)
    when_none: tuple[ArtifactId, ...] = ()
    route: ModelAlias

    def matches(self, flagged: frozenset[str]) -> bool:
        """Return whether this rule holds for one request's flagged categories."""
        return all(category in flagged for category in self.when_all) and not any(
            category in flagged for category in self.when_none
        )


class TagRulesRouterPolicy(ArtifactEnvelope):
    """A frozen tag-rules router policy: candidates, one classifier, rules, a default.

    Attributes:
        kind: Always ``tag_rules``.
        policy_id: Content-addressed router-policy identity.
        candidates: Candidate aliases with their frozen model identities.
        default_alias: Candidate serving requests no rule matches.
        classifier: Decision-model document scoring the request before dispatch.
        rules: Ordered rules; the first match routes the request.
    """

    kind: Literal["tag_rules"] = "tag_rules"
    policy_id: ArtifactId
    candidates: tuple[RoutedCandidateSnapshot, ...] = Field(min_length=1)
    default_alias: ModelAlias
    classifier: DecisionModelAdapterDocument
    rules: tuple[TagRule, ...] = Field(min_length=1, max_length=_MAX_RULES)

    @model_validator(mode="after")
    def _require_consistent_rules(self) -> TagRulesRouterPolicy:
        """Require unique candidates and rules that name only candidates and categories.

        Returns:
            The validated policy.

        Raises:
            ValueError: An alias repeats, the default or a rule's route is not a
                candidate, or a rule names an unknown or contradictory category.
        """
        aliases = tuple(candidate.alias for candidate in self.candidates)
        if len(set(aliases)) != len(aliases):
            raise ValueError("tag router candidate aliases must be unique")
        if self.default_alias not in aliases:
            raise ValueError("tag router default_alias must name a candidate")
        categories = set(self.classifier.categories)
        for rule in self.rules:
            if rule.route not in aliases:
                raise ValueError("every tag rule must route to a candidate")
            if any(name not in categories for name in (*rule.when_all, *rule.when_none)):
                raise ValueError("tag rules may name only the classifier's categories")
            if set(rule.when_all) & set(rule.when_none):
                raise ValueError("a tag rule cannot require and forbid the same category")
        return self


@dataclass(frozen=True)
class TagRulesActivation:
    """Verified immutable material for one tag-rules exact-model selector."""

    project_ref: str
    activation_ref: ArtifactId
    policy: TagRulesRouterPolicy

    def __post_init__(self) -> None:
        """Reject identifiers that differ from the frozen policy."""
        if not self.project_ref.strip():
            raise ValueError("project activation requires a non-empty project reference")
        if self.activation_ref != self.policy.policy_id:
            raise ValueError("project activation reference differs from its frozen policy")

    @property
    def candidate_aliases(self) -> tuple[str, ...]:
        """Return the ordered logical aliases the rules may select."""
        return tuple(candidate.alias for candidate in self.policy.candidates)


class TagRulesSelector:
    """Choose one candidate per episode from a decision model's category flags.

    The selector is thread-safe: the blocking native admission path and the
    event-loop path share one instance and its sticky episode state. The
    classifier call runs outside the lock under the tighter of the classifier
    timeout and the request deadline.
    """

    def __init__(
        self,
        policy: TagRulesRouterPolicy,
        catalog: RuntimeModelCatalog,
        *,
        classifier: DecisionModelClassifier | None = None,
        decision_capacity: int = 4_096,
        decision_ttl_seconds: float = 24 * 60 * 60,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Verify every candidate's frozen identity and bind the classifier.

        Identity comes from the catalog's static snapshot: the gateway dispatches
        the candidates, so no provider client is built here.

        Args:
            policy: Frozen tag-rules policy.
            catalog: Runtime catalog resolving the candidate aliases.
            classifier: Optional injected classifier; ``None`` builds one from the policy.
            decision_capacity: Most retained episode and request decisions, default 4,096.
            decision_ttl_seconds: Seconds a retained decision stays valid, default one day.
            clock: Monotonic clock for decision expiry.

        Raises:
            ValueError: A decision bound is not positive.
            TagRouterIntegrityError: A candidate does not resolve to its frozen identity.
        """
        if decision_capacity < 1 or decision_ttl_seconds <= 0:
            raise ValueError("tag router decision bounds must be positive")
        for candidate in policy.candidates:
            try:
                snapshot, capabilities = catalog.snapshot(candidate.alias)
            except Exception as exc:  # noqa: BLE001 - normalize catalog lookup errors
                raise TagRouterIntegrityError(
                    "runtime model catalog cannot resolve tag router candidates"
                ) from exc
            if (
                snapshot != candidate.model
                or capabilities.identity_sha256() != candidate.model.capabilities_sha256
            ):
                raise TagRouterIntegrityError(
                    f"runtime alias {candidate.alias!r} differs from its frozen identity"
                )
        self.policy = policy
        self._classifier = classifier or DecisionModelClassifier(policy.classifier)
        self._check = GuardrailCheck(
            check_id="tag-router",
            capability=GuardrailCapabilityKind.CONTENT_SAFETY,
            stage=GuardrailCheckStage.INPUT,
            action=GuardrailAction.ALLOW,
            timeout_ms=policy.classifier.timeout_ms,
            adapter_id=policy.classifier.adapter_id,
        )
        self._policy_sha256 = sha256_json(policy)
        self._capacity = decision_capacity
        self._ttl_seconds = decision_ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._episodes: OrderedDict[Sha256, tuple[float, RoutingDecision]] = OrderedDict()
        self._requests: OrderedDict[tuple[Sha256, Sha256], tuple[float, RoutingDecision]] = (
            OrderedDict()
        )

    @classmethod
    def from_activation(
        cls, activation: TagRulesActivation, catalog: RuntimeModelCatalog
    ) -> TagRulesSelector:
        """Bind one verified activation to runtime-owned model resolution."""
        return cls(activation.policy, catalog)

    async def select(
        self, request: GatewayRequest, *, episode_id: str, deadline: RequestDeadline
    ) -> RoutingDecision:
        """Return the episode's decision, classifying the request when it has none.

        Args:
            request: Canonical gateway request before dispatch.
            episode_id: Tenant-isolated sticky episode identity.
            deadline: Request-wide deadline that also bounds the classifier.

        Returns:
            The retained decision for this request.

        Raises:
            ProviderDeadlineExceeded: The request deadline expired.
        """
        keys = self._keys(request, episode_id)
        reused = self._reuse(keys)
        if reused is not None:
            return reused
        flagged, reason = await self._classify(request, deadline)
        return self._retain(keys, self._decision(keys, flagged, reason))

    def select_blocking(
        self, request: GatewayRequest, *, episode_id: str, deadline: RequestDeadline
    ) -> RoutingDecision:
        """Run :meth:`select` from a thread that owns no event loop.

        The classifier runs on the shared native-callback loop, the same one
        guardrail inspection uses, so the calling bridge thread only waits.
        """
        keys = self._keys(request, episode_id)
        reused = self._reuse(keys)
        if reused is not None:
            return reused
        flagged, reason = run_on_native_loop(self._classify(request, deadline))
        return self._retain(keys, self._decision(keys, flagged, reason))

    def _keys(self, request: GatewayRequest, episode_id: str) -> tuple[Sha256, Sha256]:
        """Return the episode and request digests: the request is the classified window."""
        if not episode_id.strip() or len(episode_id) > 512:
            raise ValueError("episode_id must be 1 to 512 non-blank characters")
        window = classification_window(request, self.policy.classifier.window)
        return (
            hashlib.sha256(episode_id.encode("utf-8")).hexdigest(),
            hashlib.sha256(window.encode("utf-8")).hexdigest(),
        )

    async def _classify(
        self, request: GatewayRequest, deadline: RequestDeadline
    ) -> tuple[frozenset[str], str | None]:
        """Return the flagged categories, or none with the reason routing falls back.

        Raises:
            ProviderDeadlineExceeded: The request deadline expired before the call.
        """
        timeout = deadline.attempt_timeout(self.policy.classifier.timeout_ms / 1000.0)
        try:
            async with asyncio.timeout(timeout):
                verdict = await self._classifier.inspect_input(request=request, check=self._check)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - routing falls back; it never fails the request.
            return frozenset(), CLASSIFIER_UNAVAILABLE
        if not isinstance(verdict, ScoredVerdict):
            return frozenset(), NO_CLASSIFIABLE_TEXT
        return frozenset(score.category for score in verdict.scores if score.flagged), None

    def _decision(
        self,
        keys: tuple[Sha256, Sha256],
        flagged: frozenset[str],
        reason: str | None,
    ) -> RoutingDecision:
        """Apply the first matching rule, else the default, as one content-addressed decision."""
        selected = self.policy.default_alias
        if reason is None:
            rule = next((rule for rule in self.policy.rules if rule.matches(flagged)), None)
            if rule is None:
                reason = NO_RULE_MATCHED
            else:
                selected = rule.route
        episode_sha256, request_sha256 = keys
        return _content_addressed(
            RoutingDecision(
                decision_id=_PENDING_DECISION_ID,
                policy_id=self.policy.policy_id,
                policy_sha256=self._policy_sha256,
                request_sha256=request_sha256,
                episode_id_sha256=episode_sha256,
                selected_alias=selected,
                baseline_alias=self.policy.default_alias,
                neighbor_count=0,
                paired_count=0,
                fallback_reason=reason,
            )
        )

    def _reuse(self, keys: tuple[Sha256, Sha256]) -> RoutingDecision | None:
        """Return this request's decision, or the episode's choice bound to this request."""
        with self._lock:
            cached = _live(self._requests, keys, self._clock())
            if cached is not None:
                return cached
            episode = _live(self._episodes, keys[0], self._clock())
            if episode is None:
                return None
            return self._store(self._requests, keys, _for_request(episode, keys[1]))

    def _retain(self, keys: tuple[Sha256, Sha256], decision: RoutingDecision) -> RoutingDecision:
        """Publish a fresh decision unless a concurrent request already chose for the episode."""
        with self._lock:
            now = self._clock()
            cached = _live(self._requests, keys, now)
            if cached is not None:
                return cached
            episode = _live(self._episodes, keys[0], now)
            if episode is None:
                self._store(self._episodes, keys[0], decision)
            else:
                decision = _for_request(episode, keys[1])
            return self._store(self._requests, keys, decision)

    def _store[KeyT](
        self,
        store: OrderedDict[KeyT, tuple[float, RoutingDecision]],
        key: KeyT,
        decision: RoutingDecision,
    ) -> RoutingDecision:
        """Retain one decision in insertion order under the TTL and capacity bounds.

        Every entry lives exactly ``decision_ttl_seconds``, so insertion order is
        expiry order and trimming from the oldest end is enough. Callers hold the lock.
        """
        now = self._clock()
        store[key] = (now + self._ttl_seconds, decision)
        while store and (len(store) > self._capacity or next(iter(store.values()))[0] <= now):
            store.popitem(last=False)
        return decision


def _live[KeyT](
    store: OrderedDict[KeyT, tuple[float, RoutingDecision]], key: KeyT, now: float
) -> RoutingDecision | None:
    """Return an unexpired retained decision, dropping an expired one."""
    entry = store.get(key)
    if entry is None:
        return None
    if entry[0] <= now:
        del store[key]
        return None
    return entry[1]


def _for_request(decision: RoutingDecision, request_sha256: Sha256) -> RoutingDecision:
    """Bind an episode's earlier choice to a later request of the same episode."""
    return _content_addressed(
        decision.model_copy(
            update={"decision_id": _PENDING_DECISION_ID, "request_sha256": request_sha256}
        )
    )


def _content_addressed(decision: RoutingDecision) -> RoutingDecision:
    """Replace the provisional identifier with the decision's content identity."""
    material = decision.model_dump(mode="json")
    del material["decision_id"]
    return decision.model_copy(update={"decision_id": stable_id("routing-decision", material)})
