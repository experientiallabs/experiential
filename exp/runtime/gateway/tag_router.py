"""Tag-rules routing: a decision model's category flags choose the serving candidate.

A tag-rules router policy names its candidate aliases, pinned to their model
identities like every router policy, one decision-model classifier (the
``decision_model`` document classification guardrails use), ordered rules and a
default candidate. Before dispatch the classifier reads its window of the request
once; the first rule whose ``when_all`` categories all reached their thresholds
and whose ``when_none`` categories all stayed below them names the candidate.
No match routes to the default, and so does a classifier that fails, times out,
or finds no text, so routing never fails a request; only an expired request
deadline does. A selected candidate whose declared capabilities rule out the
request (tools, or its output-token limit) yields to the default or the first
eligible candidate, as in learned routing. Like learned routing, the first
decision of an episode serves its later turns while the episode stays active; a
decision made without a classifier answer is never pinned to it.
Tag-rules policies are stored as their own artifact type, so commands that read
learned router policies never see them.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ContractModel,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.models import ModelAlias, ModelCapabilities, RoutedCandidateSnapshot
from exp.common.models.capability_fit import capabilities_support
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
from exp.runtime.models.providers.async_transport import (
    ProviderDeadlineExceeded,
    RequestDeadline,
)
from exp.runtime.openai_protocol.model_adapter import model_request

TAG_RULES_POLICY_ARTIFACT_TYPE: Final = "tag-router-policy"
TAG_ROUTER_ROUTE_REASON: Final = "tag_router"
NO_RULE_MATCHED: Final = "no_rule_matched"
CLASSIFIER_UNAVAILABLE: Final = "classifier_unavailable"
NO_CLASSIFIABLE_TEXT: Final = "no_classifiable_text"
CAPABILITY_ELIGIBILITY: Final = "capability_eligibility"
CLASSIFIER_BUSY: Final = "classifier_busy"
_UNPINNED: Final = frozenset({CLASSIFIER_UNAVAILABLE, CLASSIFIER_BUSY, NO_CLASSIFIABLE_TEXT})
_MAX_RULES: Final = 64
_PENDING_DECISION_ID: Final = "routing-decision-pending"
# Every selector in the process takes permits from this one pool, so the bound on
# decision-model calls in flight covers all activations and reloaded generations.
_CLASSIFICATIONS: Final = threading.BoundedSemaphore(32)

type TagDecisionSink = Callable[[RoutingDecision], None]
"""Recorder for every tag routing decision, with the learned router's sink contract."""


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
        policy_id: Identity derived from the candidates, default, classifier and rules.
        candidates: Candidate aliases with their frozen model identities.
        default_alias: Candidate serving requests no rule matches.
        classifier: Decision-model document scoring the request before dispatch. It
            names no credential and its URL has no query string, so the endpoint must
            accept requests without a credential.
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
        if self.classifier.bearer_env is not None:
            raise ValueError("a tag-rules policy names no credential; omit classifier.bearer_env")
        if urlsplit(self.classifier.url).query:
            raise ValueError("a tag-rules classifier URL carries no query string")
        categories = set(self.classifier.categories)
        for rule in self.rules:
            if rule.route not in aliases:
                raise ValueError("every tag rule must route to a candidate")
            if any(name not in categories for name in (*rule.when_all, *rule.when_none)):
                raise ValueError("tag rules may name only the classifier's categories")
            if set(rule.when_all) & set(rule.when_none):
                raise ValueError("a tag rule cannot require and forbid the same category")
        expected = tag_rules_policy_id(
            candidates=self.candidates,
            default_alias=self.default_alias,
            classifier=self.classifier,
            rules=self.rules,
        )
        if self.policy_id != expected:
            raise ValueError("tag router policy_id differs from its content identity")
        return self


def tag_rules_policy_id(
    *,
    candidates: Sequence[RoutedCandidateSnapshot],
    default_alias: str,
    classifier: DecisionModelAdapterDocument,
    rules: Sequence[TagRule],
) -> ArtifactId:
    """Return the identity of a tag-rules policy's routing content.

    It covers the pinned candidates, the default, the classifier and the rules,
    not provenance such as the authoring time, so equal content is one policy.
    """
    return stable_id(
        "tag-router-policy",
        {
            "version": "tag-rules-policy-v1",
            "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
            "default_alias": default_alias,
            "classifier": classifier.model_dump(mode="json"),
            "rules": [rule.model_dump(mode="json") for rule in rules],
        },
    )


@dataclass(frozen=True)
class TagRulesActivation:
    """Verified immutable material for one tag-rules exact-model selector.

    Attributes:
        project_ref: Project that stores the policy.
        activation_ref: Activated policy identity, equal to ``policy.policy_id``.
        policy: The frozen tag-rules policy.
    """

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
        decision_sink: TagDecisionSink | None = None,
        classifications: threading.BoundedSemaphore = _CLASSIFICATIONS,
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
            decision_sink: Optional recorder called with every decision before it is used.
            classifications: Permits for decision-model calls in flight. The default
                pool of 32 is shared by every selector in the process; a request
                beyond it takes the default unclassified.
            decision_capacity: Most retained episode decisions, default 4,096.
            decision_ttl_seconds: Seconds an idle episode's decision stays valid, default one day.
            clock: Monotonic clock for decision expiry.

        Raises:
            ValueError: A decision bound is not positive.
            TagRouterIntegrityError: A candidate does not resolve to its frozen identity.
        """
        if decision_capacity < 1 or decision_ttl_seconds <= 0:
            raise ValueError("tag router decision bounds must be positive")
        self._capabilities: dict[str, ModelCapabilities] = {}
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
            self._capabilities[candidate.alias] = capabilities
        self._restricted = any(
            item.supports_tools is False or item.maximum_output_tokens is not None
            for item in self._capabilities.values()
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
        self._decision_sink = decision_sink
        self._classifications = classifications
        self._capacity = decision_capacity
        self._ttl_seconds = decision_ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._episodes: OrderedDict[Sha256, tuple[float, RoutingDecision]] = OrderedDict()

    @classmethod
    def from_activation(
        cls,
        activation: TagRulesActivation,
        catalog: RuntimeModelCatalog,
        *,
        decision_sink: TagDecisionSink | None = None,
    ) -> TagRulesSelector:
        """Bind one verified activation to runtime-owned model resolution."""
        return cls(activation.policy, catalog, decision_sink=decision_sink)

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
        reused = self._reuse(request, keys)
        if reused is not None:
            return reused
        flagged, reason = await self._bounded_classify(request, deadline)
        deadline.attempt_timeout()
        return self._publish(request, keys, self._decision(keys, flagged, reason))

    def select_blocking(
        self, request: GatewayRequest, *, episode_id: str, deadline: RequestDeadline
    ) -> RoutingDecision:
        """Run :meth:`select` from a thread that owns no event loop.

        The classifier runs on the shared native-callback loop, the same one
        guardrail inspection uses, so the calling bridge thread only waits.
        """
        keys = self._keys(request, episode_id)
        reused = self._reuse(request, keys)
        if reused is not None:
            return reused
        flagged, reason = run_on_native_loop(self._bounded_classify(request, deadline))
        deadline.attempt_timeout()
        return self._publish(request, keys, self._decision(keys, flagged, reason))

    def _eligible(self, request: GatewayRequest, decision: RoutingDecision) -> RoutingDecision:
        """Keep the choice, or move this request to a candidate that can serve it.

        Only an explicit declaration rules a candidate out. The default is preferred,
        then candidate order; when no candidate can serve the request the choice
        stands and admission refuses it.
        """
        if not self._restricted:
            return decision
        try:
            provider_request = model_request(request)
        except ValueError:
            # A shape the provider projection cannot express is left to admission;
            # routing itself never fails the request.
            return decision
        if capabilities_support(self._capabilities[decision.selected_alias], provider_request):
            return decision
        eligible = [
            candidate.alias
            for candidate in self.policy.candidates
            if capabilities_support(self._capabilities[candidate.alias], provider_request)
        ]
        if not eligible:
            return decision
        alias = self.policy.default_alias if self.policy.default_alias in eligible else eligible[0]
        return _content_addressed(
            decision.model_copy(
                update={
                    "decision_id": _PENDING_DECISION_ID,
                    "selected_alias": alias,
                    "fallback_reason": CAPABILITY_ELIGIBILITY,
                }
            )
        )

    def _record(self, decision: RoutingDecision) -> RoutingDecision:
        """Send one decision to the sink before it is used, as learned routing does."""
        if self._decision_sink is not None:
            self._decision_sink(decision)
        return decision

    def _keys(self, request: GatewayRequest, episode_id: str) -> tuple[Sha256, Sha256]:
        """Return the episode and request digests: the request is the classified window."""
        if not episode_id.strip() or len(episode_id) > 512:
            raise ValueError("episode_id must be 1 to 512 non-blank characters")
        window = classification_window(request, self.policy.classifier.window)
        return (
            hashlib.sha256(episode_id.encode("utf-8")).hexdigest(),
            hashlib.sha256(window.encode("utf-8")).hexdigest(),
        )

    async def _bounded_classify(
        self, request: GatewayRequest, deadline: RequestDeadline
    ) -> tuple[frozenset[str], str | None]:
        """Classify within the in-flight bound; a request beyond it is not classified.

        Shedding instead of queueing keeps bursts from holding admission threads
        or overloading the decision model; such a request takes the default unpinned.
        """
        if not self._classifications.acquire(blocking=False):
            return frozenset(), CLASSIFIER_BUSY
        try:
            return await self._classify(request, deadline)
        finally:
            self._classifications.release()

    async def _classify(
        self, request: GatewayRequest, deadline: RequestDeadline
    ) -> tuple[frozenset[str], str | None]:
        """Return the flagged categories, or none with the reason routing falls back.

        Raises:
            ProviderDeadlineExceeded: The request deadline expired before the call.
        """
        classifier_seconds = self.policy.classifier.timeout_ms / 1000.0
        budget = deadline.attempt_timeout(classifier_seconds)
        try:
            async with asyncio.timeout(budget):
                verdict = await self._classifier.inspect_input(request=request, check=self._check)
        except TimeoutError:
            # A budget cut short by the request's deadline fails the request, as
            # learned routing does; only the classifier's own timeout falls back.
            if budget < classifier_seconds:
                raise ProviderDeadlineExceeded("tag router selection deadline exceeded") from None
            return frozenset(), CLASSIFIER_UNAVAILABLE
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

    def _reuse(
        self, request: GatewayRequest, keys: tuple[Sha256, Sha256]
    ) -> RoutingDecision | None:
        """Serve the episode's choice to this request, or ``None`` when it has none."""
        with self._lock:
            episode = self._hit(keys[0])
            return None if episode is None else self._serve_pinned(request, keys, episode)

    def _publish(
        self, request: GatewayRequest, keys: tuple[Sha256, Sha256], decision: RoutingDecision
    ) -> RoutingDecision:
        """Record and pin a fresh decision unless the episode already chose.

        The decision reaches the sink before it is pinned, so a recording failure
        pins nothing. A concurrent request's earlier choice for the episode wins.
        A decision made without a classifier answer (failed, timed out, or shed at
        the in-flight bound), or for a request with no classifiable text, is never
        pinned, so a later request asks again instead of keeping the default.
        """
        used = self._eligible(request, decision)
        with self._lock:
            episode = self._hit(keys[0])
            if episode is not None:
                return self._serve_pinned(request, keys, episode)
            self._record(used)
            if decision.fallback_reason not in _UNPINNED:
                # Pin what was served, so later turns stay on the model that began
                # the episode even when a capability fallback chose it.
                self._pin(keys[0], used)
            return used

    def _serve_pinned(
        self, request: GatewayRequest, keys: tuple[Sha256, Sha256], episode: RoutingDecision
    ) -> RoutingDecision:
        """Serve an episode's pinned choice, repinning when this request needs another.

        A capability fallback replaces the pin, as learned routing does, so an
        episode never alternates models as its requests' needs change. Callers
        hold the lock.
        """
        used = self._record(self._eligible(request, _for_request(episode, keys[1])))
        if used.selected_alias != episode.selected_alias:
            self._pin(keys[0], used)
        return used

    def _pin(self, episode_sha256: Sha256, decision: RoutingDecision) -> None:
        """Retain an episode's decision with a fresh lifetime.

        Expired episodes, and the least recent beyond capacity, are evicted.
        Callers hold the lock.
        """
        now = self._clock()
        self._episodes[episode_sha256] = (now + self._ttl_seconds, decision)
        self._episodes.move_to_end(episode_sha256)
        while self._episodes and (
            len(self._episodes) > self._capacity or next(iter(self._episodes.values()))[0] <= now
        ):
            self._episodes.popitem(last=False)

    def _hit(self, episode_sha256: Sha256) -> RoutingDecision | None:
        """Return an episode's unexpired decision and renew it, or drop an expired one.

        A hit moves to the newest end with a fresh lifetime, so an active episode
        outlives idle ones and insertion order stays expiry order. Callers hold the lock.
        """
        entry = self._episodes.get(episode_sha256)
        if entry is None:
            return None
        now = self._clock()
        if entry[0] <= now:
            del self._episodes[episode_sha256]
            return None
        self._episodes.move_to_end(episode_sha256)
        self._episodes[episode_sha256] = (now + self._ttl_seconds, entry[1])
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
