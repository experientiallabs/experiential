"""Tests for tag-rules routing: policy rules, fallbacks, stickiness, loading, resolution."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue, canonical_json_bytes
from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelRoles,
    RoutedCandidateSnapshot,
)
from exp.common.project import ArtifactStore, ProjectPaths
from exp.common.routing.policy import RoutingDecision
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
    ProjectTarget,
)
from exp.runtime.gateway.guardrails.classification import (
    DecisionModelAdapterDocument,
    DecisionModelClassifier,
)
from exp.runtime.gateway.guardrails.contracts import (
    CategoryScore,
    ClassifierVerdict,
    GuardrailCheck,
    ScoredVerdict,
)
from exp.runtime.gateway.project_activation import load_project_activation
from exp.runtime.gateway.routing import RouterProjectTargetResolver
from exp.runtime.gateway.tag_router import (
    CAPABILITY_ELIGIBILITY,
    CLASSIFIER_BUSY,
    CLASSIFIER_UNAVAILABLE,
    NO_CLASSIFIABLE_TEXT,
    NO_RULE_MATCHED,
    TAG_ROUTER_ROUTE_REASON,
    TAG_RULES_POLICY_ARTIFACT_TYPE,
    TagRouterIntegrityError,
    TagRule,
    TagRulesActivation,
    TagRulesRouterPolicy,
    TagRulesSelector,
    tag_rules_policy_id,
)
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.async_transport import ProviderDeadlineExceeded, RequestDeadline

_ALIASES = ("coder", "secure", "generalist")
_THRESHOLDS = {"coding": 0.7, "cyber": 0.8}


def _catalog(*, without_tools: tuple[str, ...] = ()) -> RuntimeModelCatalog:
    """Return a credential-free runtime catalog with three candidates."""

    def capabilities(alias: str) -> ModelCapabilities:
        """Declare tools for every candidate except those named."""
        return ModelCapabilities(
            supports_completions=True,
            supports_tools=alias not in without_tools,
            supports_structured_output=True,
            context_window_tokens=128_000,
            maximum_output_tokens=16_000,
        )

    return RuntimeModelCatalog(
        ModelCatalog(
            connections={
                "provider": ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY")
            },
            models={
                alias: ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="provider",
                    model=f"model-{alias}",
                    capabilities=capabilities(alias),
                )
                for alias in _ALIASES
            },
            roles=ModelRoles(candidates=_ALIASES, incumbent="generalist"),
        ),
        environment={},
    )


def _document(catalog: RuntimeModelCatalog, **overrides: JsonValue) -> JsonObject:
    """Return a policy routing cyber work to ``secure`` and other coding to ``coder``."""
    document: JsonObject = {
        "schema_version": 1,
        "created_at": datetime(2026, 11, 1, tzinfo=UTC).isoformat(),
        "code_revision": "test",
        "candidates": [
            RoutedCandidateSnapshot(alias=alias, model=catalog.snapshot(alias)[0]).model_dump(
                mode="json"
            )
            for alias in _ALIASES
        ],
        "default_alias": "generalist",
        "classifier": {
            "adapter_id": "task-tags",
            "kind": "decision_model",
            "url": "https://decisions.example.invalid/systemone",
            "model": "Cloudflare/clef",
            "categories": {
                name: {"threshold": threshold, "questions": {f"{name}_q": {"instructions": name}}}
                for name, threshold in _THRESHOLDS.items()
            },
        },
        "rules": [
            {"when_all": ["cyber"], "route": "secure"},
            {"when_all": ["coding"], "when_none": ["cyber"], "route": "coder"},
        ],
    }
    document.update(overrides)
    return _identified(document)


def _identified(document: JsonObject) -> JsonObject:
    """Return the document carrying the content identity of its current fields."""
    rules = cast("list[JsonObject]", document["rules"])
    candidates = cast("list[JsonObject]", document["candidates"])
    policy_id = tag_rules_policy_id(
        candidates=[RoutedCandidateSnapshot.model_validate(item) for item in candidates],
        default_alias=str(document["default_alias"]),
        classifier=DecisionModelAdapterDocument.model_validate(document["classifier"]),
        rules=[TagRule.model_validate(rule) for rule in rules],
    )
    return {**document, "policy_id": policy_id}


def _policy(catalog: RuntimeModelCatalog) -> TagRulesRouterPolicy:
    """Return the validated test policy."""
    return TagRulesRouterPolicy.model_validate(_document(catalog))


class _ScriptedClassifier:
    """Answer each inspection from a queue of scores, failures, or text-free verdicts."""

    def __init__(
        self,
        *answers: dict[str, float] | Exception | None,
        delay: float = 0.0,
        delays: tuple[float, ...] = (),
    ) -> None:
        """Queue answers in call order, each given after its delay (``delay`` by default)."""
        self._answers = list(answers)
        self._delay = delay
        self._delays = list(delays)
        self.calls = 0

    async def inspect_input(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Return the next scripted verdict."""
        del request, check
        self.calls += 1
        answer = self._answers.pop(0)
        await asyncio.sleep(self._delays.pop(0) if self._delays else self._delay)
        if isinstance(answer, Exception):
            raise answer
        if answer is None:
            return ClassifierVerdict(flagged=False)
        scores = tuple(
            CategoryScore(category=name, score=value, flagged=value >= _THRESHOLDS[name])
            for name, value in answer.items()
        )
        return ScoredVerdict(
            flagged=any(score.flagged for score in scores),
            scores=scores,
            subject_sha256="d" * 64,
            cached=False,
        )


def _selector(
    *answers: dict[str, float] | Exception | None,
    delay: float = 0.0,
    delays: tuple[float, ...] = (),
    capacity: int = 4_096,
    without_tools: tuple[str, ...] = (),
) -> tuple[TagRulesSelector, _ScriptedClassifier]:
    """Return a selector over the test policy and its scripted classifier."""
    catalog = _catalog(without_tools=without_tools)
    classifier = _ScriptedClassifier(*answers, delay=delay, delays=delays)
    selector = TagRulesSelector(
        _policy(catalog),
        catalog,
        classifier=cast("DecisionModelClassifier", classifier),
        decision_capacity=capacity,
    )
    return selector, classifier


def _request(text: str) -> GatewayRequest:
    """Return one chat request whose latest user message is ``text``."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content=text),),
    )


def _deadline() -> RequestDeadline:
    """Return a deadline five seconds out."""
    return RequestDeadline(time.monotonic() + 5)


@pytest.mark.parametrize(
    ("scores", "selected", "reason"),
    [
        ({"coding": 0.9, "cyber": 0.85}, "secure", None),
        ({"coding": 0.9, "cyber": 0.1}, "coder", None),
        ({"coding": 0.2, "cyber": 0.1}, "generalist", NO_RULE_MATCHED),
    ],
)
def test_the_first_matching_rule_routes_and_no_match_takes_the_default(
    scores: dict[str, float], selected: str, reason: str | None
) -> None:
    """Rules apply in order; ``when_none`` keeps cyber work off the coding rule."""
    selector, _ = _selector(scores)
    decision = selector.select_blocking(_request("hi"), episode_id="episode", deadline=_deadline())
    assert (decision.selected_alias, decision.fallback_reason) == (selected, reason)
    assert decision.baseline_alias == "generalist"


@pytest.mark.parametrize(
    ("answer", "reason"),
    [(RuntimeError("down"), CLASSIFIER_UNAVAILABLE), (None, NO_CLASSIFIABLE_TEXT)],
)
def test_a_classifier_without_an_answer_routes_to_the_default(
    answer: Exception | None, reason: str
) -> None:
    """Routing never fails a request: a failed or text-free check takes the default."""
    selector, _ = _selector(answer)
    decision = asyncio.run(
        selector.select(_request("hi"), episode_id="episode", deadline=_deadline())
    )
    assert (decision.selected_alias, decision.fallback_reason) == ("generalist", reason)


def test_an_episode_keeps_its_first_choice_without_asking_again() -> None:
    """A later turn reuses the episode's alias; a new episode classifies afresh."""
    selector, classifier = _selector({"coding": 0.9, "cyber": 0.1}, {"coding": 0.1, "cyber": 0.9})
    first = selector.select_blocking(_request("fix"), episode_id="one", deadline=_deadline())
    later = selector.select_blocking(_request("now"), episode_id="one", deadline=_deadline())
    other = selector.select_blocking(_request("now"), episode_id="two", deadline=_deadline())
    assert (first.selected_alias, later.selected_alias, other.selected_alias) == (
        "coder",
        "coder",
        "secure",
    )
    assert later.request_sha256 != first.request_sha256
    assert later.decision_id != first.decision_id
    assert classifier.calls == 2


def test_an_active_episode_outlives_idle_ones_at_capacity() -> None:
    """Reuse renews an episode, so other traffic evicts idle episodes first."""
    selector, classifier = _selector(
        {"coding": 0.9, "cyber": 0.1},
        {"coding": 0.1, "cyber": 0.9},
        {"coding": 0.1, "cyber": 0.9},
        capacity=2,
    )
    turns = [("a", "fix"), ("b", "hi"), ("a", "more"), ("c", "hi"), ("a", "again")]
    selected = [
        selector.select_blocking(_request(text), episode_id=episode, deadline=_deadline())
        for episode, text in turns
    ]
    assert [decision.selected_alias for decision in selected] == [
        "coder",
        "secure",
        "coder",
        "secure",
        "coder",
    ]
    assert classifier.calls == 3


def test_a_failed_classifier_does_not_pin_the_default_to_the_episode() -> None:
    """The next turn asks again instead of keeping the fallback for the conversation."""
    selector, classifier = _selector(RuntimeError("down"), {"coding": 0.9, "cyber": 0.1})
    first = selector.select_blocking(_request("fix"), episode_id="one", deadline=_deadline())
    later = selector.select_blocking(_request("fix"), episode_id="one", deadline=_deadline())
    assert (first.selected_alias, first.fallback_reason) == ("generalist", CLASSIFIER_UNAVAILABLE)
    assert (later.selected_alias, later.fallback_reason) == ("coder", None)
    assert classifier.calls == 2


def test_a_deadline_that_ends_during_classification_fails_and_keeps_nothing() -> None:
    """Running out of request time is a request failure, not a routing fallback."""
    selector, classifier = _selector({"coding": 0.9}, {"coding": 0.9, "cyber": 0.1}, delay=0.3)
    with pytest.raises(ProviderDeadlineExceeded):
        selector.select_blocking(
            _request("fix"), episode_id="one", deadline=RequestDeadline(time.monotonic() + 0.1)
        )
    later = selector.select_blocking(_request("fix"), episode_id="one", deadline=_deadline())
    assert (later.selected_alias, classifier.calls) == ("coder", 2)


def test_a_candidate_that_cannot_serve_the_request_yields_to_an_eligible_one() -> None:
    """A tool request never lands on a candidate that declares no tools."""
    selector, _ = _selector({"coding": 0.9, "cyber": 0.1}, without_tools=("coder",))
    request = _request("fix").model_copy(
        update={"tools": (GatewayToolDefinition(name="run_tests", parameters={"type": "object"}),)}
    )
    decision = selector.select_blocking(request, episode_id="one", deadline=_deadline())
    assert (decision.selected_alias, decision.fallback_reason) == (
        "generalist",
        CAPABILITY_ELIGIBILITY,
    )
    plain = selector.select_blocking(_request("more"), episode_id="one", deadline=_deadline())
    assert (plain.selected_alias, plain.fallback_reason) == ("generalist", CAPABILITY_ELIGIBILITY)


def test_a_later_capability_fallback_moves_the_episode_for_good() -> None:
    """A turn the pinned candidate cannot serve repins the episode to its fallback."""
    selector, classifier = _selector({"coding": 0.9, "cyber": 0.1}, without_tools=("coder",))
    tools = _request("run them").model_copy(
        update={"tools": (GatewayToolDefinition(name="run_tests", parameters={"type": "object"}),)}
    )
    aliases = [
        selector.select_blocking(request, episode_id="one", deadline=_deadline()).selected_alias
        for request in (_request("fix"), tools, _request("thanks"))
    ]
    assert aliases == ["coder", "generalist", "generalist"]
    assert classifier.calls == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("bearer_env", "DECISION_TOKEN", "names no credential"),
        ("url", "https://decisions.example.invalid/systemone?api_key=secret", "no query string"),
    ],
)
def test_a_policy_names_no_classifier_credential(field: str, value: str, message: str) -> None:
    """Neither a credential reference nor a query-borne key enters a policy artifact."""
    document = _document(_catalog())
    classifier = {**cast("JsonObject", document["classifier"]), field: value}
    with pytest.raises(ValueError, match=message):
        TagRulesRouterPolicy.model_validate({**document, "classifier": classifier})


def test_a_failed_classification_follows_a_concurrent_episode_choice() -> None:
    """One episode never mixes models because its classifier failed for one request."""
    selector, _ = _selector(RuntimeError("down"), {"coding": 0.9, "cyber": 0.1}, delays=(0.2, 0.0))

    async def both() -> tuple[RoutingDecision, RoutingDecision]:
        """Run a slow failing request and a fast answered one in the same episode."""
        return await asyncio.gather(
            selector.select(_request("first"), episode_id="one", deadline=_deadline()),
            selector.select(_request("second"), episode_id="one", deadline=_deadline()),
        )

    failed, answered = asyncio.run(both())
    assert (answered.selected_alias, failed.selected_alias) == ("coder", "coder")


def test_a_burst_across_activations_sheds_at_one_shared_bound() -> None:
    """Selectors share one in-flight bound, so a burst sheds instead of queueing."""
    catalog = _catalog()
    permits = threading.BoundedSemaphore(1)
    slow = _ScriptedClassifier({"coding": 0.9, "cyber": 0.1}, delay=0.2)
    other = _ScriptedClassifier({"coding": 0.9, "cyber": 0.1})
    first, second = (
        TagRulesSelector(
            _policy(catalog),
            catalog,
            classifier=cast("DecisionModelClassifier", classifier),
            classifications=permits,
        )
        for classifier in (slow, other)
    )

    async def burst() -> tuple[RoutingDecision, RoutingDecision]:
        """Start a slow classification, then another activation's request while it runs."""
        answered = asyncio.create_task(
            first.select(_request("fix"), episode_id="one", deadline=_deadline())
        )
        await asyncio.sleep(0.05)
        shed = await second.select(_request("fix"), episode_id="two", deadline=_deadline())
        return await answered, shed

    answered, shed = asyncio.run(burst())
    assert (answered.selected_alias, shed.selected_alias) == ("coder", "generalist")
    assert shed.fallback_reason == CLASSIFIER_BUSY
    later = second.select_blocking(_request("fix"), episode_id="two", deadline=_deadline())
    assert (later.selected_alias, slow.calls, other.calls) == ("coder", 1, 1)


def test_an_expired_deadline_fails_the_request_before_the_classifier() -> None:
    """Routing does not outlive its request."""
    selector, classifier = _selector({"coding": 0.9})
    with pytest.raises(ProviderDeadlineExceeded):
        selector.select_blocking(
            _request("hi"), episode_id="episode", deadline=RequestDeadline(time.monotonic() - 1)
        )
    assert classifier.calls == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"default_alias": "absent"}, "default_alias must name a candidate"),
        ({"rules": [{"when_all": ["cyber"], "route": "absent"}]}, "route to a candidate"),
        ({"rules": [{"when_all": ["weather"], "route": "coder"}]}, "classifier's categories"),
        (
            {"rules": [{"when_all": ["cyber"], "when_none": ["cyber"], "route": "coder"}]},
            "require and forbid",
        ),
    ],
)
def test_a_policy_names_only_its_candidates_and_categories(
    overrides: dict[str, JsonValue], message: str
) -> None:
    """A rule that could never route correctly is refused when the policy is built."""
    with pytest.raises(ValueError, match=message):
        TagRulesRouterPolicy.model_validate(_document(_catalog(), **overrides))


def test_a_candidate_whose_identity_drifted_refuses_activation() -> None:
    """The frozen candidate identity must match what the runtime catalog resolves."""
    catalog = _catalog()
    document = _document(catalog)
    candidates = cast("list[JsonObject]", document["candidates"])
    model = cast("JsonObject", candidates[0]["model"])
    model["model_id"] = "a-different-model"
    drifted = TagRulesRouterPolicy.model_validate(_identified(document))
    with pytest.raises(TagRouterIntegrityError, match="frozen identity"):
        TagRulesSelector(drifted, catalog)


def test_a_policy_id_must_be_its_content_identity() -> None:
    """Two different policies can never share an identity."""
    document = {**_document(_catalog()), "policy_id": "tag-router-policy-chosen-by-hand"}
    with pytest.raises(ValueError, match="content identity"):
        TagRulesRouterPolicy.model_validate(document)


def test_every_decision_reaches_the_sink_before_it_is_used() -> None:
    """Fresh and reused decisions are both recorded, each bound to its request."""
    catalog = _catalog()
    records: list[RoutingDecision] = []
    selector = TagRulesSelector(
        _policy(catalog),
        catalog,
        classifier=cast(
            "DecisionModelClassifier", _ScriptedClassifier({"coding": 0.9, "cyber": 0.1})
        ),
        decision_sink=records.append,
    )
    first = selector.select_blocking(_request("fix"), episode_id="one", deadline=_deadline())
    later = selector.select_blocking(_request("more"), episode_id="one", deadline=_deadline())
    assert records == [first, later]
    assert [record.selected_alias for record in records] == ["coder", "coder"]


def test_a_stored_tag_rules_policy_loads_as_its_own_activation(tmp_path: Path) -> None:
    """The activation loader reads the policy kind and skips kNN fit evidence."""
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    policy = _policy(_catalog())
    store.write_or_verify_exact(
        artifact_id=policy.policy_id,
        artifact_type=TAG_RULES_POLICY_ARTIFACT_TYPE,
        envelope=policy,
        files={"policy.json": canonical_json_bytes(policy)},
    )
    activation = load_project_activation(
        store, project_ref="project-a", activation_ref=policy.policy_id
    )
    assert isinstance(activation, TagRulesActivation)
    assert activation.policy == policy
    assert activation.candidate_aliases == _ALIASES


def test_the_project_resolver_reports_tag_routing_and_the_selected_exact_model() -> None:
    """A tag-rules activation resolves without the learned lane and names its reason."""
    selector, _ = _selector({"coding": 0.9, "cyber": 0.1})
    key = ("project-a", selector.policy.policy_id, "e" * 64)
    resolver = RouterProjectTargetResolver(
        {key: selector},
        {(*key, alias): f"exact-{alias}" for alias in _ALIASES},
    )
    selection = resolver.select_blocking(
        target=ProjectTarget(project_ref=key[0], activation_ref=key[1], catalog_sha256=key[2]),
        request=_request("fix the build"),
        episode_namespace=("org", "identity", "revision", "episode"),
        deadline_monotonic=time.monotonic() + 5,
    )
    assert (selection.selected_alias, selection.exact_model_id) == ("coder", "exact-coder")
    assert selection.route_reason == TAG_ROUTER_ROUTE_REASON
    assert selection.fallback_reason is None


def test_a_decision_the_sink_refuses_is_not_pinned() -> None:
    """A failed recording leaves the episode free, so a retry classifies again."""
    catalog = _catalog()
    refusals = [RuntimeError("journal unavailable")]

    def sink(decision: RoutingDecision) -> None:
        """Refuse the first recording only."""
        del decision
        if refusals:
            raise refusals.pop()

    classifier = _ScriptedClassifier({"coding": 0.9, "cyber": 0.1}, {"coding": 0.1, "cyber": 0.9})
    selector = TagRulesSelector(
        _policy(catalog),
        catalog,
        classifier=cast("DecisionModelClassifier", classifier),
        decision_sink=sink,
    )
    with pytest.raises(RuntimeError, match="journal unavailable"):
        selector.select_blocking(_request("fix"), episode_id="one", deadline=_deadline())
    retried = selector.select_blocking(_request("fix"), episode_id="one", deadline=_deadline())
    assert (retried.selected_alias, classifier.calls) == ("secure", 2)
