"""Tests for tag-rules routing: policy rules, fallbacks, stickiness, loading, resolution."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject, canonical_json_bytes
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
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    ProjectTarget,
)
from exp.runtime.gateway.guardrails.classification import DecisionModelClassifier
from exp.runtime.gateway.guardrails.contracts import (
    CategoryScore,
    ClassifierVerdict,
    GuardrailCheck,
    ScoredVerdict,
)
from exp.runtime.gateway.project_activation import load_project_activation
from exp.runtime.gateway.routing import RouterProjectTargetResolver
from exp.runtime.gateway.tag_router import (
    CLASSIFIER_UNAVAILABLE,
    NO_CLASSIFIABLE_TEXT,
    NO_RULE_MATCHED,
    TAG_ROUTER_ROUTE_REASON,
    TagRouterIntegrityError,
    TagRulesActivation,
    TagRulesRouterPolicy,
    TagRulesSelector,
)
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.async_transport import ProviderDeadlineExceeded, RequestDeadline

_ALIASES = ("coder", "secure", "generalist")
_THRESHOLDS = {"coding": 0.7, "cyber": 0.8}


def _catalog() -> RuntimeModelCatalog:
    """Return a credential-free runtime catalog with three candidates."""
    capabilities = ModelCapabilities(
        supports_completions=True,
        supports_tools=True,
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
                    capabilities=capabilities,
                )
                for alias in _ALIASES
            },
            roles=ModelRoles(candidates=_ALIASES, incumbent="generalist"),
        ),
        environment={},
    )


def _document(catalog: RuntimeModelCatalog, **overrides: object) -> JsonObject:
    """Return a policy routing cyber work to ``secure`` and other coding to ``coder``."""
    document: JsonObject = {
        "schema_version": 1,
        "created_at": datetime(2026, 11, 1, tzinfo=UTC).isoformat(),
        "code_revision": "test",
        "policy_id": "router-policy-tags",
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
    return document


def _policy(catalog: RuntimeModelCatalog) -> TagRulesRouterPolicy:
    """Return the validated test policy."""
    return TagRulesRouterPolicy.model_validate(_document(catalog))


class _ScriptedClassifier:
    """Answer each inspection from a queue of scores, failures, or text-free verdicts."""

    def __init__(self, *answers: dict[str, float] | Exception | None) -> None:
        """Queue answers in call order."""
        self._answers = list(answers)
        self.calls = 0

    async def inspect_input(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Return the next scripted verdict."""
        del request, check
        self.calls += 1
        answer = self._answers.pop(0)
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
) -> tuple[TagRulesSelector, _ScriptedClassifier]:
    """Return a selector over the test policy and its scripted classifier."""
    catalog = _catalog()
    classifier = _ScriptedClassifier(*answers)
    selector = TagRulesSelector(
        _policy(catalog), catalog, classifier=cast("DecisionModelClassifier", classifier)
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
    overrides: dict[str, object], message: str
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
    with pytest.raises(TagRouterIntegrityError, match="frozen identity"):
        TagRulesSelector(TagRulesRouterPolicy.model_validate(document), catalog)


def test_a_stored_tag_rules_policy_loads_as_its_own_activation(tmp_path: Path) -> None:
    """The activation loader reads the policy kind and skips kNN fit evidence."""
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    policy = _policy(_catalog())
    store.write_or_verify_exact(
        artifact_id=policy.policy_id,
        artifact_type="router-policy",
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
    key = ("project-a", "router-policy-tags", "e" * 64)
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
