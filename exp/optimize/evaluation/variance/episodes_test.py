"""Tests for the episode judge contracts and the single verdict-to-reward encoding."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.common.models import BillingSource, ModelSnapshot
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.episodes import (
    JUDGE_INPUT_BYTES,
    CriterionFinding,
    EpisodeAssessment,
    EpisodeEvent,
    EpisodeEvidence,
    JudgeDefinition,
    JudgeProfile,
    episode_reward,
)

_SHA = "a" * 64


def _event(sequence: int, scope: str = "scored", owner: str = "actor") -> EpisodeEvent:
    """One recorded event."""
    return EpisodeEvent.model_validate(
        {
            "event_id": f"e{sequence}",
            "sequence": sequence,
            "kind": "assistant_message",
            "owner": owner,
            "scope": scope,
            "content": "text",
            "source": f"trace:t:step:{sequence}",
        }
    )


def _evidence(*events: EpisodeEvent, complete: bool = True) -> EpisodeEvidence:
    """Evidence over the given events."""
    return EpisodeEvidence(
        episode_id="ep", task_sha256=_SHA, world_sha256=_SHA, events=events, complete=complete
    )


def _assessment(evidence: EpisodeEvidence, **changes: object) -> EpisodeAssessment:
    """A judged pass citing the first event."""
    base: dict[str, object] = {
        "episode_id": "ep",
        "evidence_sha256": evidence.identity,
        "judge": _SHA,
        "status": "judged",
        "verdict": "pass",
        "findings": (
            CriterionFinding(criterion_id="c", status="met", evidence_refs=("e0",), rationale="r"),
        ),
        "decisive_refs": ("e0",),
    }
    return EpisodeAssessment.model_validate({**base, **changes})


def test_evidence_is_ordered_unique_and_context_first() -> None:
    """Out-of-order, repeated or context-after-scored events are refused."""
    assert _evidence(_event(0, "context", "simulator"), _event(1)).complete
    with pytest.raises(ValueError, match="numbered"):
        _evidence(_event(1))
    with pytest.raises(ValueError, match="precede"):
        _evidence(_event(0), _event(1, "context"))
    with pytest.raises(ValueError, match="missing"):
        EpisodeEvidence(
            episode_id="ep",
            task_sha256=_SHA,
            world_sha256=_SHA,
            events=(_event(0),),
            complete=True,
            missing=("tool log",),
        )


def test_only_a_judged_verdict_becomes_a_reward() -> None:
    """Pass is 1, fail is 0, and anything unjudged or unassessable is unscored."""
    evidence = _evidence(_event(0))
    assert episode_reward(_assessment(evidence)) == 1.0
    assert episode_reward(_assessment(evidence, verdict="fail")) == 0.0
    assert episode_reward(_assessment(evidence, verdict="unassessable")) is None
    malformed = _assessment(
        evidence, status="malformed", verdict="unassessable", findings=(), failure="bad schema"
    )
    assert episode_reward(malformed) is None
    with pytest.raises(ValueError, match="unjudged"):
        _assessment(evidence, status="unavailable", verdict="pass", findings=(), failure="x")


def test_citations_must_exist_in_the_judged_evidence() -> None:
    """An assessment cannot cite events the evidence does not hold, or judge other evidence."""
    evidence = _evidence(_event(0))
    _assessment(evidence).check_citations(evidence)
    with pytest.raises(ValueError, match="not in the evidence"):
        _assessment(evidence, decisive_refs=("e9",)).check_citations(evidence)
    other = _evidence(_event(0), _event(1))
    with pytest.raises(ValueError, match="does not judge"):
        _assessment(evidence).check_citations(other)
    assert ArtifactInput(artifact_id="x", sha256=_SHA).artifact_id == "x"


def _profile(**changes: object) -> JudgeProfile:
    """A judge profile with frozen placeholder digests."""
    base: dict[str, object] = {
        "model_alias": "judge",
        "model": ModelSnapshot(
            billing_source=BillingSource.CUSTOMER_MANAGED,
            provider="openai",
            model_id="judge-model",
            capabilities_sha256="6" * 64,
            connection_sha256="7" * 64,
        ),
        "prompt_version": "episode-judge-v1",
        "prompt_sha256": "1" * 64,
        "output_schema_sha256": "2" * 64,
    }
    return JudgeProfile.model_validate({**base, **changes})


def test_profile_input_budget_defaults_to_the_author_request_default() -> None:
    """The profile freezes the request ceiling, defaulting to the shared author default."""
    default = AuthorSettings.model_fields["maximum_input_bytes"].default
    assert JUDGE_INPUT_BYTES == default == _profile().maximum_input_bytes
    assert _profile(maximum_input_bytes=4096).identity != _profile().identity
    with pytest.raises(ValueError, match="maximum_input_bytes"):
        _profile(maximum_input_bytes=10)


def test_definition_requires_a_content_digest_for_the_rubric() -> None:
    """A malformed or manifest-digest rubric_sha256 is refused."""
    rubric = ArtifactInput(artifact_id="rubric-a", sha256="4" * 64)
    base: dict[str, object] = {
        "profile": _profile(),
        "rubric": rubric,
        "rubric_sha256": "5" * 64,
        "task_sha256": _SHA,
        "world_sha256": _SHA,
    }
    assert JudgeDefinition.model_validate(base).rubric_sha256 == "5" * 64
    with pytest.raises(ValueError, match="rubric_sha256"):
        JudgeDefinition.model_validate({**base, "rubric_sha256": "not-a-digest"})
    with pytest.raises(ValueError, match="file content"):
        JudgeDefinition.model_validate({**base, "rubric_sha256": rubric.sha256})
