"""Offline tests for judged labels, with the scripted episode-judge transport other tests reuse.

:class:`JudgeRouter` answers every ``assess_episode`` request from a verdict function over the
rendered evidence and passes every other author request to an inner scripted transport, so role
and author scripts keep their order while the judge's verdicts depend only on the evidence.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject, sha256_json
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.audit.panel_test import write_catalog
from exp.optimize.evaluation.continuations.admission import ContinuationAdmission
from exp.optimize.evaluation.continuations.admission_test import admitted_workbook_study
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.episodes import JudgeDefinition
from exp.optimize.evaluation.variance.judge_prompt import judge_profile
from exp.optimize.evaluation.variance.judge_test import _RUBRIC, _answer, _call
from exp.optimize.evaluation.variance.judged import (
    EpisodeJudge,
    JudgedSlot,
    judge_study,
    judged_metrics,
    slot_coverage,
    slot_rewards,
)
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    JsonHttpTransport,
    RecordedRequest,
    ScriptedJsonTransport,
)
from exp.runtime.models.registry import RuntimeModelCatalog

Verdict = Callable[[JsonObject], str]
"""Maps the rendered episode evidence to pass, fail, unassessable or malformed."""

TASK, WORLD = sha256_json("judged-task"), sha256_json("judged-world")


def always(verdict: str) -> Verdict:
    """A verdict function that ignores the evidence."""
    return lambda evidence: verdict


def state_judge(path: tuple[str, ...], expected: object) -> Verdict:
    """A judge passing a complete episode whose last recorded state holds ``expected`` at path."""

    def verdict(evidence: JsonObject) -> str:
        """Read the last state snapshot; an incomplete episode is unassessable."""
        events = evidence["events"]
        assert isinstance(events, list)
        states = [
            e["content"] for e in events if isinstance(e, dict) and e["kind"] == "state_snapshot"
        ]
        value: object = states[-1] if states else None
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value == expected:
            return "pass" if evidence["complete"] else "unassessable"
        return "fail"

    return verdict


def judge_reply(verdict: str, decisive: str) -> JsonHttpResponse:
    """One scripted ``assess_episode`` answer citing an event the evidence holds."""
    if verdict == "malformed":
        return _call("assess_episode", {"verdict": "pass", "reason": "No findings."})
    if verdict == "fail":
        return _answer("fail", decisive=decisive, right_order="not_met")
    if verdict == "unassessable":
        return _answer("unassessable", decisive=decisive, right_order="insufficient_evidence")
    return _answer("pass", decisive=decisive)


class JudgeRouter(JsonHttpTransport):
    """Answer judge requests from the evidence; send every other request to the inner script.

    Attributes:
        inner: Scripted transport for role and author requests.
        verdict: Verdict function over the rendered evidence.
        judged: Every judge request's rendered evidence, in order.
    """

    def __init__(self, inner: ScriptedJsonTransport, verdict: Verdict) -> None:
        """Bind the inner script and the verdict function."""
        self.inner, self.verdict = inner, verdict
        self.judged: list[JsonObject] = []

    @property
    def requests(self) -> list[RecordedRequest]:
        """Non-judge requests, as the inner script recorded them."""
        return self.inner.requests

    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> JsonHttpResponse:
        """Delegate a GET to the inner script."""
        return self.inner.get(url, headers=headers, timeout_seconds=timeout_seconds)

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer an ``assess_episode`` request, or delegate any other request."""
        choice = payload.get("tool_choice")
        function = choice.get("function") if isinstance(choice, dict) else None
        if not isinstance(function, dict) or function.get("name") != "assess_episode":
            return self.inner.post(
                url, headers=headers, payload=payload, timeout_seconds=timeout_seconds
            )
        messages = payload["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        content = messages[1]["content"]
        assert isinstance(content, str)
        evidence = json.loads(content)["episode_evidence"]
        assert isinstance(evidence, dict)
        self.judged.append(evidence)
        events = evidence["events"]
        assert isinstance(events, list)
        scored = [e for e in events if isinstance(e, dict) and e["scope"] == "scored"]
        decisive = scored[0] if scored else events[-1]
        assert isinstance(decisive, dict)
        return judge_reply(self.verdict(evidence), str(decisive["event_id"]))


def route_judge(
    monkeypatch: pytest.MonkeyPatch,
    roles: Sequence[JsonHttpResponse] = (),
    verdict: Verdict | None = None,
) -> JudgeRouter:
    """Route every author request through a judge router over scripted role replies."""
    router = JudgeRouter(ScriptedJsonTransport(tuple(roles)), verdict or always("pass"))
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: router
    )
    return router


def episode_judge(
    project: ProjectStore,
    settings: AuthorSettings,
    *,
    task_sha256: str = TASK,
    world_sha256: str = WORLD,
) -> EpisodeJudge:
    """Save the fixture rubric once and bind a frozen judge to it under the given settings."""
    store = project.artifacts
    if not store.exists("rubric-refund"):
        store.write_json(
            artifact_id="rubric-refund",
            artifact_type="episode-rubric",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=datetime.now(UTC), code_revision="test", inputs=()
            ),
            files={"rubric.json": _RUBRIC},
        )
    manifest = store.read("rubric-refund").manifest
    definition = JudgeDefinition(
        profile=judge_profile(
            model_alias=settings.model_alias,
            model=settings.model,
            maximum_output_tokens=settings.output_tokens,
        ),
        rubric=artifact_input(manifest),
        rubric_sha256=manifest.files[0].sha256,
        task_sha256=task_sha256,
        world_sha256=world_sha256,
    )
    return EpisodeJudge(definition=definition, settings=settings)


def judge_settings(project: ProjectStore) -> AuthorSettings:
    """Judge-alias request settings of a project whose catalog has a ``judge`` alias."""
    catalog = load_model_catalog(project.model_catalog_path)
    return AuthorSettings(
        model_alias="judge",
        model=RuntimeModelCatalog(catalog, environment={}).snapshot("judge")[0],
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        output_tokens=1024,
        deadline=datetime.now(UTC) + timedelta(days=1),
    )


def _study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, EpisodeJudge, ArtifactInput]:
    """A project holding one saved two-branch workbook study, and a frozen judge."""
    project = ProjectStore(tmp_path, "case")
    acceptance = admitted_workbook_study(project)
    write_catalog(tmp_path)
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    admission = ContinuationAdmission.model_validate_json(
        project.artifacts.read_bytes(acceptance.artifact_id, "admission.json")
    )
    return project, episode_judge(project, judge_settings(project)), admission.study


def test_every_branch_is_judged_once_and_labels_come_from_saved_assessments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pass and fail are the judge's verdicts; a replay reads them back with no call."""
    project, judge, reference = _study(tmp_path, monkeypatch)
    verdicts = iter(["pass", "fail"])
    router = route_judge(monkeypatch, verdict=lambda evidence: next(verdicts))

    saved, judged = judge_study(project, reference, judge, code_revision="test")
    rewards = slot_rewards(project, judged.branches, judge.identity)

    planned = len(judged.branches)
    assert len(router.judged) == planned and all(slot.episode for slot in judged.branches)
    assert rewards == tuple(1.0 if i % 2 == 0 else 0.0 for i in range(planned))
    coverage = slot_coverage(project, judged.branches, judge.identity)
    assert (coverage.planned, coverage.scored) == (planned, planned)
    metrics = judged_metrics(project, judged, 0.5)
    assert metrics["judge"] == judge.identity and metrics["brier"] == 0.25
    assert project.artifacts.read(saved.artifact_id).manifest.artifact_type == (
        "judged-continuation-study"
    )

    assert judge_study(project, reference, judge, code_revision="test") == (saved, judged)
    assert len(router.judged) == planned


def test_malformed_and_unassessable_judgments_stay_unscored_coverage_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither a malformed answer nor an unassessable verdict becomes 0, 0.5, or disappears."""
    project, judge, reference = _study(tmp_path, monkeypatch)
    script = iter(["malformed", "unassessable"])
    route_judge(monkeypatch, verdict=lambda evidence: next(script))

    _, judged = judge_study(project, reference, judge, code_revision="test")
    rewards = slot_rewards(project, judged.branches, judge.identity)
    coverage = slot_coverage(project, judged.branches, judge.identity)

    assert rewards == (None, None) and all(slot.episode for slot in judged.branches)
    assert (coverage.malformed, coverage.unassessable) == (1, 1)
    assert coverage.insufficient_evidence == 1 and coverage.unscored == coverage.planned == 2
    metrics = judged_metrics(project, judged, 0.5)
    assert metrics["unassessable"] == 2 and metrics["resumed_success_rate"] is None
    assert metrics["brier"] is None


def test_a_slot_judged_by_another_judge_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Labels are never reused across judge definitions by alias."""
    project, judge, reference = _study(tmp_path, monkeypatch)
    route_judge(monkeypatch)
    _, judged = judge_study(project, reference, judge, code_revision="test")

    with pytest.raises(ValueError, match="different judge"):
        slot_rewards(project, judged.branches, "0" * 64)
    with pytest.raises(ValueError, match="either an assessment or a gap"):
        JudgedSlot(position=0)
