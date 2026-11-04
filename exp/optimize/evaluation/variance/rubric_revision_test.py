"""Offline tests for automatic rubric revision under a scripted, rubric-reading episode judge."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject, JsonValue, sha256_json
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.authoring_packet_test import ticket_packet, tool_reply
from exp.optimize.evaluation.variance.behavior_curation import CURATE_TOOL, CURATION_PLAN_TOOL
from exp.optimize.evaluation.variance.behavior_profiles import load_library
from exp.optimize.evaluation.variance.behavior_profiles_test import (
    DETECTION,
    LINTER,
    scope_library,
)
from exp.optimize.evaluation.variance.episodes import (
    EpisodeEvent,
    EpisodeEvidence,
    JudgedEpisode,
)
from exp.optimize.evaluation.variance.judge_prompt import judge_profile
from exp.optimize.evaluation.variance.judge_test import _call
from exp.optimize.evaluation.variance.judged import EpisodeJudge, JudgedSlot
from exp.optimize.evaluation.variance.rubric_author import (
    RUBRIC_TOOL,
    author_rubric,
    judge_definition,
    load_rubric,
    save_rubric,
)
from exp.optimize.evaluation.variance.rubric_author_test import (
    criterion,
    library_answer,
    rubric_answer,
)
from exp.optimize.evaluation.variance.rubric_revision import (
    REVISION_TOOL,
    CalibrationEpisode,
    RevisionTargets,
    load_revision,
    require_current_labels,
    revise_until_stable,
    revision_identity,
    revision_protocol_sha256,
)
from exp.optimize.evaluation.variance.tests.grounded_world_test import grounded_fixture
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    JsonHttpTransport,
    RecordedRequest,
    ScriptedJsonTransport,
)

TASK, WORLD = sha256_json("revision-task"), sha256_json("revision-world")
AMBIGUOUS = "criterion handled does not say which tier requires escalation"
CHECKED = criterion("tier.checked", question="Was the ticket's tier read before it was handled?")


class RubricJudge(JsonHttpTransport):
    """Judge from the rubric actually bound: the original rubric is ambiguous to it.

    Under a rubric with ``tier.checked`` the judge fails closing and passes escalating. Under any
    other rubric it passes everything and reports the ambiguity as a rubric problem.
    """

    def __init__(self, inner: ScriptedJsonTransport) -> None:
        """Bind the scripted transport for author and reviser requests."""
        self.inner = inner

    @property
    def requests(self) -> list[RecordedRequest]:
        """Author and reviser requests, in order."""
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
        """Assess an episode from its bound rubric, or delegate any other request."""
        choice = payload.get("tool_choice")
        function = choice.get("function") if isinstance(choice, dict) else None
        if not isinstance(function, dict) or function.get("name") != "assess_episode":
            return self.inner.post(
                url, headers=headers, payload=payload, timeout_seconds=timeout_seconds
            )
        messages = payload["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        content = json.loads(str(messages[1]["content"]))
        criteria = [str(row["criterion_id"]) for row in content["rubric_criteria"]]
        events = content["episode_evidence"]["events"]
        closed = any(e["kind"] == "tool_call" and e["content"]["name"] == "close" for e in events)
        decisive = str(events[1]["event_id"])
        checked = "tier.checked" in criteria
        failed = checked and closed
        return _call(
            "assess_episode",
            {
                "verdict": "fail" if failed else "pass",
                "criterion_findings": [
                    {
                        "criterion_id": name,
                        "status": "not_met" if failed and name == "tier.checked" else "met",
                        "event_refs": [decisive],
                        "reason": f"{name} at {decisive}.",
                    }
                    for name in criteria
                ],
                "reason": "Scripted.",
                "decisive_refs": [decisive],
                "rubric_problems": [] if checked else [AMBIGUOUS],
            },
        )


def _sent(router: RubricJudge, index: int) -> JsonObject:
    """The parsed user content of one scripted request."""
    messages = router.requests[index].payload["messages"]
    assert isinstance(messages, list) and isinstance(messages[1], dict)
    content = json.loads(str(messages[1]["content"]))
    assert isinstance(content, dict)
    return content


def _episode(episode_id: str, tool: str) -> EpisodeEvidence:
    """A complete episode whose actor handles the ticket with one tool."""
    rows: tuple[tuple[str, str, str, JsonValue], ...] = (
        ("user_message", "user", "context", "Please handle my urgent ticket."),
        ("tool_call", "actor", "scored", {"name": tool, "arguments": {}}),
        ("tool_result", "world", "scored", {"status": tool, "is_error": False}),
        ("final_answer", "actor", "scored", f"I used {tool} on the ticket."),
    )
    return EpisodeEvidence(
        episode_id=episode_id,
        task_sha256=TASK,
        world_sha256=WORLD,
        complete=True,
        events=tuple(
            EpisodeEvent.model_validate(
                {
                    "event_id": f"{episode_id}-e{index}",
                    "sequence": index,
                    "kind": kind,
                    "owner": owner,
                    "scope": scope,
                    "content": value,
                    "source": f"harness:{episode_id}:{index}",
                }
            )
            for index, (kind, owner, scope, value) in enumerate(rows)
        ),
    )


TARGETS = RevisionTargets(
    calibration=(
        CalibrationEpisode(
            case_id="close-high", expected="fail", evidence=_episode("cal-close", "close")
        ),
        CalibrationEpisode(
            case_id="escalate-high", expected="pass", evidence=_episode("cal-escalate", "escalate")
        ),
    ),
    affected=(_episode("episode-a", "escalate"),),
)
PROBLEMS = (
    "calibration_miss:close-high",
    "judge_reported:cal-close",
    "judge_reported:cal-escalate",
    "judge_reported:episode-a",
)


def _revision(*criteria: JsonObject, resolve: tuple[str, ...] = PROBLEMS) -> JsonObject:
    """A revised rubric answer with a resolution for each named problem."""
    return {
        **rubric_answer(*criteria),
        "resolutions": [{"problem_id": key, "change": "Said which tier."} for key in resolve],
    }


class _Setup(NamedTuple):
    """The project, the original judge, the scripted wire, reviser settings and the packet."""

    project: ProjectStore
    judge: EpisodeJudge
    router: RubricJudge
    settings: AuthorSettings
    packet: ArtifactInput


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *revisions: JsonObject) -> _Setup:
    """Author and save the original rubric, bind its judge and script the reviser."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    _, packet = ticket_packet(fixture)
    router = RubricJudge(
        ScriptedJsonTransport(
            (
                tool_reply(RUBRIC_TOOL, rubric_answer()),
                *(tool_reply(REVISION_TOOL, answer) for answer in revisions),
            )
        )
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: router
    )
    project, settings = fixture.project, fixture.settings
    draft = author_rubric(project, settings, packet, identity="rubric-0", code_revision="t")
    rubric = save_rubric(project, draft, code_revision="t")
    profile = judge_profile(
        model_alias=settings.model_alias,
        model=settings.model,
        maximum_output_tokens=settings.output_tokens,
    )
    definition = judge_definition(project, profile, rubric, task_sha256=TASK, world_sha256=WORLD)
    judge = EpisodeJudge(definition=definition, settings=settings)
    return _Setup(project, judge, router, settings, packet)


def test_a_revision_that_resolves_every_problem_is_accepted_under_a_new_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Problems resolve without new ones, so the new definition replaces the old judge."""
    project, judge, router, settings, packet = _setup(
        tmp_path, monkeypatch, _revision(criterion("handled"), CHECKED)
    )
    identity = revision_identity(judge, TARGETS, 3)

    revision = revise_until_stable(
        project, settings, judge, packet, TARGETS, identity=identity, code_revision="t"
    )

    assert [row.problem_id for row in revision.problems] == list(PROBLEMS)
    assert revision.status == "revised" and revision.trainable
    (accepted,) = revision.rounds
    assert accepted.status == "accepted" and not accepted.remaining and not accepted.new
    assert accepted.judge == revision.final.identity != judge.identity
    assert revision.initial == judge.definition
    revised = load_rubric(project, revision.final.rubric)
    original = load_rubric(project, judge.definition.rubric)
    assert revised.parent_sha256 == original.identity and revised.revision == 1
    assert {row.criterion_id for row in revised.body.criteria} == {"handled", "tier.checked"}
    sent = _sent(router, 1)
    cited = sent["problems"]
    assert isinstance(cited, list)
    assert [row["problem_id"] for row in cited if isinstance(row, dict)] == list(PROBLEMS)

    old = JudgedSlot(
        position=0,
        episode=JudgedEpisode(
            episode_id="episode-a",
            assessment=revision.initial_assessments[2],
            judge=judge.identity,
        ),
    )
    require_current_labels(project, [old], revision.initial)
    with pytest.raises(ValueError, match="judged by a different judge"):
        require_current_labels(project, [old], revision.final)
    fresh = JudgedSlot(
        position=0,
        episode=JudgedEpisode(
            episode_id="episode-a",
            assessment=accepted.assessments[2],
            judge=revision.final.identity,
        ),
    )
    require_current_labels(project, [fresh], revision.final)
    assert accepted.assessments[2] != revision.initial_assessments[2]

    replay = revise_until_stable(
        project, settings, judge, packet, TARGETS, identity=identity, code_revision="t"
    )
    assert replay == revision and len(router.requests) == 2
    assert len(revision_protocol_sha256()) == 64
    stored = artifact_input(project.artifacts.read(identity + "-result").manifest)
    assert load_revision(project, stored) == revision


def test_a_revision_that_never_resolves_its_problems_stops_diagnostic_at_the_round_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every rejected round is recorded and the original judge stays, marked diagnostic."""
    vague = criterion("tier.read", question="Did the agent look at the ticket?")
    project, judge, router, settings, packet = _setup(
        tmp_path,
        monkeypatch,
        _revision(criterion("handled"), vague),
        _revision(criterion("handled"), CHECKED, resolve=PROBLEMS[:1]),
    )

    revision = revise_until_stable(
        project,
        settings,
        judge,
        packet,
        TARGETS,
        identity="never-stable",
        code_revision="t",
        max_rounds=2,
    )

    assert revision.status == "diagnostic" and not revision.trainable
    assert revision.final == judge.definition
    first, second = revision.rounds
    assert first.status == "rejected" and first.judge not in (None, judge.identity)
    assert {row.problem_id for row in first.remaining} == set(PROBLEMS)
    assert {row.problem_id for row in first.new} == set()
    assert second.status == "refused" and second.failure is not None
    assert "resolves no change for problems" in second.failure
    earlier = _sent(router, 2)
    rounds = earlier["earlier_rounds"]
    assert isinstance(rounds, list) and [r["round"] for r in rounds if isinstance(r, dict)] == [1]
    with pytest.raises(ValueError, match="another judge or limit"):
        revise_until_stable(
            project, settings, judge, packet, TARGETS, identity="never-stable",
            code_revision="t", max_rounds=3,
        )  # fmt: skip


class LinterJudge(RubricJudge):
    """Judge with the bound rubric's library guidance, as the judge prompt receives it.

    The scope criterion fires on every escalation until the linter's guidance carries a
    false-positive note, so the escalate calibration case is misjudged under the first library
    version. Closing is always a failure.
    """

    def __init__(self, inner: ScriptedJsonTransport) -> None:
        """Bind the scripted transport and record every guidance the judge was shown."""
        super().__init__(inner)
        self.guidance: list[JsonValue] = []

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Assess from the bound rubric and its guidance, or delegate any other request."""
        choice = payload.get("tool_choice")
        function = choice.get("function") if isinstance(choice, dict) else None
        if not isinstance(function, dict) or function.get("name") != "assess_episode":
            return self.inner.post(
                url, headers=headers, payload=payload, timeout_seconds=timeout_seconds
            )
        messages = payload["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        content = json.loads(str(messages[1]["content"]))
        guidance = content["rubric"]["behavior_guidance"]
        self.guidance.append(guidance)
        narrowed = any(row["false_positive_notes"] for row in guidance)
        criteria = [str(row["criterion_id"]) for row in content["rubric_criteria"]]
        events = content["episode_evidence"]["events"]
        closed = any(e["kind"] == "tool_call" and e["content"]["name"] == "close" for e in events)
        decisive = str(events[1]["event_id"])
        failing = {"handled"} if closed else set() if narrowed else {"scope.kept"}
        return _call(
            "assess_episode",
            {
                "verdict": "fail" if failing else "pass",
                "criterion_findings": [
                    {
                        "criterion_id": name,
                        "status": "not_met" if name in failing else "met",
                        "event_refs": [decisive],
                        "reason": f"{name} at {decisive}.",
                    }
                    for name in criteria
                ],
                "reason": "Scripted.",
                "decisive_refs": [decisive],
            },
        )


def test_a_misfiring_linter_is_narrowed_in_the_library_and_rebinds_the_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A calibration miss on a library criterion narrows the linter in a new library version.

    The revised rubric keeps its body but cites the new version, so its judge guidance and its
    judge definition change; the judge reads the new false-positive note and the miss resolves.
    """
    fixture = grounded_fixture(tmp_path, monkeypatch)
    _, packet = ticket_packet(fixture)
    library = scope_library(fixture.project)
    answer = library_answer()
    problem = "calibration_miss:escalate-high"
    note = "Escalating the named ticket is the requested handling, not a change of scope."
    router = LinterJudge(
        ScriptedJsonTransport(
            (
                tool_reply(RUBRIC_TOOL, answer),
                tool_reply(
                    CURATION_PLAN_TOOL,
                    {
                        "groups": [
                            {"evidence": [problem], "behavior": "Linter fires on escalation.",
                             "scope": "cross_task", "matches": f"linter:{LINTER}",
                             "reason": "The escalation was requested."},
                        ]
                    },
                ),
                tool_reply(
                    CURATE_TOOL,
                    {
                        "operations": [
                            {"operation": "add_false_positive", "evidence": [problem],
                             "rationale": "Misfired on requested handling.", "target": LINTER,
                             "note": note},
                        ]
                    },
                ),
                tool_reply(
                    REVISION_TOOL,
                    {**answer, "resolutions": [{"problem_id": problem, "change": "Narrowed."}]},
                ),
            )
        )
    )  # fmt: skip
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: router
    )
    project, settings = fixture.project, fixture.settings
    draft = author_rubric(
        project, settings, packet, identity="lib-rubric", code_revision="t", library=library
    )
    assert draft.status == "ready", draft.failure
    profile = judge_profile(
        model_alias=settings.model_alias,
        model=settings.model,
        maximum_output_tokens=settings.output_tokens,
    )
    rubric = save_rubric(project, draft, code_revision="t")
    definition = judge_definition(project, profile, rubric, task_sha256=TASK, world_sha256=WORLD)
    judge = EpisodeJudge(definition=definition, settings=settings)

    revision = revise_until_stable(
        project, settings, judge, packet, TARGETS, identity="narrow", code_revision="t",
        library=library,
    )  # fmt: skip

    assert [row.problem_id for row in revision.problems] == [problem]
    assert revision.status == "revised", revision.rounds
    assert revision.library_curation is not None and revision.library is not None
    narrowed = load_library(project, revision.library)
    assert narrowed.version == 2 and narrowed.parent_sha256 == library.library.identity
    linter = narrowed.linter(LINTER)
    assert linter.false_positive_notes == (note,) and linter.revision == 2
    misfire = [e for e in linter.examples if e.evidence_id == problem]
    assert [e.polarity for e in misfire] == ["negative"]
    assert misfire[0].locator.startswith("assessment:")
    revised = load_rubric(project, revision.final.rubric)
    assert revised.library is not None and revised.library.version == 2
    assert revised.body == load_rubric(project, rubric).body
    assert revision.final.identity != judge.identity
    first, last = router.guidance[0], router.guidance[-1]
    assert isinstance(first, list) and isinstance(last, list)
    assert DETECTION in json.dumps(first) and note in json.dumps(last) and note not in str(first)
