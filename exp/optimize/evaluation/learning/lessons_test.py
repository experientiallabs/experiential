"""Screened lessons end to end: learn fit traces, screen misses, publish, audit three ways.

Everything runs offline through the routed fake of :mod:`exp.optimize.evaluation.audit.panel_test`.
One tool (``get_order_details``) is always mispredicted, like a read of unseen instance data;
the learnable tool is mispredicted until memory names it, like a behavioural rule. Screening
decisions answer by tool, so the fit must keep the behavioural miss and drop the others.
"""

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    JsonValue,
    sha256_json,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.audit.panel import MemoryProfile, plan_held_out_panel
from exp.optimize.evaluation.audit.panel_test import (
    LEARNABLE,
    PROJECT,
    USERS,
    SimulatedProvider,
    _at,
    _tool_reply,
    _user_content,
    build_project,
    episode,
    held_out_command,
    learn_fit,
)
from exp.optimize.evaluation.fidelity.memory_policy import LESSON_SET_TYPE, LessonSelection
from exp.optimize.evaluation.learning import lesson_evolution, screening
from exp.optimize.evaluation.learning.lessons import (
    latest_lesson_set,
    load_lesson_set,
    load_library,
    preflight_screening,
    screening_plan,
)
from exp.optimize.evaluation.learning.screening import (
    SCREENING_TYPE,
    LessonScreeningSpec,
    learning_misses,
    planned_screenings,
)
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    RecordedRequest,
    ScriptedJsonTransport,
)
from exp.runtime.sessions.dispatch_ledger import DispatchLedger

INSTANCE_DATA = "get_order_details"


class LessonProvider(SimulatedProvider):
    """The routed fake, plus screening decisions by tool and a scripted library edit.

    Knowledge rules that name the learnable tool teach it, as memory does. ``edit_rules`` is
    the replacement rule text a proposer returns.
    """

    def __init__(self, episodes: list[JsonObject]) -> None:
        """Mispredict instance data always and the learnable tool until it is taught."""
        super().__init__(episodes, frozenset({INSTANCE_DATA}), learnable=True)
        self.edit_rules = f"{LEARNABLE} returns user_id, orders and payment"

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer noul screening questions and edit proposals; route the rest as before."""
        questions = payload.get("questions")
        if url.endswith("/systemone") and isinstance(questions, dict) and "covered" in questions:
            self.requests.append(RecordedRequest(url, dict(headers), payload))
            tool = _miss_tool(payload["state"])
            behavioural = 0.9 if tool == LEARNABLE else 0.2
            answers = {"behavioural": behavioural, "covered": 0.1, "supported": 0.9}
            return JsonHttpResponse(
                status_code=200,
                body={"answers": {key: {"type": "noul", "noul": p} for key, p in answers.items()}},
            )
        tools = payload.get("tools")
        if isinstance(tools, list) and "rationale" in json.dumps(tools):
            self.requests.append(RecordedRequest(url, dict(headers), payload))
            training = _user_content(payload)["training"]
            assert isinstance(training, list)
            sources = sorted(
                {
                    source
                    for row in training
                    for source in row["case"]["source_trace_ids"]  # type: ignore[index]
                }
            )
            return _tool_reply(
                "world-v1",
                str(_at(payload, "tool_choice", "function", "name")),
                {
                    "rules": self.edit_rules,
                    "rationale": "Fixture edit",
                    "source_trace_ids": sources,
                },
            )
        return super().post(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)

    def _transition(self, payload: JsonObject) -> JsonObject:
        """Knowledge rules naming the learnable tool teach it, like delivered memory."""
        request = _user_content(payload)
        knowledge = request.get("behavioral_knowledge")
        rules = knowledge.get("rules") if isinstance(knowledge, dict) else None
        taught = isinstance(rules, str) and LEARNABLE in rules
        if not taught:
            return super()._transition(payload)
        learnable, self.learnable = self.learnable, False
        try:
            return super()._transition(payload)
        finally:
            self.learnable = learnable


def _miss_tool(state: JsonValue) -> str:
    """Tool of the miss a screening request frames."""
    assert isinstance(state, dict)
    miss = state["miss"]
    assert isinstance(miss, dict)
    action = miss["action"]
    first = action[0] if isinstance(action, list) else action
    assert isinstance(first, dict)
    return str(first["name"])


def continuation(root: Path, tmp_path: Path, operation: JsonObject, *flags: str) -> str:
    """Run ``exp eval PROJECT --continuations`` and return its flattened output.

    The fixture's decision wire is unpriced, so the operation authorizes unpriced attempts.
    """
    path = tmp_path / "operation.json"
    path.write_text(json.dumps({"spend": {"unpriced_attempts": 1000}, **operation}))
    result = CliRunner().invoke(
        app,
        ["eval", PROJECT, "--root", str(root), "--continuations", str(path), *flags],
    )
    assert result.exit_code == 0, result.output
    return " ".join(result.output.split())


def screened_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, ProjectStore, LessonProvider, ArtifactInput]:
    """Build, learn the fit-only memory and screen it, returning the lesson set."""
    provider = LessonProvider([episode(i) for i in range(USERS)])
    root, store, _ = build_project(tmp_path, monkeypatch, provider)
    monkeypatch.setattr(screening, "HttpxJsonTransport", lambda: provider)
    monkeypatch.setattr(lesson_evolution, "HttpxJsonTransport", lambda: provider)
    learn_fit(root, store)
    output = continuation(
        root,
        tmp_path,
        {"action": "repair-learning", "learning_repair": {"resolution": "screen-lessons"}},
        "--yes",
        "--non-interactive",
    )
    assert '"screened_in": 1' in output, output
    plan, head = screening_plan(store, "fit")
    return root, store, provider, latest_lesson_set(store, plan.lineage, head.version)


@pytest.mark.parametrize("ledger", ["claimed", "missing"])
def test_screening_keeps_behavioural_misses_and_replays_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: str
) -> None:
    """Dry-run writes nothing; screening fits its threshold and publishes unresolved rules.

    The lost dispatch is a saved intent whose ledger slot was claimed, or whose ledger is gone:
    the ledger is created before the intent, so a missing one is lost evidence, never proof that
    nothing was sent.
    """
    provider = LessonProvider([episode(i) for i in range(USERS)])
    root, store, _ = build_project(tmp_path, monkeypatch, provider)
    monkeypatch.setattr(screening, "HttpxJsonTransport", lambda: provider)
    learn_fit(root, store)
    plan, head = screening_plan(store, "fit")
    misses = learning_misses(store, plan, head)
    tools = sorted(miss.tool or "" for miss in misses)
    assert tools.count(LEARNABLE) == 1 and set(tools) == {LEARNABLE, INSTANCE_DATA}
    # One miss was dispatched before and its answer lost: it stays unknown, never redrawn.
    planned = planned_screenings(
        store, misses, [], alias=plan.recall.jev_alias, maximum_reply_bytes=4_000
    )
    lost = next((miss, identity) for miss, _, identity in planned if miss.tool == INSTANCE_DATA)
    assert lost[1] is not None
    store.artifacts.write_json(
        artifact_id=lost[1] + "-intent",
        artifact_type=SCREENING_TYPE + "-intent",
        envelope=ArtifactEnvelope(schema_version=1, created_at=head.created_at, code_revision="c"),
        files={"request.json": {"lost": True}},
    )
    if ledger == "claimed":
        DispatchLedger(
            store.artifacts.project_directory / (lost[1] + ".sqlite"), 1, sha256_json(lost[1])
        ).claim(head.created_at.isoformat())

    operation: JsonObject = {
        "action": "repair-learning",
        "learning_repair": {"resolution": "screen-lessons"},
    }
    before = len(provider.requests)
    preview = continuation(root, tmp_path, operation, "--dry-run")
    assert f"{len(misses)} misses, {len(misses) - 1} new decisions" in preview
    assert "1 unknown" in preview
    assert len(provider.requests) == before
    assert not store.artifacts.list_ids(artifact_type=SCREENING_TYPE)

    output = continuation(root, tmp_path, operation, "--yes", "--non-interactive")
    assert len(provider.requests) == before + len(misses) - 1, "one call per new decision"
    assert '"screened_in": 1' in output and '"unknown": 1' in output
    reference = latest_lesson_set(store, plan.lineage, head.version)
    lessons = load_lesson_set(store, reference)
    assert [row.miss.tool for row in lessons.screened_in] == [LEARNABLE]
    assert {row.miss.tool for row in lessons.screened_out} == {INSTANCE_DATA}
    assert lessons.unknown[0].miss.record == lost[0].record
    # The threshold was fitted between the two kinds of miss, from same-tool deliveries.
    fit = json.loads(store.artifacts.read_bytes(lessons.fit.artifact_id, "fit.json"))
    assert fit["status"] == "fitted" and 0.162 <= fit["selected"] < 0.729
    assert lessons.selection.lessons == (lessons.screened_in[0].miss.record.artifact_id,)
    # Lessons are unresolved rules derived from their source trajectory; the trajectory
    # itself is source evidence only.
    assert lessons.library is not None
    library = load_library(store, lessons.library)
    rule = next(entry for entry in library.entries if entry.kind == "rule")
    assert rule.disposition == "unresolved"
    assert rule.derived_from == lessons.screened_in[0].miss.source
    assert [entry.kind for entry in library.entries] == ["trajectory", "rule"]
    assert library.entries[0].disposition == "source"

    purchased = len(provider.requests)
    again = continuation(root, tmp_path, operation, "--yes", "--non-interactive")
    assert len(provider.requests) == purchased, "saved decisions and the lesson set replay"
    assert '"screened_in": 1' in again
    assert len(store.artifacts.list_ids(artifact_type=LESSON_SET_TYPE)) == 1


def test_a_missing_credential_leaves_every_screening_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A credential refusal dispatched nothing, so it saves no intent and a retry answers."""
    provider = LessonProvider([episode(i) for i in range(USERS)])
    root, store, _ = build_project(tmp_path, monkeypatch, provider)
    monkeypatch.setattr(screening, "HttpxJsonTransport", lambda: provider)
    learn_fit(root, store)
    plan, head = screening_plan(store, "fit")
    misses = learning_misses(store, plan, head)
    planned = planned_screenings(
        store, misses, [], alias=plan.recall.jev_alias, maximum_reply_bytes=4_000
    )
    now = datetime.now(UTC)

    def screen(hours: float = 1) -> tuple[screening.MissScreening, ...]:
        """Screen every planned miss with a dispatch deadline ``hours`` from now."""
        return screening.screen_misses(
            store,
            planned,
            alias=plan.recall.jev_alias,
            deadline=now + timedelta(hours=hours),
            clock=lambda: datetime.now(UTC),
            code_revision="c",
        )

    monkeypatch.delenv("HELD_OUT_TEST_KEY")
    with pytest.raises(ValueError, match="HELD_OUT_TEST_KEY"):
        screen()
    assert not store.artifacts.list_ids(artifact_type=SCREENING_TYPE + "-intent")
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    # A decision wire refused before sending is just as retryable.
    with monkeypatch.context() as patch:
        patch.setattr(
            screening,
            "decision_caller",
            Mock(side_effect=ValueError("JEV alias does not expose the native decision wire")),
        )
        with pytest.raises(ValueError, match="native decision wire"):
            screen()
    assert not store.artifacts.list_ids(artifact_type=SCREENING_TYPE + "-intent")
    # The transport refuses an expired deadline after the intent but before claiming its
    # slot: nothing was sent, so the saved intents stay retryable.
    with pytest.raises(TimeoutError, match="before physical dispatch"):
        screen(hours=-1)
    assert store.artifacts.list_ids(artifact_type=SCREENING_TYPE + "-intent")
    assert not store.artifacts.list_ids(artifact_type=SCREENING_TYPE)
    preview = preflight_screening(store, LessonScreeningSpec(resolution="screen-lessons"))
    assert preview.unknown == 0 and preview.to_dispatch == len(misses)
    # A failure after the slot claim may have reached the provider: unknown, never redrawn.
    failing = ScriptedJsonTransport([JsonHttpResponse(status_code=500, body={}) for _ in misses])
    monkeypatch.setattr(screening, "HttpxJsonTransport", lambda: failing)
    assert {row.status for row in screen()} == {"unknown"}
    assert len(failing.requests) == len(misses)
    assert {row.status for row in screen()} == {"unknown"}
    assert len(failing.requests) == len(misses)


def test_audit_compares_memory_profiles_on_identical_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No memory, all learned memory and screened lessons differ only in what memory is read."""
    root, store, provider, reference = screened_project(tmp_path, monkeypatch)
    lessons = load_lesson_set(store, reference)

    def plan(profile: MemoryProfile) -> list[JsonObject]:
        """Planned specs of one memory profile, without their memory and run identity."""
        planned = plan_held_out_panel(
            store,
            2,
            profile="memory",
            code_revision="test",
            clock=lambda: store.artifacts.read(reference.artifact_id).manifest.created_at,
            memory_profile=profile,
        )
        return [
            spec.model_dump(mode="json", exclude={"run_id", "deadline", "memory"})
            for spec in planned.specs
        ]

    learned, none, screened = plan("learned"), plan("none"), plan("lessons")
    # Without memory there is no recall call per turn, so only the attempt ceiling is lower.
    assert all(
        int(str(a.pop("maximum_physical_attempts"))) >= int(str(b.pop("maximum_physical_attempts")))
        for a, b in zip(learned, none, strict=True)
    )
    assert all(row.pop("maximum_physical_attempts") for row in screened)
    assert learned == none == screened

    before = len(provider.requests)
    output = held_out_command(root, 2, "--audit-memory", "lessons")
    assert f"replays read only lesson set {reference.artifact_id}" in output
    served = [
        _user_content(request.payload)
        for request in provider.requests[before:]
        if request.url.endswith("/chat/completions")
        and '"submit_transition"' in json.dumps(request.payload.get("tool_choice"))
    ]
    assert served, "the lesson audit replayed held-out traces"
    allowed = set(lessons.selection.lessons)
    for prediction in served:
        assert not prediction.get("memory_observations"), "raw observations are not lessons"
        history = json.dumps(prediction.get("research_history") or {})
        for row in lessons.screened_out:
            assert row.miss.record.artifact_id not in history
        assert all(
            identity in allowed
            for identity in _record_ids(prediction.get("research_history") or {})
        )
    none = held_out_command(root, 2, "--audit-memory", "none", "--include-inspected")
    assert "replays read no memory" in none
    learned = held_out_command(root, 2, "--include-inspected")
    assert "replays read" not in learned.split("Audited checkpoint")[-1].split("Complete")[0]


def _record_ids(history: JsonValue) -> list[str]:
    """Record identities delivered in a prediction's research history."""
    found: list[str] = []
    pending = [history]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            identity = item.get("id")
            if isinstance(identity, str) and item.get("kind") != "original-observation":
                if "session_id" in item:
                    found.append(identity)
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return found


def test_audit_memory_needs_the_audit_and_a_known_profile(tmp_path: Path) -> None:
    """The option is refused outside an audit, and for an unknown profile."""
    result = CliRunner().invoke(app, ["eval", PROJECT, "--audit-memory", "lessons"])
    assert result.exit_code != 0
    assert "--audit-memory applies only with --audit" in " ".join(result.output.split())
    result = CliRunner().invoke(app, ["eval", PROJECT, "--audit", "2", "--audit-memory", "x"])
    assert result.exit_code != 0


def saved_lesson_set(
    store: ProjectStore, name: str, *supersedes: ArtifactInput, at: datetime
) -> ArtifactInput:
    """Save a stub lesson set of memory ``fit-memory`` v3, citing the sets it supersedes."""
    selection = LessonSelection(lineage="fit-memory", version=3, lessons=())
    return artifact_input(
        store.artifacts.write_json(
            artifact_id=name,
            artifact_type=LESSON_SET_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=at,
                code_revision="test",
                inputs=tuple(sorted(supersedes, key=lambda item: item.artifact_id)),
            ),
            files={"selection.json": selection.model_dump(mode="json")},
        )
    )


def test_the_latest_lesson_set_is_the_one_no_other_supersedes_whatever_the_clocks_claim(
    tmp_path: Path,
) -> None:
    """A superseding set wins although its clock claims it is older; unordered sets refuse."""
    store = ProjectStore(tmp_path, "case")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    first = saved_lesson_set(store, "lesson-set-first", at=now + timedelta(days=3))
    second = saved_lesson_set(store, "lesson-set-second", first, at=now)
    assert latest_lesson_set(store, "fit-memory", 3) == second
    saved_lesson_set(store, "lesson-set-rival", first, at=now + timedelta(days=9))
    with pytest.raises(ValueError, match="do not supersede one another"):
        latest_lesson_set(store, "fit-memory", 3)
    with pytest.raises(ValueError, match="no screened lesson set"):
        latest_lesson_set(store, "fit-memory", 4)
