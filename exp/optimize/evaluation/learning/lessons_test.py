"""Screened lessons end to end: learn fit traces, screen misses, publish.

Everything runs offline through the routed fake of :mod:`exp.optimize.evaluation.audit.panel_test`.
One tool (``get_order_details``) is always mispredicted, like a read of unseen instance data;
the learnable tool is mispredicted until memory names it, like a behavioural rule. Screening
decisions answer by tool, so the fit must keep the behavioural miss and drop the others.
"""

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject, JsonValue
from exp.common.project import ProjectStore
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
    learn_fit,
)
from exp.optimize.evaluation.fidelity.memory_policy import LESSON_SET_TYPE
from exp.optimize.evaluation.learning import lesson_evolution, screening
from exp.optimize.evaluation.learning.lessons import (
    latest_lesson_set,
    load_lesson_set,
    load_library,
    screening_plan,
)
from exp.optimize.evaluation.learning.screening import (
    SCREENING_TYPE,
    learning_misses,
    planned_screenings,
)
from exp.runtime.models.providers.transport import JsonHttpResponse, RecordedRequest

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
    """Run ``exp eval PROJECT --continuations`` and return its flattened output."""
    path = tmp_path / "operation.json"
    path.write_text(json.dumps(operation))
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


def test_screening_keeps_behavioural_misses_and_replays_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dry-run writes nothing; screening fits its threshold and publishes unresolved rules."""
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
