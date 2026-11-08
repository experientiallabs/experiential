"""The re-authoring loop: a large family that collapses to one task type is authored again."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    AssistantAction,
    ModelRequest,
    ModelResponse,
    OperationEconomics,
    ToolCall,
)
from exp.common.project import ProjectStore
from exp.simulation.mining.authoring_loop import (
    AuthoringLoopSpec,
    FamilyBank,
    author_family_bank,
)
from exp.simulation.mining.environment_review import ReviewerError
from exp.simulation.mining.questions import QuestionBank
from exp.simulation.mining.tag_authoring import (
    AUTHOR_VARIANTS,
    SUBMIT_TOOL,
    TagAuthor,
    TagAuthoringSpec,
)
from exp.simulation.mining.tag_authoring_test import SNAPSHOT, question
from exp.simulation.mining.tagging import TagMatrix, TagRow
from exp.simulation.mining.views import Unit

TAGGER: JsonObject = {"provider": "fixture", "model_id": "clef-fixture"}
VIEWS = ("task", "observations")
MEMBERS = 60

UNSTRUCTURED = {
    "task_shape": [("asks_output", "[R1] Does the request name an output?")],
    "environment_state": [("reads_records", "[R2] Do the tools read records?")],
    "behavior_error": [("checks_first", "[R3] Does the agent check before acting?")],
}
"""Discriminating, distinct questions that fire independently of each other: no task types."""

SEPARATING = {
    "task_shape": [
        ("network_service", "[G1] Does the request ask for a service others connect to?"),
        ("long_running", "[G2] Does the request ask for a process that keeps running?"),
    ],
    "environment_state": [("opens_ports", "[G3] Do the tools open a listening port?")],
}
"""Traits that hold mostly in planted group A: together they mark a distinct kind of task."""


def rendered() -> list[dict[str, Unit]]:
    """Members whose text plants group A (the first 24) or group B."""
    rows = []
    for index in range(MEMBERS):
        trace = f"m{index:02d}"
        group = "GROUP_A" if index < 24 else "GROUP_B"
        rows.append({view: Unit(trace, view, f"{view} {trace} {group}") for view in VIEWS})
    return rows


def coin(*parts: str) -> float:
    """A deterministic number in [0, 1) from its parts."""
    return int(hashlib.sha256("/".join(parts).encode()).hexdigest()[:8], 16) / 2**32


@dataclass
class FakeClef:
    """Answers ``[Rn]`` as an independent fair coin and ``[Gn]`` as group A with 10% flips."""

    asked: list[tuple[str, ...]] = field(default_factory=list)

    def tag(self, units: Sequence[Unit], bank: QuestionBank) -> TagMatrix:
        """Answer every question about every unit."""
        self.asked.append(bank.question_ids())
        rows = []
        for unit in units:
            values = []
            for asked in bank.questions:
                marker = re.match(r"\[(\w)(\d)\]", asked.text)
                assert marker is not None
                flip = coin(unit.trace_id, asked.text)
                if marker.group(1) == "G":
                    held = ("GROUP_A" in unit.text) != (flip < 0.1)
                else:
                    held = flip < 0.5
                values.append(0.95 if held else 0.05)
            rows.append(
                TagRow(trace_id=unit.trace_id, unit_hash=unit.unit_hash, values=tuple(values))
            )
        return TagMatrix(
            bank_id=bank.bank_id,
            bank_version=bank.version,
            question_ids=bank.question_ids(),
            view=units[0].view,
            rows=tuple(rows),
        )


@dataclass
class FakeAstra:
    """Submits ``first`` on a first pass and ``feedback`` once told the family collapsed."""

    first: dict[str, list[tuple[str, str]]]
    feedback: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    refuse: bool = False
    packets: list[JsonObject] = field(default_factory=list)
    systems: list[str] = field(default_factory=list)

    def __call__(self, request: ModelRequest, *, identity: str) -> ModelResponse:
        """Answer one draw by submitting at once."""
        if self.refuse:
            raise ReviewerError("replay must not draw")
        system = request.messages[0].content or ""
        packet = json.loads(request.messages[1].content or "{}")
        self.packets.append(packet)
        self.systems.append(system)
        variant = next(name for name, v in AUTHOR_VARIANTS.items() if v.stance in system)
        script = self.feedback if "feedback" in packet else self.first
        sample = [row["trace_id"] for row in packet["episodes"]]
        proposals = [question(slug, text, sample) for slug, text in script.get(variant, [])]
        if not proposals:  # a stance with nothing new to add still answers
            proposals = [question(f"{variant}_none", f"[R9] Is {variant} unusual?", sample)]
        call = ToolCall(call_id="s", name=SUBMIT_TOOL, arguments={"questions": proposals})
        return ModelResponse(
            output=AssistantAction(tool_calls=(call,)),
            model=SNAPSHOT,
            economics=OperationEconomics(),
        )


def shared() -> TagMatrix:
    """The fixed task-view bank every family is clustered on: two independent coins."""
    rows = tuple(
        TagRow(
            trace_id=row["task"].trace_id,
            unit_hash=f"fixed-{row['task'].trace_id}",
            values=tuple(
                0.95 if coin(row["task"].trace_id, f"fixed{i}") < 0.5 else 0.05 for i in range(2)
            ),
        )
        for row in rendered()
    )
    return TagMatrix(
        bank_id="fixed", bank_version=1, question_ids=("f0", "f1"), view="task", rows=rows
    )


def run(tmp_path: Path, astra: FakeAstra, clef: FakeClef) -> FamilyBank:
    """Author one family's bank through the loop."""
    return author_family_bank(
        ProjectStore(tmp_path, "case").artifacts,
        TagAuthor(draw=astra, model=SNAPSHOT),
        clef,
        scope="environment-family-c01",
        bank_id="authored-c01",
        rendered=rendered(),
        views=VIEWS,
        shared=(shared(),),
        tagger_identity=TAGGER,
        authoring=TagAuthoringSpec(max_turns=2, repairs=0),
        loop=AuthoringLoopSpec(min_members_to_retry=50),
        root=tmp_path / "banks",
        code_revision="test",
    )


def test_a_collapsed_family_is_authored_again_and_splits(tmp_path: Path) -> None:
    """One type from the first bank; told so, round two's traits split the planted groups."""
    astra = FakeAstra(first=UNSTRUCTURED, feedback=SEPARATING)
    result = run(tmp_path, astra, FakeClef())

    assert result.task_types == 2
    assert len(result.rounds) == 1
    round_ = result.rounds[0]
    assert round_.feedback.task_types == 1
    assert round_.outcome.startswith("split")
    assert set(round_.added) >= {"task_shape.network_service", "environment_state.opens_ports"}
    told = round_.feedback.questions
    assert {q.question_id for q in told if q.kept} == {
        "task_shape.asks_output",
        "environment_state.reads_records",
        "behavior_error.checks_first",
    }
    assert all(0.0 < q.fire_rate < 1.0 for q in told)

    # The re-authoring draws saw the feedback and the request for a 20-80 split on a new axis.
    looped = [p for p in astra.packets if "feedback" in p]
    assert len(looped) == len(AUTHOR_VARIANTS)
    assert looped[0]["feedback"]["target_split"] == [0.2, 0.8]
    assert "different axis" in astra.systems[-1]
    assert {row["question_id"] for row in looped[0]["existing_questions"]} >= {
        "task_shape.asks_output"
    }

    # The union bank is published and the members' matrices carry both rounds' questions.
    ids = result.bank.bank.question_ids()
    assert "task_shape.asks_output" in ids and "task_shape.network_service" in ids
    assert (
        result.matrices["task"].question_ids
        == result.bank.bank.families("task_shape").question_ids()
    )
    assert len(result.bank.rounds) == 1

    # Every round replays from its saved steps: no author draw on a second run.
    again = run(tmp_path, FakeAstra(first={}, refuse=True), FakeClef())
    assert again.bank == result.bank and again.task_types == 2


def test_a_family_that_already_splits_does_not_loop(tmp_path: Path) -> None:
    """A first bank that already yields two types never asks the author again."""
    astra = FakeAstra(first=SEPARATING, feedback=UNSTRUCTURED)
    result = run(tmp_path, astra, FakeClef())

    assert result.task_types == 2
    assert result.rounds == ()
    assert not any("feedback" in packet for packet in astra.packets)
