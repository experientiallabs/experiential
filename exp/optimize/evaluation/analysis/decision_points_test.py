"""Decision points label reached states from saved trajectories on two executable world kinds.

The fixtures here are shared by the cascade fit and confirmation tests: an XLSX workbook world
and a SQLite ticket world, a scripted main model that alternates a correct first action with a
mistake, and one scripted provider transport that answers the main model, a small drafting
model and System One by route.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    JsonValue,
    sha256_json,
)
from exp.common.models import (
    AssistantAction,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
    ToolCall,
    load_model_catalog,
)
from exp.common.project import ArtifactStore, ProjectStore, artifact_input
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.analysis.decision_points import (
    DecisionPointSet,
    extract_decision_points,
)
from exp.optimize.evaluation.audit.panel_test import write_catalog
from exp.optimize.evaluation.continuations.study import (
    ContinuationStudySpec,
    run_continuation_study,
    study_identity,
)
from exp.optimize.evaluation.continuations.study_test import probe as workbook_probe
from exp.optimize.evaluation.continuations.study_test import workbook_runtime
from exp.optimize.evaluation.executable.loader import ContinuationRuntime
from exp.runtime.agents.cascade import OTHER, RESPOND
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.environments.tests.world_conformance_test import ticket_case, workbook_case
from exp.runtime.environments.world_contracts import ExecutableCheckpoint
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.runtime.models.providers.transport import JsonHttpResponse, JsonHttpTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.continuations import (
    ContinuationCheckpoint,
    WorldCheckpoint,
    load_continuation_checkpoint,
    persist_continuation_checkpoint,
)
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier

WORLDS = ("tickets", "workbook")
ACTOR_CALLS = 4


def ticket_probe(checkpoint: WorldCheckpoint) -> JsonObject:
    """Read ticket 1's status from the native SQLite image, never shown to the actor."""
    assert isinstance(checkpoint, ExecutableCheckpoint)
    (image,) = checkpoint.image.parts.values()
    database = sqlite3.connect(":memory:")
    database.deserialize(image)
    row = database.execute("SELECT status FROM tickets WHERE id = 1").fetchone()
    return {"status": row[0]}


@dataclass(frozen=True)
class World:
    """One world kind bound to its task, verifier and the two scripted first actions.

    Attributes:
        kind: ``tickets`` or ``workbook``.
        instruction: Public task text; the lineage name is appended.
        correct: The first call that completes the task.
        mistake: The wrong first action of the alternating main model.
        verifier: Goal and policy predicate.
        probe_sha256: Identity of the evaluator-only projection.
    """

    kind: str
    instruction: str
    correct: ToolCall
    mistake: AssistantAction
    verifier: TaskVerifier
    probe_sha256: str = field(default="")

    @property
    def correct_key(self) -> str:
        """The correct next-action decision at the initial state."""
        return self.correct.name

    @property
    def mistake_key(self) -> str:
        """The mistaken next-action decision at the initial state."""
        return self.mistake.tool_calls[0].name if self.mistake.tool_calls else RESPOND


def world(kind: str) -> World:
    """Return the fixture definition of one world kind."""
    if kind == "workbook":
        return World(
            kind=kind,
            instruction="Delete the first data row of Orders into output.xlsx",
            correct=workbook_case().first,
            mistake=AssistantAction(content="done"),
            verifier=TaskVerifier(
                description="One data row is deleted into the output workbook.",
                outcome_schema={},
                required_paths=("/final_state/rows",),
                rules=(OutcomeRule(path="/final_state/rows", operator="equals", value=6),),
                allowed_tools=("edit_workbook",),
            ),
            probe_sha256=sha256_json({"probe": "xlsx-row-count-v1"}),
        )
    return World(
        kind=kind,
        instruction="Close ticket 1",
        correct=ToolCall(call_id="close", name="close_ticket", arguments={"ticket": 1}),
        mistake=AssistantAction(
            tool_calls=(
                ToolCall(
                    call_id="dup",
                    name="open_ticket",
                    arguments={"title": "Duplicate", "priority": "low"},
                ),
            )
        ),
        verifier=TaskVerifier(
            description="Ticket 1 is closed and no other ticket operation is used.",
            outcome_schema={},
            required_paths=("/final_state/status",),
            rules=(OutcomeRule(path="/final_state/status", operator="equals", value="closed"),),
            allowed_tools=("close_ticket",),
        ),
        probe_sha256=sha256_json({"probe": "ticket-status-v1"}),
    )


def _runtime(kind: str) -> ExecutableWorldRuntime:
    """Return a fresh runtime of one world kind."""
    return workbook_case().runtime if kind == "workbook" else ticket_case().runtime


def continuation_runtime(project: ProjectStore, configuration: JsonObject) -> ContinuationRuntime:
    """Restore either world kind for a trusted runtime file, as an operator's factory would."""
    kind = configuration["world"]
    if kind == "workbook":
        reference = ArtifactInput.model_validate(configuration["checkpoint"])
        checkpoint = load_continuation_checkpoint(project.artifacts, reference)
        return ContinuationRuntime(
            workbook_runtime(checkpoint.world), workbook_probe, world("workbook").probe_sha256
        )
    return ContinuationRuntime(ticket_case().runtime, ticket_probe, world("tickets").probe_sha256)


def capture(store: ArtifactStore, kind: str, lineage: str) -> ArtifactInput:
    """Persist the initial actor/world checkpoint of one lineage's task."""
    identity = f"authored-{kind}-{lineage}"
    if not store.exists(identity):
        store.write_json(
            artifact_id=identity,
            artifact_type="authored-task",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=datetime.now(UTC), code_revision="test"
            ),
            files={"task.json": {"origin": "authored cascade fixture, not benchmark evidence"}},
        )
    source = store.read(identity).manifest
    base = workbook_case().task if kind == "workbook" else ticket_case().task
    task = base.model_copy(
        update={
            "task_id": f"{kind}-{lineage}",
            "lineage_group_id": lineage,
            "instruction": f"{world(kind).instruction} [{lineage}].",
        }
    )
    actor = ChatAgentRuntime(maximum_model_calls=ACTOR_CALLS)
    with _runtime(kind).open(task) as session:
        checkpoint = ContinuationCheckpoint(
            task=task,
            actor=actor.initial_checkpoint(task),
            world=session.checkpoint(),
            sources=(artifact_input(source),),
        )
    return persist_continuation_checkpoint(
        store, checkpoint, created_at=datetime.now(UTC), code_revision="test"
    )


def lineage_of(text: str) -> str:
    """Recover the lineage name appended to a task instruction."""
    found = re.search(r"\[([a-z0-9-]+)\]", text)
    assert found is not None
    return found.group(1)


def main_action(kind: str, messages: list[JsonObject], forced: str | None, turn: int) -> JsonObject:
    """Decide the scripted main model's next action from the visible conversation.

    A forced tool returns that tool's correct call; a forced final answer, or any turn after a
    tool result, answers done. Otherwise even initial turns act correctly and odd ones err.
    """
    spec = world(kind)
    if forced == "none" or messages[-1]["role"] == "tool":
        return {"content": "done"}
    if forced is not None and forced != spec.correct.name:
        call = next(c for c in spec.mistake.tool_calls if c.name == forced)
        return {"call": call.model_dump(mode="json")}
    if forced is not None or turn % 2 == 0:
        return {"call": spec.correct.model_dump(mode="json")}
    if spec.mistake.tool_calls:
        return {"call": spec.mistake.tool_calls[0].model_dump(mode="json")}
    return {"content": spec.mistake.content}


def _action(value: JsonObject) -> AssistantAction:
    """Convert a scripted action description into a canonical assistant action."""
    call = value.get("call")
    if isinstance(call, dict):
        return AssistantAction(tool_calls=(ToolCall.model_validate(call),))
    content = value.get("content")
    assert isinstance(content, str)
    return AssistantAction(content=content)


class ScriptedMain:
    """In-process main model for fit sources, alternating correct and mistaken branches."""

    def __init__(self, kind: str, model: ModelSnapshot) -> None:
        """Bind the world kind and the catalog identity every response carries."""
        self._kind, self._model, self._turns = kind, model, 0
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Answer one actor request; only initial turns advance the alternation."""
        self.requests.append(request)
        messages = [message.model_dump(mode="json") for message in request.messages]
        action = main_action(self._kind, messages, None, self._turns)
        if messages[-1]["role"] != "tool":
            self._turns += 1
        return ModelResponse(
            output=_action(action), model=self._model, economics=OperationEconomics()
        )


def catalog_model(project: ProjectStore, alias: str) -> ModelSnapshot:
    """Return one alias's frozen catalog snapshot without credentials."""
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path), environment={})
    return catalog.snapshot(alias)[0]


def main_study(
    project: ProjectStore, kind: str, lineage: str, *, repeats: int = 2
) -> ArtifactInput:
    """Run one development study of the main model and return its saved result."""
    checkpoint = capture(project.artifacts, kind, lineage)
    spec = ContinuationStudySpec(
        checkpoint=checkpoint,
        actor_model=catalog_model(project, "world"),
        verifier=world(kind).verifier,
        probe_sha256=world(kind).probe_sha256,
        repeats=repeats,
        selector="all initial checkpoints; cascade fixture",
    )
    runtime = continuation_runtime(
        project, {"world": kind, "checkpoint": checkpoint.model_dump(mode="json")}
    )
    run_continuation_study(
        project.artifacts,
        spec,
        actor=ChatAgentRuntime(maximum_model_calls=ACTOR_CALLS),
        model=ScriptedMain(kind, spec.actor_model),
        world=runtime.world,
        probe=runtime.probe,
        probe_sha256=runtime.probe_sha256,
        code_revision="test",
    )
    return artifact_input(project.artifacts.read(study_identity(spec, "test") + "-result").manifest)


class ScriptedProviders(JsonHttpTransport):
    """Answer the main model, the small drafter and System One by route and visible state.

    Attributes:
        kind: World kind whose scripted actions are returned.
        initial: Per-lineage System One answer at the initial state: (choice, probability).
        requests: Every routed request as (route, payload).
    """

    def __init__(self, kind: str, initial: Mapping[str, tuple[str, float]] | None = None) -> None:
        """Bind the world kind and the decider's initial-state answers (default: correct, .95)."""
        self.kind, self.initial = kind, dict(initial or {})
        self.requests: list[tuple[str, JsonObject]] = []
        self._turns: dict[str, int] = {}

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Route one POST to the scripted owner of its wire."""
        del headers, timeout_seconds
        if url.endswith("/systemone"):
            self.requests.append(("systemone", payload))
            return JsonHttpResponse(status_code=200, body=self._systemone(payload))
        model = payload["model"]
        self.requests.append((str(model), payload))
        messages = payload["messages"]
        assert isinstance(messages, list)
        rows = [row for row in messages if isinstance(row, dict)]
        lineage = lineage_of(json.dumps(rows))
        choice = payload.get("tool_choice")
        forced = (
            choice
            if choice == "none"
            else choice["function"]["name"]
            if isinstance(choice, dict) and isinstance(choice.get("function"), dict)
            else None
        )
        if model == "judge-v1":
            action = main_action(self.kind, rows, None, 0)
        else:
            turn = self._turns.get(lineage, 0)
            action = main_action(self.kind, rows, forced if isinstance(forced, str) else None, turn)
            if rows[-1]["role"] != "tool":
                self._turns[lineage] = turn + 1
        return JsonHttpResponse(status_code=200, body=_chat(str(model), action))

    def _systemone(self, payload: JsonObject) -> JsonObject:
        """Answer the next-action choice from the conversation's latest turn."""
        state = payload["state"]
        questions = payload["questions"]
        assert isinstance(state, dict) and isinstance(questions, dict)
        criteria = questions["next_action"]["criteria"]
        conversation = state["conversation"]
        assert isinstance(criteria, dict) and isinstance(conversation, list)
        latest = conversation[-1]
        assert isinstance(latest, dict)
        if latest["role"] == "tool":
            choice, probability = RESPOND, 0.9
        else:
            lineage = lineage_of(json.dumps(conversation))
            choice, probability = self.initial.get(lineage, (world(self.kind).correct_key, 0.95))
        probabilities: dict[str, JsonValue] = {key: 0.0 for key in criteria}
        probabilities[choice] = probability
        probabilities[OTHER] = round(1 - probability, 6)
        return {
            "answers": {
                "next_action": {
                    "type": "choice",
                    "choice": choice,
                    "confidence": probability,
                    "probabilities": probabilities,
                }
            }
        }


def _chat(model: str, action: JsonObject) -> JsonObject:
    """Encode one scripted action as an OpenAI-compatible chat completion."""
    call = action.get("call")
    if isinstance(call, dict):
        message: JsonObject = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call["call_id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
                }
            ],
        }
        finish = "tool_calls"
    else:
        message, finish = {"role": "assistant", "content": action["content"]}, "stop"
    return {
        "id": "scripted",
        "model": model,
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
    }


def project_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    """Configure the shared main, small and System One catalog with a fixture credential."""
    write_catalog(tmp_path)
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    return ProjectStore(tmp_path, "case")


@pytest.mark.parametrize("kind", WORLDS)
def test_reached_states_carry_the_main_choice_and_verified_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Both branches share the initial state; the mistake is labelled by its failure."""
    project = project_fixture(tmp_path, monkeypatch)
    study = main_study(project, kind, "d1")
    points = extract_decision_points(
        project.artifacts, (study,), main_model=catalog_model(project, "world")
    )
    spec = world(kind)
    assert [(row.branch, row.outcome) for row in points.trajectories] == [(0, "pass"), (1, "fail")]
    firsts = [next(p for p in points.points if p.branch == branch) for branch in (0, 1)]
    assert [p.main_choice for p in firsts] == [spec.correct_key, spec.mistake_key]
    assert firsts[0].state_sha256 == firsts[1].state_sha256
    assert firsts[0].call != firsts[1].call
    labels = points.labels(by="decision")
    world_id, state = firsts[0].checkpoint.artifact_id, firsts[0].state_sha256
    assert labels[world_id, state, spec.correct_key] == "verified_correct"
    assert labels[world_id, state, spec.mistake_key] == "verified_error"
    assert points.lineages == ("d1",)
    assert all(point.candidates[-1] == RESPOND for point in points.points)
    if kind == "tickets":
        mistakes = [p for p in points.points if p.branch == 1]
        assert {p.policy for p in mistakes} == {"fail"}


def test_studies_of_other_models_or_with_a_cascade_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A named study is refused unless it is the main model's own, cascade-free trajectory."""
    project = project_fixture(tmp_path, monkeypatch)
    study = main_study(project, "tickets", "d1")
    with pytest.raises(ValueError, match="main-model-only"):
        extract_decision_points(
            project.artifacts, (study,), main_model=catalog_model(project, "judge")
        )
    with pytest.raises(ValueError, match="not a continuation study"):
        extract_decision_points(
            project.artifacts,
            (study.model_copy(update={"artifact_id": "authored-tickets-d1"}),),
            main_model=catalog_model(project, "world"),
        )


def points_for(project: ProjectStore, kind: str, lineages: tuple[str, ...]) -> DecisionPointSet:
    """Run one main-model study per lineage and label every reached state."""
    studies = tuple(main_study(project, kind, lineage) for lineage in lineages)
    return extract_decision_points(
        project.artifacts, studies, main_model=catalog_model(project, "world")
    )


def task_of(project: ProjectStore, reference: ArtifactInput) -> TaskCase:
    """Return the public task of a saved checkpoint."""
    return load_continuation_checkpoint(project.artifacts, reference).task
