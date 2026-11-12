"""Offline tests for TraceML step labels, pivots, returns, tier comparison and the artifact."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject, SourceIdentity
from exp.common.models import (
    AssistantAction,
    BillingSource,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
    ToolCall,
)
from exp.common.project import ProjectStore
from exp.common.project.paths import ProjectPaths
from exp.common.traces import Trace, TraceSource, TraceSpan
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.questions import Question, QuestionBank, QuestionBankError
from exp.simulation.mining.tagging import DecisionCache
from exp.simulation.mining.traceml import (
    ACTION_LABELS,
    STEP_VIEW,
    STRATEGY_QUESTIONS,
    TRACEML_ARTIFACT_TYPE,
    TraceTrajectory,
    candidate_points,
    label_traces,
    load_traceml_labels,
    persist_traceml_labels,
    require_traceml_bank,
    step_units,
    traceml_bank,
    trajectory_from_trace,
    trajectory_from_workflow_steps,
    workflow_step_turns,
)
from exp.simulation.mining.traceml_moves import (
    DEFAULT_MOVE_VARIANTS,
    MOVE_TOOL,
    MOVE_VARIANTS,
    MoveJudgeError,
    MoveWindowSpec,
)

_SNAPSHOT = ModelSnapshot(
    provider="fixture",
    model_id="decisions-fixture",
    billing_source=BillingSource.CUSTOMER_MANAGED,
    capabilities_sha256="a" * 64,
    connection_sha256="b" * 64,
)

_TOOL_LABELS = {
    "search": "explore",
    "open": "read",
    "edit": "implement",
    "run_tests": "test",
    "revert": "recover",
}
"""Which action label the scripted decision model gives each fixture tool."""


def _bank(*, strategy: bool = True) -> QuestionBank:
    """Return a step bank with every action column and, optionally, the strategy columns."""
    questions = [
        Question(question_id=f"action.{label}", family="action", text=f"Is the step {label}?")
        for label in ACTION_LABELS
    ]
    if strategy:
        questions.extend(
            Question(question_id=name, family="strategy", text=f"Does {name} hold?")
            for name in STRATEGY_QUESTIONS
        )
    return QuestionBank(bank_id="traceml-action-v1", version=1, questions=tuple(questions))


def _focal(state: str) -> str:
    """Return the focal-step section of a rendered step view."""
    return state.split("## FOCAL STEP", 1)[1]


@dataclass
class _Scripted:
    """A decision boundary labelling each step from the tool named in its focal section.

    Attributes:
        refuse: Trajectory-identifying substring whose steps get a response with no answers.
        requests: Every request received, in order.
    """

    refuse: str | None = None
    requests: list[DecisionRequest] = field(default_factory=list)

    def __call__(self, request: DecisionRequest) -> JsonObject:
        """Answer every asked proposition about one step view."""
        self.requests.append(request)
        focal = _focal(str(request.state))
        if self.refuse is not None and self.refuse in focal:
            return {"usage": {"cost": 0.0}}
        if "call ponder" in focal:
            wanted, level = None, 0.5
        elif "agent message" in focal:
            wanted, level = ("ask" if "?" in focal else "give_up"), 0.9
        else:
            tool = focal.split("call ", 1)[1].split()[0]
            wanted, level = _TOOL_LABELS[tool], 0.9
        answers: JsonObject = {}
        for question_id in request.questions:
            label = question_id.removeprefix("action.")
            if question_id.startswith("action."):
                value = level if wanted is None or label == wanted else 0.05
            else:
                value = 0.1
            answers[question_id] = {"type": "noul", "noul": value}
        return {"answers": answers, "usage": {"input_tokens": 10, "cost": 0.001}}


def _step(
    name: str, arguments: JsonObject, *, status: str = "valid", after: JsonValue = None
) -> JsonObject:
    """Return one saved continuation step mapping."""
    row: JsonObject = {"action": {"name": name, "arguments": arguments}, "status": status}
    row["reply"] = {"ok": True}
    if after is not None:
        row["after"] = after
    return row


def _research(tier: str = "glm-5.3", trace_id: str = "attempt-0") -> TraceTrajectory:
    """A trajectory that explores, edits, fails a test, pivots, then returns to its first file.

    Steps (0-based): 0 search src, 1 open a.py, 2 edit a.py, 3 run_tests fails, 4 search docs
    (pivot after the failure), 5 open b.md, 6 edit a.py again (approach return to step 2),
    7 revert, restoring the state step 1 left (state return).
    """
    steps = [
        _step("search", {"query": "src"}, after={"v": 0}),
        _step("open", {"path": "a.py"}, after={"v": 0}),
        _step("edit", {"path": "a.py"}, after={"v": 1}),
        _step("run_tests", {"target": "all"}, status="failure", after={"v": 1}),
        _step("search", {"query": "docs"}, after={"v": 1}),
        _step("open", {"path": "b.md"}, after={"v": 1}),
        _step("edit", {"path": "a.py"}, after={"v": 2}),
        _step("revert", {"path": "a.py"}, after={"v": 0}),
    ]
    return trajectory_from_workflow_steps(
        trace_id,
        tier,
        steps,
        task="Fix the failing build.",
        turns=range(len(steps)),
        locator_prefix="attempt:0/step",
        initial_state={"v": 0},
    )


def _tagger(decide: _Scripted, cache: DecisionCache | None = None) -> DecisionsTraceTagger:
    """Return the real decisions tagger over a scripted boundary."""
    return DecisionsTraceTagger(decide, _SNAPSHOT, cache=cache, maximum_concurrency=2)


Claim = tuple[str, str, str | None]
"""One scripted move: kind, step locator, return target locator (None for a pivot)."""


@dataclass
class _Judge:
    """A move judge answering from scripted claims per variant, saving responses by identity.

    Attributes:
        claims: Moves each variant claims, keyed by variant; a variant missing here claims none.
        broken: Variants whose first draw cites a step outside the window.
        refuse: Variants whose draws cannot be bought.
        requests: Every request actually answered (a saved identity replays without one).
        saved: Responses by draw identity.
    """

    claims: dict[str, list[Claim]] = field(default_factory=dict)
    broken: frozenset[str] = frozenset()
    refuse: frozenset[str] = frozenset()
    requests: list[ModelRequest] = field(default_factory=list)
    saved: dict[str, ModelResponse] = field(default_factory=dict)

    def __call__(self, request: ModelRequest, *, identity: str) -> ModelResponse:
        """Replay a saved draw or answer the window from the scripted claims."""
        if identity in self.saved:
            return self.saved[identity]
        system = request.messages[0].content or ""
        variant = next(name for name, text in MOVE_VARIANTS.items() if text in system)
        if variant in self.refuse:
            raise MoveJudgeError("scripted outage")
        self.requests.append(request)
        payload = json.loads(request.messages[1].content or "{}")
        focal = payload["focal_locators"]
        repair = "previous_draft_problems" in payload
        moves: list[JsonValue] = [
            {
                "kind": kind,
                "step": step,
                "evidence": ["attempt:0/step:99"]
                if variant in self.broken and not repair
                else [step],
                "returns_to": target,
                "rationale": f"{kind} judged by {variant}",
            }
            for kind, step, target in self.claims.get(variant, [])
            if step in focal
        ]
        answer: JsonObject = {"judged_steps": focal, "moves": moves, "limitations": ["fixture"]}
        response = ModelResponse(
            output=AssistantAction(
                tool_calls=(ToolCall(call_id="moves", name=MOVE_TOOL, arguments=answer),)
            ),
            model=_SNAPSHOT,
            economics=OperationEconomics(),
        )
        self.saved[identity] = response
        return response


_RESEARCH_MOVES: list[Claim] = [
    ("pivot", "attempt:0/step:5", None),
    ("return", "attempt:0/step:7", "attempt:0/step:3"),
    ("return", "attempt:0/step:8", "attempt:0/step:2"),
]
"""What every stance judges in :func:`_research`: the pivot after the failed test, two returns."""


def _unanimous(claims: list[Claim]) -> _Judge:
    """A judge whose every variant claims the same moves."""
    return _Judge(claims={variant: list(claims) for variant in DEFAULT_MOVE_VARIANTS})


def test_steps_are_labelled_through_the_decisions_tagger_one_row_per_step(tmp_path: Path) -> None:
    """Every step is one step-view unit; a rerun replays the decision cache and buys nothing."""
    cache = DecisionCache(ProjectPaths(root=tmp_path, project_id="fixture"))
    decide = _Scripted()
    labels = label_traces(
        [_research()], tagger=_tagger(decide, cache), bank=_bank(), move_judge=None
    )

    trace = labels.traces[0]
    assert labels.view == STEP_VIEW
    assert [step.dominant for step in trace.steps] == [
        "explore",
        "read",
        "implement",
        "test",
        "explore",
        "read",
        "implement",
        "recover",
    ]
    assert trace.steps[3].failed and trace.steps[3].locator == "attempt:0/step:4"
    assert [step.boundary for step in trace.steps] == list(range(8))
    assert len(decide.requests) == 8
    assert all("FOCAL STEP" in str(request.state) for request in decide.requests)

    replay = _Scripted()
    again = label_traces(
        [_research()], tagger=_tagger(replay, cache), bank=_bank(), move_judge=None
    )
    assert replay.requests == []
    assert again == labels


def test_moves_are_judged_per_window_and_variant_kept_by_majority_and_replayed() -> None:
    """Every stance judges every window; agreed moves are kept with each stance's rationale."""
    judge = _unanimous(_RESEARCH_MOVES)
    labels = label_traces(
        [_research()], tagger=_tagger(_Scripted()), bank=_bank(), move_judge=judge
    )
    trace = labels.traces[0]

    assert len(judge.requests) == len(DEFAULT_MOVE_VARIANTS)
    assert trace.coverage.moves == "judged" and labels.move_variants == DEFAULT_MOVE_VARIANTS
    assert [move.step for move in trace.pivots] == [4]
    pivot = trace.pivots[0]
    assert pivot.boundary == 4 and pivot.locator == "attempt:0/step:5"
    assert pivot.support == pivot.judged == 3 and not pivot.uncertain
    assert pivot.triggers[0] == "evidence-first: pivot judged by evidence-first"
    returns = {move.step: move for move in trace.returns}
    assert returns[6].returns_to == 2 and returns[7].returns_to == 1
    request = json.loads(judge.requests[0].messages[1].content or "{}")
    focal = request["focal_steps"]
    assert focal[3]["status"] == "failed" and focal[3]["labels"]["dominant"] == "test"
    assert focal[7]["state"] == {
        "recorded": True,
        "changed": True,
        "same_state_as_after": ["initial", "attempt:0/step:1", "attempt:0/step:2"],
    }

    again = label_traces([_research()], tagger=_tagger(_Scripted()), bank=_bank(), move_judge=judge)
    assert again == labels and len(judge.requests) == len(DEFAULT_MOVE_VARIANTS)


def test_a_pivot_on_the_same_tool_and_target_is_judged() -> None:
    """The removed rules treated any step on the previous target as a retry, never a pivot.

    Here the agent edits the same file after a failed test, but abandons patching the parser for
    rewriting the config loader. Every stance judges that a pivot, and it is kept.
    """
    steps = [
        _step("edit", {"path": "a.py", "change": "patch the parser"}),
        _step("run_tests", {"target": "all"}, status="failure"),
        _step("edit", {"path": "a.py", "change": "rewrite the config loader instead"}),
    ]
    trajectory = trajectory_from_workflow_steps(
        "same-target",
        "kimi-k3",
        steps,
        task="Fix the build.",
        turns=range(len(steps)),
        locator_prefix="attempt:0/step",
    )
    judge = _unanimous([("pivot", "attempt:0/step:3", None)])
    labels = label_traces([trajectory], tagger=_tagger(_Scripted()), bank=_bank(), move_judge=judge)
    (pivot,) = labels.traces[0].pivots
    assert pivot.step == 2 and pivot.support == 3
    # The removed rules keyed a step's target by tool and its "path" argument: equal here.
    assert trajectory.steps[0].tool == trajectory.steps[2].tool == "edit"
    first, last = trajectory.steps[0].arguments, trajectory.steps[2].arguments
    assert isinstance(first, dict) and isinstance(last, dict) and first["path"] == last["path"]


def test_a_minority_claim_is_contested_and_a_failed_stance_marks_moves_uncertain() -> None:
    """One of three stances claiming a move is not enough; an outage leaves moves uncertain."""
    split = _Judge(
        claims={
            "evidence-first": [("pivot", "attempt:0/step:5", None)],
            "state-first": [
                ("pivot", "attempt:0/step:5", None),
                ("pivot", "attempt:0/step:2", None),
            ],
            "intent-first": [("pivot", "attempt:0/step:2", None)],
        }
    )
    trace = label_traces(
        [_research()], tagger=_tagger(_Scripted()), bank=_bank(), move_judge=split
    ).traces[0]
    assert [move.step for move in trace.pivots] == [1, 4]
    assert all(move.support == 2 and move.uncertain for move in trace.pivots)

    lone = _Judge(claims={"intent-first": [("pivot", "attempt:0/step:5", None)]})
    trace = label_traces(
        [_research()], tagger=_tagger(_Scripted()), bank=_bank(), move_judge=lone
    ).traces[0]
    assert trace.pivots == () and [move.step for move in trace.contested] == [4]

    outage = _Judge(
        claims={variant: [("pivot", "attempt:0/step:5", None)] for variant in MOVE_VARIANTS},
        refuse=frozenset({"intent-first"}),
    )
    trace = label_traces(
        [_research()], tagger=_tagger(_Scripted()), bank=_bank(), move_judge=outage
    ).traces[0]
    assert trace.coverage.moves == "judged"
    (pivot,) = trace.pivots
    assert pivot.support == pivot.judged == 2 and pivot.uncertain
    failed = [row for row in trace.judgments if row.answer is None]
    assert [row.variant for row in failed] == ["intent-first"]
    assert failed[0].failure is not None and "scripted outage" in failed[0].failure


def test_a_miscited_judgment_is_repaired_and_long_trajectories_are_windowed() -> None:
    """A citation outside the window is shown back once; each window is judged on its own."""
    judge = _Judge(
        claims={variant: list(_RESEARCH_MOVES) for variant in MOVE_VARIANTS},
        broken=frozenset({"state-first"}),
    )
    trace = label_traces(
        [_research()],
        tagger=_tagger(_Scripted()),
        bank=_bank(),
        move_judge=judge,
        move_window=MoveWindowSpec(focal_steps=5, context_steps=2),
    ).traces[0]
    assert [(row.window, row.variant, row.repairs) for row in trace.judgments] == [
        (0, "evidence-first", 0),
        (0, "state-first", 1),
        (0, "intent-first", 0),
        (1, "evidence-first", 0),
        (1, "state-first", 1),
        (1, "intent-first", 0),
    ]
    assert [move.step for move in trace.pivots] == [4]
    assert [move.step for move in trace.returns] == [6, 7]
    assert len(judge.requests) == 8
    repair = json.loads(judge.requests[2].messages[1].content or "{}")
    assert "unshown or later steps" in repair["previous_draft_problems"][0]
    second = json.loads(judge.requests[4].messages[1].content or "{}")
    assert [row["locator"] for row in second["context_steps"]] == [
        "attempt:0/step:4",
        "attempt:0/step:5",
    ]


def test_without_a_judge_moves_are_recorded_as_not_judged() -> None:
    """Labels are kept and no rule fills in the moves."""
    labels = label_traces([_research()], tagger=_tagger(_Scripted()), bank=_bank(), move_judge=None)
    trace = labels.traces[0]
    assert trace.pivots == trace.returns == trace.judgments == ()
    assert trace.coverage.moves == "not_judged" and labels.moves_protocol is None
    with pytest.raises(ValueError, match="move variants"):
        label_traces(
            [_research()],
            tagger=_tagger(_Scripted()),
            bank=_bank(),
            move_judge=None,
            move_variants=("rules",),
        )


def test_uncertain_steps_are_kept_and_counted() -> None:
    """A step answered inside the ambiguity band is kept, flagged and counted, never dropped."""
    steps = [_step("search", {"query": "x"}), _step("ponder", {"about": "y"})]
    trajectory = trajectory_from_workflow_steps(
        "unsure", "gemma-4-26b-a4b-it", steps, turns=range(len(steps))
    )
    labels = label_traces([trajectory], tagger=_tagger(_Scripted()), bank=_bank(), move_judge=None)

    step = labels.traces[0].steps[1]
    assert step.uncertain and set(step.uncertain_labels) == set(ACTION_LABELS)
    assert labels.coverage.labelled_steps == 2 and labels.coverage.uncertain_steps == 1


def test_a_refused_trace_and_an_empty_trace_are_recorded_in_the_coverage_manifest() -> None:
    """One trace failing to tag does not abort the others; every planned step is counted."""
    refused = trajectory_from_workflow_steps(
        "refused", "glm-5.3-flash", [_step("search", {"query": "poison"})], turns=(0,)
    )
    empty = TraceTrajectory(trace_id="empty", tier="glm-5.3-flash")
    labels = label_traces(
        [_research(), refused, empty],
        tagger=_tagger(_Scripted(refuse="poison")),
        bank=_bank(),
        move_judge=None,
    )

    coverage = labels.coverage
    assert coverage.planned_traces == 3 and coverage.labelled_traces == 1
    assert coverage.empty_traces == 1
    assert set(coverage.failed_traces) == {"refused"}
    assert "TraceTaggingError" in coverage.failed_traces["refused"]
    assert coverage.planned_steps == 9 and coverage.labelled_steps == 8
    assert coverage.tiers == {"glm-5.3": 1, "glm-5.3-flash": 2}


def test_tiers_are_compared_with_human_as_the_reference() -> None:
    """Distributions differ across tiers, so their divergence is positive."""
    explorer = trajectory_from_workflow_steps(
        "human-0", "human", [_step("search", {"query": str(n)}) for n in range(3)], turns=range(3)
    )
    labels = label_traces(
        [_research("mimo-v2.6-pro"), explorer],
        tagger=_tagger(_Scripted()),
        bank=_bank(),
        move_judge=_unanimous([("pivot", "attempt:0/step:5", None)]),
    )

    comparison = labels.tiers
    assert comparison.reference_tier == "human"
    profiles = {profile.tier: profile for profile in comparison.tiers}
    assert profiles["human"].dominant_share["explore"] == 1.0
    assert profiles["mimo-v2.6-pro"].dominant_share["implement"] == pytest.approx(0.25)
    assert profiles["mimo-v2.6-pro"].pivots_per_step == pytest.approx(1 / 8)
    (divergence,) = comparison.divergences
    assert 0.0 < divergence.jensen_shannon <= 1.0


def test_candidate_points_rank_moves_and_respect_the_offered_pool() -> None:
    """Points are the boundaries before pivots and returns, strongest first."""
    judge = _unanimous(_RESEARCH_MOVES)
    judge.claims["evidence-first"] = _RESEARCH_MOVES[:2]
    trace = label_traces(
        [_research()], tagger=_tagger(_Scripted()), bank=_bank(), move_judge=judge
    ).traces[0]

    assert candidate_points(trace) == (4, 6, 7)
    assert candidate_points(trace, include_uncertain=False) == (4, 6)
    assert candidate_points(trace, offered={0, 1, 6, 7}) == (6, 7)
    assert candidate_points(trace, limit=1) == (4,)


def test_the_labels_artifact_round_trips_and_is_idempotent(tmp_path: Path) -> None:
    """The saved record carries labels, probabilities, moves and locators."""
    project = ProjectStore(tmp_path, "case")
    labels = label_traces(
        [_research()],
        tagger=_tagger(_Scripted()),
        bank=_bank(),
        move_judge=_unanimous(_RESEARCH_MOVES),
    )

    reference = persist_traceml_labels(project.artifacts, labels, code_revision="test")
    assert persist_traceml_labels(project.artifacts, labels, code_revision="test") == reference
    assert project.artifacts.read(reference.artifact_id).manifest.artifact_type == (
        TRACEML_ARTIFACT_TYPE
    )
    loaded = load_traceml_labels(project.artifacts, reference)
    assert loaded == labels
    saved = json.loads(project.artifacts.read_bytes(reference.artifact_id, "labels.json"))
    step = saved["traces"][0]["steps"][0]
    assert step["locator"] == "attempt:0/step:1" and step["probabilities"]["explore"] == 0.9
    assert saved["traces"][0]["pivots"][0]["boundary"] == 4


def test_a_bank_without_every_action_column_is_refused(tmp_path: Path) -> None:
    """The step bank must ask one proposition per action label."""
    partial = QuestionBank(
        bank_id="traceml-action-v1",
        version=1,
        questions=(Question(question_id="action.explore", family="action", text="Explore?"),),
    )
    with pytest.raises(QuestionBankError, match="action.read"):
        require_traceml_bank(partial)
    rows = "\n".join(
        f'[[questions]]\nquestion_id = "{q.question_id}"\nfamily = "{q.family}"\n'
        f'text = "{q.text}"\n'
        for q in _bank().questions
    )
    (tmp_path / "traceml-action-v1.toml").write_text(
        f'bank_id = "traceml-action-v1"\nversion = 1\n{rows}', encoding="utf-8"
    )
    assert traceml_bank(tmp_path).question_ids() == _bank().question_ids()


def test_a_canonical_trace_becomes_tool_and_message_steps() -> None:
    """Calls pair with their results, failures mark the step, messages are their own steps."""
    started = datetime(2026, 10, 1, tzinfo=UTC)

    def span(index: int, attributes: JsonObject) -> TraceSpan:
        """Return one span at a fixed offset."""
        return TraceSpan(
            span_id=f"span-{index}",
            name="agent.step",
            started_at=started + timedelta(seconds=index),
            ended_at=started + timedelta(seconds=index + 1),
            attributes=attributes,
        )

    trace = Trace(
        trace_id="human-trace",
        task="Find the invoice.",
        spans=(
            span(
                0, {"gen_ai.tool.name": "search", "gen_ai.tool.call.arguments": '{"query": "inv"}'}
            ),
            span(
                1,
                {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "search",
                    "gen_ai.tool.message": '{"error": "no index"}',
                },
            ),
            span(2, {"gen_ai.completion": "Which month do you mean?"}),
        ),
        source=TraceSource(
            identity=SourceIdentity(kind="file", source_id="fixture"),
            semantic_convention_version="1.37.0",
        ),
    )
    trajectory = trajectory_from_trace(trace, "human")

    tool, message = trajectory.steps
    assert tool.kind == "tool" and tool.tool == "search" and tool.failed
    assert tool.arguments == {"query": "inv"} and tool.locator == "span:0"
    assert message.kind == "message" and message.boundary == 1 and message.locator == "span:2"
    units = step_units(trajectory)
    assert units[1].trace_id == "human-trace::span:2"
    assert "<result>" in units[1].text and "Which month" in _focal(units[1].text)
    labels = label_traces(
        [trajectory], tagger=_tagger(_Scripted()), bank=_bank(strategy=False), move_judge=None
    )
    assert [step.dominant for step in labels.traces[0].steps] == ["explore", "ask"]
    assert labels.traces[0].steps[0].strategy == {}


def _receipt(*calls: tuple[str, str], finish: str = "completed") -> JsonObject:
    """One saved actor call receipt making ``(name, call_id)`` calls in one response."""
    tool_calls: list[JsonValue] = [
        {"call_id": call_id, "name": name, "arguments": {}} for name, call_id in calls
    ]
    output: JsonObject = {"content": None if calls else "done", "tool_calls": tool_calls}
    return {"response": {"output": output, "finish_reason": finish}}


def _called(name: str, call_id: str) -> JsonObject:
    """One saved step for the call."""
    return {"action": {"call_id": call_id, "name": name, "arguments": {}}, "status": "valid"}


def test_steps_of_one_assistant_turn_share_its_point() -> None:
    """Parallel calls of one response are decided at the same point, the turn's index."""
    steps = [_called("read", "a"), _called("grep", "b"), _called("edit", "c"), _called("test", "d")]
    calls = [
        _receipt(("read", "a"), ("grep", "b")),
        _receipt(("edit", "c"), finish="length"),
        _receipt(("edit", "c"), ("test", "d")),
        _receipt(),
    ]
    # The length-stopped response dispatched nothing; its re-issued calls belong to turn 2.
    assert workflow_step_turns(steps, calls) == (0, 0, 2, 2)
    trajectory = trajectory_from_workflow_steps(
        "t", "kimi-k3", steps, turns=(0, 0, 2, 2), final_answer="done"
    )
    assert [step.boundary for step in trajectory.steps] == [0, 0, 2, 2, 3]
    # A prefix of the steps (a labelling bound) takes the prefix of the turns.
    assert workflow_step_turns(steps[:3], calls) == (0, 0, 2)


def test_steps_that_disagree_with_the_actor_calls_are_refused() -> None:
    """A step that matches no call, or the wrong one, has no known fork point."""
    calls = [_receipt(("read", "a"))]
    with pytest.raises(ValueError, match="disagree"):
        workflow_step_turns([_called("read", "z")], calls)
    with pytest.raises(ValueError, match="match no recorded actor call"):
        workflow_step_turns([_called("read", "a"), _called("edit", "b")], calls)
    with pytest.raises(ValueError, match="steps but 1 turns"):
        trajectory_from_workflow_steps("t", "kimi-k3", [_called("read", "a")] * 2, turns=(0,))
    with pytest.raises(ValueError, match="in order"):
        trajectory_from_workflow_steps("t", "kimi-k3", [_called("read", "a")] * 2, turns=(1, 0))
