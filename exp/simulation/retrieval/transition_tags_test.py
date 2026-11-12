"""Transition tags: authored keys, append-only bank, measured pruning, predicted recall."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.model import ModelRequest
from exp.common.project import ProjectStore
from exp.runtime.gateway.decisions_contracts import MAX_DECISION_QUESTIONS, DecisionRequest
from exp.simulation.mining.tag_cluster_names_test import SNAPSHOT, ScriptedNamer
from exp.simulation.retrieval.contracts import RAGTransition
from exp.simulation.retrieval.state_rag import RecallState
from exp.simulation.retrieval.state_tags import PreTurnState
from exp.simulation.retrieval.state_tags_test import transition
from exp.simulation.retrieval.transition_tags import (
    FocalGroup,
    TransitionAuthor,
    TransitionRAG,
    TransitionRetrievalPolicy,
    TransitionStep,
    TransitionTagBank,
    TransitionTagIndex,
    TransitionTagRow,
    UsefulnessPolicy,
    auc,
    author_transition_tags,
    evaluate_tags,
    load_transition_bank,
    load_transition_index,
    observation_overlap,
    parse_transition_tags,
    phi,
    plan_transition_rag_calls,
    prune_version,
    rank_transitions,
    save_transition_bank,
    save_transition_index,
    tag_transitions,
    transition_requests,
    transition_steps,
    transition_tag,
)

FAIL = transition_tag(
    "test_run_fails_after_edit",
    "a source file that was just edited",
    "runs the test suite",
    "a failing assertion",
)
WRITE = transition_tag(
    "write_echoed",
    "a workspace directory",
    "writes a file",
    "a confirmation that the file was written",
)
EMPTY = transition_tag("listing_empty", "an empty directory", "lists files", "an empty result")


def step(
    name: str, tool: str, observation: str | None, lineage: str = "", task: str = "Fix the parser"
) -> TransitionStep:
    """A step whose action names ``tool``; the observation is the true next observation."""
    return TransitionStep(
        transition_id=f"t-{name}",
        trace_id=f"trace-{name}",
        lineage_id=lineage or f"lineage-{name}",
        pre_state=PreTurnState(task=task, step_index=1),
        action={"kind": "tool_call", "tool_name": tool, "tool_arguments": {}, "content": None},
        observation=observation,
    )


def facts(rendered: JsonObject, tag_name: str, framing: str) -> float:
    """Fixture decision model: the observed framing reads the result, predicted reads the tool."""
    action = rendered["action"]
    assert isinstance(action, dict)
    tool = action["tool_name"]
    observation = str(rendered.get("next_observation", ""))
    if framing == "observed":
        hit = {
            "test_run_fails_after_edit": tool == "pytest" and "FAILED" in observation,
            "write_echoed": tool == "write" and "written" in observation,
            "listing_empty": tool == "ls" and observation == "",
        }[tag_name]
    else:
        hit = {
            "test_run_fails_after_edit": tool == "pytest",
            "write_echoed": tool == "write",
            "listing_empty": False,
        }[tag_name]
    return 0.92 if hit else 0.04


class Decisions:
    """Answers packed transition requests from the fixture facts, recording requests."""

    def __init__(self) -> None:
        """Start with no requests."""
        self.requests: list[DecisionRequest] = []

    def __call__(self, request: DecisionRequest) -> JsonObject:
        """Answer every question of one request."""
        self.requests.append(request)
        state = request.state
        assert isinstance(state, dict)
        steps = state["steps"]
        assert isinstance(steps, dict)
        answers: JsonObject = {}
        for question_id, question in request.questions.items():
            text = str(question.instructions)
            key = text.split("`")[1].split(".")[1]
            name = next(t.name for t in (FAIL, WRITE, EMPTY) if t.change in text)
            framing = "predicted" if "not shown" in text else "observed"
            rendered = steps[key]
            assert isinstance(rendered, dict)
            assert ("next_observation" in rendered) == (framing == "observed")
            answers[question_id] = {"type": "noul", "noul": facts(rendered, name, framing)}
        return {"answers": answers, "usage": {"cost": 0.0001}}


def test_tag_identity_is_stable_and_questions_describe_the_transition() -> None:
    """Intent: the same clauses give the same tag; the predicted framing hides the result."""
    again = transition_tag(
        "other_name", *(" ".join(c.split()) for c in (FAIL.exists, FAIL.action, FAIL.change))
    )
    assert again.tag_id == FAIL.tag_id
    observed = FAIL.question("observed", "steps.s0")
    predicted = FAIL.question("predicted", "steps.s0")
    assert "`steps.s0.next_observation` shows a failing assertion" in observed
    assert "not shown" in predicted and "will show a failing assertion" in predicted


def test_requests_pack_several_steps_within_the_question_limit() -> None:
    """Intent: fan-out packs steps per request; the predicted framing never carries the result."""
    steps = [step(str(i), "pytest", "FAILED x") for i in range(25)]
    requests = transition_requests((FAIL, WRITE, EMPTY), steps, "observed")
    per = MAX_DECISION_QUESTIONS // 3
    assert len(requests) == -(-25 // per)
    assert all(len(request.questions) <= MAX_DECISION_QUESTIONS for request, _ in requests)
    legend = [row for _, rows in requests for row in rows]
    assert len(legend) == 75 and len({(t, g) for _, t, g in legend}) == 75
    predicted = transition_requests((FAIL,), [step("live", "pytest", None)], "predicted")
    assert "next_observation" not in json.dumps(predicted[0][0].state)
    with pytest.raises(ValueError, match="no observation"):
        transition_requests((FAIL,), [step("live", "pytest", None)], "observed")


def test_tagging_reads_packed_answers_back_to_each_step() -> None:
    """Intent: each answer lands on its own step and tag even when packed."""
    steps = [
        step("a", "pytest", "FAILED test_x"),
        step("b", "pytest", "3 passed"),
        step("c", "write", "file written"),
    ]
    decide = Decisions()
    answers, responses = tag_transitions((FAIL, WRITE), steps, "observed", decide)
    assert len(decide.requests) == 1 and len(responses) == 1
    assert answers["t-a"][FAIL.tag_id] > 0.5 and answers["t-b"][FAIL.tag_id] < 0.5
    assert answers["t-c"][WRITE.tag_id] > 0.5 and answers["t-c"][FAIL.tag_id] < 0.5


def test_the_bank_is_append_only_and_restores_any_version(tmp_path: Path) -> None:
    """Intent: retired tags stay recorded, nothing is re-added, and old versions restore."""
    first = TransitionTagBank(scope="family-x").appended(added=(FAIL, WRITE), environments=("f1",))
    second = first.appended(added=(EMPTY,), retired=(WRITE.tag_id,), environments=("f2",))
    assert [tag.name for tag in second.active()] == [FAIL.name, EMPTY.name]
    assert second.retired() == {WRITE.tag_id: 2}
    assert set(second.tags()) == {FAIL.tag_id, WRITE.tag_id, EMPTY.tag_id}
    assert second.at(1) == first and second.at(1).identity == first.identity
    assert second.versions[1].parent_sha256 == first.identity
    assert second.appended(added=(WRITE,)).versions[-1].added == ()  # never re-added
    with pytest.raises(ValueError, match="retires inactive"):
        second.appended(retired=(WRITE.tag_id,))
    store = ProjectStore(tmp_path, "fixture").artifacts
    one = save_transition_bank(store, first, code_revision="test")
    two = save_transition_bank(store, second, code_revision="test")
    assert one != two
    assert load_transition_bank(store, one) == first and load_transition_bank(store, two) == second


def test_parse_reads_complete_tags_and_drops_partial_ones() -> None:
    """Intent: a tag needs all three clauses."""
    reply = (
        'Here: {"tags": [{"name": "a", "exists": "x", "action": "y", "change": "z"}, '
        '{"name": "b", "exists": "x"}]}'
    )
    (tag,) = parse_transition_tags(reply)
    assert (tag.exists, tag.action, tag.change) == ("x", "y", "z")
    with pytest.raises(ValueError):
        parse_transition_tags('{"tags": []}')


def test_authoring_shows_contrasts_and_the_bank_and_never_repeats_a_tag() -> None:
    """Intent: the author sees focal steps, contrasts and results; known tags are not re-added."""

    def reply(request: ModelRequest) -> str:
        return json.dumps(
            {
                "tags": [
                    {
                        "name": FAIL.name,
                        "exists": FAIL.exists,
                        "action": FAIL.action,
                        "change": FAIL.change,
                    },
                    {
                        "name": EMPTY.name,
                        "exists": EMPTY.exists,
                        "action": EMPTY.action,
                        "change": EMPTY.change,
                    },
                ]
            }
        )

    client = ScriptedNamer(reply)
    bank = TransitionTagBank(scope="f").appended(added=(FAIL,))
    group = FocalGroup(
        focal=step("a", "pytest", "FAILED test_x"), contrasts=(step("b", "pytest", "3 passed"),)
    )
    tags, drafts = author_transition_tags(
        "f", (group,), TransitionAuthor(client=client, model=SNAPSHOT), bank=bank
    )
    assert [tag.tag_id for tag in tags] == [EMPTY.tag_id]
    assert len(drafts) == 4 and drafts[-1].variant == "reconcile"
    prompt = client.requests[0].messages[-1].content
    assert isinstance(prompt, str)
    assert "FAILED test_x" in prompt and "3 passed" in prompt and "contrast 0.0" in prompt
    assert "TAGS ALREADY IN THE BANK" in prompt and FAIL.change in prompt


def test_auc_and_phi() -> None:
    """Intent: the usefulness statistics behave at their edges."""
    assert auc([0.9, 0.8, 0.1], [True, True, False]) == 1.0
    assert auc([0.5, 0.5], [True, False]) == 0.5
    assert auc([0.1], [True]) is None
    assert phi([True, False, True, False], [True, False, True, False]) == pytest.approx(1.0)
    assert phi([True, True], [True, False]) == 0.0


def rows() -> list[TransitionTagRow]:
    """A sample: FAIL fires in two lineages, WRITE duplicates FAIL, EMPTY never fires."""
    out = []
    for index in range(10):
        hit = index < 3
        out.append(
            TransitionTagRow(
                transition_id=f"t-{index}",
                trace_id=f"trace-{index}",
                lineage_id=f"lineage-{index % 5}",
                answers={
                    FAIL.tag_id: 0.9 if hit else 0.1,
                    WRITE.tag_id: 0.9 if hit else 0.1,
                    EMPTY.tag_id: 0.05,
                },
            )
        )
    return out


def test_evaluation_prunes_rare_redundant_unpredictable_and_ungrounded_tags() -> None:
    """Intent: each bar of usefulness is measured and named in the verdict."""
    development = {
        "d1": ({FAIL.tag_id: 0.9, WRITE.tag_id: 0.2}, {FAIL.tag_id: 0.95, WRITE.tag_id: 0.9}),
        "d2": ({FAIL.tag_id: 0.1, WRITE.tag_id: 0.8}, {FAIL.tag_id: 0.05, WRITE.tag_id: 0.1}),
    }
    evidence = {
        row.tag_id: row
        for row in evaluate_tags(
            (FAIL, WRITE, EMPTY),
            rows(),
            UsefulnessPolicy(),
            development=development,
            grounding=lambda kept: 0.3 * (FAIL.tag_id in kept),
        )
    }
    assert evidence[FAIL.tag_id].verdict == "keep"
    assert evidence[FAIL.tag_id].predictability == 1.0
    assert evidence[FAIL.tag_id].grounding_gain == pytest.approx(0.3)
    assert evidence[FAIL.tag_id].lineages == 3 and evidence[FAIL.tag_id].fire_rate == 0.3
    assert any("redundant with" in r for r in evidence[WRITE.tag_id].reasons)
    assert any("predictability" in r for r in evidence[WRITE.tag_id].reasons)
    assert any("fire rate" in r for r in evidence[EMPTY.tag_id].reasons)
    bank = prune_version(
        TransitionTagBank(scope="f"), (FAIL, WRITE, EMPTY), tuple(evidence.values())
    )
    assert [tag.tag_id for tag in bank.active()] == [FAIL.tag_id]
    assert len(bank.versions[0].evidence) == 3  # pruned candidates stay on record
    bank = prune_version(bank, (), (evidence[FAIL.tag_id].model_copy(update={"verdict": "prune"}),))
    assert bank.active() == () and bank.retired() == {FAIL.tag_id: 2}


def test_ranking_matches_the_predicted_transition_and_excludes_own_lineage() -> None:
    """Intent: steps where the predicted transition happened outrank look-alikes."""
    policy = TransitionRetrievalPolicy(top_k=2)
    index = TransitionTagIndex(
        bank_sha256="0" * 64,
        bank_version=1,
        decision_model={"model": "fixture"},
        rows=tuple(rows()),
    )
    query = {FAIL.tag_id: 0.9, EMPTY.tag_id: 0.1}
    matches, _, eligible = rank_transitions(
        index, (FAIL.tag_id, EMPTY.tag_id), policy, query, ("lineage-0",)
    )
    assert eligible == 8
    assert [m.transition_id for m in matches] == ["t-1", "t-2"]
    assert all(m.agreement == 1.0 for m in matches)


def test_observation_overlap() -> None:
    """Intent: the free grounding proxy rewards shared content."""
    assert observation_overlap("FAILED test_parse", "FAILED test_parse") == 1.0
    assert observation_overlap("FAILED test_parse", "3 passed") == 0.0


def corpus() -> tuple[dict[str, RAGTransition], list[TransitionStep]]:
    """Two traces with a failing test run and a passing one, and their steps."""
    traces = {
        "fail": (
            transition("fail", 0, "write", "file written"),
            transition("fail", 1, "pytest", "FAILED test_x"),
        ),
        "pass": (
            transition("pass", 0, "write", "file written"),
            transition("pass", 1, "pytest", "3 passed"),
        ),
    }
    steps = [s for ordered in traces.values() for s in transition_steps(ordered)]
    return {t.transition_id: t for ordered in traces.values() for t in ordered}, steps


def test_live_recall_tags_the_prediction_and_records_a_receipt(tmp_path: Path) -> None:
    """Intent: the live step is asked in the predicted framing and matched to observed rows."""
    transitions, steps = corpus()
    assert steps[1].pre_state.step_index == 1 and steps[1].observation == "FAILED test_x"
    decide = Decisions()
    bank = TransitionTagBank(scope="f").appended(added=(FAIL, WRITE))
    answers, _ = tag_transitions(bank.active(), steps, "observed", decide)
    index = TransitionTagIndex(
        bank_sha256=bank.identity,
        bank_version=bank.version,
        decision_model={"model": "fixture"},
        rows=tuple(
            TransitionTagRow(
                transition_id=s.transition_id,
                trace_id=s.trace_id,
                lineage_id=s.lineage_id,
                answers=answers[s.transition_id],
            )
            for s in sorted(steps, key=lambda s: s.transition_id)
        ),
    )
    store = ProjectStore(tmp_path, "fixture").artifacts
    assert (
        load_transition_index(store, save_transition_index(store, index, code_revision="t"))
        == index
    )
    records: list[JsonObject] = []
    rag = TransitionRAG(
        bank,
        index,
        TransitionRetrievalPolicy(top_k=1),
        transitions,
        decide=decide,
        record=records.append,
    )
    live = PreTurnState(task="Fix the parser", step_index=3)
    matched, receipt = rag.retrieve(
        live,
        ("lineage-x",),
        action={"kind": "tool_call", "tool_name": "pytest", "tool_arguments": {}},
    )
    assert [t.transition_id for t in matched] == ["t-fail-1"]
    assert receipt.fired == (FAIL.tag_id,)
    assert receipt.transitions[-1].target == RecallState.RECORDED
    assert records[-1]["protocol"] == "transition-tags-v1"
    stale = bank.appended(added=(EMPTY,))
    with pytest.raises(ValueError, match="another bank version"):
        TransitionRAG(
            stale,
            index,
            TransitionRetrievalPolicy(),
            transitions,
            decide=decide,
            record=records.append,
        )


def test_index_grows_with_new_rows_and_tags() -> None:
    """Intent: a later round merges answers into the same index for the newer bank."""
    bank = TransitionTagBank(scope="f").appended(added=(FAIL,))
    base = rows()[:2]
    index = TransitionTagIndex(
        bank_sha256=bank.identity, bank_version=1, decision_model={}, rows=tuple(base)
    )
    newer = bank.appended(added=(EMPTY,))
    grown = index.grown(
        (base[0].model_copy(update={"answers": {EMPTY.tag_id: 0.7}}), rows()[5]), newer
    )
    assert grown.bank_version == 2 and len(grown.rows) == 3
    assert grown.rows[0].answers[EMPTY.tag_id] == 0.7 and FAIL.tag_id in grown.rows[0].answers


def test_call_plan_counts_packed_requests() -> None:
    """Intent: a round is priced before any call."""
    plan = plan_transition_rag_calls(
        authoring_calls=4, tags=8, sample_steps=5000, development_steps=200, live_steps=10
    )
    assert plan == {
        "authoring_completions": 4,
        "sample_decisions": 1250,
        "development_decisions": 100,
        "live_decisions": 10,
    }


def test_one_round_authors_measures_prunes_and_never_shows_development_steps() -> None:
    """Intent: the loop runs end to end; development lineages never reach the author or sample."""
    from exp.simulation.retrieval.transition_tags import RoundSpec, development_lineage, run_round

    def reply(request: ModelRequest) -> str:
        return json.dumps(
            {
                "tags": [
                    {"name": t.name, "exists": t.exists, "action": t.action, "change": t.change}
                    for t in (FAIL, WRITE, EMPTY)
                ]
            }
        )

    client = ScriptedNamer(reply)
    steps = []
    for index in range(40):
        tool, result = [
            ("pytest", "FAILED test_x"),
            ("pytest", "3 passed"),
            ("write", "file written"),
        ][index % 3]
        steps.append(
            step(str(index), tool, result, lineage=f"lineage-{index}", task=f"<task-{index}>")
        )
    spec = RoundSpec(sample_steps=30, development_steps=10, development_share=0.3, focal_steps=3)
    decide = Decisions()
    result = run_round(
        "f",
        steps,
        TransitionTagBank(scope="f"),
        TransitionAuthor(client=client, model=SNAPSHOT),
        decide,
        spec,
        environments=("f",),
    )
    held = {s.transition_id for s in steps if development_lineage(s.lineage_id, spec)}
    assert held and set(result.development_ids) <= held
    prompts = " ".join(str(r.messages[-1].content) for r in client.requests)
    assert "<task-" in prompts
    assert not any(f"<task-{tid[2:]}>" in prompts for tid in held)
    assert all(row.transition_id not in held for row in result.rows)
    active = {t.tag_id for t in result.bank.active()}
    assert FAIL.tag_id in active and EMPTY.tag_id not in active  # EMPTY never fires
    assert len(result.bank.versions[0].evidence) == 3
