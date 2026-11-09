"""Pre-turn state retrieval: matching states win over lexical look-alikes, held-out never enters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectStore
from exp.common.project.paths import ProjectPaths
from exp.common.state_machine import IllegalTransition, MachineRun
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.simulation.mining.tagging import DecisionCache
from exp.simulation.retrieval.contracts import RAGLineageBinding, RAGTransition
from exp.simulation.retrieval.embedding import HashingRAGEmbedder
from exp.simulation.retrieval.scope import CorpusScope
from exp.simulation.retrieval.state_rag import (
    RECALL_MACHINE,
    RecallState,
    StateEmbedder,
    StateRAG,
    StateRecallReceipt,
    StateRetrievalPolicy,
    StateTagIndex,
    build_state_tag_index,
    load_state_tag_index,
    plan_state_rag_calls,
    replay_state_recall,
    save_state_tag_index,
    state_agreement,
)
from exp.simulation.retrieval.state_tags import (
    PreTurnState,
    StateTagBank,
    cached_state_decider,
    pre_turn_steps,
)
from exp.simulation.retrieval.state_tags_test import ScriptedDecisions, bank, transition

LIVE_TASK = "Update the budget workbook"


def _hashing(texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
    """Embed with the provider-free hashing embedder."""
    return tuple(item.values for item in HashingRAGEmbedder().embed(texts))


HASHING = StateEmbedder(model={"embedder": "hashing-fixture"}, embed=_hashing)


def corpus() -> dict[str, tuple[RAGTransition, ...]]:
    """Three fit traces: a lexical look-alike, a state match on another task, and a twin."""
    return {
        # Same task and actions as the live step, but the write succeeded: no error now.
        "alike": (
            transition("alike", 0, "open_sheet", "Sheet Budget opened"),
            transition("alike", 1, "write_cell", "cell B2 written"),
            transition("alike", 2, "save", "saved"),
        ),
        # Another task entirely, but a sheet is open and the last call failed, as live.
        "match": (
            transition("match", 0, "open_sheet", "Ledger opened", task="Archive old invoices"),
            transition(
                "match", 1, "move_rows", "error: permission denied", task="Archive old invoices"
            ),
            transition("match", 2, "retry", "moved", task="Archive old invoices"),
        ),
        "twin": (
            transition("twin", 0, "open_sheet", "Sheet Budget opened"),
            transition("twin", 1, "write_cell", "cell B9 written"),
            transition("twin", 2, "save", "saved"),
        ),
    }


def steps(traces: dict[str, tuple[RAGTransition, ...]]) -> list[tuple[RAGTransition, PreTurnState]]:
    """Every step of every trace with its pre-turn state."""
    return [pair for rows in traces.values() for pair in pre_turn_steps(rows)]


def fit(*traces: str, held_out: Sequence[str] = ()) -> list[RAGLineageBinding]:
    """Bind each trace to its own lineage."""
    return [
        RAGLineageBinding(
            trace_id=trace,
            lineage_id=f"lineage-{trace}",
            partition="held_out" if trace in held_out else "fit",
        )
        for trace in (*traces, *held_out)
    ]


def live_state() -> PreTurnState:
    """The live pre-turn state: the budget sheet is open and the last write failed."""
    rows = (
        transition("live", 0, "open_sheet", "Sheet Budget opened"),
        transition("live", 1, "write_cell", "error: cell B2 locked"),
        transition("live", 2, "save", "pending"),
    )
    return pre_turn_steps(rows)[2][1]


def served(
    index_bank: StateTagBank,
    *,
    policy: StateRetrievalPolicy,
    embedder: StateEmbedder | None = None,
    decide: Callable[[DecisionRequest], JsonObject] | None = None,
) -> tuple[StateRAG, list[JsonObject], Callable[[DecisionRequest], JsonObject]]:
    """Build an index over the corpus and serve it, recording receipts."""
    traces = corpus()
    decisions = decide or ScriptedDecisions()
    index = build_state_tag_index(
        index_bank,
        steps(traces),
        fit(*traces),
        decide=decisions,
        decision_model={"model": "clef"},
        embedder=embedder,
    )
    receipts: list[JsonObject] = []
    transitions = {row.transition_id: row for rows in traces.values() for row in rows}
    rag = StateRAG(
        index, policy, transitions, decide=decisions, record=receipts.append, embedder=embedder
    )
    return rag, receipts, decisions


def test_matching_pre_turn_state_beats_a_lexically_similar_one() -> None:
    """The state match on another task outranks the look-alike the embedder would prefer."""
    rag, receipts, _ = served(bank(), policy=StateRetrievalPolicy(top_k=1), embedder=HASHING)
    live = live_state()
    look_alike = steps({"alike": corpus()["alike"]})[2][1]
    state_match = steps({"match": corpus()["match"]})[2][1]
    (live_vector, alike_vector, match_vector) = _hashing(
        [live.text(), look_alike.text(), state_match.text()]
    )
    lexical = [
        sum(a * b for a, b in zip(live_vector, other, strict=True))
        for other in (alike_vector, match_vector)
    ]
    assert lexical[0] > lexical[1], "the look-alike is lexically closer to the live step"

    (retrieved,), receipt = rag.retrieve(live, excluded_lineage_ids=("lineage-live",))
    assert retrieved.transition_id == "t-match-2"
    assert receipt.selected[0].agreement == 1.0 and receipt.selected[0].similarity is None
    assert receipt.similarities == {}, "no tie, so no embedding was consulted"
    assert receipt.usd == pytest.approx(0.0002)
    assert len([row for row in receipts if "selected" in row]) == 1
    assert [row.event for row in receipt.transitions] == ["tagged", "ranked", "recorded"]
    assert receipt.fired and receipt.eligible == 9


def test_embedding_only_breaks_exact_agreement_ties() -> None:
    """Twins with equal agreement are ordered by state similarity, never above a better match."""
    # After a successful write of B9: sheet open, no error. Every step that only has a sheet open
    # agrees fully; the twin's last step is the lexically closest, though it sorts last.
    rows = (
        transition("live", 0, "open_sheet", "Sheet Budget opened"),
        transition("live", 1, "write_cell", "cell B9 written"),
        transition("live", 2, "save", "saved"),
    )
    live = pre_turn_steps(rows)[2][1]
    plain, _, _ = served(bank(), policy=StateRetrievalPolicy(top_k=1))
    (unbroken,), plain_receipt = plain.retrieve(live)
    assert unbroken.transition_id == "t-alike-1" and plain_receipt.similarities == {}

    rag, _, _ = served(bank(), policy=StateRetrievalPolicy(top_k=1), embedder=HASHING)
    (broken,), receipt = rag.retrieve(live)
    assert broken.transition_id == "t-twin-2"
    tied = {"t-alike-1", "t-alike-2", "t-match-1", "t-twin-1", "t-twin-2"}
    assert set(receipt.similarities) == tied
    assert max(receipt.similarities, key=receipt.similarities.__getitem__) == "t-twin-2"
    assert receipt.selected[0].agreement == 1.0
    assert [row.event for row in receipt.transitions] == [
        "tagged",
        "ranked",
        "tie_broken",
        "recorded",
    ]


def test_recall_lifecycle_refuses_illegal_moves_and_records_failures() -> None:
    """A failed tagging ends in failed; moves outside the table are refused and recorded."""
    moves: list[JsonObject] = []
    run = MachineRun(RECALL_MACHINE, "step/fixture", moves.append)
    with pytest.raises(IllegalTransition, match="cannot apply 'recorded' in state pending"):
        run.fire("recorded")
    assert run.state is RecallState.PENDING and not run.history[-1].accepted
    assert moves == [{"transition": run.history[-1].model_dump(mode="json")}]
    answering = ScriptedDecisions()
    broken = {"now": False}

    def flaky(request: DecisionRequest) -> JsonObject:
        """Answer while building the index, then nothing."""
        return {"answers": {}} if broken["now"] else answering(request)

    rag, receipts, _ = served(bank(), policy=StateRetrievalPolicy(), decide=flaky)
    broken["now"] = True
    with pytest.raises(ValueError, match="no proposition answer"):
        rag.retrieve(live_state())
    last = receipts[-1]["transition"]
    assert isinstance(last, dict) and (last["event"], last["target"]) == ("tag_failed", "failed")
    assert "    ranked --> tie_broken: tie_broken" in RECALL_MACHINE.mermaid()


def test_uncertain_answers_count_less_than_confident_ones() -> None:
    """A confident shared tag outweighs a hesitant one."""
    policy = StateRetrievalPolicy()
    query = (0.95, 0.6)
    assert state_agreement(query, (0.95, 0.05), policy) > state_agreement(
        query, (0.05, 0.95), policy
    )
    assert state_agreement((0.05, 0.05), (0.05, 0.05), policy) == 0.0
    with pytest.raises(ValueError, match="certain_below"):
        StateRetrievalPolicy(certain_below=0.6)


def test_held_out_traces_are_refused_and_scope_and_own_lineage_are_respected() -> None:
    """Held-out traces never enter; a scope narrows the steps; the query's lineage is excluded."""
    traces = corpus()
    with pytest.raises(ValueError, match="held out"):
        build_state_tag_index(
            bank(),
            steps(traces),
            fit("alike", "twin", held_out=("match",)),
            decide=ScriptedDecisions(),
            decision_model={"model": "clef"},
        )
    with pytest.raises(ValueError, match="no lineage binding"):
        build_state_tag_index(
            bank(),
            steps(traces),
            fit("alike", "twin"),
            decide=ScriptedDecisions(),
            decision_model={"model": "clef"},
        )
    scope = CorpusScope.declare(
        name="state-retrieval",
        kind="lineage",
        ids=("lineage-alike", "lineage-twin"),
        source_plan="0" * 64,
    )
    scoped = build_state_tag_index(
        bank(),
        steps(traces),
        fit(*traces),
        decide=ScriptedDecisions(),
        decision_model={"model": "clef"},
        scope=scope,
    )
    assert {row.trace_id for row in scoped.rows} == {"alike", "twin"}
    assert scoped.corpus_scope == scope

    rag, _, _ = served(bank(), policy=StateRetrievalPolicy(top_k=9))
    matched, receipt = rag.retrieve(live_state(), excluded_lineage_ids=("lineage-match",))
    assert all(row.lineage_id != "lineage-match" for row in matched)
    assert receipt.eligible == 6


def test_receipts_replay_without_calls_and_the_index_round_trips(tmp_path: Path) -> None:
    """A receipt recomputes its selection offline; a cached step makes no decision call."""
    cache = DecisionCache(ProjectPaths(root=tmp_path, project_id="fixture"))
    decisions = ScriptedDecisions()
    traces = corpus()
    cached = cached_state_decider(cache, decisions, {"model": "clef"})
    index = build_state_tag_index(
        bank(),
        steps(traces),
        fit(*traces),
        decide=cached,
        decision_model={"model": "clef"},
        embedder=HASHING,
    )
    policy = StateRetrievalPolicy(top_k=2)
    receipts: list[JsonObject] = []
    transitions = {row.transition_id: row for rows in traces.values() for row in rows}
    rag = StateRAG(
        index, policy, transitions, decide=cached, record=receipts.append, embedder=HASHING
    )
    first, receipt = rag.retrieve(live_state())
    calls = len(decisions.requests)
    again, _ = rag.retrieve(live_state())
    assert again == first and len(decisions.requests) == calls
    recorded = StateRecallReceipt.model_validate(next(row for row in receipts if "selected" in row))
    assert replay_state_recall(index, policy, recorded) == receipt.selected
    with pytest.raises(ValueError, match="another index or policy"):
        replay_state_recall(index, StateRetrievalPolicy(top_k=3), recorded)
    tampered = recorded.model_copy(update={"probabilities": (0.05,) * 4})
    with pytest.raises(ValueError, match="differs from the recorded selection"):
        replay_state_recall(index, policy, tampered)

    store = ProjectStore(tmp_path, "fixture").artifacts
    reference = save_state_tag_index(store, index, code_revision="t")
    assert load_state_tag_index(store, reference) == index
    assert save_state_tag_index(store, index, code_revision="t") == reference
    with pytest.raises(ValueError, match="embedder differs"):
        StateRAG(
            index,
            policy,
            transitions,
            decide=cached,
            record=receipts.append,
            embedder=StateEmbedder(model={"embedder": "other"}, embed=_hashing),
        )
    with pytest.raises(ValueError, match="no matching transition"):
        StateRAG(index, policy, {}, decide=cached, record=receipts.append)


def test_call_plan_prices_authoring_corpus_and_live_steps() -> None:
    """Planned calls count drafts plus reconcile, one decision per state per 32 questions."""
    plan = plan_state_rag_calls(authoring_calls=5, questions=40, corpus_steps=9, live_steps=16)
    assert (plan.authoring_completions, plan.corpus_decisions, plan.live_decisions) == (5, 18, 32)
    assert plan.decisions == 50


def test_tagging_steps_concurrently_saves_the_same_index_as_one_at_a_time() -> None:
    """Concurrency only speeds tagging up: the saved index is identical."""
    traces = corpus()

    def build(concurrency: int):  # noqa: ANN202
        return build_state_tag_index(
            bank(),
            steps(traces),
            fit(*traces),
            decide=ScriptedDecisions(),
            decision_model={"model": "clef"},
            concurrency=concurrency,
        )

    assert build(8).identity == build(1).identity


def test_a_step_refused_at_the_credential_boundary_is_left_out_not_fatal() -> None:
    """One credential-bearing state drops out of the index; any other failure still stops it."""
    from exp.common.project.errors import ArtifactStoreError

    traces = corpus()
    all_steps = steps(traces)
    scripted = ScriptedDecisions()
    calls = [0]

    def guarded(request: DecisionRequest) -> JsonObject:
        calls[0] += 1
        if calls[0] == 1:  # steps are tagged in order one at a time: this is the first step
            raise ArtifactStoreError(
                "provider request cannot be retained at the credential boundary; "
                "remove credential-bearing content before dispatch"
            )
        return scripted(request)

    index = build_state_tag_index(
        bank(), all_steps, fit(*traces), decide=guarded, decision_model={"model": "clef"}
    )
    assert all_steps[0][0].transition_id not in {row.transition_id for row in index.rows}
    assert len(index.rows) == len(all_steps) - 1

    def broken(request: DecisionRequest) -> JsonObject:
        raise ArtifactStoreError("disk full")

    with pytest.raises(ArtifactStoreError, match="disk full"):
        build_state_tag_index(
            bank(), all_steps, fit(*traces), decide=broken, decision_model={"model": "clef"}
        )


def test_a_partial_index_records_its_skipped_steps_and_a_complete_one_keeps_its_identity() -> None:
    """F13: a replay can tell a credential-skipped index is partial; complete ones are unchanged.

    The count is a metadata field omitted when zero, so indexes saved before it (none skipped)
    keep their content identity.
    """
    from exp.common.project.errors import ArtifactStoreError

    traces = corpus()
    all_steps = steps(traces)
    complete = build_state_tag_index(
        bank(), all_steps, fit(*traces), decide=ScriptedDecisions(), decision_model={"model": "c"}
    )
    assert complete.skipped_steps == 0
    assert "skipped_steps" not in complete.model_dump(mode="json")
    assert (
        complete.identity == StateTagIndex.model_validate(complete.model_dump(mode="json")).identity
    )

    scripted, calls = ScriptedDecisions(), [0]

    def guarded(request: DecisionRequest) -> JsonObject:
        calls[0] += 1
        if calls[0] == 1:
            raise ArtifactStoreError(
                "provider request cannot be retained at the credential boundary; "
                "remove credential-bearing content before dispatch"
            )
        return scripted(request)

    partial = build_state_tag_index(
        bank(), all_steps, fit(*traces), decide=guarded, decision_model={"model": "c"}
    )
    assert partial.skipped_steps == 1
    replayed = StateTagIndex.model_validate_json(partial.model_dump_json())
    assert replayed.skipped_steps == 1 and replayed.identity == partial.identity


def test_a_step_whose_earlier_decision_has_an_unknown_outcome_is_left_out_not_re_bought() -> None:
    """A step an earlier run left with an unknown outcome is skipped; the run carries on."""
    from exp.simulation.mining.intents import abandoned_claim_refusal
    from exp.simulation.mining.tagging import TraceTaggingError

    traces = corpus()
    all_steps = steps(traces)
    scripted = ScriptedDecisions()
    calls = [0]

    def guarded(request: DecisionRequest) -> JsonObject:
        calls[0] += 1
        if calls[0] == 1:
            raise TraceTaggingError(abandoned_claim_refusal("a" * 64))
        return scripted(request)

    index = build_state_tag_index(
        bank(), all_steps, fit(*traces), decide=guarded, decision_model={"model": "clef"}
    )
    assert len(index.rows) == len(all_steps) - 1
