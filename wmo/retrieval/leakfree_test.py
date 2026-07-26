"""Tests for the shared leak-free demo retriever."""

from __future__ import annotations

from wmo.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmo.retrieval import EmbeddingRetriever, HashingEmbedder
from wmo.retrieval.leakfree import DemoRetriever


def _trace(tid: str, loc: str, n: int = 2) -> Trace:
    return Trace(
        trace_id=tid,
        steps=[
            Step(
                action=Action(kind=ActionKind.TOOL_CALL, name="f", arguments={"i": i, "loc": loc}),
                observation=Observation(content=f"{tid}-{i}"),
                state_before=EnvState(structured={"loc": loc}),
            )
            for i in range(n)
        ],
    )


def test_excludes_same_trace_demos() -> None:
    # Two distinct traces; a step's nearest neighbor is its own sibling, which must be excluded.
    a = _trace("trace-A", "shop")
    b = _trace("trace-B", "warehouse")
    demos = DemoRetriever(EmbeddingRetriever(HashingEmbedder(dim=128)), [a, b], top_k=2)

    a_ids = {id(s) for s in a.steps}
    for step in a.steps:
        got = demos.demos_for("trace-A", step)
        assert all(id(d) not in a_ids for d in got)  # no self/sibling leakage


def test_zero_shot_when_no_retriever() -> None:
    a = _trace("trace-A", "shop")
    demos = DemoRetriever(None, [a])
    assert demos.demos_for("trace-A", a.steps[0]) == []


def test_zero_shot_when_corpus_empty() -> None:
    a = _trace("trace-A", "shop")
    demos = DemoRetriever(EmbeddingRetriever(HashingEmbedder(dim=16)), [], top_k=3)
    assert demos.demos_for("trace-A", a.steps[0]) == []


def test_respects_top_k_after_exclusion() -> None:
    # Corpus has one other trace with 3 steps; top_k=2 should return 2 (all from the other trace).
    a = _trace("trace-A", "shop", n=1)
    b = _trace("trace-B", "warehouse", n=3)
    demos = DemoRetriever(EmbeddingRetriever(HashingEmbedder(dim=64)), [a, b], top_k=2)
    got = demos.demos_for("trace-A", a.steps[0])
    assert len(got) == 2
    assert all(d.observation.content.startswith("trace-B") for d in got)
