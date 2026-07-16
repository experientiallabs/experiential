"""Tests for corpus hygiene."""

from __future__ import annotations

from wmh.core.types import Action, ActionKind, Observation, Step, Trace
from wmh.ingest.quality import drop_degenerate_traces


def _trace(tid: str, contents: list[str]) -> Trace:
    return Trace(
        trace_id=tid,
        steps=[
            Step(
                action=Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "ls"}),
                observation=Observation(content=c),
            )
            for c in contents
        ],
    )


def test_drops_traces_whose_every_observation_is_empty() -> None:
    healthy = _trace("h", ["file.py", ""])  # one real observation keeps it
    degenerate = _trace("d", [""])  # the 1-step all-empty capture-failure shape
    whitespace = _trace("w", ["   \n"])  # whitespace-only counts as empty
    kept, dropped = drop_degenerate_traces([healthy, degenerate, whitespace])
    assert [t.trace_id for t in kept] == ["h"]
    assert dropped == 2


def test_keeps_everything_when_corpus_is_healthy() -> None:
    traces = [_trace("a", ["x"]), _trace("b", ["y", ""])]
    kept, dropped = drop_degenerate_traces(traces)
    assert kept == traces
    assert dropped == 0


def test_empty_corpus_is_safe() -> None:
    assert drop_degenerate_traces([]) == ([], 0)
