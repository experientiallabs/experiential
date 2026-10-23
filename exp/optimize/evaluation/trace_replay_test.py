"""Replay boundaries and content-addressed run identities."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue, SourceIdentity
from exp.common.traces import Trace, TraceSource, TraceSpan
from exp.optimize.evaluation.fidelity_test import source_project
from exp.optimize.evaluation.trace_replay import content_run_id, recorded_exchanges, replay_boundary

NOW = datetime(2026, 10, 15, tzinfo=UTC)


def trace(context: JsonObject) -> Trace:
    """A one-span trace with a request context."""
    return Trace(
        trace_id="t",
        task="Read",
        initial_context=context,
        source=TraceSource(
            identity=SourceIdentity(kind="file", source_id="fixture"),
            semantic_convention_version="1.37.0",
        ),
        spans=(TraceSpan(span_id="s", name="chat", started_at=NOW, ended_at=NOW),),
    )


def test_replay_boundary_defaults_to_zero_and_refuses_bad_declarations() -> None:
    """``replay_from`` is optional; when declared it must be an in-range integer."""
    assert replay_boundary(trace({}), 3) == 0
    assert replay_boundary(trace({"replay_from": 2}), 3) == 2
    for bad in (-1, 4, True, "1"):
        with pytest.raises(ValueError, match="replay_from"):
            replay_boundary(trace({"replay_from": bad}), 3)


def test_run_id_addresses_content_but_not_deadline(tmp_path: Path) -> None:
    """Equal inputs share an ID whatever the deadline; any other change moves it."""
    _, spec = source_project(tmp_path)
    later = spec.model_copy(update={"deadline": spec.deadline + timedelta(hours=1)})
    changed = spec.model_copy(update={"maximum_output_tokens": 2048})
    assert content_run_id(spec, "held-out", "T 1") == content_run_id(later, "held-out", "T 1")
    assert content_run_id(spec, "held-out", "T 1") != content_run_id(changed, "held-out", "T 1")
    assert content_run_id(spec, "learn", "T 1", 3).startswith("learn-t-1-")
    assert content_run_id(spec, "learn", "T 1", 3).endswith("-r3")


@pytest.mark.parametrize("reply", ["", " \n", "  visible\n", None, False, 0])
def test_recorded_exchange_inventory_preserves_empty_results_and_prefix_offsets(
    reply: JsonValue,
) -> None:
    """A real reply must count as a measured exchange and as a public-prefix position."""
    source = trace({"replay_from": 1})
    common: JsonObject = {"gen_ai.tool.name": "read", "gen_ai.tool.call.id": "call"}
    source = source.model_copy(
        update={
            "spans": (
                TraceSpan(
                    span_id="z-call",
                    name="chat",
                    started_at=NOW,
                    ended_at=NOW,
                    attributes={**common, "gen_ai.tool.call.arguments": {"id": 7}},
                ),
                TraceSpan(
                    span_id="a-result",
                    name="read",
                    started_at=NOW,
                    ended_at=NOW,
                    attributes={
                        **common,
                        "gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.message": reply,
                    },
                ),
            )
        }
    )
    exchanges = recorded_exchanges(source, "lineage-a")
    assert len(exchanges) == 1
    assert exchanges[0].name == "read"
    assert exchanges[0].arguments == {"id": 7}
    assert type(exchanges[0].reply) is type(reply)
    assert exchanges[0].reply == reply
    assert replay_boundary(source, len(exchanges)) == 1
