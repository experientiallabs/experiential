"""Replay boundaries and content-addressed run identities."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue, SourceIdentity
from exp.common.traces import Trace, TraceSource, TraceSpan
from exp.optimize.evaluation.fidelity import replay as trace_replay
from exp.optimize.evaluation.fidelity.replay import (
    content_run_id,
    protocol_prompts_sha256,
    recorded_exchanges,
    replay_boundary,
    reply_kind,
)
from exp.optimize.evaluation.fidelity.runner import FidelityMemory
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.simulation.retrieval import evidence_reading

NOW = datetime(2026, 10, 15, tzinfo=UTC)


def test_unused_reader_changes_cannot_invalidate_baseline_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = protocol_prompts_sha256()
    reading = evidence_reading.ReadingPolicy()
    enabled = protocol_prompts_sha256(reading)
    assert enabled != baseline
    monkeypatch.setattr(evidence_reading, "READER_PROMPT", "A different evidence question protocol")
    assert protocol_prompts_sha256() == baseline
    assert protocol_prompts_sha256(reading) != enabled


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
    code = "fixture"
    assert content_run_id(spec, "held-out", "T 1", code_revision=code) == content_run_id(
        later, "held-out", "T 1", code_revision=code
    )
    assert content_run_id(spec, "held-out", "T 1", code_revision=code) != content_run_id(
        changed, "held-out", "T 1", code_revision=code
    )
    assert content_run_id(spec, "learn", "T 1", 3, code_revision=code).startswith("learn-t-1-")
    assert content_run_id(spec, "learn", "T 1", 3, code_revision=code).endswith("-r3")


def test_a_measured_run_is_keyed_on_code_and_a_learning_run_is_not(tmp_path: Path) -> None:
    """Other code addresses another measured run; learning runs stay with their memory lineage."""
    _, spec = source_project(tmp_path)
    first = content_run_id(spec, "held-out", "t", code_revision="a" * 40)
    assert content_run_id(spec, "held-out", "t", code_revision="a" * 40) == first
    assert content_run_id(spec, "held-out", "t", code_revision="b" * 40) != first
    with pytest.raises(ValueError, match="code revision"):
        content_run_id(spec, "held-out", "t", code_revision=None)
    learning = spec.model_copy(
        update={"memory": FidelityMemory(lineage="learn-fixture", mode="learn")}
    )
    learned = content_run_id(learning, "learn", "t", code_revision=None)
    assert content_run_id(learning, "learn", "t", code_revision="b" * 40) == learned


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


def test_native_prompt_change_addresses_a_new_native_run_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing native guidance cannot silently replay results from an older prompt."""
    _, spec = source_project(tmp_path)
    native = spec.model_copy(
        update={
            "task": spec.task.model_copy(
                update={"initial_context": {"native_artifact_binding": {"fixture": True}}}
            )
        }
    )
    before = content_run_id(native, "native", "t", code_revision="fixture")
    plain = content_run_id(spec, "plain", "t", code_revision="fixture")
    monkeypatch.setattr(trace_replay, "NATIVE_ARTIFACT_PLAN_GUIDANCE", "different behavior")
    assert content_run_id(native, "native", "t", code_revision="fixture") != before
    assert content_run_id(spec, "plain", "t", code_revision="fixture") == plain


@pytest.mark.parametrize(
    ("reply", "kind"),
    [
        ("", "trivial"),
        (None, "trivial"),
        ({}, "trivial"),
        ("  ok\n", "trivial"),
        ({"status": "cancelled"}, "trivial"),
        (0, "trivial"),
        ("x" * 59, "trivial"),
        ("x" * 60, "text"),
        ("The order was cancelled and a refund of $42.10 was issued to the card.", "text"),
        (
            {"order_id": "#W1001", "items": [{"name": "lamp", "price": 42.1}]},
            "structured",
        ),
        (["a" * 30, "b" * 30], "structured"),
    ],
)
def test_reply_kinds_separate_short_fixed_outputs_from_text_and_structured_replies(
    reply: JsonValue, kind: str
) -> None:
    """Short or empty replies are trivial; longer strings are text; longer JSON is structured."""
    assert reply_kind(reply) == kind
