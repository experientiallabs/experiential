"""Native bootstrap step coverage and unchanged whole-trace semantics."""

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.models import ToolCall
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.variance.bootstrap_agents import FamilyKnowledge, StartMaterial
from exp.optimize.evaluation.variance.bootstrap_contracts import BootstrapTask
from exp.simulation.rules.contracts_test import call, result, trace
from exp.simulation.rules.model import RuleBook


def material() -> StartMaterial:
    """Build admitted material with one small native row and one oversized native row."""
    evidence = trace(
        "allowed",
        (
            call("c1", 0, "read", {"text": '雪"\\\n'}, "k1"),
            result("r1", 1, "read", {"value": "small"}, "k1"),
            call("c2", 2, "read", {}, "k2"),
            result("r2", 3, "read", {"value": "x" * 40000}, "k2"),
        ),
        declare=False,
    )
    knowledge = FamilyKnowledge(
        family_id="family",
        report=None,
        rulebook=ArtifactInput(artifact_id="unused", sha256="0" * 64),
        book=RuleBook(contracts={}),
        rule_memory=None,
        traces={"allowed": evidence},
        contracts={},
    )
    task = TaskCase(
        task_id="probe",
        lineage_group_id="lineage",
        origin="authored",
        partition="fit",
        instruction="Read evidence",
        tools=(),
        workload_weight=1,
    )
    return StartMaterial(knowledge, BootstrapTask(task=task), {})


def test_native_narrow_selection_preserves_whole_read_refusal_and_opening() -> None:
    """Caller selection returns exactly one complete row, without changing the full opening."""
    source = material()
    before = source.context()
    status, body, refs = source.run(
        ToolCall(
            call_id="s",
            name="read_trace_steps",
            arguments={"trace_id": "allowed", "start_index": 0, "count": 1},
        ),
        32000,
    )
    assert status == "ok" and isinstance(body, dict)
    assert set(refs) == {"trace:allowed:span:c1", "trace:allowed:span:r1"}
    assert "trace:allowed" not in refs and "trace:allowed:span:r2" not in refs
    rows = source.views["allowed"][0]["steps"]
    assert isinstance(rows, list) and body["steps"] == rows[:1]
    assert body["coverage"] == {
        "unit": "zero-based-step-index",
        "start_index": 0,
        "end_index_exclusive": 1,
        "total_steps": 2,
        "selected_range_complete": True,
        "all_steps_selected": False,
        "next_start_index": 1,
    }
    assert source.context() == before
    assert source.run(
        ToolCall(call_id="w", name="read_trace", arguments={"trace_id": "allowed"}), 32000
    )[0::2] == ("refused", ())
    assert source.run(
        ToolCall(
            call_id="w",
            name="read_trace",
            arguments={"trace_id": "allowed", "start_index": 0, "count": 1},
        ),
        32000,
    )[0::2] == ("refused", ())
    assert source.run(
        ToolCall(
            call_id="b",
            name="read_trace_steps",
            arguments={"trace_id": "allowed", "start_index": 1, "count": 1},
        ),
        32000,
    )[0::2] == ("refused", ())


@pytest.mark.parametrize(
    "arguments",
    [
        {"trace_id": "outside", "start_index": 0, "count": 1},
        {"trace_id": "allowed", "start_index": -1, "count": 1},
        {"trace_id": "allowed", "start_index": 1, "count": 2},
        {"trace_id": "allowed", "start_index": 2, "count": 1},
        {"trace_id": "allowed", "start_index": True, "count": 1},
        {"trace_id": "allowed", "start_index": 0, "count": 0},
        {"trace_id": "allowed", "start_index": 0, "count": "1"},
        {"trace_id": "allowed", "start_index": 0, "count": 1, "alias": "trace:outside"},
    ],
)
def test_native_unknown_scope_and_nonexact_ranges_refuse(arguments: JsonObject) -> None:
    """No clamping, aliases, unknown traces, coerced indexes or empty ranges are accepted."""
    assert material().run(
        ToolCall(call_id="s", name="read_trace_steps", arguments=arguments), 32000
    )[0::2] == ("refused", ())
