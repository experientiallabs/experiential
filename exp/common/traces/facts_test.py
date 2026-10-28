"""The trace-package facts adapter, and the split of facts out of the request context."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject, SourceIdentity
from exp.common.tasks import TaskCase
from exp.common.tasks.facts import InstanceFactsError
from exp.common.tasks.facts_test import fact
from exp.common.traces import Trace, TraceSource, TraceSpan
from exp.common.traces.facts import TracePackageFacts

NOW = datetime(2026, 10, 16, tzinfo=UTC)


def trace(context: JsonObject) -> Trace:
    """A one-span trace whose request context is ``context``."""
    return Trace(
        trace_id="t1",
        task="Delete matching rows",
        initial_context=context,
        source=TraceSource(
            identity=SourceIdentity(kind="file", source_id="fixture"),
            semantic_convention_version="1.37.0",
        ),
        spans=(TraceSpan(span_id="s", name="chat", started_at=NOW, ended_at=NOW),),
    )


def task(source: str = "t1") -> TaskCase:
    """A task replaying ``source``."""
    return TaskCase(
        task_id="task",
        lineage_group_id="lineage",
        partition="fit",
        instruction="Delete matching rows",
        workload_weight=1,
        source_trace_ids=(source,),
    )


def test_package_facts_leave_the_request_context() -> None:
    """Facts move to their own field, never into request-visible context, and round-trip."""
    supplied: JsonObject = {"facts": [fact("input.xlsx/Sheet1/A2", "PQTW1")]}
    recorded = trace({"task_lineage": "x", "instance_facts": supplied})
    assert recorded.initial_context == {"task_lineage": "x"}
    assert recorded.instance_facts == supplied
    assert Trace.model_validate(recorded.model_dump(mode="json")) == recorded
    plain = trace({"task_lineage": "x"})
    assert "instance_facts" not in plain.model_dump(mode="json")
    with pytest.raises(ValidationError, match="must be a JSON object"):
        trace({"instance_facts": ["not", "an", "object"]})


def test_adapter_validates_on_read_and_serves_only_its_own_trace() -> None:
    """Validated facts come back typed; others are refused with the trace named."""
    supplied: JsonObject = {"facts": [fact("input.xlsx/Sheet1/A2", "PQTW1")]}
    facts = TracePackageFacts(trace({"instance_facts": supplied})).initial_facts(task())
    assert facts is not None and facts.facts[0].value == "PQTW1"
    assert TracePackageFacts(trace({})).initial_facts(task()) is None
    with pytest.raises(InstanceFactsError, match="does not come from it"):
        TracePackageFacts(trace({"instance_facts": supplied})).initial_facts(task("t2"))
    unvalidated: JsonObject = {"facts": [fact("a", status="unvalidated")]}
    with pytest.raises(InstanceFactsError, match="trace t1 supplies instance facts that are not"):
        TracePackageFacts(trace({"instance_facts": unvalidated})).initial_facts(task())
