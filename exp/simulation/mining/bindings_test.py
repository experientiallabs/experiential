"""Tests for build-owned complete lineage assignments."""

from datetime import UTC, datetime
from pathlib import Path

from exp.common.core.artifacts import SourceIdentity
from exp.common.project import ArtifactStore
from exp.common.project.paths import ProjectPaths
from exp.common.traces import load_trace_dataset
from exp.common.traces.ingest.chat_json import CHAT_JSON_SOURCE
from exp.common.traces.ingest.vendor_trace import SYNTHETIC_TIME_ATTRIBUTE
from exp.simulation.build import build_task_set
from exp.simulation.mining.bindings import load_task_set_lineage_bindings
from exp.simulation.mining.service import MiningSpec


def test_build_records_true_unknown_timing_and_complete_distinct_lineage_bindings(
    tmp_path: Path,
) -> None:
    """Persisted evidence retains source timing flags and verifies the corrected boundaries."""
    normalized = CHAT_JSON_SOURCE.normalize(
        tuple(
            {
                "trace_id": str(index),
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": "Recorded answer."},
                ],
            }
            for index, instruction in enumerate(
                ("Count geese beside the river", "Factor the integer 729")
            )
        ),
        source=SourceIdentity(kind="file", source_id="chat-export"),
    )
    store = ArtifactStore(ProjectPaths(tmp_path, "lineage-evidence"))
    built = build_task_set(
        normalized,
        store,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="a" * 40,
        mining_spec=MiningSpec(fit_task_budget=2, held_out_task_budget=0),
    )

    bindings = load_task_set_lineage_bindings(store, built.task_set.task_set_id)
    saved = load_trace_dataset(store, built.trace_dataset.dataset.dataset_id)

    assert len({item.lineage_id for item in bindings.bindings}) == 2
    assert {item.trace_id for item in bindings.bindings} == {
        trace.trace_id for trace in normalized.traces
    }
    assert all(item.time_bucket is None for item in built.mining.lineage_assignments)
    assert all(trace.conversation_id is None for trace in saved.traces)
    assert all(
        span.attributes[SYNTHETIC_TIME_ATTRIBUTE] is True
        for trace in saved.traces
        for span in trace.spans
    )
