"""Tests for evidence references, derived views and world-learning-only retrieval.

Evidence packets themselves are exercised end to end through a real group in actor_critic_test.py.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject
from exp.common.models import ToolCall
from exp.common.project import ArtifactStore, artifact_input
from exp.optimize.evaluation.variance.evidence import (
    ReferenceSource,
    TraceSearchSpec,
    _step_labels,
    divergence_points,
    execute_evidence_tool,
    load_references,
    open_trace_search,
)
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.simulation.retrieval.contracts import RAGAction
from exp.simulation.retrieval.scope_test import _build, _scope
from exp.simulation.retrieval.tests.retrieval_test import _store

_LABELS: JsonObject = {
    "traces": [
        {
            "trace_id": "group-attempt-1",
            "steps": [
                {"index": 0, "locator": "attempt:1/step:1", "dominant": "gather"},
                {"index": 1, "locator": "attempt:1/step:2", "dominant": "act"},
                {"index": 2, "locator": "span:9", "dominant": "act"},
            ],
            "pivots": [{"kind": "pivot", "locator": "attempt:1/step:2", "boundary": 1}],
            "returns": [],
        }
    ]
}

_ROLES = {
    "lineage-0": "world_learning",
    "lineage-1": "mining_probes",
    "lineage-2": "world_learning",
    "lineage-3": "final_evaluation",
}


def _plan(roles: dict[str, str]) -> ExposurePlan:
    """An exposure plan over the four retrieval-fixture traces, one lineage each."""
    return ExposurePlan.model_validate(
        {
            "bindings": ArtifactInput(artifact_id="bindings", sha256="b" * 64),
            "seed": 0,
            "probe_fraction": 0.3,
            "roles": roles,
            "traces": {f"trace-{i}": f"lineage-{i}" for i in range(4)},
        }
    )


def _labels_artifact(store: ArtifactStore, artifact_type: str) -> ArtifactInput:
    """Save the label fixture under an artifact type."""
    manifest = store.write_json(
        artifact_id=f"labels-{artifact_type}",
        artifact_type=artifact_type,
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test", inputs=()
        ),
        files={"labels.json": _LABELS},
    )
    return artifact_input(manifest)


def test_traceml_labels_are_an_optional_reference_keyed_by_attempt_step(tmp_path: Path) -> None:
    """Labels load only from a traceml-labels artifact and index by their attempt locators."""
    store = _store(tmp_path, "labels")
    good = ReferenceSource(
        kind="traceml_labels",
        name="steps",
        artifact=_labels_artifact(store, "traceml-labels"),
        file="labels.json",
    )
    (loaded,) = load_references(store, (good,))
    indexed = _step_labels(loaded.content)
    assert sorted(indexed) == ["1/step:1", "1/step:2"]
    records = [row.get("record") for row in indexed["1/step:2"] if isinstance(row, dict)]
    assert records == ["steps", "pivots"]

    wrong = good.model_copy(update={"artifact": _labels_artifact(store, "reference-fixture")})
    with pytest.raises(ValueError, match="not a traceml-labels artifact"):
        load_references(store, (wrong,))


def test_divergence_points_follow_every_shared_history_to_its_split() -> None:
    """Attempts sharing an action history split where they first act differently."""

    def act(name: str) -> JsonObject:
        """One saved tool turn."""
        return {"action": {"name": name}}

    documents: JsonObject = {
        "attempt:0": {"grade_status": "pass"},
        "attempt:1": {"grade_status": "fail"},
        "attempt:2": {"grade_status": "fail"},
        "attempt:3": {"grade_status": None},
        "attempt:0/step:1": act("read"),
        "attempt:1/step:1": act("read"),
        "attempt:2/step:1": act("read"),
        "attempt:3/step:1": act("write"),
        "attempt:0/step:2": act("lock"),
        "attempt:1/step:2": act("delete"),
        "attempt:2/step:2": act("delete"),
    }

    rows = divergence_points(documents)["divergence_points"]

    assert isinstance(rows, list)
    assert [(row["step"], row["shared_by"]) for row in rows] == [  # type: ignore[index]
        (1, [0, 1, 2, 3]),
        (2, [0, 1, 2]),
    ]
    branches = rows[1]["branches"]  # type: ignore[index]
    assert {b["action"]["name"]: b["grade_statuses"] for b in branches} == {  # type: ignore[index]
        "delete": ["fail", "fail"],
        "lock": ["pass"],
    }


def test_retrieval_reaches_only_world_learning_traces_and_refuses_held_out(
    tmp_path: Path,
) -> None:
    """A scoped fit-only index serves searches; held-out or probe reach is refused up front."""
    store = _store(tmp_path, "search")
    built, _ = _build(store, _scope("trace", "trace-0", "trace-2"))
    index = artifact_input(store.read(built.index.rag_id).manifest)
    spec = TraceSearchSpec(index=index, scope=_scope("trace", "trace-0", "trace-2"), task_id="t")

    search = open_trace_search(store, spec, _plan(_ROLES))
    status, found = execute_evidence_tool(
        {},
        ToolCall(
            call_id="c",
            name="search_world_traces",
            arguments={"situation": "customer asks", "message": "hello", "top_k": 2},
        ),
        maximum_bytes=64_000,
        search=search,
    )
    assert status == "ok" and isinstance(found, dict)
    matches = found["matches"]
    assert isinstance(matches, list) and matches
    assert {row["trace_id"] for row in matches} <= {"trace-0", "trace-2"}  # type: ignore[index]
    assert search.inventory()["available"] is True
    with pytest.raises(ValueError, match="top_k"):
        search.search("x", RAGAction(kind="message", content="hi"), 9)

    probe = {**_ROLES, "lineage-2": "mining_probes"}
    with pytest.raises(ValueError, match="not world learning"):
        open_trace_search(store, spec, _plan(probe))

    held_out = _store(tmp_path, "held-out")
    serving, _ = _build(held_out, None, partitions=frozenset({"fit", "held_out"}))
    serving_ref = artifact_input(held_out.read(serving.index.rag_id).manifest)
    with pytest.raises(ValueError, match="held-out"):
        open_trace_search(held_out, spec.model_copy(update={"index": serving_ref}), _plan(_ROLES))

    unscoped = _store(tmp_path, "unscoped")
    plain, _ = _build(unscoped, None)
    plain_ref = artifact_input(unscoped.read(plain.index.rag_id).manifest)
    with pytest.raises(ValueError, match="registered scope"):
        open_trace_search(unscoped, spec.model_copy(update={"index": plain_ref}), _plan(_ROLES))
