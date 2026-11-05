"""Native rules/research documents reach predictor framing without evidence relabeling."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput
from exp.common.models import ModelRequest, ModelResponse, ToolCall, load_model_catalog
from exp.common.project import artifact_input
from exp.optimize.evaluation.executable.grounded import rule_research_context
from exp.optimize.evaluation.rules.condense import (
    MemoryEntry,
    TraceDigest,
    WorldModelMemory,
    save_memory,
)
from exp.optimize.evaluation.variance.tests.grounded_world_test import (
    INITIAL,
    grounded_fixture,
)
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.engines.text.tokens import WorldModelCapacityError
from exp.simulation.rules.model import RuleBook
from exp.simulation.world_model.grounded_session import GroundedTransitionEngine
from exp.simulation.world_model.runtime import load_grounded_world_model


class Unused:
    def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("offline preparation must not dispatch")


@pytest.mark.parametrize("research_history", [True, False])
def test_verified_document_reaches_native_request_and_preserves_ablation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, research_history: bool
) -> None:
    fixture = grounded_fixture(tmp_path, monkeypatch)
    project = fixture.project
    book = RuleBook(contracts={})
    manifest = project.artifacts.write_json(
        artifact_id="document-book",
        artifact_type="rule-book",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test"
        ),
        files={"rules.json": book},
    )
    reference = artifact_input(manifest)
    memory = WorldModelMemory(
        rulebook=reference,
        summary="Preserve admitted rules and uncertainty.",
        entries=(
            MemoryEntry(kind="behavior", text="A witnessed rule.", evidence=("trace:fixture",)),
        ),
        contradictions=(),
        unknowns=(),
        research=(TraceDigest(trace_id="fixture", resolved=("research-only-sentinel",)),),
    )
    document = save_memory(project, memory, {reference}, "doc", "test")
    definition = fixture.definition.model_copy(
        update={
            "rule_book": reference,
            "rule_research_document": document,
            "research_history": research_history,
        }
    )
    row = rule_research_context(project, definition)
    assert row is not None
    assert ("research-only-sentinel" in str(row["document"])) is research_history
    assert "A witnessed rule." in str(row["document"])
    caps = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path)).snapshot("model")[1]
    runtime = load_grounded_world_model(
        project.artifacts,
        definition.world_model.artifact_id,
        client=Unused(),
        capabilities=caps,
    )
    baseline = GroundedTransitionEngine(runtime, definition.task, maximum_output_tokens=1000)
    engine = GroundedTransitionEngine(
        runtime,
        definition.task,
        maximum_output_tokens=1000,
        research_history=research_history,
        rule_research_document=row,
    )
    assert engine.identity != baseline.identity
    prepared = engine.prepare(
        {"domains": INITIAL}, (), (ToolCall(name="read_ticket", call_id="read", arguments={}),)
    )
    messages = prepared.body["messages"]
    assert isinstance(messages, list)
    framed = [
        json.loads(message["content"])
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    assert framed[-1]["rule_research_document"] == row
    assert row not in framed[-1].get("memory_observations", [])
    with pytest.raises(ValueError, match="another rule book"):
        rule_research_context(
            project,
            definition.model_copy(
                update={
                    "rule_book": ArtifactInput(artifact_id="other-book", sha256="0" * 64),
                }
            ),
        )
    assert "rule_research_document" not in fixture.definition.model_dump(mode="json")
    oversized = GroundedTransitionEngine(
        runtime,
        definition.task,
        maximum_output_tokens=1000,
        rule_research_document={
            **row,
            "document": "grounded evidence " * (caps.context_window_tokens or 100000),
        },
    )
    with pytest.raises(WorldModelCapacityError):
        oversized.prepare(
            {"domains": INITIAL},
            (),
            (ToolCall(name="read_ticket", call_id="read", arguments={}),),
        )
