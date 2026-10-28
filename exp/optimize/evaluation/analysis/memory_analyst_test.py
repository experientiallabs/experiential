"""The opt-in analyst shares existing call accounting and replays without credentials."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from filelock import FileLock, Timeout

from exp.cli.evaluation.continuation_context_test import UNPRICED_SPEND
from exp.common.core.artifacts import ArtifactEnvelope, JsonObject, sha256_json
from exp.common.models import ModelMessage, ModelRequest, load_model_catalog
from exp.common.project import artifact_input
from exp.optimize.evaluation.analysis import memory_analyst
from exp.optimize.evaluation.analysis.memory_analyst import (
    MemoryAnalystSpec,
    diagnostic_definition,
    run_memory_analysis,
)
from exp.optimize.evaluation.analysis.memory_evidence import MemoryInspection, inspect_memory
from exp.optimize.evaluation.analysis.source_access import declare_source_access
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.evaluation.analysis import analyze_traces
from exp.simulation.evaluation.analysis_test import output


@pytest.mark.parametrize("root_inventory", [False, True])
def test_analyst_replay_uses_saved_call_and_never_changes_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root_inventory: bool
) -> None:
    """A completed diagnostic needs neither credentials nor a second provider response."""
    project, _ = source_project(tmp_path)
    catalog = load_model_catalog(project.model_catalog_path)
    model, _ = RuntimeModelCatalog(catalog, environment={}).snapshot("model")
    request = ModelRequest(messages=(ModelMessage(role="user", content="read record"),))
    source = project.artifacts.write_json(
        artifact_id="saved-call",
        artifact_type="structured-call-intent",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), inputs=(), code_revision="fixture"
        ),
        files={
            "request.json": {
                "request": request.model_dump(mode="json"),
                "consumer": "actor",
                "logical_id": "actor/0",
            }
        },
    )
    inspection = MemoryInspection(intents=(artifact_input(source),))
    if root_inventory:
        root = project.artifacts.write_json(
            artifact_id="saved-run",
            artifact_type="fixture-run",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                inputs=(artifact_input(source),),
                code_revision="fixture",
            ),
            files={"run.json": {}},
        )
        inspection = MemoryInspection(roots=(artifact_input(root),))
    for reference in (*inspection.roots, artifact_input(source)):
        declare_source_access(
            project.artifacts,
            reference,
            scope="development",
            rationale="Synthetic development fixture without protected labels",
            code_revision="fixture",
        )
    exposure = inspect_memory(project.artifacts, inspection)
    spec = MemoryAnalystSpec(
        inspection=inspection,
        exposure_sha256=sha256_json(exposure),
        analyst_alias="model",
        analyst_model=model,
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        deadline=datetime.now(UTC) + timedelta(minutes=5),
        maximum_physical_attempts=1,
    )
    answer = {
        "summary": "Insufficient exposed output.",
        "findings": [],
        "limitations": ["No reasoning or outcome was captured; no memory-use claim is supported."],
    }
    transport = ScriptedJsonTransport(
        [
            JsonHttpResponse(
                status_code=200,
                body={
                    "id": "analyst",
                    "model": "fixture-v1",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "answer",
                                        "type": "function",
                                        "function": {
                                            "name": "submit_proposal",
                                            "arguments": json.dumps(answer),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
            )
        ]
    )
    monkeypatch.setattr(memory_analyst, "HttpxJsonTransport", lambda: transport)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    from typer.testing import CliRunner

    from exp.cli.app import app
    from exp.cli.evaluation.continuations import ContinuationCommand
    from exp.common.release_revision import installed_execution_revision

    command = tmp_path / "analyse-memory.json"
    command.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="analyse-memory", memory_analyst=spec
        ).model_dump_json()
    )
    args = [
        "eval",
        "fidelity-test",
        "--root",
        str(tmp_path),
        "--continuations",
        str(command),
        "--yes",
        "--non-interactive",
    ]
    with FileLock(project.artifacts.project_directory / "source-access.lock", timeout=0):
        with pytest.raises(Timeout):
            run_memory_analysis(project, spec, code_revision="fixture")
    assert not transport.requests, "classification contention never discloses a request"
    dry = CliRunner().invoke(app, [*args, "--dry-run"])
    assert dry.exit_code == 0 and not transport.requests, dry.output
    executed = CliRunner().invoke(app, args)
    assert executed.exit_code == 0, executed.output
    assert "unqualified diagnostic" in executed.output
    revision = installed_execution_revision()
    result = run_memory_analysis(project, spec, code_revision=revision)
    assert result.summary == answer["summary"] and len(transport.requests) == 1
    manifest = project.artifacts.read(spec.identity(revision)).manifest
    assert artifact_input(source) in manifest.inputs
    assert set(inspection.roots) <= set(manifest.inputs)
    assert inspect_memory(project.artifacts, inspection) == exposure
    monkeypatch.delenv("FIDELITY_TEST_KEY")
    project.model_catalog_path.unlink()
    assert run_memory_analysis(project, spec, code_revision=revision) == result
    assert len(transport.requests) == 1
    with pytest.raises(ValueError, match="exposure changed"):
        diagnostic_definition(project, spec.model_copy(update={"exposure_sha256": "f" * 64}))
    declare_source_access(
        project.artifacts,
        artifact_input(source),
        scope="protected",
        rationale="Reserved for independent confirmation",
        code_revision="fixture",
    )
    with pytest.raises(ValueError, match="protected diagnostic source"):
        run_memory_analysis(project, spec, code_revision=revision)
    assert len(transport.requests) == 1, "protection refuses even saved diagnostic replay"


@pytest.mark.parametrize("fault", ["cross_consumer", "causal", "missing_output"])
def test_memory_analyst_cannot_turn_fragments_into_causal_or_cross_consumer_evidence(
    fault: str,
) -> None:
    """A fluent hypothesis still needs co-delivered input/output and cannot establish blame."""
    definition: JsonObject = {
        "memory_diagnostics": {
            "memory:a:input": {"call": "a", "kind": "input", "consumer": "simulator"},
            "memory:a:output": {
                "call": "b" if fault == "cross_consumer" else "a",
                "kind": "exposed_text",
                "consumer": "actor" if fault == "cross_consumer" else "simulator",
            },
        }
    }
    answer = output("memory:a:input")
    if fault != "missing_output":
        answer["findings"][0]["evidence_refs"].append("memory:a:output")
    if fault == "causal":
        answer["findings"][0]["attribution"] = "actor_model"
    with pytest.raises(ValueError, match="memory"):
        analyze_traces(
            definition, (), complete=lambda body, identity: answer, logical_id="analysis"
        )
