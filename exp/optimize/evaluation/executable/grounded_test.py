"""Build-produced simulator artifacts connect to executable worlds under one physical budget."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_json
from exp.common.models import ToolCall, load_model_catalog
from exp.optimize.evaluation.executable import grounded as grounded_runtime
from exp.optimize.evaluation.executable.grounded import (
    GroundedRuntimeDefinition,
    continuation_runtime,
)
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.runtime.environments.world_contracts import StateDomain, ToolEffectPolicy, WorldManifest
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.sessions.dispatch_ledger import DispatchCapExhaustedError
from exp.simulation.world_model.prediction_reading import PredictionReading


def test_built_world_runs_restores_and_exhausts_one_shared_attempt_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frozen trace-built model makes a real provider-protocol call, not a second simulator."""
    project, fidelity = source_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    task = fidelity.task.model_copy(
        update={"origin": "authored", "source_trace_ids": (), "partition": "fit"}
    )
    spec = GroundedRuntimeDefinition(
        task=task,
        world_model=fidelity.world_model,
        catalog_sha256=sha256_json(
            load_model_catalog(project.model_catalog_path).model_dump(mode="json")
        ),
        manifest=WorldManifest(
            domains=(StateDomain(name="records", owner="simulated"),),
            tools=(ToolEffectPolicy(name="inspect", reads=("records",), writes=()),),
        ),
        initial_state={"records": {"status": "pending"}},
        probe_paths={"status": "/records/status"},
        maximum_physical_attempts=1,
        deadline=datetime.now(UTC) + timedelta(minutes=5),
    )
    proposal = {
        "observations": [{"call_id": "read", "reply": "pending", "encoding": "text"}],
        "effect": {"kind": "edits", "edits": []},
    }
    transport = ScriptedJsonTransport(
        (
            JsonHttpResponse(
                status_code=200,
                body={
                    "id": "prediction",
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
                                        "id": "proposal",
                                        "type": "function",
                                        "function": {
                                            "name": "submit_transition",
                                            "arguments": json.dumps(proposal),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
            ),
        )
    )
    monkeypatch.setattr(grounded_runtime, "HttpxJsonTransport", lambda: transport)
    runtime = continuation_runtime(project, spec.model_dump(mode="json"))
    with runtime.world.open(task) as world:
        saved = world.checkpoint()
        assert runtime.probe(saved) == {"status": "pending"}
        result = world.execute(
            ToolCall(name="inspect", call_id="read", arguments={"key": "current"})
        )
        assert result.content == "pending"
        assert result.metadata["authority"] == "simulated"
    # A restored fork shares the same dispatch ledger. It cannot obtain a fresh budget.
    restarted = continuation_runtime(project, spec.model_dump(mode="json"))
    from exp.optimize.evaluation.continuations.study import restore_continuation_world

    with restore_continuation_world(restarted.world, task, saved) as world:
        with pytest.raises(DispatchCapExhaustedError):
            world.execute(ToolCall(name="inspect", call_id="read", arguments={"key": "current"}))
        assert world.checkpoint() == saved
    assert len(transport.requests) == 1
    assert len(project.artifacts.list_ids(artifact_type="provider-physical-attempt")) >= 2
    events = project.artifacts.list_ids(artifact_type="grounded-world-prediction")
    assert (
        len(events) == 3
    )  # Original intent/result; refused fork intent, with no fabricated result.


@pytest.mark.parametrize("allowance", [1, 100])
@pytest.mark.parametrize("research_history", [True, False])
@pytest.mark.parametrize("reading", [True, False])
def test_build_learned_memory_uses_admitted_native_decisions_in_executable_world(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allowance: int,
    research_history: bool,
    reading: bool,
) -> None:
    """The default build's native recall reaches prediction and shares its physical call cap."""
    from exp.common.core.artifacts import JsonObject
    from exp.common.release_revision import installed_execution_revision
    from exp.common.tasks import TaskCase, ToolSchema
    from exp.optimize.evaluation.audit.panel_test import SimulatedProvider, build_project, episode
    from exp.optimize.evaluation.learning.profile import learned_evaluation_memory
    from exp.runtime.sessions.dispatch_ledger import DispatchLedger
    from exp.simulation.world_model.experience_memory import memory_head

    episodes = [episode(index) for index in range(3)]
    provider = SimulatedProvider(episodes)
    _, project, _ = build_project(tmp_path, monkeypatch, provider, episodes=episodes, learn=True)
    catalog = load_model_catalog(project.model_catalog_path)
    memory = learned_evaluation_memory(project, catalog)
    assert memory is not None
    head = memory_head(project.artifacts, memory.lineage)
    task = TaskCase(
        task_id="executable-memory",
        lineage_group_id="executable-memory",
        origin="authored",
        partition="fit",
        instruction="Read the user's details.",
        workload_weight=1,
        tools=(
            ToolSchema(
                name="get_user_details",
                description="Read current user details",
                input_schema={"type": "object"},
            ),
        ),
    )
    spec = GroundedRuntimeDefinition(
        task=task,
        world_model=memory.world_model,
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        manifest=WorldManifest(
            domains=(StateDomain(name="records", owner="simulated"),),
            tools=(ToolEffectPolicy(name="get_user_details", reads=("records",)),),
        ),
        initial_state={"records": {}},
        memory=memory,
        research_history=research_history,
        prediction_reading=PredictionReading(reader_alias=memory.recall.jev_alias)
        if reading
        else None,
        probe_paths={"state": "/records"},
        maximum_physical_attempts=allowance,
        deadline=datetime.now(UTC) + timedelta(minutes=5),
    )
    original = provider._transition

    def prediction(payload: JsonObject) -> JsonObject:
        """Keep the fixture reply while satisfying the executable domain-state contract."""
        result = original(payload)
        result["effect"] = {"kind": "edits", "edits": []}
        return result

    original_post = provider.post
    from exp.optimize.evaluation.audit.panel_test import _tool_reply
    from exp.runtime.models.providers.transport import RecordedRequest

    def post(
        url: str, *, headers: Mapping[str, str], payload: JsonObject, timeout_seconds: float
    ) -> JsonHttpResponse:
        """Script decision answers and one read-more proposal; forward the rest to the fixture."""
        questions = payload.get("questions", {})
        if isinstance(questions, dict) and set(questions) == {"support", "contradiction"}:
            provider.requests.append(RecordedRequest(url, dict(headers), payload))
            return JsonHttpResponse(
                status_code=200,
                body={
                    "answers": {
                        "support": {"type": "choice", "choice": "whole"},
                        "contradiction": {"type": "choice", "choice": "none"},
                    }
                },
            )
        if "read_more" in json.dumps(payload.get("tools", [])):
            provider.requests.append(RecordedRequest(url, dict(headers), payload))
            return _tool_reply(
                "world-v1",
                "submit_proposal",
                {
                    "question": "Which reply fields does this operation return?",
                    "read_more": True,
                },
            )
        return original_post(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)

    monkeypatch.setattr(provider, "post", post)
    monkeypatch.setattr(provider, "_transition", prediction)
    monkeypatch.setattr(grounded_runtime, "HttpxJsonTransport", lambda: provider)
    start = len(provider.requests)
    runtime = continuation_runtime(project, spec.model_dump(mode="json"))
    with runtime.world.open(task) as world:
        action = ToolCall(name="get_user_details", call_id="read", arguments={"user_id": "user_00"})
        before = world.checkpoint()
        if allowance == 1:
            with pytest.raises(DispatchCapExhaustedError):
                world.execute(action)
            assert world.checkpoint() == before
        else:
            reply = world.execute(action)
            assert json.loads(reply.content)["user_id"] == "user_00"
    requests = provider.requests[start:]
    assert any(row.url.endswith("/systemone") for row in requests)
    if allowance == 1:
        assert len(requests) == 1 and requests[0].url.endswith("/systemone")
    else:
        latest = provider.predictions()[-1]
        assert latest.get("memory_observations"), (
            "eligible learned records must reach the actual predictor"
        )
        if reading:
            packet = latest["prediction_evidence"]
            assert isinstance(packet, dict)
            assert isinstance(packet["passages"], list)
            assert packet["inspected_ids"] and packet["passages"]
            assert all(
                isinstance(row, dict)
                and isinstance(row["original"], dict)
                and row["original"]["kind"] == "original-observation"
                for row in packet["passages"]
            )
        else:
            assert "prediction_evidence" not in latest
        from exp.common.project import artifact_input
        from exp.optimize.evaluation.analysis.memory_evidence import (
            MemoryInspection,
            inspect_memory,
        )

        intents = tuple(
            artifact_input(project.artifacts.read(key).manifest)
            for key in project.artifacts.list_ids(artifact_type="structured-call-intent")
            if json.loads(project.artifacts.read_bytes(key, "request.json"))
            .get("logical_id", "")
            .startswith("grounded-world-")
        )
        report = inspect_memory(project.artifacts, MemoryInspection(intents=intents))
        assert report.calls and all(call.coverage == "linked_attempts" for call in report.calls)
        assert {call.consumer for call in report.calls} == (
            {"simulator", "selector"} if reading else {"simulator"}
        )
        if not research_history:
            research = latest.get("research_history", {})
            assert isinstance(research, dict)
            assert set(research) <= {"memory_overview"}
            rows = research.get("memory_overview", [])
            assert isinstance(rows, list)
            assert all(
                isinstance(row, dict) and row["kind"] in {"observed", "omitted"} for row in rows
            )
    assert memory_head(project.artifacts, memory.lineage) == head
    identity = sha256_json(
        {"definition": spec.model_dump(mode="json"), "code": installed_execution_revision()}
    )
    ledger = DispatchLedger(
        project.artifacts.project_directory / (identity + "-world.sqlite"), allowance, identity
    )
    assert ledger.used() == len(requests)
