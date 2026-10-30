"""Build-produced simulator artifacts connect to executable worlds under one physical budget."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models import (
    ReasoningEffort,
    ToolCall,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.project import ProjectStore
from exp.common.tasks.facts import ConditioningFacts, InstanceFacts
from exp.optimize.evaluation.executable import grounded as grounded_runtime
from exp.optimize.evaluation.executable.grounded import (
    GroundedRuntimeDefinition,
    continuation_runtime,
    preflight_grounded_runtime,
)
from exp.optimize.evaluation.fidelity.instance_facts import ReplayFacts
from exp.optimize.evaluation.fidelity.runner import FidelitySpec
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.runtime.environments.world_contracts import StateDomain, ToolEffectPolicy, WorldManifest
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.sessions.dispatch_ledger import DispatchCapExhaustedError
from exp.simulation.engines.text.prompt import INSTANCE_FACTS_GUIDANCE
from exp.simulation.world_model.prediction_reading import PredictionReading


@pytest.mark.parametrize("projection", ["present", "missing", "null"])
def test_built_world_runs_restores_and_exhausts_one_shared_attempt_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, projection: str
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
            tools=(ToolEffectPolicy(name="inspect", reads=("records",), writes=("records",)),),
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
    if projection != "present":
        proposal["effect"]["edits"] = [
            {
                "op": "set",
                "path": ["domains", "records"],
                "value": {} if projection == "missing" else {"status": None},
            }
        ]
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
        with pytest.raises(ValueError, match="frozen simulator"):
            runtime.probe(saved.model_copy(update={"backend_sha256": "0" * 64}))
        result = world.execute(
            ToolCall(name="inspect", call_id="read", arguments={"key": "current"})
        )
        assert result.content == "pending"
        assert result.metadata["authority"] == "simulated"
        expected = (
            {}
            if projection == "missing"
            else {"status": None if projection == "null" else "pending"}
        )
        assert runtime.probe(world.checkpoint()) == expected
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


def _prediction(call_id: str) -> JsonHttpResponse:
    """Return one scripted provider reply proposing an unchanged-state text observation."""
    proposal = {
        "observations": [{"call_id": call_id, "reply": "pending", "encoding": "text"}],
        "effect": {"kind": "edits", "edits": []},
    }
    return JsonHttpResponse(
        status_code=200,
        body={
            "id": "prediction-" + call_id,
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
    )


def wire_messages(payload: JsonObject) -> tuple[str, str]:
    """Return the system and user message text of one physical provider request body."""
    messages = payload["messages"]
    assert isinstance(messages, list)
    text = {}
    for row in messages:
        assert isinstance(row, dict)
        content = row["content"]
        if isinstance(content, list):
            content = "".join(
                str(part["text"]) for part in content if isinstance(part, dict) and "text" in part
            )
        text[str(row["role"])] = str(content)
    return text["system"], text["user"]


def assert_wire_facts(payload: JsonObject, expected: ConditioningFacts) -> None:
    """The physical request carries each registered fact's exact canonical rendering."""
    system, user = wire_messages(payload)
    assert INSTANCE_FACTS_GUIDANCE in system
    for fact in expected.facts:
        rendered = json.dumps(
            fact.rendered(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        assert rendered in user
    channel = json.loads(user)["instance_facts"]
    assert channel["state_version"] == expected.state_version
    assert channel["facts"][: len(expected.facts)] == [row.rendered() for row in expected.facts]


def _record_spec(project: ProjectStore, fidelity: FidelitySpec) -> GroundedRuntimeDefinition:
    """Return a fact-free records world over the fixture's built simulator."""
    return GroundedRuntimeDefinition(
        task=fidelity.task.model_copy(
            update={"origin": "authored", "source_trace_ids": (), "partition": "fit"}
        ),
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
        maximum_physical_attempts=2,
        deadline=datetime.now(UTC) + timedelta(minutes=5),
    )


def with_facts(spec: GroundedRuntimeDefinition, facts: InstanceFacts) -> JsonObject:
    """Return the configuration JSON of ``spec`` with explicitly declared facts."""
    return {
        **spec.model_dump(mode="json"),
        "facts": ReplayFacts(facts=facts).model_dump(mode="json"),
    }


def test_declared_record_facts_reach_the_wire_and_survive_continuation_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Facts condition every physical prediction and re-key only the definition binding them."""
    from exp.common.release_revision import installed_execution_revision
    from exp.optimize.evaluation.continuations.study import restore_continuation_world
    from exp.simulation.world_model.executable_test import record_facts

    project, fidelity = source_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    spec = _record_spec(project, fidelity)
    facts = record_facts()
    configuration = with_facts(spec, facts)
    plain = spec.model_dump(mode="json")
    assert "facts" not in plain and GroundedRuntimeDefinition.model_validate(plain) == spec
    transport = ScriptedJsonTransport((_prediction("first"), _prediction("second")))
    monkeypatch.setattr(grounded_runtime, "HttpxJsonTransport", lambda: transport)
    runtime = continuation_runtime(project, configuration)
    task = spec.task
    with runtime.world.open(task) as world:
        world.execute(ToolCall(name="inspect", call_id="first", arguments={"key": "current"}))
        saved = world.checkpoint()
        assert runtime.probe(saved) == {"status": "pending"}
    restarted = continuation_runtime(project, configuration)
    with restore_continuation_world(restarted.world, task, saved) as world:
        world.execute(ToolCall(name="inspect", call_id="second", arguments={"key": "current"}))
    assert len(transport.requests) == 2
    for version, request in enumerate(transport.requests):
        expected = facts.conditioning(version)
        assert expected is not None and len(expected.facts) == 4
        assert_wire_facts(request.payload, expected)
        assert "private-answer-7731" not in json.dumps(request.payload)
    unbound = continuation_runtime(project, plain)
    assert isinstance(unbound.world, ExecutableWorldRuntime)
    assert isinstance(runtime.world, ExecutableWorldRuntime)
    assert unbound.world.backend_sha256 != runtime.world.backend_sha256
    with pytest.raises(ValueError, match="another backend definition"):
        with restore_continuation_world(unbound.world, task, saved):
            pass
    revision = installed_execution_revision()
    for definition in (plain, configuration):
        identity = sha256_json({"definition": definition, "code": revision})
        assert (project.artifacts.project_directory / (identity + "-world.sqlite")).exists()


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"status": "unvalidated", "method": None}, "not validated"),
        ({"state_version": 1}, "starting version 0"),
        ({"key": "native-read/" + "a" * 64}, "reserved native-read/"),
    ],
)
def test_definition_refuses_facts_an_executable_episode_cannot_honor(
    change: JsonObject, reason: str
) -> None:
    """Unvalidated, later-state and receipt-shadowing facts refuse before any composition."""
    from exp.simulation.world_model.executable_test import record_facts

    fact = record_facts().facts[0].model_dump(mode="json")
    provenance = fact["provenance"]
    assert isinstance(provenance, dict)
    if "status" in change:
        provenance["validation"] = change
    elif "state_version" in change:
        provenance["state_version"] = change["state_version"]
    else:
        fact.update(change)
    from exp.common.core.artifacts import ArtifactInput
    from exp.runtime.environments.tests.world_conformance_test import _task

    spec = GroundedRuntimeDefinition(
        task=_task("inspect"),
        world_model=ArtifactInput(artifact_id="world-model", sha256="0" * 64),
        catalog_sha256="0" * 64,
        manifest=WorldManifest(
            domains=(StateDomain(name="records", owner="simulated"),),
            tools=(ToolEffectPolicy(name="inspect", reads=("records",)),),
        ),
        initial_state={"records": {}},
        probe_paths={"state": "/records"},
        maximum_physical_attempts=1,
        deadline=datetime.now(UTC) + timedelta(minutes=5),
    )
    with pytest.raises(ValueError, match=reason):
        GroundedRuntimeDefinition.model_validate(
            {**spec.model_dump(mode="json"), "facts": {"facts": {"facts": [fact]}}}
        )


def test_facts_derived_from_a_foreign_trace_refuse_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fidelity reservation rule applies: facts cannot launder another trace's state."""
    from exp.common.tasks.facts import InstanceFactsError
    from exp.simulation.world_model.executable_test import record_facts

    project, fidelity = source_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    spec = _record_spec(project, fidelity)
    fact = record_facts().facts[0]
    foreign = fact.model_copy(
        update={
            "provenance": fact.provenance.model_copy(
                update={"derived_from_traces": ("someone-else",)}
            )
        }
    )
    transport = ScriptedJsonTransport(())
    monkeypatch.setattr(grounded_runtime, "HttpxJsonTransport", lambda: transport)
    with pytest.raises(InstanceFactsError, match="someone-else"):
        continuation_runtime(project, with_facts(spec, InstanceFacts(facts=(foreign,))))
    assert not transport.requests


def _reasoning_spec(project: ProjectStore, fidelity: FidelitySpec) -> GroundedRuntimeDefinition:
    """Declare a high-default fixture before freezing a definition, retaining model identity."""
    catalog = load_model_catalog(project.model_catalog_path)
    record = catalog.models["model"]
    assert record.capabilities is not None
    configured = record.model_copy(
        update={
            "capabilities": record.capabilities.model_copy(
                update={"supports_reasoning": True, "reasoning_effort": "high"}
            ),
            "supported_reasoning_efforts": ("none", "low", "high"),
        }
    )
    expected = RuntimeModelCatalog(catalog, environment={}).snapshot("model")[0]
    prepared = catalog.model_copy(update={"models": {"model": configured, "untouched": configured}})
    assert RuntimeModelCatalog(prepared, environment={}).snapshot("model")[0] == expected
    write_model_catalog(project.model_catalog_path, prepared)
    return _record_spec(project, fidelity)


@pytest.mark.parametrize("effort", [None, "none", "low"])
def test_predictor_reasoning_override_reaches_wire_and_receipts_without_catalog_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, effort: ReasoningEffort | None
) -> None:
    """One local predictor policy preserves model identity and other catalog aliases."""
    project, fidelity = source_project(tmp_path)
    baseline = _reasoning_spec(project, fidelity)
    spec = baseline.model_copy(update={"reasoning_effort": effort})
    catalog_bytes = project.model_catalog_path.read_bytes()
    catalog = load_model_catalog(project.model_catalog_path)
    snapshot = RuntimeModelCatalog(catalog, environment={}).snapshot("model")[0]
    if effort is None:
        assert "reasoning_effort" not in spec.model_dump(mode="json")
        assert spec.model_dump(mode="json") == baseline.model_dump(mode="json")
    else:
        assert spec.model_dump(mode="json")["reasoning_effort"] == effort
        assert sha256_json(spec) != sha256_json(baseline)
    transport = ScriptedJsonTransport((_prediction("read"),))
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    monkeypatch.setattr(grounded_runtime, "HttpxJsonTransport", lambda: transport)
    runtime = continuation_runtime(project, spec.model_dump(mode="json"))
    with runtime.world.open(spec.task) as world:
        assert (
            world.execute(
                ToolCall(name="inspect", call_id="read", arguments={"key": "current"})
            ).content
            == "pending"
        )
    assert len(transport.requests) == 1
    assert transport.requests[0].payload["reasoning_effort"] == (effort or "high")
    assert project.model_catalog_path.read_bytes() == catalog_bytes
    assert RuntimeModelCatalog(catalog, environment={}).snapshot("model")[0] == snapshot
    assert catalog.models["untouched"].capabilities is not None
    assert catalog.models["untouched"].capabilities.reasoning_effort == "high"
    intents = project.artifacts.list_ids(artifact_type="structured-call-intent")
    assert len(intents) == 1
    receipt = json.loads(project.artifacts.read_bytes(intents[0], "request.json"))
    assert receipt["plan"]["model"] == snapshot.model_dump(mode="json")
    assert receipt["plan"]["capabilities"]["reasoning_effort"] == (effort or "high")
    assert receipt["request"].get("reasoning_effort") == effort


def test_unsupported_predictor_reasoning_is_refused_before_runtime_or_provider_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unsupported explicit setting cannot publish a capture or buy a simulator call."""
    project, fidelity = source_project(tmp_path)
    baseline = _reasoning_spec(project, fidelity)
    bad = baseline.model_copy(update={"reasoning_effort": "max"})
    before = project.artifacts.list_ids()

    def forbidden() -> ScriptedJsonTransport:
        """Fail if runtime validation reaches transport construction."""
        raise AssertionError("unsupported reasoning must fail before any transport exists")

    monkeypatch.setattr(grounded_runtime, "HttpxJsonTransport", forbidden)
    with pytest.raises(ValueError, match="reasoning effort"):
        preflight_grounded_runtime(project, bad)
    with pytest.raises(ValueError, match="reasoning effort"):
        continuation_runtime(project, bad.model_dump(mode="json"))
    assert project.artifacts.list_ids() == before
