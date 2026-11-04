"""Build-produced simulator artifacts connect to executable worlds under one physical budget."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    SourceIdentity,
    sha256_json,
)
from exp.common.models import (
    ReasoningEffort,
    ToolCall,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.project import ProjectStore
from exp.common.project.manifests import artifact_input
from exp.common.tasks.facts import ConditioningFacts, InstanceFacts
from exp.common.traces import Trace, TraceSource, TraceSpan
from exp.common.traces.ingest import TraceNormalizationResult, persist_trace_dataset
from exp.optimize.evaluation.executable import grounded as grounded_runtime
from exp.optimize.evaluation.executable.grounded import (
    GroundedRuntimeDefinition,
    continuation_runtime,
    preflight_grounded_runtime,
)
from exp.optimize.evaluation.fidelity.instance_facts import ReplayFacts
from exp.optimize.evaluation.fidelity.runner import FidelitySpec
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.optimize.evaluation.variance.build_records import BINDING_TYPE, save_artifact
from exp.optimize.evaluation.variance.tests.grounded_world_test import grounded_fixture
from exp.runtime.environments.world_contracts import (
    ExecutableCheckpoint,
    StateDomain,
    ToolEffectPolicy,
    WorldManifest,
)
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.sessions.dispatch_ledger import DispatchCapExhaustedError
from exp.simulation.continuations import WorldCheckpoint
from exp.simulation.engines.text.prompt import INSTANCE_FACTS_GUIDANCE
from exp.simulation.induction.contracts import InductionEpisode, InductionStep, TransitionProgram
from exp.simulation.induction.sandbox import ProgramSandbox
from exp.simulation.induction.validation import validate_programs
from exp.simulation.retrieval import RAGLineageBinding, load_rag_index, persist_trace_rag
from exp.simulation.retrieval.tags import (
    RAGIndexTags,
    RAGTagQuestion,
    RAGTagView,
    RAGTraceTags,
    TagRetrievalBinding,
    TagRetrievalEvidence,
    TagRetrievalPolicy,
    TaskQueryTags,
)
from exp.simulation.rules.model import Predicate, Rule, RuleBook
from exp.simulation.rules.rulebook import RULES_JSON, rulebook_json
from exp.simulation.world_model.artifact import persist_grounded_world_model
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
    monkeypatch.setattr(grounded_runtime, "HttpxJsonTransport", lambda t=transport: t)
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


def _prediction(call_id: str, reply: str = "pending") -> JsonHttpResponse:
    """Return one scripted provider reply proposing an unchanged-state text observation."""
    proposal = {
        "observations": [{"call_id": call_id, "reply": reply, "encoding": "text"}],
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
    monkeypatch.setattr(grounded_runtime, "HttpxJsonTransport", lambda t=transport: t)
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
    monkeypatch.setattr(grounded_runtime, "HttpxJsonTransport", lambda t=transport: t)
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
    monkeypatch.setattr(grounded_runtime, "HttpxJsonTransport", lambda t=transport: t)
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


def pending_rules() -> RuleBook:
    """One supported observation rule: inspecting a record shows it pending."""
    return RuleBook(
        contracts={},
        rules=(
            Rule(
                rule_id="inspect.pending",
                tool="inspect",
                kind="observation",
                statement="Inspecting the record shows pending.",
                predicate=Predicate.model_validate(
                    {
                        "op": "eq",
                        "left": {"scope": "observation", "path": ""},
                        "right": {"value": "pending"},
                    }
                ),
                status="supported",
                support=2,
                contradiction=0,
            ),
        ),
    )


def save_rules(project: ProjectStore, book: RuleBook) -> ArtifactInput:
    """Save a rule book as ``rules.json`` in one immutable artifact and return its reference."""
    manifest = project.artifacts.write(
        artifact_id="rules-fixture",
        artifact_type="rule-book",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="fixture"
        ),
        files={RULES_JSON: rulebook_json(book).encode()},
    )
    return artifact_input(manifest)


def test_definition_without_rule_book_keeps_its_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent rule book leaves the configuration, and so every derived identity, unchanged."""
    project, fidelity = source_project(tmp_path)
    spec = _record_spec(project, fidelity)
    plain = spec.model_dump(mode="json")
    assert "rule_book" not in plain
    assert GroundedRuntimeDefinition.model_validate(plain) == spec
    reference = save_rules(project, pending_rules())
    checked = spec.model_copy(update={"rule_book": reference}).model_dump(mode="json")
    assert checked == {**plain, "rule_book": reference.model_dump(mode="json")}
    preflight_grounded_runtime(project, GroundedRuntimeDefinition.model_validate(checked))
    stale = reference.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="manifest digest changed"):
        preflight_grounded_runtime(project, spec.model_copy(update={"rule_book": stale}))


def test_bound_rule_book_refuses_a_rule_breaking_proposal_and_reasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runtime loads the pinned rules, re-asks a breaking proposal, and records the verdict."""
    project, fidelity = source_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    reference = save_rules(project, pending_rules())
    spec = _record_spec(project, fidelity).model_copy(
        update={"rule_book": reference, "maximum_reasks": 1}
    )
    transport = ScriptedJsonTransport((_prediction("read", "done"), _prediction("read")))
    monkeypatch.setattr(grounded_runtime, "HttpxJsonTransport", lambda t=transport: t)
    runtime = continuation_runtime(project, spec.model_dump(mode="json"))
    with runtime.world.open(spec.task) as world:
        result = world.execute(ToolCall(name="inspect", call_id="read", arguments={}))
    assert result.content == "pending" and result.metadata["refused_proposals"] == 1
    assert len(transport.requests) == 2
    assert "inspect.pending" in json.dumps(transport.requests[1].payload)
    events = {
        artifact_id: json.loads(project.artifacts.read_bytes(artifact_id, "event.json"))
        for artifact_id in project.artifacts.list_ids(artifact_type="grounded-world-prediction")
    }
    stages = sorted(event["stage"] for event in events.values())
    assert stages == ["intent", "intent", "rejection", "response", "response", "verdict"]
    (verdict,) = (event for event in events.values() if event["stage"] == "verdict")
    assert verdict["payload"]["verdict"]["allowed"] is True
    for artifact_id in events:
        inputs = project.artifacts.read(artifact_id).manifest.inputs
        assert reference in inputs


TAG_POLICY = TagRetrievalPolicy(bank_id="bank", bank_version=1, threshold=0.5, weight=10.0)
"""Treatment policy: a fired shared task tag outweighs any cosine difference."""

TASK_TAGS = TaskQueryTags(
    bank_id="bank",
    bank_version=1,
    view="task",
    threshold=0.5,
    matrix_sha256="f" * 64,
    fired=("req.export",),
)
"""The bound task's own task-view tags."""


def _inspect_trace(index: int, key: str, value: str) -> Trace:
    """One recorded ``inspect`` call on ``key`` answered with ``value``."""
    start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
    shared = {"gen_ai.tool.name": "inspect", "gen_ai.tool.call.id": f"call-{index}"}
    return Trace(
        trace_id=f"tagged-{index}",
        task=f"Read the {key} entry",
        source=TraceSource(
            identity=SourceIdentity(kind="file", source_id="tagged"),
            semantic_convention_version="1.37.0",
        ),
        spans=(
            TraceSpan(
                span_id=f"request-{index}",
                name="request",
                started_at=start,
                ended_at=start,
                attributes={**shared, "gen_ai.tool.call.arguments": json.dumps({"key": key})},
            ),
            TraceSpan(
                span_id=f"result-{index}",
                name="result",
                started_at=start + timedelta(seconds=1),
                ended_at=start + timedelta(seconds=1),
                attributes={
                    **shared,
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.message": value,
                },
            ),
        ),
    )


def tagged_world(project: ProjectStore) -> tuple[ArtifactInput, ArtifactInput]:
    """Persist three fit traces, a task-tagged index over them and a world model over it.

    Only the trace least like the records task (``tagged-2``) fires the bound task's tag.

    Returns:
        The world model and its serving index.
    """
    traces = (
        _inspect_trace(0, "current", "pending"),
        _inspect_trace(1, "current entries", "pending"),
        _inspect_trace(2, "archive", "stored"),
    )
    source = persist_trace_dataset(
        TraceNormalizationResult(traces=traces, issues=()),
        project.artifacts,
        created_at=datetime.now(UTC),
        code_revision="fixture",
    )
    fired = {trace.trace_id: () for trace in traces} | {"tagged-2": ("req.export",)}
    tags = RAGIndexTags(
        views=(
            RAGTagView(
                bank_id="bank",
                bank_version=1,
                view="task",
                threshold=0.5,
                questions=(RAGTagQuestion(question_id="req.export", family="requirement"),),
                matrix_sha256="e" * 64,
                traces=tuple(
                    RAGTraceTags(trace_id=key, fired=value) for key, value in sorted(fired.items())
                ),
            ),
        )
    )
    rag = persist_trace_rag(
        project.artifacts,
        (artifact_input(source.manifest),),
        tuple(
            RAGLineageBinding(trace_id=t.trace_id, lineage_id=f"tagged-{i}", partition="fit")
            for i, t in enumerate(traces)
        ),
        created_at=datetime.now(UTC),
        code_revision="fixture",
        trace_tags=tags,
    )
    catalog = load_model_catalog(project.model_catalog_path)
    model, _ = RuntimeModelCatalog(catalog).snapshot("model")
    world = persist_grounded_world_model(
        project.artifacts,
        artifact_input(rag.manifest),
        model_alias="model",
        model=model,
        created_at=datetime.now(UTC),
        code_revision="fixture",
        top_k=3,
    )
    return artifact_input(world.manifest), artifact_input(rag.manifest)


def save_binding(
    project: ProjectStore, policy: TagRetrievalPolicy, index: ArtifactInput
) -> ArtifactInput:
    """Save a tag retrieval binding for the records task as the variance build does."""
    return save_artifact(
        project.artifacts,
        BINDING_TYPE,
        {"binding.json": TagRetrievalBinding(policy=policy, tasks={"test": TASK_TAGS})},
        (index,),
        "fixture",
    )


def test_definition_without_tag_binding_keeps_its_identity_and_refuses_a_wrong_digest(
    tmp_path: Path,
) -> None:
    """An absent binding changes nothing; a bound one joins the definition and is verified."""
    project, fidelity = source_project(tmp_path)
    spec = _record_spec(project, fidelity)
    plain = spec.model_dump(mode="json")
    assert "tag_retrieval" not in plain
    assert GroundedRuntimeDefinition.model_validate(plain) == spec
    world, index = tagged_world(project)
    reference = save_binding(project, TAG_POLICY, index)
    tagged = spec.model_copy(update={"world_model": world, "tag_retrieval": reference})
    dumped = tagged.model_dump(mode="json")
    assert dumped == {
        **plain,
        "world_model": world.model_dump(mode="json"),
        "tag_retrieval": reference.model_dump(mode="json"),
    }
    assert sha256_json(dumped) != sha256_json({**dumped, "tag_retrieval": None})
    preflight_grounded_runtime(project, GroundedRuntimeDefinition.model_validate(dumped))
    stale = reference.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="not the pinned tag-retrieval-binding"):
        preflight_grounded_runtime(project, tagged.model_copy(update={"tag_retrieval": stale}))
    rules = save_rules(project, pending_rules())
    with pytest.raises(ValueError, match="not the pinned tag-retrieval-binding"):
        preflight_grounded_runtime(project, tagged.model_copy(update={"tag_retrieval": rules}))


def _list(value: object) -> tuple[str, ...]:
    """Return a recorded JSON list of strings, refusing any other value."""
    assert isinstance(value, list)
    return tuple(str(item) for item in value)


def test_bound_tag_retrieval_ranks_the_executable_world_and_records_its_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A treatment binding re-ranks the bound task's retrieval; a weight-0 control keeps cosine.

    Both record the policy as retrieval evidence and pin the binding in every event's inputs.
    """
    project, fidelity = source_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    world, index = tagged_world(project)
    treatment = save_binding(project, TAG_POLICY, index)
    control = save_binding(project, TAG_POLICY.model_copy(update={"weight": 0.0}), index)
    base = _record_spec(project, fidelity).model_copy(update={"world_model": world})
    evidence: dict[str, JsonObject] = {}
    checkpoints: dict[str, WorldCheckpoint] = {}
    for name, binding in (("plain", None), ("control", control), ("treatment", treatment)):
        spec = base.model_copy(update={"tag_retrieval": binding})
        transport = ScriptedJsonTransport((_prediction("read"),))
        monkeypatch.setattr(grounded_runtime, "HttpxJsonTransport", lambda t=transport: t)
        before = set(project.artifacts.list_ids(artifact_type="grounded-world-prediction"))
        runtime = continuation_runtime(project, spec.model_dump(mode="json"))
        with runtime.world.open(spec.task) as opened:
            result = opened.execute(ToolCall(name="inspect", call_id="read", arguments={}))
            checkpoints[name] = opened.checkpoint()
        assert result.content == "pending"
        (intent,) = (
            artifact_id
            for artifact_id in project.artifacts.list_ids(artifact_type="grounded-world-prediction")
            if artifact_id not in before and artifact_id.endswith("-intent")
        )
        evidence[name] = json.loads(project.artifacts.read_bytes(intent, "event.json"))["payload"][
            "evidence"
        ]
        inputs = project.artifacts.read(intent).manifest.inputs
        assert (binding in inputs) if binding is not None else len(inputs) == 1
    trace_of = {
        row.transition_id: row.trace_id
        for row in load_rag_index(project.artifacts, index.artifact_id).transitions
    }
    order = {
        name: tuple(trace_of[transition] for transition in _list(row["grounded_transition_ids"]))
        for name, row in evidence.items()
    }
    assert "tag_retrieval" not in evidence["plain"]
    assert order["control"] == order["plain"]
    assert order["plain"][0] != "tagged-2"
    assert order["treatment"][0] == "tagged-2"
    assert sorted(order["treatment"]) == sorted(order["plain"])
    for name, policy in (
        ("control", TAG_POLICY.model_copy(update={"weight": 0.0})),
        ("treatment", TAG_POLICY),
    ):
        recorded = TagRetrievalEvidence.model_validate(evidence[name]["tag_retrieval"])
        assert recorded.policy == policy and recorded.policy_sha256 == policy.identity
        assert recorded.query == TASK_TAGS and recorded.index_matrix_sha256 == "e" * 64
    # The binding is part of the world's identity: a tagged checkpoint never resumes untagged.
    from exp.optimize.evaluation.continuations.study import restore_continuation_world

    untagged = continuation_runtime(project, base.model_dump(mode="json"))
    with pytest.raises(ValueError, match="another backend definition"):
        with restore_continuation_world(untagged.world, base.task, checkpoints["treatment"]):
            pass
    tagged = continuation_runtime(
        project, base.model_copy(update={"tag_retrieval": treatment}).model_dump(mode="json")
    )
    with restore_continuation_world(tagged.world, base.task, checkpoints["treatment"]) as world:
        assert world.checkpoint() == checkpoints["treatment"]


def test_bound_programs_serve_covered_tools_and_the_model_answers_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A covered tool runs as code, an uncovered one is predicted; the digest binds programs."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    manifest = WorldManifest(
        domains=(
            StateDomain(name="tickets", owner="simulated"),
            StateDomain(name="desk", owner="simulated"),
        ),
        tools=(
            ToolEffectPolicy(name="read_ticket", reads=("desk",)),
            ToolEffectPolicy(name="check_sla", reads=("tickets",)),
            ToolEffectPolicy(name="escalate", reads=("tickets",), writes=("tickets",)),
            ToolEffectPolicy(name="close", reads=("tickets",), writes=("tickets",)),
        ),
    )
    program = TransitionProgram(
        tool="read_ticket",
        source=(
            "def transition(state, args, facts):\n"
            "    return {'ticket': 'T-1', 'status': state['desk']['status']}\n"
        ),
    )
    initial = {**fixture.definition.initial_state, "desk": {"status": "open"}}
    episode = InductionEpisode(
        episode_id="desk",
        initial_state={"desk": {"status": "open"}},
        steps=(
            InductionStep(
                tool="read_ticket", arguments={}, reply={"ticket": "T-1", "status": "open"}
            ),
        ),
    )
    with ProgramSandbox() as sandbox:
        coverage = validate_programs((program,), (episode,), manifest, sandbox).coverage
    assert coverage.induced_tools == ("read_ticket",)
    plain = GroundedRuntimeDefinition.model_validate(
        {
            **fixture.definition.model_dump(mode="json"),
            "manifest": manifest.model_dump(mode="json"),
            "initial_state": initial,
        }
    )
    served = GroundedRuntimeDefinition.model_validate(
        {**plain.model_dump(mode="json"), "programs": coverage.model_dump(mode="json")}
    )
    assert "programs" not in plain.model_dump(mode="json")
    assert sha256_json(served) != sha256_json(plain)

    with (
        continuation_runtime(fixture.project, served.model_dump(mode="json")) as runtime,
        runtime.world.open(served.task) as world,
    ):
        read = world.execute(ToolCall(name="read_ticket", call_id="read", arguments={}))
        assert fixture.predictor.requests == []
        sla = world.execute(ToolCall(name="check_sla", call_id="sla", arguments={}))
        saved = world.checkpoint()
    assert (read.metadata["transition"], json.loads(read.content)["status"]) == (
        "induced",
        "open",
    )
    assert sla.metadata["transition"] == "predicted" and len(fixture.predictor.requests) == 1
    assert json.loads(sla.content) == {"severity": "high", "lock": "none"}
    assert isinstance(saved, ExecutableCheckpoint) and "induced.json" in saved.image.parts
    with pytest.raises(ValueError, match="another world manifest"):
        GroundedRuntimeDefinition.model_validate(
            {
                **fixture.definition.model_dump(mode="json"),
                "programs": coverage.model_dump(mode="json"),
            }
        )


class _RecallWire:
    """One shared physical wire: native recall decisions and scripted predictions."""

    def __init__(self) -> None:
        """Start with no recorded requests."""
        self.decisions: list[JsonObject] = []
        self.predictions: list[JsonObject] = []

    def post(
        self, url: str, *, headers: Mapping[str, str], payload: JsonObject, timeout_seconds: float
    ) -> JsonHttpResponse:
        """Score the inspect rule as governing, everything else as irrelevant."""
        del headers, timeout_seconds
        if url.endswith("/systemone"):
            self.decisions.append(payload)
            state = payload["state"]
            assert isinstance(state, dict) and isinstance(state["index"], list)
            answers: JsonObject = {}
            for i, row in enumerate(state["index"]):
                assert isinstance(row, dict)
                relevant = row["tool"] == "inspect"
                answers[f"governs_{i}"] = {"type": "score", "score": 2.0 if relevant else 0.0}
                answers[f"applies_{i}"] = {"type": "noul", "noul": 0.9 if relevant else 0.1}
            return JsonHttpResponse(
                status_code=200, body={"answers": answers, "usage": {"cost": 0.0001}}
            )
        self.predictions.append(payload)
        return _prediction("read")


def _rule_memory(project: ProjectStore, truism: str) -> ArtifactInput:
    """Save a two-tool memory set whose inspect rule has ``truism``."""
    from exp.simulation.rules.memory_set import MemorySet, RuleFile, save_memory_set

    def rule(rule_id: str, tool: str, text: str) -> RuleFile:
        return RuleFile(
            rule_id=rule_id,
            topic=tool,
            tool=tool,
            kind="observation",
            truism=text,
            reasoning=f"Recorded replies of {tool} always showed it: {text}",
            status="supported",
            status_rationale="Seen in every recorded call.",
        )

    memory = MemorySet(
        tools=("archive", "inspect"),
        rules=(
            rule("observation.inspect.1", "inspect", truism),
            rule("observation.archive.1", "archive", "Archiving hides the record."),
        ),
    )
    return save_memory_set(
        project.artifacts,
        memory,
        artifact_id="memory-" + sha256_json(truism)[:12],
        inputs=(),
        code_revision="test",
    )


def test_bound_rule_memory_recalls_only_relevant_files_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recall injects only the governing rule, records it per step, replays, and binds identity."""
    from exp.optimize.evaluation.executable.grounded import RuleMemoryBinding
    from exp.optimize.evaluation.tests.fidelity_memory_test import memory_project
    from exp.simulation.rules.recall import RecallPolicy

    project, fidelity = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    reference = _rule_memory(project, "Inspecting a record shows its pending status.")
    base = _record_spec(project, fidelity).model_copy(update={"maximum_physical_attempts": 10})
    spec = base.model_copy(
        update={
            "rule_memory": RuleMemoryBinding(
                memory_set=reference, recall=RecallPolicy(jev_alias="jev")
            )
        }
    )
    assert "rule_memory" not in base.model_dump(mode="json")
    wire = _RecallWire()
    monkeypatch.setattr(grounded_runtime, "HttpxJsonTransport", lambda: wire)

    def first_step(definition: GroundedRuntimeDefinition) -> WorldCheckpoint:
        with (
            continuation_runtime(project, definition.model_dump(mode="json")) as runtime,
            runtime.world.open(definition.task) as world,
        ):
            assert world.execute(ToolCall(name="inspect", call_id="read", arguments={})).content
            return world.checkpoint()

    saved = first_step(spec)
    assert len(wire.decisions) == 1
    prompt = json.dumps(wire.predictions[-1])
    assert "Inspecting a record shows its pending status." in prompt
    assert "Archiving hides the record." not in prompt
    intents = [
        json.loads(project.artifacts.read_bytes(key, "event.json"))
        for key in project.artifacts.list_ids(artifact_type="grounded-world-prediction")
    ]
    (intent,) = [event for event in intents if event["stage"] == "intent"]
    recall = intent["payload"]["rule_recall"]
    assert recall["selected"] == ["observation.inspect.1"] and recall["usd"] == 0.0001
    assert project.artifacts.list_ids(artifact_type="jev-receipt")

    first_step(spec)
    assert len(wire.decisions) == 1, "a replayed step recalls from the cache without a call"

    other = spec.model_copy(
        update={
            "rule_memory": RuleMemoryBinding(
                memory_set=_rule_memory(project, "Inspecting a record shows it is done."),
                recall=RecallPolicy(jev_alias="jev"),
            )
        }
    )
    assert isinstance(saved, ExecutableCheckpoint)
    changed = first_step(other)
    assert isinstance(changed, ExecutableCheckpoint)
    assert changed.backend_sha256 != saved.backend_sha256
    assert sha256_json(other) != sha256_json(spec) != sha256_json(base)

    stale = reference.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="manifest digest changed"):
        preflight_grounded_runtime(
            project,
            spec.model_copy(
                update={
                    "rule_memory": RuleMemoryBinding(
                        memory_set=stale, recall=RecallPolicy(jev_alias="jev")
                    )
                }
            ),
        )
    with pytest.raises(ValueError, match="native decision wire"):
        preflight_grounded_runtime(
            project,
            spec.model_copy(
                update={
                    "rule_memory": RuleMemoryBinding(
                        memory_set=reference, recall=RecallPolicy(jev_alias="model")
                    )
                }
            ),
        )
