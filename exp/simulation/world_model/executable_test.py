"""A frozen learned predictor shares the executable kernel's rollback and restore rules."""

import json
from collections.abc import Sequence

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    AssistantAction,
    ModelCapabilities,
    ModelResponse,
    OperationEconomics,
    ToolCall,
)
from exp.runtime.environments.tests.world_conformance_test import _task
from exp.runtime.environments.world_contracts import (
    StateDomain,
    ToolEffectPolicy,
    UndeclaredEffectError,
    WorldManifest,
)
from exp.runtime.environments.world_responses import ReplyStateBinding
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.simulation.retrieval.evidence_reading import EvidenceReader, ReadingPolicy
from exp.simulation.world_model.executable import GroundedWorldFactory, PredictorEvent
from exp.simulation.world_model.grounded_session import GroundedTransitionEngine
from exp.simulation.world_model.prediction_reading import PredictionResolver
from exp.simulation.world_model.runtime_test import _runtime


def world_fixture(
    edits: Sequence[list[JsonObject]],
    *,
    reads: tuple[str, ...] = ("tickets", "archive"),
    reply_schema: JsonObject | None = None,
    reply_bindings: tuple[ReplyStateBinding, ...] = (),
    prediction_reader: PredictionResolver | None = None,
    state_schema: JsonObject | None = None,
    maximum_reasks: int = 0,
) -> tuple[ExecutableWorldRuntime, list[PredictorEvent], list[JsonObject]]:
    """Use the real grounded prompt/parser with scripted provider responses, never native truth."""
    predictor, _, _ = _runtime("", capabilities=ModelCapabilities(supports_tools=True))
    task = _task("escalate")
    engine = GroundedTransitionEngine(
        predictor,
        task,
        maximum_output_tokens=1024,
        output_protocol="json-tool",
        prediction_reader=prediction_reader,
    )
    manifest = WorldManifest(
        domains=tuple(
            StateDomain(
                name=name,
                owner="simulated",
                state_schema=state_schema if name == "tickets" else None,
            )
            for name in ("tickets", "archive")
        ),
        tools=(
            ToolEffectPolicy(
                name="escalate",
                reads=reads,
                writes=("tickets",),
                reply_schema=reply_schema,
                reply_bindings=reply_bindings,
            ),
        ),
    )
    events: list[PredictorEvent] = []
    requests: list[JsonObject] = []

    def complete(body: JsonObject, operation: str) -> JsonObject:
        """Require durable intent before each draw and return one exact registered proposal."""
        assert events[-1].stage == "intent" and events[-1].operation_id == operation
        index = len(requests)
        requests.append(body)
        return ModelResponse(
            model=predictor.artifact.model,
            economics=OperationEconomics(),
            output=AssistantAction(
                tool_calls=(
                    ToolCall(
                        name="submit_transition",
                        call_id="proposal",
                        arguments={
                            "observations": [
                                {
                                    "call_id": "call",
                                    "reply": {"status": "escalated"},
                                    "encoding": "json",
                                }
                            ],
                            "effect": {"kind": "edits", "edits": edits[index]},
                        },
                    ),
                )
            ),
        ).model_dump(mode="json")

    factory = GroundedWorldFactory(
        engine,
        manifest,
        {"tickets": {"status": "open"}, "archive": {}},
        complete=complete,
        record=events.append,
        maximum_reasks=maximum_reasks,
    )
    return ExecutableWorldRuntime(manifest, factory, owner="simulated"), events, requests


def test_refused_proposal_can_be_reasked_with_frozen_limit_and_no_state_change() -> None:
    """A known invalid proposal receives explicit feedback; committed work is never redrawn."""
    runtime, events, requests = world_fixture(
        [
            [{"op": "set", "path": ["environment", "logical_time"], "value": 1}],
            [{"op": "set", "path": ["domains", "tickets", "status"], "value": "escalated"}],
        ],
        maximum_reasks=1,
    )
    with runtime.open(_task("escalate")) as world:
        observation = world.execute(ToolCall(name="escalate", call_id="call", arguments={}))
        saved = world.checkpoint()
        assert saved.clock == 1
        assert observation.metadata["refused_proposals"] == 1
    assert len(requests) == 2
    assert "proposal was refused" in json.dumps(requests[1])
    assert [event.stage for event in events] == [
        "intent",
        "response",
        "rejection",
        "intent",
        "response",
    ]
    with runtime.restore(_task("escalate"), saved) as resumed:
        assert resumed.checkpoint() == saved
    assert len(requests) == 2


def test_reasks_exhaust_without_publishing_any_partial_state() -> None:
    """Two bad proposals exhaust one extra attempt and preserve every refused response."""
    edit: JsonObject = {"op": "set", "path": ["environment", "logical_time"], "value": 1}
    runtime, events, requests = world_fixture([[edit], [edit]], maximum_reasks=1)
    with runtime.open(_task("escalate")) as world:
        initial = world.checkpoint()
        with pytest.raises(ValueError, match="logical clock"):
            world.execute(ToolCall(name="escalate", call_id="call", arguments={}))
        assert world.checkpoint() == initial
    assert len(requests) == 2 and [e.stage for e in events].count("rejection") == 2


def test_provider_failure_is_never_reasked_as_a_malformed_proposal() -> None:
    """An unknown provider outcome stays unknown even when proposal correction is enabled."""
    runtime, events, requests = world_fixture([[]], maximum_reasks=3)
    assert isinstance(runtime.factory, GroundedWorldFactory)
    calls = []

    def fail(body: JsonObject, operation: str) -> JsonObject:
        """Record one physical attempt and lose its response."""
        calls.append(operation)
        raise TimeoutError("unknown provider outcome")

    runtime.factory.complete = fail
    with runtime.open(_task("escalate")) as world:
        initial = world.checkpoint()
        with pytest.raises(TimeoutError):
            world.execute(ToolCall(name="escalate", call_id="call", arguments={}))
        assert world.checkpoint() == initial
    assert len(calls) == 1 and not requests
    assert [e.stage for e in events] == ["intent"]


def test_declared_state_schema_is_delivered_and_enforced_before_commit() -> None:
    """A schema-valid reply cannot hide invalid committed state, including on restore."""
    schema: JsonObject = {
        "type": "object",
        "required": ["status"],
        "additionalProperties": False,
        "properties": {"status": {"enum": ["open", "escalated"]}},
    }
    runtime, events, requests = world_fixture(
        [[{"op": "set", "path": ["domains", "tickets", "status"], "value": 123}]],
        state_schema=schema,
    )
    with runtime.open(_task("escalate")) as world:
        initial = world.checkpoint()
        with pytest.raises(ValueError, match="state.*schema"):
            world.execute(ToolCall(name="escalate", call_id="call", arguments={}))
        assert world.checkpoint() == initial
    assert len(events) == 3
    assert "state_schemas" in json.dumps(requests[0])


def test_invalid_initial_state_refuses_before_any_provider_call() -> None:
    """An invalid starting state must not become a valid frozen simulator."""
    with pytest.raises(ValueError, match="state.*schema"):
        world_fixture([[]], state_schema={"type": "integer"})


def test_grounded_world_restores_state_history_and_clock_without_a_draw() -> None:
    """Forks resume exact context; a later prediction remains a new, explicitly recorded draw."""
    runtime, events, requests = world_fixture(
        [[{"op": "set", "path": ["domains", "tickets", "status"], "value": "escalated"}], []]
    )
    task = _task("escalate")
    action = ToolCall(name="escalate", call_id="call", arguments={})
    with runtime.open(task) as world:
        initial = world.checkpoint()
        assert json.loads(world.execute(action).content) == {"status": "escalated"}
        saved = world.checkpoint()
        assert saved.clock == 1 and len(events) == 2
        assert initial.image.domains["archive"] == saved.image.domains["archive"]
    with runtime.restore(task, saved) as resumed:
        assert resumed.checkpoint() == saved
        assert len(requests) == 1
        resumed.execute(action)
        assert "escalated" in json.dumps(requests[1])
        wire = json.loads(resumed.checkpoint().image.parts["predictor.json"])
        assert len(wire["history"]) == 2
        assert wire["state"]["tickets"]["status"] == "escalated"
    with runtime.restore(task, initial) as reset:
        assert reset.checkpoint() == initial
    assert len(requests) == 2


def test_reply_contract_is_in_predictor_context_and_refusal_restores_history() -> None:
    """A shape-invalid successful reply cannot commit its otherwise legal state edit."""
    runtime, events, requests = world_fixture(
        [[{"op": "set", "path": ["domains", "tickets", "status"], "value": "escalated"}]],
        reply_schema={
            "type": "object",
            "required": ["status"],
            "properties": {"status": {"type": "integer"}},
        },
    )
    with runtime.open(_task("escalate")) as world:
        before = world.checkpoint()
        with pytest.raises(ValueError, match="reply.*schema"):
            world.execute(ToolCall(name="escalate", call_id="call", arguments={}))
        assert world.checkpoint() == before
    assert len(requests) == 1
    assert "Successful replies must be JSON matching this schema" in json.dumps(requests[0])
    assert [event.stage for event in events] == ["intent", "response", "rejection"]


@pytest.mark.parametrize(
    "edits,reason",
    [
        (
            [{"op": "set", "path": ["domains", "archive", "secret"], "value": 1}],
            "writable state paths",
        ),
        ([{"op": "set", "path": ["environment", "logical_time"], "value": 99}], "logical clock"),
        ([{"op": "set", "path": ["domains", "invented"], "value": {}}], "writable state paths"),
    ],
)
def test_refused_proposals_keep_raw_evidence_but_rollback_all_context(
    edits: list[JsonObject], reason: str
) -> None:
    """An invalid model proposal cannot change even the predictor's next-turn history."""
    runtime, events, requests = world_fixture([edits])
    with runtime.open(_task("escalate")) as world:
        before = world.checkpoint()
        with pytest.raises((ValueError, UndeclaredEffectError), match=reason):
            world.execute(ToolCall(name="escalate", call_id="call", arguments={}))
        assert world.checkpoint() == before
    assert len(requests) == 1
    assert [event.stage for event in events] == ["intent", "response", "rejection"]


def test_unrelated_domains_are_not_delivered_to_the_predictor() -> None:
    """Per-tool read scope removes private current state before memory and request construction."""
    runtime, events, requests = world_fixture([[]], reads=("tickets",))
    with runtime.open(_task("escalate")) as world:
        before = world.checkpoint()
        world.execute(ToolCall(name="escalate", call_id="call", arguments={}))
        after = world.checkpoint()
        assert before.image.domains["archive"] == after.image.domains["archive"]
    assert len(events) == 2 and len(requests) == 1
    messages = requests[0]["messages"]
    assert isinstance(messages, list) and isinstance(messages[1], dict)
    payload = json.loads(str(messages[1]["content"]))
    assert set(payload["environment_state"]["domains"]) == {"tickets"}


def test_native_workbook_and_learned_state_share_paired_restore_without_cross_writes() -> None:
    """Native XLSX edits and model proposals share a world with separate ownership."""
    from exp.common.core.artifacts import sha256_json
    from exp.runtime.environments.mixed import (
        MixedRealityEnvironmentRuntime,
        MixedRealityPolicy,
        ToolRoute,
    )
    from exp.runtime.environments.tests.world_conformance_test import workbook_case
    from exp.runtime.environments.workbook_edits import read_supported_workbook

    simulated, events, requests = world_fixture(
        [[{"op": "set", "path": ["domains", "tickets", "status"], "value": "escalated"}]]
    )
    workbook = workbook_case()
    task = _task("edit_workbook", "escalate")
    manifest = WorldManifest(
        domains=(
            *simulated.manifest.domains,
            StateDomain(name="input.xlsx", owner="native"),
            StateDomain(name="output.xlsx", owner="native"),
        ),
        tools=(*simulated.manifest.tools, workbook.runtime.manifest.tool("edit_workbook")),
    )
    old = simulated.factory
    assert isinstance(old, GroundedWorldFactory)
    engine = GroundedTransitionEngine(
        old.engine.runtime, task, maximum_output_tokens=1024, output_protocol="json-tool"
    )
    factory = GroundedWorldFactory(
        engine, manifest, old.initial_state, complete=old.complete, record=old.record
    )
    candidate = ExecutableWorldRuntime(manifest, factory, owner="simulated")
    native = ExecutableWorldRuntime(manifest, workbook.runtime.factory)
    mixed_events = []
    mixed = MixedRealityEnvironmentRuntime(
        policy=MixedRealityPolicy(
            world=manifest,
            native_identity=sha256_json(native.factory.definition),
            candidate_identity=sha256_json(factory.definition),
            routes=(
                ToolRoute(name="edit_workbook", mode="real", effect="write"),
                ToolRoute(name="escalate", mode="simulated", effect="write"),
            ),
        ),
        native=native,
        candidate=candidate,
        record=mixed_events.append,
    )
    with mixed.open(task) as world:
        initial = world.checkpoint()
        assert world.execute(workbook.first).metadata["authority"] == "native"
        after_native = world.checkpoint()
        world.execute(ToolCall(name="escalate", call_id="call", arguments={}))
        saved = world.checkpoint()
        assert saved.native is not None and after_native.native is not None
        assert saved.native.image == after_native.native.image
        assert (
            read_supported_workbook(saved.native.image.parts["output.xlsx"])["Orders"].max_row == 6
        )
    with mixed.restore(task, saved) as resumed:
        assert resumed.checkpoint().model_dump(exclude={"episode_id"}) == saved.model_dump(
            exclude={"episode_id"}
        )
    with mixed.restore(task, initial) as reset:
        assert reset.checkpoint().model_dump(exclude={"episode_id"}) == initial.model_dump(
            exclude={"episode_id"}
        )
    assert len(requests) == 1 and len(events) == 2
    assert mixed_events


@pytest.mark.parametrize("reject_receipt", [False, True])
def test_mixed_native_probe_conditions_actual_predictor_after_workbook_edits(
    reject_receipt: bool,
) -> None:
    """Current native cells reach only instance conditioning, before each simulated dispatch."""
    from exp.common.core.artifacts import sha256_json
    from exp.runtime.environments.mixed import (
        MixedEvent,
        MixedRealityEnvironmentRuntime,
        MixedRealityPolicy,
        StepGrounding,
        ToolRoute,
    )
    from exp.runtime.environments.tests.world_conformance_test import workbook_case
    from exp.runtime.environments.world_grounding import NativeReadBridge

    original, events, requests = world_fixture([[], []])
    old = original.factory
    assert isinstance(old, GroundedWorldFactory)
    workbook = workbook_case()
    task = _task("edit_workbook", "escalate")
    manifest = WorldManifest(
        domains=(
            *original.manifest.domains,
            StateDomain(name="input.xlsx", owner="native"),
            StateDomain(name="output.xlsx", owner="native"),
        ),
        tools=(
            *original.manifest.tools,
            workbook.runtime.manifest.tool("edit_workbook"),
            ToolEffectPolicy(name="inspect_workbook", reads=("output.xlsx",), duration=0),
        ),
        native_reads=(
            NativeReadBridge(
                name="workbook-context",
                target_tool="escalate",
                probe_tool="inspect_workbook",
                arguments={"sheet": "Orders", "row_count": 2, "column_count": 2},
                argument_keys={"source": "workbook"},
            ),
        ),
    )
    factory = GroundedWorldFactory(
        GroundedTransitionEngine(
            old.engine.runtime, task, maximum_output_tokens=1024, output_protocol="json-tool"
        ),
        manifest,
        old.initial_state,
        complete=old.complete,
        record=old.record,
    )
    native = ExecutableWorldRuntime(manifest, workbook.runtime.factory)
    candidate = ExecutableWorldRuntime(manifest, factory, owner="simulated")
    mixed_events: list[MixedEvent] = []

    def record(event: MixedEvent) -> None:
        """Prove that a failed grounding sink prevents predictor dispatch."""
        if isinstance(event, StepGrounding):
            assert len(requests) == len(
                [row for row in mixed_events if isinstance(row, StepGrounding)]
            )
            if reject_receipt:
                raise OSError("evidence store unavailable")
        mixed_events.append(event)

    runtime = MixedRealityEnvironmentRuntime(
        policy=MixedRealityPolicy(
            world=manifest,
            native_identity=sha256_json(native.factory.definition),
            candidate_identity=sha256_json(factory.definition),
            routes=(
                ToolRoute(name="edit_workbook", mode="real", effect="write"),
                ToolRoute(name="escalate", mode="simulated", effect="write"),
            ),
        ),
        native=native,
        candidate=candidate,
        record=record,
    )
    action = ToolCall(name="escalate", call_id="call", arguments={"workbook": "output.xlsx"})
    with runtime.open(task) as world:
        world.execute(workbook.first)
        first = world.checkpoint()
        if reject_receipt:
            with pytest.raises(RuntimeError, match="OSError"):
                world.execute(action)
            assert not requests and not events
            return
        observation = world.execute(action)
        assert "native" not in observation.metadata and "projection" not in observation.content
        second = world.checkpoint()
        assert first.native is not None and second.native is not None
        assert first.native.image == second.native.image
        assert second.native.clock == 2
        world.execute(workbook.second)
        world.execute(action)
        saved = world.checkpoint()
    receipts = [row for row in mixed_events if isinstance(row, StepGrounding)]
    assert len(receipts) == 2
    for index, expected in enumerate((6, 5)):
        body = requests[index]["messages"]
        assert isinstance(body, list) and isinstance(body[1], dict)
        payload = json.loads(str(body[1]["content"]))
        assert "instance_facts" in payload
        facts = payload["instance_facts"]["facts"]
        assert len(facts) == 1 and facts[0]["state_version"] == index
        receipt = facts[0]["value"]
        assert receipt == receipts[index].reads[0].model_dump(mode="json")
        projection = json.loads(receipt["observation"]["content"])["projection"]["output.xlsx"]
        assert projection["max_row"] == expected
        assert projection["rows"][1][0]["value"] == index + 2
        assert receipt["logical_time"] == 1 + index * 2
        # Facts are in their own authority channel, never silently copied into simulated state.
        assert "output.xlsx" not in payload.get("state", {}).get("domains", {})
    assert saved.candidate is not None
    predictor = json.loads(saved.candidate.image.parts["predictor.json"])
    assert "native-read" not in json.dumps(predictor)
    with runtime.restore(task, saved) as restored:
        assert restored.checkpoint().model_dump(exclude={"episode_id"}) == saved.model_dump(
            exclude={"episode_id"}
        )
    assert len(requests) == 2


@pytest.mark.parametrize("failure", ["unknown", "invalid"])
def test_prediction_reader_failure_cannot_commit_or_silently_fall_back(failure: str) -> None:
    """The reader has dispatched but no final prediction is bought or applied after refusal."""
    receipts: list[JsonObject] = []

    def decide(request: DecisionRequest) -> JsonObject:
        if failure == "unknown":
            raise TimeoutError("reader outcome unknown")
        return {"answers": {"support": {"type": "choice", "choice": "invented"}}}

    reader = PredictionResolver(
        EvidenceReader(ReadingPolicy(rounds=1), decide, receipts.append),
        lambda body, operation: {
            "question": "Can a closed ticket be escalated?",
            "read_more": True,
        },
        {"prior": {"kind": "original-observation", "reply": "closed"}},
        "a" * 64,
    )
    runtime, events, requests = world_fixture([[]], prediction_reader=reader)
    with runtime.open(_task("escalate")) as world:
        before = world.checkpoint()
        with pytest.raises((TimeoutError, ValueError)):
            world.execute(ToolCall(name="escalate", call_id="call", arguments={}))
        assert world.checkpoint() == before
    assert requests == [] and events == []
    assert any("questions" in str(row) for row in receipts)


def test_executable_predictor_receives_manifest_derived_write_boundaries() -> None:
    """The actual provider schema excludes clock and unrelated-domain edits."""
    from jsonschema import Draft202012Validator

    runtime, _, requests = world_fixture([[]])
    with runtime.open(_task("escalate")) as world:
        world.execute(ToolCall(name="escalate", call_id="call", arguments={}))
    tools = requests[0]["tools"]
    assert isinstance(tools, list) and isinstance(tools[0], dict)
    schema = tools[0]["input_schema"]
    assert isinstance(schema, dict)
    validator = Draft202012Validator(schema)
    proposal = {
        "observations": [{"call_id": "call", "reply": {}, "encoding": "json"}],
        "effect": {"edits": []},
    }
    assert validator.is_valid(proposal)
    for path in (["environment", "logical_time"], ["domains", "archive"], ["domains"]):
        proposal["effect"]["edits"] = [{"op": "set", "path": path, "value": {}}]
        assert not validator.is_valid(proposal)
    assert "runtime alone advances environment.logical_time" in json.dumps(requests[0])


@pytest.mark.parametrize("change", [False, True])
def test_declared_reply_state_relation_rejects_a_schema_valid_lie(change: bool) -> None:
    """A generated escalation reply cannot contradict candidate state when the tool binds it."""
    runtime, events, requests = world_fixture(
        [
            [{"op": "set", "path": ["domains", "tickets", "status"], "value": "escalated"}]
            if change
            else []
        ],
        reply_bindings=(
            ReplyStateBinding(reply_path=("status",), state_path=("tickets", "status")),
        ),
    )
    with runtime.open(_task("escalate")) as world:
        before = world.checkpoint()
        if change:
            world.execute(ToolCall(name="escalate", call_id="call", arguments={}))
        else:
            with pytest.raises(ValueError, match="contradicts"):
                world.execute(ToolCall(name="escalate", call_id="call", arguments={}))
            assert world.checkpoint() == before
            assert world.last_execution is not None
            assert world.last_execution.diagnostic_reason is not None
            assert world.last_execution.public_refusal is None
            assert world.last_execution is not None
            assert world.last_execution.effects == "rolled_back"
    assert len(events) == (2 if change else 3) and "state fields" in json.dumps(requests[0])
    assert "page_cursor" in json.dumps(requests[0])
    assert "null at or beyond the end" in json.dumps(requests[0])
    if not change:
        assert 'reply field ["status"]' in str(events[-1].payload["reason"])
