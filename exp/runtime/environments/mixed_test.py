"""Behavioral tests for independently owned native and candidate sessions."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject, JsonValue, sha256_json
from exp.common.models import NumericMeasurement, OperationEconomics, ToolCall
from exp.common.tasks import TaskCase, ToolSchema
from exp.runtime.environments.interface import Observation
from exp.runtime.environments.mixed import (
    MixedEvent,
    MixedRealityEnvironmentRuntime,
    MixedRealityPolicy,
    StepIntent,
    StepResult,
    ToolRoute,
)
from exp.runtime.environments.record_world import RecordTool, RecordWorldFactory
from exp.runtime.environments.world_contracts import (
    RestoreError,
    StateDomain,
    ToolEffectPolicy,
    WorldAccess,
    WorldManifest,
)
from exp.runtime.environments.world_outcomes import execute_with_recovery
from exp.runtime.environments.worlds import ExecutableWorldRuntime, WorldContext


class _Runtime:
    """A session fixture that records public inputs and independently owns its state."""

    def __init__(self, replies: list[Observation | Exception]) -> None:
        """Bind deterministic tool outcomes without sharing another runtime's state."""
        self.replies = iter(replies)
        self.actions: list[ToolCall] = []
        self.tasks: list[TaskCase] = []
        self.closed = 0

    @contextmanager
    def open(self, task: TaskCase) -> Iterator["_Runtime"]:
        """Record the public task and prove cleanup on all exit paths."""
        self.tasks.append(task)
        try:
            yield self
        finally:
            self.closed += 1

    def execute(self, action: ToolCall) -> Observation:
        """Return or raise the next tool outcome."""
        self.actions.append(action)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _task(*names: str) -> TaskCase:
    """Return a public task with no reference answer in the executable boundary."""
    return TaskCase(
        task_id="retail-test",
        lineage_group_id="customer-test",
        partition="fit",
        instruction="Read and update the order.",
        workload_weight=1,
        origin="authored",
        reference_answer="private evaluator answer",
        tools=tuple(
            ToolSchema(name=name, description="Fixture tool.", input_schema={"type": "object"})
            for name in (names or ("read",))
        ),
    )


def _policy(*routes: ToolRoute) -> MixedRealityPolicy:
    """Bind both source identities to a prospective routing policy."""
    return MixedRealityPolicy(
        native_identity=sha256_json({"native": "fixture"}),
        candidate_identity=sha256_json({"candidate": "fixture"}),
        routes=routes,
    )


def _observation(content: str, cost: float = 0) -> Observation:
    """Attach explicit cost and private metadata that must not reach the actor."""
    return Observation(
        content=content,
        metadata={
            "private_state": {"secret": "not actor-visible"},
            "economics": OperationEconomics(
                cost_usd=NumericMeasurement(value=cost, provenance="observed")
            ).model_dump(mode="json"),
        },
    )


def test_shadow_returns_native_reply_and_records_both_costs() -> None:
    """A wrong candidate never replaces native truth or hides its generation charge."""
    events: list[MixedEvent] = []
    native = _Runtime([_observation("pending", 0)])
    candidate = _Runtime([_observation("delivered", 0.03)])
    runtime = MixedRealityEnvironmentRuntime(
        policy=_policy(ToolRoute(name="read", mode="shadow", effect="read")),
        native=native,
        candidate=candidate,
        record=events.append,
    )
    with runtime.open(_task()) as session:
        reply = session.execute(ToolCall(call_id="duplicate", name="read"))
    assert reply.content == "pending"
    assert "private_state" not in reply.metadata
    economics = OperationEconomics.model_validate(reply.metadata["economics"])
    assert economics.cost_usd is not None
    assert economics.cost_usd.value == 0.03
    assert isinstance(events[0], StepIntent)
    result = events[-1]
    assert isinstance(result, StepResult)
    assert result.exact_observation is False
    assert result.comparison_scope == "initial_correspondence_unverified"
    assert native.closed == candidate.closed == 1
    assert native.tasks[0].reference_answer is None
    assert candidate.tasks[0].reference_answer is None
    assert candidate.actions[0].call_id == native.actions[0].call_id


def test_shadow_failure_preserves_native_success_and_halts_shadow_track() -> None:
    """A provider failure does not force native tool reexecution or erase the receipt."""
    events: list[MixedEvent] = []
    native = _Runtime([_observation("a"), _observation("b")])
    candidate = _Runtime([TimeoutError("private provider information")])
    runtime = MixedRealityEnvironmentRuntime(
        policy=_policy(ToolRoute(name="read", mode="shadow", effect="read")),
        native=native,
        candidate=candidate,
        record=events.append,
    )
    with runtime.open(_task()) as session:
        assert session.execute(ToolCall(call_id="same", name="read")).content == "a"
        assert session.execute(ToolCall(call_id="same", name="read")).content == "b"
    results = [event for event in events if isinstance(event, StepResult)]
    assert results[0].candidate_failure == "TimeoutError"
    assert results[1].shadow_status == "halted"
    assert results[0].economics.cost_usd is None
    assert len(candidate.actions) == 1
    assert len({event.operation_id for event in results}) == 2


def test_native_transport_failure_is_never_retried_or_simulated() -> None:
    """An ambiguous native write poisons the session and retains the attempted result."""
    events: list[MixedEvent] = []
    native = _Runtime([TimeoutError("native unknown")])
    candidate = _Runtime([_observation("model guess")])
    runtime = MixedRealityEnvironmentRuntime(
        policy=_policy(ToolRoute(name="write", mode="shadow", effect="write")),
        native=native,
        candidate=candidate,
        record=events.append,
    )
    with runtime.open(_task("write")) as session:
        with pytest.raises(TimeoutError):
            session.execute(ToolCall(call_id="write", name="write"))
        with pytest.raises(RuntimeError, match="stopped"):
            session.execute(ToolCall(call_id="write-again", name="write"))
    assert len(native.actions) == len(candidate.actions) == 1
    assert isinstance(events[-1], StepResult)
    assert events[-1].authoritative_failure == "TimeoutError"
    assert events[-1].failures[0].details["phase"] == "authoritative_execution"
    assert events[-1].failures[0].details["operation_id"] == events[-1].operation_id


def test_tool_error_is_an_observation_and_divergent_prefix_stays_visible() -> None:
    """A declared native tool error continues, while earlier divergence stays qualified."""
    events: list[MixedEvent] = []
    native = _Runtime([Observation(content="missing", is_error=True), Observation(content="ok")])
    candidate = _Runtime([Observation(content="different"), Observation(content="ok")])
    runtime = MixedRealityEnvironmentRuntime(
        policy=_policy(ToolRoute(name="read", mode="shadow", effect="read")),
        native=native,
        candidate=candidate,
        record=events.append,
    )
    with runtime.open(_task()) as session:
        assert session.execute(ToolCall(call_id="1", name="read")).is_error
        session.execute(ToolCall(call_id="2", name="read"))
    assert isinstance(events[-1], StepResult)
    assert events[-1].exact_observation is True
    assert events[-1].comparison_scope == "divergent_prefix"


def test_real_write_breaks_candidate_correspondence_without_rescuing_it() -> None:
    """Unshadowed mutation is explicitly unpaired, never silently mirrored into the twin."""
    events: list[MixedEvent] = []
    native = _Runtime([Observation(content="changed"), Observation(content="value")])
    candidate = _Runtime([Observation(content="value")])
    runtime = MixedRealityEnvironmentRuntime(
        policy=_policy(
            ToolRoute(name="write", mode="real", effect="write"),
            ToolRoute(name="read", mode="shadow", effect="read"),
        ),
        native=native,
        candidate=candidate,
        record=events.append,
    )
    with runtime.open(_task("read", "write")) as session:
        session.execute(ToolCall(call_id="1", name="write"))
        session.execute(ToolCall(call_id="2", name="read"))
    assert isinstance(events[-1], StepResult)
    assert events[-1].comparison_scope == "unpaired_prefix"
    assert len(candidate.actions) == 1


def test_recording_failure_stops_before_purchase() -> None:
    """An intent that cannot be retained prevents both external calls."""
    native = _Runtime([Observation(content="never")])
    candidate = _Runtime([Observation(content="never")])

    def refuse(event: MixedEvent) -> None:
        """Model a failed durable evidence sink."""
        raise OSError("disk unavailable")

    runtime = MixedRealityEnvironmentRuntime(
        policy=_policy(ToolRoute(name="read", mode="shadow", effect="read")),
        native=native,
        candidate=candidate,
        record=refuse,
    )
    with runtime.open(_task()) as session:
        with pytest.raises(OSError):
            session.execute(ToolCall(call_id="x", name="read"))
        with pytest.raises(RuntimeError, match="stopped"):
            session.execute(ToolCall(call_id="x", name="read"))
    assert not native.actions and not candidate.actions


def test_unknown_tools_are_rejected_before_dispatch() -> None:
    """A policy cannot be widened by an actor's invented tool name."""
    native = _Runtime([])
    events: list[MixedEvent] = []
    runtime = MixedRealityEnvironmentRuntime(
        policy=_policy(ToolRoute(name="read", mode="real", effect="read")),
        native=native,
        candidate=None,
        record=events.append,
    )
    with runtime.open(_task()) as session:
        with pytest.raises(ValueError, match="not routed"):
            session.execute(ToolCall(call_id="x", name="invented"))
    assert not native.actions and not events


def test_simulated_mode_uses_only_candidate_backend() -> None:
    """An all-simulated policy has explicit candidate authority and no native dispatch."""
    events: list[MixedEvent] = []
    candidate = _Runtime([Observation(content="simulated")])
    runtime = MixedRealityEnvironmentRuntime(
        policy=_policy(ToolRoute(name="write", mode="simulated", effect="write")),
        native=None,
        candidate=candidate,
        record=events.append,
    )
    with runtime.open(_task("write")) as session:
        assert session.execute(ToolCall(call_id="x", name="write")).content == "simulated"
    assert isinstance(events[-1], StepResult)
    assert events[-1].authority == "simulated"


def test_unsafe_state_mix_and_duplicate_routes_are_refused() -> None:
    """Unimplemented cross-store coherence cannot be asserted through routing alone."""
    with pytest.raises(ValidationError, match="state ownership"):
        _policy(
            ToolRoute(name="write", mode="real", effect="write"),
            ToolRoute(name="read", mode="simulated", effect="read"),
        )
    with pytest.raises(ValidationError, match="unique"):
        _policy(
            ToolRoute(name="read", mode="real", effect="read"),
            ToolRoute(name="read", mode="shadow", effect="read"),
        )


def test_exited_session_cannot_dispatch_again() -> None:
    """The caller cannot execute after either backend has been released."""
    native = _Runtime([Observation(content="never")])
    runtime = MixedRealityEnvironmentRuntime(
        policy=_policy(ToolRoute(name="read", mode="real", effect="read")),
        native=native,
        candidate=None,
        record=lambda event: None,
    )
    with runtime.open(_task()) as session:
        pass
    with pytest.raises(RuntimeError, match="stopped"):
        session.execute(ToolCall(call_id="late", name="read"))
    assert not native.actions


def test_invalid_charge_preserves_raw_result_and_stops_session() -> None:
    """A malformed cost receipt cannot permit another ambiguous native mutation."""
    native = _Runtime([Observation(content="changed", metadata={"economics": {"cost_usd": "bad"}})])
    events: list[MixedEvent] = []
    runtime = MixedRealityEnvironmentRuntime(
        policy=_policy(ToolRoute(name="write", mode="real", effect="write")),
        native=native,
        candidate=None,
        record=events.append,
    )
    with runtime.open(_task("write")) as session:
        with pytest.raises(RuntimeError, match="accounting failed"):
            session.execute(ToolCall(call_id="write", name="write"))
        with pytest.raises(RuntimeError, match="stopped"):
            session.execute(ToolCall(call_id="again", name="write"))
    assert isinstance(events[-1], StepResult)
    assert events[-1].authoritative_observation is not None
    assert events[-1].authoritative_observation.content == "changed"
    assert events[-1].accounting_failure == "ValidationError"
    assert events[-1].economics.cost_usd is None


def test_world_policy_does_not_expand_actor_tool_permissions() -> None:
    """A broad world policy never gives an actor tools outside its public task."""
    native = _Runtime([])
    events: list[MixedEvent] = []
    runtime = MixedRealityEnvironmentRuntime(
        policy=_policy(
            ToolRoute(name="read", mode="real", effect="read"),
            ToolRoute(name="write", mode="real", effect="write"),
        ),
        native=native,
        candidate=None,
        record=events.append,
    )
    with runtime.open(_task("read")) as session:
        with pytest.raises(ValueError, match="not routed"):
            session.execute(ToolCall(call_id="x", name="write"))
    assert not native.actions and not events


def test_missing_task_routes_fail_before_backends_open() -> None:
    """An advertised tool cannot remain secretly unimplemented until after dispatch."""
    native = _Runtime([])
    runtime = MixedRealityEnvironmentRuntime(
        policy=_policy(ToolRoute(name="read", mode="real", effect="read")),
        native=native,
        candidate=None,
        record=lambda event: None,
    )
    with pytest.raises(ValueError, match="task tools"):
        with runtime.open(_task("unknown")):
            pass
    assert not native.tasks


def test_failed_result_persistence_cannot_reexecute_native_write() -> None:
    """A write that succeeded before a disk failure is retained as an unresolved intent."""
    events: list[MixedEvent] = []
    native = _Runtime([Observation(content="committed")])

    def fail_after_intent(event: MixedEvent) -> None:
        """Persist intent but model a result-store outage."""
        if isinstance(event, StepResult):
            raise OSError("disk full")
        events.append(event)

    runtime = MixedRealityEnvironmentRuntime(
        policy=_policy(ToolRoute(name="write", mode="real", effect="write")),
        native=native,
        candidate=None,
        record=fail_after_intent,
    )
    with runtime.open(_task("write")) as session:
        with pytest.raises(OSError):
            session.execute(ToolCall(call_id="x", name="write"))
        with pytest.raises(RuntimeError, match="stopped"):
            session.execute(ToolCall(call_id="again", name="write"))
    assert len(native.actions) == len(events) == 1


def test_backend_exception_payload_is_not_actor_visible() -> None:
    """A backend's raw error message cannot leak credentials through the actor bridge."""
    native = _Runtime([ValueError("secret-native-value")])
    events: list[MixedEvent] = []
    runtime = MixedRealityEnvironmentRuntime(
        policy=_policy(ToolRoute(name="read", mode="real", effect="read")),
        native=native,
        candidate=None,
        record=events.append,
    )
    with runtime.open(_task()) as session:
        with pytest.raises(RuntimeError, match="ValueError") as failed:
            session.execute(ToolCall(call_id="x", name="read"))
    assert "secret-native-value" not in str(failed.value)
    assert isinstance(events[-1], StepResult)
    assert events[-1].authoritative_failure == "ValueError"
    assert events[-1].failures[0].details["phase"] == "authoritative_execution"
    assert "secret-native-value" not in events[-1].model_dump_json()


def test_shadow_predicts_before_native_dispatch_and_cannot_edit_its_action() -> None:
    """The native call gets the original public arguments, after the candidate has returned."""
    order: list[str] = []

    class Candidate(_Runtime):
        """Try to corrupt its own action copy while recording dispatch order."""

        def execute(self, action: ToolCall) -> Observation:
            """Mutate only the candidate's argument object."""
            order.append("candidate")
            action.arguments["id"] = "wrong-target"
            return super().execute(action)

    class Native(_Runtime):
        """Observe the still-original arguments after candidate execution."""

        def execute(self, action: ToolCall) -> Observation:
            """Check that native output was unavailable while predicting."""
            assert order == ["candidate"]
            assert action.arguments == {"id": "correct-target"}
            order.append("native")
            return super().execute(action)

    events: list[MixedEvent] = []
    native = Native([Observation(content="native")])
    candidate = Candidate([Observation(content="guess")])
    runtime = MixedRealityEnvironmentRuntime(
        policy=_policy(ToolRoute(name="read", mode="shadow", effect="read")),
        native=native,
        candidate=candidate,
        record=events.append,
    )
    action = ToolCall(call_id="x", name="read", arguments={"id": "correct-target"})
    with runtime.open(_task()) as session:
        assert session.execute(action).content == "native"
    assert action.arguments == {"id": "correct-target"}
    assert isinstance(events[0], StepIntent)
    assert events[0].action.arguments == action.arguments


def test_bad_shadow_accounting_keeps_valid_native_charge_in_operator_evidence() -> None:
    """An unknown aggregate must not erase the separately observed native charge."""
    native = _Runtime([_observation("committed", 0.5)])
    candidate = _Runtime([Observation(content="guess", metadata={"economics": {"invalid": 1}})])
    events: list[MixedEvent] = []
    runtime = MixedRealityEnvironmentRuntime(
        policy=_policy(ToolRoute(name="write", mode="shadow", effect="write")),
        native=native,
        candidate=candidate,
        record=events.append,
    )
    with runtime.open(_task("write")) as session:
        with pytest.raises(RuntimeError, match="accounting failed"):
            session.execute(ToolCall(call_id="x", name="write"))
    result = events[-1]
    assert isinstance(result, StepResult)
    assert result.economics.cost_usd is None
    assert result.authoritative_observation is not None
    native_cost = OperationEconomics.model_validate(
        result.authoritative_observation.metadata["economics"]
    ).cost_usd
    assert native_cost is not None and native_cost.value == 0.5


def test_shadow_prediction_is_retained_before_native_dispatch() -> None:
    """Native work starts only after operator evidence contains the sealed prediction."""
    events: list[MixedEvent] = []

    class Native(_Runtime):
        """Inspect the durable evidence boundary when native execution begins."""

        def execute(self, action: ToolCall) -> Observation:
            """Require the prediction to have been retained before this call."""
            assert [event.kind for event in events] == ["intent", "prediction"]
            return super().execute(action)

    runtime = MixedRealityEnvironmentRuntime(
        policy=_policy(ToolRoute(name="read", mode="shadow", effect="read")),
        native=Native([Observation(content="native")]),
        candidate=_Runtime([Observation(content="guess")]),
        record=events.append,
    )
    with runtime.open(_task()) as session:
        assert session.execute(ToolCall(call_id="x", name="read")).content == "native"
    assert [event.kind for event in events] == ["intent", "prediction", "result"]


def test_failed_prediction_persistence_prevents_native_dispatch() -> None:
    """A paid prediction whose receipt cannot be stored never authorizes a native effect."""
    native = _Runtime([Observation(content="changed")])
    candidate = _Runtime([Observation(content="guess")])

    def retain(event: MixedEvent) -> None:
        """Model storage failing between prediction and native dispatch."""
        if event.kind == "prediction":
            raise OSError("prediction receipt unavailable")

    runtime = MixedRealityEnvironmentRuntime(
        policy=_policy(ToolRoute(name="write", mode="shadow", effect="write")),
        native=native,
        candidate=candidate,
        record=retain,
    )
    with runtime.open(_task("write")) as session:
        with pytest.raises(OSError, match="prediction receipt"):
            session.execute(ToolCall(call_id="x", name="write"))
        with pytest.raises(RuntimeError, match="stopped"):
            session.execute(ToolCall(call_id="again", name="write"))
    assert len(candidate.actions) == 1
    assert not native.actions


ORDERS_SCHEMA = "CREATE TABLE orders (id INTEGER PRIMARY KEY, item TEXT NOT NULL);"
REFUNDS_SCHEMA = "CREATE TABLE refunds (id INTEGER PRIMARY KEY, amount INTEGER NOT NULL);"


def _orders(arguments: JsonObject) -> WorldAccess:
    """Orders are native records."""
    del arguments
    return WorldAccess(reads=frozenset({"orders"}), writes=frozenset({"orders"}))


def _refunds(arguments: JsonObject) -> WorldAccess:
    """Refunds are simulated records."""
    del arguments
    return WorldAccess(reads=frozenset({"refunds"}), writes=frozenset({"refunds"}))


def place_order(db: sqlite3.Connection, arguments: JsonObject, context: WorldContext) -> JsonValue:
    """Insert one native order."""
    del context
    cursor = db.execute("INSERT INTO orders (item) VALUES (?)", (arguments["item"],))
    return {"order": cursor.lastrowid}


def refund(db: sqlite3.Connection, arguments: JsonObject, context: WorldContext) -> JsonValue:
    """Insert one simulated refund."""
    del context
    cursor = db.execute("INSERT INTO refunds (amount) VALUES (?)", (arguments["amount"],))
    return {"refund": cursor.lastrowid}


WORLD = WorldManifest(
    domains=(
        StateDomain(name="orders", owner="native"),
        StateDomain(name="refunds", owner="simulated"),
    ),
    tools=(
        ToolEffectPolicy(name="place_order", reads=("orders",), writes=("orders",)),
        ToolEffectPolicy(name="refund", reads=("refunds",), writes=("refunds",), duration=2),
    ),
)


def _world_policy(world: WorldManifest | None = WORLD, **modes: str) -> MixedRealityPolicy:
    """Route orders natively and refunds to the candidate under one ownership contract."""
    return MixedRealityPolicy.model_validate(
        {
            "native_identity": sha256_json({"native": "orders"}),
            "candidate_identity": sha256_json({"candidate": "refunds"}),
            "routes": [
                {
                    "name": "place_order",
                    "mode": modes.get("place_order", "real"),
                    "effect": "write",
                },
                {"name": "refund", "mode": modes.get("refund", "simulated"), "effect": "write"},
            ],
            "world": world,
        }
    )


def _world_runtime(
    events: list[MixedEvent], *, shadow: bool = False
) -> MixedRealityEnvironmentRuntime:
    """Bind a native SQLite orders world and a simulated SQLite refunds world."""
    native = ExecutableWorldRuntime(
        WORLD,
        RecordWorldFactory(
            service={"name": "orders"},
            schema=ORDERS_SCHEMA,
            tools={"place_order": RecordTool(access=_orders, run=place_order)},
        ),
    )
    candidate = ExecutableWorldRuntime(
        WORLD,
        RecordWorldFactory(
            service={"name": "refunds"},
            schema=REFUNDS_SCHEMA,
            tools={"refund": RecordTool(access=_refunds, run=refund)},
        ),
        owner="simulated",
    )
    return MixedRealityEnvironmentRuntime(
        policy=_world_policy(place_order="shadow") if shadow else _world_policy(),
        native=native,
        candidate=candidate,
        record=events.append,
    )


def _order(call_id: str, item: str = "lamp") -> ToolCall:
    """Place one native order."""
    return ToolCall(call_id=call_id, name="place_order", arguments={"item": item})


def _refund(call_id: str, amount: int = 5) -> ToolCall:
    """Issue one simulated refund."""
    return ToolCall(call_id=call_id, name="refund", arguments={"amount": amount})


def test_world_manifest_admits_disjoint_mixed_routes_and_refuses_overlap() -> None:
    """Native and simulated routes coexist only when each touches its own domains."""
    assert _world_policy().world == WORLD
    with pytest.raises(ValidationError, match="state ownership"):
        _world_policy(world=None)
    with pytest.raises(ValidationError, match="does not own"):
        _world_policy(place_order="simulated")
    with pytest.raises(ValidationError, match="does not own"):
        _world_policy(refund="shadow")
    reads_only = WORLD.model_copy(
        update={"tools": (WORLD.tools[0].model_copy(update={"writes": ()}), WORLD.tools[1])}
    )
    with pytest.raises(ValidationError, match="disagrees"):
        _world_policy(world=reads_only)
    partial = WORLD.model_copy(update={"tools": WORLD.tools[:1]})
    with pytest.raises(ValidationError, match="no declared world effect policy"):
        _world_policy(world=partial)
    with pytest.raises(ValueError, match="policy's world manifest"):
        MixedRealityEnvironmentRuntime(
            policy=_world_policy(),
            native=ExecutableWorldRuntime(
                WORLD,
                RecordWorldFactory(
                    service={"name": "orders"},
                    schema=ORDERS_SCHEMA,
                    tools={"place_order": RecordTool(access=_orders, run=place_order)},
                ),
                owner="simulated",
            ),
            candidate=_Runtime([]),
            record=lambda event: None,
        )


def test_mixed_episode_checkpoints_and_forks_across_native_and_simulated_domains() -> None:
    """A fork's native and simulated effects never reach the parent or a sibling."""
    events: list[MixedEvent] = []
    runtime = _world_runtime(events)
    task = _task("place_order", "refund")
    with runtime.open(task) as parent:
        parent.execute(_order("o1"))
        parent.execute(_refund("r1"))
        saved = parent.checkpoint()
        assert saved.sequence == 2
        assert saved.native is not None and saved.candidate is not None
        assert saved.native.clock == saved.candidate.clock == 3
        with runtime.restore(task, saved) as left, runtime.restore(task, saved) as right:
            left.execute(_order("o2", "desk"))
            right.execute(_refund("r2", 9))
            assert parent.checkpoint() == saved
            left_state, right_state = left.checkpoint(), right.checkpoint()
            assert left_state.native != saved.native and left_state.candidate is not None
            assert left_state.candidate.image == saved.candidate.image
            assert right_state.native is not None and right_state.candidate is not None
            assert right_state.native.image == saved.native.image
            assert right_state.candidate.image != saved.candidate.image
            assert left_state.native is not None
            assert left_state.native.clock == left_state.candidate.clock == 4
            assert right_state.native.clock == right_state.candidate.clock == 5
            assert {left_state.episode_id, right_state.episode_id}.isdisjoint({saved.episode_id})
    intents = [event for event in events if isinstance(event, StepIntent)]
    assert [intent.operation_id.rsplit("-", 1)[1] for intent in intents] == ["1", "2", "3", "3"]
    assert len({intent.operation_id for intent in intents}) == 4


def test_mixed_restore_refuses_foreign_checkpoints_and_unrestorable_backends() -> None:
    """Checkpoints bind policy and task; backends outside the contract cannot checkpoint."""
    events: list[MixedEvent] = []
    runtime = _world_runtime(events)
    task = _task("place_order", "refund")
    with runtime.open(task) as session:
        session.execute(_order("o1"))
        saved = session.checkpoint()
    foreign = saved.model_copy(update={"policy_sha256": "0" * 64})
    with pytest.raises(RestoreError, match="routing policy"), runtime.restore(task, foreign):
        pass
    no_native = saved.model_copy(update={"native": None})
    with pytest.raises(RestoreError, match="no native world"), runtime.restore(task, no_native):
        pass
    plain = MixedRealityEnvironmentRuntime(
        policy=_policy(ToolRoute(name="read", mode="real", effect="read")),
        native=_Runtime([_observation("ok")]),
        candidate=None,
        record=events.append,
    )
    with plain.open(_task("read")) as session:
        session.execute(ToolCall(call_id="read-1", name="read", arguments={}))
        with pytest.raises(RuntimeError, match="cannot checkpoint"):
            session.checkpoint()


def test_candidate_world_failure_stops_the_mixed_episode() -> None:
    """A failed candidate-world call is rolled back there and the episode stops, unretried."""
    events: list[MixedEvent] = []
    runtime = _world_runtime(events)
    with runtime.open(_task("place_order", "refund")) as session:
        session.execute(_refund("r1"))
        with pytest.raises(RuntimeError, match="session is stopped"):
            session.execute(ToolCall(call_id="bad", name="refund", arguments={}))
        with pytest.raises(RuntimeError, match="cannot be checkpointed"):
            session.checkpoint()


def test_owned_shadow_route_refuses_before_opening_an_unavailable_mirror() -> None:
    """An owned simulator cannot shadow native domains without a separate mirror contract."""
    events: list[MixedEvent] = []
    with pytest.raises(ValueError, match="shadow.*mirror"):
        _world_runtime(events, shadow=True)
    assert not events


def test_mixed_pre_dispatch_refusal_preserves_both_clocks_and_allows_correction() -> None:
    """A proven rejected access is an actor observation, not an irrecoverable simulator outage."""
    read_world = WORLD.model_copy(
        update={
            "tools": (
                WORLD.tools[0].model_copy(update={"writes": ()}),
                WORLD.tools[1],
            )
        }
    )
    policy = MixedRealityPolicy(
        native_identity=sha256_json({"n": 1}),
        candidate_identity=sha256_json({"c": 1}),
        world=read_world,
        routes=(
            ToolRoute(name="place_order", mode="real", effect="read"),
            ToolRoute(name="refund", mode="simulated", effect="write"),
        ),
    )
    native = ExecutableWorldRuntime(
        read_world,
        RecordWorldFactory(
            service={"name": "orders"},
            schema=ORDERS_SCHEMA,
            tools={"place_order": RecordTool(access=_orders, run=place_order)},
        ),
    )
    candidate = ExecutableWorldRuntime(
        read_world,
        RecordWorldFactory(
            service={"name": "refunds"},
            schema=REFUNDS_SCHEMA,
            tools={"refund": RecordTool(access=_refunds, run=refund)},
        ),
        owner="simulated",
    )
    runtime = MixedRealityEnvironmentRuntime(
        policy=policy, native=native, candidate=candidate, record=lambda event: None
    )
    with runtime.open(_task("place_order", "refund")) as session:
        before = session.checkpoint()
        reply = execute_with_recovery(session, _order("refused"))
        assert reply.is_error
        after = session.checkpoint()
        assert after.native == before.native and after.candidate == before.candidate
        assert session.execute(_refund("allowed")).is_error is False
        done = session.checkpoint()
        assert done.native is not None and done.candidate is not None
        assert done.native.clock == done.candidate.clock == 2
