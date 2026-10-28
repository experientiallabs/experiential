"""Kernel rules of executable worlds, exercised with a minimal in-memory backend."""

import json

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject, JsonValue, Sha256, sha256_json
from exp.common.models import ToolCall
from exp.common.tasks import TaskCase, ToolSchema
from exp.runtime.environments.interface import Observation
from exp.runtime.environments.world_contracts import (
    AuthorityError,
    NativeImage,
    RestoreError,
    StateDomain,
    ToolEffectPolicy,
    UndeclaredEffectError,
    WorldAccess,
    WorldEvent,
    WorldManifest,
)
from exp.runtime.environments.worlds import ExecutableWorldRuntime, WorldContext


class DictBackend:
    """Native string values per domain with tools that can misbehave on purpose."""

    def __init__(self, values: dict[str, str]) -> None:
        """Own one private value map."""
        self.values = values

    def access(self, action: ToolCall) -> WorldAccess:
        """Resolve access; ``lie`` and ``boom`` under-declare what they change."""
        domain = str(action.arguments.get("domain", "a"))
        if action.name in {"set", "boom", "later", "roll"}:
            return WorldAccess(writes=frozenset({domain}))
        if action.name == "lie":
            return WorldAccess(writes=frozenset({"a"}))
        return WorldAccess(reads=frozenset({domain}))

    def execute(self, action: ToolCall, context: WorldContext) -> Observation:
        """Apply one fixture tool."""
        domain = str(action.arguments.get("domain", "a"))
        if action.name == "set":
            self.values[domain] = str(action.arguments["value"])
        elif action.name == "lie":
            self.values["b"] = "smuggled"
        elif action.name == "boom":
            self.values[domain] = "partial"
            context.schedule(_call("set", domain="a", value="never"), delay=1)
            raise RuntimeError("backend crashed mid-call")
        elif action.name == "later":
            tool = str(action.arguments.get("tool", "set"))
            event = context.schedule(
                _call(tool, domain=domain, value=str(action.arguments["value"])),
                delay=int(str(action.arguments["delay"])),
            )
            if action.arguments.get("cancel") is not None:
                context.cancel(str(action.arguments["cancel"]))
            return Observation(content=event)
        elif action.name == "roll":
            self.values[domain] = f"{context.random():.12f}"
        return Observation(content=json.dumps({"value": self.values.get(domain)}))

    def domain_digests(self) -> dict[str, Sha256]:
        """Digest each domain's value."""
        return {name: sha256_json(value) for name, value in sorted(self.values.items())}

    def capture(self) -> NativeImage:
        """Capture every value as UTF-8 bytes."""
        return NativeImage(
            parts={name: value.encode() for name, value in self.values.items()},
            domains=self.domain_digests(),
        )

    def close(self) -> None:
        """Nothing to release."""


class DictFactory:
    """Create dict backends from fixed initial values."""

    def __init__(self, values: dict[str, str], name: str = "dict-v1") -> None:
        """Bind initial values and an identity."""
        self.values = values
        self.name = name

    @property
    def definition(self) -> JsonObject:
        """Identify the fixture backend."""
        return {"name": self.name, "values": dict(self.values)}

    def create(self, task: TaskCase) -> DictBackend:
        """Return a backend with a private copy of the initial values."""
        del task
        return DictBackend(dict(self.values))

    def restore(self, image: NativeImage) -> DictBackend:
        """Return a backend with a private copy of the captured values."""
        return DictBackend({name: payload.decode() for name, payload in image.parts.items()})


TOOLS = ("set", "lie", "boom", "later", "roll", "get")


def _call(name: str, **arguments: JsonValue) -> ToolCall:
    """Build one fixture call."""
    return ToolCall(call_id=f"{name}-call", name=name, arguments=arguments)


def _task(*names: str) -> TaskCase:
    """Return a public task exposing the named tools."""
    return TaskCase(
        task_id="world-kernel",
        lineage_group_id="world-kernel",
        partition="fit",
        instruction="Operate the fixture.",
        workload_weight=1,
        origin="authored",
        tools=tuple(
            ToolSchema(name=name, description="Fixture.", input_schema={"type": "object"})
            for name in (names or TOOLS)
        ),
    )


def _manifest(*, owner_b: str = "native") -> WorldManifest:
    """Declare domains ``a`` and ``b`` with every fixture tool limited to ``a``."""
    return WorldManifest(
        domains=(
            StateDomain(name="a", owner="native"),
            StateDomain.model_validate({"name": "b", "owner": owner_b}),
        ),
        tools=tuple(
            ToolEffectPolicy(name=name, reads=("a",), writes=() if name == "get" else ("a",))
            for name in TOOLS
        ),
    )


def _runtime(**options: int | tuple[WorldEvent, ...]) -> ExecutableWorldRuntime:
    """Bind the fixture backend under the default manifest."""
    seed = options.get("seed", 0)
    events = options.get("events", ())
    assert isinstance(seed, int) and isinstance(events, tuple)
    return ExecutableWorldRuntime(
        _manifest(), DictFactory({"a": "0", "b": "0"}), seed=seed, events=events
    )


def test_lying_backend_is_rolled_back_after_execution() -> None:
    """A change beyond the resolved writes is detected by digests and undone."""
    with _runtime().open(_task()) as world:
        before = world.checkpoint()
        with pytest.raises(UndeclaredEffectError, match=r"changed \['b'\]"):
            world.execute(_call("lie"))
        assert world.checkpoint() == before


def test_backend_failure_rolls_back_state_and_staged_events() -> None:
    """A crash mid-call leaves no partial write and no scheduled event."""
    with _runtime().open(_task()) as world:
        before = world.checkpoint()
        with pytest.raises(RuntimeError, match="mid-call"):
            world.execute(_call("boom", domain="a"))
        assert world.checkpoint() == before


class BlindingBackend(DictBackend):
    """A backend whose state digest fails after any write, as a crashed probe would."""

    def __init__(self, values: dict[str, str]) -> None:
        """Start with a readable digest."""
        super().__init__(values)
        self.blind = False

    def execute(self, action: ToolCall, context: WorldContext) -> Observation:
        """Apply the call, then lose the ability to digest the mutated state."""
        observation = super().execute(action, context)
        self.blind = True
        return observation

    def domain_digests(self) -> dict[str, Sha256]:
        """Raise once a call has mutated state."""
        if self.blind:
            raise OSError("state digest unavailable")
        return super().domain_digests()


class BlindingFactory(DictFactory):
    """Create blinding backends; restore yields a fresh, readable one."""

    def create(self, task: TaskCase) -> DictBackend:
        """Return a blinding backend over the initial values."""
        del task
        return BlindingBackend(dict(self.values))


class CorruptRestoreFactory(DictFactory):
    """Restore every image as different state, so a rollback cannot be verified."""

    def restore(self, image: NativeImage) -> DictBackend:
        """Return a backend that does not hold the captured image."""
        return DictBackend({name: "corrupt" for name in image.parts})


def test_failed_post_call_digest_rolls_back_the_call() -> None:
    """A digest failure after execution is inside the rollback scope, not a kept mutation."""
    runtime = ExecutableWorldRuntime(_manifest(), BlindingFactory({"a": "0", "b": "0"}))
    with runtime.open(_task()) as world:
        before = world.checkpoint()
        with pytest.raises(OSError, match="digest unavailable"):
            world.execute(_call("set", domain="a", value="mutated"))
        assert world.checkpoint() == before


def test_unverified_rollback_closes_the_world_and_stops_the_episode() -> None:
    """A restore that does not reproduce the pre-call state never leaves a world in use."""
    runtime = ExecutableWorldRuntime(_manifest(), CorruptRestoreFactory({"a": "0", "b": "0"}))
    with runtime.open(_task()) as world:
        with pytest.raises(RestoreError, match="rollback failed") as error:
            world.execute(_call("boom", domain="a"))
        assert isinstance(error.value.__cause__, RestoreError)
        with pytest.raises(RuntimeError, match="closed"):
            world.execute(_call("get"))
    with runtime.open(_task()) as world:
        # The event fires as the call's duration elapses; its failed rollback is not retained
        # as a "failed" event outcome.
        with pytest.raises(RestoreError, match="rollback failed"):
            world.execute(_call("later", tool="boom", value="x", delay=1))


class FailingBackend(DictBackend):
    """A backend whose release fails, as a lost remote sandbox would."""

    def close(self) -> None:
        """Fail to release."""
        raise OSError("backend release failed")


class FaultyRollbackFactory(DictFactory):
    """Fail one step of a rollback: the restore, the restored release or the old release."""

    def __init__(self, values: dict[str, str], fault: str) -> None:
        """Bind the fault to inject."""
        super().__init__(values)
        self.fault = fault

    def create(self, task: TaskCase) -> DictBackend:
        """Return an initial backend, failing on release for the old-release fault."""
        del task
        return (FailingBackend if self.fault == "old_close" else DictBackend)(dict(self.values))

    def restore(self, image: NativeImage) -> DictBackend:
        """Fail, or return a corrupt backend that also fails on release, or restore exactly."""
        if self.fault == "restore":
            raise OSError("restore failed")
        if self.fault == "restored_close":
            return FailingBackend({name: "corrupt" for name in image.parts})
        return super().restore(image)


@pytest.mark.parametrize("fault", ["restore", "restored_close", "old_close"])
def test_a_failing_rollback_step_always_closes_the_world(fault: str) -> None:
    """No restore or cleanup failure leaves the world usable or replaces the restore failure."""
    runtime = ExecutableWorldRuntime(
        _manifest(), FaultyRollbackFactory({"a": "0", "b": "0"}, fault)
    )
    with runtime.open(_task()) as world:
        with pytest.raises(RestoreError, match="it is closed") as error:
            world.execute(_call("later", tool="boom", value="x", delay=1))
        if fault == "restored_close":
            assert error.value.__notes__ == ["backend cleanup also failed: OSError"]
        with pytest.raises(RuntimeError, match="world is closed"):
            world.execute(_call("get"))


def test_undeclared_tools_and_domains_are_refused() -> None:
    """Every task tool needs a policy; access to an unknown domain is never admitted."""
    runtime = _runtime()
    with pytest.raises(UndeclaredEffectError, match="no declared effect policy"):
        with runtime.open(_task("set", "wipe")):
            pass
    with runtime.open(_task()) as world:
        with pytest.raises(UndeclaredEffectError, match="undeclared domain"):
            world.execute(_call("get", domain="z"))
        with pytest.raises(UndeclaredEffectError, match="no declared effect policy"):
            world.execute(_call("wipe"))


def test_simulated_domain_is_never_touched_natively() -> None:
    """Reads, not only writes, of a simulated-owned domain are refused before execution."""
    runtime = ExecutableWorldRuntime(_manifest(owner_b="simulated"), DictFactory({"a": "0"}))
    with runtime.open(_task()) as world:
        with pytest.raises(AuthorityError, match="'b', owned by the simulated side"):
            world.execute(_call("get", domain="b"))
        assert world.clock == 0


def test_backend_must_hold_exactly_its_owned_domains() -> None:
    """A backend missing an owned domain, or holding a foreign one, cannot open."""
    runtime = ExecutableWorldRuntime(
        _manifest(owner_b="simulated"), DictFactory({"a": "0", "b": "0"})
    )
    with pytest.raises(RestoreError, match="align the backend"):
        with runtime.open(_task()):
            pass


def test_events_fire_in_order_and_cancelled_events_never_fire() -> None:
    """Ties break by scheduling order; cancellation is staged with the accepting call."""
    with _runtime().open(_task()) as world:
        first = world.execute(_call("later", value="1", delay=5)).content
        world.execute(_call("later", value="2", delay=3))
        world.execute(_call("later", value="3", delay=2, cancel=first))
        assert [event.event_id for event in world.pending] == ["scheduled-1", "scheduled-2"]
        fired = world.advance(10)
        assert [(event.event_id, event.at) for event in fired] == [
            ("scheduled-1", 4),
            ("scheduled-2", 4),
        ]
        assert json.loads(world.execute(_call("get")).content) == {"value": "3"}


def test_scheduling_an_undeclared_tool_rolls_the_call_back() -> None:
    """A backend cannot schedule a tool outside the manifest."""
    with _runtime().open(_task()) as world:
        before = world.checkpoint()
        with pytest.raises(UndeclaredEffectError, match="'wipe' has no declared effect policy"):
            world.execute(_call("later", value="1", delay=1, tool="wipe"))
        assert world.checkpoint() == before


def test_event_refused_after_branch_is_retained_not_raised() -> None:
    """An event whose domain left this world's authority is recorded as refused."""
    with _runtime().open(_task()) as world:
        world.execute(_call("later", value="9", delay=2))
        world.branch("a")
        (outcome,) = world.advance(5)
        assert outcome.status == "refused"
        assert "owned by the simulated side" in outcome.detail


def test_seeded_randomness_is_restored_with_its_position() -> None:
    """The random stream is deterministic per seed and resumes after restore."""
    runtime = _runtime(seed=11)
    with runtime.open(_task()) as world:
        world.execute(_call("roll"))
        saved = world.checkpoint()
        world.execute(_call("roll"))
        expected = world.execute(_call("get")).content
        with runtime.restore(_task(), saved) as restored:
            restored.execute(_call("roll"))
            assert restored.execute(_call("get")).content == expected
    with _runtime(seed=12).open(_task()) as other:
        other.execute(_call("roll"))
        other.execute(_call("roll"))
        assert other.execute(_call("get")).content != expected


def test_outage_degrades_without_executing_and_expires() -> None:
    """A degraded tool answers with an error observation and changes nothing."""
    outage = WorldEvent(event_id="outage", due=1, kind="outage", tools=("set",), duration=2)
    with _runtime(events=(outage,)).open(_task()) as world:
        world.advance(1)
        reply = world.execute(_call("set", value="x"))
        assert reply.is_error and "unavailable until t=3" in reply.content
        assert json.loads(world.execute(_call("get")).content) == {"value": "0"}
        assert not world.execute(_call("set", value="x")).is_error


def test_restore_refuses_another_backend_or_authority() -> None:
    """Checkpoints bind the backend definition and the owner that captured them."""
    with _runtime().open(_task()) as world:
        saved = world.checkpoint()
    other = ExecutableWorldRuntime(_manifest(), DictFactory({"a": "0", "b": "0"}, name="dict-v2"))
    with pytest.raises(RestoreError, match="backend definition"), other.restore(_task(), saved):
        pass
    foreign = saved.model_copy(update={"owner": "simulated"})
    with pytest.raises(RestoreError, match="authority"), _runtime().restore(_task(), foreign):
        pass


def test_closed_world_and_bad_events_are_refused() -> None:
    """Use after exit, reserved identities and backwards time all fail loudly."""
    with _runtime().open(_task()) as world:
        with pytest.raises(ValueError, match="backwards"):
            world.advance(-1)
    with pytest.raises(RuntimeError, match="closed"):
        world.execute(_call("get"))
    with pytest.raises(ValueError, match="reserved"):
        _runtime(
            events=(WorldEvent(event_id="scheduled-0", due=1, kind="action", action=_call("get")),)
        )
    with pytest.raises(ValidationError, match="outage"):
        WorldEvent(event_id="x", due=1, kind="outage", tools=())


def test_invalid_reply_shape_rolls_back_native_state_clock_and_randomness() -> None:
    """A backend can mutate state and RNG before returning the wrong successful reply shape."""
    policy = ToolEffectPolicy(name="roll", writes=("a",)).model_copy(
        update={
            "reply_schema": {
                "type": "object",
                "required": ["first"],
                "properties": {"first": {"type": "string"}},
            }
        }
    )
    manifest = WorldManifest(domains=(StateDomain(name="a", owner="native"),), tools=(policy,))
    runtime = ExecutableWorldRuntime(manifest, DictFactory({"a": "original"}))
    with runtime.open(
        _task().model_copy(update={"tools": (_task().tools[TOOLS.index("roll")],)})
    ) as world:
        before = world.checkpoint()
        with pytest.raises(ValueError, match="reply.*schema"):
            world.execute(_call("roll"))
        assert world.checkpoint() == before
