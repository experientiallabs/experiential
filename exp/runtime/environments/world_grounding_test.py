"""Native bridges refuse stale facts, hidden effects and implicit cross-owner access."""

import pytest

from exp.common.models import ToolCall
from exp.runtime.environments.interface import Observation
from exp.runtime.environments.world_contracts import (
    AuthorityError,
    NativeImage,
    StateDomain,
    ToolEffectPolicy,
    UndeclaredEffectError,
    WorldEvent,
    WorldManifest,
    WorldPolicyError,
)
from exp.runtime.environments.world_grounding import NativeReadBridge
from exp.runtime.environments.worlds import ExecutableWorldRuntime, WorldContext
from exp.runtime.environments.worlds_test import DictBackend, DictFactory, _task


def manifest_fixture() -> WorldManifest:
    """Expose a native read dependency separately from the simulated tool's write access."""
    return WorldManifest(
        domains=(StateDomain(name="a", owner="native"), StateDomain(name="b", owner="simulated")),
        tools=(
            ToolEffectPolicy(name="get", reads=("a",), duration=0),
            ToolEffectPolicy(name="set", writes=("a",)),
            ToolEffectPolicy(name="predict", reads=("b",)),
        ),
        native_reads=(NativeReadBridge(name="current-a", target_tool="predict", probe_tool="get"),),
    )


def test_read_is_current_and_does_not_advance_any_native_state() -> None:
    """Pending events must fire first; the probe then leaves the exact native snapshot unchanged."""
    manifest = manifest_fixture()
    runtime = ExecutableWorldRuntime(
        manifest,
        DictFactory({"a": "old"}),
        events=(
            WorldEvent(
                event_id="update",
                due=0,
                order=0,
                kind="action",
                action=ToolCall(name="set", call_id="event", arguments={"value": "new"}),
            ),
        ),
    )
    target = ToolCall(name="predict", call_id="call", arguments={})
    bridge = manifest.native_reads[0]
    with runtime.open(_task("get", "set")) as native:
        with pytest.raises(WorldPolicyError, match="current-time"):
            native.observe_native(bridge, target)
        native.advance(0)
        before = native.checkpoint()
        receipt = native.observe_native(bridge, target)
        assert '"new"' in receipt.observation.content
        assert receipt.domains == {"a": before.image.domains["a"]}
        assert native.checkpoint() == before


@pytest.mark.parametrize("mutation", ["writes", "duration", "owner"])
def test_manifest_refuses_unsafe_probe_authority(mutation: str) -> None:
    """A claimed read probe cannot declare effects, time passage or simulated sources."""
    wire = manifest_fixture().model_dump(mode="json")
    if mutation == "owner":
        wire["domains"][0]["owner"] = "simulated"
    else:
        wire["tools"][0][mutation] = ["a"] if mutation == "writes" else 1
    with pytest.raises(ValueError, match="native reads"):
        WorldManifest.model_validate(wire)


def test_unmapped_arguments_and_unregistered_bridges_refuse() -> None:
    """Missing input is not a wildcard query and caller-supplied bridge changes cannot execute."""
    manifest = manifest_fixture()
    bridge = manifest.native_reads[0].model_copy(update={"argument_keys": {"id": "ticket_id"}})
    target = ToolCall(name="predict", call_id="c", arguments={})
    with pytest.raises(ValueError, match="requires target argument"):
        bridge.action(target)
    runtime = ExecutableWorldRuntime(manifest, DictFactory({"a": "old"}))
    with runtime.open(_task("get")) as world:
        with pytest.raises(AuthorityError, match="not registered"):
            world.observe_native(bridge, target)


@pytest.mark.parametrize("misbehavior", ["state", "rng", "schedule", "oversize"])
def test_bad_read_never_mutates_original_world(misbehavior: str) -> None:
    """Digest-preserving scheduler effects also refuse, not just business-state writes."""

    class BadRead(DictBackend):
        """A supposedly read-only adapter that violates one guarantee at a time."""

        def execute(self, action: ToolCall, context: WorldContext) -> Observation:
            """Deliberately exercise the outer kernel's independent enforcement."""
            if misbehavior == "state":
                self.values["a"] = "changed"
            if misbehavior == "rng":
                context.random()
            if misbehavior == "schedule":
                context.schedule(ToolCall(name="set", call_id="event", arguments={}), delay=1)
            return Observation(
                content="x" * (20000 if misbehavior == "oversize" else 1),
                is_error=False,
            )

    class BadFactory(DictFactory):
        """Restore the same faulty implementation into the isolated read fork."""

        def restore(self, image: NativeImage) -> BadRead:
            """Preserve exact snapshot values."""
            return BadRead({name: value.decode() for name, value in image.parts.items()})

    manifest = manifest_fixture()
    with ExecutableWorldRuntime(manifest, BadFactory({"a": "old"})).open(_task("get")) as world:
        before = world.checkpoint()
        with pytest.raises((WorldPolicyError, UndeclaredEffectError)):
            world.observe_native(
                manifest.native_reads[0], ToolCall(name="predict", call_id="c", arguments={})
            )
        assert world.checkpoint() == before


def test_simulated_call_refuses_missing_stale_or_wrong_action_receipts() -> None:
    """Standalone execution and mismatched receipt reuse cannot bypass native conditioning."""
    manifest = manifest_fixture()
    target = ToolCall(name="predict", call_id="c", arguments={"domain": "b"})
    with ExecutableWorldRuntime(manifest, DictFactory({"a": "old"})).open(_task("get")) as native:
        receipt = native.observe_native(manifest.native_reads[0], target)
    candidate = ExecutableWorldRuntime(manifest, DictFactory({"b": "simulated"}), owner="simulated")
    with candidate.open(_task("predict")) as world:
        before = world.checkpoint()
        for receipts in (
            (),
            (receipt.model_copy(update={"logical_time": 9}),),
            (receipt.model_copy(update={"target_sha256": "0" * 64}),),
        ):
            with pytest.raises(AuthorityError, match="exact current"):
                world.execute_grounded(target, receipts)
            assert world.checkpoint() == before
        assert "simulated" in world.execute_grounded(target, (receipt,)).content


def test_missing_grounding_refuses_before_due_events() -> None:
    """An inadmissible call cannot consume queued events or change outage state."""
    manifest = manifest_fixture()
    runtime = ExecutableWorldRuntime(
        manifest,
        DictFactory({"b": "simulated"}),
        owner="simulated",
        events=(
            WorldEvent(event_id="outage", due=0, kind="outage", tools=("predict",), duration=5),
        ),
    )
    with runtime.open(_task("predict")) as world:
        before = world.checkpoint()
        with pytest.raises(AuthorityError, match="exact current"):
            world.execute(ToolCall(name="predict", call_id="no-receipt", arguments={"domain": "b"}))
        assert world.checkpoint() == before
        assert world.last_execution is not None
        assert not world.last_execution.dispatched
        assert world.last_execution.phase == "admission"


def test_negative_read_is_retained_but_cannot_authorize_simulation() -> None:
    """A known read error leaves the native image intact and cannot enter prediction."""

    class NegativeRead(DictBackend):
        """Return an ordinary negative read without any state or scheduler effect."""

        def execute(self, action: ToolCall, context: WorldContext) -> Observation:
            """Return an explicit unavailable observation."""
            return Observation(content="Requested range exceeds the current bounds.", is_error=True)

    class NegativeFactory(DictFactory):
        """Restore the same negative-read implementation in its isolated fork."""

        def restore(self, image: NativeImage) -> NegativeRead:
            """Restore exact file values."""
            return NegativeRead({name: value.decode() for name, value in image.parts.items()})

    manifest = manifest_fixture()
    target = ToolCall(name="predict", call_id="negative", arguments={"domain": "b"})
    with ExecutableWorldRuntime(manifest, NegativeFactory({"a": "old"})).open(
        _task("get")
    ) as native:
        before = native.checkpoint()
        receipt = native.observe_native(manifest.native_reads[0], target)
        assert receipt.observation.is_error and native.checkpoint() == before
    with ExecutableWorldRuntime(manifest, DictFactory({"b": "unknown"}), owner="simulated").open(
        _task("predict")
    ) as candidate:
        with pytest.raises(AuthorityError, match="native read"):
            candidate.execute_grounded(target, (receipt,))
