"""Executable-world contracts: state domains, effect policies, manifests, events and checkpoints.

These immutable models declare what a restorable environment is and what a checkpoint binds:
the state domains and their owners, each tool's declared effects and reply contract, scheduled
events, the native image and the checkpoint that ties them to the kernel's clock, queue and
randomness. The kernel in :mod:`exp.runtime.environments.worlds` executes against them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from exp.common.core.artifacts import (
    ContractModel,
    JsonObject,
    Sha256,
)
from exp.common.models import ToolCall
from exp.runtime.environments.world_grounding import NativeReadBridge
from exp.runtime.environments.world_responses import (
    ReplyStateBinding,
    validate_reply_schema,
)
from exp.runtime.environments.world_state_paths import ArgumentKey

Owner = Literal["native", "simulated"]


class WorldPolicyError(PermissionError):
    """A call violates world policy; its receipt distinguishes refusal from rollback."""


class UndeclaredEffectError(WorldPolicyError):
    """A call reads or writes outside its tool's declared effect policy."""


class AuthorityError(WorldPolicyError):
    """A call touches a state domain that this world's backend does not own."""


class RestoreError(ValueError):
    """A checkpoint cannot be restored by this runtime, or restored state did not verify."""


class StateDomain(ContractModel):
    """One named partition of episode state with exactly one owner.

    Attributes:
        name: Exact domain name, such as a table or an artifact path.
        owner: Backend authority for the episode: ``native`` or ``simulated``.
        kind: ``state`` for business state; ``effect`` for an external effect sink.
        sandboxed: Required for a native effect domain: the native sink records effects
            instead of performing them. Unsandboxed external effects belong to the simulator.
        state_schema: Optional local schema over this domain's backend state projection,
            checked at initialization, restore and every transition, including error replies.
    """

    name: str = Field(min_length=1, max_length=256)
    owner: Owner
    kind: Literal["state", "effect"] = "state"
    sandboxed: bool = False
    state_schema: JsonObject | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def sandboxed_native_effects(self) -> Self:
        """Refuse a native external-effect domain without a sandboxed sink."""
        if self.state_schema is not None:
            validate_reply_schema(self.state_schema)
        if self.kind == "effect" and self.owner == "native" and not self.sandboxed:
            raise ValueError(
                f"effect domain {self.name!r} is native but not sandboxed; declare a sandboxed "
                "native sink or make the simulator own it"
            )
        return self


class ToolEffectPolicy(ContractModel):
    """The declared effect policy of one tool.

    Attributes:
        name: Exact tool name.
        reads: Domains the tool may read. Declared writes may also be read.
        writes: Domains the tool may change, including effect sinks.
        duration: Logical ticks the clock advances after one call of this tool.
        reply_schema: Optional local JSON Schema for successful decoded replies. Errors remain
            error observations. This validates representation, not factual correctness.
        reply_bindings: Optional equalities with private backend state. These validate local
            consistency only; missing projections refuse execution before dispatch.
    """

    name: str = Field(min_length=1, max_length=256)
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    duration: int = Field(default=1, ge=0, strict=True)
    reply_schema: JsonObject | None = Field(default=None, exclude_if=lambda value: value is None)

    reply_bindings: tuple[ReplyStateBinding, ...] = Field(
        default=(), exclude_if=lambda rows: not rows
    )

    @model_validator(mode="after")
    def local_reply_contract(self) -> Self:
        """Require a self-contained reply predicate before backend execution is possible."""
        if self.reply_schema is not None:
            validate_reply_schema(self.reply_schema)
        return self

    def domains(self) -> frozenset[str]:
        """Return every domain this tool may read or write."""
        return frozenset(self.reads) | frozenset(self.writes)


def spanning_tools(tools: Iterable[ToolEffectPolicy], owners: Mapping[str, str]) -> tuple[str, ...]:
    """Return the tools whose declared domains belong to more than one owner.

    This is the single ownership rule for routed execution: a tool executes on exactly one
    owner, so every domain it reads or writes must share that owner. Callers that partition
    domains between other executors (such as induced code and a predictor) apply it too.

    Args:
        tools: Declared tool effect policies.
        owners: Owner label of every domain the tools declare.

    Returns:
        Names of tools that span owners, in declaration order.
    """
    return tuple(tool.name for tool in tools if len({owners[name] for name in tool.domains()}) > 1)


class WorldManifest(ContractModel):
    """State domains and per-tool effect policies frozen before an episode.

    Attributes:
        domains: Unique domains with their initial owners.
        tools: Unique tool policies. A tool may only declare domains of a single owner:
            cross-ownership writes need a branch or sync protocol this contract refuses.
        native_reads: Explicit read-only observations permitted to condition simulated calls.
    """

    domains: tuple[StateDomain, ...] = Field(min_length=1)
    tools: tuple[ToolEffectPolicy, ...] = Field(min_length=1)
    native_reads: tuple[NativeReadBridge, ...] = Field(default=(), exclude_if=lambda rows: not rows)

    @model_validator(mode="after")
    def closed_declarations(self) -> Self:
        """Refuse duplicate names, undeclared domains and tools that span owners."""
        if len({domain.name for domain in self.domains}) != len(self.domains):
            raise ValueError("state domains must have unique names")
        if len({tool.name for tool in self.tools}) != len(self.tools):
            raise ValueError("tool effect policies must have unique names")
        owners = self.ownership()
        for tool in self.tools:
            unknown = sorted(tool.domains() - owners.keys())
            if unknown:
                raise ValueError(f"tool {tool.name!r} declares unknown domains {unknown!r}")
            if any(
                not binding.state_path
                or (
                    not isinstance(binding.state_path[0], ArgumentKey)
                    and (
                        not isinstance(binding.state_path[0], str)
                        or binding.state_path[0] not in tool.domains()
                    )
                )
                for binding in tool.reply_bindings
            ):
                raise ValueError(
                    "reply state bindings must stay inside the tool's declared domains"
                )
        if spanning := spanning_tools(self.tools, owners):
            raise ValueError(
                f"tool {spanning[0]!r} spans native and simulated domains; split the tool or "
                "give its domains one owner"
            )
        if len({bridge.name for bridge in self.native_reads}) != len(self.native_reads):
            raise ValueError("native read bridge names must be unique")
        for bridge in self.native_reads:
            probe, target = self.tool(bridge.probe_tool), self.tool(bridge.target_tool)
            if (
                probe.writes
                or probe.duration
                or not probe.reads
                or any(owners[name] != "native" for name in probe.domains())
                or not target.domains()
                or any(owners[name] != "simulated" for name in target.domains())
            ):
                raise ValueError(
                    "native read bridges require zero-duration native reads and simulated targets"
                )
        return self

    def ownership(self) -> dict[str, Owner]:
        """Return the initial domain ownership map."""
        return {domain.name: domain.owner for domain in self.domains}

    def tool(self, name: str) -> ToolEffectPolicy:
        """Return one tool's declared policy, refusing tools without one."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise UndeclaredEffectError(
            f"tool {name!r} has no declared effect policy; declare it in the world manifest"
        )


class WorldAccess(ContractModel):
    """The domains one exact call reads and writes, resolved before execution.

    Attributes:
        reads: Domains the call reads.
        writes: Domains the call may change.
    """

    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()


class WorldEvent(ContractModel):
    """One pending event owned by the episode clock.

    Attributes:
        event_id: Unique event identity; generated events use the ``scheduled-`` prefix.
        due: Logical time at which the event fires.
        order: Tie-breaker among events due at the same time.
        kind: ``action`` runs a tool call through the world's policy; ``outage`` makes
            named tools return error observations for ``duration`` ticks.
        action: The scheduled tool call of an action event.
        tools: Tools an outage degrades.
        duration: Ticks an outage lasts.
    """

    event_id: str = Field(min_length=1, max_length=256)
    due: int = Field(ge=0, strict=True)
    order: int = Field(default=0, ge=0, strict=True)
    kind: Literal["action", "outage"]
    action: ToolCall | None = None
    tools: tuple[str, ...] = ()
    duration: int = Field(default=0, ge=0, strict=True)

    @model_validator(mode="after")
    def complete_kind(self) -> Self:
        """Require exactly the fields each event kind uses."""
        if self.kind == "action" and (self.action is None or self.tools or self.duration):
            raise ValueError("an action event has one action and no outage fields")
        if self.kind == "outage" and (
            self.action is not None or not self.tools or not self.duration
        ):
            raise ValueError("an outage event names its tools and a positive duration")
        return self


class FiredEvent(ContractModel):
    """The retained outcome of one fired event.

    Attributes:
        event_id: Identity of the fired event.
        at: Logical time at which it fired.
        status: Applied, refused by policy, skipped by an outage, or failed and rolled back.
        detail: Observation content or the refusal reason, never a backend traceback.
    """

    event_id: str
    at: int
    status: Literal["applied", "refused", "degraded", "failed"]
    detail: str


class NativeImage(ContractModel):
    """Exact native state captured from a backend.

    Attributes:
        parts: Backend-defined byte parts, enough to rebuild the state in a fresh instance.
        domains: Digest of every domain the backend holds, used to verify restoration.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, ser_json_bytes="base64", val_json_bytes="base64"
    )

    parts: dict[str, bytes]
    domains: dict[str, Sha256]


class ExecutableCheckpoint(ContractModel):
    """Everything needed to resume one executable world in a fresh instance.

    Attributes:
        format: Contract version; other formats fail closed.
        manifest_sha256: Digest of the world manifest.
        backend_sha256: Digest of the backend factory definition.
        owner: Authority of the backend that produced the image.
        image: Exact native state.
        ownership: Current domain ownership, including branch transfers.
        clock: Logical time.
        next_event: Counter for generated event identities and tie order.
        pending: Pending events in firing order.
        degraded: Tools under an outage, mapped to the time the outage ends.
        fired: Outcomes of every fired event.
        seed: Seed of the deterministic random stream.
        draws: Random values already drawn.
        steps: Tool calls executed by the episode.
    """

    format: Literal["executable-world-v1"] = "executable-world-v1"
    manifest_sha256: Sha256
    backend_sha256: Sha256
    owner: Owner
    image: NativeImage
    ownership: dict[str, Owner]
    clock: int = Field(ge=0)
    next_event: int = Field(ge=0)
    pending: tuple[WorldEvent, ...]
    degraded: dict[str, int]
    fired: tuple[FiredEvent, ...]
    seed: int
    draws: int = Field(ge=0)
    steps: int = Field(ge=0)
