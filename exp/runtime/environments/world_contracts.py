"""Executable-world contracts: state domain ownership, tool effect policies and policy errors.

These immutable models declare the state domains of a restorable environment and their owners,
and each tool's declared effects and reply contract. The kernel in
:mod:`exp.runtime.environments.worlds` executes against them.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.runtime.environments.world_responses import ReplyStateBinding, validate_reply_schema

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
