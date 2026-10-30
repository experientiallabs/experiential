"""Explicit native read bridges used to condition one simulated call, never to transfer state."""

from copy import deepcopy
from typing import Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, Sha256, sha256_json
from exp.common.models import ToolCall
from exp.runtime.environments.interface import Observation


class NativeReadBridge(ContractModel):
    """A frozen mapping from a simulated call to one private native read tool.

    Attributes:
        name: Unique receipt key within the world manifest.
        target_tool: Simulated tool requiring this observation on every call.
        probe_tool: Native tool with no writes and zero duration.
        arguments: Fixed probe arguments, bound into the manifest identity.
        argument_keys: Probe argument names mapped to exact top-level target argument names.
        maximum_response_bytes: Bound on the complete observation before model dispatch.
    """

    name: str = Field(min_length=1, max_length=128)
    target_tool: str = Field(min_length=1)
    probe_tool: str = Field(min_length=1)
    arguments: JsonObject = Field(default_factory=dict)
    argument_keys: dict[str, str] = Field(default_factory=dict)
    maximum_response_bytes: int = Field(default=16384, ge=1, le=262144)

    @model_validator(mode="after")
    def disjoint_arguments(self) -> Self:
        """Refuse ambiguous precedence between fixed and call-derived arguments."""
        if self.arguments.keys() & self.argument_keys.keys():
            raise ValueError("native probe fixed and mapped arguments overlap")
        return self

    def action(self, target: ToolCall) -> ToolCall:
        """Build the exact private read; missing caller arguments refuse before execution."""
        if target.name != self.target_tool:
            raise ValueError("native bridge belongs to another target tool")
        arguments = deepcopy(self.arguments)
        for destination, source in self.argument_keys.items():
            if source not in target.arguments:
                raise ValueError(f"native probe requires target argument {source!r}")
            arguments[destination] = deepcopy(target.arguments[source])
        return ToolCall(
            name=self.probe_tool,
            call_id="native-probe-"
            + sha256_json(
                {"bridge": self.model_dump(mode="json"), "target": target.model_dump(mode="json")}
            ),
            arguments=arguments,
        )


class NativeReadReceipt(ContractModel):
    """A current native observation for exactly one simulated action.

    Attributes:
        bridge_sha256: Exact manifest read bridge.
        target_sha256: Exact tool call conditioned by this observation.
        checkpoint_sha256: Whole native snapshot read before the simulated call.
        backend_sha256: Native implementation and initialization identity.
        logical_time: Native clock at observation, also required at simulated dispatch.
        domains: Digests of the native domains actually read.
        action: Private probe call, absent from the actor's tool history.
        observation: Original bounded native reply, never a verifier verdict.
    """

    bridge_sha256: Sha256
    target_sha256: Sha256
    checkpoint_sha256: Sha256
    backend_sha256: Sha256
    logical_time: int = Field(ge=0)
    domains: dict[str, Sha256]
    action: ToolCall
    observation: Observation
