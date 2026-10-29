"""Explicit engine proposals, state effects and immutable prepared evidence."""

import copy
from collections.abc import Callable
from typing import Annotated, Literal, Protocol

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, Sha256
from exp.common.models import ToolCall
from exp.simulation.engines.text.generic_delta import apply_delta


class ReplaceState(ContractModel):
    """Complete-state effect from an engine whose protocol declares replacement.

    Attributes:
        kind: Discriminator forbidding simultaneous replacement and edits.
        state: Complete candidate-owned state, never a native refresh.
    """

    kind: Literal["replace"] = "replace"
    state: JsonObject


class ApplyEdits(ContractModel):
    """Sequential structural edits under the declared generic edit protocol.

    Attributes:
        kind: Explicit edit effect, distinct from replacement and RFC JSON Patch.
        edits: Ordered set/delete/append/remove_value edits.
    """

    kind: Literal["edits"] = "edits"
    edits: list[JsonObject]


class ProposedObservation(ContractModel):
    """A tool consequence whose caller identity and wire encoding are explicit.

    Attributes:
        call_id: Exact tool invocation receiving this observation.
        reply: Proposed external tool result, independent of evaluator truth.
        encoding: Text preserves exact strings; JSON uses declared JSON serialization.
        is_error: Native tool-error observation, not simulator unavailability.
    """

    call_id: str = Field(min_length=1)
    reply: JsonValue
    encoding: Literal["text", "json"]
    is_error: bool = Field(default=False, strict=True)


class ProposedTransition(ContractModel):
    """A proposal cannot commit itself or carry two competing state effects.

    Attributes:
        observations: Ordered consequences, one per supplied tool invocation.
        effect: Exactly one replacement or edit effect.
        terminal: Whether the tool protocol ends the episode; currently false.
    """

    observations: tuple[ProposedObservation, ...] = Field(min_length=1)
    effect: Annotated[ReplaceState | ApplyEdits, Field(discriminator="kind")]
    terminal: bool = Field(default=False, strict=True)

    def next_state(self, current: JsonObject) -> JsonObject:
        """Compute effects on a detached copy; failed edits never mutate current state."""
        if isinstance(self.effect, ReplaceState):
            return copy.deepcopy(self.effect.state)
        return apply_delta(current, self.effect.edits)


class PreparedPrediction(ContractModel):
    """The actual input and selected-evidence receipt before dispatch.

    Attributes:
        body: Provider-neutral messages, without credentials or evaluator references.
        evidence: Selected IDs and ordered delivered-content identities.
    """

    body: JsonObject
    evidence: JsonObject


class TransitionEngine(Protocol):
    """Prepare and parse proposals without owning persistence or provider I/O."""

    @property
    def identity(self) -> Sha256:
        """Return the frozen engine, knowledge and information-policy identity."""
        ...

    def prepare(
        self, state: JsonObject, history: tuple[JsonObject, ...], actions: tuple[ToolCall, ...]
    ) -> PreparedPrediction:
        """Validate supported actions and frame permitted state and evidence."""
        ...

    def parse(
        self,
        proposal: JsonObject,
        actions: tuple[ToolCall, ...],
        *,
        current_state: JsonObject | None = None,
    ) -> ProposedTransition:
        """Validate protocol and exact ordered call identities; never score against gold."""
        ...


Completion = Callable[[JsonObject, str], JsonObject]
