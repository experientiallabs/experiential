"""Opt-in decision cascade: a cheap next-action decider with the main model as the fallback.

At each actor model call the cascade asks a cheap decider which next action applies: one visible
tool, named by the tool, or ``respond`` (a final answer without another tool call). When the
decider's probability for its choice meets the policy threshold, and the policy's hard
restrictions allow that action, the decider is responsible for the decision. Otherwise the main
model stays responsible and receives exactly the request it would have received without the
cascade.

An accepted decision is completed in one of two ways. A decider that drafts complete actions (a
small model) supplies the action, so the main model call is avoided. A decider that only ranks
the options (System One probabilities) delegates the choice alone: the main model is called once
with the tool choice forced to the selected action and fills in its arguments. Delegated
decisions and avoided calls are therefore different counts, and every record states both.

The cascade is off unless a caller wraps a model client with a ``CascadePolicy``. The policy binds
the exact decider definition and a threshold fitted elsewhere on development trajectories; this
module never chooses a threshold and has no default one.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection
from typing import Literal, Protocol, Self

from jsonschema import Draft202012Validator
from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, Sha256, sha256_json
from exp.common.core.json_schema import expand_local_references
from exp.common.models import (
    AssistantAction,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    ToolChoice,
)
from exp.common.models.unknown_outcome import DispatchOutcomeUnknown

CASCADE_VERSION = "decision-cascade-v3"
RESPOND = "respond"
OTHER = "OTHER"


def action_key(action: AssistantAction) -> str:
    """Return the next-action decision an output makes: its first tool's name, or ``respond``."""
    return action.tool_calls[0].name if action.tool_calls else RESPOND


def action_projection(action: AssistantAction) -> Sha256:
    """Identify a complete action by ordered tool names and arguments, or its final text.

    Generated call IDs are ignored, so two drafts that call the same tools with the same arguments
    are the same action.
    """
    return sha256_json(
        {
            "content": action.content if not action.tool_calls else None,
            "calls": [
                {"name": call.name, "arguments": call.arguments} for call in action.tool_calls
            ],
        }
    )


def decision_candidates(request: ModelRequest) -> tuple[str, ...]:
    """Return the ordered next-action options for one request: its tools, then ``respond``.

    Raises:
        ValueError: A tool uses a reserved option name, so its decisions cannot be told apart.
    """
    names = tuple(tool.name for tool in request.tools)
    if RESPOND in names or OTHER in names:
        raise ValueError(
            f"a tool named {RESPOND!r} or {OTHER!r} collides with a reserved cascade option; "
            "rename the tool before enabling the cascade"
        )
    return (*names, RESPOND)


def delegates(
    selected: str, confidence: float, *, threshold: float, allowed: Collection[str]
) -> bool:
    """Apply the one cascade acceptance rule shared by deployment, fitting and measurement.

    Args:
        selected: Decider choice; ``OTHER`` abstains.
        confidence: Decider probability for that choice.
        threshold: Registered acceptance threshold.
        allowed: Options the hard policy permits the decider to take; restrictions apply before
            any confidence comparison, so a high probability never overrides them.

    Returns:
        Whether the decider is responsible for this decision.
    """
    return selected != OTHER and selected in allowed and confidence >= threshold


class DeciderDefinition(ContractModel):
    """Exact cheap-decider configuration bound into a cascade policy.

    Attributes:
        kind: ``systemone`` ranks the options through the native decision wire and never drafts
            arguments. ``sampled_drafts`` drafts complete actions with a small chat model; its
            confidence is the share of drafts making the modal decision, a readout of decision
            agreement rather than of argument correctness.
        alias: Catalog alias of the decision model.
        model: Exact snapshot, checked before use and on every drafted response.
        samples: Drafts per decision, at least two (``sampled_drafts`` only; one for ``systemone``).
        temperature: Draft sampling temperature (``sampled_drafts`` only).
        maximum_output_tokens: Per-draft output allowance (``sampled_drafts`` only).
        guidance: Additional System One instruction (``systemone`` only), empty by default.
    """

    kind: Literal["systemone", "sampled_drafts"]
    alias: str = Field(min_length=1)
    model: ModelSnapshot
    samples: int = Field(default=1, ge=1, le=16, strict=True)
    temperature: float = Field(default=1.0, ge=0, le=2, allow_inf_nan=False)
    maximum_output_tokens: int = Field(default=1024, gt=0, le=65536, strict=True)
    guidance: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        """Refuse settings the selected kind would silently ignore."""
        if self.kind == "systemone" and (
            self.samples != 1 or self.temperature != 1.0 or self.maximum_output_tokens != 1024
        ):
            raise ValueError("a systemone decider takes no draft samples, temperature or tokens")
        if self.kind == "sampled_drafts" and (self.samples < 2 or self.guidance):
            raise ValueError(
                "a sampled_drafts decider needs at least two samples and takes no guidance"
            )
        return self


class CascadePolicy(ContractModel):
    """An explicit, versioned cascade runtime option; absent means no cascade.

    Attributes:
        version: Cascade protocol; other versions fail closed.
        decider: Exact decider every decision must come from.
        threshold: Acceptance probability, fitted on development trajectories by a registered
            objective. There is no default.
        never_delegate: Options the decider may never take, sorted; the main model stays
            responsible for them whatever the confidence.
    """

    version: Literal["decision-cascade-v3"] = CASCADE_VERSION
    decider: DeciderDefinition
    threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    never_delegate: tuple[str, ...] = ()

    @model_validator(mode="after")
    def sorted_restrictions(self) -> Self:
        """Keep restrictions canonical so equal policies share one identity."""
        if self.never_delegate != tuple(sorted(set(self.never_delegate))) or (
            OTHER in self.never_delegate
        ):
            raise ValueError("never_delegate must be sorted, unique and exclude OTHER")
        return self

    @property
    def identity(self) -> Sha256:
        """Bind the version, decider, threshold and hard restrictions."""
        return sha256_json(self)

    def allowed(self, candidates: tuple[str, ...]) -> tuple[str, ...]:
        """Return the offered options the decider may take under the hard restrictions."""
        return tuple(name for name in candidates if name not in self.never_delegate)


class NextActionDecision(ContractModel):
    """One decider answer over the offered options plus ``OTHER``.

    Attributes:
        selected: Chosen option, or ``OTHER`` to abstain.
        probabilities: Distribution over every offered option and ``OTHER``, summing to one.
        action: Complete drafted action making the selected decision, when the decider drafts.
        response: The drafter's own response for that action, keeping its identity and usage.
    """

    selected: str = Field(min_length=1)
    probabilities: dict[str, float]
    action: AssistantAction | None = None
    response: ModelResponse | None = None

    @model_validator(mode="after")
    def valid(self) -> Self:
        """Reject invalid distributions and drafts that do not make the selected decision."""
        values = self.probabilities.values()
        if (
            OTHER not in self.probabilities
            or self.selected not in self.probabilities
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values)
            or not math.isclose(sum(values), 1, abs_tol=1e-6)
        ):
            raise ValueError("decision probabilities must be finite, include OTHER and sum to one")
        if (self.action is None) != (self.response is None):
            raise ValueError("a drafted action needs the response that produced it")
        if self.action is not None and (
            action_key(self.action) != self.selected
            or self.response is None
            or self.response.output != self.action
        ):
            raise ValueError("drafted action differs from the selected decision or its response")
        return self

    @property
    def confidence(self) -> float:
        """Return the decider's probability for its own choice."""
        return self.probabilities[self.selected]


class NextActionDecider(Protocol):
    """A cheap model answering which next action applies at a reached actor state."""

    @property
    def definition(self) -> DeciderDefinition:
        """Return the exact configuration this decider implements."""
        ...

    def decide(self, request: ModelRequest, candidates: tuple[str, ...]) -> NextActionDecision:
        """Answer over ``candidates`` plus ``OTHER`` for the actor-visible request.

        Args:
            request: Exact actor request at the reached state, with its visible tools.
            candidates: Ordered options from ``decision_candidates``.

        Returns:
            The decision. A ``ValueError`` about the answer's content leaves the main model
            responsible; every other failure, and ``DispatchOutcomeUnknown``, stops the step.
        """
        ...


class CascadeDecisionRecord(ContractModel):
    """Which model decided one actor step, and with what confidence.

    Attributes:
        step: Zero-based model call index within one cascade client.
        policy_sha256: Exact deployed policy.
        candidates: Ordered next-action options offered at this step.
        selected: Decider choice, or None when the decider was not asked or failed.
        confidence: Decider probability for its choice, or None without a choice.
        threshold: Policy threshold in force.
        decided_by: ``decider`` when the decision was delegated, otherwise ``main``.
        decision_model: Snapshot of the model responsible for the decision.
        main_called: Whether the main model was called for this step.
        followed: For a forced main call, whether its output made the selected decision.
        note: Why the main model stayed responsible, such as ``below_threshold``, ``abstained``,
            ``restricted``, ``invalid_draft``, ``caller_tool_choice`` or
            ``decider_failure:<type>``; None for a delegated decision.
    """

    step: int = Field(ge=0)
    policy_sha256: Sha256
    candidates: tuple[str, ...]
    selected: str | None = None
    confidence: float | None = None
    threshold: float
    decided_by: Literal["decider", "main"]
    decision_model: ModelSnapshot
    main_called: bool
    followed: bool | None = None
    note: str | None = None


class CascadeModelClient:
    """Wrap a main model client with a cheap decider under one explicit policy."""

    def __init__(
        self,
        *,
        main: ModelClient,
        decider: NextActionDecider,
        policy: CascadePolicy,
        record: Callable[[CascadeDecisionRecord], None] | None = None,
    ) -> None:
        """Bind the main model, the policy's exact decider and an optional record observer.

        Args:
            main: Model responsible for every decision the decider does not take.
            decider: Cheap decider; its definition must equal the policy's.
            policy: Frozen cascade option, including the fitted threshold.
            record: Trusted observer called with each decision record before it is returned.

        Raises:
            ValueError: The decider differs from the policy's frozen decider.
        """
        if decider.definition != policy.decider:
            raise ValueError("cascade decider differs from the policy's frozen decider")
        self._main, self._decider, self._policy, self._record = main, decider, policy, record
        self.records: list[CascadeDecisionRecord] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Let the decider decide when the policy accepts its answer, else call the main model.

        Args:
            request: Actor request at the reached state.

        Only a decider content error, such as a malformed distribution or an off-option answer,
        leaves the main model responsible. Transport failures, deadlines, dispatch caps and
        unknown dispatch outcomes propagate: the decider's step stays undecided rather than being
        redrawn from the main model.

        Returns:
            The drafted response, the forced main response, or the unchanged main response.

        Raises:
            DispatchOutcomeUnknown: The decider's dispatch may have been answered and billed.
        """
        candidates: tuple[str, ...] = ()
        if request.tool_choice not in (None, "auto"):
            return self._main_decides(request, candidates, None, "caller_tool_choice")
        try:
            candidates = decision_candidates(request)
            decision = self._decider.decide(request, candidates)
            if set(decision.probabilities) != {*candidates, OTHER}:
                raise ValueError("decision distribution differs from the offered options")
        except DispatchOutcomeUnknown:
            raise
        except ValueError as error:
            return self._main_decides(
                request, candidates, None, f"decider_failure:{type(error).__name__}"
            )
        allowed = self._policy.allowed(candidates)
        if not delegates(
            decision.selected,
            decision.confidence,
            threshold=self._policy.threshold,
            allowed=allowed,
        ):
            note = (
                "abstained"
                if decision.selected == OTHER
                else "restricted"
                if decision.selected not in allowed
                else "below_threshold"
            )
            return self._main_decides(request, candidates, decision, note)
        if decision.action is not None:
            if decision.response is None or not admitted_draft(
                request, decision.action, self._policy.never_delegate
            ):
                return self._main_decides(request, candidates, decision, "invalid_draft")
            return self._finish(
                decision.response,
                candidates,
                decision,
                decided_by="decider",
                decision_model=self._policy.decider.model,
                main_called=False,
            )
        forced = request.model_copy(
            update={
                "tool_choice": "none"
                if decision.selected == RESPOND
                else ToolChoice(name=decision.selected)
            }
        )
        response = self._main.complete(forced)
        return self._finish(
            response,
            candidates,
            decision,
            decided_by="decider",
            decision_model=self._policy.decider.model,
            main_called=True,
            followed=action_key(response.output) == decision.selected,
        )

    def _main_decides(
        self,
        request: ModelRequest,
        candidates: tuple[str, ...],
        decision: NextActionDecision | None,
        note: str,
    ) -> ModelResponse:
        """Call the main model with the original request and record why it stayed responsible."""
        response = self._main.complete(request)
        return self._finish(
            response,
            candidates,
            decision,
            decided_by="main",
            decision_model=response.model,
            main_called=True,
            note=note,
        )

    def _finish(
        self,
        response: ModelResponse,
        candidates: tuple[str, ...],
        decision: NextActionDecision | None,
        *,
        decided_by: Literal["decider", "main"],
        decision_model: ModelSnapshot,
        main_called: bool,
        followed: bool | None = None,
        note: str | None = None,
    ) -> ModelResponse:
        """Record the step before its response can affect the episode."""
        row = CascadeDecisionRecord(
            step=len(self.records),
            policy_sha256=self._policy.identity,
            candidates=candidates,
            selected=None if decision is None else decision.selected,
            confidence=None if decision is None else decision.confidence,
            threshold=self._policy.threshold,
            decided_by=decided_by,
            decision_model=decision_model,
            main_called=main_called,
            followed=followed,
            note=note,
        )
        self.records.append(row)
        if self._record is not None:
            self._record(row)
        return response


def admitted_draft(
    request: ModelRequest, action: AssistantAction, never_delegate: Collection[str]
) -> bool:
    """Accept a draft only when every call names a visible, delegable tool with valid arguments.

    The acceptance rule sees only the draft's first call, so the hard restrictions are applied
    again to every later call: a draft never executes a tool the decider may not take.
    """
    tools = {tool.name: tool for tool in request.tools}
    return all(
        call.name in tools
        and call.name not in never_delegate
        and Draft202012Validator(expand_local_references(tools[call.name].input_schema)).is_valid(
            call.arguments
        )
        for call in action.tool_calls
    )
