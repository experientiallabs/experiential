"""Reconstructed text proposals over permitted evidence and candidate-owned state."""

import copy
import json
from importlib.resources import files
from typing import Literal

from jsonschema import Draft202012Validator
from pydantic import Field, StrictInt, StrictStr

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, Sha256, sha256_json
from exp.common.models import ToolCall
from exp.simulation.engines.transition import (
    ApplyEdits,
    PreparedPrediction,
    ProposedObservation,
    ProposedTransition,
)
from exp.simulation.reconstruction.models import ReconstructionBundle, check_schema
from exp.simulation.retrieval.selection import (
    AllOriginals,
    EvidencePolicy,
    SelectionQuery,
    selected_evidence,
)


class _JsonObservation(ProposedObservation):
    """The reconstructed tool protocol serializes every result as JSON."""

    encoding: Literal["json"] = "json"


class _StateReference(ContractModel):
    """A model-selected pre-action value with an optional literal reply envelope.

    Attributes:
        path: Literal object keys or array indices, without wildcard or prefix matching.
        wrapper: Literal reply envelope; omitted returns the selected value directly.
        view: Select the prestate or the result of this proposal and its own edits.
        field: Envelope field receiving the value; required only with a wrapper.
    """

    path: tuple[StrictStr | StrictInt, ...]
    wrapper: JsonObject | None = None
    field: str | None = None
    view: Literal["before", "after"] = "before"

    def resolve(self, state: JsonObject) -> JsonValue:
        """Select only existing candidate values; never consult source or native state."""
        value: JsonValue = state
        for part in self.path:
            if isinstance(value, dict) and isinstance(part, str) and part in value:
                value = value[part]
            elif isinstance(value, list) and type(part) is int and 0 <= part < len(value):
                value = value[part]
            else:
                raise ValueError("reply state reference does not resolve")
        if self.wrapper is None:
            if self.field is not None:
                raise ValueError("reply field requires a wrapper")
            return copy.deepcopy(value)
        if self.field is None or self.field in self.wrapper:
            raise ValueError("reply wrapper requires an unused destination field")
        return {**copy.deepcopy(self.wrapper), self.field: copy.deepcopy(value)}


class _ReferencedObservation(ContractModel):
    """A reply projection disjoint from a model-authored literal reply.

    Attributes:
        call_id: Exact current caller identity.
        reply_from_state: Explicit candidate-state reference; no literal reply allowed.
        encoding: Referenced values use JSON serialization.
        is_error: Whether this is a supported native error consequence.
    """

    call_id: str = Field(min_length=1)
    reply_from_state: _StateReference
    encoding: Literal["json"] = "json"
    is_error: bool = Field(default=False, strict=True)


class _ReconstructedProposal(ContractModel):
    """The exact advertised and parsed compact-edit wire contract."""

    observations: tuple[_JsonObservation | _ReferencedObservation, ...] = Field(
        min_length=1, max_length=1
    )
    effect: ApplyEdits
    terminal: Literal[False] = False


class InformationChannels(ContractModel):
    """Runtime removals leave construction exposure and inferred interface explicit.

    Attributes:
        rules: Deliver generated behavioral rules and claims.
        originals: Deliver selected original observations at runtime.
        history: Deliver this instance's public chronology, including to selection.
    """

    rules: bool = Field(default=True, strict=True)
    originals: bool = Field(default=True, strict=True)
    history: bool = Field(default=True, strict=True)


class ReconstructedTextEngine:
    """Prepare one single-tool text proposal; session code owns all state commits."""

    def __init__(
        self,
        bundle: ReconstructionBundle,
        *,
        channels: InformationChannels | None = None,
        selection: EvidencePolicy | None = None,
        initial_coverage: JsonObject | None = None,
        observation_schema: JsonObject | None = None,
    ) -> None:
        """Freeze the interface, knowledge, information treatment and evidence policy."""
        self.bundle = ReconstructionBundle.model_validate_json(bundle.model_dump_json())
        self.channels = channels or InformationChannels()
        self.selection = selection or AllOriginals()
        self.initial_coverage = copy.deepcopy(initial_coverage or {})
        self.observation_schema = copy.deepcopy(observation_schema)
        if self.observation_schema is not None:
            check_schema(self.observation_schema)
        self._prompt = (
            files("exp.simulation.reconstruction")
            .joinpath("prompts/predict.txt")
            .read_text(encoding="utf-8")
        )

    @property
    def identity(self) -> Sha256:
        """Bind the actual behavioral artifact, prompt and information policy."""
        return sha256_json(
            {
                "bundle": self.bundle.digest,
                "channels": self.channels.model_dump(mode="json"),
                "selection": self.selection.identity,
                "initial_coverage": self.initial_coverage,
                "observation_schema": self.observation_schema,
                "reply_protocol": "literal-or-candidate-state-v3",
                "prompt": self._prompt,
            }
        )

    def prepare(
        self, state: JsonObject, history: tuple[JsonObject, ...], actions: tuple[ToolCall, ...]
    ) -> PreparedPrediction:
        """Reject unsupported batches before retrieval, budget admission or dispatch."""
        if len(actions) != 1:
            raise ValueError("reconstructed text supports exactly one tool call per operation")
        action = actions[0]
        contract = next(
            (tool for tool in self.bundle.interface.tools if tool.name == action.name), None
        )
        if contract is None:
            raise ValueError("tool is absent from the proposed interface")
        Draft202012Validator(contract.parameters).validate(action.arguments)
        permitted_history = history if self.channels.history else ()
        catalog = self.bundle.sources.source_map() if self.channels.originals else {}
        sources, receipt = selected_evidence(
            self.selection,
            SelectionQuery(action=action, state=state, history=permitted_history),
            catalog,
        )
        knowledge: JsonObject = {"interface": self.bundle.interface.model_dump(mode="json")}
        if self.channels.rules:
            knowledge.update(
                text_rules=self.bundle.text_rules,
                claims=[claim.model_dump(mode="json") for claim in self.bundle.claims],
            )
        output_schema = _ReconstructedProposal.model_json_schema()
        if self.observation_schema is not None:
            output_schema["$defs"]["_JsonObservation"]["properties"]["reply"] = (
                self.observation_schema
            )
        for name in ("_JsonObservation", "_ReferencedObservation"):
            output_schema["$defs"][name]["properties"]["call_id"] = {
                "type": "string",
                "const": action.call_id,
            }
        payload: JsonObject = {
            "knowledge": knowledge,
            "selected_sources": sources,
            "state": copy.deepcopy(state),
            "initial_state_coverage": copy.deepcopy(self.initial_coverage),
            "history": list(permitted_history),
            "action": action.model_dump(mode="json"),
            "output_schema": output_schema,
        }
        body: JsonObject = {
            "response_schema": output_schema,
            "messages": [
                {"role": "system", "content": self._prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, allow_nan=False),
                },
            ],
        }
        receipt.update(
            channels=self.channels.model_dump(mode="json"), request_sha256=sha256_json(body)
        )
        return PreparedPrediction(body=body, evidence=receipt)

    def parse(
        self,
        proposal: JsonObject,
        actions: tuple[ToolCall, ...],
        *,
        current_state: JsonObject | None = None,
    ) -> ProposedTransition:
        """Validate ordered identities and declared reply types without native answers."""
        transition = _ReconstructedProposal.model_validate(proposal)
        if tuple(row.call_id for row in transition.observations) != tuple(
            row.call_id for row in actions
        ):
            raise ValueError("proposal observation identity/order differs from the action")
        observations = []
        after: JsonObject | None = None
        for observation, action in zip(transition.observations, actions, strict=True):
            if isinstance(observation, _ReferencedObservation):
                if current_state is None:
                    raise ValueError("reply projection requires the actual candidate prestate")
                source_state = current_state
                if observation.reply_from_state.view == "after":
                    if after is None:
                        from exp.simulation.engines.text.generic_delta import apply_delta

                        after = apply_delta(current_state, transition.effect.edits)
                    source_state = after
                resolved = ProposedObservation(
                    call_id=observation.call_id,
                    encoding="json",
                    is_error=observation.is_error,
                    reply=observation.reply_from_state.resolve(source_state),
                )
            else:
                resolved = observation
            tool = next(tool for tool in self.bundle.interface.tools if tool.name == action.name)
            if tool.reply_schema is not None:
                Draft202012Validator(tool.reply_schema).validate(resolved.reply)
            if self.observation_schema is not None:
                Draft202012Validator(self.observation_schema).validate(resolved.reply)
            observations.append(resolved)
        return ProposedTransition(
            observations=tuple(observations), effect=transition.effect, terminal=transition.terminal
        )
