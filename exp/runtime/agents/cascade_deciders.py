"""The two maintained cheap deciders: System One option probabilities and sampled small drafts.

Both read only the actor-visible request: its messages and tool schemas. Neither receives task
references, evaluator state or future outcomes. Provider admission, deadlines and attempt
accounting stay with the injected decision caller or model client.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.models import ModelClient, ModelFinishReason, ModelRequest
from exp.runtime.agents.cascade import (
    OTHER,
    RESPOND,
    DeciderDefinition,
    NextActionDecision,
    action_key,
)
from exp.runtime.gateway.decisions_contracts import (
    MAX_DECISION_CRITERIA,
    MAX_DECISION_INPUT_BYTES,
    ChoiceQuestion,
    DecisionRequest,
)

# Identity of the rendering and readout below, bound into saved decisions in place of source
# text. Bump it when either changes; exp/tests/code_protocols_test.py records the covered source.
DECIDER_PROTOCOL = "cascade-decider-v1"
QUESTION = "next_action"
_INSTRUCTIONS = (
    "Which single next action should the assistant take in this conversation? Choose the tool "
    "it should call next, respond when it should answer the user without another tool call, or "
    "OTHER when the visible conversation does not determine the next action. "
)
_RESPOND = "Answer the user now without calling another tool."
_OTHER = "The visible conversation does not determine the next action."
_STATE_ALLOWANCE = MAX_DECISION_INPUT_BYTES - 32_768


def _size(value: JsonValue) -> int:
    """Return the strict UTF-8 JSON byte count the decision wire bounds."""
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))


def next_action_request(definition: DeciderDefinition, request: ModelRequest) -> DecisionRequest:
    """Render one actor-visible state as a System One choice over its next actions.

    The oldest messages after the first are omitted, and counted, when the conversation would
    exceed the decision wire's input bound; the first message and the latest turns are kept.

    Args:
        definition: System One decider configuration, supplying optional guidance.
        request: Actor request at the reached state.

    Returns:
        The native decision request with one ``choice`` question.

    Raises:
        ValueError: The options exceed the wire's category limit or a single message is too big.
    """
    criteria: dict[str, JsonValue] = {tool.name: tool.description for tool in request.tools}
    criteria[RESPOND] = _RESPOND
    criteria[OTHER] = _OTHER
    if len(criteria) > MAX_DECISION_CRITERIA:
        raise ValueError("next-action options exceed the decision wire's category limit")
    messages: list[JsonValue] = [message.model_dump(mode="json") for message in request.messages]
    tools: list[JsonValue] = [tool.model_dump(mode="json") for tool in request.tools]
    omitted = 0
    while _size({"conversation": messages, "tools": tools}) > _STATE_ALLOWANCE:
        if len(messages) <= 2:
            raise ValueError("actor state exceeds the decision wire's input bound")
        del messages[1]
        omitted += 1
    state: JsonObject = {
        "conversation": messages,
        "tools": tools,
        "omitted_earlier_messages": omitted,
        "authority": "Actor-visible conversation and tool schemas only; no outcomes supplied.",
    }
    return DecisionRequest(
        state=state,
        questions={
            QUESTION: ChoiceQuestion(
                instructions=_INSTRUCTIONS + definition.guidance, criteria=criteria
            )
        },
    )


def next_action_answer(response: JsonObject, candidates: tuple[str, ...]) -> NextActionDecision:
    """Read the choice and its normalized distribution, refusing any other answer shape.

    Wire probabilities are rounded, so they are renormalized over the offered options.

    Raises:
        ValueError: The answer is missing, malformed or names an option that was not offered.
    """
    answers = response.get("answers")
    answer = answers.get(QUESTION) if isinstance(answers, dict) else None
    options = (*candidates, OTHER)
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise ValueError("System One returned no next-action choice")
    distribution = answer.get("probabilities")
    choice = answer.get("choice")
    if (
        not isinstance(distribution, dict)
        or set(distribution) != set(options)
        or not isinstance(choice, str)
        or choice not in options
    ):
        raise ValueError("System One next-action answer differs from the offered options")
    values: dict[str, float] = {}
    for key in options:
        value = distribution[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError("System One next-action probabilities must lie in [0, 1]")
        values[key] = float(value)
    total = sum(values.values())
    if total <= 0:
        raise ValueError("System One next-action probabilities sum to zero")
    return NextActionDecision(
        selected=choice, probabilities={key: value / total for key, value in values.items()}
    )


class SystemOneDecider:
    """Rank next actions through the native decision wire; never drafts arguments."""

    def __init__(
        self, definition: DeciderDefinition, decide: Callable[[DecisionRequest], JsonObject]
    ) -> None:
        """Bind an admitted decision caller that owns HTTP, deadlines and accounting.

        Raises:
            ValueError: The definition is not a System One decider.
        """
        if definition.kind != "systemone":
            raise ValueError("SystemOneDecider requires a systemone definition")
        self._definition, self._decide = definition, decide

    @property
    def definition(self) -> DeciderDefinition:
        """Return the bound System One configuration."""
        return self._definition

    def decide(self, request: ModelRequest, candidates: tuple[str, ...]) -> NextActionDecision:
        """Ask one choice question and return its normalized distribution."""
        return next_action_answer(
            self._decide(next_action_request(self._definition, request)), candidates
        )


class SampledDraftDecider:
    """Draft complete actions with a small model; confidence is modal-decision agreement."""

    def __init__(self, definition: DeciderDefinition, model: ModelClient) -> None:
        """Bind an admitted small-model client.

        Raises:
            ValueError: The definition is not a sampled-drafts decider.
        """
        if definition.kind != "sampled_drafts":
            raise ValueError("SampledDraftDecider requires a sampled_drafts definition")
        self._definition, self._model = definition, model

    @property
    def definition(self) -> DeciderDefinition:
        """Return the bound drafting configuration."""
        return self._definition

    def decide(self, request: ModelRequest, candidates: tuple[str, ...]) -> NextActionDecision:
        """Draw every registered draft once and keep the first draft of the modal decision.

        Truncated drafts and drafts calling an unoffered tool count as ``OTHER``. Ties go to the
        decision drafted first.

        Raises:
            ValueError: A draft came from a model other than the frozen snapshot.
        """
        draft = request.model_copy(
            update={
                "temperature": self._definition.temperature,
                "maximum_output_tokens": self._definition.maximum_output_tokens,
            }
        )
        responses = [self._model.complete(draft) for _ in range(self._definition.samples)]
        if any(response.model != self._definition.model for response in responses):
            raise ValueError("draft response identity differs from the frozen decider")
        keys = [
            action_key(response.output)
            if response.finish_reason == ModelFinishReason.COMPLETED
            and action_key(response.output) in candidates
            else OTHER
            for response in responses
        ]
        counts = Counter(keys)
        selected = max(keys, key=lambda key: (counts[key], -keys.index(key)))
        chosen = None if selected == OTHER else responses[keys.index(selected)]
        return NextActionDecision(
            selected=selected,
            probabilities={key: counts[key] / len(keys) for key in (*candidates, OTHER)},
            action=None if chosen is None else chosen.output,
            response=chosen,
        )
