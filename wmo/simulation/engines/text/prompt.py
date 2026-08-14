"""Versioned text-only world-model prompt framing and strict transition parsing."""

from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import ValidationError

from wmo.common.core.artifacts import ContractModel, JsonObject, sha256_json
from wmo.common.models import AssistantAction, ModelMessage, ModelRequest
from wmo.common.tasks import TaskCase
from wmo.simulation.retrieval import RAGMatch
from wmo.simulation.retrieval.contracts import RAG_KEY_SCHEMA_VERSION

WORLD_MODEL_TEXT_PROMPT_VERSION = "text-world-model-v1"
WORLD_MODEL_TEXT_PROMPT_ID = "world-model-text-v1"
WORLD_MODEL_TEXT_GROUNDING_SCHEMA_VERSION = "fit-rag-examples-v1"
WORLD_MODEL_TEXT_SYSTEM_PROMPT = """Protocol version: text-world-model-v1.
You simulate the next visible user or environment message in a text-only
customer-agent scenario. You receive the task, safe initial context, the candidate's visible
conversation, and its latest visible assistant response. Do not execute, invent, or describe tools.
Do not infer or reveal candidate hidden reasoning. You may reason internally if your provider
supports it, but return only this JSON object:
{"message":"the next visible user or environment message","terminal":false}
Set terminal to true only when the scenario has reached a visible terminal state. The message may be
empty only for a terminal state. Do not include markdown fences or other keys."""


class TextWorldModelTransition(ContractModel):
    """One parsed visible text turn emitted by the versioned world-model prompt."""

    message: str
    terminal: bool = False

    @property
    def visible_message(self) -> ModelMessage:
        """Return the one user-visible transcript message represented by this transition."""
        return ModelMessage(role="user", content=self.message)


class TextWorldModelProtocolError(ValueError):
    """The world model did not return the pinned text-transition JSON contract."""


def build_world_model_request(
    task: TaskCase,
    *,
    visible_messages: Sequence[ModelMessage],
    candidate_response: AssistantAction,
    grounded_examples: Sequence[RAGMatch],
    maximum_output_tokens: int,
) -> ModelRequest:
    """Build one tool-free request from visible candidate evidence only.

    Args:
        task: Current canonical representative task.
        visible_messages: Candidate-visible request messages, including prior simulated turns.
        candidate_response: Candidate's visible text response. Tool calls are rejected by the
            caller before this function is reached.
        grounded_examples: Nearest immutable real transitions after current-lineage exclusion.
        maximum_output_tokens: Explicit non-truncating provider output budget.

    Returns:
        A text-only provider request with the pinned prompt and no candidate hidden state.

    Raises:
        TextWorldModelProtocolError: The candidate response lacks visible text or includes tools.
    """
    if candidate_response.content is None or candidate_response.tool_calls:
        raise TextWorldModelProtocolError(
            "text world-model framing requires one visible candidate text response without tools"
        )
    evidence: JsonObject = {
        "task": {
            "task_id": task.task_id,
            "instruction": task.instruction,
            "initial_context": task.initial_context,
        },
        "visible_conversation": [
            message.model_dump(mode="json", exclude_none=True) for message in visible_messages
        ],
        "candidate_response": candidate_response.content,
        "grounding_schema_version": WORLD_MODEL_TEXT_GROUNDING_SCHEMA_VERSION,
        "grounded_examples": [
            {
                "transition_id": match.transition.transition_id,
                "task": match.transition.task,
                "initial_context": match.transition.initial_context,
                "action": match.transition.action.model_dump(mode="json", exclude_none=True),
                "observation": match.transition.observation.model_dump(mode="json"),
            }
            for match in grounded_examples
        ],
    }
    return ModelRequest(
        messages=(
            ModelMessage(role="system", content=WORLD_MODEL_TEXT_SYSTEM_PROMPT),
            ModelMessage(
                role="user",
                content=json.dumps(
                    evidence,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            ),
        ),
        tool_choice="none",
        maximum_output_tokens=maximum_output_tokens,
    )


def parse_world_model_transition(output: AssistantAction) -> TextWorldModelTransition:
    """Parse a strict visible transition and reject tools, prose, or hidden-output stand-ins.

    Args:
        output: Provider-normalized visible assistant action from the world model.

    Returns:
        Parsed next user or environment message plus terminal state.

    Raises:
        TextWorldModelProtocolError: The output is not the pinned JSON-only text protocol.
    """
    if output.tool_calls or output.content is None:
        raise TextWorldModelProtocolError(
            "text world models must return one JSON transition without tool calls"
        )
    try:
        value = json.loads(output.content)
    except json.JSONDecodeError as exc:
        raise TextWorldModelProtocolError(
            "text world model must return the pinned JSON transition without surrounding prose"
        ) from exc
    try:
        transition = TextWorldModelTransition.model_validate(value)
    except ValidationError as exc:
        raise TextWorldModelProtocolError(
            "text world model transition must contain only message and terminal fields"
        ) from exc
    if not transition.message and not transition.terminal:
        raise TextWorldModelProtocolError(
            "a nonterminal text world-model transition needs a visible message"
        )
    return transition


def text_prompt_sha256() -> str:
    """Return the digest pinned in every text-world-model simulator snapshot.

    Returns:
        Canonical digest covering the prompt and grounding schema identities.
    """
    return sha256_json(
        {
            "prompt_id": WORLD_MODEL_TEXT_PROMPT_ID,
            "prompt_version": WORLD_MODEL_TEXT_PROMPT_VERSION,
            "system_prompt": WORLD_MODEL_TEXT_SYSTEM_PROMPT,
            "grounding_schema_version": WORLD_MODEL_TEXT_GROUNDING_SCHEMA_VERSION,
            "rag_key_schema_version": RAG_KEY_SCHEMA_VERSION,
        }
    )
