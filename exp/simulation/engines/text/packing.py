"""Fit whole optional grounding examples around an unchanged required world prompt."""

from __future__ import annotations

import json

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models import ModelRequest
from exp.simulation.engines.text.prompt import WORLD_MODEL_TEXT_GROUNDING_SCHEMA_VERSION
from exp.simulation.engines.text.tokens import TokenCounter


def pack_world_model_request(
    request: ModelRequest, *, maximum_input_tokens: int, token_counter: TokenCounter
) -> tuple[ModelRequest, tuple[str, ...]]:
    """Keep the highest-priority whole examples that fit the exact input allowance.

    Only ``grounded_examples`` and their referenced contexts in the second message may change.
    Every selected example retains its entire context, action and observation. Required task,
    tools, history, state, system instructions, correction messages and output controls stay
    intact. An oversized required prompt is returned without examples for normal admission
    to reject; packing never truncates it or increases the input allowance.

    Args:
        request: Canonical world request, optionally followed by protocol-correction messages.
        maximum_input_tokens: Available context after the unchanged output reservation.
        token_counter: Full-request counter including provider framing.

    Returns:
        Packed request and the exact ordered transition IDs included in it.

    Raises:
        ValueError: The request does not contain canonical grounding evidence.
    """
    if len(request.messages) < 2 or request.messages[1].content is None:
        raise ValueError("world-model packing requires the canonical evidence message")
    evidence: JsonValue = json.loads(request.messages[1].content)
    if not isinstance(evidence, dict) or not isinstance(evidence.get("grounded_examples"), list):
        raise ValueError("world-model packing requires a grounded_examples list")
    examples = evidence["grounded_examples"]
    assert isinstance(examples, list)
    if evidence.get("grounding_schema_version") != WORLD_MODEL_TEXT_GROUNDING_SCHEMA_VERSION:
        raise ValueError("world-model grounding schema differs; prepare a new world-model build")
    contexts = evidence.get("grounding_contexts")
    if not isinstance(contexts, dict):
        raise ValueError("world-model packing requires a grounding_contexts table")
    _require_context_closure(examples, contexts)
    identifiers = tuple(_transition_id(example) for example in examples)
    if 0 <= token_counter.count(request) <= maximum_input_tokens:
        return request, identifiers

    def framed(selected: list[JsonValue]) -> ModelRequest:
        """Replace only optional examples while retaining every other request field."""
        references = {_context_ref(example) for example in selected}
        payload: JsonObject = {
            **evidence,
            "grounded_examples": selected,
            "grounding_contexts": {
                key: value for key, value in contexts.items() if key in references
            },
        }
        message = request.messages[1].model_copy(
            update={
                "content": json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                )
            }
        )
        return request.model_copy(
            update={"messages": (request.messages[0], message, *request.messages[2:])}
        )

    selected: list[JsonValue] = []
    included: list[str] = []
    packed = framed(selected)
    if not 0 <= token_counter.count(packed) <= maximum_input_tokens:
        return packed, ()
    for identifier, example in zip(identifiers, examples, strict=True):
        candidate = framed([*selected, example])
        if 0 <= token_counter.count(candidate) <= maximum_input_tokens:
            selected.append(example)
            included.append(identifier)
            packed = candidate
    return packed, tuple(included)


def _transition_id(example: JsonValue) -> str:
    """Require an explicit source identity for each optional example."""
    if not isinstance(example, dict) or not isinstance(example.get("transition_id"), str):
        raise ValueError("grounding examples require an exact transition_id")
    identifier = example["transition_id"]
    assert isinstance(identifier, str)
    return identifier


def _context_ref(example: JsonValue) -> str:
    """Require one exact context reference for a complete optional example."""
    if not isinstance(example, dict) or not isinstance(example.get("context_ref"), str):
        raise ValueError("grounding examples require an exact context_ref")
    reference = example["context_ref"]
    assert isinstance(reference, str)
    return reference


def _require_context_closure(examples: list[JsonValue], contexts: JsonObject) -> None:
    """Reject missing, unused or altered contexts before considering request capacity.

    Args:
        examples: Ordered optional observations with content-addressed context references.
        contexts: Complete context table, containing exactly the referenced entries.

    Raises:
        ValueError: A reference, context schema or content digest differs.
    """
    if {_context_ref(example) for example in examples} != set(contexts):
        raise ValueError("grounding context table must contain exactly the referenced contexts")
    for reference, context in contexts.items():
        if (
            not isinstance(context, dict)
            or set(context) != {"task", "initial_context"}
            or not isinstance(context["task"], str)
            or not isinstance(context["initial_context"], dict)
            or reference != "context-" + sha256_json(context)
        ):
            raise ValueError("grounding context differs from its exact content reference")
