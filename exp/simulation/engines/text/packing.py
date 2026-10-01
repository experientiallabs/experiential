"""Fit whole optional grounding examples around an unchanged required world prompt."""

from __future__ import annotations

import json

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelRequest
from exp.simulation.engines.text.tokens import TokenCounter


def pack_world_model_request(
    request: ModelRequest, *, maximum_input_tokens: int, token_counter: TokenCounter
) -> tuple[ModelRequest, tuple[str, ...]]:
    """Keep the highest-priority whole examples that fit the exact input allowance.

    Only ``grounded_examples`` in the canonical second message may change. Required task,
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
    identifiers = tuple(_transition_id(example) for example in examples)
    if 0 <= token_counter.count(request) <= maximum_input_tokens:
        return request, identifiers

    def framed(selected: list[JsonValue]) -> ModelRequest:
        """Replace only optional examples while retaining every other request field."""
        payload: JsonObject = {**evidence, "grounded_examples": selected}
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
