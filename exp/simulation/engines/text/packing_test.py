"""Required world evidence and output controls survive context-aware example selection."""

import json

from exp.common.models import AssistantAction, ModelMessage
from exp.common.tasks import ToolSchema
from exp.simulation.engines.text.packing import pack_world_model_request
from exp.simulation.engines.text.prompt import build_world_model_request, retry_world_model_request
from exp.simulation.engines.text.recording_test import _grounding_example, _task
from exp.simulation.engines.text.tokens import Utf8UpperBoundTokenCounter


def test_packing_preserves_required_state_tools_messages_and_generation_controls() -> None:
    """Only complete optional examples may change, including with Unicode and correction turns."""
    task = _task().model_copy(
        update={
            "tools": (
                ToolSchema(
                    name="lookup_record",
                    description="Required tool definition",
                    input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
                ),
            )
        }
    )
    action = AssistantAction(content="Required candidate action")
    required = build_world_model_request(
        task,
        visible_messages=(ModelMessage(role="developer", content="Required developer rule"),),
        candidate_response=action,
        grounded_examples=(),
        maximum_output_tokens=393_216,
        state={"required_state": {"records": ["é", "保留"]}},
    ).model_copy(update={"reasoning_effort": "max", "json_object_output": True})
    required = retry_world_model_request(required, action, "invalid result")
    evidence = json.loads(required.messages[1].content or "")
    optional = build_world_model_request(
        task,
        visible_messages=(ModelMessage(role="developer", content="Required developer rule"),),
        candidate_response=action,
        grounded_examples=(_grounding_example("oversized", 10_000), _grounding_example("fits", 10)),
        maximum_output_tokens=393_216,
        state={"required_state": {"records": ["é", "保留"]}},
    ).model_copy(update={"reasoning_effort": "max", "json_object_output": True})
    optional = retry_world_model_request(optional, action, "invalid result")
    counter = Utf8UpperBoundTokenCounter()
    ceiling = counter.count(required) + 1_000

    packed, identifiers = pack_world_model_request(
        optional, maximum_input_tokens=ceiling, token_counter=counter
    )

    assert identifiers == ("fits",)
    assert counter.count(packed) <= ceiling
    packed_evidence = json.loads(packed.messages[1].content or "")
    assert packed_evidence.pop("grounded_examples")[0]["transition_id"] == "fits"
    evidence.pop("grounded_examples")
    assert packed_evidence == evidence
    assert packed.messages[0] == required.messages[0]
    assert packed.messages[2:] == required.messages[2:]
    assert packed.model_dump(exclude={"messages"}) == required.model_dump(exclude={"messages"})


def test_required_overflow_is_retained_without_truncation() -> None:
    """An impossible required prompt remains oversized for ordinary context admission to reject."""
    request = build_world_model_request(
        _task(),
        visible_messages=(ModelMessage(role="user", content="Must remain " * 2_000),),
        candidate_response=AssistantAction(content="Required action"),
        grounded_examples=(_grounding_example("optional", 1),),
        maximum_output_tokens=16_000,
    )
    packed, identifiers = pack_world_model_request(
        request, maximum_input_tokens=2_000, token_counter=Utf8UpperBoundTokenCounter()
    )
    before = json.loads(request.messages[1].content or "")
    after = json.loads(packed.messages[1].content or "")
    before["grounded_examples"] = []
    assert after == before
    assert identifiers == ()
    assert packed.maximum_output_tokens == request.maximum_output_tokens
    assert Utf8UpperBoundTokenCounter().count(packed) > 2_000


def test_fitting_request_replays_byte_for_byte_without_repacking() -> None:
    """An already-admitted prompt keeps its original serialization and complete grounding."""
    request = build_world_model_request(
        _task(),
        visible_messages=(),
        candidate_response=AssistantAction(content="Answer"),
        grounded_examples=(_grounding_example("fits", 10),),
        maximum_output_tokens=16_000,
    )
    packed, identifiers = pack_world_model_request(
        request, maximum_input_tokens=100_000, token_counter=Utf8UpperBoundTokenCounter()
    )
    assert packed is request
    assert identifiers == ("fits",)
