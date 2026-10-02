"""Token counting is exercised by the simulator context preflight tests."""

from exp.common.models import ModelMessage, ModelRequest
from exp.runtime.gateway.json_object import JSON_OBJECT_SYSTEM_INSTRUCTION
from exp.simulation.engines.text.tokens import Utf8UpperBoundTokenCounter


def test_json_mode_reserves_provider_added_instruction_bytes() -> None:
    """Instruction-only adapters cannot exceed admission by adding uncounted JSON guidance."""
    request = ModelRequest(messages=(ModelMessage(role="user", content="JSON please"),))
    json_request = request.model_copy(update={"json_object_output": True})
    counter = Utf8UpperBoundTokenCounter()
    assert counter.count(json_request) >= counter.count(request) + len(
        JSON_OBJECT_SYSTEM_INSTRUCTION.encode("utf-8")
    )
