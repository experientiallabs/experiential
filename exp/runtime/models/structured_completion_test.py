"""Request replay, raw-output retention and shared actor accounting at the model seam."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    AssistantAction,
    BillingSource,
    ModelCapabilities,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    ToolCall,
)
from exp.common.project import ProjectConfig, ProjectStore
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.structured_completion import (
    CompletionPlan,
    ReturnedOutputRejected,
    StructuredCompletion,
)


class _Client:
    """Provide exact output bytes to test admission without network work."""

    def __init__(self, model: ModelSnapshot, output: AssistantAction) -> None:
        """Retain the response and each request independently of accounting."""
        self.model, self.output = model, output
        self.served_model_id: str | None = None
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return one typed provider result with explicitly unknown cost."""
        self.requests.append(request)
        return ModelResponse.completed(
            output=self.output,
            configured_model=self.model,
            served_model_id=self.served_model_id,
            usage=None,
            latency_seconds=0,
        )


def setup(
    root: Path, output: AssistantAction, *, capped: bool = False
) -> tuple[StructuredCompletion, _Client, RequestBudget]:
    """Bind real project persistence and budget accounting to deterministic model output."""
    project = ProjectStore(root, "completion")
    project.initialize(ProjectConfig(project_id="completion"))
    caps = ModelCapabilities(context_window_tokens=10000, maximum_output_tokens=1000)
    model = ModelSnapshot(
        provider="fixture",
        model_id="fixture",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        capabilities_sha256=caps.identity_sha256(),
        connection_sha256="a" * 64,
    )
    client = _Client(model, output)
    budget = RequestBudget(project, identity="fixture", maximum_cost_usd=10 if capped else None)
    complete = StructuredCompletion(
        client=client,
        plan=CompletionPlan(
            model=model,
            capabilities=caps,
            deadline=datetime.now(UTC) + timedelta(minutes=1),
            output_tokens=1000,
            maximum_cost_per_call=1,
            code_revision="fixture",
        ),
        budget=budget,
        artifacts=project.artifacts,
    )
    return complete, client, budget


@pytest.mark.parametrize("content", ['{"a":1,"a":2}', '{"a":1e999}', '{"a":NaN}', "not-json"])
def test_invalid_json_is_retained_and_replay_never_repurchases(
    tmp_path: Path, content: str
) -> None:
    """Duplicate keys, overflow and invalid syntax stay paid output failures."""
    complete, client, budget = setup(tmp_path, AssistantAction(content=content))
    body: JsonObject = {"messages": [{"role": "user", "content": "return JSON"}]}
    for _ in range(2):
        with pytest.raises(ValueError):
            complete(body, "one-logical-call")
    assert len(client.requests) == 1
    assert budget.accounted_usd == 1


def test_unknown_tariff_cannot_start_a_capped_run(tmp_path: Path) -> None:
    """An arbitrary dollar reservation is not proof of a complete price bound."""
    complete, client, _ = setup(tmp_path, AssistantAction(content="{}"), capped=True)
    with pytest.raises(ValueError, match="complete applicable token prices"):
        complete({"messages": [{"role": "user", "content": "JSON"}]}, "capped")
    assert not client.requests


def test_actor_uses_same_retained_client_and_replays_its_turn(tmp_path: Path) -> None:
    """Native tool calls are actor output, while the JSON proposal adapter rejects them."""
    output = AssistantAction(tool_calls=(ToolCall(call_id="call", name="list", arguments={}),))
    complete, client, budget = setup(tmp_path, output)
    request = ModelRequest(messages=(ModelMessage(role="user", content="list"),))
    first = complete.as_model_client("episode").complete(request)
    repeated = complete.as_model_client("episode").complete(request)
    assert first == repeated and first.output == output
    assert len(client.requests) == 1 and budget.accounted_usd == 1
    with pytest.raises(ValueError, match="cannot execute provider tools"):
        complete({"messages": [{"role": "user", "content": "JSON"}]}, "structured")
    assert len(client.requests) == 2


def test_capacity_refusal_makes_no_provider_call_or_reservation(tmp_path: Path) -> None:
    """The typed local refusal establishes non-dispatch rather than an unknown purchase."""
    from exp.common.models.client import RequestCapacityExceeded

    complete, client, budget = setup(tmp_path, AssistantAction(content="{}"))
    body: JsonObject = {"messages": [{"role": "user", "content": "x" * 11000}]}
    with pytest.raises(RequestCapacityExceeded):
        complete(body, "too-large")
    assert client.requests == []
    assert budget.accounted_usd == 0


def test_capacity_counts_decoded_content_not_json_transport_escapes(tmp_path: Path) -> None:
    """Nested JSON fits the declared bound without counting its wire escaping twice."""
    complete, client, budget = setup(tmp_path, AssistantAction(content="{}"))
    request = ModelRequest(
        messages=(ModelMessage(role="user", content='{"key":"value"}\n' * 500),),
        maximum_output_tokens=1000,
    )
    assert len(request.model_dump_json().encode()) + 1000 > 10000
    complete.preflight_request(request)
    assert not client.requests
    assert budget.accounted_usd == 0


def test_named_tool_output_uses_actual_schema_and_validates_response(tmp_path: Path) -> None:
    """The answer tool submits data only; it never executes a generated function."""
    from exp.common.models import ToolChoice

    complete, client, _ = setup(
        tmp_path,
        AssistantAction(
            tool_calls=(
                ToolCall(call_id="answer", name="submit_proposal", arguments={"value": "current"}),
            )
        ),
    )
    caps = complete.plan.capabilities.model_copy(update={"supports_tools": True})
    model = complete.plan.model.model_copy(update={"capabilities_sha256": caps.identity_sha256()})
    client.model = model
    complete.plan = complete.plan.model_copy(
        update={"capabilities": caps, "model": model, "output_protocol": "tool"}
    )
    schema: JsonObject = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    body: JsonObject = {
        "messages": [{"role": "user", "content": "Submit JSON."}],
        "response_schema": schema,
    }
    assert complete(body, "tool-answer") == {"value": "current"}
    assert client.requests[0].tool_choice == ToolChoice(name="submit_proposal")
    assert client.requests[0].tools[0].input_schema == schema
    client.output = AssistantAction(
        tool_calls=(
            ToolCall(call_id="invalid", name="submit_proposal", arguments={"invented": True}),
        )
    )
    from jsonschema import ValidationError

    with pytest.raises(ReturnedOutputRejected) as refused:
        complete(body, "tool-invalid")
    assert isinstance(refused.value.__cause__, ValidationError)


def test_response_under_the_catalog_served_model_id_is_admitted(tmp_path: Path) -> None:
    """A provider may answer under its pinned served model ID, and under no other ID."""
    complete, client, _ = setup(tmp_path, AssistantAction(content="{}"))
    client.served_model_id = "fixture-2026-10-01"
    body: JsonObject = {"messages": [{"role": "user", "content": "return JSON"}]}
    with pytest.raises(ValueError, match="model identity differs"):
        complete(body, "unpinned")
    complete.plan = complete.plan.model_copy(update={"served_model_id": "fixture-2026-10-01"})
    assert complete(body, "pinned") == {}
    complete.plan = complete.plan.model_copy(update={"served_model_id": "another-model"})
    with pytest.raises(ValueError, match="model identity differs"):
        complete(body, "other-pin")
