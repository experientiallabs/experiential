"""Request replay, raw-output retention and shared actor accounting at the model seam."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject
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
from exp.common.project import ArtifactStore, ProjectConfig, ProjectStore, artifact_input
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.call_evidence import canonical_call
from exp.runtime.models.providers.openai_compatible import OpenAICompatibleClient
from exp.runtime.models.providers.openai_compatible_test import _snapshot
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.spend_ceiling_test import _ceiling, _transport
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


@pytest.mark.parametrize(
    "output",
    [
        AssistantAction(content='{"value": "current"}'),
        AssistantAction(
            tool_calls=(
                ToolCall(call_id="one", name="submit_proposal", arguments={"value": "a"}),
                ToolCall(call_id="two", name="submit_proposal", arguments={"value": "b"}),
            )
        ),
    ],
)
def test_named_tool_output_refuses_what_an_automatic_choice_can_return(
    tmp_path: Path, output: AssistantAction
) -> None:
    """A model steered to the tool by ``auto`` must still return exactly one call.

    Text alone or a second call is retained, refused, and never bought again on replay.
    """
    complete, client, budget = setup(tmp_path, output)
    caps = complete.plan.capabilities.model_copy(
        update={"supports_tools": True, "supports_forced_tool_choice": False}
    )
    model = complete.plan.model.model_copy(update={"capabilities_sha256": caps.identity_sha256()})
    client.model = model
    complete.plan = complete.plan.model_copy(
        update={"capabilities": caps, "model": model, "output_protocol": "tool"}
    )
    body: JsonObject = {
        "messages": [{"role": "user", "content": "Submit JSON."}],
        "response_schema": {"type": "object"},
    }
    for _ in range(2):
        with pytest.raises(ReturnedOutputRejected, match="exactly one named"):
            complete(body, "steered-answer")
    assert len(client.requests) == 1 and budget.accounted_usd == 1


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


def test_completion_profile_pins_inference_settings_not_scheduling(tmp_path: Path) -> None:
    """A rubric cannot reuse qualification across different output protocols or allowances."""
    complete, _, _ = setup(tmp_path, AssistantAction(content="{}"))
    original = complete.completion_profile_sha256
    complete.plan = complete.plan.model_copy(
        update={"deadline": complete.plan.deadline + timedelta(hours=1), "maximum_cost_per_call": 2}
    )
    assert complete.completion_profile_sha256 == original
    for update in (
        {"output_tokens": 500},
        {"output_protocol": "tool"},
        {"json_object_output": True},
    ):
        plan = complete.plan
        complete.plan = plan.model_copy(update=update)
        assert complete.completion_profile_sha256 != original
        complete.plan = plan


def test_a_resumption_window_admits_past_the_plan_deadline_and_still_replays(
    tmp_path: Path,
) -> None:
    """The window extends admission; the plan, so the replay fingerprint, is unchanged."""
    complete, client, budget = setup(tmp_path, AssistantAction(content='{"a": 1}'))
    body: JsonObject = {"messages": [{"role": "user", "content": "return JSON"}]}
    assert complete(body, "answered") == {"a": 1}
    later = complete.plan.deadline + timedelta(hours=4)
    with pytest.raises(TimeoutError):
        StructuredCompletion(
            client=client,
            plan=complete.plan,
            budget=budget,
            artifacts=budget_artifacts(tmp_path),
            clock=lambda: later,
        )(body, "answered")
    resumed = StructuredCompletion(
        client=client,
        plan=complete.plan,
        budget=budget,
        artifacts=budget_artifacts(tmp_path),
        clock=lambda: later,
        window=later + timedelta(hours=3),
    )
    assert resumed(body, "answered") == {"a": 1}
    assert len(client.requests) == 1, "the saved answer replays without a provider call"
    assert resumed(body, "new") == {"a": 1}
    assert len(client.requests) == 2


def budget_artifacts(root: Path) -> ArtifactStore:
    """The fixture project's artifact store."""
    return ProjectStore(root, "completion").artifacts


@pytest.mark.parametrize("with_caller", [False, True])
def test_physical_structured_intent_preserves_caller_and_protected_source_ancestry(
    tmp_path: Path, with_caller: bool
) -> None:
    """Physical ownership stays structured; its outer source remains an immutable ancestor."""
    response = JsonHttpResponse(
        200,
        {
            "choices": [{"message": {"content": '{"ok":true}'}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        },
    )
    inner = ScriptedJsonTransport([response])
    admitted, project, physical = _transport(tmp_path, inner, _ceiling(tmp_path, 1))
    caps = ModelCapabilities(context_window_tokens=100000, maximum_output_tokens=1024)
    model = _snapshot().model_copy(update={"capabilities_sha256": caps.identity_sha256()})
    client = OpenAICompatibleClient(
        model=model, api_key="fixture", base_url="https://fixture.invalid/v1", transport=admitted
    )
    complete = StructuredCompletion(
        client=client,
        plan=CompletionPlan(
            model=model,
            capabilities=caps,
            deadline=datetime.now(UTC) + timedelta(minutes=1),
            output_tokens=1024,
            maximum_cost_per_call=1,
            code_revision="fixture",
        ),
        budget=RequestBudget(project, identity="ancestry"),
        artifacts=project.artifacts,
    )

    def source(identity: str, inputs: tuple[ArtifactInput, ...] = ()) -> ArtifactInput:
        return artifact_input(
            project.artifacts.write_json(
                artifact_id=identity,
                artifact_type="test-source",
                envelope=ArtifactEnvelope(
                    schema_version=1,
                    created_at=datetime.now(UTC),
                    code_revision="fixture",
                    inputs=inputs,
                ),
                files={"source.json": {"classification": "protected"}},
            )
        )

    protected = source("protected-source")
    caller = source("caller-intent", (protected,))
    body: JsonObject = {"messages": [{"role": "user", "content": "Assess the task."}]}
    if with_caller:
        with canonical_call(caller):
            assert complete(body, "one-cell") == {"ok": True}
            assert complete(body, "one-cell") == {"ok": True}
    else:
        assert complete(body, "one-cell") == {"ok": True}
        assert complete(body, "one-cell") == {"ok": True}
    assert len(inner.requests) == physical.used() == 1
    [intent_id] = [
        i
        for i in project.artifacts.list_ids(artifact_type="provider-physical-attempt")
        if i.endswith("-intent")
    ]
    receipt = json.loads(project.artifacts.read_bytes(intent_id, "receipt.json"))
    structured = project.artifacts.read(receipt["canonical_request"]["artifact_id"]).manifest
    assert structured.artifact_type == "structured-call-intent"
    assert project.artifacts.read(intent_id).manifest.inputs == (artifact_input(structured),)
    assert structured.inputs == ((caller,) if with_caller else ())
    if with_caller:
        assert project.artifacts.read(caller.artifact_id).manifest.inputs == (protected,)
        replacement = source("different-caller")
        with canonical_call(replacement), pytest.raises(ValueError):
            complete(body, "one-cell")
        assert len(inner.requests) == 1
        altered = caller.model_copy(update={"sha256": "0" * 64})
        with canonical_call(altered), pytest.raises(ValueError, match="exact artifact"):
            complete(body, "one-cell")
        assert len(inner.requests) == 1
