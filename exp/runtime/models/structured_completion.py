"""Budgeted structured proposals through the maintained model-client interface."""

import copy
import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from pydantic import AwareDatetime, Field, TypeAdapter

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.core.json_schema import expand_local_references
from exp.common.models import (
    ModelCapabilities,
    ModelClient,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    ToolChoice,
)
from exp.common.models.client import RequestCapacityExceeded
from exp.common.project import ArtifactStore
from exp.common.tasks import ToolSchema
from exp.common.traces.ingest.json_strict import reject_duplicate_json_keys
from exp.runtime.models.budget import RequestBudget

_JSON_OBJECT = TypeAdapter(JsonObject)


class CompletionNotDispatched(TimeoutError):
    """The completion was refused before any provider request was sent.

    Nothing was purchased, so a caller may close the attempt and try again later.
    """


class ReturnedOutputRejected(ValueError):
    """A provider response returned and was retained, but its output was refused.

    Raised only after the client returned, so callers may treat the attempt as closed and
    explain the refusal on a new attempt. Any other error leaves the purchase unresolved.
    """


class CompletionPlan(ContractModel):
    """Frozen limits for structured calls over one registered client.

    Attributes:
        model: Expected configured and returned model identity.
        served_model_id: Catalog pin for an alternate provider-reported model ID; a response
            under it must keep the model's provider, revision, capability and connection digests.
        capabilities: Context and output limits belonging to that model.
        deadline: Latest time a returned proposal can be admitted.
        output_tokens: Output reservation for every call.
        maximum_cost_per_call: Conservative single-attempt USD reservation.
        code_revision: Source identity recorded in immutable call artifacts.
        json_object_output: Explicit route capability; false uses strict local parsing.
        output_protocol: JSON text or one named tool carrying the response schema.
        cost_is_upper_bound: Caller has established a complete applicable tariff bound.
            False requires explicit uncapped spending authority; unknown charges remain unknown.
    """

    model: ModelSnapshot
    served_model_id: str | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )
    capabilities: ModelCapabilities
    deadline: AwareDatetime
    output_tokens: int = Field(gt=0, strict=True)
    maximum_cost_per_call: float = Field(gt=0, allow_inf_nan=False)
    code_revision: str = Field(min_length=1)
    json_object_output: bool = Field(default=False, strict=True)
    cost_is_upper_bound: bool = Field(default=False, strict=True)
    output_protocol: Literal["json", "tool"] = "json"


class StructuredCompletion:
    """Meter and retain model outputs without adding transport retries.

    The injected client owns transport and its prospectively configured retry policy.
    The admitted provider transport counts every physical attempt, including retries.
    The shared RequestBudget retains unknown charges and exact response replay.
    Generated code is never executed here. Callers must not expose this client to it.
    """

    def __init__(
        self,
        *,
        client: ModelClient,
        plan: CompletionPlan,
        budget: RequestBudget,
        artifacts: ArtifactStore,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Bind already configured provider, budget, physical counter and artifacts."""
        if plan.capabilities.identity_sha256() != plan.model.capabilities_sha256:
            raise ValueError("completion capabilities differ from the model snapshot")
        if plan.json_object_output and not plan.capabilities.supports_structured_output:
            raise ValueError("JSON output mode requires declared provider support")
        if plan.output_protocol == "tool" and plan.capabilities.supports_tools is not True:
            raise ValueError("Named structured output requires declared tool support")
        if plan.output_protocol == "tool" and plan.json_object_output:
            raise ValueError("Named-tool and JSON-text output modes are mutually exclusive")
        self.plan = plan
        self._client, self._budget = client, budget
        self._artifacts, self._clock = artifacts, clock or _utc_now

    def __call__(self, body: JsonObject, logical_id: str) -> JsonObject:
        """Dispatch one structured request or return its exact saved proposal.

        Args:
            body: Provider-neutral messages and optional explicit response schema.
            logical_id: Unique registered role/operation identity, stable on replay.

        Returns:
            Strict JSON object after raw model output has been durably retained.
        """
        request, schema = self._prepare(body)
        response = self.complete_request(request, logical_id)
        try:
            return self._accept(response, schema)
        except ReturnedOutputRejected:
            raise
        except Exception as error:  # noqa: BLE001 - only output checks run after return.
            raise ReturnedOutputRejected(f"{type(error).__name__}: {error}") from error

    def _accept(self, response: ModelResponse, schema: JsonObject | None) -> JsonObject:
        """Check returned output; every failure here concerns a retained response."""
        from jsonschema import Draft202012Validator

        if response.finish_reason != ModelFinishReason.COMPLETED:
            raise ReturnedOutputRejected(
                "structured proposal did not complete; raw response retained"
            )
        if self.plan.output_protocol == "tool":
            calls = response.output.tool_calls
            if len(calls) != 1 or calls[0].name != "submit_proposal":
                raise ReturnedOutputRejected("Expected exactly one named structured proposal")
            if not isinstance(schema, dict):
                raise ValueError("named-tool output requires a response schema")
            Draft202012Validator(schema).validate(calls[0].arguments)
            return copy.deepcopy(calls[0].arguments)
        if response.output.tool_calls:
            raise ReturnedOutputRejected("structured proposals cannot execute provider tools")
        return _JSON_OBJECT.validate_python(
            json.loads(
                response.output.content or "",
                object_pairs_hook=reject_duplicate_json_keys,
                parse_constant=_reject_constant,
                parse_float=_finite_float,
            )
        )

    def _prepare(self, body: JsonObject) -> tuple[ModelRequest, JsonObject | None]:
        """Build the typed request used by both local preflight and dispatch."""
        messages = body.get("messages")
        if set(body) - {"messages", "response_schema"} or not isinstance(messages, list):
            raise ValueError(
                "structured completion expects messages and an optional response schema"
            )
        tools: tuple[ToolSchema, ...] = ()
        schema = body.get("response_schema")
        if schema is not None and not isinstance(schema, dict):
            raise ValueError("Response schema must be a JSON object")
        if self.plan.output_protocol == "tool":
            if not isinstance(schema, dict):
                raise ValueError("Named-tool output requires an explicit response schema")
            tools = (
                ToolSchema(
                    name="submit_proposal",
                    description="Submit the requested structured proposal.",
                    input_schema=expand_local_references(schema),
                ),
            )
        return ModelRequest(
            messages=tuple(ModelMessage.model_validate(row) for row in messages),
            tools=tools,
            maximum_output_tokens=self.plan.output_tokens,
            tool_choice=ToolChoice(name="submit_proposal") if tools else "none",
            json_object_output=self.plan.json_object_output,
        ), schema

    def preflight(self, body: JsonObject) -> None:
        """Refuse local capacity before a session marks a possible external purchase."""
        request, _ = self._prepare(body)
        self._check_capacity(request)

    def preflight_request(self, request: ModelRequest) -> None:
        """Check a fully framed model request without changing its wire protocol."""
        self._check_capacity(request)

    def _check_capacity(self, request: ModelRequest) -> None:
        """Validate the actual typed request and pinned output reservation without I/O."""
        requested_output = request.maximum_output_tokens or self.plan.output_tokens
        if requested_output > self.plan.output_tokens:
            raise RequestCapacityExceeded("request exceeds the pinned output reservation")
        # Providers tokenize decoded content, not its extra JSON transport escaping.
        framing = request.model_copy(
            update={
                "messages": tuple(
                    message.model_copy(update={"content": ""})
                    if message.content is not None
                    else message
                    for message in request.messages
                )
            }
        )
        input_bound = (
            len(framing.model_dump_json().encode())
            + sum(len((message.content or "").encode()) for message in request.messages)
            + 256 * len(request.messages)
        )
        caps = self.plan.capabilities
        if (
            caps.context_window_tokens is None
            or input_bound + requested_output > caps.context_window_tokens
        ):
            raise RequestCapacityExceeded(
                "complete structured request exceeds the declared context capacity"
            )
        if caps.maximum_output_tokens is not None and requested_output > caps.maximum_output_tokens:
            raise RequestCapacityExceeded("output reservation exceeds the declared route capacity")
        if self._clock() >= self.plan.deadline:
            raise CompletionNotDispatched("completion deadline reached before dispatch")

    def complete_request(self, request: ModelRequest, logical_id: str) -> ModelResponse:
        """Retain and meter a typed request, including actor tool calls, through one owner.

        This boundary preserves length stops for the caller. Only the structured JSON
        adapter requires a fully completed, tool-free output.
        """
        requested_output = request.maximum_output_tokens or self.plan.output_tokens
        request = request.model_copy(update={"maximum_output_tokens": requested_output})
        self._check_capacity(request)
        fingerprint = sha256_json(
            {"request": request.model_dump(mode="json"), "plan": self.plan.model_dump(mode="json")}
        )

        def operation() -> ModelResponse:
            """Claim physical capacity and retain intent before using the injected client."""
            if self._clock() >= self.plan.deadline:
                raise TimeoutError("completion deadline reached during admission")
            identity = stable_id(
                "structured-intent", {"logical_id": logical_id, "fingerprint": fingerprint}
            )
            self._artifacts.write_json(
                artifact_id=identity,
                artifact_type="structured-call-intent",
                envelope=ArtifactEnvelope(
                    schema_version=1,
                    created_at=self._clock(),
                    inputs=(),
                    code_revision=self.plan.code_revision,
                ),
                files={
                    "request.json": {
                        "request": request.model_dump(mode="json"),
                        "plan": self.plan.model_dump(mode="json"),
                    }
                },
            )
            return self._client.complete(request)

        with self._budget.scope(logical_id):
            response = self._budget.call(
                role="structured",
                fingerprint=fingerprint,
                maximum_cost_usd=self.plan.maximum_cost_per_call,
                operation=operation,
                encode=_encode,
                decode=ModelResponse.model_validate_json,
                charge=_charge,
                cost_is_upper_bound=self.plan.cost_is_upper_bound,
            )
        if response.model != self.plan.model and response.model != self._served_model():
            raise ValueError("model identity differs; raw response retained")
        if self._clock() >= self.plan.deadline:
            raise TimeoutError("late model response retained without admission")
        return response

    def _served_model(self) -> ModelSnapshot | None:
        """Return the pinned identity a provider may report under the catalog's served model ID."""
        if self.plan.served_model_id is None:
            return None
        return self.plan.model.model_copy(update={"model_id": self.plan.served_model_id})

    def as_model_client(self, episode_id: str) -> ModelClient:
        """Bind one sequential actor episode to the same request and budget boundary."""
        if not episode_id:
            raise ValueError("actor episode identity is required before dispatch")
        return _EpisodeClient(self, episode_id)


class ModelCompletion:
    """Preserve typed requests and raw responses at the durable proposal seam.

    The engine's own parser validates the returned model output. This adapter never
    strips prose, repairs JSON, or turns a returned invalid proposal into a lost reply.
    """

    def __init__(self, owner: StructuredCompletion) -> None:
        """Reuse one configured provider, request ledger and physical-attempt owner."""
        self._owner = owner

    def preflight(self, body: JsonObject) -> None:
        """Validate the actual typed request before the session claims dispatch."""
        self._owner.preflight_request(ModelRequest.model_validate(body))

    def __call__(self, body: JsonObject, logical_id: str) -> JsonObject:
        """Retain the raw response under the session's exact logical operation identity."""
        return self._owner.complete_request(
            ModelRequest.model_validate(body), logical_id
        ).model_dump(mode="json")


class _EpisodeClient:
    """Give the maintained actor a typed client without another provider or retry loop."""

    def __init__(self, owner: StructuredCompletion, episode_id: str) -> None:
        """Start at turn zero; reconstructing this adapter replays the same saved requests."""
        self._owner, self._episode_id, self._turn = owner, episode_id, 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Consume one exact turn coordinate before calling the existing budget owner."""
        turn = self._turn
        self._turn += 1
        return self._owner.complete_request(request, f"{self._episode_id}/actor-{turn}")


def _utc_now() -> datetime:
    """Return the aware UTC clock used for dispatch and response admission."""
    return datetime.now(UTC)


def _encode(response: ModelResponse) -> str:
    """Persist the full model result before protocol or semantic validation."""
    return response.model_dump_json()


def _charge(response: ModelResponse) -> float | None:
    """Keep missing cost unknown rather than treating it as a free generation."""
    cost = response.economics.cost_usd
    return None if cost is None else cost.value


def _reject_constant(value: str) -> None:
    """Reject NaN and infinity rather than admitting non-JSON model values."""
    raise ValueError("non-finite JSON constant: " + value)


def _finite_float(value: str) -> float:
    """Reject overflowing numeric literals as well as explicit non-finite constants."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number")
    return number
