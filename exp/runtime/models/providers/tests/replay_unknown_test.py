"""A lost completed keyed result remains unknown; another sampled answer is not recovery."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models import ModelCapabilities
from exp.common.models.unknown_outcome import DispatchOutcomeUnknown
from exp.common.project import ProjectConfig, ProjectStore
from exp.common.project.request_budget import RequestBudgetStore
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.openai_compatible import OpenAICompatibleClient
from exp.runtime.models.providers.openai_compatible_test import _snapshot
from exp.runtime.models.providers.transport import (
    HttpxJsonTransport,
    ProviderTransportError,
    RetryPolicy,
)
from exp.runtime.models.structured_completion import CompletionPlan, StructuredCompletion
from exp.runtime.sessions.dispatch_ledger import DispatchLedger


def test_lost_keyed_result_never_rotates_or_buys_available_third_response(tmp_path: Path) -> None:
    """Actual HTTP503→409 must stop on the same key/body and retain an unknown logical receipt."""
    project = ProjectStore(tmp_path, "unknown-replay")
    project.initialize(ProjectConfig(project_id="unknown-replay"))
    requests: list[httpx.Request] = []

    def response(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        bodies = [
            (503, {"error": {"code": "deadline_exceeded"}}),
            (409, {"error": {"code": "idempotency_replay_unavailable"}}),
            (
                200,
                {
                    "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            ),
        ]
        status, body = bodies[len(requests) - 1]
        return httpx.Response(status, json=body)

    ledger = DispatchLedger(tmp_path / "physical.sqlite", 3, "a" * 64)
    caps = ModelCapabilities(
        supports_structured_output=True, context_window_tokens=10000, maximum_output_tokens=1024
    )
    model = _snapshot().model_copy(update={"capabilities_sha256": caps.identity_sha256()})
    budget = RequestBudget(project, identity="replay-boundary")
    with httpx.Client(transport=httpx.MockTransport(response)) as http:
        admitted = AdmittedJsonTransport(
            HttpxJsonTransport(http),
            ledger=ledger,
            artifacts=project.artifacts,
            code_revision="fixture",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )
        client = OpenAICompatibleClient(
            model=model,
            base_url="https://fixture.test/v1",
            api_key="fixture",
            transport=admitted,
            retry_policy=RetryPolicy(maximum_attempts=3, initial_delay_seconds=0),
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
                json_object_output=True,
            ),
            budget=budget,
            artifacts=project.artifacts,
        )
        body: JsonObject = {"messages": [{"role": "user", "content": "Return a JSON object."}]}
        with pytest.raises(ProviderTransportError) as failure:
            complete(body, "one-logical-request")
        assert failure.value.error_code == "idempotency_replay_unavailable"
        assert len(requests) == ledger.used() == 2
        assert requests[0].content == requests[1].content
        assert requests[0].headers["Idempotency-Key"] == requests[1].headers["Idempotency-Key"]
        key = sha256_json({"scope": "one-logical-request", "role": "structured", "ordinal": 0})
        receipt = RequestBudgetStore(project, "replay-boundary").read(key)
        assert receipt is not None and receipt.state == "unknown" and receipt.response is None
        with pytest.raises(DispatchOutcomeUnknown):
            complete(body, "one-logical-request")
        assert len(requests) == 2
    physical_ids = [
        x
        for x in project.artifacts.list_ids()
        if x.startswith("provider-attempt-") and x.endswith("-intent")
    ]
    saved = [json.loads(project.artifacts.read_bytes(x, "receipt.json")) for x in physical_ids]
    assert len(saved) == 2
    assert len({x["idempotency_key"] for x in saved}) == 1
    assert len({x["request_sha256"] for x in saved}) == 1
    assert len({x["canonical_request"]["artifact_id"] for x in saved}) == 1
    assert all(x["request_body"] == json.loads(requests[0].content) for x in saved)


@pytest.mark.parametrize("status", [429, 503])
def test_safe_idempotent_status_retry_keeps_original_key(status: int) -> None:
    """Ordinary safe endpoint retries may obtain the original operation's returned result."""
    from exp.runtime.models.providers.openai_compatible_test import _request
    from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport

    inner = ScriptedJsonTransport(
        [
            JsonHttpResponse(status, {"error": {"code": "overloaded"}}),
            JsonHttpResponse(200, {"choices": [{"message": {"content": "done"}}]}),
        ]
    )
    client = OpenAICompatibleClient(
        model=_snapshot(),
        api_key="fixture",
        base_url="https://fixture.test/v1",
        transport=inner,
        retry_policy=RetryPolicy(maximum_attempts=3, initial_delay_seconds=0),
    )
    assert client.complete(_request()).output.content == "done"
    assert len(inner.requests) == 2
    assert (
        inner.requests[0].headers["Idempotency-Key"] == inner.requests[1].headers["Idempotency-Key"]
    )


class UnrepeatableClient(OpenAICompatibleClient):
    """Exercise the same real wire implementation with deduplication explicitly unavailable."""

    supports_idempotency_key = False


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("surface", ["completion", "raw-embedding"])
def test_nonidempotent_wire_does_not_infer_safe_resend_from_http_status(
    status: int, surface: str
) -> None:
    """Both completion and raw-post owners refuse an unproven resend on a non-deduplicating wire."""
    from exp.runtime.models.providers.openai_compatible_test import _request
    from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport

    inner = ScriptedJsonTransport(
        [
            JsonHttpResponse(status, {"error": {"code": "overloaded"}}),
            JsonHttpResponse(200, {"choices": [{"message": {"content": "must not consume"}}]}),
        ]
    )
    client = UnrepeatableClient(
        model=_snapshot(),
        api_key="fixture",
        base_url="https://fixture.test/v1",
        transport=inner,
        retry_policy=RetryPolicy(maximum_attempts=3, initial_delay_seconds=0),
    )
    with pytest.raises(ProviderTransportError):
        if surface == "completion":
            client.complete(_request())
        else:
            client.embed(["visible input"])
    assert len(inner.requests) == 1
