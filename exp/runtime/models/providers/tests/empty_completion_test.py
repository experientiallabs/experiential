"""Empty completed Chat responses receive bounded retries without accepting malformed tools."""

import asyncio
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectConfig, ProjectStore
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.errors import ProviderRetryableResponseError
from exp.runtime.models.providers.openai_compatible import OpenAICompatibleClient
from exp.runtime.models.providers.openai_compatible_test import _request, _snapshot
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    ProviderTransportError,
    RetryPolicy,
    ScriptedJsonTransport,
)
from exp.runtime.sessions.dispatch_ledger import DispatchLedger


@pytest.mark.parametrize(
    "empty",
    [
        {"choices": []},
        {"choices": [{"finish_reason": "stop", "message": {"content": None}}]},
    ],
)
def test_empty_chat_response_retries_then_returns_real_output(empty: JsonObject) -> None:
    """Empty output without a known allowance stop may retry within its finite policy."""
    transport = ScriptedJsonTransport(
        [
            JsonHttpResponse(status_code=200, body=empty),
            JsonHttpResponse(status_code=200, body={"choices": [{"message": {"content": "done"}}]}),
        ]
    )
    client = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://example.test/v1",
        api_key="fixture",
        transport=transport,
        retry_policy=RetryPolicy(maximum_attempts=2, initial_delay_seconds=0),
    )
    assert client.complete(_request()).output.content == "done"
    assert len(transport.requests) == 2
    assert (
        transport.requests[0].headers["Idempotency-Key"]
        != transport.requests[1].headers["Idempotency-Key"]
    )


@pytest.mark.parametrize("explicit_key", [None, "caller-owned-operation"])
def test_lost_completed_replay_stops_without_rotating_any_key(explicit_key: str | None) -> None:
    """A lost completed result is unknown; neither generated nor explicit keys authorize redraw."""
    transport = ScriptedJsonTransport(
        [
            JsonHttpResponse(
                status_code=409, body={"error": {"code": "idempotency_replay_unavailable"}}
            ),
            JsonHttpResponse(status_code=200, body={"choices": [{"message": {"content": "done"}}]}),
        ]
    )
    client = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://example.test/v1",
        api_key="fixture",
        transport=transport,
        retry_policy=RetryPolicy(maximum_attempts=2, initial_delay_seconds=0),
    )
    with pytest.raises(ProviderTransportError) as failure:
        if explicit_key is None:
            client.complete(_request())
        else:
            asyncio.run(client.complete_async(_request(), idempotency_key=explicit_key))
    assert failure.value.error_code == "idempotency_replay_unavailable"
    assert len(transport.requests) == 1
    if explicit_key is not None:
        assert transport.requests[0].headers["Idempotency-Key"] == explicit_key


def test_empty_chat_response_stops_at_the_retry_bound() -> None:
    """Repeated emptiness remains invalid and cannot cause an unbounded provider loop."""
    transport = ScriptedJsonTransport(
        [JsonHttpResponse(status_code=200, body={"choices": []}) for _ in range(2)]
    )
    client = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://example.test/v1",
        api_key="fixture",
        transport=transport,
        retry_policy=RetryPolicy(maximum_attempts=2, initial_delay_seconds=0),
    )
    with pytest.raises(ProviderRetryableResponseError) as raised:
        client.complete(_request())
    assert raised.value.hit_length_limit is False
    assert raised.value.response_body is None
    assert len(transport.requests) == 2


@pytest.mark.parametrize("idempotent_wire", [True, False])
def test_empty_chat_response_at_the_output_allowance_reports_the_length_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, idempotent_wire: bool
) -> None:
    """Keep the raw known-budget stop after one dispatch despite headroom and later success."""
    message: JsonObject = {"role": "assistant", "content": None}
    empty: JsonObject = {
        "id": "private-returned-body-marker",
        "choices": [{"finish_reason": "length", "message": message}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 4096,
            "completion_tokens_details": {"reasoning_tokens": 4096},
        },
    }
    expected = deepcopy(empty)
    inner = ScriptedJsonTransport(
        [
            JsonHttpResponse(status_code=200, body=empty),
            JsonHttpResponse(status_code=200, body={"choices": [{"message": {"content": "done"}}]}),
        ]
    )
    project = ProjectStore(tmp_path, "attempt")
    project.initialize(ProjectConfig(project_id="attempt"))
    ledger = DispatchLedger(tmp_path / "physical.sqlite", 3, "a" * 64)
    transport = AdmittedJsonTransport(
        inner,
        ledger=ledger,
        artifacts=project.artifacts,
        code_revision="fixture",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
    )
    client = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://example.test/v1",
        api_key="credential-never-in-receipts",
        transport=transport,
        retry_policy=RetryPolicy(maximum_attempts=3, initial_delay_seconds=0),
    )
    # Exercise both shared retry classifiers with the same real wire parser.
    monkeypatch.setattr(client, "supports_idempotency_key", idempotent_wire)
    with pytest.raises(ProviderRetryableResponseError) as raised:
        client.complete(_request().model_copy(update={"maximum_output_tokens": 4096}))
    assert raised.value.hit_length_limit is True
    retained = raised.value.response_body
    assert retained is not None and retained == expected
    assert retained is not empty
    assert retained["choices"] is not empty["choices"]
    assert "private-returned-body-marker" not in str(raised.value)
    assert "private-returned-body-marker" not in repr(raised.value)
    assert ledger.used() == len(inner.requests) == 1, "remaining capacity is not redraw authority"
    artifacts = project.artifacts.list_ids(artifact_type="provider-physical-attempt")
    assert len(artifacts) == 2
    response = next(identity for identity in artifacts if identity.endswith("-response"))
    receipt_bytes = project.artifacts.read_bytes(response, "receipt.json")
    receipt = json.loads(receipt_bytes)
    assert receipt["http_status"] == 200 and receipt["response_body"] == expected
    assert b"credential-never-in-receipts" not in receipt_bytes
    assert b"Authorization" not in receipt_bytes
    message["content"] = "mutated after return"
    assert raised.value.response_body == expected, "retained evidence is a defensive deep copy"
    assert json.loads(project.artifacts.read_bytes(response, "receipt.json")) == receipt
