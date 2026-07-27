"""Tests for the Bedrock provider's Nova request/response path and region resolution (no network).

The region tests never reach a backend either: building a boto3 client signs nothing, and every
ambient AWS source is pointed at nothing so the answers do not depend on the machine.
"""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, cast

import boto3
import pytest

from wmo.config.config import PROVIDER_ENV_VARS
from wmo.providers.base import ChatRequest, Message, ProviderConfig, ProviderKind
from wmo.providers.bedrock import (
    AWS_DEFAULT_REGION_ENV,
    AWS_REGION_ENV,
    NO_REGION_ERROR,
    REGION_SOURCES,
    BedrockProvider,
    _is_nova,
    resolve_region,
)

if TYPE_CHECKING:
    from pathlib import Path

    from botocore.client import BaseClient


class _StubClient:
    """Captures invoke_model calls and returns a canned body."""

    def __init__(self, response: dict) -> None:  # noqa: ANN401 - boto3 responses are untyped dicts
        self._response = response
        self.model_id: str | None = None
        self.body: dict | None = None

    def invoke_model(self, *, modelId: str, body: str) -> dict:  # noqa: N803 - boto3 kwarg name
        self.model_id = modelId
        self.body = json.loads(body)
        return {"body": io.BytesIO(json.dumps(self._response).encode("utf-8"))}


class _StubConverseClient:
    """Captures structured Converse requests and returns one text response."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def converse(self, **kwargs: object) -> dict[str, object]:
        self.requests.append(kwargs)
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 2, "outputTokens": 1},
        }


def test_is_nova_matches_nova_model_ids_only() -> None:
    assert _is_nova("us.amazon.nova-lite-v1:0")
    assert _is_nova("amazon.nova-micro-v1:0")
    assert not _is_nova("us.anthropic.claude-opus-4-8")
    assert not _is_nova("amazon.titan-embed-text-v2:0")


def test_nova_complete_builds_nova_body_and_parses_response() -> None:
    provider = BedrockProvider(
        ProviderConfig(kind=ProviderKind.BEDROCK, model="us.amazon.nova-lite-v1:0")
    )
    stub = _StubClient(
        {
            "output": {"message": {"content": [{"text": "hello "}, {"text": "world"}]}},
            "usage": {"inputTokens": 12, "outputTokens": 5},
        }
    )
    provider._client = cast("BaseClient", stub)  # inject; _get_client returns it

    completion = provider.complete(
        "be brief",
        [Message(role="user", content="hi")],
        temperature=0.4,
        max_tokens=64,
    )

    assert completion.text == "hello world"
    assert completion.usage.input_tokens == 12
    assert completion.usage.output_tokens == 5
    assert stub.model_id == "us.amazon.nova-lite-v1:0"
    assert stub.body == {
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
        "inferenceConfig": {"maxTokens": 64, "temperature": 0.4},
        "system": [{"text": "be brief"}],
    }


def test_nova_complete_omits_empty_system() -> None:
    provider = BedrockProvider(
        ProviderConfig(kind=ProviderKind.BEDROCK, model="us.amazon.nova-lite-v1:0")
    )
    stub = _StubClient(
        {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
    )
    provider._client = cast("BaseClient", stub)

    provider.complete("", [Message(role="user", content="hi")])
    assert stub.body is not None
    assert "system" not in stub.body


def test_structured_chat_normalizes_temperature_for_unsupported_model() -> None:
    provider = BedrockProvider(
        ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model_type="claude-opus-4-8",
            model="us.anthropic.claude-opus-4-8",
        )
    )
    stub = _StubConverseClient()
    provider._client = cast("BaseClient", stub)
    request = ChatRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.3,
            "max_completion_tokens": 64,
        }
    )

    provider.complete_chat(request)

    assert request.temperature == 0.3  # normalization does not mutate the reusable request
    assert stub.requests[0]["inferenceConfig"] == {"maxTokens": 64}


def test_structured_chat_preserves_temperature_for_supported_model() -> None:
    provider = BedrockProvider(
        ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model_type="claude-sonnet-4-6",
            model="us.anthropic.claude-sonnet-4-6",
        )
    )
    stub = _StubConverseClient()
    provider._client = cast("BaseClient", stub)

    provider.complete_chat(
        ChatRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.3,
                "max_completion_tokens": 64,
            }
        )
    )

    assert stub.requests[0]["inferenceConfig"] == {"maxTokens": 64, "temperature": 0.3}


def test_claude_invoke_normalizes_cache_read_and_write_to_subsets() -> None:
    # Bedrock's Anthropic body reports cache reads/writes BESIDE input_tokens; TokenUsage's
    # contract is subsets of input_tokens, so the provider must sum at the boundary.
    provider = BedrockProvider(
        ProviderConfig(kind=ProviderKind.BEDROCK, model="us.anthropic.claude-opus-4-8")
    )
    stub = _StubClient(
        {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 3,
                "cache_read_input_tokens": 40,
                "cache_creation_input_tokens": 50,
            },
        }
    )
    provider._client = cast("BaseClient", stub)

    completion = provider.complete("sys", [Message(role="user", content="hi")])

    assert completion.usage.input_tokens == 100  # 10 fresh + 40 read + 50 written
    assert completion.usage.cached_input_tokens == 40
    assert completion.usage.cache_write_input_tokens == 50


class _StubCacheConverseClient:
    """Converse stub whose usage carries both cache tiers."""

    def converse(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 7,
                "outputTokens": 2,
                "cacheReadInputTokens": 20,
                "cacheWriteInputTokens": 30,
            },
        }


def test_converse_normalizes_cache_read_and_write_to_subsets() -> None:
    provider = BedrockProvider(
        ProviderConfig(kind=ProviderKind.BEDROCK, model="us.moonshotai.kimi-k2-6")
    )
    provider._client = cast("BaseClient", _StubCacheConverseClient())

    completion = provider.complete("sys", [Message(role="user", content="hi")])

    assert completion.usage.input_tokens == 57  # 7 fresh + 20 read + 30 written
    assert completion.usage.cached_input_tokens == 20
    assert completion.usage.cache_write_input_tokens == 30


def _regionless_config() -> ProviderConfig:
    """A Bedrock config carrying no region of its own, so the environment decides."""
    return ProviderConfig(kind=ProviderKind.BEDROCK, model="us.anthropic.claude-opus-4-8")


@pytest.fixture
def no_ambient_aws(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point every AWS region source at nothing, so these tests answer the same on any machine.

    boto3 falls back to `~/.aws/config`, so on a developer box with a configured profile a region
    resolves no matter what the environment says and every assertion below would pass by
    accident. Both credential files are aimed at paths that do not exist and the metadata
    endpoint is disabled (a belt: no test may touch the network). boto3's cached process-wide
    default session is cleared too, because `boto3.client` resolves through it and it snapshots
    the environment as of whenever some earlier test first built a client.
    """
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-aws-credentials"))
    for name in (AWS_REGION_ENV, AWS_DEFAULT_REGION_ENV, "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)


def test_client_is_built_in_the_region_aws_region_names(
    no_ambient_aws: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The bug this fixes: botocore's session reads only AWS_DEFAULT_REGION (its mapping for
    # `region` is `("region", "AWS_DEFAULT_REGION", None, None)`), so passing `region_name=None`
    # with AWS_REGION correctly set raised NoRegionError, "You must specify a region."
    monkeypatch.setenv(AWS_REGION_ENV, "us-east-1")

    client = BedrockProvider(_regionless_config())._get_client()

    assert client.meta.region_name == "us-east-1"


def test_explicit_config_region_beats_aws_region(
    no_ambient_aws: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The entry/config region is the most specific source, so an ambient AWS_REGION may not
    # silently redirect a pool entry that named its own region.
    monkeypatch.setenv(AWS_REGION_ENV, "us-east-1")
    config = ProviderConfig(
        kind=ProviderKind.BEDROCK, model="us.anthropic.claude-opus-4-8", region="eu-central-1"
    )

    assert BedrockProvider(config)._get_client().meta.region_name == "eu-central-1"


def test_aws_region_beats_aws_default_region(
    no_ambient_aws: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(AWS_REGION_ENV, "us-east-1")
    monkeypatch.setenv(AWS_DEFAULT_REGION_ENV, "ap-southeast-2")

    assert BedrockProvider(_regionless_config())._get_client().meta.region_name == "us-east-1"


def test_aws_default_region_still_resolves_through_boto3(
    no_ambient_aws: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reading AWS_REGION ourselves must not take anything away from boto3: with AWS_REGION unset,
    # `resolve_region` returns None and the whole boto3 chain (AWS_DEFAULT_REGION, the active
    # profile, an instance role) is still what decides.
    monkeypatch.setenv(AWS_DEFAULT_REGION_ENV, "ap-southeast-2")

    assert resolve_region(None) is None
    assert BedrockProvider(_regionless_config())._get_client().meta.region_name == "ap-southeast-2"


def test_prepare_accepts_aws_region_alone(
    no_ambient_aws: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The pre-flight resolves the region the same way the client does, or a `route sweep` would
    # refuse a Bedrock candidate that is in fact perfectly callable.
    monkeypatch.setenv(AWS_REGION_ENV, "us-east-1")

    BedrockProvider(_regionless_config()).prepare()  # does not raise


def test_no_region_anywhere_names_every_source_in_resolution_order(no_ambient_aws: None) -> None:
    with pytest.raises(ValueError, match=AWS_REGION_ENV) as excinfo:
        BedrockProvider(_regionless_config()).prepare()

    message = str(excinfo.value)
    positions = [message.find(source) for source in REGION_SOURCES]
    assert all(position >= 0 for position in positions), message  # every source is named
    assert positions == sorted(positions), message  # in the order they are consulted


def test_verify_reports_the_actionable_region_failure_not_botocores(no_ambient_aws: None) -> None:
    # `verify_via_ping` reports whatever the first call raised, and that detail is what
    # `wmo build`'s pre-flight, `wmo providers set` and `wmo providers verify` all print.
    result = BedrockProvider(_regionless_config()).verify()

    assert result.ok is False
    assert result.detail == NO_REGION_ERROR
    assert "You must specify a region." not in result.detail


def test_provider_env_vars_name_the_region_variable_this_provider_reads() -> None:
    # PROVIDER_ENV_VARS holds literals (importing a provider module from config would invert the
    # dependency), so the pair is pinned here, as openrouter's and tinker's key vars are.
    assert PROVIDER_ENV_VARS[ProviderKind.BEDROCK][0] == AWS_REGION_ENV
    # AWS_DEFAULT_REGION is honoured by `resolve_region` but deliberately absent here: the dict
    # is an all-must-be-set presence contract (`wmo.cli.ui.has_credentials`) whose prompt writes
    # each missing name into `.env`, so listing an ALTERNATIVE spelling of the same value would
    # call a correct environment incomplete and ask for the region twice.
    assert AWS_DEFAULT_REGION_ENV not in PROVIDER_ENV_VARS[ProviderKind.BEDROCK]
