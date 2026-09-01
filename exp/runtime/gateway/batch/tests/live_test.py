"""Credential-gated live batch tests against the real provider batch APIs.

Each test runs one tiny real batch (two cheap lines) end to end: submit,
poll to completion, retrieve, and per-line accounting assertions. A missing
provider key skips with a note per house norms; these tests spend real money
in the smallest possible amounts and run only when explicitly selected:

    EXP_LIVE_BATCH=1 OPENAI_API_KEY=... uv run pytest exp/runtime/gateway/batch/live_test.py -q
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC

import pytest

from exp.runtime.gateway.batch.contracts import (
    TERMINAL_STATUSES,
    BatchCounts,
    BatchJob,
    BatchLine,
    BatchStatus,
    BatchSurface,
)
from exp.runtime.gateway.batch.providers import (
    AnthropicBatchClient,
    OpenAIBatchClient,
    OpenRouterBatchClient,
    ProviderBatchClient,
)

_LIVE = os.environ.get("EXP_LIVE_BATCH") == "1"
_POLL_SECONDS = 30.0
_DEADLINE_SECONDS = 30 * 60.0


def _job(provider: str, surface: BatchSurface, provider_model: str, body: dict) -> BatchJob:
    """Build one two-line live job for the given provider."""
    from datetime import datetime, timedelta

    created = datetime.now(UTC)
    lines = tuple(
        BatchLine(
            custom_id=f"live-{index}",
            surface=surface,
            model="live-batch-model",
            provider_model=provider_model,
            body=body,
            estimated_input_tokens=8,
            maximum_output_tokens=8,
        )
        for index in range(2)
    )
    return BatchJob(
        batch_id="batch_live",
        organization_id="org_live",
        identity_id="id_live",
        surface=surface,
        provider=provider,
        credential_reference="secret://fixture",
        input_file_id="file_live",
        counts=BatchCounts(total=2),
        lines=lines,
        created_at=created,
        expires_at=created + timedelta(hours=24),
    )


def _run(client: ProviderBatchClient, job: BatchJob, api_key: str) -> None:
    """Drive one live job to a terminal state and assert per-line results."""

    async def flow() -> None:
        provider_batch_id = await client.submit(job=job, api_key=api_key)
        assert provider_batch_id
        live = job.model_copy(update={"provider_batch_id": provider_batch_id})
        deadline = time.monotonic() + _DEADLINE_SECONDS
        while True:
            snapshot = await client.poll(job=live, api_key=api_key)
            if snapshot.status in TERMINAL_STATUSES:
                assert snapshot.status is BatchStatus.COMPLETED, snapshot
                break
            assert time.monotonic() < deadline, "live batch did not finish in 30 minutes"
            await asyncio.sleep(_POLL_SECONDS)
        results = await client.results(job=live, api_key=api_key)
        by_id = {result.custom_id: result for result in results}
        assert set(by_id) == {"live-0", "live-1"}
        for result in by_id.values():
            assert result.error is None, result
            assert result.output_tokens > 0

    asyncio.run(flow())


@pytest.mark.skipif(
    not (_LIVE and os.environ.get("OPENAI_API_KEY")),
    reason="set EXP_LIVE_BATCH=1 and OPENAI_API_KEY to run the live OpenAI batch",
)
def test_live_openai_chat_batch() -> None:
    """One real two-line gpt-4.1-nano chat batch completes and settles."""
    _run(
        OpenAIBatchClient(),
        _job(
            "openai",
            "/v1/chat/completions",
            "gpt-4.1-nano",
            {"messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 8},
        ),
        os.environ["OPENAI_API_KEY"],
    )


@pytest.mark.skipif(
    not (_LIVE and os.environ.get("ANTHROPIC_API_KEY")),
    reason="set EXP_LIVE_BATCH=1 and ANTHROPIC_API_KEY to run the live Anthropic batch",
)
def test_live_anthropic_messages_batch() -> None:
    """One real two-line haiku message batch completes and settles."""
    _run(
        AnthropicBatchClient(),
        _job(
            "anthropic",
            "/v1/messages",
            "claude-haiku-4-5-20251001",
            {"messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 8},
        ),
        os.environ["ANTHROPIC_API_KEY"],
    )


@pytest.mark.skipif(
    not (_LIVE and os.environ.get("OPENROUTER_API_KEY")),
    reason="set EXP_LIVE_BATCH=1 and OPENROUTER_API_KEY to run the live OpenRouter batch",
)
def test_live_openrouter_chat_batch() -> None:
    """One real two-line batch through OpenRouter's beta batches completes."""
    _run(
        OpenRouterBatchClient(),
        _job(
            "openrouter",
            "/v1/chat/completions",
            "openai/gpt-oss-20b:batch",
            {"messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 8},
        ),
        os.environ["OPENROUTER_API_KEY"],
    )
