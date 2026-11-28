"""A completed upstream cannot freeze optional input evidence before its decision."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayEvent, GatewayFailure, GatewayServiceTierSettlement
from exp.runtime.gateway.group_commit import SyncGroupCommitLedger
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.guardrails.store_test import _authorization
from exp.runtime.gateway.tests import parallel_input_guardrails_test as parallel


def _collect_terminal_evidence(monkeypatch: pytest.MonkeyPatch) -> list[JsonObject]:
    """Declare the accounting extension while preserving the real SQLite terminal write."""
    finish = SyncGroupCommitLedger.finish_attempt
    documents: list[JsonObject] = []

    def write(
        self: SyncGroupCommitLedger,
        *,
        attempt_id: str,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
        first_token_at: datetime | None = None,
        retry_after_seconds: int | None = None,
        ratelimit_limit_requests: int | None = None,
        ratelimit_remaining_requests: int | None = None,
        ratelimit_limit_tokens: int | None = None,
        ratelimit_remaining_tokens: int | None = None,
        upstream_provider: str | None = None,
        web_search_requests: int = 0,
        tool_search_requests: int = 0,
        service_tier: GatewayServiceTierSettlement | None = None,
        input_guardrail_outcomes: tuple[JsonObject, ...] = (),
    ) -> None:
        """Keep the observation alongside a successful durable financial commit."""
        finish(
            self,
            attempt_id=attempt_id,
            terminal_event=terminal_event,
            failure=failure,
            finalize_request=finalize_request,
            first_token_at=first_token_at,
            retry_after_seconds=retry_after_seconds,
            ratelimit_limit_requests=ratelimit_limit_requests,
            ratelimit_remaining_requests=ratelimit_remaining_requests,
            ratelimit_limit_tokens=ratelimit_limit_tokens,
            ratelimit_remaining_tokens=ratelimit_remaining_tokens,
            upstream_provider=upstream_provider,
            web_search_requests=web_search_requests,
            tool_search_requests=tool_search_requests,
            service_tier=service_tier,
        )
        documents.extend(input_guardrail_outcomes)

    monkeypatch.setattr(SyncGroupCommitLedger, "finish_attempt", write)
    return documents


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("outcome", ["allow", "block"])
def test_completed_provider_waits_for_optional_input_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
    streaming: bool,
    outcome: str,
) -> None:
    """The real native server records a timely decision after an upstream has finished."""
    records = _collect_terminal_evidence(monkeypatch)
    build = parallel._engine  # noqa: SLF001 - reuse the real HTTP fixture's policy.

    def optional(classifier: parallel._Classifier, timeout_ms: int = 3000) -> GuardrailEngine:
        """Use the optional native path with ample time for the held classifier result."""
        template = build(classifier, timeout_ms=timeout_ms)
        policy = template.policies_for(_authorization())[0].model_copy(
            update={
                "fail_open": True,
                "input_max_wait_ms": 500,
                "organization_id": None,
                "identity_id": None,
            }
        )
        template.close()
        return GuardrailEngine(
            store=MappingGuardrailStore((policy,)),
            client=DirectClassifierClient(ClassifierRegistry({"criminal": classifier})),
            monotonic=time.monotonic,
        )

    monkeypatch.setattr(parallel, "_engine", optional)
    parallel.test_parallel_inspection_holds_text_tools_and_customer_settlement(
        tmp_path,
        surface=surface,
        streaming=streaming,
        outcome=outcome,
        tool_call=False,
        provider_phase="completed",
    )
    (record,) = records
    assert record["outcome"] == outcome
    assert record["recovery_reason"] is None
