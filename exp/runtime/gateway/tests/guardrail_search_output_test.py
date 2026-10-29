"""Inspect gateway search context and the answer as one complete HTTP output subject."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Literal, cast

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.config import engine_from_document
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailCompletion,
    GuardrailPolicy,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.tests.guardrail_policy_integrity_test import _body, _provider
from exp.runtime.gateway.tests.mandatory_guardrails_paths_test import _serving
from exp.runtime.gateway.tests.native_tool_search_test import _configure, _Provider
from exp.runtime.gateway.tests.native_waterfall_test import (
    _content_chunk,
    _sse_frame,
    _terminal_frames,
)
from exp.runtime.gateway.tests.web_search_backend_fixture_test import StaticWebSearchBackend
from exp.runtime.gateway.web_search.contracts import GatewayWebSearchResult

_SEARCH_MARKER = "search-context-marker"
_ANSWER = "answer-marker [1]"
_Ending = Literal["completed", "refusal", "failed"]


class _ContextClassifier:
    """Reject an answer only when its accompanying search context is present.

    Attributes:
        marker: Search-context text required before an answer is flagged.
        completions: Complete subjects actually delivered to the classifier.
    """

    def __init__(self, marker: str = _SEARCH_MARKER) -> None:
        """Bind the contextual condition without rejecting either component alone."""
        self.marker = marker
        self.completions: list[GuardrailCompletion] = []

    async def inspect_input(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Permit the prompt and search expansions so only output decides the result."""
        del request, check
        return ClassifierVerdict(flagged=False)

    async def inspect_output(
        self, *, completion: GuardrailCompletion, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Reject the relation between an answer and its returned search context."""
        del check
        self.completions.append(completion)
        return ClassifierVerdict(
            flagged=bool(completion.text or completion.refusal)
            and any(self.marker in item for item in completion.context)
        )


def _context_engine(
    classifier: _ContextClassifier,
    *,
    max_response_bytes: int = 1_048_576,
    action: GuardrailAction = GuardrailAction.BLOCK,
) -> GuardrailEngine:
    """Bind one model-style output check to the normal request-owned engine."""
    return GuardrailEngine(
        store=MappingGuardrailStore(
            (
                GuardrailPolicy(
                    policy_id="complete-output",
                    protected=True,
                    max_response_bytes=max_response_bytes,
                    checks=(
                        GuardrailCheck(
                            check_id="context",
                            stage=GuardrailCheckStage.OUTPUT,
                            action=action,
                            capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                            adapter_id="context",
                            timeout_ms=1000,
                        ),
                    ),
                ),
            )
        ),
        client=DirectClassifierClient(ClassifierRegistry({"context": classifier})),
        monotonic=time.monotonic,
    )


def _native_engine(action: Literal["block", "modify"], limit: int) -> GuardrailEngine:
    """Use a native detector whose modify action supports incremental redaction."""
    return engine_from_document(
        {
            "adapters": [{"adapter_id": "email", "kind": "regex", "builtin_patterns": ["email"]}],
            "policies": [
                {
                    "policy_id": "complete-output",
                    "protected": True,
                    "max_response_bytes": limit,
                    "checks": [
                        {
                            "check_id": "email",
                            "stage": "output",
                            "action": action,
                            "capability": "pii",
                            "adapter_id": "email",
                            "timeout_ms": 1000,
                        }
                    ],
                }
            ],
        }
    )


def _frames(ending: _Ending, answer: str) -> bytes:
    """Return a completed answer or a withheld refusal ending in a typed failure."""
    if ending == "completed":
        return _content_chunk(answer) + _terminal_frames()
    refusal = _sse_frame({"choices": [{"index": 0, "delta": {"refusal": answer}}]})
    if ending == "refusal":
        terminal = _sse_frame(
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}]}
        )
    else:
        terminal = _sse_frame(
            {"error": {"type": "invalid_request_error", "message": "Synthetic failure"}}
        )
    return refusal + terminal + b"data: [DONE]\n\n"


def _web_response(
    root: Path,
    engine: GuardrailEngine,
    surface: str,
    stream: bool,
    *,
    ending: _Ending = "completed",
    answer: str = _ANSWER,
) -> tuple[httpx.Response, Path]:
    """Serve one searched request through the real native HTTP endpoint."""
    with _provider(_frames(ending, answer)) as (base_url, received):
        key = _configure(root, base_url)
        manager = GatewayManagement(root)
        alias = manager.aliases()[0]
        manager.activate_direct_alias(
            alias_id="coding",
            alias_name="coding",
            revision_id="refusal-enabled",
            pool_id="coding",
            snapshot_ref=str(alias.snapshot_ref),
            catalog_sha256=str(alias.catalog_sha256),
            refusal_failover=ending != "refusal",
        )
        components = load_gateway_components(root, environment={"TEST_PROVIDER_KEY": "fixture"})
        search = StaticWebSearchBackend(
            (GatewayWebSearchResult(url="https://synthetic.invalid/result", title=_SEARCH_MARKER),)
        )
        control = NativeControlPlane(components, guardrails=engine, web_search=search)
        route, body = _body(surface, stream)
        body["model"] = "coding:online"
        if surface == "messages":
            body["model"] = "coding"
            body["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
        with _serving(control) as url:
            response = httpx.post(
                url + route, headers={"authorization": f"Bearer {key}"}, json=body, timeout=10
            )
        assert len(received) == 1
        return response, components.ledger.database_path


def _assert_guardrail_settlement(database_path: Path, *, attempts: int = 1) -> None:
    """Require a terminal guardrail failure without an unfinished accepted request."""
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "select state, failure_class from gateway_attempts order by rowid"
        ).fetchall()
        assert len(rows) == attempts
        assert rows[-1] == ("failed", "guardrail")
        assert connection.execute(
            "select terminal_state, terminal_at is not null from gateway_requests"
        ).fetchall() == [("failed", 1)]


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("ending", ["completed", "refusal", "failed"])
def test_web_search_and_answer_are_one_output_subject(
    tmp_path: Path, surface: str, stream: bool, ending: _Ending
) -> None:
    """Contextual policies block the entire answer before any search or answer bytes."""
    classifier = _ContextClassifier()
    response, database_path = _web_response(
        tmp_path, _context_engine(classifier), surface, stream, ending=ending
    )
    assert response.status_code == 400, response.text
    assert "answer-marker" not in response.text
    assert _SEARCH_MARKER not in response.text
    assert any(
        (completion.text or completion.refusal)
        and any(_SEARCH_MARKER in item for item in completion.context)
        for completion in classifier.completions
    )
    _assert_guardrail_settlement(database_path)


def _assert_web_rendered_once(response: httpx.Response, surface: str, stream: bool) -> None:
    """Check one logical search result and the exact answer in the public protocol."""
    documents = (
        [
            cast(JsonObject, json.loads(line[6:]))
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        if stream
        else [cast(JsonObject, response.json())]
    )
    if surface == "messages":
        if stream:
            blocks = [
                document["content_block"]
                for document in documents
                if document.get("type") == "content_block_start"
            ]
            text = "".join(
                str(delta["text"])
                for document in documents
                if isinstance(delta := document.get("delta"), dict) and "text" in delta
            )
        else:
            blocks = documents[0]["content"]
            assert isinstance(blocks, list)
            text = "".join(
                str(block["text"])
                for block in blocks
                if isinstance(block, dict) and block.get("type") == "text"
            )
        assert text == _ANSWER
        assert (
            sum(
                isinstance(block, dict) and block.get("type") == "server_tool_use"
                for block in blocks
            )
            == 1
        )
        results = [
            block
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "web_search_tool_result"
        ]
        assert len(results) == 1
        assert json.dumps(results).count(_SEARCH_MARKER) == 1
    elif surface == "responses":
        document = documents[0]
        if stream:
            terminal = next(item for item in documents if item.get("type") == "response.completed")
            document = cast(JsonObject, terminal["response"])
        output = document["output"]
        assert isinstance(output, list)
        assert json.dumps(output).count(_SEARCH_MARKER) == 1
        assert json.dumps(output).count(_ANSWER) == 1
    else:
        messages: list[JsonObject] = []
        for document in documents:
            choices = document.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            assert isinstance(choice, dict)
            message = choice["delta" if stream else "message"]
            assert isinstance(message, dict)
            messages.append(cast(JsonObject, message))
        assert "".join(str(message.get("content") or "") for message in messages) == _ANSWER
        assert json.dumps(messages).count(_SEARCH_MARKER) == 1


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_allowed_complete_output_is_checked_and_rendered_once(
    tmp_path: Path, surface: str, stream: bool
) -> None:
    """An allowed answer retains one search rendering after one complete inspection."""
    classifier = _ContextClassifier("absent marker")
    response, _database_path = _web_response(tmp_path, _context_engine(classifier), surface, stream)
    assert response.status_code == 200, response.text
    assert len(classifier.completions) == 1
    assert classifier.completions[0].text == _ANSWER
    assert any(_SEARCH_MARKER in item for item in classifier.completions[0].context)
    _assert_web_rendered_once(response, surface, stream)


class _ModifyingContextClassifier(_ContextClassifier):
    """Request a text-only replacement when search context and answer match together."""

    async def inspect_output(
        self, *, completion: GuardrailCompletion, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Require removal of generated context that protocol encoders would reinsert."""
        verdict = await super().inspect_output(completion=completion, check=check)
        return ClassifierVerdict(flagged=verdict.flagged, replacement_text="Sanitized answer.")


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_generated_search_context_cannot_return_after_output_modification(
    tmp_path: Path, surface: str, stream: bool
) -> None:
    """A text-only rewrite cannot silently preserve or reinsert rejected generated metadata."""
    classifier = _ModifyingContextClassifier()
    response, database_path = _web_response(
        tmp_path,
        _context_engine(classifier, action=GuardrailAction.MODIFY),
        surface,
        stream,
    )
    assert response.status_code == 400, response.text
    assert "answer-marker" not in response.text
    assert _SEARCH_MARKER not in response.text
    assert "Sanitized answer." not in response.text
    _assert_guardrail_settlement(database_path)


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("strategy", ["async", "native-block", "native-modify"])
def test_output_bound_includes_search_context_and_answer(
    tmp_path: Path, surface: str, stream: bool, strategy: str
) -> None:
    """Each component fits the cap alone, but their complete output must be refused."""
    limit = 900
    engine = (
        _context_engine(_ContextClassifier("absent marker"), max_response_bytes=limit)
        if strategy == "async"
        else _native_engine("modify" if strategy == "native-modify" else "block", limit)
    )
    response, database_path = _web_response(
        tmp_path, engine, surface, stream, answer="x " * 350 + "[1]"
    )
    assert response.status_code == 400, response.text
    assert "x x x" not in response.text
    assert _SEARCH_MARKER not in response.text
    _assert_guardrail_settlement(database_path)


def _tool_body(surface: str, stream: bool) -> tuple[str, JsonObject]:
    """Request one deferred weather tool on a surface that renders search results."""
    route, body = _body(surface, stream)
    if surface == "responses":
        body["tools"] = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Current weather for a city",
                "parameters": {"type": "object"},
                "defer_loading": True,
            },
            {"type": "tool_search"},
        ]
    else:
        body["tools"] = [
            {
                "name": "get_weather",
                "description": "Current weather for a city",
                "input_schema": {"type": "object"},
                "defer_loading": True,
            },
            {"type": "tool_search_tool_bm25", "name": "tool_search_tool_bm25"},
        ]
    return route, body


@pytest.mark.parametrize("surface", ["responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_tool_search_result_and_answer_are_one_output_subject(
    tmp_path: Path, surface: str, stream: bool
) -> None:
    """The matched tool context and final answer reach a contextual check together."""
    provider = _Provider()
    classifier = _ContextClassifier("get_weather")
    try:
        key = _configure(tmp_path, f"http://127.0.0.1:{provider.server.server_port}/v1")
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        control = NativeControlPlane(components, guardrails=_context_engine(classifier))
        route, body = _tool_body(surface, stream)
        with _serving(control) as url:
            response = httpx.post(
                url + route, headers={"authorization": f"Bearer {key}"}, json=body, timeout=10
            )
        assert response.status_code == 400, response.text
        assert "get_weather" not in response.text
        assert "18C in Bern" not in response.text
        assert len(provider.requests) == 2
        assert any(
            completion.text and any("get_weather" in item for item in completion.context)
            for completion in classifier.completions
        )
        _assert_guardrail_settlement(components.ledger.database_path, attempts=2)
    finally:
        provider.close()
