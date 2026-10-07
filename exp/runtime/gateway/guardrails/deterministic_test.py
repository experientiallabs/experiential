"""Native deterministic resolution and its parity with the python adapter."""

from __future__ import annotations

import asyncio

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
    ToolCall,
)
from exp.runtime.gateway.guardrails import text_coverage
from exp.runtime.gateway.guardrails.config import engine_from_document
from exp.runtime.gateway.guardrails.contracts import (
    GuardrailAction,
    GuardrailRejected,
    request_content_bytes,
)
from exp.runtime.gateway.guardrails.deterministic import (
    compile_native_detectors,
    native_input_request,
    native_output_plan,
)
from exp.runtime.gateway.guardrails.regex import (
    BuiltinPattern,
    RegexAdapterDocument,
    RegexClassifier,
)
from exp.runtime.gateway.guardrails.text_coverage_test import (
    SUPPORTED_TEXT_CONTEXTS,
    UNSUPPORTED_TEXT_INPUTS,
    text_context_request,
    tool_error_request,
    unsupported_text_request,
)
from exp.runtime.models.providers.wire_messages import anthropic_blocks, responses_items


@pytest.mark.parametrize("execution", ["async", "native"])
@pytest.mark.parametrize("field", ["input_examples", "allowed_callers"])
@pytest.mark.parametrize(
    "payload", ["x" * 8192, "\x00" * 1024, "\U0001f642" * 1536], ids=["ascii", "escaped", "utf8"]
)
@pytest.mark.parametrize("protected", [True, False])
def test_private_tool_context_is_bounded_before_local_projection(
    execution: str,
    field: str,
    payload: str,
    protected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Private examples and callers cannot bypass authored bounds in either execution path."""
    tool = GatewayToolDefinition(name="lookup", parameters={})
    tool = tool.model_copy(
        update={field: ({"value": payload},) if field == "input_examples" else (payload,)}
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="benign"),),
        tools=(tool,),
    )
    assert request_content_bytes(request) < 4096
    engine = engine_from_document(_adapter(builtin_patterns=["email"]))
    policy = engine.policies_for("org", "identity")[0].model_copy(
        update={"protected": protected, "max_request_bytes": 4096}
    )
    detectors = compile_native_detectors(engine.deterministic_specifications)
    projected: list[bool] = []

    def forbidden_projection(_context: JsonObject) -> bytes:
        """Expose any scanner allocation attempted before the shared admission bound."""
        projected.append(True)
        raise AssertionError("oversized context reached local projection")

    monkeypatch.setattr(text_coverage, "canonical_json_bytes", forbidden_projection)
    try:
        with pytest.raises(GuardrailRejected) as rejected:
            if execution == "async":
                asyncio.run(
                    engine.enforce_input(policy=policy, request=request, deadline_monotonic=1e12)
                )
            else:
                native_input_request(
                    policy, detectors, request, monotonic=lambda: 0.0, deadline_monotonic=1e12
                )
        assert rejected.value.failure.failure_class is GatewayFailureClass.UNSUPPORTED_CAPABILITY
        assert not rejected.value.failure.retryable_same_deployment
        assert not rejected.value.failure.failover_eligible
        assert not projected
    finally:
        engine.close(timeout_seconds=2)


@pytest.mark.parametrize("content", ["", "benign"])
@pytest.mark.parametrize(
    "action", [GuardrailAction.ALLOW, GuardrailAction.BLOCK, GuardrailAction.MODIFY]
)
@pytest.mark.parametrize("protected", [True, False])
def test_native_and_python_match_exact_tool_error_wire_text(
    content: str, action: GuardrailAction, protected: bool
) -> None:
    """Anchored rules see the provider prefix; modify cannot erase the underlying error flag."""
    engine = engine_from_document(_adapter(patterns=[rf"^\[tool error\] {content}$"]))
    original_policy = engine.policies_for("org", "identity")[0]
    policy = original_policy.model_copy(
        update={
            "protected": protected,
            "checks": tuple(
                check.model_copy(update={"action": action}) for check in original_policy.checks
            ),
        }
    )
    detectors = compile_native_detectors(engine.deterministic_specifications)
    try:
        for failed in (False, True):
            request = tool_error_request(failed=failed, content=content)
            original = request.model_copy(deep=True)
            wire = responses_items(request.messages[0])
            assert wire[0]["output"] == (f"[tool error] {content}" if failed else content)
            if failed and action is not GuardrailAction.ALLOW:
                with pytest.raises(GuardrailRejected) as python:
                    asyncio.run(
                        engine.enforce_input(
                            policy=policy, request=request, deadline_monotonic=1e12
                        )
                    )
                with pytest.raises(GuardrailRejected) as native:
                    native_input_request(
                        policy, detectors, request, monotonic=lambda: 0.0, deadline_monotonic=1e12
                    )
                assert native.value.failure == python.value.failure
                assert native.value.failure.failure_class.value == "guardrail"
            else:
                assert (
                    asyncio.run(
                        engine.enforce_input(
                            policy=policy, request=request, deadline_monotonic=1e12
                        )
                    )
                    is request
                )
                assert (
                    native_input_request(
                        policy, detectors, request, monotonic=lambda: 0.0, deadline_monotonic=1e12
                    )
                    is request
                )
            assert request == original
            assert responses_items(request.messages[0]) == wire
    finally:
        engine.close(timeout_seconds=2)


@pytest.mark.parametrize("field", UNSUPPORTED_TEXT_INPUTS)
@pytest.mark.parametrize("protected", [True, False])
def test_native_and_python_input_refuse_incomplete_coverage(field: str, protected: bool) -> None:
    """The native fast path cannot turn a Python coverage failure into a partial allow."""
    engine = engine_from_document(_adapter(builtin_patterns=["email"]))
    policy = engine.policies_for("org", "identity")[0].model_copy(update={"protected": protected})
    detectors = compile_native_detectors(engine.deterministic_specifications)
    request = unsupported_text_request(field)
    with pytest.raises(GuardrailRejected) as python:
        asyncio.run(engine.enforce_input(policy=policy, request=request, deadline_monotonic=1e12))
    with pytest.raises(GuardrailRejected) as native:
        native_input_request(
            policy, detectors, request, monotonic=lambda: 0.0, deadline_monotonic=1e12
        )
    assert native.value.failure == python.value.failure
    assert native.value.failure.failure_class.value == "unsupported_capability"


@pytest.mark.parametrize("field", SUPPORTED_TEXT_CONTEXTS)
def test_native_and_python_context_matches_preserve_immutable_constraints(field: str) -> None:
    """Native and Python regex agree on benign context and refuse matched constraints."""
    engine = engine_from_document(_adapter(builtin_patterns=["email"]))
    policy = engine.policies_for("org", "identity")[0]
    detectors = compile_native_detectors(engine.deterministic_specifications)
    benign = text_context_request(field, "benign")
    assert (
        asyncio.run(engine.enforce_input(policy=policy, request=benign, deadline_monotonic=1e12))
        is benign
    )
    assert (
        native_input_request(
            policy, detectors, benign, monotonic=lambda: 0.0, deadline_monotonic=1e12
        )
        is benign
    )
    request = text_context_request(field, "alice@example.com")
    with pytest.raises(GuardrailRejected) as python:
        asyncio.run(engine.enforce_input(policy=policy, request=request, deadline_monotonic=1e12))
    with pytest.raises(GuardrailRejected) as native:
        native_input_request(
            policy, detectors, request, monotonic=lambda: 0.0, deadline_monotonic=1e12
        )
    assert native.value.failure == python.value.failure


@pytest.mark.parametrize("protected", [True, False])
def test_matched_cache_carried_text_cannot_escape_a_modify_rule(protected: bool) -> None:
    """Refuse rewriting only the canonical text while the provider replays its original blocks."""
    engine = engine_from_document(_adapter(builtin_patterns=["email"]))
    policy = engine.policies_for("org", "identity")[0].model_copy(update={"protected": protected})
    detectors = compile_native_detectors(engine.deterministic_specifications)
    dispatched: list[tuple[str, list[JsonObject]]] = []
    for text in ("benign", "alice@example.com"):
        request = GatewayRequest(
            surface=GatewayApiSurface.MESSAGES,
            messages=(
                GatewayMessage(
                    role="user",
                    content=text,
                    provider_text_blocks=(
                        {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}},
                    ),
                ),
            ),
        )
        if text == "benign":
            python_result = asyncio.run(
                engine.enforce_input(policy=policy, request=request, deadline_monotonic=1e12)
            )
            native_result = native_input_request(
                policy, detectors, request, monotonic=lambda: 0.0, deadline_monotonic=1e12
            )
            assert python_result is native_result is request
            assert anthropic_blocks(request.messages[0]) == (
                "user",
                [{"type": "text", "text": "benign", "cache_control": {"type": "ephemeral"}}],
            )
            continue
        with pytest.raises(GuardrailRejected) as python:
            python_result = asyncio.run(
                engine.enforce_input(policy=policy, request=request, deadline_monotonic=1e12)
            )
            dispatched.append(anthropic_blocks(python_result.messages[0]))
        with pytest.raises(GuardrailRejected) as native:
            native_result = native_input_request(
                policy, detectors, request, monotonic=lambda: 0.0, deadline_monotonic=1e12
            )
            assert native_result is not None
            dispatched.append(anthropic_blocks(native_result.messages[0]))
        assert native.value.failure == python.value.failure
        assert native.value.failure.failure_class.value == "guardrail"
        assert dispatched == []


_CORPUS = (
    "",
    "No personal information",
    "Contact alice@example.com today",
    "日本語 😀 alice@example.com fin",
    "alice@example.com and bob@example.org",
    "naïve 🙂 ada@example.com 🙂 ok",
    "Cards: 4111111111111111 5555555555554444",
    "4111 1111 1111 1111",
    "4111-1111-1111-1111",
    "4111111111111111",
    "4111111111111111 5555555555554444",
    "4111 1111 1111 1111 5555 5555 5555 4444",
    "4111-1111-1111-1111 378282246310005",
    "4111111111111111, 5555555555554444",
    "4111111111111112",
    "0000000000000000",
    "141111111111111111111",
    "4111 1111 1111 1111 0000",
    "41111111111111111111",
    "token: sk-proj-" + "a" * 40,
    "token: ghp_" + "b" * 36,
    "token: xpl_" + "c" * 32,
    "token: AKIAABCDEFGHIJKLMNOP tail",
    "token: github_pat_" + "d" * 30,
    "id-42 and id-7 and id-٤٢",
    "xabcdey",
    "a" * 500 + "@" + "example.com",
    "a b",
    "a\tb",
    "a\fb",
    "a\rb",
    "a\vb",
)


def _adapter(**overrides: object) -> JsonObject:
    """Author one regex adapter bound to both stages of a protected identity."""
    return {
        "adapters": [{"kind": "regex", "adapter_id": "patterns", **overrides}],
        "policies": [
            {
                "policy_id": "redact",
                "organization_id": "org",
                "identity_id": "identity",
                "protected": True,
                "checks": [
                    {
                        "check_id": stage,
                        "capability": "pii",
                        "stage": stage,
                        "action": "modify",
                        "adapter_id": "patterns",
                        "timeout_ms": 500,
                    }
                    for stage in ("input", "output")
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    "document",
    [
        RegexAdapterDocument(
            adapter_id="builtins",
            builtin_patterns=(
                BuiltinPattern.EMAIL,
                BuiltinPattern.CREDIT_CARD,
                BuiltinPattern.API_KEY,
            ),
        ),
        RegexAdapterDocument(adapter_id="cards", builtin_patterns=(BuiltinPattern.CREDIT_CARD,)),
        RegexAdapterDocument(adapter_id="custom", patterns=("abc", "cde"), replacement=r"\1"),
        RegexAdapterDocument(adapter_id="ascii", patterns=(r"\bid-\d+\b",), replacement="[ID]"),
        RegexAdapterDocument(adapter_id="space", patterns=(r"a\sb",), replacement="[WS]"),
        RegexAdapterDocument(
            adapter_id="mixed",
            patterns=(r"internal-[0-9]+", r"\w+@corp\.test"),
            builtin_patterns=(BuiltinPattern.EMAIL,),
            replacement="*",
        ),
    ],
    ids=["builtins", "cards", "literal_replacement", "ascii_classes", "whitespace", "mixed"],
)
def test_native_detector_output_matches_the_python_adapter(
    document: RegexAdapterDocument,
) -> None:
    """The same corpus redacts identically through RE2 and the native rule.

    This is the anti-drift gate: the python adapter is the specification, so
    any behavior change on either side has to be made on both.
    """
    classifier = RegexClassifier(document)
    detectors = compile_native_detectors({document.adapter_id: classifier.native_specification()})
    detector = detectors[document.adapter_id]
    for subject in _CORPUS:
        flagged, expected = classifier._redact(subject)
        rewritten = detector.redact(subject)
        assert detector.matches(subject) is flagged
        assert (expected if flagged else None) == rewritten


def test_native_detector_shares_the_python_bounds() -> None:
    """Both implementations fail closed on the same subject and match ceilings."""
    classifier = RegexClassifier(RegexAdapterDocument(adapter_id="dense", patterns=("a",)))
    detector = compile_native_detectors({"dense": classifier.native_specification()})["dense"]
    for subject in ("a" * 5000, "a" * (1_048_576 + 1)):
        with pytest.raises(ValueError):
            classifier._redact(subject)
        with pytest.raises(ValueError):
            detector.redact(subject)


def _request(*, content: str = "", tool_email: bool = False) -> GatewayRequest:
    """Build one chat request carrying the subject as user content."""
    calls = (
        (ToolCall(call_id="call", name="lookup", arguments={"email": "alice@example.com"}),)
        if tool_email
        else ()
    )
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(
                role="assistant" if tool_email else "user",
                content=content,
                tool_calls=calls,
            ),
        ),
    )


def test_the_native_input_chain_matches_the_python_engine() -> None:
    """Inline deterministic admission rewrites the request exactly as the engine does."""
    engine = engine_from_document(_adapter(builtin_patterns=["email", "credit_card", "api_key"]))
    policy = engine.policies_for("org", "identity")[0]
    assert policy is not None
    detectors = compile_native_detectors(engine.deterministic_specifications)
    for subject in _CORPUS:
        request = _request(content=subject)
        expected = asyncio.run(
            engine.enforce_input(policy=policy, request=request, deadline_monotonic=1e12)
        )
        native = native_input_request(
            policy,
            detectors,
            request,
            monotonic=lambda: 0.0,
            deadline_monotonic=1e12,
        )
        assert native == expected


def test_the_native_input_chain_refuses_matched_tool_arguments() -> None:
    """A tool argument match is refused inline, never rewritten."""
    engine = engine_from_document(_adapter(builtin_patterns=["email"]))
    policy = engine.policies_for("org", "identity")[0]
    assert policy is not None
    detectors = compile_native_detectors(engine.deterministic_specifications)
    request = _request(tool_email=True)
    with pytest.raises(GuardrailRejected):
        asyncio.run(engine.enforce_input(policy=policy, request=request, deadline_monotonic=1e12))
    with pytest.raises(GuardrailRejected):
        native_input_request(
            policy,
            detectors,
            request,
            monotonic=lambda: 0.0,
            deadline_monotonic=1e12,
        )


def test_the_native_input_chain_fails_closed_on_an_overrun_scan() -> None:
    """A scan that finishes past its authored budget never returns a verdict."""
    engine = engine_from_document(_adapter(builtin_patterns=["email"]))
    policy = engine.policies_for("org", "identity")[0]
    assert policy is not None
    detectors = compile_native_detectors(engine.deterministic_specifications)
    ticks = iter((0.0, 0.0, 5.0))

    def _clock() -> float:
        """Advance past the authored 500 ms budget during the scan itself."""
        return next(ticks)

    with pytest.raises(GuardrailRejected) as exc:
        native_input_request(
            policy,
            detectors,
            _request(content="alice@example.com"),
            monotonic=_clock,
            deadline_monotonic=1e12,
        )
    assert exc.value.failure.failure_class.value == "unavailable"


def test_the_native_input_chain_declines_a_hosted_adapter() -> None:
    """A chain the data plane cannot evaluate returns to the python engine."""
    engine = engine_from_document(_adapter(builtin_patterns=["email"]))
    policy = engine.policies_for("org", "identity")[0]
    assert policy is not None
    assert (
        native_input_request(
            policy,
            {},
            _request(content="alice@example.com"),
            monotonic=lambda: 0.0,
            deadline_monotonic=1e12,
        )
        is None
    )


def test_a_deterministic_policy_resolves_into_an_admission_plan() -> None:
    """A regex-only output chain is handed to the data plane in authored order."""
    engine = engine_from_document(_adapter(builtin_patterns=["email"]))
    policy = engine.policies_for("org", "identity")[0]
    assert policy is not None
    detectors = compile_native_detectors(engine.deterministic_specifications)
    plan = native_output_plan(policy, detectors)
    assert plan == {
        "protected": True,
        "max_response_bytes": policy.max_response_bytes,
        "policy_id": "redact",
        "organization_id": "org",
        "identity_id": "identity",
        "checks": [
            {
                "action": "modify",
                "adapter_id": "patterns",
                "check_id": "output",
                "capability": "pii",
                "timeout_ms": 500,
            }
        ],
    }


def test_a_nondeterministic_chain_keeps_the_python_boundary() -> None:
    """One hosted adapter in the chain sends the whole stage back to python."""
    document = _adapter(builtin_patterns=["email"])
    adapters = document["adapters"]
    policies = document["policies"]
    assert isinstance(adapters, list)
    assert isinstance(policies, list)
    adapters.append(
        {"kind": "http_json", "adapter_id": "hosted", "url": "https://classifier.test/inspect"}
    )
    policy_document = policies[0]
    assert isinstance(policy_document, dict)
    checks = policy_document["checks"]
    assert isinstance(checks, list)
    checks.append(
        {
            "check_id": "hosted-output",
            "capability": "pii",
            "stage": "output",
            "action": "block",
            "adapter_id": "hosted",
            "timeout_ms": 500,
        }
    )
    engine = engine_from_document(document)
    policy = engine.policies_for("org", "identity")[0]
    assert policy is not None
    detectors = compile_native_detectors(engine.deterministic_specifications)
    assert native_output_plan(policy, detectors) is None


def test_an_unguarded_policy_has_no_plan() -> None:
    """An input-only policy leaves the buffered output path untouched."""
    document = _adapter(builtin_patterns=["email"])
    policies = document["policies"]
    assert isinstance(policies, list)
    policy_document = policies[0]
    assert isinstance(policy_document, dict)
    checks = policy_document["checks"]
    assert isinstance(checks, list)
    policy_document["checks"] = [check for check in checks if check["stage"] == "input"]
    engine = engine_from_document(document)
    policy = engine.policies_for("org", "identity")[0]
    assert policy is not None
    detectors = compile_native_detectors(engine.deterministic_specifications)
    assert native_output_plan(policy, detectors) is None


@pytest.mark.parametrize("pattern", ["[a&&b]", "[a~~b]", "[a[b]", "[a-z&&b]"])
def test_native_character_classes_preserve_re2_literal_members(pattern: str) -> None:
    """Rust set operators must not narrow the authored RE2 match set."""
    classifier = RegexClassifier(RegexAdapterDocument(adapter_id="classes", patterns=(pattern,)))
    detector = compile_native_detectors({"classes": classifier.native_specification()})["classes"]
    for subject in ("a", "b", "&", "~", "[", "abc", "z", "secret & token"):
        flagged, expected = classifier._redact(subject)
        assert detector.redact(subject) == (expected if flagged else None)
