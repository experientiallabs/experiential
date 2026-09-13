"""Native deterministic resolution and its parity with the python adapter."""

from __future__ import annotations

import asyncio

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    ToolCall,
)
from exp.runtime.gateway.guardrails.config import engine_from_document
from exp.runtime.gateway.guardrails.contracts import GuardrailRejected
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
    policy = engine.policy_for("org", "identity")
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
    policy = engine.policy_for("org", "identity")
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
    policy = engine.policy_for("org", "identity")
    assert policy is not None
    detectors = compile_native_detectors(engine.deterministic_specifications)
    ticks = iter((0.0, 0.0, 5.0))

    def _clock() -> float:
        """Advance past the authored 500 ms budget during the scan itself."""
        return next(ticks)

    with pytest.raises(GuardrailRejected):
        native_input_request(
            policy,
            detectors,
            _request(content="alice@example.com"),
            monotonic=_clock,
            deadline_monotonic=1e12,
        )


def test_the_native_input_chain_declines_a_hosted_adapter() -> None:
    """A chain the data plane cannot evaluate returns to the python engine."""
    engine = engine_from_document(_adapter(builtin_patterns=["email"]))
    policy = engine.policy_for("org", "identity")
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
    policy = engine.policy_for("org", "identity")
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
    policy = engine.policy_for("org", "identity")
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
    policy = engine.policy_for("org", "identity")
    assert policy is not None
    detectors = compile_native_detectors(engine.deterministic_specifications)
    assert native_output_plan(policy, detectors) is None
