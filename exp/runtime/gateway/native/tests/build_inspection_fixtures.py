"""Generate independent policy ordering, streaming and Unicode casefold reference data."""

from __future__ import annotations

import json
import time
from pathlib import Path

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry, KeywordClassifier
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.contracts import GuardrailPolicy
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.native import (
    enforce_native_output,
    enforce_native_output_segment,
)
from exp.runtime.gateway.guardrails.native_owner import output_registration
from exp.runtime.gateway.guardrails.regex import (
    BuiltinPattern,
    RegexAdapterDocument,
    RegexClassifier,
)
from exp.runtime.gateway.guardrails.session import GuardrailSession
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore


def _engine() -> GuardrailEngine:
    registry = ClassifierRegistry(
        {
            "email": RegexClassifier(
                RegexAdapterDocument(adapter_id="email", builtin_patterns=(BuiltinPattern.EMAIL,))
            ),
            "secret": RegexClassifier(
                RegexAdapterDocument(
                    adapter_id="secret", patterns=("secret",), replacement="replacement"
                )
            ),
            "replacement": RegexClassifier(
                RegexAdapterDocument(
                    adapter_id="replacement", patterns=("replacement",), replacement="secret"
                )
            ),
            "keyword": KeywordClassifier(("secret", "strasse", "i̇")),
        }
    )
    return GuardrailEngine(
        store=MappingGuardrailStore(),
        client=DirectClassifierClient(registry),
        monotonic=time.monotonic,
    )


def _policy(index: int, adapter: str, action: str, protected: bool, limit: int) -> GuardrailPolicy:
    return GuardrailPolicy.model_validate(
        {
            "policy_id": f"policy-{index}",
            "organization_id": "org",
            "identity_id": "identity",
            "protected": protected,
            "max_response_bytes": limit,
            "checks": [
                {
                    "check_id": f"check-{index}",
                    "adapter_id": adapter,
                    "stage": "output",
                    "capability": "pii",
                    "action": action,
                    "timeout_ms": 1000,
                }
            ],
        }
    )


def output_cases(engine: GuardrailEngine) -> list[JsonObject]:
    """Exercise final-subject validation and complete output inspection bounds."""
    cases: list[JsonObject] = []
    for index in range(320):
        action = ["allow", "block", "error", "modify"][index % 4]
        adapter = ["secret", "email", "keyword"][(index // 4) % 3]
        policies = (
            _policy(0, adapter, action, bool(index % 2), 80 if index % 17 == 0 else 1_048_576),
        )
        if index % 5 == 0:
            policies += (_policy(1, "replacement", "modify", bool(index % 3), 1_048_576),)
        session = GuardrailSession(engine, policies, time.monotonic() + 30)
        data: JsonObject = {
            "text": ["", "secret", "user@example.test", "Straße", "İ", "ordinary 世界"][
                (index // 2) % 6
            ],
            "context": ["secret"] if index % 7 == 0 else [],
            "refusal": index % 11 == 0,
            "tool_calls": [{"call_id": "call", "name": "lookup", "arguments": '{"secret":1}'}]
            if index % 13 == 0
            else [],
        }
        cases.append(
            {
                "checks": output_registration(session),
                "input": data,
                "expected": json.loads(enforce_native_output(session, json.dumps(data))),
            }
        )
    return cases


def stream_cases(engine: GuardrailEngine) -> list[JsonObject]:
    """Check every split through a builtin match, including long withheld tails."""
    policy = _policy(0, "email", "modify", True, 1_048_576)
    session = GuardrailSession(engine, (policy,), time.monotonic() + 30)
    cases: list[JsonObject] = []
    for text in [
        "before user@example.test after",
        "世界 user@example.test",
        "a" * 550 + "@example.test",
    ]:
        for cut in range(len(text) + 1):
            data: JsonObject = {
                "pending": text[:cut],
                "final": cut == len(text),
                "settled_bytes": 0,
            }
            cases.append(
                {
                    "checks": output_registration(session),
                    "input": data,
                    "expected": json.loads(
                        enforce_native_output_segment(session, json.dumps(data))
                    ),
                }
            )
    return cases


if __name__ == "__main__":
    destination = Path(__file__).with_name("generated")
    engine = _engine()
    try:
        for name, cases in (
            ("inspection", output_cases(engine)),
            ("inspection_stream", stream_cases(engine)),
        ):
            destination.joinpath(f"{name}.jsonl").write_text(
                "".join(json.dumps(case, separators=(",", ":")) + "\n" for case in cases)
            )
    finally:
        engine.close()
    fold = {chr(i): chr(i).casefold() for i in range(0x110000) if chr(i).casefold() != chr(i)}
    destination.parent.parent.joinpath("src", "generated", "casefold.json").write_text(
        json.dumps(fold, ensure_ascii=True, separators=(",", ":")) + "\n"
    )
