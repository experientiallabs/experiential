"""Build Python-reference cases for native read-only input checks and exact scoring."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry, KeywordClassifier
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.contracts import GuardrailPolicy, GuardrailRejected
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.native_owner import input_registration
from exp.runtime.gateway.guardrails.regex import (
    BuiltinPattern,
    RegexAdapterDocument,
    RegexClassifier,
)
from exp.runtime.gateway.guardrails.session import GuardrailSession, denied_input_failure
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore


def input_cases() -> list[JsonObject]:
    """Compare ordered policies, Unicode, limits and complete context coverage."""
    registry = ClassifierRegistry(
        {
            "keyword": KeywordClassifier(("secret", "strasse", "i̇")),
            "email": RegexClassifier(
                RegexAdapterDocument(adapter_id="email", builtin_patterns=(BuiltinPattern.EMAIL,))
            ),
        }
    )
    engine = GuardrailEngine(
        store=MappingGuardrailStore(),
        client=DirectClassifierClient(registry),
        monotonic=time.monotonic,
    )
    cases: list[JsonObject] = []
    for i in range(288):
        request = GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(
                GatewayMessage(role="system", content="secret" if i % 13 == 0 else "instruction"),
                GatewayMessage(
                    role="user",
                    content=["ordinary", "secret", "Straße", "İ", "user@example.test", "世界"][
                        i % 6
                    ],
                ),
            ),
        )
        policies = tuple(
            GuardrailPolicy.model_validate(
                {
                    "policy_id": f"policy-{n}",
                    "organization_id": "org",
                    "identity_id": "identity",
                    "protected": bool(i % 2),
                    "input_execution": "parallel",
                    "max_request_bytes": 64 if i % 17 == 0 else 1_048_576,
                    "checks": [
                        {
                            "check_id": f"check-{n}",
                            "adapter_id": ["keyword", "email"][(i // 6 + n) % 2],
                            "stage": "input",
                            "capability": "pii",
                            "action": ["allow", "block", "error"][(i // 12 + n) % 3],
                            "timeout_ms": 1000,
                        }
                    ],
                }
            )
            for n in range(2 if i % 5 == 0 else 1)
        )
        session = GuardrailSession(engine, policies, time.monotonic() + 30)
        program = input_registration(session, (request,))
        expected: JsonObject = {"action": "allow"}
        try:
            session.inspect_input(request)
        except GuardrailRejected as exc:
            expected = {
                "action": "error",
                "failure": denied_input_failure(exc.failure).model_dump(mode="json"),
            }
        cases.append({"checks": program, "expected": expected})
    return cases


def score_cases() -> list[JsonObject]:
    """Exercise correctly rounded positive sums at exact threshold boundaries."""
    rng = random.Random(15971)
    cases: list[JsonObject] = []
    for i in range(1024):
        values = [rng.random() * 2.0 ** -rng.randrange(0, 1000) for _ in range(1 + i % 40)]
        if i % 3 == 0:
            values[:1] = [1.0]
        if i % 5 == 0:
            values += [2.0**-53, 2.0**-54]
        aggregate = "mean" if i % 2 else "max"
        score = math.fsum(values) / len(values) if aggregate == "mean" else max(values)
        cases.append(
            {
                "adapter": {
                    "categories": {
                        "category": {
                            "questions": {f"q{n}": {} for n in range(len(values))},
                            "aggregate": aggregate,
                            "threshold": score,
                        }
                    }
                },
                "response": {
                    "answers": {f"q{n}": {"type": "noul", "noul": v} for n, v in enumerate(values)}
                },
                "expected": [["category", score]],
            }
        )
    return cases


def main() -> None:
    """Write deterministic cases beside the native regression tests."""
    target = Path(__file__).with_name("generated")
    for name, cases in [("input", input_cases()), ("input_scores", score_cases())]:
        (target / f"{name}.jsonl").write_text(
            "".join(
                json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n" for case in cases
            )
        )


if __name__ == "__main__":
    main()
