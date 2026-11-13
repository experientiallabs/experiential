"""Compare post-search input rewrites, authority restoration, and final constraint validation."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from exp.common.models.model import ToolCall
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry, KeywordClassifier
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.contracts import GuardrailPolicy, GuardrailRejected
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.regex import RegexAdapterDocument, RegexClassifier
from exp.runtime.gateway.guardrails.session import GuardrailSession
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.guardrails.subjects import _subject_value
from exp.runtime.gateway.native_search_subject import input_templates, subject_registration


async def cases() -> list[dict[str, object]]:
    """Exercise every action on message text, immutable tool material, and rewrites."""
    registry = ClassifierRegistry(
        {
            "keyword": KeywordClassifier(("secret", "strasse")),
            "redact": RegexClassifier(
                RegexAdapterDocument(
                    adapter_id="redact", patterns=("secret",), replacement="public"
                )
            ),
            "undo": RegexClassifier(
                RegexAdapterDocument(adapter_id="undo", patterns=("public",), replacement="secret")
            ),
            "grow": RegexClassifier(
                RegexAdapterDocument(adapter_id="grow", patterns=("secret",), replacement="secret!")
            ),
        }
    )
    engine = GuardrailEngine(
        store=MappingGuardrailStore(),
        client=DirectClassifierClient(registry),
        monotonic=time.monotonic,
    )
    result: list[dict[str, object]] = []
    for i in range(432):
        messages = [GatewayMessage(role="user", content="secret" if i % 6 == 0 else "question")]
        if i % 3:
            messages += [
                GatewayMessage(
                    role="assistant",
                    tool_calls=(
                        ToolCall(
                            call_id="call",
                            name="fn",
                            arguments={"z": "secret" if i % 7 == 0 else "ok", "a": 1.0},
                            raw_arguments=None
                            if i % 3 == 1
                            else json.dumps({"z": "secret" if i % 7 == 0 else "ok", "a": 1.0}),
                        ),
                    ),
                )
            ]
            messages += [
                GatewayMessage(
                    role="tool",
                    content="secret" if i % 5 else "safe",
                    tool_call_id="call",
                    tool_is_error=i % 11 == 0,
                )
            ]
        request = GatewayRequest(
            surface=GatewayApiSurface.MESSAGES,
            messages=tuple(messages),
            tools=(
                GatewayToolDefinition(
                    name="fn", description="secret" if i % 17 == 0 else None, parameters={}
                ),
            ),
        )
        policies = tuple(
            GuardrailPolicy.model_validate(
                {
                    "policy_id": f"p{j}",
                    "organization_id": "org",
                    "identity_id": "identity",
                    "protected": bool(i % 2),
                    "max_request_bytes": 64 if i % 29 == 0 else 1_048_576,
                    "checks": [
                        {
                            "check_id": f"c{j}",
                            "adapter_id": ["redact", "undo", "grow", "keyword"][(i // 9 + j) % 4],
                            "stage": "input",
                            "capability": "pii",
                            "action": ["allow", "block", "error", "modify"][(i // 3) % 4],
                            "timeout_ms": 1000,
                        }
                    ],
                }
            )
            for j in range(1 + i % 3)
        )
        session = GuardrailSession(engine, policies, time.monotonic() + 30)
        try:
            inspected = await session._inspect_input(request)
            expected = {"request": _subject_value(inspected)}
        except GuardrailRejected as exc:
            expected = {"failure": exc.failure.model_dump(mode="json")}
        result.append(
            {
                "subject": subject_registration(request, request.tools),
                "checks": input_templates(session),
                "expected": expected,
            }
        )
    return result


if __name__ == "__main__":
    Path(__file__).with_name("generated").joinpath("input_programs.jsonl").write_text(
        "".join(
            json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n"
            for case in asyncio.run(cases())
        )
    )
