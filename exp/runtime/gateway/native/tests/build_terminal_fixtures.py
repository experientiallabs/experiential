"""Generate independent Python-reference pricing and terminal normalization cases."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

from exp.common.core.artifacts import JsonObject
from exp.common.models.catalog import GatewayDeploymentCapabilities
from exp.common.models.catalog_prices import (
    GatewayLongContextTier,
    GatewayServiceTierPrices,
    GatewayTokenPrices,
)
from exp.runtime.gateway.attempt_tokens import counted_input_tokens
from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.gateway.disconnect_estimate import (
    bill_within_output_ceiling,
    estimate_disconnect_usage,
    estimate_unreported_truncation_usage,
)
from exp.runtime.gateway.disconnect_estimate_test import _request
from exp.runtime.gateway.ledger_test import _deployment
from exp.runtime.gateway.native_attempt_pricing import priced_attempt
from exp.runtime.gateway.native_attempt_pricing_test import _deepseek
from exp.runtime.gateway.native_service_tiers import settlement_kwarg, tier_ceiling
from exp.runtime.gateway.native_settlement import (
    streamed_output_from_settlement,
    terminal_from_settlement,
)
from exp.runtime.gateway.rate_limit_headers import rate_limit_observation
from exp.runtime.gateway.service_tiers import tier_admission


def pricing_cases() -> list[JsonObject]:
    """Exercise premium boundaries, missing legs, processing tiers and weekly price guards."""
    rng = random.Random(429811)
    cases: list[JsonObject] = []
    for index in range(320):
        instant = datetime(2026, 10, 19, tzinfo=UTC) + timedelta(
            minutes=index - 20, seconds=index % 60
        )
        card = GatewayTokenPrices(
            input_nano_usd_per_million_tokens=rng.choice([1, 100_000_000, 1_000_000_000]),
            output_nano_usd_per_million_tokens=rng.choice([0, 5, 100_000_000]),
            cached_input_nano_usd_per_million_tokens=rng.choice([None, 3, 20_000_000]),
            cache_creation_input_nano_usd_per_million_tokens=rng.choice([None, 31, 200_000_000]),
            reasoning_nano_usd_per_million_tokens=rng.choice([None, 0, 100_000_000]),
        )
        if index % 3 == 0:
            card = card.model_copy(
                update={
                    "long_context": GatewayLongContextTier(
                        input_threshold_tokens=1000,
                        input_nano_usd_per_million_tokens=3_000_000_000,
                        output_nano_usd_per_million_tokens=4_000_000_000,
                    )
                }
            )
        if index % 4 == 0:
            card = card.model_copy(
                update={
                    "priority": GatewayServiceTierPrices(
                        input_nano_usd_per_million_tokens=3, output_nano_usd_per_million_tokens=17
                    )
                }
            )
        deployment = _deployment().model_copy(
            update={
                "gateway": _deployment().gateway.model_copy(
                    update={
                        "prices": card,
                        "capabilities": GatewayDeploymentCapabilities(
                            reports_cached_input_tokens=index % 2 == 0,
                            reports_cache_creation_input_tokens=index % 5 == 0,
                            reports_reasoning_tokens=index % 7 == 0,
                        ),
                    }
                )
            }
        )
        if index % 5 == 0:
            deployment = _deepseek()
        output = rng.choice([1, 16, 150, 4096])
        request = _request().model_copy(
            update={
                "maximum_output_tokens": output,
                "prompt_cache_ttl": "12h" if index % 8 == 0 else None,
            }
        )
        input_tokens = rng.choice([0, 1, 799, 800, 999, 1000, 1001, 100_000])
        requested = "priority" if index % 2 == 0 else None
        forwards = index % 3 != 0
        effective, admission = priced_attempt(
            deployment, requested, forwards_tier=forwards, instant=instant
        )
        cap = deployment.gateway.capabilities
        cases.append(
            {
                "prices": deployment.gateway.prices.model_dump(mode="json"),
                "requested": requested,
                "forwards": forwards,
                "customer": deployment.billing_source == "customer_managed",
                "at": instant.isoformat(),
                "ceiling": {
                    "kind": "tokens",
                    "input": input_tokens,
                    "output": output,
                    "cached": cap.reports_cached_input_tokens,
                    "writes": cap.reports_cache_creation_input_tokens,
                    "hour": False,
                    "twelve": request.prompt_cache_ttl == "12h",
                    "reasoning": cap.reports_reasoning_tokens,
                },
                "effective": effective.gateway.prices.model_dump(mode="json"),
                "admission": None if admission is None else admission.model_dump(mode="json"),
                "amount": tier_ceiling(request, effective, admission, input_tokens=input_tokens),
            }
        )
    return cases


def terminal_cases() -> list[JsonObject]:
    """Compare exact meters, missing-meter estimates, Unicode, truncation and tier evidence."""
    rng = random.Random(87342)
    cases: list[JsonObject] = []
    request = _request()
    cards = GatewayTokenPrices(
        input_nano_usd_per_million_tokens=3,
        output_nano_usd_per_million_tokens=7,
        priority=GatewayServiceTierPrices(
            input_nano_usd_per_million_tokens=5, output_nano_usd_per_million_tokens=11
        ),
    )
    admission = tier_admission(cards, "priority", forwards_tier=True, customer_managed=False)
    for index in range(320):
        phase = index % 4
        usage: JsonObject | None = (
            None
            if index % 3 == 0
            else {
                "input_tokens": rng.choice([None, 0, 37, 1000]),
                "output_tokens": rng.choice([None, 0, 1, 100]),
                "cached_input_tokens": rng.choice([None, 0, 20]),
                "cache_creation_input_tokens": rng.choice([None, 0, 10]),
            }
        )
        data: JsonObject = {
            "outcome": ["completed", "failed", "incomplete", "failed"][phase],
            "request_id": "request",
            "attempt_id": "attempt",
            "finalize": True,
            "dispatched": True,
            "opened": index % 11 != 0,
            "usage": usage,
            "tool_names": ["search"] if index % 7 == 0 else [],
            "web_search_requests": index % 2,
            "tool_search_requests": index % 3,
            "streamed_output": {
                "text": rng.choice(["", "hello 👋🏽", "中国語の文章\n", "👨‍👩‍👧‍👦", "a b c\n"]),
                "reasoning": rng.choice(["", "why?", "αβγ" * 20]),
                "text_overflow_chars": rng.choice([0, 17, 900]),
                "reasoning_overflow_chars": rng.choice([0, 3]),
                "images": int(index % 13 == 0),
            },
        }
        if phase in (1, 3):
            data["failure"] = {
                "failure_class": "cancelled" if phase == 1 else "provider_authentication",
                "safe_message": "fixture",
                "customer_owned": phase == 3,
            }
        if phase == 1:
            data["usage_incomplete_due_to_disconnect"] = True
        if phase == 2:
            data["usage_unreported_truncation"] = True
        selected = admission if index % 2 == 0 else None
        self_served = index % 5 == 0
        if selected is not None and index % 3 == 0:
            data["service_tier"] = {
                "resolution": "confirmed",
                "served": "default" if index % 4 else "priority",
            }
        cap = None if index % 3 == 0 else 64
        cached = rng.choice([0.0, 0.5, 0.9])
        terms: JsonObject = {
            "surface": "chat_completions",
            "counted_input_tokens": counted_input_tokens(request),
            "output_ceiling": cap,
            "output_budget": 100,
            "service_tier": None if selected is None else selected.model_dump(mode="json"),
            "self_served": self_served,
            "paid_prework": False,
        }
        case: JsonObject = {"input": data, "terms": terms, "cached": cached}
        try:
            event, failure = terminal_from_settlement(
                data, surface=GatewayApiSurface.CHAT_COMPLETIONS
            )
            streamed = streamed_output_from_settlement(data)
            event = estimate_disconnect_usage(
                event,
                request=request,
                surface=GatewayApiSurface.CHAT_COMPLETIONS,
                opened=data["opened"] is True,
                streamed=streamed,
                cached_fraction=cached,
            )
            event = estimate_unreported_truncation_usage(
                event,
                request=request,
                surface=GatewayApiSurface.CHAT_COMPLETIONS,
                opened=data["opened"] is True,
                streamed=streamed,
                cached_fraction=cached,
                output_budget=100,
            )
            event = bill_within_output_ceiling(event, output_ceiling=cap)
            tier = settlement_kwarg(selected, data, self_served=self_served).get("service_tier")
            case["expected"] = {
                "usage": None if event.usage is None else event.usage.model_dump(mode="json"),
                "failure": None
                if event.failure is None
                else {
                    "failure_class": event.failure.failure_class.value,
                    "safe_message": event.failure.safe_message,
                    "provider_detail": event.failure.provider_detail,
                    "customer_owned": event.failure.customer_owned,
                    "retry_after_seconds": event.failure.retry_after_seconds,
                },
                "usage_estimated": event.usage_estimated,
                "usage_incomplete_due_to_disconnect": event.usage_incomplete_due_to_disconnect,
                "service_tier": None if tier is None else tier.model_dump(mode="json"),
            }
        except ValueError:
            case["invalid"] = True
        cases.append(case)
    return cases


def observation_cases() -> list[JsonObject]:
    """Keep invalid values unknown while preserving bounded integer and plan headers."""
    cases: list[JsonObject] = []
    for text in [
        "0",
        "-0",
        "-1",
        "+13",
        "1_000",
        "12__0",
        "nan",
        "inf",
        "0.95",
        " 16 ",
        "1000000000000001",
    ]:
        for header in [
            "retry-after",
            "x-ratelimit-limit-requests",
            "anthropic-ratelimit-tokens-remaining",
            "anthropic-ratelimit-unified-5h-utilization",
            "x-codex-primary-used-percent",
        ]:
            headers = {header: text}
            result = rate_limit_observation(headers)
            cases.append({"input": headers, "expected": result.model_dump(mode="json")})
    return cases


if __name__ == "__main__":
    destination = Path(__file__).parent / "generated"
    for name, cases in [
        ("prices", pricing_cases()),
        ("terminal", terminal_cases()),
        ("observations", observation_cases()),
    ]:
        (destination / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(case, sort_keys=True, ensure_ascii=False) for case in cases) + "\n"
        )
