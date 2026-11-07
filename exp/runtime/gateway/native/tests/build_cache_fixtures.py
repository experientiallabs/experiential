"""Freeze cache offers and callback decisions from the existing Python owner."""

from __future__ import annotations

import json
import random
import tempfile
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.explicit_cache import (
    CacheCreator,
    CacheOffer,
    CachePrefix,
    CacheReady,
    CacheRenewal,
    CacheResult,
    CacheUnavailable,
    GoogleCacheAuthority,
    prepare_cache_offer,
    validate_cache_claim,
    validate_cache_result,
)
from exp.runtime.gateway.explicit_cache_test import _authority
from exp.runtime.gateway.native_bridge_test import _start_first
from exp.runtime.gateway.native_cache_registration import cache_registration
from exp.runtime.gateway.native_explicit_cache import NativeCacheState
from exp.runtime.gateway.native_explicit_cache_test import (
    _admission,
    _Clock,
    _control,
    _finish_argument,
    _Host,
    _selector,
)


def offers() -> list[JsonObject]:
    """Include exact integer ceilings, Unicode scopes and denied size bounds."""
    rng = random.Random(6718)
    cases: list[JsonObject] = []
    for index in range(240):
        authority = asdict(_authority())
        authority["tenant_scope"] = [
            "tenant",
            "café 水",
            "😀",
            "x\u200by",
            "a\nb",
            " x",
            "x\u00a0y",
        ][index % 7]
        authority["minimum_input_tokens"] = rng.choice([1, 1024, 8192, 2**63 - 1])
        authority["create_input_nano_usd_per_million"] = rng.choice([0, 1, 7, 10**9, 2**63 - 1])
        authority["storage_nano_usd_per_million_token_hour"] = rng.choice([1, 1001, 2**63 - 1])
        inputs = {
            "operation_id": "operation",
            "request_id": "request",
            "attempt_id": "attempt",
            "account_key_fingerprint": "accounté",
            "plan_scope_digest": "a" * 64,
            "resource_prefix": "cachedContents/",
            "prefix_bytes": rng.choice([0, 1, 1000, 8192, 10**8, 2**63 - 1]),
            "framing_tokens": 4096,
            "requested_at": 1_800_000_000.0,
            "renewable": True,
        }
        try:
            result = prepare_cache_offer(authority=GoogleCacheAuthority(**authority), **inputs)
            expected = None if result is None else asdict(result)
            error = False
        except ValueError:
            expected, error = None, True
        cases.append(
            {
                "kind": "offer",
                "authority": authority,
                "input": inputs,
                "expected": expected,
                "error": error,
            }
        )
    return cases


class Host(_Host):
    """Return chosen synthetic durable facts and retain the exact callback writes."""

    def __init__(self, kind: str) -> None:
        super().__init__(None if kind == "disabled" else _authority())
        self.kind = kind
        self.calls: list[JsonObject] = []

    def claim(self, offer: CacheOffer) -> CacheCreator | CacheReady | CacheUnavailable:
        """Simulate one durable claim without changing its identity or horizon."""
        self.calls.append({"action": "claim", "offer": asdict(offer)})
        if self.kind == "unavailable":
            return CacheUnavailable("unknown")
        if self.kind.startswith("ready"):
            return CacheReady(
                "cachedContents/existing",
                1_800_000_100.0,
                1536,
                "renew-operation" if self.kind == "ready_renew" else None,
                1_800_000_300.8 if self.kind == "ready_renew" else None,
            )
        return CacheCreator(offer.operation_id, offer.expires_at)

    def record(self, result: CacheResult) -> None:
        """Save exact observation bytes, including writes with lost acknowledgements."""
        self.calls.append({"action": "record", "result": asdict(result)})
        if self.kind == "record_failure":
            raise RuntimeError("synthetic lost acknowledgement")

    def record_renewal(self, renewal: CacheRenewal) -> None:
        """Save the first renewal outcome independently of generation success."""
        self.calls.append({"action": "renew", "renewal": asdict(renewal)})

    def select_prefix(
        self, authority: GoogleCacheAuthority, request_id: str, candidates: tuple[CachePrefix, ...]
    ) -> str | None:
        """Select an offered prefix while retaining only content-free metadata."""
        del request_id
        self.calls.append(
            {
                "action": "select",
                "authority": asdict(authority),
                "candidates": [asdict(c) for c in candidates],
            }
        )
        return candidates[-1].prefix_sha256


def callbacks(*, automatic: bool = False) -> list[JsonObject]:
    """Run the real callbacks, including repeated results and contradictory facts."""
    cases: list[JsonObject] = []
    edits = [
        {},
        {"outcome": "unknown"},
        {"name": "cachedContents/foreign/path"},
        {"name": "cachedContents/_name"},
        {"http_status": 429},
        {"http_status": 204},
        {"total_tokens": 1},
        {"total_tokens": 10**9},
        {"total_tokens": None},
        {"expire_time": "2027-01-15T08:00:00"},
        {"expire_time": None},
        {"expire_time": "2027-01-15T08:00:00+00:00"},
        {"create_time": "2027-01-15T08:00:00+00:00"},
        {"create_time": "invalid"},
        {"create_time": "2028-01-15T08:00:00+00:00"},
        {"outcome": "not_created"},
        {
            "outcome": "not_created",
            "name": None,
            "http_status": None,
            "total_tokens": None,
            "expire_time": None,
            "create_time": None,
        },
        {"unexpected": True},
    ]
    with tempfile.TemporaryDirectory() as temporary:
        host = Host("creator")
        control, key = _control(Path(temporary), host)
        admitted = _admission(control, key)
        _start_first(control, admitted)
        entry = control._accounting.entry(str(admitted["request_id"]))
        assert entry is not None and entry.explicit_cache_state is not None
        bindings = entry.explicit_cache_state.bindings
        if automatic:
            bindings = tuple(
                None if b is None else replace(b, automatic_plans=(b.plan,)) for b in bindings
            )
        clock = _Clock()
        with (
            patch("exp.runtime.gateway.native_explicit_cache.time", clock),
            patch("exp.runtime.gateway.native_explicit_cache.uuid4", return_value=UUID(int=1071)),
        ):
            for kind in [
                "creator",
                "record_failure",
                "ready",
                "ready_renew",
                "disabled",
                "unavailable",
            ]:
                for edit in edits if kind in {"creator", "record_failure"} else [{}]:
                    host = Host(kind)
                    control._explicit_cache = host
                    control._automatic_cache = host if automatic else None
                    entry.explicit_cache_state = NativeCacheState(bindings, automatic=automatic)
                    config = cache_registration(entry)
                    steps: list[JsonObject] = []

                    def invoke(method: str, value: JsonObject) -> JsonObject | None:
                        step: JsonObject
                        try:
                            expected = json.loads(getattr(control, method)(json.dumps(value)))
                            step = {
                                "method": method,
                                "input": value,
                                "expected": expected,
                                "error": False,
                            }
                        except Exception:  # noqa: BLE001 - fixture records the public error boundary.
                            expected = None
                            step = {
                                "method": method,
                                "input": value,
                                "expected": None,
                                "error": True,
                            }
                        steps.append(step)  # noqa: B023 - invoked synchronously within this iteration.
                        return expected

                    creation = invoke("prepare_explicit_cache", _selector(admitted))
                    assert creation is not None
                    if creation["state"] == "create":
                        value = {**_finish_argument(admitted, creation), **edit}
                        invoke("finish_explicit_cache", value)
                        invoke("finish_explicit_cache", value)
                        invoke("finish_explicit_cache", {**value, "outcome": "unknown"})
                    if kind == "ready_renew":
                        value = {
                            **_selector(admitted),
                            "operation_id": "renew-operation",
                            "outcome": "renewed",
                            "expire_time": datetime.fromtimestamp(clock.now + 300, UTC).isoformat(),
                        }
                        invoke("renew_explicit_cache", value)
                        invoke("renew_explicit_cache", value)
                    invoke("prepare_explicit_cache", _selector(admitted))
                    cases.append(
                        {
                            "kind": "callbacks",
                            "configuration": config,
                            "active": {
                                "attempt": entry.active_attempt_id,
                                "depth": 0,
                                "deployment": _selector(admitted)["deployment_id"],
                            },
                            "authority": None
                            if host.authority_value is None
                            else asdict(host.authority_value),
                            "host_kind": kind,
                            "steps": steps,
                            "calls": host.calls,
                        }
                    )
    return cases


def observations() -> list[JsonObject]:
    """Validate namespace, measurement, clock and renewal authority independently."""
    offer = prepare_cache_offer(
        authority=_authority(),
        operation_id="operation",
        request_id="request",
        attempt_id="attempt",
        account_key_fingerprint="account",
        plan_scope_digest="a" * 64,
        resource_prefix="cachedContents/",
        prefix_bytes=8192,
        framing_tokens=4096,
        requested_at=1_800_000_000.0,
        renewable=True,
    )
    assert offer is not None
    cases: list[JsonObject] = []
    for tokens in [1, 1023, 1024, 12288, 12289, 2**63 - 1]:
        for seconds in [-1, 0, 4, 5, 6, 300, 301, 3600, 3601]:
            claim = CacheReady("cachedContents/resource", offer.requested_at + seconds, tokens)
            result = CacheResult(
                "operation",
                "ready",
                offer.requested_at,
                "cachedContents/resource",
                tokens,
                offer.requested_at + seconds,
                200,
            )
            for kind, item in [("claim", claim), ("observation", result)]:
                try:
                    if isinstance(item, CacheReady):
                        validate_cache_claim(offer, item, offer.requested_at)
                    else:
                        validate_cache_result(offer, item)
                    valid = True
                except ValueError:
                    valid = False
                cases.append(
                    {"kind": kind, "offer": asdict(offer), "input": asdict(item), "valid": valid}
                )
    return cases


if __name__ == "__main__":
    target = Path(__file__).parent / "generated" / "cache.jsonl"
    cases = offers() + observations() + callbacks() + callbacks(automatic=True)
    target.write_text(
        "".join(json.dumps(c, ensure_ascii=False, separators=(",", ":")) + "\n" for c in cases)
    )
