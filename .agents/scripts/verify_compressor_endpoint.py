"""End-to-end verification of the deployed compressor endpoint (track C1 / C3 serving).

Drives the LIVE H100 endpoint from a client machine and reports the numbers the deploy is
accountable for: determinism across separate HTTP calls, auth rejection, rate limiting, and
measured latency/cost per 10k tokens through the whole network path.

The token denominator is C1's GPT-2 count for the same 120 audit transcripts (135,859 tokens),
so the s/10k and $/10k here are directly comparable with the on-box bench numbers
(0.1966 s/10k, $0.000534/10k at batch 1) rather than being quoted on a different tokenizer.

    uv run python .agents/scripts/verify_compressor_endpoint.py

Reads WMO_COMPRESSOR_URL / WMO_COMPRESSOR_API_KEY / WMO_COMPRESSOR_CERT from .env (gitignored).
Prints no secrets.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import statistics
import time
from pathlib import Path

import httpx

from wmo.optimize.compression_endpoint import (
    CompressionConfig,
    CompressorEndpointError,
    LLMLingua2EndpointCompressor,
)

log = logging.getLogger("verify_endpoint")

TRANSCRIPTS = Path.home() / "Desktop/Projects/wmh-compression-data/cache/audit-transcripts.jsonl"
# C1's GPT-2 token total over these exact 120 transcripts (cache/gpu-bench.json, raw_tokens).
C1_GPT2_TOKENS = 135_859
C1_S_PER_10K = 0.1966
C1_USD_PER_10K = 0.000534
THRESHOLD = 0.5


def load_env(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into the process environment."""
    if not path.exists():
        raise SystemExit(f"{path} not found; see deploy/compressor-endpoint/README.md")
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def check_health(client: LLMLingua2EndpointCompressor) -> None:
    """Report what is actually serving."""
    health = client.health()
    log.info("HEALTH ok: %s", json.dumps(health, indent=2))


def check_determinism(client: LLMLingua2EndpointCompressor, transcripts: list[list[str]]) -> bool:
    """Two separate HTTP calls with the same payload must return identical bytes."""
    config = CompressionConfig(compressor_id="llmlingua2-endpoint", aggressiveness=THRESHOLD)
    sample = transcripts[3]
    first = client.compress(sample, config)
    second = client.compress(sample, config)
    identical = first.segments == second.segments
    log.info(
        "DETERMINISM %s (%d segments, %d -> %d proxy tokens)",
        "PASS: byte-identical across two calls" if identical else "FAIL: outputs differ",
        len(sample),
        first.tokens_in_raw,
        first.tokens_in_compressed,
    )
    return identical


def check_auth(base_url: str, cert: str) -> bool:
    """A wrong token and a missing token must both be rejected with 401."""
    with httpx.Client(verify=cert, timeout=15) as raw:
        body = {"segments": ["hello world"], "threshold": THRESHOLD}
        bad = raw.post(
            f"{base_url}/v1/compress",
            json=body,
            headers={"Authorization": "Bearer " + "0" * 64},
        )
        none = raw.post(f"{base_url}/v1/compress", json=body)
        health = raw.get(f"{base_url}/healthz")
    ok = bad.status_code == 401 and none.status_code == 401 and health.status_code == 200
    log.info(
        "AUTH %s: wrong token -> %d, no token -> %d, unauthenticated health -> %d",
        "PASS" if ok else "FAIL",
        bad.status_code,
        none.status_code,
        health.status_code,
    )
    return ok


def check_bank_fit_batch(client: LLMLingua2EndpointCompressor) -> bool:
    """An 800-scenario bank fit must go through as ONE round trip, not a 413 and not a split.

    The seam's CompressingEmbedder compresses every fit scenario in a single `compress` call, so
    the endpoint's request caps have to clear a realistic fit batch. This sends one.
    """
    scenarios = [
        f"Scenario {index}: the user asks the agent to reconcile an invoice against the "
        f"quarterly ledger and explain any discrepancy in plain language."
        for index in range(800)
    ]
    config = CompressionConfig(compressor_id="llmlingua2-endpoint", aggressiveness=THRESHOLD)
    start = time.perf_counter()
    result = client.compress(scenarios, config)
    elapsed = time.perf_counter() - start
    ok = len(result.segments) == len(scenarios)
    log.info(
        "BANK FIT %s: 800 scenarios in one call, %.2fs, %d -> %d proxy tokens, $%.6f",
        "PASS" if ok else "FAIL",
        elapsed,
        result.tokens_in_raw,
        result.tokens_in_compressed,
        result.cost_usd,
    )
    return ok


def check_queue_not_billed(base_url: str, api_key: str, cert: str) -> bool:
    """Concurrent callers must not bill each other's queueing as their own GPU-seconds.

    The GPU is serialized by a lock. When the clock started before that lock, a request queued
    behind N others billed the whole wait, inflating cost_usd by roughly the concurrency factor
    and making a busy minute look like an expensive one. This fires a serial baseline and then
    a concurrent burst of the SAME payload: cost must stay flat while queue_ms absorbs the wait.
    """
    body = {
        "segments": [
            "The quarterly revenue report shows that total revenue increased by 12 percent "
            "year over year, driven primarily by strong performance in the enterprise segment."
        ]
        * 8,
        "threshold": THRESHOLD,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    concurrency = 8
    with httpx.Client(verify=cert, timeout=60) as raw:
        baseline = raw.post(f"{base_url}/v1/compress", json=body, headers=headers).json()

        def fire(_: int) -> dict[str, float]:
            return raw.post(f"{base_url}/v1/compress", json=body, headers=headers).json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            burst = list(pool.map(fire, range(concurrency)))

    serial_cost = float(baseline["cost_usd"])
    worst_cost = max(float(row["cost_usd"]) for row in burst)
    worst_queue = max(float(row["queue_ms"]) for row in burst)
    # Generous bound: GPU time per request is unchanged by concurrency, so the worst biller
    # should sit near the serial cost, nowhere near concurrency x it.
    ok = worst_cost < serial_cost * 2.5 and worst_queue > 0
    log.info(
        "QUEUE BILLING %s: serial $%.6f, worst concurrent $%.6f (%.2fx, would be ~%dx if "
        "queueing were billed), worst queue_ms %.1f",
        "PASS" if ok else "FAIL",
        serial_cost,
        worst_cost,
        worst_cost / serial_cost if serial_cost else 0.0,
        concurrency,
        worst_queue,
    )
    return ok


def check_selection_rule(client: LLMLingua2EndpointCompressor) -> bool:
    """Verify the rule behind the append-stability attestation, and adopt the box's caps."""
    try:
        client.handshake()
    except CompressorEndpointError as error:
        log.info("SELECTION RULE FAIL: %s", error)
        return False
    log.info("SELECTION RULE PASS: endpoint reports fixed-absolute-threshold")
    return True


def wait_for_capacity(base_url: str, api_key: str, cert: str, need: int = 50) -> None:
    """Idle until the rate limiter has refilled enough budget for the measurement run.

    Only matters when this script is re-run: the rate-limit check deliberately drains the
    bucket, and a throttled measurement would report latency of the limiter, not the model.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    body = {"segments": ["ping"], "threshold": THRESHOLD}
    with httpx.Client(verify=cert, timeout=30) as raw:
        for _ in range(30):
            if raw.post(f"{base_url}/v1/compress", json=body, headers=headers).status_code == 200:
                break
            time.sleep(5)
        else:
            raise SystemExit("endpoint stayed rate limited; wait a minute and re-run")
    # The bucket refills at the sustained rate (1/s by default); give it room for the run.
    log.info("waiting %ds for rate-limit budget before measuring", need)
    time.sleep(need)


def measure(
    client: LLMLingua2EndpointCompressor, transcripts: list[list[str]]
) -> dict[str, float]:
    """Push all 120 transcripts through the endpoint and measure latency and cost."""
    config = CompressionConfig(compressor_id="llmlingua2-endpoint", aggressiveness=THRESHOLD)
    # Batched into groups so the run stays inside the endpoint's burst allowance; each request
    # carries one group's turns as segments, which is the shape the serving seam sends.
    group_size = 8
    latencies: list[float] = []
    server_ms: list[float] = []
    cost = 0.0
    raw_proxy = 0
    compressed_proxy = 0
    start = time.perf_counter()
    for index in range(0, len(transcripts), group_size):
        segments = [turn for transcript in transcripts[index : index + group_size] for turn in transcript]
        call_start = time.perf_counter()
        result = client.compress(segments, config)
        latencies.append(time.perf_counter() - call_start)
        cost += result.cost_usd
        raw_proxy += result.tokens_in_raw
        compressed_proxy += result.tokens_in_compressed
    wall = time.perf_counter() - start

    # Per-request p50 on single transcripts: the latency one serving call actually adds.
    single: list[float] = []
    for transcript in transcripts[:30]:
        call_start = time.perf_counter()
        client.compress(transcript, config)
        single.append(time.perf_counter() - call_start)

    stats = {
        "wall_s": wall,
        "s_per_10k_gpt2": wall / C1_GPT2_TOKENS * 10_000,
        "usd_per_10k_gpt2": cost / C1_GPT2_TOKENS * 10_000,
        "total_cost_usd": cost,
        "keep_ratio_proxy": compressed_proxy / raw_proxy,
        "batched_request_p50_s": statistics.median(latencies),
        "single_transcript_p50_s": statistics.median(single),
        "single_transcript_p95_s": sorted(single)[int(0.95 * len(single)) - 1],
    }
    del server_ms
    log.info("THROUGHPUT %s", json.dumps({k: round(v, 6) for k, v in stats.items()}, indent=2))
    log.info(
        "vs C1 on-box bench: %.4f s/10k here vs %.4f on box (network adds the difference); "
        "$%.6f/10k here vs $%.6f on box",
        stats["s_per_10k_gpt2"],
        C1_S_PER_10K,
        stats["usd_per_10k_gpt2"],
        C1_USD_PER_10K,
    )
    return stats


def check_rate_limit(base_url: str, api_key: str, cert: str) -> bool:
    """Hammer the endpoint with cheap calls until the limiter answers 429.

    Runs LAST: it deliberately drains the token bucket, so anything measured after it would be
    throttled.
    """
    body = {"segments": ["hi"], "threshold": THRESHOLD}
    headers = {"Authorization": f"Bearer {api_key}"}
    statuses: list[int] = []
    with httpx.Client(verify=cert, timeout=20) as raw:

        def fire(_: int) -> int:
            return raw.post(f"{base_url}/v1/compress", json=body, headers=headers).status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            statuses = list(pool.map(fire, range(200)))
    limited = statuses.count(429)
    served = statuses.count(200)
    ok = limited > 0
    log.info(
        "RATE LIMIT %s: %d served, %d rejected with 429 out of %d requests",
        "PASS" if ok else "FAIL (no 429 seen)",
        served,
        limited,
        len(statuses),
    )
    return ok


def main() -> None:
    """Run every live check and summarize."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # httpx logs a line per request; at 200 rate-limit probes that buries the results.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_env(Path(".env"))
    transcripts = [json.loads(line)["turns"] for line in TRANSCRIPTS.open()]
    log.info("loaded %d transcripts", len(transcripts))

    client = LLMLingua2EndpointCompressor.from_env(timeout_s=120.0)
    base_url = client.base_url
    cert = os.environ["WMO_COMPRESSOR_CERT"]
    api_key = os.environ["WMO_COMPRESSOR_API_KEY"]

    check_health(client)
    rule_ok = check_selection_rule(client)
    deterministic = check_determinism(client, transcripts)
    authorized = check_auth(base_url, cert)
    bank_fit = check_bank_fit_batch(client)
    queue_billing = check_queue_not_billed(base_url, api_key, cert)
    wait_for_capacity(base_url, api_key, cert)
    measure(client, transcripts)
    rate_limited = check_rate_limit(base_url, api_key, cert)

    # A down endpoint must raise, never quietly pass the input through.
    down = LLMLingua2EndpointCompressor("https://127.0.0.1:1", api_key, timeout_s=2.0)
    try:
        down.compress(["x"], CompressionConfig(compressor_id="x", aggressiveness=0.5))
    except CompressorEndpointError:
        honest_failure = True
    else:
        honest_failure = False
    log.info("HONEST FAILURE %s: a down endpoint raises", "PASS" if honest_failure else "FAIL")

    checks = {
        "selection_rule": rule_ok,
        "determinism": deterministic,
        "bank_fit_one_round_trip": bank_fit,
        "queue_not_billed": queue_billing,
        "auth": authorized,
        "rate_limit": rate_limited,
        "honest_failure": honest_failure,
    }
    log.info("SUMMARY %s", json.dumps(checks))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
