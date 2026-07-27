"""The team's LLMLingua-2 compressor endpoint: fp32, fixed-threshold, self-hosted on one H100.

This is the serving side of track C1's learned-family survivor. It exposes exactly one useful
operation, `POST /v1/compress`, which takes conversation segments and returns them with
low-information words removed, plus honest per-call metering (server-measured latency and the
GPU-seconds cost that latency implies at the box's hourly rate).

Two properties are load-bearing and are asserted at startup rather than assumed:

- **fp32 only.** C1 measured fp16 keep/drop decisions flipping with batch composition. A
  compressor whose output depends on which other requests it was batched with cannot be
  byte-deterministic in serving, where batch composition is traffic-dependent. The server
  refuses to run in any other dtype.
- **Fixed absolute threshold, never percentile.** Stock LLMLingua-2 keeps the top-k fraction per
  input, which makes every keep/drop decision a function of the whole input's score
  distribution: appending one turn rewrites 45-81% of the already-emitted compressed prefix and
  forfeits the provider's prompt cache. An absolute threshold on the keep probability decides
  each word locally, so an unchanged segment always compresses to the same bytes.

Deviation from C1's bench harness, deliberate: the bench chunked at a fixed 400-word window and
gave any word the 512-subword truncation cut off a keep probability of 0.0, i.e. it silently
DROPPED unscored text. Here chunks are packed against a subword budget so words are scored, and
any word that still ends up unscored is KEPT. A compressor may never drop content it did not
score.

Run (on the box, under systemd):

    WMO_COMPRESSOR_TOKEN=... python server.py --port 8443 --ssl-certfile cert.pem \\
        --ssl-keyfile key.pem
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import logging
import os
import re
import threading
import time
from collections.abc import Iterator, Sequence

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from transformers import AutoModelForTokenClassification, AutoTokenizer

log = logging.getLogger("wmo_compressor")

# The model C1 benched: LLMLingua-2's 177M multilingual BERT token classifier, Apache-2.0
# weights. Overridable so the box can point at a local snapshot without network access.
DEFAULT_MODEL_ID = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

# Standard_NC80adis_H100_v5, centralindia retail consumption, 2 GPUs on the box. We serve on
# one, so a call's cost is its GPU-seconds against half the box rate. This is the honest
# amortized number: the box bills whether or not a request is in flight, but per-call cost is
# only meaningful as the share of the box a call actually occupied.
BOX_USD_PER_HOUR = 19.544
GPUS_ON_BOX = 2
USD_PER_GPU_SECOND = (BOX_USD_PER_HOUR / GPUS_ON_BOX) / 3600

# Whitespace-delimited words with their trailing whitespace attached, so "".join(words) is
# lossless. Same unit C1's audit used, so ratios stay comparable.
WORD_RE = re.compile(r"\S+\s*")

# BERT's 512-position limit minus [CLS] and [SEP].
SUBWORD_BUDGET = 510

# Chunks per forward pass. Matches C1's GPU bench, where fp32 output was identical at batch
# 1, 4, and 16, so this is a throughput knob and not a behavior knob.
FORWARD_BATCH = 32

SELECTION_RULE = "fixed-absolute-threshold"
SERVICE_VERSION = "1"


class CompressRequest(BaseModel):
    """One compression call: the segments to rewrite and the keep-probability cutoff."""

    segments: list[str]
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class CompressionOutcome(BaseModel):
    """What one compress() call produced, including the GPU time it actually occupied.

    `compute_s` is measured INSIDE the serializing GPU lock and after a CUDA sync, so it is the
    time this request held the device, not the time it spent waiting for another request to let
    go of it.
    """

    segments: list[str]
    tokens_in: int
    tokens_out: int
    compute_s: float


class CompressResponse(BaseModel):
    """Compressed segments plus this call's own metering.

    `tokens_in`/`tokens_out` are counted with the compressor's OWN tokenizer, which is what it
    actually processed; they are not the serving model's token counts and are not billable
    truth.

    `latency_ms` is GPU compute only, measured inside the serializing lock, and `cost_usd`
    prices exactly that as GPU-seconds. `queue_ms` is the wait for the lock, reported separately
    and deliberately NOT billed: the box charges for wall time either way, but attributing one
    request's queueing to another request's GPU cost would inflate cost_usd by roughly the
    concurrency factor and make a busy minute look like an expensive one. Add the two to get
    the server-side time a caller waited.
    """

    segments: list[str]
    tokens_in: int
    tokens_out: int
    latency_ms: float
    queue_ms: float
    cost_usd: float
    compressor_version: str
    model_fingerprint: str


class HealthResponse(BaseModel):
    """Unauthenticated liveness plus the identity of what is actually loaded."""

    status: str
    compressor_version: str
    model_fingerprint: str
    uptime_s: float
    model_id: str
    device: str
    dtype: str
    selection_rule: str
    self_test: str
    # Published so a client can size its batches against the box's real configuration instead
    # of a constant that could drift out of sync with it.
    max_segments: int
    max_chars: int


class TokenBucket:
    """Per-token rate limit: `rate_per_min` sustained, `burst` capacity, monotonic refill."""

    def __init__(self, rate_per_min: float, burst: float) -> None:
        self._rate_per_s = rate_per_min / 60.0
        self._burst = burst
        self._lock = threading.Lock()
        self._level: dict[str, float] = {}
        self._last: dict[str, float] = {}

    def take(self, key: str) -> float:
        """Consume one unit for `key`; return 0.0 if allowed, else seconds until one is free."""
        now = time.monotonic()
        with self._lock:
            level = self._level.get(key, self._burst)
            last = self._last.get(key, now)
            level = min(self._burst, level + (now - last) * self._rate_per_s)
            self._last[key] = now
            if level < 1.0:
                self._level[key] = level
                return (1.0 - level) / self._rate_per_s
            self._level[key] = level - 1.0
            return 0.0


def _pack_chunks(words: Sequence[str], subword_lengths: Sequence[int]) -> Iterator[tuple[int, int]]:
    """Yield (start, end) word ranges that each fit the model's subword budget.

    Greedy and purely a function of the segment, so an unchanged segment always chunks the same
    way. A single word longer than the budget gets its own chunk (and is truncated by the
    tokenizer, which the caller compensates for by keeping unscored words).
    """
    start = 0
    used = 0
    for index in range(len(words)):
        length = subword_lengths[index]
        if used and used + length > SUBWORD_BUDGET:
            yield start, index
            start = index
            used = 0
        used += length
    if start < len(words):
        yield start, len(words)


class LLMLingua2FixedThreshold:
    """LLMLingua-2 keep-probabilities under an absolute threshold, fp32, on one GPU."""

    def __init__(self, model_id: str, device: str) -> None:
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForTokenClassification.from_pretrained(model_id, dtype=torch.float32)
        self.model_fingerprint = _fingerprint(model)
        self.model = model.to(device=device, dtype=torch.float32).eval()
        if self.model.dtype != torch.float32:
            raise RuntimeError(
                f"refusing to serve in {self.model.dtype}; fp32 only (see module docstring)"
            )
        id2label = self.model.config.id2label
        keep_ids = [i for i, label in id2label.items() if str(label).lower() in ("1", "preserve")]
        self.keep_index = keep_ids[0] if keep_ids else 1
        self.version = f"llmlingua2-{SELECTION_RULE}-fp32/{SERVICE_VERSION}"
        # One GPU, one model instance: serialize so batch composition (and therefore output) is
        # never a function of concurrent traffic.
        self._lock = threading.Lock()

    def _subword_lengths(self, words: Sequence[str]) -> list[int]:
        """Subword count of each word, batched through the fast tokenizer."""
        if not words:
            return []
        encoded = self.tokenizer(
            [word.strip() for word in words],
            add_special_tokens=False,
        )["input_ids"]
        return [max(1, len(ids)) for ids in encoded]

    def compress(self, segments: Sequence[str], threshold: float) -> CompressionOutcome:
        """Compress every segment in one batched pass, timing only the GPU-held portion.

        All of the request's chunks share forward passes, which is what makes the endpoint
        worth its network hop (one forward per chunk costs ~2.4x more GPU per token). That
        batching is safe only because fp32 keep probabilities do not move with batch
        composition, which is exactly what `run_self_test` proves at startup before the port
        opens. Requests never batch with each other: the lock serializes the GPU, so a
        response is a function of its own request and nothing else in flight.
        """
        with self._lock:
            # The clock starts here, not before the lock: a request queued behind another must
            # not bill its wait as GPU-seconds.
            started = time.perf_counter()
            words_per_segment = [WORD_RE.findall(segment) for segment in segments]
            leading = [segment[: len(segment) - len(segment.lstrip())] for segment in segments]
            flat_lengths = self._subword_lengths(
                [word for words in words_per_segment for word in words]
            )
            lengths: list[list[int]] = []
            cursor = 0
            for words in words_per_segment:
                lengths.append(flat_lengths[cursor : cursor + len(words)])
                cursor += len(words)

            chunks: list[tuple[int, int, int]] = []
            for index, words in enumerate(words_per_segment):
                chunks.extend(
                    (index, start, end) for start, end in _pack_chunks(words, lengths[index])
                )
            # Unscored words default to KEEP: a compressor may never drop what it did not score.
            probabilities = [[1.0] * len(words) for words in words_per_segment]
            for offset in range(0, len(chunks), FORWARD_BATCH):
                batch = chunks[offset : offset + FORWARD_BATCH]
                encoded = self.tokenizer(
                    [
                        [word.strip() for word in words_per_segment[index][start:end]]
                        for index, start, end in batch
                    ],
                    is_split_into_words=True,
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    logits = self.model(**encoded).logits
                keep = torch.softmax(logits.float(), dim=-1)[:, :, self.keep_index].cpu()
                for row, (index, start, end) in enumerate(batch):
                    sums = [0.0] * (end - start)
                    counts = [0] * (end - start)
                    for position, word_index in enumerate(encoded.word_ids(row)):
                        if word_index is None:
                            continue
                        sums[word_index] += float(keep[row, position])
                        counts[word_index] += 1
                    for word in range(end - start):
                        if counts[word]:
                            probabilities[index][start + word] = sums[word] / counts[word]

            out: list[str] = []
            tokens_in = 0
            tokens_out = 0
            for index, words in enumerate(words_per_segment):
                scores = probabilities[index]
                out.append(
                    leading[index]
                    + "".join(word for word, p in zip(words, scores, strict=True) if p >= threshold)
                )
                tokens_in += sum(lengths[index])
                tokens_out += sum(
                    length
                    for length, p in zip(lengths[index], scores, strict=True)
                    if p >= threshold
                )
            # Sync before stopping the clock, still holding the lock: CUDA work is queued
            # asynchronously, so without this the timer would stop before the GPU is done and
            # under-bill the request.
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            compute_s = time.perf_counter() - started
        return CompressionOutcome(
            segments=out, tokens_in=tokens_in, tokens_out=tokens_out, compute_s=compute_s
        )


def _fingerprint(model: torch.nn.Module) -> str:
    """Stable short hash of the loaded weights, so a health check identifies what is serving."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


SELF_TEST_SEGMENTS = [
    "The quarterly revenue report shows that total revenue increased by 12 percent "
    "year over year, driven primarily by strong performance in the enterprise segment.",
    '{"tool": "search_flights", "args": {"origin": "SFO", "destination": "JFK"}}',
    "Traceback (most recent call last): File main.py line 42 in run KeyError: 'user_id'",
]


def run_self_test(compressor: LLMLingua2FixedThreshold) -> str:
    """Prove the three serving invariants or refuse to start.

    Determinism (same input three times, byte-identical), batch-composition invariance (a
    segment compresses the same alone as alongside others, the property fp16 breaks), and
    losslessness at threshold 0.0 (nothing is dropped when nothing may be).
    """
    first = compressor.compress(SELF_TEST_SEGMENTS, 0.5).segments
    for _ in range(2):
        again = compressor.compress(SELF_TEST_SEGMENTS, 0.5).segments
        if again != first:
            raise RuntimeError("self-test FAILED: same input produced different output")
    isolated = compressor.compress(SELF_TEST_SEGMENTS[:1], 0.5).segments
    if isolated[0] != first[0]:
        raise RuntimeError(
            "self-test FAILED: a segment compressed differently alone than in a batch; "
            "output depends on batch composition and cannot be served"
        )
    passthrough = compressor.compress(SELF_TEST_SEGMENTS, 0.0).segments
    if passthrough != list(SELF_TEST_SEGMENTS):
        raise RuntimeError("self-test FAILED: threshold 0.0 is not lossless")
    return "determinism+batch-invariance+threshold-0-lossless PASS"


def build_app(compressor: LLMLingua2FixedThreshold, token: str, self_test: str) -> FastAPI:
    """Wire the routes, auth, and rate limit around a loaded compressor."""
    rate_per_min = float(os.environ.get("WMO_COMPRESSOR_RATE_PER_MIN", "60"))
    burst = float(os.environ.get("WMO_COMPRESSOR_BURST", "120"))
    # Sized for the largest legitimate caller, not the typical one: fitting a routing bank
    # compresses every fit scenario in ONE call (the seam's CompressingEmbedder batches per
    # embed, not per text), so an 800-scenario fit arrives as a single 800-segment request and
    # has to stay a single round trip. These are caps against abuse, not a batching policy.
    max_segments = int(os.environ.get("WMO_COMPRESSOR_MAX_SEGMENTS", "1024"))
    max_chars = int(os.environ.get("WMO_COMPRESSOR_MAX_CHARS", "8000000"))
    bucket = TokenBucket(rate_per_min, burst)
    started = time.monotonic()
    app = FastAPI(title="WMO compressor endpoint", version=compressor.version)
    scheme = HTTPBearer(auto_error=False)

    # B008: a call in an argument default is exactly how FastAPI declares a dependency, and the
    # Annotated form cannot be used here because `scheme` is a local and this module uses
    # postponed annotation evaluation.
    def authorize(
        credentials: HTTPAuthorizationCredentials | None = Depends(scheme),  # noqa: B008
    ) -> str:
        """Constant-time bearer check, then the per-token rate limit."""
        presented = credentials.credentials if credentials else ""
        if not hmac.compare_digest(presented, token):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")
        wait = bucket.take(hashlib.sha256(presented.encode()).hexdigest())
        if wait > 0:
            raise HTTPException(
                status_code=429,
                detail=f"rate limit exceeded ({rate_per_min:g}/min, burst {burst:g})",
                headers={"Retry-After": str(max(1, int(wait + 0.999)))},
            )
        return presented

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        """Liveness and identity; deliberately unauthenticated so a probe needs no secret."""
        return HealthResponse(
            status="ok",
            compressor_version=compressor.version,
            model_fingerprint=compressor.model_fingerprint,
            uptime_s=round(time.monotonic() - started, 3),
            model_id=compressor.model_id,
            device=compressor.device,
            dtype="float32",
            selection_rule=SELECTION_RULE,
            self_test=self_test,
            max_segments=max_segments,
            max_chars=max_chars,
        )

    @app.post("/v1/compress", response_model=CompressResponse)
    def compress(
        body: CompressRequest,
        request: Request,
        _: str = Depends(authorize),  # noqa: B008 - see the note on authorize above
    ) -> CompressResponse:
        """Compress the given segments at an absolute keep-probability threshold."""
        if len(body.segments) > max_segments:
            raise HTTPException(
                status_code=413,
                detail=f"too many segments ({len(body.segments)} > {max_segments})",
            )
        total_chars = sum(len(segment) for segment in body.segments)
        if total_chars > max_chars:
            raise HTTPException(
                status_code=413, detail=f"payload too large ({total_chars} > {max_chars} chars)"
            )
        arrived = time.perf_counter()
        outcome = compressor.compress(body.segments, body.threshold)
        # Everything between arrival and the end of compute that was NOT compute was spent
        # waiting for the GPU lock. Reported, never billed.
        queue_s = max(0.0, (time.perf_counter() - arrived) - outcome.compute_s)
        log.info(
            "compress client=%s segments=%d tokens_in=%d tokens_out=%d threshold=%.3f "
            "compute=%.1fms queue=%.1fms",
            request.client.host if request.client else "?",
            len(body.segments),
            outcome.tokens_in,
            outcome.tokens_out,
            body.threshold,
            outcome.compute_s * 1000,
            queue_s * 1000,
        )
        return CompressResponse(
            segments=outcome.segments,
            tokens_in=outcome.tokens_in,
            tokens_out=outcome.tokens_out,
            latency_ms=round(outcome.compute_s * 1000, 3),
            queue_ms=round(queue_s * 1000, 3),
            cost_usd=outcome.compute_s * USD_PER_GPU_SECOND,
            compressor_version=compressor.version,
            model_fingerprint=compressor.model_fingerprint,
        )

    return app


def main() -> None:
    """Load the model, prove the invariants, then serve over TLS."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 - the point is remote access
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--device", default=os.environ.get("WMO_COMPRESSOR_DEVICE", "cuda:0"))
    parser.add_argument("--model", default=os.environ.get("WMO_COMPRESSOR_MODEL", DEFAULT_MODEL_ID))
    parser.add_argument("--ssl-certfile", default=os.environ.get("WMO_COMPRESSOR_CERT"))
    parser.add_argument("--ssl-keyfile", default=os.environ.get("WMO_COMPRESSOR_KEY"))
    args = parser.parse_args()

    token = os.environ.get("WMO_COMPRESSOR_TOKEN", "")
    if len(token) < 32:
        raise SystemExit(
            "WMO_COMPRESSOR_TOKEN is missing or shorter than 32 chars; generate one with "
            "`openssl rand -hex 32` and put it in the systemd EnvironmentFile"
        )

    log.info("loading %s on %s (fp32)", args.model, args.device)
    compressor = LLMLingua2FixedThreshold(args.model, args.device)
    log.info("loaded, fingerprint %s; running self-test", compressor.model_fingerprint)
    self_test = run_self_test(compressor)
    log.info("self-test: %s", self_test)

    app = build_app(compressor, token, self_test)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
        log_level="info",
    )


if __name__ == "__main__":
    main()
