"""Client for the team's self-hosted LLMLingua-2 compressor endpoint (H100, fp32).

The learned compressor runs on a GPU box, not in the serving process: a 177M token classifier
needs a GPU to be worth its latency (C1 measured 0.197 s/10k tokens on an H100 against 4.5 s on
a laptop CPU), and one warm copy serves the whole team. This module is the thin client side of
that, registered into the D-COMPRESS seam like any other compressor, so a policy can name it
without anything else in the harness knowing a network hop exists.

Configuration is two environment variables, `WMO_COMPRESSOR_URL` and `WMO_COMPRESSOR_API_KEY`.
There is no fallback. If the endpoint is unreachable this raises: a compressor that silently
served uncompressed text on failure would make cost and accuracy results depend on the health
of a box nobody was watching, and would quietly invalidate any grid that hit a bad minute.

Importing this module registers a FACTORY for `llmlingua2-endpoint`, not the compressor: the
compressor is built on the first policy that names it. That split is the point. Building it
reads credentials and calls the box, and neither belongs in `import wmo`, but a policy naming
the id has to resolve without the caller remembering to register anything first. So
`wmo optimize route fit --compressor llmlingua2-endpoint` works with only the two env vars set.

Construction VERIFIES the live endpoint's selection rule before attesting `append_stable`, so
the admission ticket the seam checks is a measured fact about the server that is actually
answering, not a claim this file makes about a server it never contacted. It also adopts the
box's published request cap as `max_segments_per_call`, which is what the seam chunks against.
"""

from __future__ import annotations

import logging
import os
import ssl
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
from pydantic import BaseModel, JsonValue, ValidationError

from wmo.optimize.compression import (
    CompressionConfig,
    CompressionResult,
    estimate_tokens,
    register_compressor,
    register_compressor_factory,
)

log = logging.getLogger(__name__)

URL_ENV = "WMO_COMPRESSOR_URL"
KEY_ENV = "WMO_COMPRESSOR_API_KEY"
CERT_ENV = "WMO_COMPRESSOR_CERT"

# The pinned self-signed certificate shipped beside the deploy scripts. The endpoint has no
# domain, so clients trust exactly this certificate rather than a public CA. Resolved from this
# file rather than the working directory, so it is found from anywhere in a source checkout; in
# an installed wheel `deploy/` is absent and WMO_COMPRESSOR_CERT must point at a copy.
DEFAULT_CERT_PATH = (
    Path(__file__).resolve().parents[2] / "deploy" / "compressor-endpoint" / "compressor-cert.pem"
)

# The only selection rule whose output is append-stable. The server hard-codes it and publishes
# it on /healthz; this client refuses to register against anything else (see `append_stable`).
REQUIRED_SELECTION_RULE = "fixed-absolute-threshold"

# Generous by default: a routing bank fit compresses every fit scenario in ONE embed batch
# (CompressingEmbedder), so an 800-scenario fit arrives here as a single 800-segment call and
# should stay a single round trip. These bounds only split batches bigger than the server will
# accept, and they split on fixed boundaries so the same input always splits the same way.
MAX_SEGMENTS_PER_REQUEST = 1024
MAX_CHARS_PER_REQUEST = 8_000_000

# Compressing a whole fit corpus in one call is seconds of GPU work, so the default read budget
# is minutes rather than seconds. An endpoint that is DOWN still fails immediately (the
# connection is refused); this only bounds a server that has gone quiet mid-request.
DEFAULT_TIMEOUT_S = 120.0


class CompressorEndpointError(RuntimeError):
    """The compressor endpoint could not be reached or returned something unusable."""


class EndpointReply(BaseModel):
    """The endpoint's `/v1/compress` response.

    Mirrors the server's `CompressResponse` (deploy/compressor-endpoint/server.py). Parsing it
    into a model rather than reading a loose dict means a shape change on the box surfaces as
    one clear error here instead of a confusing failure deeper in the serving path.
    """

    segments: list[str]
    tokens_in: int = 0
    tokens_out: int = 0
    # GPU compute only, measured inside the box's serializing lock. `queue_ms` is the wait for
    # that lock, reported separately and NOT included in cost_usd.
    latency_ms: float = 0.0
    queue_ms: float = 0.0
    cost_usd: float = 0.0
    compressor_version: str = ""
    model_fingerprint: str = ""


class LLMLingua2EndpointCompressor:
    """Remote LLMLingua-2 compression at a fixed absolute keep-probability threshold.

    This compressor's reading of the seam's `aggressiveness` dial is the absolute keep
    probability a word must clear to survive. It satisfies both of the dial's invariants: 0.0 is
    a strict bit-for-bit no-op (returned locally, without a network call), and the dial is
    monotone, since raising the bar can only drop more words. It cannot hit an exact removal
    fraction, which is the general case the seam documents for learned compressors: the achieved
    ratio is an outcome, read per call off `CompressionResult`.

    `append_stable` is True because selection is per-word and local: whether a word survives
    depends on its own keep probability against a fixed bar, never on the rest of the input, so
    appending a segment cannot rewrite bytes already emitted. That is a property of the SERVER's
    selection rule, so `register_endpoint_compressor` verifies the live server is running it
    before the attestation reaches the seam. The percentile rule it replaces rewrote 45-81% of
    the emitted prefix per appended turn (C1 round 0).
    """

    id = "llmlingua2-endpoint"
    version = "1"
    append_stable = True
    # The seam reads this to chunk before calling. Starts at the client default and is replaced
    # by the box's own published cap during `handshake()`, so it tracks the running server.
    max_segments_per_call = MAX_SEGMENTS_PER_REQUEST

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        cert_path: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: httpx.Client | None = None,
        max_segments_per_request: int = MAX_SEGMENTS_PER_REQUEST,
        max_chars_per_request: int = MAX_CHARS_PER_REQUEST,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._max_segments = max_segments_per_request
        self._max_chars = max_chars_per_request
        if client is not None:
            self._client = client
        else:
            # Pinning: when a certificate is given it becomes the client's ENTIRE trust store,
            # so a public CA cannot be substituted for the box.
            verify: ssl.SSLContext | bool = (
                ssl.create_default_context(cafile=cert_path) if cert_path else True
            )
            self._client = httpx.Client(timeout=timeout_s, verify=verify)

    @classmethod
    def from_env(cls, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> LLMLingua2EndpointCompressor:
        """Build from `WMO_COMPRESSOR_URL` / `WMO_COMPRESSOR_API_KEY`, or say what is missing."""
        base_url = os.environ.get(URL_ENV, "").strip()
        api_key = os.environ.get(KEY_ENV, "").strip()
        missing = [name for name, value in ((URL_ENV, base_url), (KEY_ENV, api_key)) if not value]
        if missing:
            raise CompressorEndpointError(
                f"compressor endpoint is not configured: {' and '.join(missing)} "
                f"{'are' if len(missing) > 1 else 'is'} unset. Set {URL_ENV} to the endpoint "
                f"(for example https://40.80.93.150:8443) and {KEY_ENV} to the bearer token "
                "from the box (see deploy/compressor-endpoint/README.md), or select the "
                "'identity' compressor to run without compression."
            )
        cert_path = os.environ.get(CERT_ENV, "").strip() or None
        if cert_path is None and DEFAULT_CERT_PATH.is_file():
            cert_path = str(DEFAULT_CERT_PATH)
        return cls(base_url, api_key, cert_path=cert_path, timeout_s=timeout_s)

    def handshake(self) -> None:
        """Read the running box once: verify its selection rule and adopt its request caps.

        Two things a client should not assume about a box it did not start. The selection rule
        decides whether `append_stable` is honest, so it is checked rather than trusted. The
        request caps decide where batches have to split, so they are read from the server rather
        than hardcoded here, where they could drift out of sync with the box's configuration.

        One round trip, called at registration, so a misconfigured or wrongly-deployed endpoint
        fails at mount instead of mid-grid.
        """
        health = self.health()
        rule = health.get("selection_rule", "<absent>")
        if rule != REQUIRED_SELECTION_RULE:
            raise CompressorEndpointError(
                f"compressor endpoint at {self.base_url} reports selection rule '{rule}', not "
                f"'{REQUIRED_SELECTION_RULE}'. This client attests append stability, which only "
                "holds for absolute-threshold selection; a percentile rule rewrites the "
                "already-compressed prefix on every appended turn and must not be served. "
                "Redeploy the box from deploy/compressor-endpoint/ before using this compressor."
            )
        self._max_segments = int(health.get("max_segments", self._max_segments))
        self._max_chars = int(health.get("max_chars", self._max_chars))
        # The seam chunks against this before calling, so it must reflect the box, not a guess.
        self.max_segments_per_call = self._max_segments

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        """Compress the mutable segments remotely; raise rather than degrade to no compression."""
        raw = sum(estimate_tokens(segment) for segment in segments)
        if config.aggressiveness == 0.0 or not segments:
            return CompressionResult(
                segments=list(segments),
                tokens_in_raw=raw,
                tokens_in_compressed=raw,
                latency_s=0.0,
            )
        start = time.monotonic()
        out: list[str] = []
        cost = 0.0
        gpu_ms = 0.0
        queue_ms = 0.0
        for begin, end in self._batches(segments):
            reply = self._post(
                {"segments": segments[begin:end], "threshold": config.aggressiveness}
            )
            if len(reply.segments) != end - begin:
                raise CompressorEndpointError(
                    f"compressor endpoint returned {len(reply.segments)} segments for "
                    f"{end - begin} inputs; a compressor may not merge, split, or reorder "
                    "segments. Check that the endpoint version matches this client."
                )
            out.extend(reply.segments)
            cost += reply.cost_usd
            gpu_ms += reply.latency_ms
            queue_ms += reply.queue_ms
        # Wall clock, not the endpoint's compute time: the round trip is what the serving path
        # actually waits for, and effective-cost accounting has to see the real latency.
        latency_s = time.monotonic() - start
        log.debug(
            "compressed %d segments via %s (round trip %.1fms, gpu %.1fms, queue %.1fms, $%.6f)",
            len(segments),
            self.base_url,
            latency_s * 1000,
            gpu_ms,
            queue_ms,
            cost,
        )
        return CompressionResult(
            segments=out,
            # Seam proxy counts, not the endpoint's own tokenizer counts, so this compressor's
            # accounting is comparable with identity and truncate in the same grid.
            tokens_in_raw=raw,
            tokens_in_compressed=sum(estimate_tokens(segment) for segment in out),
            latency_s=latency_s,
            cost_usd=cost,
        )

    def _batches(self, segments: list[str]) -> Iterator[tuple[int, int]]:
        """Split an oversized call into request-sized index ranges.

        Normally yields exactly one range: the seam hands a compressor a whole request at once
        and the endpoint is sized to take it. Splitting matters for bank fits, which compress
        every fit scenario in a single call. Boundaries depend only on the segment list, so the
        same input always splits identically, and per-segment output is unchanged either way
        because the server is batch-invariant (asserted at its startup).
        """
        begin = 0
        chars = 0
        for index, segment in enumerate(segments):
            too_many = index - begin >= self._max_segments
            too_long = chars + len(segment) > self._max_chars
            if index > begin and (too_many or too_long):
                yield begin, index
                begin = index
                chars = 0
            chars += len(segment)
        yield begin, len(segments)

    def health(self) -> dict[str, str]:
        """Fetch the endpoint's unauthenticated health document (version, fingerprint, uptime)."""
        try:
            response = self._client.get(f"{self.base_url}/healthz", timeout=self._timeout_s)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise CompressorEndpointError(
                f"compressor endpoint at {self.base_url} is not answering health checks: "
                f"{error}. Check {URL_ENV} and that the box is up "
                "(`ssh h100-dev-box-6 systemctl status wmo-compressor`)."
            ) from error
        return {key: str(value) for key, value in response.json().items()}

    def _post(self, body: dict[str, JsonValue]) -> EndpointReply:
        """POST to /v1/compress with one retry, translating every failure into a clear error."""
        url = f"{self.base_url}/v1/compress"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        last_error: Exception | None = None
        # One retry only, and only for transport faults: a 401 or a 429 will not fix itself,
        # and retrying a rate limit makes the rate limit worse.
        for attempt in (1, 2):
            try:
                response = self._client.post(url, json=body, headers=headers)
            except httpx.HTTPError as error:
                last_error = error
                if attempt == 1:
                    log.warning("compressor endpoint call failed (%s); retrying once", error)
                    continue
                break
            if response.status_code == 401:
                raise CompressorEndpointError(
                    f"compressor endpoint at {self.base_url} rejected the bearer token. Check "
                    f"{KEY_ENV}; rotate it per deploy/compressor-endpoint/README.md if needed."
                )
            if response.status_code == 429:
                raise CompressorEndpointError(
                    f"compressor endpoint at {self.base_url} is rate limiting this token "
                    f"(retry after {response.headers.get('Retry-After', '?')}s). Lower "
                    "concurrency or ask for a raised limit on the box."
                )
            if response.status_code == 413:
                raise CompressorEndpointError(
                    f"compressor endpoint at {self.base_url} rejected the request as too large: "
                    f"{response.text[:200]}. This client splits at {self._max_segments} segments "
                    f"and {self._max_chars} chars per request, so the box is configured below "
                    "that; raise WMO_COMPRESSOR_MAX_SEGMENTS / WMO_COMPRESSOR_MAX_CHARS there, "
                    "or lower max_segments_per_request here to match."
                )
            if response.status_code >= 400:
                raise CompressorEndpointError(
                    f"compressor endpoint at {self.base_url} returned HTTP "
                    f"{response.status_code}: {response.text[:300]}"
                )
            try:
                return EndpointReply.model_validate(response.json())
            except ValidationError as error:
                raise CompressorEndpointError(
                    f"compressor endpoint at {self.base_url} returned a body this client does "
                    f"not understand: {error}. Redeploy the box so its server.py matches this "
                    "client (deploy/compressor-endpoint/deploy.sh)."
                ) from error
        raise CompressorEndpointError(
            f"compressor endpoint at {self.base_url} is unreachable ({last_error}). The "
            f"compressor never silently falls back to uncompressed input. Check {URL_ENV}, "
            f"{KEY_ENV}, and the pinned certificate ({CERT_ENV}), or select the 'identity' "
            "compressor to run without compression."
        ) from last_error


def build_endpoint_compressor() -> LLMLingua2EndpointCompressor:
    """Construct the endpoint compressor from the environment and handshake with the box.

    The factory body behind the lazy registration below. Reads the credentials, then makes one
    call to the box to verify its selection rule and adopt its request caps. Deliberately does
    NOT register the result: `get_compressor` registers what a factory returns, and registering
    here too would just be a second path to the same registry entry.
    """
    compressor = LLMLingua2EndpointCompressor.from_env()
    compressor.handshake()
    return compressor


def register_endpoint_compressor(
    *, timeout_s: float = DEFAULT_TIMEOUT_S
) -> LLMLingua2EndpointCompressor:
    """Build and register the endpoint compressor NOW, instead of at first use.

    The eager path, for a script that would rather find out immediately that the box is
    unreachable than discover it partway into a grid. Ordinary callers do not need this: the
    factory registered at import resolves `llmlingua2-endpoint` on demand.

        from wmo.optimize.compression_endpoint import register_endpoint_compressor

        register_endpoint_compressor()   # raises here if the endpoint is not usable
    """
    compressor = LLMLingua2EndpointCompressor.from_env(timeout_s=timeout_s)
    compressor.handshake()
    register_compressor(compressor)
    return compressor


# The module's ONE import side effect, and deliberately a cheap one: it registers the FACTORY,
# never the compressor. Building the compressor reads credentials and calls the box, which must
# not happen on `import wmo`. The cost and the failure mode move to the first policy that
# actually names 'llmlingua2-endpoint', which is what makes
# `wmo optimize route fit --compressor llmlingua2-endpoint` work with nothing but env vars set.
register_compressor_factory(LLMLingua2EndpointCompressor.id, build_endpoint_compressor)
