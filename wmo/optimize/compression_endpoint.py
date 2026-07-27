"""Client for the team's self-hosted LLMLingua-2 compressor endpoint (H100, fp32).

The learned compressor runs on a GPU box, not in the serving process: a 177M token classifier
needs a GPU to be worth its latency (C1 measured 0.197 s/10k tokens on an H100 against 4.5 s on
a laptop CPU), and one warm copy serves the whole team. This module is the thin client side of
that: it implements the same segment-aware `Compressor` shape as `identity` and `truncate`, so
a policy can name it without anything else in the harness knowing a network hop exists.

Configuration is two environment variables, `WMO_COMPRESSOR_URL` and `WMO_COMPRESSOR_API_KEY`.
There is no fallback. If the endpoint is unreachable this raises: a compressor that silently
serves uncompressed text on failure would make cost and accuracy results depend on the health
of a box nobody was watching, and would quietly invalidate any grid that hit a bad minute.

PIN (temporary, remove when PR #265 lands): the D-COMPRESS seam types live in
`wmo/optimize/compression.py` on branch `compress/c3`, which is not on main yet. The
`CompressionConfig`, `CompressionResult`, and `estimate_tokens` definitions below are
field-identical mirrors of that module at commit 7f0b0efa. When the seam merges, delete them,
import the real ones, and register this compressor with the seam's `register_compressor()`.
Nothing else in this file changes.
"""

from __future__ import annotations

import logging
import math
import os
import ssl
import time
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, JsonValue, ValidationError

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

_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Deterministic proxy token count of `text` (ceil of chars/4; 0 only for empty text)."""
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


class CompressionConfig(BaseModel):
    """Per-cluster compression choice carried on the policy artifact (D-COMPRESS shape)."""

    compressor_id: str = Field(min_length=1)
    compressor_version: str = "1"
    aggressiveness: float = Field(default=0.0, ge=0.0, le=1.0)


class CompressionResult(BaseModel):
    """What one compress() call did: the output segments plus its own accounting."""

    segments: list[str]
    tokens_in_raw: int
    tokens_in_compressed: int
    latency_s: float
    cost_usd: float = 0.0


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
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    compressor_version: str = ""
    model_fingerprint: str = ""


class LLMLingua2EndpointCompressor:
    """Remote LLMLingua-2 compression at a fixed absolute keep-probability threshold.

    `aggressiveness` is passed through as that threshold, which is a DEVIATION from the seam's
    documented "fraction of content the compressor may remove" reading, and a deliberate one:
    hitting an exact removal fraction requires per-input percentile selection, the rule C1
    measured rewriting 45-81% of the already-emitted compressed prefix on every appended turn.
    A fixed threshold decides each word locally and keeps the prompt cache intact. Removal
    fraction is therefore an outcome, reported per call, not an input. 0.0 remains a strict
    no-op (returned bit-for-bit, without a network call), so the seam's 0.0 contract holds.
    """

    id = "llmlingua2-endpoint"
    version = "1"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        cert_path: str | None = None,
        timeout_s: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
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
    def from_env(cls, *, timeout_s: float = 30.0) -> LLMLingua2EndpointCompressor:
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
        reply = self._post({"segments": segments, "threshold": config.aggressiveness})
        # Wall clock, not the endpoint's compute time: the round trip is what the serving path
        # actually waits for, and effective-cost accounting has to see the real latency. The
        # endpoint's own compute latency is logged separately.
        latency_s = time.monotonic() - start
        if len(reply.segments) != len(segments):
            raise CompressorEndpointError(
                f"compressor endpoint returned {len(reply.segments)} "
                f"segments for {len(segments)} inputs; a compressor may not merge, split, or "
                "reorder segments. Check that the endpoint version matches this client."
            )
        log.debug(
            "compressed %d segments via %s (server %.1fms, round trip %.1fms, $%.6f)",
            len(segments),
            self.base_url,
            reply.latency_ms,
            latency_s * 1000,
            reply.cost_usd,
        )
        return CompressionResult(
            segments=reply.segments,
            # Seam proxy counts, not the endpoint's own tokenizer counts, so this compressor's
            # accounting is comparable with identity and truncate in the same grid.
            tokens_in_raw=raw,
            tokens_in_compressed=sum(estimate_tokens(segment) for segment in reply.segments),
            latency_s=latency_s,
            cost_usd=reply.cost_usd,
        )

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
