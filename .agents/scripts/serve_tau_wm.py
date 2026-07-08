"""Serve the prebuilt tau-bench WM in the BENCH-B training or eval configuration.

One script for both serving configs so fixes (failover chain, ports, warm-up behavior)
cannot diverge between the training env and the eval env:

- ``train``: env + reward judge on Bedrock Haiku 4.5 (dated profile id — cost control;
  the artifact's built-in Opus provider is overridden at load). ``WMH_ENV_TEMPERATURE``
  optionally pins the env's sampling temperature (judge unaffected): at the provider
  default 0.7, the WM imagines materially different case circumstances per session —
  measured live: identical 4-step action replays scored {0.95, 0.15, 0.3, 0.15, 0.65,
  0.95} (stdev 0.34) across fresh sessions, which drowns group-relative advantages in
  environment luck. Pinning ~0 makes same-actions → same-verdict, so within-group
  reward differences reflect the policy again.
- ``eval`` (D30): env on the pinned GPT-5.5 (OpenAI; strongest circularity blunting vs
  the haiku training env), reward judge on Opus 4.8 (D12/D21 — third family vs both WM
  backends). Requires OPENAI_API_KEY, read from the gitignored ``.env`` at the repo root.

Both providers are wrapped in same-model FallbackProvider chains (D18): throttles fail
over instantly; a hung read fails over at the bedrock client's 600s bound instead of
killing the episode.

Promotion note (AGENTS rule 7): serving a built WM with an overridden provider is
dataset-agnostic and now has three consumers — it belongs in `wmh serve` as a provider
override flag; tracked in DECISIONS.

Run from the wmh repo root:
    uv run python .agents/scripts/serve_tau_wm.py train [port]   # default port 8000
    uv run python .agents/scripts/serve_tau_wm.py eval [port]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

from wmh.engine.world_model import WorldModel
from wmh.providers import get_provider
from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    Message,
    Provider,
    ProviderConfig,
    ProviderKind,
    VerifyResult,
)
from wmh.serving.server import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
# WMH_MODEL_DIR/WMH_WM_NAME swap the served artifact for the D67 cross-benchmark
# smokes (terminal/swe/gui reuse this script unchanged apart from these two).
_DEFAULT_MODEL_DIR = (
    REPO_ROOT / "packages" / "environment-capture" / "tau-bench" / "models" / "tau-bench"
)
MODEL_DIR = Path(os.environ.get("WMH_MODEL_DIR", _DEFAULT_MODEL_DIR))
WM_NAME = os.environ.get("WMH_WM_NAME", "tau-bench")
HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # dated profile id (required)
EVAL_ENV_MODEL = "gpt-5.5"
JUDGE_MODEL = "us.anthropic.claude-opus-4-8"  # the artifact's own serve model id


class FallbackProvider:
    """Sequential same-call failover that FORWARDS temperature (unlike WaterfallProvider,
    which drops sampling params by design). The temperature pass-through is load-bearing
    here: WMH_ENV_TEMPERATURE (D62/D64) and the judge's explicit temperature=0.0 must
    reach the backend or the training substrate silently changes.
    """

    def __init__(self, chain: list[Provider]) -> None:
        if not chain:
            raise ValueError("FallbackProvider needs at least one provider")
        self._chain = chain
        self.config = chain[0].config

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        last: Exception | None = None
        for provider in self._chain:
            try:
                return provider.complete(
                    system, messages, temperature=temperature, max_tokens=max_tokens
                )
            except Exception as exc:  # noqa: BLE001 - any backend failure moves down the chain
                last = exc
        assert last is not None
        raise last

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._chain[0].embed(texts)

    def verify(self) -> VerifyResult:
        return self._chain[0].verify()


def _fallback_chain(config: ProviderConfig) -> Provider:
    """Same-model failover: a second same-region instance, then us-west-2 for Bedrock.

    The cross-region link rides through regional brownouts (observed live:
    us-east-1 ServiceUnavailableException storms on the Opus judge stall whole evals —
    both same-region links 503 together).
    """
    chain = [get_provider(config), get_provider(config)]
    if config.kind is ProviderKind.BEDROCK and config.region != "us-west-2":
        chain.append(get_provider(config.model_copy(update={"region": "us-west-2"})))
    # Cross-PROVIDER last resort for Opus 4.8: the Anthropic direct API (own quota pool,
    # rides through Bedrock-wide storms). Key distributed to the box .envs (Silen ack'd
    # direct-key use after the D68 OpenAI termination).
    if "opus-4-8" in config.model and os.environ.get("ANTHROPIC_API_KEY"):
        chain.append(
            get_provider(ProviderConfig(kind=ProviderKind.ANTHROPIC, model="claude-opus-4-8"))
        )
    return FallbackProvider(chain)


class PinnedTemperatureProvider:
    """Forces one sampling temperature on every ``complete`` call it forwards.

    Wraps ONLY the WM's serve provider: callers' temperature arguments (the WM never
    passes one, so it otherwise gets the 0.7 provider default) are replaced, while the
    reward judge keeps its own unwrapped provider and its explicit temperature=0.0.
    """

    def __init__(self, inner: Provider, temperature: float) -> None:
        self._inner = inner
        self._temperature = temperature
        self.config = inner.config

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        del temperature
        return self._inner.complete(
            system, messages, temperature=self._temperature, max_tokens=max_tokens
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed(texts)

    def verify(self) -> VerifyResult:
        return self._inner.verify()


def _load_dotenv() -> None:
    """Minimal .env loader: KEY=VALUE lines, optional `export `, optional quotes."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "eval"):
        raise SystemExit(f"usage: {sys.argv[0]} train|eval [port]")
    mode = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    _load_dotenv()  # OPENAI/ANTHROPIC keys for backend swaps + the direct-API opus link

    top_k: int | None = None
    if mode == "train":
        # Fidelity->transfer curve (D67): WMH_ENV_MODEL/WMH_ENV_PROVIDER swap the env
        # backend per curve point; the reward judge stays PINNED on the haiku chain so
        # points differ by environment fidelity only. WMH_TOP_K=0 = the no-RAG point.
        haiku = ProviderConfig(kind=ProviderKind.BEDROCK, model=HAIKU_MODEL, region="us-east-1")
        reward_provider = _fallback_chain(haiku)
        env_model = os.environ.get("WMH_ENV_MODEL")
        env_kind = ProviderKind(os.environ.get("WMH_ENV_PROVIDER", "bedrock"))
        if env_model is None:
            serve_provider = reward_provider
        else:
            if env_kind is ProviderKind.OPENAI:
                _load_dotenv()
            serve_provider = _fallback_chain(
                ProviderConfig(
                    kind=env_kind,
                    model=env_model,
                    region="us-east-1" if env_kind is ProviderKind.BEDROCK else None,
                )
            )
        raw_top_k = os.environ.get("WMH_TOP_K")
        if raw_top_k is not None:
            top_k = int(raw_top_k)
        env_temp = os.environ.get("WMH_ENV_TEMPERATURE")
        if env_temp is not None:
            serve_provider = PinnedTemperatureProvider(serve_provider, float(env_temp))
    else:
        _load_dotenv()
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(
                "OPENAI_API_KEY missing: put it in the gitignored .env at the repo root"
            )
        serve_provider = _fallback_chain(
            ProviderConfig(kind=ProviderKind.OPENAI, model=EVAL_ENV_MODEL)
        )
        reward_provider = _fallback_chain(
            ProviderConfig(kind=ProviderKind.BEDROCK, model=JUDGE_MODEL, region="us-east-1")
        )

    wm = WorldModel.load(
        str(MODEL_DIR), serve_provider, reward_provider=reward_provider, top_k=top_k
    )
    app = create_app(world_models={WM_NAME: wm})
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
