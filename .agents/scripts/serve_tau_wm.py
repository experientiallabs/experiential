"""Serve the prebuilt tau-bench WM in the BENCH-B training or eval configuration.

One script for both serving configs so fixes (failover chain, ports, warm-up behavior)
cannot diverge between the training env and the eval env:

- ``train``: env + reward judge on Bedrock Haiku 4.5 (dated profile id — cost control;
  the artifact's built-in Opus provider is overridden at load).
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
from wmh.providers.base import Provider, ProviderConfig, ProviderKind
from wmh.providers.fallback import FallbackProvider
from wmh.serving.server import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "examples" / "tau-bench" / "models" / "tau-bench"
HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # dated profile id (required)
EVAL_ENV_MODEL = "gpt-5.5"
JUDGE_MODEL = "us.anthropic.claude-opus-4-8"  # the artifact's own serve model id


def _fallback_chain(config: ProviderConfig) -> Provider:
    """Two independent provider instances (each owns its boto/http client), same model."""
    return FallbackProvider([get_provider(config), get_provider(config)])


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

    if mode == "train":
        haiku = ProviderConfig(kind=ProviderKind.BEDROCK, model=HAIKU_MODEL, region="us-east-1")
        serve_provider = _fallback_chain(haiku)
        reward_provider = serve_provider
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

    wm = WorldModel.load(str(MODEL_DIR), serve_provider, reward_provider=reward_provider)
    app = create_app(world_models={"tau-bench": wm})
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
