"""Literal CLI defaults for the route command family."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wmo.common.vendor.waterfall import ChatMaxTokensField

_MAX_TOKENS_FIELDS: tuple[ChatMaxTokensField, ...] = ("max_tokens", "max_completion_tokens")
DEFAULT_MATRIX_FILENAME = "matrix.json"
_DEFAULT_HISTORY_CHARS = 2000
_DEFAULT_POOL_PATH = ".wmo/pool.toml"
_POLICY_FILENAME = "policy.json"
_HASHING_EMBEDDER_DIM = 512
_LOCAL_EMBEDDER_DIM = 1024
_AZURE_EMBEDDER_DIM = 3072
_AZURE_EMBEDDER_ENV = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT")
_WM_SIMULATED = "wm_simulated"
_REAL_EPISODE = "real_episode"
_DEFAULT_WM_JUDGE = "world-model verifier"
COMPRESSOR_IDS_HELP = "identity | truncate"
_MATRIX_DIGEST_MARK = "sha256="
