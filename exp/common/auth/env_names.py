"""Canonical and generated environment-variable names for provider connections.

Known providers keep one documented override name. OpenAI-compatible and other custom
connections derive a stable name from the connection ID so operators never type it.
"""

from __future__ import annotations

import re

CANONICAL_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "trustedrouter": "TRUSTEDROUTER_API_KEY",
    "openai-compatible": "OPENAI_COMPATIBLE_API_KEY",
    "tinker": "TINKER_API_KEY",
}
_UNSAFE = re.compile(r"[^A-Za-z0-9_]+")


def derived_api_key_env(provider: str, connection_id: str) -> str | None:
    """Return the internal environment override name for one connection.

    Args:
        provider: Catalog provider kind.
        connection_id: Exact connection name used as the credential-store key.

    Returns:
        Environment-variable name, or ``None`` for Bedrock.
    """
    if provider == "bedrock":
        return None
    canonical = CANONICAL_API_KEY_ENV.get(provider)
    if canonical is not None and provider != "openai-compatible":
        return canonical
    if canonical is not None and connection_id in {provider, "openai-compatible"}:
        return canonical
    stem = _UNSAFE.sub("_", connection_id).strip("_").upper()
    if not stem:
        stem = "CONNECTION"
    if stem[0].isdigit():
        stem = f"CONN_{stem}"
    if not stem.endswith("_API_KEY"):
        stem = f"{stem}_API_KEY"
    return stem
