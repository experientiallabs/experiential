"""Fold the camelCase ``promptCacheKey`` alias onto ``prompt_cache_key``.

Coding clients built on the Vercel AI SDK (opencode, mimocode) place the
provider option ``promptCacheKey`` verbatim in the OpenAI request body instead
of the documented snake_case wire field. The alias is renamed here, before the
compatibility manifest sees the body, so the request proceeds exactly as if
``prompt_cache_key`` had been sent. This is the one camelCase spelling the
OpenAI surfaces admit; every other unknown top-level field stays rejected by
name.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject

PROMPT_CACHE_KEY_ALIAS = "promptCacheKey"
"""Vercel AI SDK spelling of the OpenAI ``prompt_cache_key`` wire field."""

PROMPT_CACHE_KEY_ALIAS_IGNORED = f"{PROMPT_CACHE_KEY_ALIAS}->ignored(explicit_prompt_cache_key)"
"""Disclosure emitted when both spellings arrive and the alias is dropped."""

_CANONICAL = "prompt_cache_key"


def fold_prompt_cache_key_alias(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Rename a top-level ``promptCacheKey`` to ``prompt_cache_key``.

    Args:
        payload: Parsed Chat Completions or Responses body.

    Returns:
        The original payload when the alias is absent; otherwise a shallow copy
        without the alias, carrying its value under ``prompt_cache_key`` unless
        the canonical field was already present, in which case the canonical
        value wins and the dropped alias is named in the returned disclosures.
    """
    if PROMPT_CACHE_KEY_ALIAS not in payload:
        return payload, ()
    folded = {key: value for key, value in payload.items() if key != PROMPT_CACHE_KEY_ALIAS}
    if _CANONICAL in payload:
        return folded, (PROMPT_CACHE_KEY_ALIAS_IGNORED,)
    folded[_CANONICAL] = payload[PROMPT_CACHE_KEY_ALIAS]
    return folded, ()
