"""The mechanical leakage check of a rewritten prompt: no held-out input or reference text.

A rewrite is refused when it contains, from any held-out input or any reference output:

- an exact string of at least ``leak_characters`` characters (every string leaf of the parsed
  user content and of the reference answer, and the raw user content itself);
- the same string after normalization (case folded, every run of non-alphanumerics collapsed to
  one space);
- a normalized word n-gram of ``leak_ngram`` words that the allowed text (the code prompt, the
  role's fixed system prompt and its answer schema) does not already contain.

These are substring and n-gram comparisons only. They are not a semantic judgment, and a
paraphrase can pass them; the held-out parity is reported on inputs Astra's rewriter never saw.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence

from exp.common.core.artifacts import JsonValue

_NON_WORD = re.compile(r"[^0-9a-z]+")


def normalize(text: str) -> str:
    """Case fold and collapse every run of non-alphanumerics to one space."""
    return _NON_WORD.sub(" ", text.casefold()).strip()


def strings(value: JsonValue) -> list[str]:
    """Every string leaf (keys included) of a JSON value."""
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.append(key)
            found.extend(strings(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(strings(item))
    return found


def held_out_texts(user: str, reference: JsonValue) -> list[str]:
    """The texts of one held-out input and its reference that a rewrite must not contain."""
    texts = [user, *strings(reference)]
    try:
        texts.extend(strings(json.loads(user)))
    except ValueError:
        pass
    return texts


def ngrams(text: str, size: int) -> set[str]:
    """Every normalized word n-gram of ``text``."""
    words = normalize(text).split()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def leaks(
    candidate: str,
    held_out: Iterable[tuple[str, Sequence[str]]],
    allowed: Sequence[str],
    *,
    characters: int,
    ngram: int,
) -> tuple[str, ...]:
    """Return every leak of held-out text into ``candidate``; empty when it is clean.

    Args:
        candidate: The rewritten prompt.
        held_out: Per held-out example id, its texts (:func:`held_out_texts`).
        allowed: Texts whose content a rewrite may repeat (code prompt, fixed system, schema).
        characters: Shortest string checked verbatim and normalized.
        ngram: Word n-gram length.

    Returns:
        One line per finding, naming the example and the kind of overlap.
    """
    folded = normalize(candidate)
    grams = ngrams(candidate, ngram)
    permitted: set[str] = set()
    permitted_text = " ".join(normalize(text) for text in allowed)
    for text in allowed:
        permitted |= ngrams(text, ngram)
    found: list[str] = []
    for example_id, texts in held_out:
        for text in texts:
            stripped = text.strip()
            if len(stripped) < characters:
                continue
            short = stripped[:60].replace("\n", " ")
            if stripped in candidate:
                found.append(f"{example_id}: verbatim held-out text {short!r}")
                continue
            norm = normalize(stripped)
            if len(norm) >= characters and norm in folded and norm not in permitted_text:
                found.append(f"{example_id}: normalized held-out text {short!r}")
                continue
            shared = sorted((ngrams(text, ngram) & grams) - permitted)
            if shared:
                found.append(f"{example_id}: held-out {ngram}-gram {shared[0]!r}")
    return tuple(dict.fromkeys(found))
