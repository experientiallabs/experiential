"""Tests for the packaged, network-free reservation tokenizer."""

from __future__ import annotations

import gzip
import hashlib
import random
from pathlib import Path

import pytest
import tiktoken
import tiktoken.load
import tiktoken.registry

from exp.runtime.gateway import reservation_tokenizer
from exp.runtime.gateway.reservation_tokenizer import (
    PACKAGED_RANKS_RESOURCE,
    PACKAGED_RANKS_SHA256,
    RESERVATION_ENCODING,
    ReservationTokenizerError,
    count_reservation_tokens,
    packaged_ranks,
    reservation_encoder,
    reservation_token_bound,
)


def _forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every tiktoken download, cache lookup, or registry path fail loudly."""

    def refuse(*args: object, **kwargs: object) -> object:
        """Fail any call: the packaged encoder must never reach these seams."""
        raise AssertionError(f"reservation tokenizer touched a network path: {args} {kwargs}")

    monkeypatch.setattr(tiktoken.load, "read_file", refuse)
    monkeypatch.setattr(tiktoken.load, "read_file_cached", refuse)
    monkeypatch.setattr(tiktoken.load, "load_tiktoken_bpe", refuse)
    monkeypatch.setattr(tiktoken.registry, "get_encoding", refuse)
    monkeypatch.setattr(tiktoken, "get_encoding", refuse)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")


def test_encoder_builds_from_package_data_with_every_network_path_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cold host with no egress gets the published o200k vocabulary from the wheel."""
    _forbid_network(monkeypatch)
    reservation_encoder.cache_clear()
    try:
        encoder = reservation_encoder()
        assert encoder.name == RESERVATION_ENCODING
        assert encoder.n_vocab == 200_019
        # "Hello, world!" is four o200k tokens; the published ids pin the vocabulary.
        assert encoder.encode_ordinary("Hello, world!") == [13225, 11, 2375, 0]
        assert (
            encoder.decode(encoder.encode_ordinary("深度学习 and ünïcödé"))
            == "深度学习 and ünïcödé"
        )
        assert reservation_encoder() is encoder
    finally:
        reservation_encoder.cache_clear()


def test_packaged_table_is_the_published_vocabulary() -> None:
    """The shipped file decompresses to tiktoken's own pinned o200k digest and rank count."""
    resource = Path(reservation_tokenizer.__file__).with_name(PACKAGED_RANKS_RESOURCE)
    table = gzip.decompress(resource.read_bytes())
    assert hashlib.sha256(table).hexdigest() == PACKAGED_RANKS_SHA256
    ranks = packaged_ranks()
    assert len(ranks) == 199_998
    assert ranks[b"Hello"] == 13225
    assert max(ranks.values()) < 200_000


def test_corrupt_or_missing_table_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """Startup names the packaged file and the digest mismatch instead of guessing."""

    class _Resource:
        """Stand-in package resource returning fixed bytes."""

        def __init__(self, payload: bytes) -> None:
            """Hold the bytes the fake resource returns."""
            self._payload = payload

        def joinpath(self, name: str) -> _Resource:
            """Return the same resource for the requested name."""
            assert name == PACKAGED_RANKS_RESOURCE
            return self

        def read_bytes(self) -> bytes:
            """Return the fixed payload."""
            return self._payload

    monkeypatch.setattr(
        reservation_tokenizer.resources,
        "files",
        lambda _package: _Resource(gzip.compress(b"QQ== 0\n")),
    )
    with pytest.raises(ReservationTokenizerError, match="has digest .* expected"):
        packaged_ranks()

    monkeypatch.setattr(
        reservation_tokenizer.resources, "files", lambda _package: _Resource(b"not gzip")
    )
    with pytest.raises(ReservationTokenizerError, match="missing or unreadable"):
        packaged_ranks()


_ALPHABET = (
    "abcXYZ",
    "éßЖ漢字",
    "0123456789",
    " ",
    "  \t",
    "\n",
    "\r\n",
    "\n\n",
    "{}[]();:,.!?'\"/\\-_=+*&^%$#@",
    "'s 're 'll",
    "\u0301",  # a combining mark
    "😀",
    "<|endoftext|>",
)


def _random_text(rng: random.Random, pieces: int) -> str:
    """Adversarial mixtures of every o200k pre-tokenizer character class."""
    return "".join(rng.choice(_ALPHABET) for _ in range(pieces))


def test_buffer_count_equals_encode_ordinary_including_specials_and_surrogates() -> None:
    encoder = reservation_encoder()
    rng = random.Random(11)
    texts = [_random_text(rng, rng.randint(0, 400)) for _ in range(200)]
    texts += ["", "<|endoftext|><|endofprompt|>", "lone \ud800 surrogate", "x" * 10_000]
    for text in texts:
        assert count_reservation_tokens(text) == len(encoder.encode_ordinary(text))


def test_newline_letter_boundaries_split_the_count_exactly() -> None:
    """The reset the bounded count relies on: no pre-token spans "\n" + letter."""
    encoder = reservation_encoder()
    rng = random.Random(23)
    for _ in range(300):
        text = _random_text(rng, rng.randint(2, 300)) + "\n" + rng.choice("aZéЖ漢")
        text += _random_text(rng, rng.randint(0, 300))
        for index in range(1, len(text)):
            if text[index - 1] == "\n" and text[index].isalpha():
                whole = len(encoder.encode_ordinary(text))
                halves = len(encoder.encode_ordinary(text[:index])) + len(
                    encoder.encode_ordinary(text[index:])
                )
                assert whole == halves, repr(text[max(0, index - 8) : index + 8])


def test_bound_is_exact_within_the_window_and_never_below_the_count() -> None:
    encoder = reservation_encoder()
    rng = random.Random(5)
    for _ in range(200):
        text = _random_text(rng, rng.randint(0, 600))
        exact = len(encoder.encode_ordinary(text))
        assert reservation_token_bound(text) == exact
        for window in (0, 1, 16, 64, 256):
            assert reservation_token_bound(text, exact_characters=window) >= exact


def test_bound_counts_the_prefix_and_bounds_the_rest_by_utf8_length() -> None:
    prefix = "def handler(event):\n    return event\n"
    suffix = "Reply in prose. " * 50
    text = prefix + suffix
    counted = reservation_token_bound(text, exact_characters=len(prefix) + 3)
    assert counted == count_reservation_tokens(prefix) + len(suffix.encode())
    # No reset inside the window: the whole text is bounded by its bytes.
    flat = "word " * 200
    assert reservation_token_bound(flat, exact_characters=100) == len(flat.encode())
