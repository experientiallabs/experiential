"""Tests for the packaged, network-free reservation tokenizer."""

from __future__ import annotations

import gzip
import hashlib
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
    packaged_ranks,
    reservation_encoder,
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
