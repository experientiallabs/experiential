"""Tests for exact per-token byte offsets."""

from __future__ import annotations

from wmo.distill.xtoken.byte_offsets import (
    BYTE_DECODER,
    ByteOffsetTokenizer,
    span_byte_ends,
    token_bytes,
)


class FakeByteLevelTokenizer:
    """A byte-level BPE tokenizer over a fixed vocabulary, plus special tokens.

    Ordinary tokens carry byte-level surface forms (built by encoding their raw
    bytes through the inverse of `BYTE_DECODER`); special tokens carry their
    literal text, exactly as HuggingFace fast tokenizers report them.
    """

    def __init__(self, pieces: dict[int, bytes], specials: dict[int, str] | None = None) -> None:
        encoder = {value: char for char, value in BYTE_DECODER.items()}
        self._surfaces = {
            token_id: "".join(encoder[byte] for byte in raw) for token_id, raw in pieces.items()
        }
        self._raw = dict(pieces)
        for token_id, text in (specials or {}).items():
            self._surfaces[token_id] = text
            self._raw[token_id] = text.encode("utf-8")

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str | None]:
        """Surface forms, None for ids outside the vocabulary."""
        return [self._surfaces.get(token_id) for token_id in ids]

    def decode(self, token_ids: list[int]) -> str:
        """Concatenate the raw bytes and decode as UTF-8."""
        return b"".join(self._raw.get(token_id, b"") for token_id in token_ids).decode(
            "utf-8", errors="replace"
        )


class NormalizingTokenizer(FakeByteLevelTokenizer):
    """A tokenizer whose decode rewrites whitespace, like SentencePiece does."""

    def decode(self, token_ids: list[int]) -> str:
        """Decode, then collapse every run of spaces to a single space."""
        text = super().decode(token_ids)
        while "  " in text:
            text = text.replace("  ", " ")
        return text


def test_token_bytes_round_trips_byte_level_surface_forms() -> None:
    encoder = {value: char for char, value in BYTE_DECODER.items()}
    raw = b"ls -la\n"
    surface = "".join(encoder[byte] for byte in raw)
    assert token_bytes(surface) == raw


def test_ascii_special_token_byte_decodes_to_its_own_literal_bytes() -> None:
    # Every character of '<|im_end|>' is printable ASCII, which IS in the
    # byte-level alphabet, so the byte-decode path and the literal-text
    # fallback agree. The offsets are correct either way; this pins the
    # coincidence so a future alphabet change surfaces here.
    assert token_bytes("<|im_end|>") == b"<|im_end|>"


def test_token_bytes_rejects_surface_forms_outside_the_alphabet() -> None:
    # A raw space is byte 32, which byte-level BPE renders as 'G-dot' rather
    # than ' ', so a literal space marks a special token whose surface form is
    # plain text and must go through the literal-encode fallback.
    assert token_bytes("<|start of text|>") is None
    # CJK sits past the byte-level alphabet's top (chr(288)); real byte-level
    # surface forms spell such characters out as their UTF-8 bytes instead.
    assert token_bytes("\N{CJK UNIFIED IDEOGRAPH-4F60}") is None


def test_empty_span_has_no_offsets() -> None:
    tokenizer = FakeByteLevelTokenizer({1: b"a"})
    assert span_byte_ends(tokenizer, []) == ([], b"")


def test_offsets_are_cumulative_over_token_bytes() -> None:
    tokenizer = FakeByteLevelTokenizer({1: b"ls", 2: b" -", 3: b"la"})
    result = span_byte_ends(tokenizer, [1, 2, 3])
    assert result is not None
    ends, span = result
    assert ends == [2, 4, 6]
    assert span == b"ls -la"
    # Each token's slice is recoverable from the offsets.
    assert span[0:2] == b"ls"
    assert span[2:4] == b" -"
    assert span[4:6] == b"la"


def test_special_tokens_contribute_their_literal_bytes() -> None:
    tokenizer = FakeByteLevelTokenizer({1: b"hi"}, specials={99: "<|im_end|>"})
    result = span_byte_ends(tokenizer, [1, 99])
    assert result is not None
    ends, span = result
    assert span == b"hi<|im_end|>"
    assert ends == [2, 12]


def test_multi_byte_character_split_across_tokens_keeps_exact_offsets() -> None:
    # A single emoji is 4 UTF-8 bytes; byte-level BPE may split it across two
    # tokens, which is precisely the case a decode-per-token approach corrupts.
    rocket = "\N{ROCKET}".encode()
    tokenizer = FakeByteLevelTokenizer({1: rocket[:2], 2: rocket[2:]})
    result = span_byte_ends(tokenizer, [1, 2])
    assert result is not None
    ends, span = result
    assert ends == [2, 4]
    assert span == rocket
    assert span.decode("utf-8") == "\N{ROCKET}"


def test_non_canonical_segmentation_is_handled_like_any_other() -> None:
    # Sampling emits 'S' + 'olver' where canonical BPE gives 'Solver'. Nothing
    # is re-encoded, so both segmentations produce exact offsets.
    tokenizer = FakeByteLevelTokenizer({1: b"S", 2: b"olver", 3: b"Solver"})
    split = span_byte_ends(tokenizer, [1, 2])
    merged = span_byte_ends(tokenizer, [3])
    assert split is not None
    assert merged is not None
    assert split[1] == merged[1] == b"Solver"
    assert split[0] == [1, 6]
    assert merged[0] == [6]


def test_unknown_token_id_reports_unusable() -> None:
    tokenizer = FakeByteLevelTokenizer({1: b"a"})
    assert span_byte_ends(tokenizer, [1, 4242]) is None


def test_normalizing_tokenizer_reports_unusable_instead_of_wrong_offsets() -> None:
    # The reconstruction says 5 bytes ('a  b'), the decode says 3 ('a b'), so
    # byte alignment is invalid for this vocabulary and must be refused.
    tokenizer = NormalizingTokenizer({1: b"a", 2: b"  ", 3: b"b"})
    assert span_byte_ends(tokenizer, [1, 2, 3]) is None


def test_protocol_is_satisfied_structurally() -> None:
    tokenizer: ByteOffsetTokenizer = FakeByteLevelTokenizer({1: b"a"})
    assert span_byte_ends(tokenizer, [1]) is not None
