"""Freeze existing Python search and tokenizer behavior for the native continuation owner."""

from __future__ import annotations

import json
import random
from pathlib import Path

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayToolDefinition
from exp.runtime.gateway.guardrails.regex import RegexAdapterDocument, RegexClassifier
from exp.runtime.gateway.native_search_counter import reset_letters
from exp.runtime.gateway.reservation_tokenizer import (
    count_reservation_tokens,
    reservation_token_bound,
)
from exp.runtime.gateway.tool_search.search import (
    ToolSearchPatternError,
    bm25_search,
    clamp_limit,
    regex_search,
)


def tokenizer_cases() -> list[JsonObject]:
    """Cover Unicode, special spellings, letter resets and bounded exact windows."""
    rng = random.Random(66218)
    pieces = [
        "",
        "word",
        " world!",
        "\n",
        "\r\n",
        "\t",
        " ",
        "\u0301",
        "Ⅶ",
        "\u0345",
        "世界",
        "日本語",
        "Здравствуйте",
        "é",
        "İ",
        "Straße",
        "αβγ",
        "عربي",
        " हिन्दी",
        "😀",
        "👩‍💻",
        "<|endoftext|>",
        "<|endofprompt|>",
        ' {"name":1.3e-9}',
        "\0",
    ]
    texts = ["", "\nA", "\nⅦ", "\n\u0345", "a\n" * 4098 + "\n" * 4097 + "word"]
    texts += ["".join(rng.choices(pieces, k=1 + i % 50)) for i in range(480)]
    return [
        {
            "text": text,
            "limit": limit,
            "expected": reservation_token_bound(text, exact_characters=limit),
            "exact": count_reservation_tokens(text),
        }
        for i, text in enumerate(texts)
        for limit in (0, max(1, len(text) // 2), len(text), i % 31)
    ]


def search_cases() -> list[JsonObject]:
    """Exercise ranking, declaration order, limits, and model-authored RE2 syntax."""
    rng = random.Random(67498)
    names = ["getWeather", "setWeather", "weatherReport", "bookFlight", "list_files", "éFoo"]
    words = ["weather", "temperature", "city", "read", "write", "flight", "files", "Straße"]
    cases: list[JsonObject] = []
    patterns = [
        "weather",
        "^get",
        "(?i)GET",
        r"[\D]",
        r"[a\W]",
        r"\123",
        r"\12",
        r"\077",
        r"\QWeather\E",
        r"\C",
        r"\bflight\b",
        r"[[:alpha:]]",
        r"[[]",
        r"[^\D]",
        r"(?x)weather",
        r"(?u)weather",
        r"a{1001}",
        r"(?=weather)",
        r"(weather)\1",
        r"\u0061",
        r"(?R)weather",
        r"a{",
        "[",
        "",
        "a" * 201,
        r"(?:a{100}){100}",
        r"\p{Greek}",
        r"\p{Emoji}",
    ]
    for i in range(420):
        tools = [
            GatewayToolDefinition(
                name=f"{name}_{n}",
                description=" ".join(rng.choices(words, k=n % 7)) if n % 5 else None,
                parameters={
                    "type": "object",
                    "properties": {"city": {"description": "weather city", "type": "string"}},
                },
            )
            for n, name in enumerate(rng.choices(names, k=i % 16))
        ]
        raw_limit = [None, True, False, -1, 0, 1, 5, 90, 2.4, "8", 10**18][i % 11]
        default = 1 + i % 10
        limit = clamp_limit(raw_limit, default=default)
        query = " ".join(rng.choices(words + ["the", "please"], k=1 + i % 4))
        pattern = patterns[i % len(patterns)] if i % 3 else None
        try:
            matched = (
                bm25_search(query, tools, limit=limit)
                if pattern is None
                else regex_search(pattern, tools, limit=limit)
            )
            expected = [tools.index(tool) for tool in matched]
        except ToolSearchPatternError as exc:
            expected = {"error": str(exc)}
        cases.append(
            {
                "tools": [tool.model_dump(mode="json") for tool in tools],
                "limit": raw_limit,
                "default": default,
                "query": query,
                "pattern": pattern,
                "expected": expected,
            }
        )
    return cases


def regex_cases() -> list[JsonObject]:
    """Compare exact redaction spans, including bytes, captures, and empty matches."""
    patterns = [
        r"[\w-z]",
        r"[\s-z]",
        r"[\p{L}-z]",
        r"a{01}",
        r"a{1,02}",
        r"\C",
        r"\C\C",
        r"(\C)\C",
        r"\C(\C)",
        r"(?P<x>\C+)",
        r"(\C)*",
        r"\C*",
        r"é\C",
        r"\Cé",
        r"\C|x",
        r"$",
        r"a*",
        r"a?",
        r".*?",
        r"[\D]",
        r"[^\D]",
        r"[x\W]",
        r"[\S-]",
        r"\12",
        r"\123",
        r"\0",
        r"\Q.a[?]\E",
        r"\Qunclosed",
        r"[[:alpha:]]",
        r"[[:^alpha:]]",
        r"[[]",
        r"[]a]",
        r"[^]a]",
        r"[&~]",
        r"[\d-a]",
        r"[--a]",
        r"[-a-]",
        r"a{",
        r"(?U)a*",
        r"\b",
        r"\B",
        r"(?m)^a$",
        r"(?s).",
        r"\p{^L}",
        r"\PL",
    ]
    groups_file = Path(__file__).parents[1] / "src/generated/re2_groups.json"
    patterns += [rf"\p{{{name}}}" for name in json.loads(groups_file.read_text())]
    patterns += [f"(?i){pattern}" for pattern in patterns]
    patterns += [
        r"(?i)a(?-i)b",
        r"a(?i)|b",
        r"(?:a(?i)|b)c",
        r"(?i:a)b",
        r"(?i)[^a-z]",
        r"(?i)[[:^upper:]]",
        r"(?i)[^\W]",
        r"(?i)[a-ſ]",
        r"(?i)[ß-ẞ]",
    ]
    patterns += [f"(?i){chr(code)}" for code in (0x10D50, 0x10D70, 0x1C89, 0x1C8A, 0xA7CB, 0x264)]
    subjects = ["", "é中 aS?", "Straße İı Kſ", "\n\r\v aaaaa\n", ".a[?] unclosed", "\0\nS123-x[]&~"]
    rng = random.Random(41629)
    subjects += [
        "abABabcABCſkKKßẞ"
        + "".join(chr(code) for code in (0x10D50, 0x10D70, 0x1C89, 0x1C8A, 0xA7CB, 0x264))
    ]
    subjects += [
        "".join(chr(c) for c in rng.choices(range(0x110000), k=100) if not 0xD800 <= c <= 0xDFFF)
    ]
    cases: list[JsonObject] = []
    for pattern in patterns:
        classifier = RegexClassifier(RegexAdapterDocument(adapter_id="test", patterns=(pattern,)))
        for subject in subjects:
            flagged, rewritten = classifier._redact(subject)
            cases.append(
                {"pattern": pattern, "text": subject, "flagged": flagged, "rewritten": rewritten}
            )
    for pattern in (r"$", r"a*", r"a?", r"\C"):
        classifier = RegexClassifier(RegexAdapterDocument(adapter_id="test", patterns=(pattern,)))
        for subject in ("ab" * 2048, "é" * 2048):
            try:
                flagged, rewritten = classifier._redact(subject)
                expected: JsonObject = {"flagged": flagged, "rewritten": rewritten}
            except ValueError:
                expected = {"error": True}
            cases.append({"pattern": pattern, "text": subject, **expected})
    return cases


def main() -> None:
    """Regenerate the checked-in Python reference without a network or provider call."""
    directory = Path(__file__).with_name("generated")
    directory.joinpath("tokenizer_letters.txt").write_text(reset_letters() + "\n")
    for name, cases in (
        ("tokenizer", tokenizer_cases()),
        ("search", search_cases()),
        ("regex", regex_cases()),
    ):
        directory.joinpath(f"{name}.jsonl").write_text(
            "".join(
                json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n" for case in cases
            )
        )


if __name__ == "__main__":
    main()
