"""Generate native Unicode membership from the pinned RE2 group's numeric ranges.

Inputs are re2/unicode_groups.cc and re2/unicode_casefold.cc from
https://github.com/google/re2/tree/2025-11-05. Run with their local paths;
generation and compilation perform no network calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def groups(source: str) -> dict[str, str]:
    """Convert Unicode interval data into UTF-8-safe native character classes."""
    arrays = {
        name: [
            (int(first), int(last)) for first, last in re.findall(r"\{\s*(\d+),\s*(\d+)\s*\}", body)
        ]
        for name, body in re.findall(
            r"static const URange(?:16|32) (\w+)\[\] = \{(.*?)\};", source, re.S
        )
    }
    result: dict[str, str] = {}
    for name, low, high in re.findall(
        r'\{\s*"(\w+)",\s*\+1,\s*(\w+),\s*\d+,\s*(\w+),\s*\d+\s*\}', source
    ):
        pieces: list[str] = []
        for first, last in [*arrays.get(low, []), *arrays.get(high, [])]:
            for floor, ceiling in ((0, 0xD7FF), (0xE000, 0x10FFFF)):
                start, end = max(first, floor), min(last, ceiling)
                if start <= end:
                    pieces.append(f"\\x{{{start:x}}}-\\x{{{end:x}}}")
        result[name] = "[" + "".join(pieces) + "]" if pieces else "[a&&b]"
    if len(result) != 199:
        message = "Expected the 199 Unicode groups from RE2 2025-11-05"
        raise ValueError(message)
    return result


def main() -> None:
    """Write generated membership data from an explicitly supplied source file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("casefold", type=Path)
    args = parser.parse_args()
    source = args.source.read_bytes()
    if (
        hashlib.sha256(source).hexdigest()
        != "e00735cd8ba1f87ea5c0c78c07de7836e9ad02cc84f4ef8cf90e2b0edfd99e68"
    ):
        message = "Expected the pinned RE2 2025-11-05 Unicode range source"
        raise ValueError(message)
    destination = Path(__file__).parents[1] / "src/generated/re2_groups.json"
    destination.write_text(json.dumps(groups(source.decode()), separators=(",", ":")) + "\n")
    fold_source = args.casefold.read_bytes()
    if hashlib.sha256(fold_source).hexdigest() != (
        "e2fcd199b8cee45ac053c26e55a810756c530060e90c2ab267513adb4bfdb707"
    ):
        message = "Expected the pinned RE2 2025-11-05 case-folding source"
        raise ValueError(message)
    destination.with_name("re2_casefold.json").write_text(
        json.dumps(folds(fold_source.decode()), separators=(",", ":")) + "\n"
    )


def folds(source: str) -> list[tuple[int, list[int]]]:
    """Expand simple-fold cycles without using the host's Unicode version."""
    source = source.split("const int num_unicode_casefold")[0]
    pairs: dict[int, int] = {}
    for start, end, delta in re.findall(r"\{\s*(\d+),\s*(\d+),\s*([-\w]+)\s*\}", source):
        for code in range(int(start), int(end) + 1):
            if delta == "EvenOdd":
                target = code + (1 if code % 2 == 0 else -1)
            elif delta == "OddEven":
                target = code + (1 if code % 2 else -1)
            else:
                target = code + int(delta)
            pairs[code] = target
    result: list[tuple[int, list[int]]] = []
    for code, first in sorted(pairs.items()):
        cycle: set[int] = set()
        target = first
        while target != code:
            if target in cycle or target not in pairs:
                message = "Invalid simple-fold cycle"
                raise ValueError(message)
            cycle.add(target)
            target = pairs[target]
        result.append((code, sorted(cycle)))
    return result


if __name__ == "__main__":
    main()
