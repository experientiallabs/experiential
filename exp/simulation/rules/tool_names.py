"""One canonical key per tool name, for coverage and matching only.

Harnesses spell the same tool differently: ``Write``, ``write`` and ``write_file`` name one
tool across trajectory pools. A probe task refused a family over such a spelling, or a family
counting one tool twice, is a false difference. :func:`tool_key` folds case and drops the
separators (``_``, ``-``, space and camelCase boundaries); :class:`ToolNames` also drops a trailing
``file`` or ``tool`` word, but only when the shortened key names exactly one known tool.

Stored names are never rewritten: contracts, rules and task schemas keep their own spelling, and
these keys are used only to decide whether two names denote the same tool. Everything here is a
pure function of the names given, so the same names always match the same way.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

DROPPED_SUFFIXES: frozenset[str] = frozenset({"file", "tool"})
"""Trailing words dropped from a key when the shortened key is unambiguous."""

_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z0-9])|[A-Z]?[a-z0-9]+|[A-Z]+")


def _words(name: str) -> tuple[str, ...]:
    """Split a tool name into case-folded words at separators and camelCase boundaries."""
    return tuple(
        word.casefold() for part in re.split(r"[\s_\-]+", name) for word in _WORD.findall(part)
    )


def tool_key(name: str) -> str:
    """Return the case- and separator-insensitive key of one tool name.

    ``Write``, ``write`` and ``WRITE`` share a key, as do ``write_file``, ``write-file`` and
    ``WriteFile``; ``write`` and ``write_file`` do not (see :class:`ToolNames`).
    """
    return "".join(_words(name)) or name.casefold()


def _short_key(name: str) -> str:
    """Return the key with one trailing ``file`` or ``tool`` word dropped, when one is there."""
    words = _words(name)
    if len(words) > 1 and words[-1] in DROPPED_SUFFIXES:
        words = words[:-1]
    return "".join(words) or name.casefold()


class ToolNames:
    """The tools a set of stored names denotes, matched by canonical key.

    A name matches the stored names sharing its :func:`tool_key`. Failing that, it matches by
    its key with a trailing ``file``/``tool`` word dropped, but only when exactly one stored tool
    shortens to that key: with both ``write`` and ``write_file`` stored, ``Write`` matches
    ``write`` alone and ``write_file`` is never folded into it.
    """

    def __init__(self, names: Iterable[str]) -> None:
        """Index the stored names."""
        self._by_key: dict[str, set[str]] = {}
        for name in names:
            self._by_key.setdefault(tool_key(name), set()).add(name)
        self._by_short: dict[str, set[str]] = {}
        for key, stored in self._by_key.items():
            self._by_short.setdefault(_short_key(next(iter(stored))), set()).add(key)

    def match(self, name: str) -> tuple[str, ...]:
        """Return the stored names that denote the same tool as ``name``, sorted, or ()."""
        key = tool_key(name)
        if key not in self._by_key:
            keys = self._by_short.get(_short_key(name), set())
            if len(keys) != 1:
                return ()
            key = next(iter(keys))
        return tuple(sorted(self._by_key[key]))

    def uncovered(self, names: Iterable[str]) -> tuple[str, ...]:
        """Return the names, in order without repeats, that match no stored tool."""
        return tuple(dict.fromkeys(name for name in names if not self.match(name)))

    def covers(self, names: Iterable[str]) -> bool:
        """Return whether every name matches a stored tool."""
        return not self.uncovered(names)

    def distinct(self) -> int:
        """Return how many distinct tools the stored names denote (spellings counted once)."""
        return len(self._by_key)
