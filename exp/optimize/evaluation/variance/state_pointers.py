"""RFC 6901 pointer keys and the files-map shape of a start's simulated state.

A files map (:func:`keyed_by_paths`) holds each file under one flat key such as
``confcheck/__init__.py``; the simulator never nests directories in it. A start's domain
``state_schema`` is derived from that shape, and authored pointers are split into keys here.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonValue


def split(pointer: str) -> list[str]:
    """A pointer's unescaped keys; raises ValueError for a relative pointer or bad escape."""
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise ValueError("pointer must be absolute")
    keys = []
    for raw in pointer[1:].split("/"):
        if "~" in raw.replace("~0", "").replace("~1", ""):
            raise ValueError("invalid pointer escape")
        keys.append(raw.replace("~1", "/").replace("~0", "~"))
    return keys


def _path_key(key: str) -> bool:
    """Whether a key names a file by path or file name (it holds a ``/`` or ``.``)."""
    return "/" in key or "." in key


def keyed_by_paths(value: JsonValue) -> bool:
    """Whether an object is a map keyed by file paths or names, not an object of named fields.

    This is the shape contract a start's domain ``state_schema`` is derived from: an object any
    of whose keys holds a ``/`` or ``.`` is a files map, which may gain and lose entries, each
    under one key. An empty object says neither.
    """
    return isinstance(value, dict) and any(_path_key(key) for key in value)
