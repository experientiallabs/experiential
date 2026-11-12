"""Pointer keys and the files-map shape."""

from __future__ import annotations

import pytest

from exp.optimize.evaluation.variance.state_pointers import keyed_by_paths, split


def test_a_files_map_is_keyed_by_paths_and_named_fields_are_not() -> None:
    """A key holding a slash or a dot names a file; an empty object says neither."""
    assert keyed_by_paths({".teich-prompt.txt": "the task", "other_entries": [".pi"]})
    assert keyed_by_paths({"confcheck/__init__.py": ""})
    assert not keyed_by_paths({"oracle": {}}) and not keyed_by_paths({})
    assert not keyed_by_paths(["a.py"])


def test_pointers_split_into_unescaped_keys() -> None:
    """Escapes decode per RFC 6901; relative pointers and bad escapes are refused."""
    assert split("") == []
    assert split("/files/pkg~1mod.py/a~0b") == ["files", "pkg/mod.py", "a~b"]
    with pytest.raises(ValueError, match="absolute"):
        split("files")
    with pytest.raises(ValueError, match="escape"):
        split("/files/~2")
