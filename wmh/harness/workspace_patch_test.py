# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Tests for the transport-neutral incremental workspace patch protocol."""

from __future__ import annotations

import io
import tarfile

import pytest

from wmh.harness.workspace_patch import (
    WorkspacePatchError,
    build_workspace_patch,
    parse_workspace_patch,
)


def _archive(files: dict[str, tuple[bytes, int]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path, (content, mode) in files.items():
            info = tarfile.TarInfo(path)
            info.size = len(content)
            info.mode = mode
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_patch_round_trip_carries_add_modify_delete_and_mode() -> None:
    before = _archive(
        {
            "changed.txt": (b"before", 0o644),
            "deleted.txt": (b"remove", 0o644),
            "mode.txt": (b"same", 0o644),
        }
    )
    after = _archive(
        {
            "changed.txt": (b"after", 0o644),
            "added.txt": (b"new", 0o600),
            "mode.txt": (b"same", 0o755),
        }
    )

    content = build_workspace_patch(before, after)

    assert content is not None
    patch = parse_workspace_patch(content)
    assert [operation.path for operation in patch.operations] == [
        "added.txt",
        "changed.txt",
        "deleted.txt",
        "mode.txt",
    ]
    assert patch.files["added.txt"] == b"new"
    assert patch.files["changed.txt"] == b"after"
    assert "deleted.txt" not in patch.files
    assert patch.files["mode.txt"] == b"same"
    assert patch.operations[-1].after is not None
    assert patch.operations[-1].after.mode == 0o755


def test_patch_builder_returns_none_when_archives_match() -> None:
    archive = _archive({"same.txt": (b"same", 0o644)})

    assert build_workspace_patch(archive, archive) is None


def test_patch_parser_rejects_content_that_does_not_match_manifest() -> None:
    before = _archive({"changed.txt": (b"before", 0o644)})
    after = _archive({"changed.txt": (b"after", 0o644)})
    content = build_workspace_patch(before, after)
    assert content is not None

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as output:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as source:
            for member in source.getmembers():
                body = source.extractfile(member)
                data = body.read() if body is not None else b""
                if member.name == "data/changed.txt":
                    data = b"tampered"
                    member.size = len(data)
                output.addfile(member, io.BytesIO(data))

    with pytest.raises(WorkspacePatchError, match="digest"):
        parse_workspace_patch(buffer.getvalue())
