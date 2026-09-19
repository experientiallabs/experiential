"""Bounded regular-file reads beneath a trusted local gateway directory."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from typing import BinaryIO

from exp.common.config.settings import GatewayResourceSettings
from exp.runtime.gateway.snapshot_file_windows import windows_snapshot_stream


@contextmanager
def snapshot_stream(root: Path, relative_path: str) -> Iterator[BinaryIO]:
    """Open one descendant without following symlink or reparse-point components.

    The caller trusts root itself. POSIX uses directory-relative handles; Windows
    holds each checked directory against rename and reparse mutation until close.
    """
    relative = Path(relative_path)
    windows = PureWindowsPath(relative_path)
    if (
        relative.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or not relative.parts
        or any(part in (".", "..") for part in relative.parts)
    ):
        raise ValueError("budget catalog snapshot reference escapes gateway state")
    if os.name == "nt":
        with windows_snapshot_stream(root, relative_path) as stream:
            yield stream
        return
    if os.open not in os.supports_dir_fd or any(
        not hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    ):
        raise ValueError("secure snapshot reads are unsupported on this operating system")
    directory = os.open(root.resolve(), os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in relative.parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(
            relative.parts[-1], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory
        )
        try:
            stream = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("budget catalog snapshot must be a regular file")
            yield stream
    finally:
        os.close(directory)


def read_snapshot_bytes(root: Path, relative_path: str, maximum_bytes: int) -> bytes:
    """Read with an explicit resource budget, detecting growth, replacement, and truncation."""
    maximum = GatewayResourceSettings(
        budget_snapshot_max_bytes=maximum_bytes
    ).budget_snapshot_max_bytes
    with snapshot_stream(root, relative_path) as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > maximum:
            raise _size_error(before.st_size, maximum)
        chunks: list[bytes] = []
        size = 0
        while chunk := stream.read(min(1024 * 1024, maximum - size + 1)):
            chunks.append(chunk)
            size += len(chunk)
            if size > maximum:
                raise _size_error(size, maximum)
        after = os.fstat(stream.fileno())
        if (
            size != before.st_size
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ino != before.st_ino
            or after.st_dev != before.st_dev
        ):
            raise ValueError("budget catalog snapshot changed or was truncated during read")
    return b"".join(chunks)


def _size_error(size: int, maximum: int) -> ValueError:
    """Name the operator setting and constructor override that permit a larger reviewed file."""
    return ValueError(
        f"budget snapshot needs {size} bytes, above the configured {maximum}-byte resource budget; "
        "raise [gateway].budget_snapshot_max_bytes in the project settings.toml, "
        "or pass snapshot_max_bytes to SQLiteBudgetStore"
    )
