"""Bounded regular-file reads and operation-scoped path proofs beneath gateway state."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from typing import BinaryIO

from exp.common.config.settings import GatewayResourceSettings
from exp.runtime.gateway.snapshot_file_windows import (
    windows_change_time,
    windows_snapshot_anchor,
    windows_snapshot_observation,
)

_PathIdentity = tuple[tuple[int, bytes], ...]
_FileStamp = tuple[int, int, int, int, int, int, int | None]


class SnapshotSizeError(ValueError):
    """A secure snapshot read exceeded its explicit resource policy.

    Attributes:
        size: Observed bytes needed, possibly a lower bound when the file grew.
        maximum: Configured positive per-file byte limit.
    """

    def __init__(self, size: int, maximum: int) -> None:
        """Retain measured bounds so each consumer can name its own repair setting."""
        self.size = size
        self.maximum = maximum
        super().__init__(
            f"budget snapshot needs {size} bytes, above the configured {maximum}-byte "
            "resource budget; raise [gateway].budget_snapshot_max_bytes in the project "
            "settings.toml, or pass snapshot_max_bytes to SQLiteBudgetStore"
        )


@contextmanager
def _snapshot_observation(
    root: Path, relative_path: str
) -> Iterator[tuple[BinaryIO | None, _PathIdentity]]:
    """Retain all no-follow handles, including ancestors of the first absent component."""
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
        with windows_snapshot_observation(root, relative_path) as observation:
            yield observation
        return
    if os.open not in os.supports_dir_fd or any(
        not hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    ):
        raise ValueError("secure snapshot reads are unsupported on this operating system")
    directories = [os.open(root.resolve(), os.O_RDONLY | os.O_DIRECTORY)]
    identities = [_handle_identity(directories[0])]
    try:
        try:
            for part in relative.parts[:-1]:
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directories[-1]
                )
                directories.append(child)
                identities.append(_handle_identity(child))
            descriptor = os.open(
                relative.parts[-1],
                os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
                dir_fd=directories[-1],
            )
        except FileNotFoundError:
            yield None, tuple(identities)
            return
        try:
            stream = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("budget catalog snapshot must be a regular file")
            identities.append(_handle_identity(stream.fileno()))
            yield stream, tuple(identities)
    finally:
        for directory in reversed(directories):
            os.close(directory)


def _handle_identity(descriptor: int) -> tuple[int, bytes]:
    """Identify a retained POSIX inode without treating sibling activity as path drift."""
    info = os.fstat(descriptor)
    return info.st_dev, info.st_ino.to_bytes(16, "big")


def _file_stamp(stream: BinaryIO) -> _FileStamp:
    """Capture content-sensitive file metadata; Windows handles additionally deny writers."""
    info = os.fstat(stream.fileno())
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        windows_change_time(stream.fileno()) if os.name == "nt" else info.st_ctime_ns,
    )


@contextmanager
def snapshot_stream(root: Path, relative_path: str) -> Iterator[BinaryIO]:
    """Open one required descendant without following symlink or reparse-point components."""
    with _snapshot_observation(root, relative_path) as (stream, _identities):
        if stream is None:
            raise FileNotFoundError(relative_path)
        yield stream


def _read_stream_bytes(stream: BinaryIO, maximum_bytes: int) -> bytes:
    """Bound an already-secure stream and detect growth, rewriting, or truncation."""
    maximum = GatewayResourceSettings(
        budget_snapshot_max_bytes=maximum_bytes
    ).budget_snapshot_max_bytes
    before = _file_stamp(stream)
    if before[3] > maximum:
        raise SnapshotSizeError(before[3], maximum)
    chunks: list[bytes] = []
    size = 0
    while chunk := stream.read(min(1024 * 1024, maximum - size + 1)):
        chunks.append(chunk)
        size += len(chunk)
        if size > maximum:
            raise SnapshotSizeError(size, maximum)
    if size != before[3] or _file_stamp(stream) != before:
        raise ValueError("budget catalog snapshot changed or was truncated during read")
    return b"".join(chunks)


def read_snapshot_bytes(root: Path, relative_path: str, maximum_bytes: int) -> bytes:
    """Read with an explicit resource budget, detecting growth, replacement, and truncation."""
    with snapshot_stream(root, relative_path) as stream:
        return _read_stream_bytes(stream, maximum_bytes)


class PreparedSnapshotFile:
    """Live path observation owned exclusively by one preflight context.

    Attributes:
        root: Trusted gateway directory, not an inferred settings root.
        relative_path: Unresolved descendant spelling used for every secure walk.
    """

    def __init__(
        self,
        root: Path,
        relative_path: str,
        stream: BinaryIO | None,
        identities: _PathIdentity,
        content: bytes | None,
    ) -> None:
        """Bind retained handles and release parsed bytes when the consumer takes them."""
        self.root = root
        self.relative_path = relative_path
        self._stream = stream
        self._identities = identities
        self._stamp = None if stream is None else _file_stamp(stream)
        self._content = content
        self._closed = False

    @property
    def generation(self) -> tuple[_PathIdentity, _FileStamp | None]:
        """Return immutable path and content stamps, with absence represented explicitly."""
        return self._identities, self._stamp

    def read_bytes(self, maximum_bytes: int) -> bytes | None:
        """Read the already-open operation handle only when classification is not reusable."""
        if self._closed:
            raise ValueError("snapshot preflight is closed")
        content = None if self._stream is None else _read_stream_bytes(self._stream, maximum_bytes)
        self.validate_current()
        return content

    def retain_leaf(self) -> int | None:
        """Retain only a present leaf's identity; the caller owns and must close its descriptor."""
        if self._closed:
            raise ValueError("snapshot preflight is closed")
        if self._stream is None:
            return None
        if self._stamp is None or self._stamp[6] is None:
            raise ValueError("snapshot change identity is unavailable for memo reuse")
        return (
            windows_snapshot_anchor(self._stream)
            if os.name == "nt"
            else os.dup(self._stream.fileno())
        )

    def anchor_matches(self, descriptor: int | None) -> bool:
        """Compare the retained cache inode with this independently opened operation inode."""
        if self._stamp is None:
            return descriptor is None
        if descriptor is None:
            return False
        info = os.fstat(descriptor)
        return self._stamp == (
            info.st_dev,
            info.st_ino,
            stat.S_IFMT(info.st_mode),
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            windows_change_time(descriptor) if os.name == "nt" else info.st_ctime_ns,
        )

    def take_bytes(self) -> bytes | None:
        """Transfer the bounded bytes once; an absent path returns no bytes."""
        content, self._content = self._content, None
        return content

    def validate_current(self) -> None:
        """Fence path and retained inode identity with no content reads or parsing."""
        if self._closed:
            raise ValueError("snapshot preflight is closed")
        with _snapshot_observation(self.root, self.relative_path) as (current, identities):
            stamp = None if current is None else _file_stamp(current)
            if (
                identities != self._identities
                or stamp != self._stamp
                or (self._stream is not None and _file_stamp(self._stream) != self._stamp)
            ):
                raise ValueError("serving snapshot changed after preflight; retry the operation")


@contextmanager
def prepare_snapshot_file(
    root: Path,
    relative_path: str,
    maximum_bytes: int,
    *,
    read_content: bool = True,
) -> Iterator[PreparedSnapshotFile]:
    """Open and bound the file, optionally reading now, while retaining the secure path proof."""
    with _snapshot_observation(root, relative_path) as (stream, identities):
        prepared = PreparedSnapshotFile(root, relative_path, stream, identities, None)
        try:
            if prepared._stamp is not None and prepared._stamp[3] > maximum_bytes:
                raise SnapshotSizeError(prepared._stamp[3], maximum_bytes)
            if read_content:
                prepared._content = prepared.read_bytes(maximum_bytes)
            prepared.validate_current()
            yield prepared
        finally:
            prepared._closed = True
            prepared._content = None
