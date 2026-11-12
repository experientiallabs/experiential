"""Durable publication and secure reads of digest-addressed external artifact files."""

import hashlib
import os
import stat
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from uuid import uuid4

_BlobKey = tuple[str, str, str]
_BlobStat = tuple[int, int, int, int, int]


class VerifiedBlobs:
    """Process-local reuse of digest-verified blob bytes, bounded by total size.

    A blob is named by its SHA-256 and never rewritten, so bytes already read and verified
    against that digest can be reused while the file is provably the same one: same device,
    inode, size, modification and change time (any rewrite, replacement or metadata change of
    the file moves its change time). The secure descendant walk, the open and the ``fstat`` still
    happen on every read; only re-reading and re-hashing the bytes is skipped. Large immutable
    inputs (a 1.4 GB trace dataset, 2.5 GB RAG indexes) were otherwise read and hashed tens of
    times per front-door start.

    Attributes:
        maximum_bytes: Total payload bytes kept; least recently used blobs are forgotten first.
    """

    def __init__(self, maximum_bytes: int) -> None:
        """Start empty; a zero bound disables reuse."""
        self.maximum_bytes = maximum_bytes
        self._entries: OrderedDict[_BlobKey, tuple[_BlobStat, bytes]] = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()

    def get(self, key: _BlobKey, signature: _BlobStat) -> bytes | None:
        """Return remembered bytes for an unchanged file, or None."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry[0] != signature:
                self._size -= len(entry[1])
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return entry[1]

    def put(self, key: _BlobKey, signature: _BlobStat, payload: bytes) -> None:
        """Remember verified bytes, forgetting the oldest beyond the bound."""
        if len(payload) > self.maximum_bytes:
            return
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._size -= len(old[1])
            self._entries[key] = (signature, payload)
            self._size += len(payload)
            while self._size > self.maximum_bytes:
                _, (_, dropped) = self._entries.popitem(last=False)
                self._size -= len(dropped)

    def clear(self) -> None:
        """Forget every blob."""
        with self._lock:
            self._entries.clear()
            self._size = 0


VERIFIED_BLOBS = VerifiedBlobs(int(os.environ.get("EXP_VERIFIED_BLOB_CACHE_BYTES", 8 << 30)))
"""Verified blob bytes reused within this process (``EXP_VERIFIED_BLOB_CACHE_BYTES``, 0 = off)."""


def _read_artifact_file_snapshot(
    directory: Path,
    relative_path: str,
    *,
    digest: str | None = None,
    mismatch: Callable[[], Exception] | None = None,
) -> bytes:
    """Read a regular descendant without following a replaced path component.

    Args:
        directory: Content root, the boundary for every descendant open.
        relative_path: Validated descendant path.
        digest: The SHA-256 the bytes must have. When given, the bytes are verified against it
            and reused from :data:`VERIFIED_BLOBS` while the open file is provably unchanged.
        mismatch: Builds the error raised when the bytes do not match ``digest``.
    """
    if not hasattr(os, "O_NOFOLLOW"):
        raise OSError("secure artifact reads require O_NOFOLLOW")
    directory_fd = os.open(directory, os.O_RDONLY | os.O_NOFOLLOW)
    current_fd = directory_fd
    try:
        if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
            raise OSError("artifact directory is not a directory")
        parts = PurePosixPath(relative_path).parts
        for component in parts[:-1]:
            next_fd = os.open(component, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current_fd)
            try:
                if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                    raise OSError("artifact data path has a non-directory component")
            except OSError:
                os.close(next_fd)
                raise
            os.close(current_fd)
            current_fd = next_fd
        file_descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current_fd)
        try:
            status = os.fstat(file_descriptor)
            if not stat.S_ISREG(status.st_mode):
                raise OSError("artifact data path is not a regular file")
            if digest is None:
                return _read_file_descriptor(file_descriptor)
            key = (os.fspath(directory), relative_path, digest)
            signature = (
                status.st_dev,
                status.st_ino,
                status.st_size,
                status.st_mtime_ns,
                status.st_ctime_ns,
            )
            reused = VERIFIED_BLOBS.get(key, signature)
            if reused is not None:
                return reused
            payload = _read_file_descriptor(file_descriptor)
            if hashlib.sha256(payload).hexdigest() != digest:
                raise (mismatch or (lambda: OSError("artifact blob digest mismatch")))()
            VERIFIED_BLOBS.put(key, signature, payload)
            return payload
        finally:
            os.close(file_descriptor)
    finally:
        os.close(current_fd)


def _read_file_descriptor(file_descriptor: int) -> bytes:
    """Read all available bytes from one already-open regular file descriptor."""
    chunks: list[bytes] = []
    while chunk := os.read(file_descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _fsync_directory_strict(directory: Path) -> None:
    """Persist a staged directory before atomically exposing it to readers."""
    file_descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def publish_blob(root: Path, relative_directory: PurePosixPath, payload: bytes) -> str:
    """Publish and fsync immutable bytes before their database reference can commit.

    Args:
        root: Explicit content root, the boundary for all descendant directory opens.
        relative_directory: Validated project blob directory beneath the root.
        payload: Exact artifact bytes whose digest is the final blob name.

    Returns:
        SHA-256 digest naming the durably published blob.
    """
    if relative_directory.is_absolute() or ".." in relative_directory.parts:
        raise ValueError("Blob directory must be a relative descendant.")
    digest = hashlib.sha256(payload).hexdigest()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    _fsync_directory_strict(root.parent)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    staging = f".{uuid4().hex}.partial"
    try:
        for component in relative_directory.parts:
            try:
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            else:
                os.fsync(descriptor)
            next_descriptor = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = next_descriptor
        file_descriptor = os.open(
            staging,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            mode=0o600,
            dir_fd=descriptor,
        )
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(
                    staging,
                    digest,
                    src_dir_fd=descriptor,
                    dst_dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError:
                existing = os.open(digest, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
                try:
                    if not stat.S_ISREG(os.fstat(existing).st_mode) or (
                        _read_file_descriptor(existing) != payload
                    ):
                        raise ValueError(
                            "Existing artifact blob differs from its digest."
                        ) from None
                finally:
                    os.close(existing)
            os.fsync(descriptor)
        finally:
            os.unlink(staging, dir_fd=descriptor)
    finally:
        os.close(descriptor)
    return digest
