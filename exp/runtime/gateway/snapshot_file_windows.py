"""Windows handle-backed snapshot reads that reject reparse points and replacement races."""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path, PureWindowsPath
from typing import BinaryIO

if sys.platform == "win32":
    import msvcrt

_READ_ATTRIBUTES = 0x0080
_GENERIC_READ = 0x80000000
_SHARE_READ = 0x00000001
_OPEN_EXISTING = 3
_BACKUP_SEMANTICS = 0x02000000
_OPEN_REPARSE_POINT = 0x00200000
_ATTRIBUTE_DIRECTORY = 0x10
_ATTRIBUTE_REPARSE_POINT = 0x400
_FILE_TYPE_DISK = 1
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)


def _windows_error() -> OSError:
    """Capture the thread-local Win32 error only on the supported platform."""
    if sys.platform != "win32":
        return OSError("Windows snapshot backend requires Windows")
    return ctypes.WinError(ctypes.get_last_error())


class _AttributeTag(ctypes.Structure):
    """The exact FILE_ATTRIBUTE_TAG_INFO layout returned by kernel32."""

    _fields_ = [("attributes", wintypes.DWORD), ("tag", wintypes.DWORD)]


class _BasicInfo(ctypes.Structure):
    """FILE_BASIC_INFO carries metadata change time independently of file creation time."""

    _fields_ = [
        ("created", ctypes.c_longlong),
        ("accessed", ctypes.c_longlong),
        ("written", ctypes.c_longlong),
        ("changed", ctypes.c_longlong),
        ("attributes", wintypes.DWORD),
    ]


class _FileId(ctypes.Structure):
    """The FILE_ID_INFO volume and 128-bit identity returned by kernel32."""

    _fields_ = [("volume", ctypes.c_ulonglong), ("identity", ctypes.c_ubyte * 16)]


class _WindowsFiles:
    """Small typed-at-ABI kernel32 handle owner; constructed only on Windows."""

    api: ctypes.CDLL

    def __init__(self) -> None:
        """Declare pointer-sized handles and every used argument before making a call."""
        if sys.platform != "win32":
            raise RuntimeError("Windows snapshot backend requires Windows")
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        self.api.CreateFileW.restype = wintypes.HANDLE
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.api.GetFileInformationByHandleEx.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self.api.GetFileInformationByHandleEx.restype = wintypes.BOOL
        self.api.GetFinalPathNameByHandleW.argtypes = [
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        self.api.GetFinalPathNameByHandleW.restype = wintypes.DWORD
        self.api.GetFileType.argtypes = [wintypes.HANDLE]
        self.api.GetFileType.restype = wintypes.DWORD

    def open(self, path: str, *, directory: bool, metadata_anchor: bool = False) -> int:
        """Open without following the leaf reparse point or sharing writes and renames.

        Args:
            path: Absolute path reached through already checked ancestors.
            directory: Whether the opened object must be a directory rather than a disk file.
            metadata_anchor: Retain identity only, sharing reads, writes, deletes and renames.

        Returns:
            An owned handle the caller must close or transfer exactly once to the CRT.

        Raises:
            OSError: The open or handle inspection fails, including sharing conflicts.
            ValueError: The handle identifies a reparse point or an unexpected object type.
        """
        handle = self.api.CreateFileW(
            path,
            0 if metadata_anchor else (_READ_ATTRIBUTES if directory else _GENERIC_READ),
            7 if metadata_anchor else _SHARE_READ,
            None,
            _OPEN_EXISTING,
            _BACKUP_SEMANTICS | _OPEN_REPARSE_POINT,
            None,
        )
        if handle == _INVALID_HANDLE_VALUE or handle is None:
            raise _windows_error()
        result = int(handle)
        try:
            info = _AttributeTag()
            if not self.api.GetFileInformationByHandleEx(
                result, 9, ctypes.byref(info), ctypes.sizeof(info)
            ):
                raise _windows_error()
            if info.attributes & _ATTRIBUTE_REPARSE_POINT:
                raise ValueError("budget snapshot path contains a reparse point")
            if bool(info.attributes & _ATTRIBUTE_DIRECTORY) != directory:
                raise ValueError("budget snapshot path has an unexpected file type")
            if not directory and self.api.GetFileType(result) != _FILE_TYPE_DISK:
                raise ValueError("budget snapshot must be a regular disk file")
            return result
        except BaseException:
            self.close(result)
            raise

    def final_path(self, handle: int) -> PureWindowsPath:
        """Read normalized DOS or UNC identity from the opened handle, never the input spelling."""
        size = self.api.GetFinalPathNameByHandleW(handle, None, 0, 0)
        if size == 0 or size > 32768:
            raise _windows_error()
        buffer = ctypes.create_unicode_buffer(size + 1)
        written = self.api.GetFinalPathNameByHandleW(handle, buffer, len(buffer), 0)
        if written == 0 or written >= len(buffer):
            raise _windows_error()
        return PureWindowsPath(buffer.value)

    def identity(self, handle: int) -> tuple[int, bytes]:
        """Return stable volume and file identity without transferring handle ownership."""
        info = _FileId()
        if not self.api.GetFileInformationByHandleEx(
            handle, 18, ctypes.byref(info), ctypes.sizeof(info)
        ):
            raise _windows_error()
        return int(info.volume), bytes(info.identity)

    def close(self, handle: int) -> None:
        """Close one owned handle, surfacing OS failure rather than leaking silently."""
        if not self.api.CloseHandle(handle):
            raise _windows_error()


@contextmanager
def windows_snapshot_observation(
    root: Path, relative_path: str
) -> Iterator[tuple[BinaryIO | None, tuple[tuple[int, bytes], ...]]]:
    """Hold checked ancestors even when the first missing component prevents a leaf open.

    Handles deny write and delete sharing, so a directory cannot be retargeted to
    a junction or renamed between component validation and the leaf open. The
    final file denies writers too. A conflicting existing writer fails closed.
    """
    if os.name != "nt":
        raise RuntimeError("Windows snapshot backend requires Windows")
    relative = PureWindowsPath(relative_path)
    if (
        relative.is_absolute()
        or relative.drive
        or not relative.parts
        or any(
            part in (".", "..")
            or ":" in part
            or part.endswith((".", " "))
            or part.split(".", 1)[0].upper() in _RESERVED_NAMES
            for part in relative.parts
        )
    ):
        raise ValueError("budget snapshot path must be a relative ordinary file path")
    api = _WindowsFiles()
    handles: list[int] = []
    leaf: int | None = None
    try:
        root_handle = api.open(str(root.resolve()), directory=True)
        handles.append(root_handle)
        current = api.final_path(root_handle)
        trusted = current
        identities = [api.identity(root_handle)]
        try:
            for part in relative.parts[:-1]:
                expected = current / part
                handle = api.open(str(expected), directory=True)
                handles.append(handle)
                identities.append(api.identity(handle))
                current = api.final_path(handle)
                if current != expected or not current.is_relative_to(trusted):
                    raise ValueError("budget snapshot directory identity changed")
            expected = current / relative.parts[-1]
            leaf = api.open(str(expected), directory=False)
        except FileNotFoundError:
            yield None, tuple(identities)
            return
        final = api.final_path(leaf)
        if final != expected or not final.is_relative_to(trusted):
            raise ValueError("budget snapshot file identity changed")
        identities.append(api.identity(leaf))
        descriptor = msvcrt.open_osfhandle(leaf, os.O_RDONLY | os.O_BINARY)
        leaf = None  # Ownership transferred once to the CRT descriptor.
        try:
            stream = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            yield stream, tuple(identities)
    finally:
        close_error: OSError | None = None
        if leaf is not None:
            handles.append(leaf)
        for handle in reversed(handles):
            try:
                api.close(handle)
            except OSError as error:
                close_error = error
        if close_error is not None and sys.exc_info()[0] is None:
            raise close_error


def windows_change_time(descriptor: int) -> int | None:
    """Read FILE_BASIC_INFO.ChangeTime; unsupported queries disable optional memo reuse."""
    api = _WindowsFiles()
    info = _BasicInfo()
    if not api.api.GetFileInformationByHandleEx(
        msvcrt.get_osfhandle(descriptor), 0, ctypes.byref(info), ctypes.sizeof(info)
    ):
        return None
    return int(info.changed)


def windows_snapshot_anchor(stream: BinaryIO) -> int:
    """Keep file identity without retaining the operation's deny-write sharing policy.

    CreateFileW permits metadata access with desired access zero. All three share
    flags permit the publisher's FileRenameInfoEx POSIX replacement, whereas an
    ordinary MoveFileEx replacement can still refuse this open destination.
    The live strict operation handle protects the path while identity is checked.
    """
    if os.name != "nt":
        raise RuntimeError("Windows snapshot anchors require Windows")
    api = _WindowsFiles()
    original = msvcrt.get_osfhandle(stream.fileno())
    anchor = api.open(str(api.final_path(original)), directory=False, metadata_anchor=True)
    try:
        if api.identity(anchor) != api.identity(original):
            raise ValueError("snapshot anchor identity differs from the opened operation file")
        descriptor = msvcrt.open_osfhandle(anchor, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        api.close(anchor)
        raise
    return descriptor


@contextmanager
def windows_snapshot_stream(root: Path, relative_path: str) -> Iterator[BinaryIO]:
    """Require a present regular file while retaining every checked Windows handle."""
    with windows_snapshot_observation(root, relative_path) as (stream, _directories):
        if stream is None:
            raise FileNotFoundError(relative_path)
        yield stream
