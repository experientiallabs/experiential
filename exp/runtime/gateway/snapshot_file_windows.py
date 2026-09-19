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

    def open(self, path: str, *, directory: bool) -> int:
        """Open without following the leaf reparse point or sharing writes and renames."""
        handle = self.api.CreateFileW(
            path,
            _READ_ATTRIBUTES if directory else _GENERIC_READ,
            _SHARE_READ,
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

    def close(self, handle: int) -> None:
        """Close one owned handle, surfacing OS failure rather than leaking silently."""
        if not self.api.CloseHandle(handle):
            raise _windows_error()


@contextmanager
def windows_snapshot_stream(root: Path, relative_path: str) -> Iterator[BinaryIO]:
    """Hold every checked directory while opening and reading its regular descendant.

    Handles deny write and delete sharing, so a directory cannot be retargeted to
    a junction or renamed between component validation and the leaf open. The
    final file denies writers too. A conflicting existing writer fails closed.
    """
    if os.name != "nt":
        raise RuntimeError("Windows snapshot backend requires Windows")
    import msvcrt  # Windows-only CRT bridge; importing elsewhere is not supported.

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
        for part in relative.parts[:-1]:
            expected = current / part
            handle = api.open(str(expected), directory=True)
            handles.append(handle)
            current = api.final_path(handle)
            if current != expected or not current.is_relative_to(trusted):
                raise ValueError("budget snapshot directory identity changed")
        expected = current / relative.parts[-1]
        leaf = api.open(str(expected), directory=False)
        final = api.final_path(leaf)
        if final != expected or not final.is_relative_to(trusted):
            raise ValueError("budget snapshot file identity changed")
        descriptor = msvcrt.open_osfhandle(leaf, os.O_RDONLY | os.O_BINARY)
        leaf = None  # Ownership transferred once to the CRT descriptor.
        try:
            stream = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            yield stream
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
