"""Every cross-process lock goes through the single-host guard in `exp.common.core.locks`.

OS file locks only exclude processes of one machine. `CoordinationLock` checks the directory's
host binding before each acquisition; a bare `filelock` lock or a raw `flock` would silently
skip that check, so production code may not construct one.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_OWNER = REPO_ROOT / "exp" / "common" / "core" / "locks.py"
_RAW_LOCKS = frozenset({"FileLock", "SoftFileLock", "UnixFileLock", "WindowsFileLock"})
_RAW_CALLS = frozenset({("fcntl", "flock"), ("fcntl", "lockf"), ("msvcrt", "locking")})


def _raw_lock_calls(source: str) -> list[int]:
    """Return line numbers of calls that create an unguarded cross-process lock."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name) and function.id in _RAW_LOCKS:
            lines.append(node.lineno)
        elif isinstance(function, ast.Attribute):
            owner = function.value
            if function.attr in _RAW_LOCKS or (
                isinstance(owner, ast.Name) and (owner.id, function.attr) in _RAW_CALLS
            ):
                lines.append(node.lineno)
    return lines


def test_production_code_creates_cross_process_locks_only_through_the_guard() -> None:
    """A new lock site must use `CoordinationLock` or `file_write_lock`."""
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for path in sorted((REPO_ROOT / "exp").rglob("*.py"))
        if not path.name.endswith("_test.py") and path != _OWNER and "tests" not in path.parts
        for line in _raw_lock_calls(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"unguarded cross-process locks: {offenders}"


def test_the_scan_recognizes_each_raw_lock_shape() -> None:
    """The scan is not vacuous: each forbidden construction is detected."""
    source = "\n".join(
        (
            "FileLock(p)",
            "filelock.SoftFileLock(p)",
            "fcntl.flock(fd, 2)",
            "msvcrt.locking(fd, 1, 1)",
            "CoordinationLock(p)",
        )
    )
    assert _raw_lock_calls(source) == [1, 2, 3, 4]
