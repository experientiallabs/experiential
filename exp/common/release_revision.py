"""Package-owned producer revision resolution for persisted artifacts."""

from __future__ import annotations

import hashlib
import os
import re
from importlib import metadata
from pathlib import Path

_DISTRIBUTION = "experiential"
_REVISION_ENV = "EXP_RELEASE_REVISION"
_EXACT_GIT_REVISION = re.compile(r"[0-9a-f]{40}")
_SAFE_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+!-]*")


class ReleaseRevisionError(ValueError):
    """Report unavailable or invalid package-owned producer provenance."""


def installed_release_revision() -> str:
    """Resolve the producer identity without consulting the host repository.

    An explicit release build pin is authoritative when present. Normal installed packages use
    their immutable distribution name and version, which identifies the package that owns the
    executing code without inheriting Git state from the caller's working directory.

    Returns:
        An exact Git revision from ``EXP_RELEASE_REVISION`` or the installed distribution identity.

    Raises:
        ReleaseRevisionError: The configured revision or installed package metadata is invalid.
    """
    configured = os.environ.get(_REVISION_ENV)
    if configured is not None:
        if _EXACT_GIT_REVISION.fullmatch(configured) is None:
            raise ReleaseRevisionError(
                f"{_REVISION_ENV} must be a full lowercase 40-hex Git revision"
            )
        return configured
    try:
        version = metadata.version(_DISTRIBUTION)
    except metadata.PackageNotFoundError as exc:
        raise ReleaseRevisionError(
            "installed experiential package metadata is unavailable; "
            f"set {_REVISION_ENV} to the exact producer revision"
        ) from exc
    if _SAFE_VERSION.fullmatch(version) is None:
        raise ReleaseRevisionError("installed experiential version metadata is invalid")
    return f"{_DISTRIBUTION}=={version}"


def installed_execution_revision() -> str:
    """Bind the producer release to the installed EXP source and runtime resource bytes.

    Editable installs can change without changing their distribution version or Git commit.
    Executable studies therefore bind the package's actual Python and resource files, using
    relative names so an identical installation may move. Tests and build caches are excluded.
    Third-party libraries, the interpreter, and explicitly selected external factories remain
    separately owned dependencies; this is not an environment image or a sandbox certificate.
    """
    digest = _package_source_digest(Path(__file__).resolve().parents[1])
    return f"{installed_release_revision()};source-sha256={digest}"


def _package_source_digest(package: Path) -> str:
    """Hash production source/resources without Git, credentials, timestamps or bytecode."""
    ignored = {"__pycache__", "tests", "target", "node_modules"}
    rows = []
    for directory, directories, files in package.walk():
        directories[:] = [
            name for name in directories if name not in ignored and not name.startswith(".")
        ]
        for name in directories:
            if (directory / name).is_symlink():
                raise ReleaseRevisionError(
                    "execution source must not contain symlinked directories"
                )
        for name in files:
            path = directory / name
            if name.startswith(".") or name.endswith("_test.py") or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink() or not path.is_file():
                raise ReleaseRevisionError(
                    "execution source must contain only regular package files"
                )
            before = path.stat()
            payload = path.read_bytes()
            after = path.stat()
            if (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) != (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ):
                raise ReleaseRevisionError(
                    "execution source changed while its identity was computed"
                )
            rows.append((path.relative_to(package).as_posix(), hashlib.sha256(payload).hexdigest()))
    if not rows or not any(name.endswith(".py") for name, _ in rows):
        raise ReleaseRevisionError("execution identity requires readable installed Python source")
    digest = hashlib.sha256()
    for name, content in sorted(rows):
        digest.update(name.encode() + b"\0" + content.encode() + b"\n")
    return digest.hexdigest()
