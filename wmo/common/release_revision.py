"""Package-owned producer revision resolution for persisted artifacts."""

from __future__ import annotations

import os
import re
from importlib import metadata

_DISTRIBUTION = "world-model-optimizer"
_REVISION_ENV = "WMO_RELEASE_REVISION"
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
        An exact Git revision from ``WMO_RELEASE_REVISION`` or the installed distribution identity.

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
            "installed world-model-optimizer package metadata is unavailable; "
            f"set {_REVISION_ENV} to the exact producer revision"
        ) from exc
    if _SAFE_VERSION.fullmatch(version) is None:
        raise ReleaseRevisionError("installed world-model-optimizer version metadata is invalid")
    return f"{_DISTRIBUTION}=={version}"
