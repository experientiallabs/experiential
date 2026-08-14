"""Tests for package-owned producer revision resolution."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

import pytest

from wmo.common import release_revision
from wmo.common.release_revision import ReleaseRevisionError, installed_release_revision


def test_explicit_release_revision_is_authoritative(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the exact release pin without reading installed metadata."""
    expected = "a" * 40
    monkeypatch.setenv("WMO_RELEASE_REVISION", expected)

    def unexpected_metadata_read(distribution: str) -> str:
        """Fail if the explicit release pin does not short-circuit metadata lookup."""
        raise AssertionError(f"unexpected metadata read for {distribution}")

    monkeypatch.setattr(release_revision.metadata, "version", unexpected_metadata_read)

    assert installed_release_revision() == expected


@pytest.mark.parametrize("configured", ["", "HEAD", "A" * 40, "a" * 39, "a" * 41])
def test_explicit_release_revision_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
) -> None:
    """Reject symbolic, empty, uppercase, and incorrectly sized producer pins."""
    monkeypatch.setenv("WMO_RELEASE_REVISION", configured)

    with pytest.raises(ReleaseRevisionError, match="full lowercase 40-hex"):
        installed_release_revision()


def test_installed_distribution_identity_is_package_owned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ignore the caller's Git checkout and bind the installed package version."""
    monkeypatch.delenv("WMO_RELEASE_REVISION", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(release_revision.metadata, "version", lambda _name: "0.3.0")

    assert installed_release_revision() == "world-model-optimizer==0.3.0"


def test_missing_or_malformed_installed_metadata_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject missing and malformed installed distribution provenance."""
    monkeypatch.delenv("WMO_RELEASE_REVISION", raising=False)

    def missing(_name: str) -> str:
        """Model an environment without the installed distribution metadata."""
        raise metadata.PackageNotFoundError("world-model-optimizer")

    monkeypatch.setattr(release_revision.metadata, "version", missing)
    with pytest.raises(ReleaseRevisionError, match="metadata is unavailable"):
        installed_release_revision()

    monkeypatch.setattr(release_revision.metadata, "version", lambda _name: "bad version")
    with pytest.raises(ReleaseRevisionError, match="version metadata is invalid"):
        installed_release_revision()
