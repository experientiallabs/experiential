"""Tests for package-owned producer revision resolution."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

import pytest

from exp.common import release_revision
from exp.common.release_revision import ReleaseRevisionError, installed_release_revision


def test_explicit_release_revision_is_authoritative(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the exact release pin without reading installed metadata."""
    expected = "a" * 40
    monkeypatch.setenv("EXP_RELEASE_REVISION", expected)

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
    monkeypatch.setenv("EXP_RELEASE_REVISION", configured)

    with pytest.raises(ReleaseRevisionError, match="full lowercase 40-hex"):
        installed_release_revision()


def test_installed_distribution_identity_is_package_owned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ignore the caller's Git checkout and bind the installed package version."""
    monkeypatch.delenv("EXP_RELEASE_REVISION", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(release_revision.metadata, "version", lambda _name: "0.3.0")

    assert installed_release_revision() == "experiential==0.3.0"


def test_missing_or_malformed_installed_metadata_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject missing and malformed installed distribution provenance."""
    monkeypatch.delenv("EXP_RELEASE_REVISION", raising=False)

    def missing(_name: str) -> str:
        """Model an environment without the installed distribution metadata."""
        raise metadata.PackageNotFoundError("experiential")

    monkeypatch.setattr(release_revision.metadata, "version", missing)
    with pytest.raises(ReleaseRevisionError, match="metadata is unavailable"):
        installed_release_revision()

    monkeypatch.setattr(release_revision.metadata, "version", lambda _name: "bad version")
    with pytest.raises(ReleaseRevisionError, match="version metadata is invalid"):
        installed_release_revision()


def test_execution_source_identity_tracks_actual_bytes_without_git_or_test_noise(
    tmp_path: Path,
) -> None:
    """Moving a package preserves identity; same-length edits with restored mtime do not."""
    import os
    import shutil

    package = tmp_path / "package"
    package.mkdir()
    source = package / "implementation.py"
    source.write_text("value = 1\n")
    prompt = package / "prompt.txt"
    prompt.write_text("Original instructions")
    baseline = release_revision._package_source_digest(package)
    (package / "implementation_test.py").write_text("irrelevant test changes")
    assert release_revision._package_source_digest(package) == baseline
    moved = tmp_path / "moved"
    shutil.copytree(package, moved)
    assert release_revision._package_source_digest(moved) == baseline
    stamp = source.stat()
    source.write_text("value = 2\n")
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert release_revision._package_source_digest(package) != baseline
    source.write_text("value = 1\n")
    prompt.write_text("Different instructions")
    assert release_revision._package_source_digest(package) != baseline


def test_execution_revision_includes_source_when_release_pin_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A release label cannot erase an editable package's behavior identity."""
    monkeypatch.setenv("EXP_RELEASE_REVISION", "a" * 40)
    monkeypatch.setattr(release_revision, "_package_source_digest", lambda package: "b" * 64)
    first = release_revision.installed_execution_revision()
    monkeypatch.setattr(release_revision, "_package_source_digest", lambda package: "c" * 64)
    assert release_revision.installed_execution_revision() != first
    assert first == "a" * 40 + ";source-sha256=" + "b" * 64


def test_execution_identity_ignores_read_access_timestamp_changes(tmp_path: Path) -> None:
    """Reading a fresh copied package can update atime without any source modification."""
    import os

    path = tmp_path / "implementation.py"
    path.write_text("value = 1\n")
    stamp = path.stat()
    os.utime(path, ns=(0, stamp.st_mtime_ns))
    first = release_revision._package_source_digest(tmp_path)
    assert release_revision._package_source_digest(tmp_path) == first
