"""Tests for the local code revision recorded by CLI commands."""

from __future__ import annotations

import subprocess

import pytest

from wmo.cli import revision


def test_current_revision_reads_head_and_falls_back_when_git_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A readable HEAD is used verbatim; a failed or empty read stays explicitly unversioned."""

    def completed(returncode: int, stdout: str) -> subprocess.CompletedProcess[str]:
        """Return one stubbed `git rev-parse` result."""
        return subprocess.CompletedProcess(args=["git"], returncode=returncode, stdout=stdout)

    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: completed(0, "e7aad17\n"))
    assert revision.current_revision() == "e7aad17"

    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: completed(128, ""))
    assert revision.current_revision() == "local-unversioned"

    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: completed(0, "\n"))
    assert revision.current_revision() == "local-unversioned"
