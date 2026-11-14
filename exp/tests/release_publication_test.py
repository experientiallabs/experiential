"""Execute the release source gate against protected and unprotected Git refs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NATIVE = "exp/runtime/gateway/native"


def _source_gate() -> str:
    """Extract the executable source validation step from the publishing workflow."""
    workflow = (ROOT / ".github/workflows/python-package.yml").read_text()
    return textwrap.dedent(
        workflow.split("          python3 - <<'PY'\n", 1)[1].split("          PY\n", 1)[0]
    )


@pytest.mark.parametrize(
    ("changes", "success"),
    [
        ({}, True),
        ({"REF_PROTECTED": "false"}, False),
        ({"GITHUB_REPOSITORY": "other/experiential"}, False),
        ({"PROJECT": "0.7.243rc2"}, False),
        ({"CARGO": "0.3.190-rc.2"}, False),
        ({"NATIVE": "0.3.190rc2"}, False),
        ({"CARGO": "0.3.190rc1"}, False),
        ({"DEFERRED": "true"}, False),
        ({"GITHUB_REF": "refs/tags/v0.7.243beta1"}, False),
        ({"GITHUB_REF": "refs/heads/candidate", "PUBLISH": "true"}, False),
        ({"GITHUB_REF": "refs/heads/candidate"}, True),
        ({"NATIVE": "0.3.188", "CARGO": "0.3.188"}, True),
        (
            {"PROJECT": "0.7.243", "GITHUB_REF": "refs/tags/v0.7.243", "ON_MAIN": "true"},
            False,
        ),
        (
            {
                "PROJECT": "0.7.243",
                "GITHUB_REF": "refs/tags/v0.7.243",
                "NATIVE": "0.3.190",
                "CARGO": "0.3.190",
            },
            False,
        ),
        (
            {
                "PROJECT": "0.7.243",
                "GITHUB_REF": "refs/tags/v0.7.243",
                "NATIVE": "0.3.190",
                "CARGO": "0.3.190",
                "ON_MAIN": "true",
            },
            True,
        ),
        ({"GITHUB_EVENT_NAME": "workflow_dispatch", "PUBLISH": "true"}, False),
        (
            {
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "GITHUB_REF": "refs/heads/main",
                "SOURCE": "latest-release",
                "PUBLISH": "true",
                "PROJECT": "0.7.243",
                "NATIVE": "0.3.190",
                "CARGO": "0.3.190",
                "ON_MAIN": "true",
            },
            True,
        ),
        (
            {
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "GITHUB_REF": "refs/heads/candidate",
                "SOURCE": "latest-release",
                "PUBLISH": "true",
            },
            False,
        ),
    ],
)
def test_publication_requires_protected_tags_and_stable_main_ancestry(
    tmp_path: Path, changes: dict[str, str], success: bool
) -> None:
    """Run the real gate in a local repository without build or publishing credentials.

    Args:
        tmp_path: Isolated repository and source metadata directory.
        changes: Trigger or package metadata differences from a valid branch candidate.
        success: Whether the source gate should admit this invocation.
    """
    values = {
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/tags/v0.7.243rc1",
        "GITHUB_REPOSITORY": "experientiallabs/experiential",
        "REF_PROTECTED": "true",
        "SOURCE": "selected-ref",
        "PUBLISH": "false",
        "PROJECT": "0.7.243rc1",
        "NATIVE": "0.3.190rc1",
        "CARGO": "0.3.190-rc.1",
        "ON_MAIN": "false",
        "DEFERRED": "false",
        **changes,
    }

    def git(*args: str) -> str:
        """Run a local Git operation with hooks and credential-backed signing disabled."""
        return subprocess.check_output(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "tag.gpgSign=false",
                "-c",
                "user.name=Release Test",
                "-c",
                "user.email=release@example.invalid",
                *args,
            ],
            cwd=tmp_path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    git("init", "-q", "-b", "main")
    git("commit", "--allow-empty", "-qm", "base")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    native = tmp_path / NATIVE
    native.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "experiential"\nversion = "{values["PROJECT"]}"\n'
    )
    (native / "pyproject.toml").write_text(
        f'[project]\nname = "exp-gateway-native"\nversion = "{values["NATIVE"]}"\n'
    )
    (native / "Cargo.toml").write_text(f'[package]\nversion = "{values["CARGO"]}"\n')
    if values["DEFERRED"] == "true":
        (tmp_path / ".github").mkdir()
        (tmp_path / ".github/deferred-native-release.toml").touch()
    git("add", ".")
    git("commit", "-qm", "candidate")
    revision = git("rev-parse", "HEAD")
    git("tag", "v" + values["PROJECT"])
    ref = values["GITHUB_REF"]
    if ref.startswith("refs/tags/") and ref != "refs/tags/v" + values["PROJECT"]:
        git("tag", ref.removeprefix("refs/tags/"))
    if values["ON_MAIN"] == "true":
        git("update-ref", "refs/remotes/origin/main", "HEAD")

    commands = tmp_path / "bin"
    commands.mkdir()
    release = {
        "tag_name": "v" + values["PROJECT"],
        "draft": False,
        "prerelease": "rc" in values["PROJECT"],
    }
    gh = commands / "gh"
    gh.write_text(f"#!{sys.executable}\nimport sys\nsys.stdout.write({json.dumps(release)!r})\n")
    gh.chmod(0o755)
    output = tmp_path / "output"
    result = subprocess.run(
        [sys.executable, "-c", _source_gate()],
        cwd=tmp_path,
        env={
            **os.environ,
            **values,
            "PATH": str(commands) + os.pathsep + os.environ["PATH"],
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is success, result.stdout + result.stderr
    if success:
        publishing = values["PUBLISH"] == "true" or ref.startswith("refs/tags/")
        assert output.read_text() == f"revision={revision}\npublish={str(publishing).lower()}\n"
    else:
        assert not output.exists()
