"""Execute CI release-deferral guards without GitHub or package publication."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MARKER = ".github/deferred-native-release.toml"
NATIVE = "exp/runtime/gateway/native"


def _version_gate() -> str:
    """Extract the exact executable version guard from its owning workflow."""
    workflow = (ROOT / ".github/workflows/gate.yml").read_text()
    step = workflow.split("      - name: Crate changes require a crate version bump\n", 1)[1]
    return textwrap.dedent(step.split("        run: |\n", 1)[1].split("      - name:", 1)[0])


@pytest.mark.parametrize(
    ("deferred", "project", "native", "base_project", "files", "success"),
    [
        (True, "0.7.106", "0.3.96", "0.7.106", f"{NATIVE}/src/lib.rs", True),
        (True, "0.7.107", "0.3.96", "0.7.106", f"{NATIVE}/src/lib.rs", False),
        (True, "0.7.106", "0.3.97", "0.7.106", f"{NATIVE}/src/lib.rs", False),
        (True, "0.7.106", "0.3.96", "0.7.105", f"{NATIVE}/src/lib.rs", False),
        (False, "0.7.106", "0.3.96", "0.7.106", f"{NATIVE}/src/lib.rs", False),
        (False, "0.7.107", "0.3.97", "0.7.106", MARKER, True),
        (False, "0.7.107", "0.3.96", "0.7.106", MARKER, False),
        (False, "0.7.106", "0.3.97", "0.7.106", MARKER, False),
        (False, "0.7.106", "0.3.96", "0.7.106", "docs/usage.md", True),
    ],
)
def test_release_deferral_requires_unchanged_versions_and_a_combined_bump(
    tmp_path: Path,
    deferred: bool,
    project: str,
    native: str,
    base_project: str,
    files: str,
    success: bool,
) -> None:
    """Exercise the actual shell gate against explicit local source and base versions."""
    (tmp_path / NATIVE).mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion = "{project}"\n')
    (tmp_path / NATIVE / "Cargo.toml").write_text(f'[package]\nversion = "{native}"\n')
    (tmp_path / NATIVE / "pyproject.toml").write_text(f'[project]\nversion = "{native}"\n')
    if deferred:
        (tmp_path / ".github").mkdir()
        (tmp_path / MARKER).write_text('experiential = "0.7.106"\nexp-gateway-native = "0.3.96"\n')
    stubs = r"""
    gh() {
      case "$*" in
        */files*) printf '%s\n' "$CHANGED_FILES";;
        *Cargo.toml*) printf 'version = "0.3.96"\n' | base64;;
        *pyproject.toml*) printf 'version = "%s"\n' "$BASE_PROJECT" | base64;;
        *) return 1;;
      esac
    }
    uv() { shift 2; "$TEST_PYTHON" "$@"; }
    """
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", textwrap.dedent(stubs) + _version_gate()],
        cwd=tmp_path,
        env={
            **os.environ,
            "TEST_PYTHON": sys.executable,
            "BASE_PROJECT": base_project,
            "CHANGED_FILES": files,
            "GITHUB_REPOSITORY": "example/repo",
            "GITHUB_BASE_REF": "main",
            "PR_NUMBER": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) == success, result.stdout + result.stderr


@pytest.mark.parametrize("deferred", [False, True])
def test_publishing_cannot_pass_with_a_deferred_native_release(
    tmp_path: Path, deferred: bool
) -> None:
    """Both release events and explicit replay publishing must pass the build guard."""
    workflow = (ROOT / ".github/workflows/python-package.yml").read_text()
    build, publish = workflow.split("  build:\n", 1)[1].split("  publish:\n", 1)
    step = build.split("      - name: Block publishing unreleased native changes\n", 1)[1]
    assert (
        "if: github.event_name == 'release' || inputs.publish == true"
        in step.split("      - name:", 1)[0]
    )
    assert "    needs: [build, capture-release-smoke]\n" in publish
    command = re.search(r"^        run: (.+)$", step, re.MULTILINE)
    assert command is not None
    if deferred:
        (tmp_path / ".github").mkdir()
        (tmp_path / MARKER).touch()
    result = subprocess.run(["bash", "-ec", command[1]], cwd=tmp_path, check=False)
    assert (result.returncode == 0) is not deferred
