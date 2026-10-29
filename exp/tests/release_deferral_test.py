"""Execute CI release-deferral guards without a CI runner or package publication."""

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


def _ci_job(area: str, name: str) -> str:
    """Return the raw YAML block of one top-level job in a GitLab CI area file.

    Args:
        area: Area file name under ``.gitlab/ci`` without the ``.yml`` suffix.
        name: Top-level job name.

    Returns:
        The job's indented block, without the job name line.
    """
    block = (ROOT / ".gitlab" / "ci" / f"{area}.yml").read_text().split(f"\n{name}:\n", 1)[1]
    return re.split(r"\n(?=\S)", block, maxsplit=1)[0]


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


@pytest.mark.parametrize(
    ("trigger", "deferred", "success"),
    [
        ({"CI_COMMIT_TAG": "v0.7.107"}, True, False),
        ({"PYTHON_PACKAGE_PUBLISH": "true"}, True, False),
        ({"CI_COMMIT_TAG": "v0.7.107"}, False, True),
        ({"PYTHON_PACKAGE_PUBLISH": "true"}, False, True),
        ({"PYTHON_PACKAGE_PUBLISH": "false"}, True, True),
    ],
)
def test_publishing_cannot_pass_with_a_deferred_native_release(
    tmp_path: Path, trigger: dict[str, str], deferred: bool, success: bool
) -> None:
    """Both release tags and explicit replay publishing must pass the build guard."""
    build = _ci_job("python-package", "python-package-build")
    publish = _ci_job("python-package", "python-package-publish")
    assert "    - job: python-package-build\n" in publish
    assert "    - job: python-package-capture-smoke\n" in publish
    guard = next(item for item in build.split("    - |\n")[1:] if MARKER in item)
    if deferred:
        (tmp_path / ".github").mkdir()
        (tmp_path / MARKER).touch()
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in {"CI_COMMIT_TAG", "PYTHON_PACKAGE_PUBLISH"}
    }
    result = subprocess.run(
        ["bash", "-ec", textwrap.dedent(guard)],
        cwd=tmp_path,
        env={**environment, **trigger},
        check=False,
    )
    assert (result.returncode == 0) is success
