"""Execute the CI scope selectors that decide which expensive runners a change needs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def _step_script(workflow: str, step: str) -> str:
    """Extract the exact run script of one named workflow step."""
    text = (WORKFLOWS / workflow).read_text()
    body = text.split(f"      - name: {step}\n", 1)[1]
    return textwrap.dedent(body.split("        run: |\n", 1)[1].split("\n\n", 1)[0])


def _darwin_scope_program() -> str:
    """The Python selector embedded in the darwin-scope job."""
    text = (WORKFLOWS / "gate.yml").read_text()
    marker = "      - name: Does this pull request touch the Darwin evidence scope\n"
    step = text.split(marker, 1)[1]
    return textwrap.dedent(step.split("<<'PY'\n", 1)[1].split("\n          PY\n", 1)[0])


VERSION_BUMP_LOCK = '@@ -1 +1 @@\n-version = "0.7.138"\n+version = "0.7.139"'
NATIVE_PIN_BUMP = (
    "@@ -11 +11 @@\n"
    '-    "exp-gateway-native>=0.3.119,<0.4",  # compiled gateway data plane\n'
    '+    "exp-gateway-native>=0.3.120,<0.4",  # compiled gateway data plane'
)
DEPENDENCY_CHANGE = (
    '@@ -5 +5 @@\n-    { name = "certifi" },\n+    { name = "certifi", marker = "x" },'
)


@pytest.mark.parametrize(
    ("files", "required"),
    [
        ([{"filename": "exp/runtime/gateway/server.py", "patch": "+x"}], False),
        ([{"filename": "uv.lock", "patch": VERSION_BUMP_LOCK}], False),
        ([{"filename": "pyproject.toml", "patch": NATIVE_PIN_BUMP}], False),
        ([{"filename": "uv.lock", "patch": DEPENDENCY_CHANGE}], True),
        ([{"filename": "uv.lock"}], True),
        ([{"filename": "exp/runtime/capture/proxy.py", "patch": "+x"}], True),
        ([{"filename": "exp/simulation/engines/sandbox.py", "patch": "+x"}], True),
        ([{"filename": ".github/workflows/gate.yml", "patch": "+x"}], True),
        ([{"filename": "docs/usage.md", "patch": "+x"}], False),
    ],
)
def test_darwin_scope_skips_version_bumps_but_not_dependency_changes(
    tmp_path: Path, files: list[dict[str, str]], required: bool
) -> None:
    """Every release bumps the manifests, so a version-only bump must not select macOS."""
    (tmp_path / "files.jsonl").write_text("".join(json.dumps(f) + "\n" for f in files))
    output = tmp_path / "output"
    subprocess.run(
        [sys.executable, "-c", _darwin_scope_program()],
        cwd=tmp_path,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        check=True,
    )
    assert output.read_text() == f"required={str(required).lower()}\n"


@pytest.mark.skipif(shutil.which("jq") is None, reason="the runner image provides jq")
@pytest.mark.parametrize(("changed", "count"), [("true", 5), ("false", 1)])
def test_unchanged_crate_builds_only_the_wheel_that_build_smokes(
    tmp_path: Path, changed: str, count: int
) -> None:
    """Skipping cross-platform wheels must still leave the linux-x86_64 wheel `build` needs."""
    output = tmp_path / "output"
    script = _step_script("python-package.yml", "Select wheel targets")
    subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        env={**os.environ, "NATIVE_CHANGED": changed, "GITHUB_OUTPUT": str(output)},
        check=True,
    )
    matrix = json.loads(output.read_text().removeprefix("matrix="))
    targets = [entry["target"] for entry in matrix["include"]]
    assert len(targets) == count
    assert "linux-x86_64" in targets
