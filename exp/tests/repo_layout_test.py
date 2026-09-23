"""Permanent executable guards for repository-level ownership boundaries."""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MAX_HAND_AUTHORED_LINES = 999
HAND_AUTHORED_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".rs",
    ".rst",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
ALLOWED_TOP_DIRS = {".claude", ".github", "assets", "docs", "exp"}
ALLOWED_TOP_FILES = {
    ".env.example",
    ".gitignore",
    "AGENTS.md",
    "CLAUDE.md",
    "LICENSE",
    "README.md",
    "conftest.py",
    "justfile",
    "pyproject.toml",
    "SETUP.md",
    "uv.lock",
}


def _tracked_files() -> tuple[str, ...]:
    """Read the complete Git-tracked repository path set.

    Returns:
        Tracked paths relative to the repository root.

    Raises:
        pytest.skip.Exception: Git is unavailable or the checkout cannot enumerate tracked files.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git is required for repository-layout checks")
    if result.returncode != 0:
        pytest.skip("repository-layout checks require a Git checkout")
    return tuple(result.stdout.splitlines())


def _physical_lines(path: Path) -> int:
    """Return physical UTF-8 line count without inventing a final blank line."""
    text = path.read_text(encoding="utf-8")
    return text.count("\n") + int(bool(text) and not text.endswith("\n"))


def _is_line_limit_exempt(path: Path) -> bool:
    """Return whether a generated lock or Python test module is exempt."""
    return path.name == "package-lock.json" or path.name.endswith("_test.py")


def test_non_test_hand_authored_files_stay_below_one_thousand_lines() -> None:
    """Prove every covered non-test hand-authored file remains within the executable limit.

    Tracked generated lock files and cohesive Python test modules are the only narrow exemptions.
    """
    oversized: list[tuple[str, int]] = []
    for relative_path in _tracked_files():
        path = REPO_ROOT / relative_path
        if (
            not path.is_file()
            or path.suffix.lower() not in HAND_AUTHORED_SUFFIXES
            or _is_line_limit_exempt(path)
        ):
            continue
        lines = _physical_lines(path)
        if lines > MAX_HAND_AUTHORED_LINES:
            oversized.append((relative_path, lines))
    assert not oversized, f"hand-authored files exceed 999 physical lines: {oversized}"


def test_top_level_paths_are_allowlisted() -> None:
    """Prove tracked repository-root directories and files stay on the closed allowlist.

    New root surfaces require an explicit allowlist edit instead of landing silently.
    """
    tracked = _tracked_files()
    actual_dirs = {path.split("/", 1)[0] for path in tracked if "/" in path}
    actual_files = {path for path in tracked if "/" not in path}
    assert actual_dirs == ALLOWED_TOP_DIRS
    assert actual_files == ALLOWED_TOP_FILES


def test_package_wide_tests_live_in_tests_directories() -> None:
    """Keep tests without a same-stem production module in package test directories."""
    offenders: list[str] = []
    for relative_path in _tracked_files():
        relative = Path(relative_path)
        if not relative.name.endswith("_test.py"):
            continue
        module_name = relative.name.removesuffix("_test.py") + ".py"
        module_path = relative.with_name(module_name)
        if relative.parent.name == "tests" or (REPO_ROOT / module_path).is_file():
            continue
        offenders.append(relative_path)
    assert not offenders, (
        "package-wide tests without a same-stem module must live in a tests directory: "
        f"{sorted(offenders)}"
    )


def test_review_surfaces_are_python_only() -> None:
    """Prove review services remain Python-owned without a browser application or adapter."""
    assert not (REPO_ROOT / "web").exists()
    assert not (REPO_ROOT / "exp" / "cli" / "review_server.py").exists()
    assert not (REPO_ROOT / "exp" / "cli" / "review_server_test.py").exists()


def test_no_local_state_or_cache_is_tracked() -> None:
    """Prove generated settings, caches, bytecode, and local environment files stay untracked.

    The guard scans Git ownership rather than the developer's unrelated untracked local state.
    """
    offenders = [
        path
        for path in _tracked_files()
        if Path(path).name in {".env", "settings.toml"}
        or "__pycache__" in Path(path).parts
        or path.endswith(".pyc")
    ]
    assert not offenders, f"local state or cache files are tracked: {offenders}"


def test_native_crate_versions_stay_in_lockstep() -> None:
    """The extension's two manifests and the flagship exact dependency agree.

    The crate version feeds ``exp_gateway_native.__version__`` from Cargo and
    the published wheel from its pyproject; the flagship's exact pin selects
    only that reviewed companion. If any of the three drift, a release
    can pair new python code with a stale compiled engine (or fail to publish
    at all, since PyPI rejects re-uploads of an existing version's files).
    """
    crate_dir = REPO_ROOT / "exp" / "runtime" / "gateway" / "native"
    cargo_version = re.search(
        r'^version = "([^"]+)"',
        (crate_dir / "Cargo.toml").read_text(),
        re.MULTILINE,
    )
    assert cargo_version is not None
    with (crate_dir / "pyproject.toml").open("rb") as handle:
        wheel_version = tomllib.load(handle)["project"]["version"]
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        dependencies = tomllib.load(handle)["project"]["dependencies"]
    native_requirements = [
        requirement for requirement in dependencies if requirement.startswith("exp-gateway-native")
    ]
    assert len(native_requirements) == 1, "exactly one native dependency is required"
    _assert_native_version_contract(cargo_version.group(1), wheel_version, native_requirements[0])


def _assert_native_version_contract(
    cargo_version: str, wheel_version: str, requirement: str
) -> None:
    """Require equivalent package versions and the release's intended native pin.

    Args:
        cargo_version: SemVer version declared by the compiled crate.
        wheel_version: PEP 440 version declared by the Python distribution.
        requirement: Flagship dependency requirement for the native distribution.

    Raises:
        AssertionError: Manifest versions or the native dependency pin disagree.
    """
    version = Version(wheel_version)
    assert Version(cargo_version) == version, "native manifest versions differ"
    expected = f"exp-gateway-native=={version}"
    assert requirement == expected, "native dependency does not match the release version"


@pytest.mark.parametrize(
    ("cargo_version", "wheel_version", "requirement"),
    (
        ("0.3.82", "0.3.82", "exp-gateway-native==0.3.82"),
        ("0.3.82-rc.1", "0.3.82rc1", "exp-gateway-native==0.3.82rc1"),
    ),
)
def test_native_version_contract_accepts_exact_stable_and_prerelease(
    cargo_version: str, wheel_version: str, requirement: str
) -> None:
    """Equivalent stable and prerelease versions require exact companion pins.

    Args:
        cargo_version: Crate version under test.
        wheel_version: Distribution version under test.
        requirement: Native dependency requirement under test.
    """
    _assert_native_version_contract(cargo_version, wheel_version, requirement)


@pytest.mark.parametrize(
    ("cargo_version", "wheel_version", "requirement"),
    (
        ("0.3.81", "0.3.82", "exp-gateway-native==0.3.82"),
        ("0.3.82", "0.3.82", "exp-gateway-native>=0.3.82,<0.4"),
        ("0.3.82", "0.3.82", "exp-gateway-native==0.3.81"),
        ("0.3.82", "0.3.82", "exp-gateway-native==0.3.*"),
        ("0.3.82", "0.3.82", 'exp-gateway-native==0.3.82; sys_platform == "linux"'),
        ("0.3.82", "0.3.82", ""),
        ("0.3.82-rc.1", "0.3.82rc2", "exp-gateway-native==0.3.82rc2"),
        ("0.3.82-rc.1", "0.3.82rc1", "exp-gateway-native>=0.3.82rc1,<0.4"),
        ("0.3.82-rc.1", "0.3.82rc1", "exp-gateway-native==0.3.82"),
    ),
)
def test_native_version_contract_rejects_drift_and_unpinned_dependencies(
    cargo_version: str, wheel_version: str, requirement: str
) -> None:
    """Manifest drift, ranges, missing requirements and conditional pins fail closed.

    Args:
        cargo_version: Crate version under test.
        wheel_version: Distribution version under test.
        requirement: Native dependency requirement under test.
    """
    with pytest.raises(AssertionError):
        _assert_native_version_contract(cargo_version, wheel_version, requirement)
