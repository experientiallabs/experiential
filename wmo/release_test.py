"""Release archive and metadata checks for the current single-package distribution."""

from __future__ import annotations

import os
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import pytest

BUILT_DIST_ENV = "WMO_BUILT_DIST_DIR"
RETIRED_REQUIREMENT = re.compile(
    r"(?mi)^Requires-Dist:\s*(?:anthropic|boto3|environment-capture|gepa|mlx-lm|openai|"
    r"opentelemetry-proto|scikit-learn|transformers)(?:\s|[<>=;~!])"
)
REQUIRED_CORE_REQUIREMENTS = frozenset(
    {
        "click",
        "fastapi",
        "filelock",
        "httpx",
        "numpy",
        "posthog",
        "pydantic",
        "rich",
        "tomli-w",
        "typer",
        "uvicorn",
    }
)
REQUIRED_WHEEL_MODULES = frozenset(
    {
        "wmo/common/models/model.py",
        "wmo/runtime/environments/local.py",
        "wmo/runtime/models/registry.py",
        "wmo/runtime/router/application.py",
        "wmo/simulation/comparison.py",
        "wmo/simulation/engines/sandbox.py",
        "wmo/workflow/router.py",
    }
)
REQUIRED_SDIST_MEMBERS = frozenset({"README.md", "pyproject.toml", "wmo/workflow/router.py"})
FORBIDDEN_ARCHIVE_PREFIXES = (
    "assets/",
    "wmo/common/providers/",
    "wmo/common/vendor/",
    "wmo/optimize/research/",
)
FORBIDDEN_ARCHIVE_MEMBERS = frozenset(
    {
        "docs/reference/repository_guardrails.md",
        "wmo/cli/repo_metrics.py",
        "wmo/common/core/parsing.py",
        "wmo/common/core/render.py",
        "wmo/common/core/types.py",
        "wmo/repo_docstrings_test.py",
        "wmo/repo_structure_test.py",
        "wmo/repository_guardrails.toml",
        "wmo/simulation/hub.py",
    }
)


def _normalized_path(member_name: str) -> str:
    """Remove the versioned sdist root from one archive path."""
    parts = PurePosixPath(member_name).parts
    if parts and parts[0].startswith("world_model_optimizer-"):
        parts = parts[1:]
    return PurePosixPath(*parts).as_posix()


def _wheel_metadata(archive: zipfile.ZipFile) -> str:
    """Return the wheel's unique core metadata document."""
    paths = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
    assert len(paths) == 1, f"wheel has unexpected METADATA paths: {paths}"
    return archive.read(paths[0]).decode("utf-8")


def _sdist_metadata(archive: tarfile.TarFile) -> str:
    """Return the sdist's unique core metadata document."""
    members = [
        member
        for member in archive.getmembers()
        if member.isfile() and member.name.endswith("/PKG-INFO")
    ]
    assert len(members) == 1, f"sdist has unexpected PKG-INFO paths: {members}"
    extracted = archive.extractfile(members[0])
    assert extracted is not None
    return extracted.read().decode("utf-8")


def _core_requirement_names(metadata: str) -> frozenset[str]:
    """Return normalized non-extra dependency names from package metadata."""
    names: set[str] = set()
    for line in metadata.splitlines():
        if not line.startswith("Requires-Dist:") or "; extra ==" in line:
            continue
        requirement = line.removeprefix("Requires-Dist:").strip()
        name = re.split(r"[<>=;~!\s]", requirement, maxsplit=1)[0].casefold()
        names.add(name)
    return frozenset(names)


def _assert_current_archive_members(
    names: tuple[str, ...],
    *,
    required: frozenset[str],
    allow_tests: bool,
) -> None:
    """Reject missing current members or any retired package, module, test, or asset member."""
    file_names = frozenset(name for name in names if name and not name.endswith("/"))
    assert required.issubset(file_names), (
        f"archive is missing current members: {required - file_names}"
    )
    forbidden = sorted(
        name
        for name in file_names
        if name in FORBIDDEN_ARCHIVE_MEMBERS
        or any(name.startswith(prefix) for prefix in FORBIDDEN_ARCHIVE_PREFIXES)
        or not allow_tests
        and (name.endswith("_test.py") or name == "wmo/conftest.py")
    )
    assert not forbidden, f"archive contains forbidden stale members: {forbidden}"


def _tracked_sdist_members() -> frozenset[str]:
    """Return current tracked members admitted by the explicit sdist include contract."""
    result = subprocess.run(
        ["git", "ls-files", ".gitignore", "README.md", "pyproject.toml", "conftest.py", "wmo"],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return frozenset(result.stdout.splitlines())


def _tracked_wheel_members() -> frozenset[str]:
    """Return the exact current source members admitted by the wheel contract."""
    return frozenset(
        name
        for name in _tracked_sdist_members()
        if name.startswith("wmo/") and not name.endswith("_test.py") and name != "wmo/conftest.py"
    )


def test_built_archives_match_current_package_contract() -> None:
    """Fresh wheel and sdist contain current code and minimal dependency metadata."""
    configured_dir = os.environ.get(BUILT_DIST_ENV)
    if configured_dir is None:
        pytest.skip(f"set {BUILT_DIST_ENV} to scan freshly built release archives")
    dist_dir = Path(configured_dir)
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    assert len(wheels) == 1, f"expected one wheel in {dist_dir}, found {wheels}"
    assert len(sdists) == 1, f"expected one sdist in {dist_dir}, found {sdists}"

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = tuple(_normalized_path(name) for name in wheel.namelist())
        metadata = _wheel_metadata(wheel)
        _assert_current_archive_members(
            names,
            required=REQUIRED_WHEEL_MODULES,
            allow_tests=False,
        )
        assert frozenset(name for name in names if name.startswith("wmo/")) == (
            _tracked_wheel_members()
        )
        assert RETIRED_REQUIREMENT.search(metadata) is None
        assert _core_requirement_names(metadata) == REQUIRED_CORE_REQUIREMENTS

    with tarfile.open(sdists[0], mode="r:gz") as sdist:
        names = tuple(
            _normalized_path(member.name) for member in sdist.getmembers() if member.isfile()
        )
        metadata = _sdist_metadata(sdist)
        _assert_current_archive_members(names, required=REQUIRED_SDIST_MEMBERS, allow_tests=True)
        assert frozenset(name for name in names if name and not name.endswith("/")) == (
            _tracked_sdist_members() | {"PKG-INFO"}
        )
        assert RETIRED_REQUIREMENT.search(metadata) is None
        assert _core_requirement_names(metadata) == REQUIRED_CORE_REQUIREMENTS


def test_requirement_scanner_rejects_removed_dependencies() -> None:
    """The release check detects every removed dependency family directly."""
    for dependency in (
        "anthropic",
        "boto3",
        "environment-capture",
        "gepa",
        "mlx-lm",
        "openai",
        "opentelemetry-proto",
        "scikit-learn",
        "transformers",
    ):
        assert RETIRED_REQUIREMENT.search(f"Requires-Dist: {dependency}>=1\n") is not None


def test_w16_public_evidence_apis_resolve_from_release_owners() -> None:
    """W16 customer and comparison workflows resolve without test-only API owners."""
    import wmo
    from wmo.runtime.environments import LocalProcessEnvironmentRuntime
    from wmo.simulation import compare_text_and_sandbox
    from wmo.simulation.engines import SandboxSimulator

    assert callable(wmo.compose_router)
    assert callable(wmo.load_project_router)
    assert callable(wmo.create_project_router_app)
    assert callable(compare_text_and_sandbox)
    assert SandboxSimulator.__module__ == "wmo.simulation.engines.sandbox"
    assert LocalProcessEnvironmentRuntime.__module__ == "wmo.runtime.environments.local"


def test_documentation_index_commands_and_release_scope_are_current() -> None:
    """Every indexed doc exists and release docs name current commands and explicit exclusions."""
    repository = Path(__file__).resolve().parent.parent
    docs = repository / "docs"
    index = (docs / "README.md").read_text(encoding="utf-8")
    indexed_paths = re.findall(r"\| `([^`]+\.md)` \|", index)
    assert indexed_paths
    assert not [path for path in indexed_paths if not (docs / path).is_file()]

    usage = (docs / "usage.md").read_text(encoding="utf-8")
    assert "wmo optimize router" in usage
    assert "wmo optimize model" in usage
    assert "wmo optimize route" not in usage.replace("wmo optimize router", "")

    scope = (docs / "release-scope.md").read_text(encoding="utf-8")
    for exclusion in (
        "No paid E2B or Harbor cloud smoke ran",
        "No real Tinker training ran",
        "No trained-versus-base behavioral comparison ran",
        "exactly $0.00 observed service spend",
    ):
        assert exclusion in scope

    ingest = (docs / "reference" / "ingest.md").read_text(encoding="utf-8")
    assert "PostHogPullRequest" in ingest
    assert "pull_posthog_traces" in ingest


@pytest.mark.parametrize(
    "stale_member",
    [
        "assets/world-model-agent-loop.svg",
        "wmo/common/providers/openai.py",
        "wmo/common/vendor/sdk.py",
        "wmo/optimize/research/runner.py",
        "wmo/common/core/types.py",
        "wmo/simulation/hub.py",
        "wmo/repository_guardrails.toml",
    ],
)
def test_archive_member_scanner_rejects_synthetic_stale_members(stale_member: str) -> None:
    """Every removed asset, owner, helper, and guard family fails a direct synthetic scan."""
    with pytest.raises(AssertionError, match="forbidden stale members"):
        _assert_current_archive_members(
            tuple((*REQUIRED_WHEEL_MODULES, stale_member)),
            required=REQUIRED_WHEEL_MODULES,
            allow_tests=False,
        )
