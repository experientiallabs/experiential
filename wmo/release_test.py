"""Release archive and metadata checks for the current single-package distribution."""

from __future__ import annotations

import os
import re
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
        assert "wmo/common/models/model.py" in names
        assert "wmo/runtime/models/registry.py" in names
        assert "wmo/workflow/router.py" in names
        assert not any(name.endswith("_test.py") or name == "wmo/conftest.py" for name in names)
        assert RETIRED_REQUIREMENT.search(metadata) is None
        assert _core_requirement_names(metadata) == REQUIRED_CORE_REQUIREMENTS

    with tarfile.open(sdists[0], mode="r:gz") as sdist:
        names = tuple(_normalized_path(member.name) for member in sdist.getmembers())
        metadata = _sdist_metadata(sdist)
        assert "README.md" in names
        assert "pyproject.toml" in names
        assert "wmo/workflow/router.py" in names
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
