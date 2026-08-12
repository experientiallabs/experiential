"""Release regressions for retired W8.6 documentation, modules, and dependencies."""

from __future__ import annotations

import os
import re
import subprocess
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Final

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILT_DIST_ENV: Final[str] = "WMO_BUILT_DIST_DIR"

FORBIDDEN_DOC_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "deleted optimizer module": re.compile(r"wmo[/\.]optimize[/\.](?:base|gepa)(?:\.py)?"),
    "deleted simulation owner": re.compile(
        r"wmo[/\.]simulation[/\.](?:environment(?:\.py)?|model|evaluation|retrieval)"
    ),
    "deleted scenarios command": re.compile(r"\bwmo scenarios build\b"),
    "deleted serve or eval command": re.compile(r"\bwmo (?:serve|eval)\b"),
    "deleted WorldModel API": re.compile(r"\bWorldModel\b"),
}

FORBIDDEN_ARCHIVE_PATHS: Final[frozenset[str]] = frozenset(
    {
        "wmo/optimize/base.py",
        "wmo/optimize/base_test.py",
        "wmo/optimize/gepa.py",
        "wmo/optimize/gepa_test.py",
        "wmo/simulation/environment.py",
        "wmo/simulation/environment_test.py",
    }
)
FORBIDDEN_ARCHIVE_PREFIXES: Final[tuple[str, ...]] = (
    "wmo/simulation/model/",
    "wmo/simulation/evaluation/",
    "wmo/simulation/retrieval/",
)
GEPA_REQUIREMENT = re.compile(r"(?mi)^Requires-Dist:\s*gepa(?:\s|[<>=;~!])")


def _tracked_document_paths() -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "*.md", "*.rst", "*.txt"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = (REPO_ROOT / relative_path for relative_path in result.stdout.splitlines())
    return tuple(path for path in paths if path.is_file())


def _normalized_archive_path(member_name: str) -> str:
    path = PurePosixPath(member_name)
    parts = path.parts
    if parts and parts[0].startswith("world_model_optimizer-"):
        parts = parts[1:]
    return PurePosixPath(*parts).as_posix()


def _retired_archive_members(member_names: Iterable[str]) -> tuple[str, ...]:
    retired: list[str] = []
    for member_name in member_names:
        path = _normalized_archive_path(member_name)
        if path in FORBIDDEN_ARCHIVE_PATHS or path.startswith(FORBIDDEN_ARCHIVE_PREFIXES):
            retired.append(member_name)
    return tuple(sorted(retired))


def _wheel_metadata(archive: zipfile.ZipFile) -> str:
    metadata_paths = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
    assert len(metadata_paths) == 1, f"wheel has unexpected METADATA paths: {metadata_paths}"
    return archive.read(metadata_paths[0]).decode("utf-8")


def _sdist_metadata(archive: tarfile.TarFile) -> str:
    metadata_members = [
        member
        for member in archive.getmembers()
        if member.isfile() and member.name.endswith("/PKG-INFO")
    ]
    assert len(metadata_members) == 1, f"sdist has unexpected PKG-INFO paths: {metadata_members}"
    extracted = archive.extractfile(metadata_members[0])
    assert extracted is not None
    return extracted.read().decode("utf-8")


def test_tracked_docs_do_not_publish_retired_w8_workflows() -> None:
    findings: list[str] = []
    for path in _tracked_document_paths():
        text = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        for label, pattern in FORBIDDEN_DOC_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{relative_path}:{line}: {label}: {match.group(0)}")
    assert not findings, "retired W8 documentation returned:\n" + "\n".join(findings)


def test_built_archives_exclude_retired_w8_content() -> None:
    configured_dir = os.environ.get(BUILT_DIST_ENV)
    if configured_dir is None:
        pytest.skip(f"set {BUILT_DIST_ENV} to scan freshly built release archives")
    dist_dir = Path(configured_dir)
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    assert len(wheels) == 1, f"expected one wheel in {dist_dir}, found {wheels}"
    assert len(sdists) == 1, f"expected one sdist in {dist_dir}, found {sdists}"

    with zipfile.ZipFile(wheels[0]) as wheel:
        assert not _retired_archive_members(wheel.namelist())
        assert GEPA_REQUIREMENT.search(_wheel_metadata(wheel)) is None
    with tarfile.open(sdists[0], mode="r:gz") as sdist:
        assert not _retired_archive_members(member.name for member in sdist.getmembers())
        assert GEPA_REQUIREMENT.search(_sdist_metadata(sdist)) is None


def test_archive_scanner_rejects_retired_module_descendants() -> None:
    members = (
        "world_model_optimizer-0.3.0/wmo/simulation/model/compat.py",
        "wmo/optimize/gepa.py",
        "wmo/runtime/router/runtime.py",
    )
    assert frozenset(_retired_archive_members(members)) == frozenset(members[:2])
