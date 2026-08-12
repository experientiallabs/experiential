"""Report the frozen production LOC boundary and pull-request dependency changes."""

from __future__ import annotations

import argparse
import json
import subprocess
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".py", ".ts", ".sh", ".toml", ".yaml", ".yml", ".json"}
)
TEST_DIRECTORY_NAMES: Final[frozenset[str]] = frozenset({"test", "tests", "testdata", "fixtures"})
GENERATED_PRODUCTION_EXEMPTIONS: Final[frozenset[str]] = frozenset()


@dataclass(frozen=True)
class ProductionSnapshot:
    """The reproducible production file and physical LOC count at one Git revision."""

    revision: str
    file_count: int
    line_count: int
    paths: tuple[str, ...]


@dataclass(frozen=True)
class ProductionLocReport:
    """A per-PR production LOC and direct-dependency change report."""

    base: ProductionSnapshot
    head: ProductionSnapshot
    production_files_added: int
    production_files_removed: int
    production_files_net: int
    production_lines_added: int
    production_lines_removed: int
    production_lines_net: int
    direct_dependencies_added: tuple[str, ...]
    direct_dependencies_removed: tuple[str, ...]


def _git_output(arguments: Sequence[str]) -> str:
    """Run one local Git read command and return its standard output."""
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout


def _physical_line_count(text: str) -> int:
    """Count lines while treating a final newline as a terminator, not an extra blank line."""
    if not text:
        return 0
    return text.count("\n") + int(not text.endswith("\n"))


def is_production_path(relative_path: str) -> bool:
    """Return whether a tracked path belongs to the approved production LOC boundary.

    Args:
        relative_path: Repository-relative tracked path to classify.

    Returns:
        True when the path contributes to the production LOC report.
    """
    path = PurePosixPath(relative_path)
    if not path.parts or path.parts[0] != "wmo" or path.suffix not in PRODUCTION_SUFFIXES:
        return False
    if relative_path in GENERATED_PRODUCTION_EXEMPTIONS:
        return False
    if TEST_DIRECTORY_NAMES & set(path.parts):
        return False
    name = path.name
    return not (
        name.endswith("_test.py")
        or name == "conftest.py"
        or name.endswith(".test.ts")
        or name.startswith("vitest")
        and name.endswith(".config.ts")
    )


def _revision_paths(revision: str) -> tuple[str, ...]:
    """Return tracked paths at one revision."""
    return tuple(_git_output(["ls-tree", "-r", "--name-only", revision]).splitlines())


def _file_text(revision: str, relative_path: str) -> str:
    """Return one UTF-8 text object from a Git revision."""
    return _git_output(["show", f"{revision}:{relative_path}"])


def _resolved_revision(revision: str) -> str:
    """Resolve a user-supplied Git revision to its immutable object ID."""
    return _git_output(["rev-parse", revision]).strip()


def production_snapshot(revision: str) -> ProductionSnapshot:
    """Count approved production files and physical LOC at one Git revision.

    Args:
        revision: A Git revision that resolves in this checkout.

    Returns:
        The resolved revision plus its production file paths and physical LOC.

    Raises:
        RuntimeError: If the revision or one of its Git objects cannot be read.
    """
    resolved_revision = _resolved_revision(revision)
    paths = tuple(path for path in _revision_paths(resolved_revision) if is_production_path(path))
    line_count = sum(_physical_line_count(_file_text(resolved_revision, path)) for path in paths)
    return ProductionSnapshot(
        revision=resolved_revision,
        file_count=len(paths),
        line_count=line_count,
        paths=paths,
    )


def _changed_production_lines(base: str, head: str) -> tuple[int, int]:
    """Return production lines added and removed by the merge-base pull-request comparison."""
    output = _git_output(["diff", "--no-renames", "--numstat", f"{base}...{head}", "--", "wmo"])
    added = 0
    removed = 0
    for line in output.splitlines():
        fields = line.split("\t", maxsplit=2)
        if len(fields) != 3 or not is_production_path(fields[2]):
            continue
        if fields[0] == "-" or fields[1] == "-":
            continue
        added += int(fields[0])
        removed += int(fields[1])
    return added, removed


def _string_entries(value: object, label: str) -> tuple[str, ...]:
    """Validate one TOML string-array dependency declaration."""
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a list of dependency strings")
    entries: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise RuntimeError(f"{label} must be a list of dependency strings")
        entries.append(item)
    return tuple(entries)


def _direct_dependencies(revision: str) -> frozenset[str]:
    """Return project, optional-extra, and dependency-group declarations at one revision."""
    config = tomllib.loads(_file_text(revision, "pyproject.toml"))
    project = config.get("project")
    if not isinstance(project, dict):
        raise RuntimeError("pyproject.toml must contain a [project] table")
    dependencies = {
        f"project: {dependency}"
        for dependency in _string_entries(project["dependencies"], "dependencies")
    }
    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, dict):
        raise RuntimeError("project.optional-dependencies must be a table")
    for extra, entries in optional.items():
        if not isinstance(extra, str):
            raise RuntimeError("optional dependency names must be strings")
        dependencies.update(
            f"optional.{extra}: {dependency}"
            for dependency in _string_entries(entries, f"optional dependency {extra}")
        )
    groups = config.get("dependency-groups", {})
    if not isinstance(groups, dict):
        raise RuntimeError("dependency-groups must be a table")
    for group, entries in groups.items():
        if not isinstance(group, str):
            raise RuntimeError("dependency group names must be strings")
        dependencies.update(
            f"group.{group}: {dependency}"
            for dependency in _string_entries(entries, f"dependency group {group}")
        )
    return frozenset(dependencies)


def production_loc_report(base: str, head: str) -> ProductionLocReport:
    """Build the documented per-PR production LOC and dependency report.

    Args:
        base: The target branch or immutable base revision.
        head: The pull-request head revision.

    Returns:
        Production file and LOC deltas plus direct dependency additions and removals.
    """
    base_snapshot = production_snapshot(base)
    head_snapshot = production_snapshot(head)
    lines_added, lines_removed = _changed_production_lines(base, head)
    base_dependencies = _direct_dependencies(base_snapshot.revision)
    head_dependencies = _direct_dependencies(head_snapshot.revision)
    return ProductionLocReport(
        base=base_snapshot,
        head=head_snapshot,
        production_files_added=len(set(head_snapshot.paths) - set(base_snapshot.paths)),
        production_files_removed=len(set(base_snapshot.paths) - set(head_snapshot.paths)),
        production_files_net=head_snapshot.file_count - base_snapshot.file_count,
        production_lines_added=lines_added,
        production_lines_removed=lines_removed,
        production_lines_net=lines_added - lines_removed,
        direct_dependencies_added=tuple(sorted(head_dependencies - base_dependencies)),
        direct_dependencies_removed=tuple(sorted(base_dependencies - head_dependencies)),
    )


def _report_payload(report: ProductionLocReport) -> dict[str, object]:
    """Return the stable JSON object emitted by the command-line reporting surface."""
    payload = asdict(report)
    payload["base"].pop("paths")
    payload["head"].pop("paths")
    return payload


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    """Parse the private module command's explicit revision arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="target branch or immutable base revision")
    parser.add_argument("--head", default="HEAD", help="pull-request head revision")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Emit one reproducible JSON production LOC report.

    Args:
        arguments: Optional command arguments, primarily for direct tests.

    Returns:
        Zero after emitting the report, or argparse's process exit on invalid input.
    """
    parsed = _parse_arguments(arguments)
    report = production_loc_report(parsed.base, parsed.head)
    Console().print(json.dumps(_report_payload(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
