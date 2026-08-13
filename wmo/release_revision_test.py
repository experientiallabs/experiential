"""Exact-checkout provenance checks shared by the executable W16 release evidence."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import JsonValue

from wmo.common.core.artifacts import sha256_json
from wmo.common.project import ArtifactStore

_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def exact_checkout_revision() -> str:
    """Return one verified 40-hex checkout revision or fail closed."""
    revision = _git("rev-parse", "HEAD")
    _require_revision(revision, label="checked-out revision")
    for variable in ("WMO_RELEASE_REVISION", "GITHUB_SHA"):
        configured = os.environ.get(variable)
        if configured is None:
            continue
        _require_revision(configured, label=variable)
        if configured != revision:
            raise RuntimeError(f"{variable} does not match the checked-out revision")
    if _git_status("diff", "--quiet", "--", ".") != 0:
        raise RuntimeError("release evidence requires a checkout with no tracked source changes")
    if _git_status("diff", "--cached", "--quiet", "--", ".") != 0:
        raise RuntimeError("release evidence requires a checkout with no staged source changes")
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", ".")
    if untracked:
        raise RuntimeError("release evidence source paths contain untracked files")
    return revision


def verify_release_evidence(
    store: ArtifactStore,
    *,
    expected_revision: str,
    report_name: str,
    claims: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Recursively verify exact artifact inputs and emit one optional machine-readable report."""
    _require_revision(expected_revision, label="expected release revision")
    artifact_ids = store.list_ids()
    verified_inputs = 0
    artifact_types: dict[str, int] = {}
    for artifact_id in artifact_ids:
        stored = store.read(artifact_id)
        manifest = stored.manifest
        if manifest.code_revision != expected_revision:
            raise AssertionError(
                f"artifact {artifact_id} was written by {manifest.code_revision}, "
                f"not {expected_revision}"
            )
        artifact_types[manifest.artifact_type] = artifact_types.get(manifest.artifact_type, 0) + 1
        for input_record in manifest.inputs:
            input_manifest = store.read(input_record.artifact_id).manifest
            if sha256_json(input_manifest) != input_record.sha256:
                raise AssertionError(
                    f"artifact {artifact_id} input {input_record.artifact_id} has a stale digest"
                )
            verified_inputs += 1
        for file_record in manifest.files:
            if file_record.path.endswith(".json"):
                value = json.loads(store.read_bytes(artifact_id, file_record.path))
                _require_nested_revisions(value, expected_revision, artifact_id)
            elif file_record.path.endswith(".jsonl"):
                for line in store.read_bytes(artifact_id, file_record.path).splitlines():
                    if line:
                        _require_nested_revisions(json.loads(line), expected_revision, artifact_id)
    report: dict[str, JsonValue] = {
        "schema_version": 1,
        "code_revision": expected_revision,
        "artifact_count": len(artifact_ids),
        "verified_input_count": verified_inputs,
        "artifact_types": dict(sorted(artifact_types.items())),
        "claims": dict(claims),
    }
    destination = os.environ.get("WMO_RELEASE_EVIDENCE_DIR")
    if destination is not None:
        output_directory = Path(destination)
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / f"{report_name}.json").write_text(
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    return report


def _require_nested_revisions(
    value: JsonValue,
    expected_revision: str,
    artifact_id: str,
) -> None:
    """Reject any nested persisted code revision that differs from the checkout."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "code_revision" and item != expected_revision:
                raise AssertionError(
                    f"artifact {artifact_id} contains nested code revision {item!r}"
                )
            _require_nested_revisions(item, expected_revision, artifact_id)
    elif isinstance(value, list):
        for item in value:
            _require_nested_revisions(item, expected_revision, artifact_id)


def _require_revision(value: str, *, label: str) -> None:
    """Require one full lowercase Git commit identity."""
    if _REVISION_PATTERN.fullmatch(value) is None:
        raise RuntimeError(f"{label} must be a full lowercase 40-hex Git revision")


def _git(*arguments: str) -> str:
    """Run one bounded read-only Git query against the release checkout."""
    result = subprocess.run(
        ["git", *arguments],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return result.stdout.strip()


def _git_status(*arguments: str) -> int:
    """Return one bounded read-only Git status code."""
    return subprocess.run(
        ["git", *arguments],
        cwd=_REPOSITORY_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    ).returncode


def test_exact_checkout_revision_rejects_symbolic_or_mismatched_ci_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release provenance cannot be replaced by a label or a different commit."""
    monkeypatch.setenv("WMO_RELEASE_REVISION", "w16-release")
    with pytest.raises(RuntimeError, match="full lowercase 40-hex"):
        exact_checkout_revision()
    monkeypatch.setenv("WMO_RELEASE_REVISION", "0" * 40)
    with pytest.raises(RuntimeError, match="does not match"):
        exact_checkout_revision()


@pytest.mark.parametrize("relative_path", ["release-shadow.py", "wmo/release-shadow.py"])
def test_exact_checkout_revision_rejects_untracked_checkout_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    """Root and package untracked inputs cannot hide outside the exact-checkout proof."""
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.email", "release@example.invalid"], cwd=repository)
    subprocess.run(["git", "config", "user.name", "Release Test"], cwd=repository)
    tracked = repository / "tracked.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repository, check=True)
    untracked = repository / relative_path
    untracked.parent.mkdir(parents=True, exist_ok=True)
    untracked.write_text("untracked\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "_REPOSITORY_ROOT", repository)
    monkeypatch.delenv("WMO_RELEASE_REVISION", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)

    with pytest.raises(RuntimeError, match="untracked files"):
        exact_checkout_revision()
