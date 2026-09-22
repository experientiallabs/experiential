"""Process-death recovery proofs for bundle publication across filesystem and SQLite."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from exp.common.project import (
    ProjectConfig,
    ProjectStore,
    export_project_bundle,
    restore_project_bundle,
)
from exp.common.project.bundle import ProjectBundleError
from exp.common.project.bundle_test import _REVISION, _project_store
from exp.common.project.records import ProjectRecords


@pytest.mark.parametrize("boundary", ("before_rename", "after_rename", "before_commit"))
def test_killed_restore_retries_exact_bundle_without_partial_selection(
    tmp_path: Path, boundary: str
) -> None:
    """A process killed outside Python cleanup leaves a recoverable exact-bundle intent."""
    source = _project_store(tmp_path, with_blob=True)
    exported = export_project_bundle(
        source, tmp_path / "source.bundle", producer_revision=_REVISION
    )
    root = tmp_path / "restored"
    sibling = ProjectStore(root, "sibling")
    sibling.initialize(ProjectConfig(project_id="sibling"))
    sibling.write_review({"retained": True})
    script = """
import os
import sys
from contextlib import contextmanager
from pathlib import Path
import exp.common.project.restore as restore
from exp.common.project import restore_project_bundle
boundary = sys.argv[4]
rename = restore.os.rename
transaction = restore.project_connection

def crash_rename(source, destination):
    if boundary == "before_rename":
        os._exit(73)
    rename(source, destination)
    if boundary == "after_rename":
        os._exit(73)

@contextmanager
def crash_commit(root, **kwargs):
    with transaction(root, **kwargs) as connection:
        yield connection
        if boundary == "before_commit" and connection.execute(
            "SELECT 1 FROM project_config_heads WHERE project_id='portable-project'"
        ).fetchone():
            os._exit(73)

restore.os.rename = crash_rename
restore.project_connection = crash_commit
restore_project_bundle(Path(sys.argv[1]), root=Path(sys.argv[2]), expected_sha256=sys.argv[3])
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(exported.path), str(root), exported.sha256, boundary],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 73, completed.stderr.decode()
    pending = ProjectRecords(root, "portable-project", "bundle-restore")
    assert pending.read("pending-publication") == exported.sha256.encode()
    unpublished = ProjectStore(root, "portable-project")
    assert not unpublished.exists()
    assert unpublished.artifacts.list_ids() == ()
    assert sibling.read_review() == {"retained": True}
    restored = restore_project_bundle(exported.path, root=root, expected_sha256=exported.sha256)
    assert restored.load_project() == source.load_project()
    for artifact_id in restored.artifacts.list_ids():
        assert (
            restored.artifacts.read(artifact_id).payloads
            == source.artifacts.read(artifact_id).payloads
        )
    assert pending.read("pending-publication") is None
    assert sibling.read_review() == {"retained": True}
    with pytest.raises(ProjectBundleError, match="destination must be absent"):
        restore_project_bundle(exported.path, root=root, expected_sha256=exported.sha256)


def test_interrupted_restore_rejects_changed_published_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An intent cannot authorize tampered bytes or permit a different bundle to adopt them."""
    source = _project_store(tmp_path, with_blob=True)
    exported = export_project_bundle(
        source, tmp_path / "source.bundle", producer_revision=_REVISION
    )
    root = tmp_path / "restored"
    original_rename = os.rename

    def interrupt_after_rename(source_path: Path, destination_path: Path) -> None:
        """Leave the filesystem published but abort the uncommitted database mutation."""
        original_rename(source_path, destination_path)
        raise OSError("injected power-loss boundary")

    with monkeypatch.context() as patch:
        patch.setattr("exp.common.project.restore.os.rename", interrupt_after_rename)
        with pytest.raises(ProjectBundleError, match="power-loss boundary"):
            restore_project_bundle(exported.path, root=root, expected_sha256=exported.sha256)
    destination = ProjectStore(root, "portable-project")
    blob = next((destination.paths.project_directory / "blobs").iterdir())
    payload = blob.read_bytes()
    blob.write_bytes(b"corrupt")
    with pytest.raises(ProjectBundleError, match="differs from verified bundle"):
        restore_project_bundle(exported.path, root=root, expected_sha256=exported.sha256)
    assert not destination.exists()
    blob.write_bytes(payload)
    alternative = export_project_bundle(
        source, tmp_path / "different.bundle", producer_revision="different"
    )
    assert alternative.sha256 != exported.sha256
    with pytest.raises(ProjectBundleError, match="interrupted bundle"):
        restore_project_bundle(alternative.path, root=root, expected_sha256=alternative.sha256)
    restored = restore_project_bundle(exported.path, root=root, expected_sha256=exported.sha256)
    assert restored.load_project() == source.load_project()
