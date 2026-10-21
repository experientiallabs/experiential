"""Binary checkpoint persistence retains exact bytes without treating base64 as recorded prose."""

import base64
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import SecretBoundaryError, assert_secret_free, sha256_bytes
from exp.common.project import ArtifactStore, ProjectPaths
from exp.common.project.errors import ArtifactStoreError
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.environments.world_artifacts import load_world_image, persist_world_image
from exp.runtime.environments.world_contracts import NativeImage


def test_binary_snapshot_does_not_scan_base64_as_text_but_preserves_actual_secret_checks(
    tmp_path: Path,
) -> None:
    """An encoding coincidence is not a credential; an actual textual credential still refuses."""
    store = ArtifactStore(ProjectPaths(tmp_path, "images"))
    case = workbook_case()
    with case.runtime.open(case.task) as session:
        checkpoint = session.checkpoint()
    encoded = "AKIA" + "A" * 16
    with pytest.raises(SecretBoundaryError):
        assert_secret_free({"encoded": encoded})
    payload = base64.b64decode(encoded)
    image = NativeImage(
        parts={"opaque.bin": payload}, domains={"opaque.bin": sha256_bytes(payload)}
    )
    checkpoint = checkpoint.model_copy(update={"image": image})
    ref = persist_world_image(store, checkpoint, created_at=datetime.now(UTC), code_revision="test")
    assert load_world_image(store, ref) == checkpoint
    assert any(row.path.endswith(".bin") for row in store.read(ref.artifact_id).manifest.files)
    actual_text = encoded.encode()
    bad = checkpoint.model_copy(
        update={
            "image": NativeImage(
                parts={"opaque.bin": actual_text}, domains={"opaque.bin": sha256_bytes(actual_text)}
            )
        }
    )
    with pytest.raises(ArtifactStoreError, match="secret"):
        persist_world_image(store, bad, created_at=datetime.now(UTC), code_revision="test")


def test_native_workbook_snapshot_restores_after_artifact_round_trip(tmp_path: Path) -> None:
    """Persisted bytes reconstruct a usable world, not merely an equal metadata object."""
    store = ArtifactStore(ProjectPaths(tmp_path, "images"))
    case = workbook_case()
    with case.runtime.open(case.task) as world:
        world.execute(case.first)
        snapshot = world.checkpoint()
    ref = persist_world_image(store, snapshot, created_at=datetime.now(UTC), code_revision="test")
    restored = load_world_image(store, ref)
    assert type(restored) is type(snapshot)
    with case.runtime.restore(case.task, restored) as session:
        assert session.checkpoint() == snapshot
        assert not session.execute(case.second).is_error
