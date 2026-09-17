"""Checkpoint digest and scope rejection tests without executing model payloads."""

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import sha256_json
from exp.optimize.claas.backends.checkpoints import (
    CheckpointManifest,
    checkpoint_snapshot,
    hash_file,
    recover_training_result,
    verify_checkpoint,
    verify_training_result,
)
from exp.optimize.claas.training_contracts import (
    TrainingCheckpoint,
    TrainingResult,
    next_policy_revision,
)
from exp.optimize.claas.training_contracts_test import job, spec


def checkpoint(root: Path) -> TrainingCheckpoint:
    """Write inert files with a complete manifest for verification-only tests."""
    paths = (
        "student/adapter_config.json",
        "student/adapter_model.safetensors",
        "teacher/adapter_config.json",
        "teacher/adapter_model.safetensors",
        "verl/actor/model_world_size_1_rank_0.pt",
        "verl/actor/optim_world_size_1_rank_0.pt",
        "verl/actor/extra_state_world_size_1_rank_0.pt",
        "verl/actor/fsdp_config.json",
        "verl/teacher/model_world_size_1_rank_0.pt",
        "verl/teacher/fsdp_config.json",
    )
    for name in paths:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"inert-test-payload")
    manifest = CheckpointManifest(
        schema_version=2,
        training_backend="verl-fsdp-0.9.0",
        spec=spec(),
        policy_revision="policy-1",
        parent_policy_revision="policy-0",
        policy_history=("policy-1", "policy-0"),
        step=1,
        batch_id="batch-1",
        batch_sha256=sha256_json(job(root).batch),
        metrics={},
        consumed_experience_ids=("experience-1",),
        files={name: hash_file(root / name) for name in paths},
    )
    (root / "manifest.json").write_text(manifest.model_dump_json())
    return TrainingCheckpoint(
        scope=spec().scope,
        adapter_id=spec().adapter_id,
        policy_revision="policy-1",
        policy_history=manifest.policy_history,
        step=1,
        path=str(root),
        manifest_sha256=sha256_json(manifest),
    )


def test_verifies_before_loading_and_rejects_changed_payload(tmp_path: Path) -> None:
    """A valid manifest does not authorize a subsequently changed optimizer pickle."""
    receipt = checkpoint(tmp_path)
    assert verify_checkpoint(receipt, spec()).step == 1
    (tmp_path / "verl/actor/optim_world_size_1_rank_0.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        verify_checkpoint(receipt, spec())


def test_checkpoint_requires_selected_serving_adapter_payloads(tmp_path: Path) -> None:
    """A valid student checkpoint cannot claim a separate serving export that is absent."""
    receipt = checkpoint(tmp_path)
    manifest = verify_checkpoint(receipt, spec())
    assert manifest.serving_adapter_directory == "student"
    missing_serving = manifest.model_copy(update={"serving_adapter_directory": "serving"})
    (tmp_path / "manifest.json").write_text(missing_serving.model_dump_json())
    changed = receipt.model_copy(update={"manifest_sha256": sha256_json(missing_serving)})
    with pytest.raises(ValueError, match="lacks.*serving"):
        verify_checkpoint(changed, spec())


def test_rejects_unbound_ancestry_and_extra_payload(tmp_path: Path) -> None:
    """A claimed replay ancestor must be contained in the immutable manifest."""
    receipt = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        verify_checkpoint(
            receipt.model_copy(update={"policy_history": ("policy-1", "foreign")}), spec()
        )
    (tmp_path / "extra.pt").write_bytes(b"untracked")
    with pytest.raises(ValueError, match="outside"):
        verify_checkpoint(receipt, spec())


def test_private_snapshot_keeps_verified_bytes_after_source_replacement(tmp_path: Path) -> None:
    """Concurrent source writes after staging cannot change the paths used by loaders."""

    receipt = checkpoint(tmp_path)
    with checkpoint_snapshot(receipt, spec()) as staged:
        assert staged is not None
        staged_root = Path(staged.path)
        (tmp_path / "verl/actor/optim_world_size_1_rank_0.pt").write_bytes(b"replacement")
        assert (
            staged_root / "verl/actor/optim_world_size_1_rank_0.pt"
        ).read_bytes() == b"inert-test-payload"
        verify_checkpoint(staged, spec())
    assert not staged_root.exists()


def test_snapshot_rejects_content_changed_during_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing source bytes between verification and copy cannot become resumable state."""

    receipt = checkpoint(tmp_path)
    original = shutil.copyfile

    def changed_copy(source: Path, target: Path) -> Path:
        """Replace a payload immediately before its staging read."""
        source.write_bytes(b"changed-during-copy")
        return original(source, target)

    monkeypatch.setattr(shutil, "copyfile", changed_copy)
    with (
        pytest.raises(ValueError, match="changed while staging"),
        checkpoint_snapshot(receipt, spec()),
    ):
        pytest.fail("unverified state must never reach a loader")


@pytest.mark.parametrize("field", ["batch_id", "parent_policy_revision", "consumed_experience_ids"])
def test_result_verification_rejects_manifest_for_another_update(
    tmp_path: Path, field: str
) -> None:
    """Valid file hashes and plausible outer receipt labels cannot hide a different batch."""

    submitted = job(tmp_path)
    receipt = checkpoint(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = CheckpointManifest.model_validate_json(manifest_path.read_text()).model_copy(
        update={
            "policy_revision": next_policy_revision(submitted),
            "policy_history": (next_policy_revision(submitted), "policy-0"),
        }
    )
    manifest = manifest.model_copy(
        update={field: ("foreign",) if field == "consumed_experience_ids" else "foreign"}
    )
    manifest_path.write_text(manifest.model_dump_json())
    receipt = receipt.model_copy(
        update={
            "manifest_sha256": sha256_json(manifest),
            "policy_revision": manifest.policy_revision,
            "policy_history": manifest.policy_history,
        }
    )
    result = TrainingResult(
        checkpoint=receipt, consumed_experience_ids=("experience-1",), metrics={}
    )
    with pytest.raises(ValueError, match="submitted update"):
        verify_training_result(submitted, result)


def test_custom_optimizer_checkpoint_cannot_masquerade_as_native_verl(tmp_path: Path) -> None:
    """A legacy manifest or standalone optimizer payload cannot satisfy native resume."""

    receipt = checkpoint(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    raw = json.loads(manifest_path.read_text())
    del raw["schema_version"]
    del raw["training_backend"]
    manifest_path.write_text(json.dumps(raw))
    with pytest.raises(ValidationError, match="schema_version"):
        verify_checkpoint(receipt, spec())


def test_exact_batch_recovery_preserves_metrics_and_rejects_id_reuse(tmp_path: Path) -> None:
    """A crash after publication can replay its receipt, but changed feedback cannot reuse an ID."""
    root = tmp_path / "claas-committed"
    receipt = checkpoint(root)
    manifest = verify_checkpoint(receipt, spec()).model_copy(update={"metrics": {"loss": 0.25}})
    (root / "manifest.json").write_text(manifest.model_dump_json())
    batch = job(tmp_path).batch
    recovered = recover_training_result(tmp_path, spec(), batch, "main")
    assert recovered is not None
    assert recovered.metrics == {"loss": 0.25}
    assert recovered.checkpoint.step == 1
    changed = batch.model_copy(
        update={"examples": (batch.examples[0].model_copy(update={"text_feedback": "Different"}),)}
    )
    with pytest.raises(ValueError, match="different immutable update"):
        recover_training_result(tmp_path, spec(), changed, "main")
    with pytest.raises(ValueError, match="different immutable update"):
        recover_training_result(tmp_path, spec(), batch, "another-lineage")


def test_checkpoint_metrics_must_be_finite(tmp_path: Path) -> None:
    """A receipt cannot publish NaN or infinity as durable optimizer evidence."""
    receipt = checkpoint(tmp_path)
    raw = verify_checkpoint(receipt, spec()).model_dump(mode="python")
    raw["metrics"] = {"loss": float("inf")}
    with pytest.raises(ValidationError, match="finite"):
        CheckpointManifest.model_validate(raw)
