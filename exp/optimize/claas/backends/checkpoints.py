"""Content verification for complete CLaaS student, teacher, and optimizer checkpoints."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, FiniteFloat

from exp.common.core.artifacts import ContractModel, Sha256, sha256_json
from exp.optimize.claas.training_contracts import (
    ClaasTrainingSpec,
    TrainingBatch,
    TrainingCheckpoint,
    TrainingJob,
    TrainingResult,
    next_policy_revision,
)


class CheckpointManifest(ContractModel):
    """Exact frozen configuration and digests for every resumable state file."""

    schema_version: Literal[2]
    training_backend: Literal["verl-fsdp-0.9.0"]
    spec: ClaasTrainingSpec
    policy_revision: str = Field(min_length=1)
    parent_policy_revision: str = Field(min_length=1)
    policy_history: tuple[str, ...] = Field(min_length=1)
    step: int = Field(strict=True, ge=1)
    batch_id: str = Field(min_length=1)
    batch_sha256: Sha256
    metrics: dict[str, FiniteFloat]
    consumed_experience_ids: tuple[str, ...] = Field(min_length=1)
    files: dict[str, Sha256] = Field(min_length=1)
    lineage_id: str = Field(default="main", min_length=1, max_length=512)
    serving_adapter_directory: Literal["student", "serving"] = "student"


def hash_file(path: Path) -> str:
    """Hash a regular payload without loading model state or executing pickle."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(
    checkpoint: TrainingCheckpoint, spec: ClaasTrainingSpec
) -> CheckpointManifest:
    """Fail before compute if the checkpoint, scope, recipe, or any payload has drifted."""
    root = Path(checkpoint.path)
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("checkpoint must name an absolute immutable directory, without symlinks")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("checkpoint manifest cannot be a symlink")
    manifest = CheckpointManifest.model_validate_json(manifest_path.read_text())
    if (
        sha256_json(manifest) != checkpoint.manifest_sha256
        or manifest.spec != spec
        or manifest.policy_revision != checkpoint.policy_revision
        or manifest.step != checkpoint.step
        or manifest.policy_history != checkpoint.policy_history
        or manifest.policy_history[0] != checkpoint.policy_revision
        or len(set(manifest.policy_history)) != len(manifest.policy_history)
        or len(manifest.policy_history) > min(checkpoint.step + 1, spec.max_policy_lag + 1)
        or checkpoint.scope != spec.scope
        or checkpoint.adapter_id != spec.adapter_id
    ):
        raise ValueError("checkpoint identity, training recipe, or manifest digest does not match")
    for relative, expected in manifest.files.items():
        parts = PurePosixPath(relative)
        if parts.is_absolute() or ".." in parts.parts or relative == "manifest.json":
            raise ValueError("checkpoint contains an invalid payload path")
        payload = root / relative
        if any(part.is_symlink() for part in (payload, *payload.parents)):
            raise ValueError("checkpoint payload cannot be a symlink")
        if not payload.is_file() or hash_file(payload) != expected:
            raise ValueError(f"checkpoint payload is missing or changed: {relative}")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual != set(manifest.files) | {"manifest.json"}:
        raise ValueError("checkpoint contains files outside its manifest")
    required = {
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
        f"{manifest.serving_adapter_directory}/adapter_config.json",
        f"{manifest.serving_adapter_directory}/adapter_model.safetensors",
    }
    if not required.issubset(manifest.files):
        raise ValueError(
            "checkpoint lacks student, teacher, serving, or native veRL resumable optimizer state"
        )
    return manifest


@contextmanager
def checkpoint_snapshot(
    checkpoint: TrainingCheckpoint | None, spec: ClaasTrainingSpec
) -> Iterator[TrainingCheckpoint | None]:
    """Bind resume loading to verified bytes copied into a private worker directory.

    A writer may replace the original checkpoint concurrently. Only copies whose
    content matches the receipt's manifest are exposed to model/optimizer loaders.
    The private directory is removed after loading and training finish.
    """
    if checkpoint is None:
        yield None
        return
    manifest = verify_checkpoint(checkpoint, spec)
    with tempfile.TemporaryDirectory(prefix="claas-resume-") as directory:
        root = Path(directory).resolve()
        for relative, expected in manifest.files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(checkpoint.path) / relative, target)
            if hash_file(target) != expected:
                raise ValueError("checkpoint changed while staging verified resume state")
            target.chmod(0o400)
        manifest_path = root / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json())
        manifest_path.chmod(0o400)
        staged = checkpoint.model_copy(update={"path": str(root)})
        verify_checkpoint(staged, spec)
        yield staged


def verify_training_result(job: TrainingJob, result: TrainingResult) -> CheckpointManifest:
    """Bind a complete checkpoint's manifest and receipt to the exact submitted update."""
    manifest = verify_checkpoint(result.checkpoint, job.spec)
    expected_ids = tuple(item.experience.experience_id for item in job.batch.examples)
    expected_revision = next_policy_revision(job)
    prior_history = (
        job.resume_checkpoint.policy_history
        if job.resume_checkpoint
        else (job.spec.initial_policy_revision,)
    )
    expected_history = (expected_revision, *prior_history)[: job.spec.max_policy_lag + 1]
    if (
        result.checkpoint.policy_revision != expected_revision
        or result.checkpoint.step
        != (job.resume_checkpoint.step if job.resume_checkpoint else 0) + 1
        or result.consumed_experience_ids != expected_ids
        or manifest.consumed_experience_ids != expected_ids
        or manifest.batch_id != job.batch.batch_id
        or manifest.batch_sha256 != sha256_json(job.batch)
        or manifest.metrics != result.metrics
        or manifest.parent_policy_revision != job.batch.expected_policy_revision
        or manifest.policy_history != expected_history
        or manifest.lineage_id != job.lineage_id
    ):
        raise ValueError("checkpoint manifest and worker receipt do not match the submitted update")
    return manifest


def checkpoint_receipt(root: Path, manifest: CheckpointManifest) -> TrainingCheckpoint:
    """Describe immutable state without loading executable model or optimizer payloads."""
    return TrainingCheckpoint(
        scope=manifest.spec.scope,
        adapter_id=manifest.spec.adapter_id,
        policy_revision=manifest.policy_revision,
        policy_history=manifest.policy_history,
        step=manifest.step,
        path=str(root),
        manifest_sha256=sha256_json(manifest),
    )


def recover_training_result(
    root: Path, spec: ClaasTrainingSpec, batch: TrainingBatch, lineage_id: str
) -> TrainingResult | None:
    """Recover a published update before policy freshness checks, rejecting ID collisions."""
    found: TrainingResult | None = None
    for path in root.glob("claas-*/manifest.json"):
        manifest = CheckpointManifest.model_validate_json(path.read_bytes())
        if manifest.batch_id != batch.batch_id:
            continue
        receipt = checkpoint_receipt(path.parent, manifest)
        verify_checkpoint(receipt, spec)
        if (
            manifest.lineage_id != lineage_id
            or manifest.batch_sha256 != sha256_json(batch)
            or manifest.parent_policy_revision != batch.expected_policy_revision
            or manifest.consumed_experience_ids
            != tuple(item.experience.experience_id for item in batch.examples)
        ):
            raise ValueError("batch ID already names a different immutable update")
        if found is not None:
            raise ValueError("batch ID has multiple committed checkpoints; inspect the lineage")
        found = TrainingResult(
            checkpoint=receipt,
            consumed_experience_ids=manifest.consumed_experience_ids,
            metrics=manifest.metrics,
        )
    return found
