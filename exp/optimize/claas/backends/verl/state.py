"""Atomic publication of native veRL state and separately consumable LoRA exports."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import cast

import torch
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from verl.workers.engine_workers import TrainingWorker

from exp.common.core.artifacts import sha256_json
from exp.common.core.files import fsync_directory_best_effort
from exp.optimize.claas.backends.checkpoints import (
    CheckpointManifest,
    hash_file,
    verify_checkpoint,
    verify_training_result,
)
from exp.optimize.claas.backends.verl.engine import ClaasFeedbackEngine
from exp.optimize.claas.training_contracts import TrainingCheckpoint, TrainingJob, TrainingResult


def _module(worker: TrainingWorker) -> FSDP:
    """Reject a mismatched engine before accessing its one-rank FSDP state."""
    if not isinstance(worker.engine, ClaasFeedbackEngine):
        raise ValueError("CLaaS state requires the configured veRL feedback engine")
    module = worker.engine.module
    if not isinstance(module, FSDP):
        raise ValueError("CLaaS state requires the native veRL FSDP module")
    if module.sharding_strategy != ShardingStrategy.NO_SHARD:
        raise ValueError("CLaaS adapter export supports only the single-rank veRL FSDP engine")
    return module


def update_teacher(actor: TrainingWorker, teacher: TrainingWorker, rate: float) -> None:
    """Copy an EMA of actor LoRA parameters; the frozen base and optimizer are untouched."""
    student_model = cast(PeftModel, _module(actor).module)
    student = {
        name: value.detach().float().cpu().clone()
        for name, value in get_peft_model_state_dict(student_model).items()
    }
    teacher_model = cast(PeftModel, _module(teacher).module)
    old = get_peft_model_state_dict(teacher_model)
    if student.keys() != old.keys():
        raise ValueError("veRL actor and teacher have incompatible adapter parameter sets")
    with torch.no_grad():
        updated = {
            name: old[name].float().cpu() * (1 - rate) + value * rate
            for name, value in student.items()
        }
        set_peft_model_state_dict(teacher_model, updated)


def _export_adapter(worker: TrainingWorker, destination: Path) -> None:
    """Export complete NO_SHARD parameters using their original full-model PEFT names."""
    model = cast(PeftModel, _module(worker).module)
    model.save_pretrained(str(destination), safe_serialization=True)


def publish_checkpoint(
    job: TrainingJob,
    actor: TrainingWorker,
    teacher: TrainingWorker,
    metrics: dict[str, float],
    destination: Path,
) -> TrainingResult:
    """Ask veRL to save native state, then atomically publish its hash-bound receipt."""
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=destination.parent))
    step = (job.resume_checkpoint.step if job.resume_checkpoint else 0) + 1
    try:
        actor.save_checkpoint(str(temporary / "verl" / "actor"), global_step=step)
        _export_adapter(actor, temporary / "student")
        actor.to("cpu", model=True, optimizer=True, grad=True)
        teacher.save_checkpoint(str(temporary / "verl" / "teacher"), global_step=step)
        _export_adapter(teacher, temporary / "teacher")
        files = {
            str(path.relative_to(temporary)): hash_file(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        manifest = CheckpointManifest(
            schema_version=2,
            training_backend="verl-fsdp-0.9.0",
            spec=job.spec,
            policy_revision=destination.name,
            parent_policy_revision=job.batch.expected_policy_revision,
            policy_history=(
                destination.name,
                *(
                    job.resume_checkpoint.policy_history
                    if job.resume_checkpoint
                    else (job.spec.initial_policy_revision,)
                ),
            )[: job.spec.max_policy_lag + 1],
            step=step,
            batch_id=job.batch.batch_id,
            batch_sha256=sha256_json(job.batch),
            metrics=metrics,
            consumed_experience_ids=tuple(
                item.experience.experience_id for item in job.batch.examples
            ),
            files=files,
            lineage_id=job.lineage_id,
        )
        (temporary / "manifest.json").write_text(manifest.model_dump_json(indent=2))
        provisional = TrainingCheckpoint(
            scope=job.spec.scope,
            adapter_id=job.spec.adapter_id,
            policy_revision=manifest.policy_revision,
            policy_history=manifest.policy_history,
            step=step,
            path=str(temporary),
            manifest_sha256=sha256_json(manifest),
        )
        verify_checkpoint(provisional, job.spec)
        for path in temporary.rglob("*"):
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        for directory in sorted(
            (path for path in temporary.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            fsync_directory_best_effort(directory)
        fsync_directory_best_effort(temporary)
        temporary.rename(destination)
        fsync_directory_best_effort(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    checkpoint = TrainingCheckpoint(
        scope=job.spec.scope,
        adapter_id=job.spec.adapter_id,
        policy_revision=manifest.policy_revision,
        policy_history=manifest.policy_history,
        step=step,
        path=str(destination),
        manifest_sha256=sha256_json(manifest),
    )
    verify_checkpoint(checkpoint, job.spec)
    result = TrainingResult(
        checkpoint=checkpoint,
        metrics=metrics,
        consumed_experience_ids=manifest.consumed_experience_ids,
    )
    verify_training_result(job, result)
    return result
