"""One resident actor, EMA teacher and process group with native veRL state ownership."""

from __future__ import annotations

import gc
import math
from contextlib import ExitStack
from pathlib import Path
from typing import cast

import torch
from filelock import FileLock
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model_state_dict
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from verl.utils import tensordict_utils as tu
from verl.utils.fsdp_utils import offload_fsdp_model_to_cpu
from verl.workers.engine_workers import TrainingWorker

from exp.common.core.artifacts import sha256_json
from exp.optimize.claas.backends.checkpoints import (
    CheckpointManifest,
    checkpoint_receipt,
    checkpoint_snapshot,
    recover_training_result,
    verify_checkpoint,
)
from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings
from exp.optimize.claas.backends.verl.inputs import build_engine_batch
from exp.optimize.claas.backends.verl.native import (
    _validate_model_reference,
    require_worker_runtime,
    single_rank_process_group,
    worker_config,
)
from exp.optimize.claas.backends.verl.objective import FeedbackLoss
from exp.optimize.claas.backends.verl.state import _module, publish_checkpoint, update_teacher
from exp.optimize.claas.training_contracts import (
    ClaasTrainingError,
    ClaasTrainingSpec,
    TrainingBatch,
    TrainingCheckpoint,
    TrainingJob,
    TrainingResult,
    next_policy_revision,
)


class ResidentTrainer:
    """Own native veRL objects until close; all calls execute on one controller thread."""

    def __init__(self, spec: ClaasTrainingSpec, settings: ResidentVerlSettings) -> None:
        """Bind immutable configuration without loading model weights."""
        self.spec, self.settings = spec, settings
        self.root = (
            settings.checkpoint_root.resolve()
            / sha256_json(
                {"scope": spec.scope.model_dump(mode="json"), "adapter_id": spec.adapter_id}
            )
            / sha256_json({"lineage_id": settings.lineage_id})
        )
        self.stack = ExitStack()
        self.actor: TrainingWorker | None = None
        self.teacher: TrainingWorker | None = None
        self.checkpoint: TrainingCheckpoint | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.model_path: Path | None = None
        self.tokenizer_path: Path | None = None

    def initialize(self, resume: TrainingCheckpoint | None) -> None:
        """Acquire the lineage and initialize native workers exactly once per resident run."""
        require_worker_runtime()
        _validate_model_reference(self.spec.base_model, self.spec.model_revision)
        _validate_model_reference(self.spec.tokenizer_id, self.spec.tokenizer_revision)
        self.root.mkdir(parents=True, exist_ok=True)
        self.stack.enter_context(FileLock(self.root / ".training.lock", timeout=0))
        try:
            resume = self._latest(resume)
            self.model_path = Path(
                snapshot_download(self.spec.base_model, revision=self.spec.model_revision)
            )
            self.tokenizer_path = Path(
                snapshot_download(self.spec.tokenizer_id, revision=self.spec.tokenizer_revision)
            )
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.tokenizer_path), trust_remote_code=False
            )
            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise ValueError("resident runtime requires a Hugging Face text tokenizer")
            self.tokenizer = tokenizer
            self.stack.enter_context(single_rank_process_group())
            torch.manual_seed(self.spec.seed)
            self.actor = TrainingWorker(
                worker_config(self.spec, self.model_path, self.tokenizer_path, teacher=False)
            )
            self.actor.reset()
            self.actor.to("cpu")
            self.teacher = TrainingWorker(
                worker_config(self.spec, self.model_path, self.tokenizer_path, teacher=True)
            )
            self.teacher.reset()
            offload_fsdp_model_to_cpu(_module(self.teacher))
            with checkpoint_snapshot(resume, self.spec) as private:
                if private is not None:
                    self.teacher.load_checkpoint(
                        str(Path(private.path) / "verl/teacher"), del_local_after_load=False
                    )
                    self.actor.load_checkpoint(
                        str(Path(private.path) / "verl/actor"), del_local_after_load=False
                    )
                else:
                    update_teacher(self.actor, self.teacher, 1.0)
            self.checkpoint = resume
            self.offload()
        except BaseException:
            self.close()
            raise

    def _latest(self, resume: TrainingCheckpoint | None) -> TrainingCheckpoint | None:
        """Restore the last committed update when a caller crashed before saving its receipt."""
        if resume is not None:
            manifest = verify_checkpoint(resume, self.spec)
            if manifest.lineage_id != self.settings.lineage_id:
                raise ValueError("resume checkpoint belongs to another lineage")
        latest = resume
        records: dict[int, tuple[CheckpointManifest, TrainingCheckpoint]] = {}
        for path in self.root.glob("claas-*/manifest.json"):
            manifest = CheckpointManifest.model_validate_json(path.read_bytes())
            receipt = checkpoint_receipt(path.parent, manifest)
            verify_checkpoint(receipt, self.spec)
            if manifest.lineage_id != self.settings.lineage_id or receipt.step in records:
                raise ValueError("checkpoint lineage is ambiguous; inspect its immutable records")
            if (
                resume is not None
                and receipt.step == resume.step
                and (
                    receipt.policy_revision != resume.policy_revision
                    or receipt.manifest_sha256 != resume.manifest_sha256
                )
            ):
                raise ValueError("resume checkpoint conflicts with the committed lineage")
            records[receipt.step] = (manifest, receipt)
            if latest is None or receipt.step > latest.step:
                latest = receipt
        previous: TrainingCheckpoint | None = None
        for step in sorted(records):
            manifest, receipt = records[step]
            if previous is not None and (
                step != previous.step + 1
                or manifest.parent_policy_revision != previous.policy_revision
            ):
                raise ValueError("checkpoint lineage has a gap or conflicting parent")
            previous = receipt
        if resume is not None and records and resume.step > max(records):
            raise ValueError("resume checkpoint is newer than this committed lineage")
        return latest

    @property
    def policy_revision(self) -> str:
        """Return the last committed student identity, never an in-progress update."""
        return (
            self.checkpoint.policy_revision
            if self.checkpoint
            else self.spec.initial_policy_revision
        )

    def offload(self) -> None:
        """Release actor GPU state; forward-only veRL owns teacher CPU offload."""
        if self.actor is not None:
            self.actor.to("cpu", model=True, optimizer=True, grad=True)
        torch.cuda.empty_cache()

    def adapter(self) -> tuple[dict[str, torch.Tensor], LoraConfig]:
        """Snapshot complete NO_SHARD LoRA state on CPU for upstream rollout transfer."""
        if self.actor is None:
            raise ClaasTrainingError("resident actor is not initialized")
        model = cast(PeftModel, _module(self.actor).module)
        tensors = {
            name: value.detach().cpu().clone()
            for name, value in get_peft_model_state_dict(model).items()
        }
        return tensors, cast(LoraConfig, model.peft_config["default"])

    def train(self, batch: TrainingBatch) -> TrainingResult:
        """Update persistent native state once or recover an exact already-committed batch."""
        recovered = recover_training_result(self.root, self.spec, batch, self.settings.lineage_id)
        if recovered is not None:
            return recovered
        if self.actor is None or self.teacher is None or self.tokenizer is None:
            raise ClaasTrainingError("resident workers are not initialized")
        job = TrainingJob(
            spec=self.spec,
            batch=batch,
            checkpoint_root=str(self.settings.checkpoint_root),
            resume_checkpoint=self.checkpoint,
            lineage_id=self.settings.lineage_id,
        )
        student = build_engine_batch(job, self.tokenizer)
        loss = FeedbackLoss(job)
        if self.spec.objective != "reinforce":
            target = FeedbackLoss(job, teacher=True)
            self.teacher.set_loss_fn(target)
            self.teacher.infer_batch(build_engine_batch(job, self.tokenizer, teacher=True))
            loss.teacher_logits = target.teacher_logits
        self.actor.to("device")
        self.actor.set_loss_fn(loss)
        output = self.actor.train_batch(student)
        reported = tu.get(output, "metrics")
        metrics = {name: float(reported[name]) for name in ("loss", "grad_norm")}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ClaasTrainingError("veRL returned nonfinite metrics; checkpoint refused")
        metrics["response_tokens"] = float(loss.response_tokens)
        update_teacher(self.actor, self.teacher, self.spec.teacher_update_rate)
        result = publish_checkpoint(
            job, self.actor, self.teacher, metrics, self.root / next_policy_revision(job)
        )
        self.checkpoint = result.checkpoint
        self.actor.set_loss_fn(None)
        self.teacher.set_loss_fn(None)
        self.offload()
        return result

    def close(self) -> None:
        """Release workers before destroying the owned process group and lineage lock."""
        self.actor = self.teacher = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.stack.close()
