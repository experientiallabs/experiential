"""Resident checkpoint recovery plus opt-in native CUDA multi-update and resume proof."""

import os
from pathlib import Path
from typing import cast

import pytest
import torch
from safetensors.torch import load_file

from exp.common.core.artifacts import sha256_json
from exp.optimize.claas.backends.checkpoints import verify_checkpoint
from exp.optimize.claas.backends.checkpoints_test import checkpoint
from exp.optimize.claas.backends.verl import resident
from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings
from exp.optimize.claas.backends.verl.engine import ClaasFeedbackEngine
from exp.optimize.claas.backends.verl.native_test import tiny_snapshot
from exp.optimize.claas.backends.verl.resident import ResidentTrainer
from exp.optimize.claas.training_contracts import TrainingBatch
from exp.optimize.claas.training_contracts_test import example, job, spec


def _assert_trainable_adam_checkpoint(
    path: Path,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    trainable_count: int,
) -> None:
    """Match saved Adam entries to every live trainable LoRA tensor by parameter group ID."""
    trainable = {
        id(parameter): name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    assert len(trainable) == trainable_count > 0
    assert all("lora_" in name for name in trainable.values())
    saved = torch.load(path, weights_only=True)
    parameters = {
        index: parameter
        for saved_group, live_group in zip(
            saved["param_groups"], optimizer.param_groups, strict=True
        )
        for index, parameter in zip(saved_group["params"], live_group["params"], strict=True)
    }
    expected = {index for index, parameter in parameters.items() if id(parameter) in trainable}
    assert len(expected) == len(trainable)
    active = {index for index, value in saved["state"].items() if value}
    assert active == expected
    for index in expected:
        value = saved["state"][index]
        assert {"step", "exp_avg", "exp_avg_sq"} <= value.keys()
        assert float(value["step"]) == step
        for key in ("exp_avg", "exp_avg_sq"):
            assert value[key].shape == parameters[index].shape
            assert torch.isfinite(value[key]).all()


@pytest.mark.parametrize("corruption", [None, "missing", "empty", "step", "moment", "frozen"])
def test_adam_checkpoint_assertion_requires_every_trainable_parameter(
    tmp_path: Path, corruption: str | None
) -> None:
    """Empty frozen state is valid; missing, incomplete or stale trainable state is rejected."""
    module = torch.nn.Module()
    module.register_parameter("lora_A", torch.nn.Parameter(torch.ones(2)))
    module.register_parameter("lora_B", torch.nn.Parameter(torch.ones(3)))
    module.register_parameter("frozen", torch.nn.Parameter(torch.ones(4), requires_grad=False))
    parameters = dict(module.named_parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameters["frozen"], parameters["lora_B"]]},
            {"params": [parameters["lora_A"]]},
        ]
    )
    for _ in range(3):
        for parameter in parameters.values():
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad()
    optimizer.state[parameters["frozen"]] = {}
    saved = optimizer.state_dict()
    if corruption == "missing":
        del saved["state"][1]
    elif corruption == "empty":
        saved["state"][1] = {}
    elif corruption == "step":
        saved["state"][1]["step"] = torch.tensor(1.0)
    elif corruption == "moment":
        del saved["state"][1]["exp_avg"]
    elif corruption == "frozen":
        saved["state"][0] = {"step": torch.tensor(3.0)}
    path = tmp_path / "optimizer.pt"
    torch.save(saved, path)
    if corruption is None:
        _assert_trainable_adam_checkpoint(path, module, optimizer, step=3, trainable_count=2)
    else:
        with pytest.raises(AssertionError):
            _assert_trainable_adam_checkpoint(path, module, optimizer, step=3, trainable_count=2)


def test_published_exact_batch_recovers_before_stale_policy_or_worker_checks(
    tmp_path: Path,
) -> None:
    """Recovery returns committed evidence even when no optimizer has been initialized yet."""
    trainer = ResidentTrainer(spec(), ResidentVerlSettings(checkpoint_root=tmp_path))
    trainer.root.mkdir(parents=True)
    receipt = checkpoint(trainer.root / "claas-complete")
    trainer.checkpoint = receipt
    result = trainer.train(job(tmp_path).batch)
    assert result.checkpoint == receipt
    assert trainer.actor is None
    assert trainer._latest(None) == receipt
    changed = job(tmp_path).batch.model_copy(
        update={"examples": (example().model_copy(update={"scalar_reward": -1.0}),)}
    )
    with pytest.raises(ValueError, match="different immutable update"):
        trainer.train(changed)


def test_resume_rejects_a_valid_checkpoint_from_another_lineage(tmp_path: Path) -> None:
    """Valid payload hashes cannot substitute a checkpoint from another training lineage."""
    receipt = checkpoint(tmp_path / "foreign")
    manifest = verify_checkpoint(receipt, spec()).model_copy(update={"lineage_id": "other"})
    (Path(receipt.path) / "manifest.json").write_text(manifest.model_dump_json())
    receipt = receipt.model_copy(update={"manifest_sha256": sha256_json(manifest)})
    trainer = ResidentTrainer(spec(), ResidentVerlSettings(checkpoint_root=tmp_path / "owned"))
    with pytest.raises(ValueError, match="another lineage"):
        trainer._latest(receipt)


@pytest.mark.skipif(
    os.environ.get("CLAAS_RUN_CUDA_INTEGRATION") != "1" or not torch.cuda.is_available(),
    reason="requires explicit CLAAS_RUN_CUDA_INTEGRATION=1 and an authorized CUDA GPU",
)
def test_cuda_resident_native_updates_recovery_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep actual veRL workers alive across two updates and recover their native state."""
    snapshot = tiny_snapshot(tmp_path / "base")
    # Only immutable asset resolution is replaced by a deterministic local fixture.
    monkeypatch.setattr(resident, "snapshot_download", lambda *_args, **_kwargs: str(snapshot))
    monkeypatch.setattr(resident, "_validate_model_reference", lambda _id, _revision: None)
    settings = ResidentVerlSettings(checkpoint_root=tmp_path / "results")
    training_spec = spec().model_copy(update={"target_modules": ("q_proj", "v_proj")})
    trainer = ResidentTrainer(training_spec, settings)
    trainer.initialize(None)
    actor, teacher = trainer.actor, trainer.teacher
    try:
        first = trainer.train(job(tmp_path).batch)
        repeated = trainer.train(job(tmp_path).batch)
        assert repeated == first
        second_batch = TrainingBatch(
            batch_id="batch-2",
            expected_policy_revision=first.checkpoint.policy_revision,
            examples=(example(policy=first.checkpoint.policy_revision),),
        )
        second = trainer.train(second_batch)
        assert trainer.actor is actor and trainer.teacher is teacher
        assert second.checkpoint.step == 2
        first_teacher = load_file(
            str(Path(first.checkpoint.path) / "teacher/adapter_model.safetensors")
        )
        second_student = load_file(
            str(Path(second.checkpoint.path) / "student/adapter_model.safetensors")
        )
        second_teacher = load_file(
            str(Path(second.checkpoint.path) / "teacher/adapter_model.safetensors")
        )
        rate = training_spec.teacher_update_rate
        for name in second_teacher:
            torch.testing.assert_close(
                second_teacher[name].float(),
                first_teacher[name].float() * (1 - rate) + second_student[name].float() * rate,
                atol=1e-5,
                rtol=0.02,
            )
    finally:
        trainer.close()
    reopened = ResidentTrainer(training_spec, settings)
    reopened.initialize(first.checkpoint)
    try:
        assert reopened.checkpoint == second.checkpoint
        assert reopened.train(second_batch) == second
        third = reopened.train(
            TrainingBatch(
                batch_id="batch-3",
                expected_policy_revision=second.checkpoint.policy_revision,
                examples=(example(policy=second.checkpoint.policy_revision),),
            )
        )
        assert reopened.actor is not None
        engine = cast(ClaasFeedbackEngine, reopened.actor.engine)
        assert isinstance(engine.optimizer, torch.optim.AdamW)
        _assert_trainable_adam_checkpoint(
            Path(third.checkpoint.path) / "verl/actor/optim_world_size_1_rank_0.pt",
            engine.module,
            engine.optimizer,
            step=3,
            trainable_count=4,
        )
    finally:
        reopened.close()
