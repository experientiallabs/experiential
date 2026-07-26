"""On-policy distillation optimizer mode.

Trains a Tinker LoRA student from rollouts harbor's own terminus-2 agent
produces on harbor tasks (TerminalBench-2), sampling the student through
Tinker. The teacher scores the student's sampled tokens via
compute_logprobs, and the reverse-KL advantages feed the advantage-weighted
loss `train.loss` names (`importance_sampling` or `ppo`).
The public surface is deliberately minimal for now: the per-run TOML config
model and its loader.
"""

from wmo.distill.config import DistillConfig, load_distill_config

__all__ = ["DistillConfig", "load_distill_config"]
