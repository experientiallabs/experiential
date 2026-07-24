"""On-policy distillation optimizer mode.

Trains a Tinker LoRA student from rollouts the pi agent produces on harbor
tasks (TerminalBench-2). The teacher scores the student's sampled tokens via
compute_logprobs, and reverse-KL advantages feed the importance_sampling loss.
The public surface is deliberately minimal for now: the per-run TOML config
model and its loader.
"""

from wmh.distill.config import DistillConfig, load_distill_config

__all__ = ["DistillConfig", "load_distill_config"]
