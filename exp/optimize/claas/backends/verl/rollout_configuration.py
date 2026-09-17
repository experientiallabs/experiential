"""Pinned public veRL rollout configuration for one colocated GPU."""

from verl.workers.config import HFModelConfig, RolloutConfig

from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings
from exp.optimize.claas.training_contracts import ClaasTrainingSpec


def rollout_config(
    settings: ResidentVerlSettings, model: HFModelConfig, spec: ClaasTrainingSpec
) -> RolloutConfig:
    """Freeze context, sampling and native sleep mode.

    Temperature, nucleus and top-k transforms are disabled. veRL's native valid-
    vocabulary mask still excludes OOV and image placeholders; emitted behavior
    log probabilities describe that actual sampling distribution.
    """
    return RolloutConfig(
        name="vllm",
        mode="async",
        dtype="bfloat16",
        load_format="auto",
        tensor_model_parallel_size=1,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        gpu_memory_utilization=settings.rollout_gpu_memory_utilization,
        max_model_len=spec.max_sequence_tokens,
        max_num_seqs=1,
        max_num_batched_tokens=spec.max_sequence_tokens,
        prompt_length=spec.max_sequence_tokens,
        response_length=settings.maximum_output_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        free_cache_engine=True,
        enable_sleep_mode=True,
        calculate_log_probs=True,
        logprobs_mode="raw_logprobs",
        seed=spec.seed,
        engine_kwargs={"vllm": {"tokenizer": model.tokenizer_path}},
    )
