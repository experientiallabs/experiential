"""Real public veRL configuration coverage on hosts without a CUDA runtime."""

from pathlib import Path

from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings
from exp.optimize.claas.backends.verl.native import worker_config
from exp.optimize.claas.backends.verl.native_test import tiny_snapshot
from exp.optimize.claas.backends.verl.rollout_configuration import rollout_config
from exp.optimize.claas.training_contracts_test import spec


def test_colocated_rollout_freezes_exact_probability_and_memory_configuration(
    tmp_path: Path,
) -> None:
    """Upstream typed config selects one device, sleep mode and unfiltered behavior logprobs."""
    snapshot = tiny_snapshot(tmp_path / "model")
    model = worker_config(spec(), snapshot, snapshot, teacher=False).model_config
    settings = ResidentVerlSettings(checkpoint_root=tmp_path, maximum_output_tokens=17)
    config = rollout_config(settings, model, spec())
    assert config.tensor_model_parallel_size == config.data_parallel_size == 1
    assert config.pipeline_model_parallel_size == 1
    assert config.enable_sleep_mode and config.free_cache_engine
    assert config.logprobs_mode == "raw_logprobs"
    assert config.temperature == config.top_p == config.repetition_penalty == 1
    assert config.top_k == -1
    assert config.response_length == 17
    assert config.max_model_len == spec().max_sequence_tokens
    assert config.load_format == "auto"
    assert config.engine_kwargs["vllm"]["tokenizer"] == str(snapshot)
