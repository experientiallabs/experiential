"""Tests for the tau2 rollout source (episode identity, joins, dispatch)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmo.distill.config import DistillConfig
from wmo.distill.rollouts import collect_rollouts
from wmo.distill.tau2 import (
    _assemble_record,
    _EpisodeSpec,
    _tau2_command,
    _wipe_stale_episode_dir,
    collect_tau2_rollouts,
    parse_tau2_task_id,
)
from wmo.harness.doc import HarnessDoc
from wmo.providers.base import ProviderConfig, ProviderKind


def _cfg(**tau2_overrides: object) -> DistillConfig:
    tau2 = {"tau2_bin": "/venv/bin/tau2", "data_dir": "/data"} | dict(tau2_overrides)
    return DistillConfig.model_validate(
        {
            "student": {"base_model": "Qwen/Qwen3.5-9B"},
            "teacher": {"model": "Qwen/Qwen3.6-27B"},
            "tau2": tau2,
        }
    )


def _provider_config(model: str = "tinker://run/sampler_weights/x") -> ProviderConfig:
    return ProviderConfig(kind=ProviderKind.TINKER, model=model, model_type="Qwen/Qwen3.5-9B")


# The shape a real tau2 results.json carries (captured from a live airline
# episode; reward_info trimmed to the fields the join reads).
def _write_results(
    episode_dir: Path, *, reward: float | None = 1.0, termination: str = "user_stop"
) -> None:
    episode_dir.mkdir(parents=True, exist_ok=True)
    reward_info = {"reward": reward} if reward is not None else {}
    payload = {
        "simulations": [
            {
                "id": "df099383",
                "task_id": "0",
                "termination_reason": termination,
                "reward_info": reward_info,
            }
        ]
    }
    (episode_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")


class TestParseTaskId:
    def test_parses_composite_ids(self) -> None:
        assert parse_tau2_task_id("airline/12") == ("airline", "12")
        assert parse_tau2_task_id("telecom/[mixed_id_7]") == ("telecom", "[mixed_id_7]")

    @pytest.mark.parametrize("bad", ["12", "airline/", "mars/3", "/3", ""])
    def test_rejects_malformed_ids(self, bad: str) -> None:
        with pytest.raises(ValueError, match="invalid tau2 task id"):
            parse_tau2_task_id(bad)


class TestCommand:
    def test_command_pins_the_benchmark_surface(self, tmp_path: Path) -> None:
        cfg = _cfg(user_llm="azure/gpt-5.4-mini")
        spec = _EpisodeSpec("airline/7", 2, tmp_path / "step-0000")
        command = _tau2_command(spec, cfg, "http://127.0.0.1:9999/v1")
        text = " ".join(command)
        assert "--domain airline" in text
        assert "--task-ids 7" in text
        assert "--num-trials 1" in text
        assert "--user-llm azure/gpt-5.4-mini" in text
        assert "--auto-resume" in text
        assert f"--agent-llm openai/{spec.name}" in text
        agent_args = json.loads(command[command.index("--agent-llm-args") + 1])
        assert agent_args["api_base"] == "http://127.0.0.1:9999/v1"
        assert agent_args["temperature"] == cfg.sampling.temperature
        assert agent_args["max_tokens"] == cfg.sampling.max_tokens
        # airline uses tau2's default split; only telecom carries an override
        assert "--task-split-name" not in text
        # tau2-internal retries are pinned OFF: a runner-level retry would re-run
        # the simulation into the same span sink, splicing an abandoned attempt's
        # tokens under another attempt's reward. Retries are episode-level.
        assert command[command.index("--max-retries") + 1] == "0"

    def test_telecom_needs_the_full_split(self, tmp_path: Path) -> None:
        spec = _EpisodeSpec("telecom/900", 1, tmp_path / "step-0000")
        command = _tau2_command(spec, _cfg(), "http://x/v1")
        assert command[command.index("--task-split-name") + 1] == "full"

    def test_save_names_are_scoped_per_step_dir(self, tmp_path: Path) -> None:
        # Same episode, different step dirs (another step, another run, another
        # eval key) must never share a tau2 simulations dir: --auto-resume
        # would silently replay the other context's episode.
        a = _EpisodeSpec("airline/7", 1, tmp_path / "run-a" / "tau2" / "step-0000")
        b = _EpisodeSpec("airline/7", 1, tmp_path / "run-a" / "tau2" / "step-0001")
        c = _EpisodeSpec("airline/7", 1, tmp_path / "run-b" / "tau2" / "step-0000")
        assert len({a.save_name, b.save_name, c.save_name}) == 3
        # ... while a resume of the SAME step reuses the same name.
        again = _EpisodeSpec("airline/7", 1, tmp_path / "run-a" / "tau2" / "step-0000")
        assert again.save_name == a.save_name


class TestAssembleRecord:
    def test_joins_reward_and_stop_reason(self, tmp_path: Path) -> None:
        spec = _EpisodeSpec("airline/0", 1, tmp_path)
        _write_results(spec.episode_dir, reward=1.0, termination="user_stop")
        record = _assemble_record(spec)
        assert record.task_id == "airline/0"
        assert record.passed is True
        assert record.reward == 1.0
        assert record.stop_reason == "submitted"
        assert record.infra_failed is False
        assert record.spans == []  # no sink file: died-before-first-completion shape

    @pytest.mark.parametrize(
        ("termination", "expected"),
        [
            ("agent_stop", "submitted"),
            ("max_steps", "max_turns"),
            ("timeout", "budget"),
            ("too_many_errors", "unparsed_tool_call"),
            # the datum builder's whole-episode drop keys on this exact string
            ("context_window_exceeded", "context_overflow"),
            ("unexpected_error", "error"),
        ],
    )
    def test_termination_mapping(self, tmp_path: Path, termination: str, expected: str) -> None:
        spec = _EpisodeSpec("retail/3", 1, tmp_path)
        _write_results(spec.episode_dir, reward=0.0, termination=termination)
        assert _assemble_record(spec).stop_reason == expected

    def test_infrastructure_error_is_infra_failed(self, tmp_path: Path) -> None:
        spec = _EpisodeSpec("airline/0", 1, tmp_path)
        _write_results(spec.episode_dir, reward=None, termination="infrastructure_error")
        record = _assemble_record(spec)
        assert record.infra_failed is True
        assert record.passed is False
        assert record.stop_reason is None

    def test_missing_results_is_infra_failed(self, tmp_path: Path) -> None:
        spec = _EpisodeSpec("airline/0", 1, tmp_path)
        record = _assemble_record(spec)
        assert record.infra_failed is True
        assert record.reward == 0.0

    def test_fractional_reward_is_not_a_pass(self, tmp_path: Path) -> None:
        spec = _EpisodeSpec("airline/0", 1, tmp_path)
        _write_results(spec.episode_dir, reward=0.5, termination="user_stop")
        record = _assemble_record(spec)
        assert record.passed is False
        assert record.reward == 0.5


class TestStaleEpisodeWipe:
    def test_wipes_only_a_different_provider(self, tmp_path: Path) -> None:
        spec = _EpisodeSpec("airline/0", 1, tmp_path / "step-0000")
        current = _provider_config()
        _write_results(spec.episode_dir)
        (spec.episode_dir / "provider.json").write_text(
            json.dumps(
                _provider_config("tinker://OTHER/sampler_weights/y").model_dump(mode="json")
            ),
            encoding="utf-8",
        )
        assert _wipe_stale_episode_dir(spec, current) is True
        assert not spec.episode_dir.exists()

    def test_keeps_a_matching_provider(self, tmp_path: Path) -> None:
        spec = _EpisodeSpec("airline/0", 1, tmp_path / "step-0000")
        current = _provider_config()
        _write_results(spec.episode_dir)
        (spec.episode_dir / "provider.json").write_text(
            json.dumps(current.model_dump(mode="json")), encoding="utf-8"
        )
        assert _wipe_stale_episode_dir(spec, current) is False
        assert (spec.episode_dir / "results.json").exists()

    def test_leaves_an_unreadable_snapshot_alone(self, tmp_path: Path) -> None:
        spec = _EpisodeSpec("airline/0", 1, tmp_path / "step-0000")
        _write_results(spec.episode_dir)
        assert _wipe_stale_episode_dir(spec, _provider_config()) is False


class TestCollectValidation:
    def test_rejects_non_tinker_provider(self, tmp_path: Path) -> None:
        provider = ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-x", model_type="gpt-x")
        with pytest.raises(ValueError, match="kind 'tinker'"):
            collect_tau2_rollouts(
                0, ["airline/0"], _cfg(), HarnessDoc.baseline(), provider, tmp_path
            )

    def test_rejects_negative_step(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="step_index"):
            collect_tau2_rollouts(
                -1, ["airline/0"], _cfg(), HarnessDoc.baseline(), _provider_config(), tmp_path
            )

    def test_e2b_backend_names_the_gap(self, tmp_path: Path) -> None:
        with pytest.raises(NotImplementedError, match="e2b"):
            collect_tau2_rollouts(
                0,
                ["airline/0"],
                _cfg(backend="e2b"),
                HarnessDoc.baseline(),
                _provider_config(),
                tmp_path,
            )


class TestDispatch:
    def test_collect_rollouts_dispatches_on_the_config_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        def _fake_tau2(*args: object, **kwargs: object) -> tuple[list[object], object]:
            calls.append("tau2")
            return [], object()

        monkeypatch.setattr("wmo.distill.tau2.collect_tau2_rollouts", _fake_tau2)
        collect_rollouts(
            0, ["airline/0"], _cfg(), HarnessDoc.baseline(), _provider_config(), tmp_path
        )
        assert calls == ["tau2"]


class TestEpisodeRetry:
    def test_a_failed_attempt_retries_fresh_and_a_graded_one_stops(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # First attempt dies without evidence, second lands a graded simulation;
        # the third would fail the test by over-running the retry budget.
        cfg = _cfg(episode_retries=1)
        cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"group_size": 1})})
        attempts: list[int] = []

        async def _fake_subprocess(spec: _EpisodeSpec, *_args: object) -> None:
            attempts.append(1)
            if len(attempts) == 2:
                _write_results(spec.episode_dir, reward=1.0, termination="user_stop")

        monkeypatch.setattr("wmo.distill.tau2._run_episode_subprocess", _fake_subprocess)
        records, stats = collect_tau2_rollouts(
            0, ["airline/0"], cfg, HarnessDoc.baseline(), _provider_config(), tmp_path
        )
        assert len(attempts) == 2
        [record] = records
        assert record.infra_failed is False
        assert record.passed is True
        assert stats.executed_trials == 1

    def test_exhausted_retries_leave_an_infra_failed_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _cfg(episode_retries=1)
        cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"group_size": 1})})
        attempts: list[int] = []

        async def _never_lands(*_args: object) -> None:
            attempts.append(1)

        monkeypatch.setattr("wmo.distill.tau2._run_episode_subprocess", _never_lands)
        records, stats = collect_tau2_rollouts(
            0, ["airline/0"], cfg, HarnessDoc.baseline(), _provider_config(), tmp_path
        )
        assert len(attempts) == 2  # the original try plus one retry
        [record] = records
        assert record.infra_failed is True
        assert stats.infra_failed_trials == 1
