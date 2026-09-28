"""Verify explicit egress, history ordering, bounded cleanup, and real offline SDK recording."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import wandb

from exp.common.core.artifacts import SecretBoundaryError
from exp.common.observability.metrics import MetricRecord
from exp.runtime.observability.wandb import WandbMetricSink


class FakeRun:
    def __init__(self) -> None:
        self.step = 7
        self.disabled = False
        self.offline = False
        self.url = "https://wandb.ai/example/project/runs/fixture"
        self.records: list[tuple[dict[str, float], int, bool]] = []
        self.finish_calls = 0
        self.close_error: Exception | None = None

    def log(self, values: dict[str, float], *, step: int, commit: bool) -> None:
        self.records.append((values, step, commit))

    def finish(self, *, exit_code: int = 0) -> None:
        self.finish_calls += 1
        if self.close_error:
            raise self.close_error


def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeRun, dict[str, object]]:
    run = FakeRun()
    captured: dict[str, object] = {}

    def initialize(**arguments: object) -> wandb.Run:
        captured.update(arguments)
        return cast(wandb.Run, run)

    monkeypatch.setattr(wandb, "init", initialize)
    monkeypatch.setattr(wandb, "setup", lambda: SimpleNamespace(settings=wandb.Settings()))
    monkeypatch.setattr(wandb, "login", lambda *, timeout, force: True)
    return run, captured


def test_explicit_payload_history_and_privacy_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, arguments = fake_sdk(monkeypatch)
    with WandbMetricSink(
        project="project", run_id="fixture", directory=tmp_path, config={"lora_rank": 16}
    ) as sink:
        assert sink.next_step == 7 and sink.url == run.url
        sink.record(MetricRecord(event_id="gpu-1", step=7, values={"gpu/utilization": 50}))
        sink.record(MetricRecord(event_id="update-1", values={"train/optimizer_step": 1}))
        sink.record(MetricRecord(event_id="eval-1", step=12, values={"eval/successes": 8}))
        assert sink.next_step == 13
        with pytest.raises(ValueError, match="precedes next_step"):
            sink.record(MetricRecord(event_id="old", step=7, values={"loss": 1}))
    sink.close()
    assert run.finish_calls == 1
    assert run.records == [
        ({"gpu/utilization": 50.0}, 7, True),
        ({"train/optimizer_step": 1.0}, 8, True),
        ({"eval/successes": 8.0}, 12, True),
    ]
    assert arguments["config"] == {"lora_rank": 16}
    assert arguments["reinit"] == "create_new" and arguments["sync_tensorboard"] is False
    settings = cast(wandb.Settings, arguments["settings"])
    assert settings.console == "off" and settings.disable_code and settings.disable_git
    assert settings.x_disable_meta and settings.x_disable_stats and settings.x_disable_machine_info
    assert settings.config_paths == [] and settings.capture_loggers == {}
    assert not settings.x_save_requirements and settings.disable_job_creation
    assert settings.finish_timeout == settings.init_timeout == 30 and settings.finish_timeout_raises
    assert settings.stop_fn is not None
    settings.stop_fn()  # The metrics UI must never send SIGINT to the learning process.
    with pytest.raises(ValueError, match="sink is closed"):
        sink.record(MetricRecord(event_id="late", values={"loss": 1}))


def test_bad_config_rejected_before_sdk_or_directory_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, captured = fake_sdk(monkeypatch)
    directory = tmp_path / "not-created"
    with pytest.raises(SecretBoundaryError):
        WandbMetricSink(
            project="p",
            run_id="r",
            directory=directory,
            config={"credentials": {"api_key": "private"}},
        )
    with pytest.raises(ValueError, match="timeouts"):
        WandbMetricSink(project="p", run_id="r", directory=directory, finish_timeout_seconds=0)
    assert captured == {} and not directory.exists()


def test_global_config_is_rejected_before_run_egress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, captured = fake_sdk(monkeypatch)
    inherited = wandb.Settings(config_paths=["unrelated-private-config.yaml"])
    monkeypatch.setattr(wandb, "setup", lambda: SimpleNamespace(settings=inherited))
    with pytest.raises(ValueError, match="separate metrics reporter process"):
        WandbMetricSink(project="p", run_id="r", directory=tmp_path / "not-created")
    assert not captured and inherited.config_paths == ["unrelated-private-config.yaml"]


def test_online_auth_failure_does_not_create_a_disabled_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, captured = fake_sdk(monkeypatch)
    monkeypatch.setattr(wandb, "login", lambda *, timeout, force: False)
    with pytest.raises(ValueError, match="authentication failed"):
        WandbMetricSink(project="p", run_id="r", directory=tmp_path)
    assert not captured
    monkeypatch.setattr(wandb, "login", lambda *, timeout, force: True)
    run.disabled = True
    with pytest.raises(ValueError, match="requested reporting mode"):
        WandbMetricSink(project="p", run_id="r", directory=tmp_path)
    assert run.finish_calls == 1


def test_mutation_is_revalidated_and_cleanup_cannot_replace_caller_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    run, _ = fake_sdk(monkeypatch)
    sink = WandbMetricSink(project="p", run_id="r", directory=tmp_path)
    record = MetricRecord(event_id="r", values={"loss": 1})
    record.values["loss"] = float("nan")
    with pytest.raises(ValueError):
        sink.record(record)
    assert not run.records and sink.next_step == 7
    run.close_error = TimeoutError("upload deadline")
    with pytest.raises(RuntimeError, match="original"):
        with sink:
            raise RuntimeError("original")
    assert "cleanup failed (TimeoutError)" in caplog.text


def test_offline_sdk_writes_real_history_without_provider_access(tmp_path: Path) -> None:
    script = """
import pathlib, sys
import wandb
from exp.common.observability.metrics import MetricRecord
from exp.runtime.observability.wandb import WandbMetricSink
directory=pathlib.Path(sys.argv[1])
with WandbMetricSink(
    project='existing-caller',run_id='existing',directory=directory,mode='offline'
) as existing:
  with WandbMetricSink(
    project='exp-offline-test',run_id='offline-fixture',directory=directory,
    mode='offline',config={'rank':16}
  ) as sink:
    first=sink.next_step
    sink.record(MetricRecord(event_id='eval-1',values={'eval/success':1},step=first))
    sink.record(MetricRecord(event_id='update-1',values={'train/optimizer_step':1,'train/loss':-0.5}))
    assert sink.next_step==first+2
    assert sink.url is None
  existing.record(MetricRecord(event_id='still-open',values={'caller/success':1}))
assert len(list(directory.glob('wandb/offline-run-*/run-*.wandb')))==2
target=next(directory.glob('wandb/offline-run-*-offline-fixture/run-*.wandb')).read_bytes()
existing=next(directory.glob('wandb/offline-run-*-existing/run-*.wandb')).read_bytes()
assert b'eval/success' in target and b'train/loss' in target
assert b'caller/success' not in target
assert b'caller/success' in existing
assert b'train/loss' not in existing

"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    files = list(tmp_path.glob("wandb/offline-run-*/files/*"))
    assert all(path.name not in {"output.log", "wandb-metadata.json"} for path in files)
    assert not [
        file
        for directory in tmp_path.rglob("code")
        for file in directory.rglob("*")
        if file.is_file()
    ]
