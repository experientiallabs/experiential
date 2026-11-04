"""Durable bootstrap validation over real grounded worlds and scripted provider transports."""

import json
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import Mock

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.variance import bootstrap, bootstrap_capture
from exp.optimize.evaluation.variance.bootstrap_test import BootScript, _setup
from exp.runtime.models.providers.transport import JsonHttpResponse, ProviderTransportError


@pytest.mark.parametrize(("task", "known_steps"), [("auth", 0), ("recon", 0), ("recon", 1)])
def test_lost_validation_response_never_buys_a_new_sample_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str, known_steps: int
) -> None:
    """Unresolved reveal or prefix dispatches stay failed across repeated task resumes."""
    setup = _setup(tmp_path, monkeypatch)
    predictor = setup.fixture.predictor
    original = predictor.post
    keys: list[str | None] = []

    def lose_response(
        url: str, *, headers: Mapping[str, str], payload: JsonObject, timeout_seconds: float
    ) -> JsonHttpResponse:
        """Execute a prediction, then lose every response after the selected known steps."""
        keys.append({key.lower(): value for key, value in headers.items()}.get("idempotency-key"))
        response = original(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)
        if len(keys) > known_steps:
            raise ProviderTransportError("response lost after provider execution")
        return response

    monkeypatch.setattr(predictor, "post", lose_response)
    first = setup.run(BootScript(), monkeypatch, tasks={task})
    assert first.by_status("failed") == (task,)
    attempts = len(predictor.requests)
    assert attempts > known_steps
    assert len(set(keys[known_steps:])) == 1
    store = setup.project.artifacts
    saved_steps = {
        artifact_id: store.read_bytes(artifact_id, "observation.json")
        for artifact_id in store.list_ids(artifact_type="start-bootstrap-probe-step")
    }
    assert len(saved_steps) == known_steps
    if known_steps:
        observation = json.loads(next(iter(saved_steps.values())))
        assert json.loads(observation["content"]) == {"ticket": "T-1", "status": "open"}
    monkeypatch.setattr(predictor, "post", original)
    for _ in range(2):
        silent = BootScript(lambda role, opening: True)
        resumed = setup.run(silent, monkeypatch, tasks={task})
        assert resumed.by_status("failed") == (task,)
        assert "unfinished" in (resumed.entries[0].failure or "")
        assert len(predictor.requests) == attempts
        assert silent.requests == []
        assert all(
            store.read_bytes(artifact_id, "observation.json") == payload
            for artifact_id, payload in saved_steps.items()
        )


@pytest.mark.parametrize("task", ["auth", "recon"])
def test_completed_validation_probes_replay_after_later_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str
) -> None:
    """A completed reveal or prefix survives interruption in review or job registration."""
    setup = _setup(tmp_path, monkeypatch)
    module = bootstrap_capture if task == "auth" else bootstrap
    function = "check_reveals" if task == "auth" else "mining_job"
    with monkeypatch.context() as interruption:
        interruption.setattr(module, function, Mock(side_effect=RuntimeError("interrupted")))
        first = setup.run(BootScript(), monkeypatch, tasks={task})
    assert first.by_status("failed") == (task,)
    attempts = len(setup.fixture.predictor.requests)
    assert attempts > 0
    resumed = setup.run(BootScript(), monkeypatch, tasks={task})
    assert resumed.by_status("registered") == (task,)
    assert len(setup.fixture.predictor.requests) == attempts
