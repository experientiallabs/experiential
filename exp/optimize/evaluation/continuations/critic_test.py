"""Critics see only pre-outcome actor context, with durable draws and exact calibration binding."""

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_json
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations import critic as actor_critic
from exp.optimize.evaluation.continuations.critic import (
    CriticSpec,
    critic_request,
    load_critic_prediction,
    predict_success,
    require_prediction_binding,
)
from exp.optimize.evaluation.continuations.job import ContinuationJob, execute_continuation_job
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.simulation.continuations import load_continuation_checkpoint


def critic_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, probability: float = 0.75
) -> tuple[ProjectStore, ContinuationJob, CriticSpec, ScriptedJsonTransport]:
    """Produce an explicit typed estimate through the real provider parser, without a live API."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    spec = CriticSpec(
        checkpoint=job.study.checkpoint,
        actor_model=job.study.actor_model,
        verifier_sha256=sha256_json(job.study.verifier),
        probe_sha256=job.study.probe_sha256,
        maximum_model_calls=job.maximum_model_calls,
        system_prompt=job.system_prompt,
        temperature=job.study.temperature,
        actor_output_tokens=job.study.maximum_output_tokens,
        critic_alias=job.actor_alias,
        critic_model=job.study.actor_model,
        catalog_sha256=job.catalog_sha256,
        deadline=job.deadline,
    )
    transport = ScriptedJsonTransport(
        (
            JsonHttpResponse(
                status_code=200,
                body={
                    "id": "critic",
                    "model": "world-v1",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "estimate",
                                        "type": "function",
                                        "function": {
                                            "name": "estimate_success",
                                            "arguments": json.dumps(
                                                {
                                                    "probability": probability,
                                                    "rationale": "Public evidence only.",
                                                }
                                            ),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
            ),
        )
    )
    monkeypatch.setattr(actor_critic, "HttpxJsonTransport", lambda: transport)
    return project, job, spec, transport


def test_critic_request_excludes_world_verifier_and_future_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The projection excludes outcomes even when completed trajectories exist in the store."""
    project, job, spec, transport = critic_fixture(tmp_path, monkeypatch)
    request = critic_request(project.artifacts, spec)
    assert request.messages[-1].content is not None
    public = json.loads(request.messages[-1].content)
    checkpoint = load_continuation_checkpoint(project.artifacts, spec.checkpoint)
    assert set(public) == {
        "messages",
        "tools",
        "actor_model",
        "remaining_calls",
        "actor_temperature",
        "actor_output_tokens",
    }
    assert public["messages"] == [row.model_dump(mode="json") for row in checkpoint.actor.messages]
    assert public["remaining_calls"] == job.maximum_model_calls - checkpoint.actor.calls
    assert "verifier" not in public and "world" not in public and "outcome" not in public
    reference = predict_success(project, spec, code_revision="test")
    prediction = load_critic_prediction(project.artifacts, reference)
    assert prediction.estimate.probability == 0.75
    assert len(transport.requests) == 1
    assert predict_success(project, spec, code_revision="test") == reference
    assert len(transport.requests) == 1


def test_critic_estimate_binds_later_continuations_and_cannot_change_actor_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saved prospective probability is scored on fresh resumed branches, not supplied labels."""
    project, job, spec, transport = critic_fixture(tmp_path, monkeypatch)
    reference = predict_success(project, spec, code_revision="test")
    job = job.model_copy(
        update={
            "study": job.study.model_copy(
                update={
                    "critic_prediction": reference,
                    "predicted_success": 0.75,
                }
            )
        }
    )
    result = execute_continuation_job(project, job, code_revision="test")
    assert result.metrics()["brier"] == (0.25**2 + 0.75**2) / 2
    assert len(transport.requests) == 1
    with pytest.raises(ValueError, match="continuation policy"):
        require_prediction_binding(
            project.artifacts,
            reference,
            checkpoint=spec.checkpoint,
            actor_model=spec.actor_model,
            temperature=0,
            output_tokens=spec.actor_output_tokens,
            probability=0.75,
            verifier_sha256=spec.verifier_sha256,
            probe_sha256=spec.probe_sha256,
        )


def test_invalid_saved_estimate_is_not_redrawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A returned out-of-range probability stays a failed estimate instead of buying a new one."""
    project, _, spec, transport = critic_fixture(tmp_path, monkeypatch, probability=2)
    for _ in range(2):
        with pytest.raises(ValueError, match="less than or equal"):
            predict_success(project, spec, code_revision="test")
    assert len(transport.requests) == 1
    assert project.artifacts.exists(spec.identity("test") + "-response")


def test_unknown_critic_dispatch_refuses_a_second_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lost provider outcome cannot be selected away by resubmitting the same prediction."""
    project, _, spec, _ = critic_fixture(tmp_path, monkeypatch)
    empty = ScriptedJsonTransport(())
    monkeypatch.setattr(actor_critic, "HttpxJsonTransport", lambda: empty)
    with pytest.raises((AssertionError, RuntimeError)):
        predict_success(project, spec, code_revision="test")
    with pytest.raises(ValueError, match="unknown outcome"):
        predict_success(project, spec, code_revision="test")
    assert len(empty.requests) == 1


def test_critic_oversized_public_context_refuses_before_any_intent_or_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Large prefixes are not silently shortened or sent beyond the registered input allowance."""
    project, _, spec, transport = critic_fixture(tmp_path, monkeypatch)
    spec = spec.model_copy(update={"maximum_input_bytes": 512})
    with pytest.raises(ValueError, match="byte allowance"):
        predict_success(project, spec, code_revision="test")
    assert not transport.requests
    assert not project.artifacts.exists(spec.identity("test") + "-intent")
