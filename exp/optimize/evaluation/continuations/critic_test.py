"""Critics see only pre-outcome actor context, with durable draws and exact calibration binding."""

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_json
from exp.common.models import ModelCatalog, load_model_catalog, write_model_catalog
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
            reasoning_effort=spec.actor_reasoning_effort,
            probability=0.75,
            verifier_sha256=spec.verifier_sha256,
            probe_sha256=spec.probe_sha256,
        )


def test_critic_dispatches_the_catalog_its_preflight_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A catalog edited after preflight cannot redirect the frozen request to another model."""
    project, job, spec, transport = critic_fixture(tmp_path, monkeypatch)
    path = project.model_catalog_path

    def load_then_drift(source: Path) -> ModelCatalog:
        """Return the checked catalog, then point the critic alias at another model."""
        checked = load_model_catalog(source)
        record = checked.models[job.actor_alias].model_copy(update={"model": "drifted"})
        write_model_catalog(
            path,
            checked.model_copy(update={"models": {**checked.models, job.actor_alias: record}}),
        )
        return checked

    monkeypatch.setattr(actor_critic, "load_model_catalog", load_then_drift)
    predict_success(project, spec, code_revision="test")
    (request,) = transport.requests
    assert request.payload["model"] == spec.critic_model.model_id


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


def test_critic_reasoning_effort_reaches_the_request_and_keys_the_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registered effort is sent with the estimate; unset, the earlier identity is kept."""
    project, _, spec, _ = critic_fixture(tmp_path, monkeypatch)
    assert "reasoning_effort" not in spec.model_dump(mode="json")
    assert critic_request(project.artifacts, spec).reasoning_effort is None
    low = spec.model_copy(update={"reasoning_effort": "low"})
    assert critic_request(project.artifacts, low).reasoning_effort == "low"
    assert low.identity("test") != spec.identity("test")


def test_critic_reasoning_effort_must_be_declared_by_its_deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An effort the critic deployment cannot preserve refuses before any intent or call."""
    project, _, spec, transport = critic_fixture(tmp_path, monkeypatch)
    low = spec.model_copy(update={"reasoning_effort": "low"})
    with pytest.raises(ValueError, match="reasoning effort 'low' is not supported"):
        actor_critic.preflight_critic(project, low)
    with pytest.raises(ValueError, match="reasoning effort 'low' is not supported"):
        predict_success(project, low, code_revision="test")
    assert not transport.requests
    assert not project.artifacts.exists(low.identity("test") + "-intent")


def test_critic_failure_names_the_saved_defect_or_the_lost_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused estimate is classified from saved evidence, so its reason replays exactly."""
    project, _, spec, _ = critic_fixture(tmp_path, monkeypatch, probability=2)
    with pytest.raises(ValueError) as invalid:
        predict_success(project, spec, code_revision="test")
    assert (
        actor_critic.critic_failure(
            project.artifacts, spec, code_revision="test", error=invalid.value
        )
        == "invalid_typed_estimate"
    )
    lost = spec.model_copy(update={"critic_output_tokens": 512})
    monkeypatch.setattr(actor_critic, "HttpxJsonTransport", lambda: ScriptedJsonTransport(()))
    with pytest.raises((AssertionError, RuntimeError)) as dispatch:
        predict_success(project, lost, code_revision="test")
    assert isinstance(dispatch.value, Exception)
    assert (
        actor_critic.critic_failure(
            project.artifacts, lost, code_revision="test", error=dispatch.value
        )
        == "unknown_dispatch_outcome"
    )
    early = spec.model_copy(update={"maximum_input_bytes": 512})
    with pytest.raises(ValueError) as refused:
        predict_success(project, early, code_revision="test")
    assert (
        actor_critic.critic_failure(
            project.artifacts, early, code_revision="test", error=refused.value
        )
        == "ValueError"
    )


def test_missing_critic_credential_refuses_before_intent_and_stays_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A credential refusal dispatched nothing, so it is not saved as an unknown-outcome draw."""
    project, _, spec, transport = critic_fixture(tmp_path, monkeypatch)
    monkeypatch.delenv("HELD_OUT_TEST_KEY")
    with pytest.raises(ValueError):
        predict_success(project, spec, code_revision="test")
    assert not transport.requests
    assert not project.artifacts.exists(spec.identity("test") + "-intent")
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    reference = predict_success(project, spec, code_revision="test")
    assert load_critic_prediction(project.artifacts, reference).estimate.probability == 0.75
    assert len(transport.requests) == 1


def test_actor_reasoning_effort_is_shown_to_the_critic_and_binds_its_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The critic forecasts the registered actor effort; a different effort cannot reuse it."""
    project, _, spec, _ = critic_fixture(tmp_path, monkeypatch)
    assert "actor_reasoning_effort" not in spec.model_dump(mode="json")
    low = spec.model_copy(update={"actor_reasoning_effort": "low"})
    assert low.identity("test") != spec.identity("test")
    content = critic_request(project.artifacts, low).messages[-1].content
    assert content is not None and json.loads(content)["actor_reasoning_effort"] == "low"
    reference = predict_success(project, low, code_revision="test")
    binding = {
        "checkpoint": low.checkpoint,
        "actor_model": low.actor_model,
        "temperature": low.temperature,
        "output_tokens": low.actor_output_tokens,
        "probability": 0.75,
        "verifier_sha256": low.verifier_sha256,
        "probe_sha256": low.probe_sha256,
    }
    require_prediction_binding(project.artifacts, reference, reasoning_effort="low", **binding)
    with pytest.raises(ValueError, match="continuation policy"):
        require_prediction_binding(project.artifacts, reference, reasoning_effort=None, **binding)
