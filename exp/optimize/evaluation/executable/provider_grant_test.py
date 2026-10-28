"""Fresh provider authority restores old frozen worlds without refilling resumed trials."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, sha256_bytes
from exp.common.models import AssistantAction, ToolCall
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.capture import CaptureDefinition, capture_initial_world
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.study import ContinuationStudySpec
from exp.optimize.evaluation.executable import grounded as grounded_runtime
from exp.optimize.evaluation.executable import mixed as mixed_runtime
from exp.optimize.evaluation.executable.loader import load_runtime_definition
from exp.optimize.evaluation.executable.mixed_test import mixed_fixture
from exp.optimize.evaluation.executable.provider_grant import ProviderGrant, ProviderLimits
from exp.optimize.evaluation.executable.task import ExecutableTask, persist_executable_task
from exp.optimize.evaluation.executable.trial import open_executable_harbor_trial
from exp.runtime.agents.chat_test import _response
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.models.providers.transport import ScriptedJsonTransport
from exp.simulation.evaluation.tasks import TaskVerifier


def expired_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ProjectStore, ArtifactInput, ExecutableTask, ScriptedJsonTransport]:
    """Capture a real mixed checkpoint whose original simulator admission is already expired."""
    project, spec, image, transport = mixed_fixture(tmp_path, monkeypatch)
    spec = spec.model_copy(
        update={
            "simulated": spec.simulated.model_copy(
                update={
                    "deadline": datetime.now(UTC) - timedelta(days=1),
                }
            )
        }
    )
    file = Path(mixed_runtime.__file__)
    configuration = spec.model_dump(mode="json")
    configuration["simulated"]["deadline"] = spec.simulated.deadline.isoformat()
    capture = CaptureDefinition(
        task=spec.simulated.task,
        sources=(image, spec.simulated.world_model),
        runtime_file=file,
        runtime_sha256=sha256_bytes(file.read_bytes()),
        configuration=configuration,
        probe_sha256=spec.probe_sha256,
        maximum_model_calls=4,
    )
    checkpoint = capture_initial_world(project, capture, code_revision="test")
    definition = ExecutableTask(
        intent="Native workbook plus frozen learned read.",
        job=ContinuationJob(
            study=ContinuationStudySpec(
                checkpoint=checkpoint,
                actor_model=_response(AssistantAction(content="done")).model,
                verifier=TaskVerifier(description="Diagnostic native task", outcome_schema={}),
                probe_sha256=spec.probe_sha256,
                selector="grant conformance",
                repeats=1,
            ),
            runtime_file=file,
            runtime_sha256=capture.runtime_sha256,
            configuration=capture.configuration,
            actor_alias="world",
            catalog_sha256=spec.simulated.catalog_sha256,
            maximum_model_calls=4,
            maximum_physical_attempts=1,
            deadline=datetime.now(UTC) - timedelta(days=1),
        ),
    )
    reference = persist_executable_task(
        project, definition, created_at=datetime.now(UTC), code_revision="test"
    )
    return project, reference, definition, transport


def test_new_grant_runs_expired_world_and_resume_cannot_refill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh authority preserves the world, saved replies and one lifetime attempt cap."""
    project, reference, definition, transport = expired_task(tmp_path, monkeypatch)
    limits = ProviderLimits(
        maximum_physical_attempts=1, deadline=datetime.now(UTC) + timedelta(minutes=5)
    )
    read = ToolCall(name="inspect", call_id="read", arguments={"key": "output.xlsx"})
    with open_executable_harbor_trial(
        project, reference, trial_id="grant", code_revision="test", provider_limits=limits
    ) as trial:
        trial.call(workbook_case().first)
        reply = trial.call(read)
        registration = trial.registration
        after = trial.session.checkpoint()
    assert len(transport.requests) == 1 and reply["content"] == "predicted reply"
    with open_executable_harbor_trial(
        project, reference, trial_id="grant", code_revision="test", provider_limits=limits
    ) as trial:
        # Restoration starts a new recording episode; both owners, clocks and pending events
        # must otherwise reproduce exactly, including the predictor's saved history.
        assert trial.session.checkpoint().model_dump(exclude={"episode_id"}) == after.model_dump(
            exclude={"episode_id"}
        )
        assert trial.call(read) == reply
        with pytest.raises(RuntimeError, match="DispatchCapExhaustedError"):
            trial.call(read.model_copy(update={"call_id": "fresh-draw"}))
        assert trial.result()["outcome_unknown"] is False
        assert trial.result()["execution_failed"] is True
    assert len(transport.requests) == 1
    changed = limits.model_copy(update={"deadline": limits.deadline + timedelta(minutes=1)})
    with pytest.raises(ValueError, match="authority changed"):
        with open_executable_harbor_trial(
            project, reference, trial_id="grant", code_revision="test", provider_limits=changed
        ):
            raise AssertionError("changed authority must not enter runtime")
    grant = ProviderGrant(registration=registration, limits=limits)
    grant.require_configuration(project, definition.job.configuration)
    events = project.artifacts.list_ids(artifact_type="grounded-world-prediction")
    assert events and all(
        registration in project.artifacts.read(key).manifest.inputs for key in events
    )


def test_grant_refuses_model_configuration_or_implementation_changes_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real trial receipt cannot authorize a different local implementation or predictor."""
    project, reference, definition, transport = expired_task(tmp_path, monkeypatch)
    limits = ProviderLimits(
        maximum_physical_attempts=1, deadline=datetime.now(UTC) + timedelta(minutes=5)
    )
    with open_executable_harbor_trial(
        project, reference, trial_id="grant", code_revision="test", provider_limits=limits
    ) as trial:
        grant = ProviderGrant(registration=trial.registration, limits=limits)
    with pytest.raises(ValueError, match="frozen runtime configuration"):
        grant.require_configuration(project, {})
    bad = tmp_path / "must-not-import.py"
    bad.write_text('raise AssertionError("imported")')
    with pytest.raises(ValueError, match="runtime implementation"):
        load_runtime_definition(
            project,
            runtime_file=bad,
            runtime_sha256=sha256_bytes(bad.read_bytes()),
            configuration=definition.job.configuration,
            probe_sha256=definition.job.study.probe_sha256,
            provider_grant=grant,
        )
    with pytest.raises(ValueError, match="durable trial registration"):
        grant.model_copy(
            update={"limits": limits.model_copy(update={"maximum_physical_attempts": 2})}
        ).verify(project)
    assert not transport.requests


def test_distinct_authorized_trials_have_distinct_finite_ledgers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two explicitly authorized trials each have a separate one-attempt ceiling."""
    project, reference, _, transport = expired_task(tmp_path, monkeypatch)
    transports = [deepcopy(transport), deepcopy(transport)]
    pending = iter(transports)
    monkeypatch.setattr(grounded_runtime, "HttpxJsonTransport", lambda: next(pending))
    limits = ProviderLimits(
        maximum_physical_attempts=1, deadline=datetime.now(UTC) + timedelta(minutes=5)
    )
    starts = []
    for identity in ("first", "second"):
        with open_executable_harbor_trial(
            project, reference, trial_id=identity, code_revision="test", provider_limits=limits
        ) as trial:
            starts.append(trial.registration)
            trial.call(workbook_case().first)
            reply = trial.call(
                ToolCall(name="inspect", call_id="read", arguments={"key": "output.xlsx"})
            )
            assert reply["content"] == "predicted reply"
    assert starts[0] != starts[1]
    assert [len(item.requests) for item in transports] == [1, 1]


def test_fresh_grant_cannot_exceed_frozen_limit_or_start_expired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operational refresh cannot enlarge the configured predictor's admitted ceiling."""
    project, reference, _, transport = expired_task(tmp_path, monkeypatch)
    limits = ProviderLimits(
        maximum_physical_attempts=2, deadline=datetime.now(UTC) + timedelta(minutes=5)
    )
    with pytest.raises(ValueError, match="frozen simulator ceiling"):
        with open_executable_harbor_trial(
            project, reference, trial_id="oversized", code_revision="test", provider_limits=limits
        ):
            raise AssertionError("oversized authority must refuse")
    with pytest.raises(ValueError, match="has expired"):
        with open_executable_harbor_trial(
            project,
            reference,
            trial_id="expired",
            code_revision="test",
            provider_limits=limits.model_copy(
                update={"deadline": datetime.now(UTC) - timedelta(seconds=1)}
            ),
        ):
            raise AssertionError("expired authority must refuse")
    assert not transport.requests
