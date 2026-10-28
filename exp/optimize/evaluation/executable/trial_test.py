"""Real workbook Harbor calls prove durable restoration, private grading and no unknown reroll."""

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.optimize.evaluation.executable.qualification import qualify_executable_task
from exp.optimize.evaluation.executable.qualification_test import qualified_fixture
from exp.optimize.evaluation.executable.trial import open_executable_harbor_trial
from exp.optimize.evaluation.harbor.bridge import serve_harbor_trial
from exp.runtime.environments.interface import Observation
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.environments.workbook_edits import read_supported_workbook


def _post(url: str, token: str, body: JsonObject) -> JsonObject:
    """Send a bounded local bridge request with a runtime-only capability."""
    request = Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def test_harbor_executes_restores_and_qualifies_real_workbook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual artifact changes survive owner restart; repeat call IDs do not apply edits twice."""
    project, task, spec = qualified_fixture(tmp_path, monkeypatch)
    qualification = qualify_executable_task(
        project, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    action = workbook_case().first
    with open_executable_harbor_trial(
        project, task, trial_id="one", code_revision="test", qualification=qualification
    ) as trial:
        reply = trial.call(action)
        assert trial.call(action) == reply
        with pytest.raises(ValueError, match="different arguments"):
            trial.call(action.model_copy(update={"arguments": {}}))
    with open_executable_harbor_trial(
        project, task, trial_id="one", code_revision="test", qualification=qualification
    ) as trial:
        assert trial.call(action) == reply
        assert trial.runtime.probe(trial.session.checkpoint())["rows"] == 6
        with serve_harbor_trial(trial) as endpoint:
            with pytest.raises(HTTPError) as refused:
                _post(endpoint.url + "/grade", endpoint.actor_token, {})
            assert refused.value.code == 403
            assert (
                _post(endpoint.url + "/submit", endpoint.actor_token, {"answer": "done"})[
                    "submitted"
                ]
                is True
            )
            result = _post(endpoint.url + "/grade", endpoint.verifier_token, {})
            assert result["status"] == "completed" and result["reward"] == 1
            files = _post(endpoint.url + "/outputs", endpoint.actor_token, {})["files"]
            assert isinstance(files, list) and len(files) == 1
            row = files[0]
            assert isinstance(row, dict) and row["path"] == "output.xlsx"
            content = row["content_base64"]
            assert isinstance(content, str)
            assert read_supported_workbook(base64.b64decode(content))["Orders"].max_row == 6
        with pytest.raises(ValueError, match="closed"):
            trial.call(workbook_case().second)


def test_unknown_native_call_closes_trial_instead_of_redrawing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lost reply after the edit cannot replay the tool or emit a successful reward."""
    project, task, _ = qualified_fixture(tmp_path, monkeypatch)
    with open_executable_harbor_trial(
        project, task, trial_id="unknown", code_revision="test"
    ) as trial:
        execute = trial.session.execute

        def lost_response(action: ToolCall) -> Observation:
            """Apply the real edit, then emulate a crash before response persistence."""
            execute(action)
            raise RuntimeError("response lost")

        monkeypatch.setattr(trial.session, "execute", lost_response)
        with pytest.raises(RuntimeError, match="lost"):
            trial.call(workbook_case().first)
        assert trial.result()["outcome_unknown"] is True
    with open_executable_harbor_trial(
        project, task, trial_id="unknown", code_revision="test"
    ) as resumed:
        with pytest.raises(ValueError, match="unknown"):
            resumed.call(workbook_case().first)
        assert resumed.result()["reward"] is None
        assert resumed.result()["outcome_unknown"] is True


def test_changed_execution_revision_refuses_before_loading_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task qualified under one implementation cannot execute under another as the same task."""
    project, task, _ = qualified_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="execution revision"):
        with open_executable_harbor_trial(
            project, task, trial_id="changed-code", code_revision="different-implementation"
        ):
            pass


def test_submission_requires_declared_outputs_and_actor_can_correct_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing deliverable is caught before closing the trial or acknowledging submission."""
    project, task, _ = qualified_fixture(tmp_path, monkeypatch)
    with open_executable_harbor_trial(
        project, task, trial_id="missing-output", code_revision="test"
    ) as trial:
        with pytest.raises(ValueError, match="output.*missing"):
            trial.submit({"answer": "done"})
        assert trial.result()["reward"] is None
        trial.call(workbook_case().first)
        assert trial.submit({"answer": "done"}) == {"submitted": True}
        assert trial.output_files()["files"]


@pytest.mark.parametrize("bad_arguments", [False, True])
def test_harbor_access_refusal_replays_and_actor_can_correct_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_arguments: bool
) -> None:
    """The exported execution path shares continuation recovery without repeating the refusal."""
    project, task, _ = qualified_fixture(tmp_path, monkeypatch)
    case = workbook_case()
    rejected = (
        case.first.model_copy(update={"call_id": "malformed", "arguments": {}})
        if bad_arguments
        else case.outside_policy
    )
    with open_executable_harbor_trial(
        project, task, trial_id="refusal", code_revision="test"
    ) as trial:
        refused = trial.call(rejected)
        assert refused["is_error"] is True
        assert trial.call(rejected) == refused
    with open_executable_harbor_trial(
        project, task, trial_id="refusal", code_revision="test"
    ) as trial:
        assert trial.call(rejected) == refused
        assert trial.call(case.first)["is_error"] is False
        trial.submit({"answer": "done"})
        result = trial.result()
        assert result["outcome_unknown"] is False
        assert result["reward"] is None  # Fresh diagnostic run has no qualified reward authority.
    records = [
        json.loads(project.artifacts.read_bytes(key, "record.json"))
        for key in project.artifacts.list_ids(artifact_type="executable-trial")
        if key.endswith("-call-0")
    ]
    assert len(records) == 1
    assert records[0]["step"]["execution"]["effects"] == "not_started"


def test_proven_rollback_stops_trial_without_claiming_ambiguous_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kernel rollback is an operational failure, retained across restart without a redraw."""
    from exp.runtime.environments.worlds import ExecutableWorld, WorldContext

    project, task, _ = qualified_fixture(tmp_path, monkeypatch)
    with open_executable_harbor_trial(
        project, task, trial_id="rollback", code_revision="test"
    ) as trial:
        assert isinstance(trial.session, ExecutableWorld)
        backend = trial.session._backend
        execute = backend.execute
        before = trial.session.checkpoint()

        def rollback(action: ToolCall, context: WorldContext) -> Observation:
            """Fail after actual workbook mutation so the kernel must restore native bytes."""
            execute(action, context)
            raise RuntimeError("native response validation failed")

        monkeypatch.setattr(backend, "execute", rollback)
        with pytest.raises(RuntimeError, match="validation failed"):
            trial.call(workbook_case().first)
        assert trial.session.checkpoint() == before
        result = trial.result()
        assert result["execution_failed"] is True
        assert result["outcome_unknown"] is False
        assert result["reward"] is None
    with open_executable_harbor_trial(
        project, task, trial_id="rollback", code_revision="test"
    ) as resumed:
        assert resumed.result() == result
        with pytest.raises(ValueError, match="closed"):
            resumed.call(workbook_case().first)
