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
from exp.optimize.evaluation.executable.ledger import LedgerKind, trial_ledger
from exp.optimize.evaluation.executable.qualification import qualify_executable_task
from exp.optimize.evaluation.executable.qualification_test import qualified_fixture
from exp.optimize.evaluation.executable.task import load_executable_task, persist_executable_task
from exp.optimize.evaluation.executable.trial import (
    ExecutableHarborTrial,
    open_executable_harbor_trial,
)
from exp.optimize.evaluation.harbor.bridge import serve_harbor_trial
from exp.optimize.evaluation.variance.judged_test import episode_judge, judge_settings, route_judge
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


def test_ledger_records_dialogue_calls_and_submission_in_order_without_changing_reward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every actor operation through the real bridge lands in one sequenced, durable ledger."""
    project, task, spec = qualified_fixture(tmp_path, monkeypatch)
    qualification = qualify_executable_task(
        project, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    case = workbook_case()
    malformed = case.first.model_copy(update={"call_id": "malformed", "arguments": {}})
    with open_executable_harbor_trial(
        project, task, trial_id="ledger", code_revision="test", qualification=qualification
    ) as trial:
        with serve_harbor_trial(trial) as endpoint:
            actor = endpoint.actor_token
            assert _post(endpoint.url + "/message", actor, {"text": "Looking now."}) == {
                "recorded": True,
                "sequence": 1,
            }
            with pytest.raises(HTTPError) as refused:
                _post(endpoint.url + "/message", endpoint.verifier_token, {"text": "no"})
            assert refused.value.code == 403
            with pytest.raises(HTTPError) as empty:
                _post(endpoint.url + "/message", actor, {"text": "  "})
            assert empty.value.code == 400
            failed = _post(endpoint.url + "/call", actor, malformed.model_dump(mode="json"))
            assert failed["is_error"] is True
            done = _post(endpoint.url + "/call", actor, case.first.model_dump(mode="json"))
            assert done["is_error"] is False
            # A replayed call returns the saved reply and records nothing new.
            assert _post(endpoint.url + "/call", actor, case.first.model_dump(mode="json")) == done
            _post(endpoint.url + "/message", actor, {"text": "Row deleted."})
            _post(endpoint.url + "/submit", actor, {"answer": "done"})
            with pytest.raises(HTTPError) as closed:
                _post(endpoint.url + "/message", actor, {"text": "one more thing"})
            assert closed.value.code == 400
            result = _post(endpoint.url + "/grade", endpoint.verifier_token, {})
        assert result["status"] == "completed" and result["reward"] == 1
        identity = trial.identity
        assert trial.ledger() == trial_ledger(project, identity)
    ledger = trial_ledger(project, identity)
    assert [entry.kind for entry in ledger] == [
        "start",
        "message",
        "tool_call",
        "tool_reply",
        "tool_call",
        "tool_reply",
        "message",
        "submission",
    ]
    assert [entry.sequence for entry in ledger] == list(range(8))
    assert not any(entry.reconciled for entry in ledger)
    assert ledger[1].content == {"text": "Looking now."} and ledger[1].record is None
    assert ledger[3].record == identity + "-call-0"
    step = ledger[3].content["step"]
    assert isinstance(step, dict) and isinstance(step["reply"], dict)
    assert step["reply"]["is_error"] is True
    assert ledger[7].content["answer"] == {"answer": "done"}
    # Reopening neither duplicates entries nor changes the saved verdict.
    with open_executable_harbor_trial(
        project, task, trial_id="ledger", code_revision="test", qualification=qualification
    ) as resumed:
        assert resumed.ledger() == ledger
        assert resumed.result() == result


def test_ledger_reconciles_a_committed_reply_after_the_process_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reply saved before its ledger entry is appended on restart, marked reconciled."""
    project, task, _ = qualified_fixture(tmp_path, monkeypatch)
    append = ExecutableHarborTrial._append

    def stop_before_reply(
        self: ExecutableHarborTrial,
        kind: LedgerKind,
        content: JsonObject,
        record: str | None = None,
        *,
        reconciled: bool = False,
    ) -> int:
        """Emulate a crash after the call record and before its ledger entry."""
        if kind == "tool_reply":
            raise RuntimeError("process stopped")
        return append(self, kind, content, record, reconciled=reconciled)

    with open_executable_harbor_trial(
        project, task, trial_id="crash", code_revision="test"
    ) as trial:
        monkeypatch.setattr(ExecutableHarborTrial, "_append", stop_before_reply)
        with pytest.raises(RuntimeError, match="stopped"):
            trial.call(workbook_case().first)
        monkeypatch.setattr(ExecutableHarborTrial, "_append", append)
    with open_executable_harbor_trial(
        project, task, trial_id="crash", code_revision="test"
    ) as resumed:
        ledger = resumed.ledger()
        assert [(entry.kind, entry.reconciled) for entry in ledger] == [
            ("start", False),
            ("tool_call", False),
            ("tool_reply", True),
        ]
        assert resumed.call(workbook_case().first)["is_error"] is False
        assert resumed.ledger() == ledger


def test_a_judged_task_is_rewarded_only_by_its_frozen_judge_over_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same submitted episode under pass, fail and unassessable verdicts: 1, 0 and none."""
    project, task, spec = qualified_fixture(tmp_path, monkeypatch)
    judge = episode_judge(project, judge_settings(project))
    judged = load_executable_task(project, task).model_copy(update={"judge": judge})
    reference = persist_executable_task(
        project, judged, created_at=datetime.now(UTC), code_revision="test"
    )
    qualification = qualify_executable_task(
        project,
        spec.model_copy(update={"task": reference}),
        created_at=datetime.now(UTC),
        code_revision="test",
    )
    verdicts = iter(["pass", "fail", "unassessable"])
    router = route_judge(monkeypatch, verdict=lambda evidence: next(verdicts))
    rewards = []
    for trial_id in ("judged-pass", "judged-fail", "judged-unscored"):
        with open_executable_harbor_trial(
            project, reference, trial_id=trial_id, code_revision="test", qualification=qualification
        ) as trial:
            before = len(router.judged)
            assert trial.result()["reward"] is None and len(router.judged) == before
            trial.message("Deleting the row now.")
            trial.call(workbook_case().first)
            trial.submit({"answer": "done"})
            result = trial.result()
            binding = result["judge"]
            assert isinstance(binding, dict) and binding["definition"] == judge.identity
            assert "mechanical" not in result
            assert trial.result() == result
            rewards.append((result["status"], result["reward"]))
    assert rewards == [("completed", 1), ("completed", 0), ("incomplete", None)]
    assert len(router.judged) == 3
    events = router.judged[0]["events"]
    assert isinstance(events, list)
    kinds = [e["kind"] for e in events if isinstance(e, dict) and e["scope"] == "scored"]
    assert kinds[:2] == ["assistant_message", "tool_call"] and "final_answer" in kinds
