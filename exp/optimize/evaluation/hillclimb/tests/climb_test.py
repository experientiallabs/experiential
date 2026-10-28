"""Climbs on native workbook and record worlds with a scripted, prompt-sensitive actor.

Each world gets four independent lineages, one authored case each, captured at the actor's
initial turn. The scripted actor passes every episode when its system prompt carries the helpful
cue, passes only for its first four episodes per case under the fragile cue, and otherwise
alternates pass and fail per case, so the baseline sits at one half with a zero A/A floor.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    sha256_bytes,
    sha256_json,
)
from exp.common.models import AssistantAction, ToolCall, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.audit.panel_test import write_catalog
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.study import ContinuationStudySpec
from exp.optimize.evaluation.harbor.tests.package_trial_test import RECORD_RUNTIME
from exp.optimize.evaluation.hillclimb.design import ClimbSpec, climb_split
from exp.optimize.evaluation.hillclimb.history import load_history
from exp.optimize.evaluation.hillclimb.prompts import PromptEdit, load_prompt_version
from exp.optimize.evaluation.hillclimb.report import ClimbReportSpec, run_climb_report
from exp.optimize.evaluation.hillclimb.rounds import ClimbRound, run_climb_round
from exp.runtime.agents.chat import ChatAgentRuntime, ChatCheckpoint
from exp.runtime.agents.chat_test import _Model, _response
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    RecordedRequest,
    ScriptedJsonTransport,
)
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.continuations import ContinuationCheckpoint, persist_continuation_checkpoint
from exp.simulation.evaluation.record_execution_test import fixture, source
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier

HELPFUL = "Before editing, re-read the request and change exactly what it asks."
FRAGILE = "Work quickly and trust your first reading."
WORKBOOK_RUNTIME = """from exp.optimize.evaluation.executable.loader import ContinuationRuntime
from exp.optimize.evaluation.continuations.study_test import probe, workbook_runtime
from exp.simulation.continuations import load_continuation_checkpoint
from exp.common.core.artifacts import ArtifactInput
def continuation_runtime(project, configuration):
    reference = ArtifactInput.model_validate(configuration["checkpoint"])
    checkpoint = load_continuation_checkpoint(project.artifacts, reference)
    return ContinuationRuntime(workbook_runtime(checkpoint.world), probe, configuration["probe"])
"""


def _workbook_call(count: int) -> ToolCall:
    """Delete ``count`` data rows; one row passes the workbook verifier."""
    call = workbook_case().first
    return call.model_copy(
        update={
            "arguments": {
                **call.arguments,
                "deletions": [{"sheet": "Orders", "start": 2, "count": count}],
            }
        }
    )


@dataclass(frozen=True)
class WorldKind:
    """One native world: its passing and failing actor calls.

    Attributes:
        name: World label.
        good: Call that passes the case verifier.
        bad: Call that misses the goal.
    """

    name: str
    good: ToolCall
    bad: ToolCall


KINDS = {
    "workbook": WorldKind("workbook", _workbook_call(1), _workbook_call(2)),
    "records": WorldKind(
        "records",
        ToolCall(call_id="move", name="change", arguments={"id": "a", "address": "new"}),
        ToolCall(call_id="move", name="change", arguments={"id": "a", "address": "partial"}),
    ),
}


class PromptActor(ScriptedJsonTransport):
    """A scripted actor whose first call depends on its system prompt and per-case history.

    Attributes:
        kind: World whose calls it makes.
        episodes: Episodes opened so far per (case, system prompt).
    """

    def __init__(self, kind: WorldKind) -> None:
        """Answer for one world kind."""
        super().__init__()
        self.kind = kind
        self.episodes: dict[tuple[str, str], int] = {}

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Open an episode with a pass or fail call, then finish with ``done``."""
        del timeout_seconds
        self.requests.append(RecordedRequest(url, dict(headers), payload))
        messages = payload["messages"]
        assert isinstance(messages, list)
        rows = [row for row in messages if isinstance(row, dict)]
        system = next((str(row["content"]) for row in rows if row["role"] == "system"), "")
        case = next(str(row["content"]) for row in rows if row["role"] == "user")
        message: JsonObject = {"role": "assistant", "content": "done"}
        if not any(row["role"] == "assistant" for row in rows):
            episode = self.episodes.get((case, system), 0)
            self.episodes[(case, system)] = episode + 1
            good = (
                HELPFUL in system
                or (FRAGILE in system and episode < 4)
                or (FRAGILE not in system and episode % 2 == 0)
            )
            call = self.kind.good if good else self.kind.bad
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": call.call_id,
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                ],
            }
        return JsonHttpResponse(
            status_code=200,
            body={
                "id": f"actor-{len(self.requests)}",
                "model": "world-v1",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
                        "message": message,
                    }
                ],
            },
        )


def _capture(
    project: ProjectStore, kind: str, index: int, system_prompt: str | None
) -> tuple[ArtifactInput, TaskVerifier, str]:
    """Capture one lineage's case at the actor's initial turn; return its verifier and probe."""
    if kind == "workbook":
        world_case = workbook_case()
        task, runtime = world_case.task, world_case.runtime
        verifier = TaskVerifier(
            description="Delete one data row and preserve the input.",
            outcome_schema={},
            required_paths=("/final_state/rows",),
            rules=(OutcomeRule(path="/final_state/rows", operator="equals", value=6),),
            allowed_tools=("edit_workbook",),
            preserved_paths=("/input_digest",),
        )
        probe_sha256 = sha256_json({"probe": "xlsx-row-count-v1"})
        if not project.artifacts.exists("authored-source"):
            project.artifacts.write_json(
                artifact_id="authored-source",
                artifact_type="authored-task",
                envelope=ArtifactEnvelope(
                    schema_version=1, created_at=datetime.now(UTC), code_revision="test"
                ),
                files={"task.json": {"origin": "authored climb fixture, not benchmark evidence"}},
            )
        origin = artifact_input(project.artifacts.read("authored-source").manifest)
    else:
        replay, task = fixture()
        runtime = replay.world
        verifier = TaskVerifier(
            description="Move order a to the new address without collateral effects.",
            outcome_schema={},
            rules=(
                OutcomeRule(
                    path="/final_state/tables/orders/rows/0/1", operator="equals", value="new"
                ),
            ),
            allowed_tools=("change",),
            preserved_paths=("/tables/orders/rows/1", "/tables/outbox"),
        )
        probe_sha256 = replay.probe_sha256
        origin = source(project)
    task = task.model_copy(
        update={
            "task_id": f"{kind}-case-{index}",
            "lineage_group_id": f"{kind}-lineage-{index}",
            "instruction": f"{task.instruction} Case {index}.",
        }
    )
    captured: list[ContinuationCheckpoint] = []
    with runtime.open(task) as session:

        def capture(checkpoint: ChatCheckpoint) -> None:
            """Pair the initial actor context with the exact native world image."""
            captured.append(
                ContinuationCheckpoint(
                    task=task, actor=checkpoint, world=session.checkpoint(), sources=(origin,)
                )
            )

        ChatAgentRuntime(maximum_model_calls=4, system_prompt=system_prompt).run(
            task,
            model=_Model((_response(AssistantAction(content="stop")),)),
            environment=session,
            on_checkpoint=capture,
        )
    reference = persist_continuation_checkpoint(
        project.artifacts, captured[0], created_at=datetime.now(UTC), code_revision="test"
    )
    return reference, verifier, probe_sha256


def climb_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    *,
    cases: int = 4,
    system_prompt: str | None = None,
) -> tuple[ProjectStore, ClimbSpec, PromptActor]:
    """Configure ``cases`` independent lineages of one native world and the scripted actor."""
    project = ProjectStore(tmp_path, kind)
    write_catalog(tmp_path)
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    catalog = load_model_catalog(project.model_catalog_path)
    model = RuntimeModelCatalog(catalog).resolve("world")
    runtime = tmp_path / f"{kind}_runtime.py"
    runtime.write_text(WORKBOOK_RUNTIME if kind == "workbook" else RECORD_RUNTIME)
    jobs = []
    for index in range(cases):
        checkpoint, verifier, probe_sha256 = _capture(project, kind, index, system_prompt)
        study = ContinuationStudySpec(
            checkpoint=checkpoint,
            actor_model=model.snapshot,
            verifier=verifier,
            probe_sha256=probe_sha256,
            repeats=1,
            selector="authored climb case",
        )
        jobs.append(
            ContinuationJob(
                study=study,
                runtime_file=runtime,
                runtime_sha256=sha256_bytes(runtime.read_bytes()),
                configuration={
                    "checkpoint": checkpoint.model_dump(mode="json"),
                    "probe": probe_sha256,
                },
                actor_alias="world",
                catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
                maximum_model_calls=4,
                system_prompt=system_prompt,
                maximum_physical_attempts=40,
                deadline=datetime.now(UTC) + timedelta(hours=1),
            )
        )
    actor = PromptActor(KINDS[kind])
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: actor
    )
    return project, ClimbSpec(cases=tuple(jobs), minimum_lineages=2, final_repeats=4), actor


def deadline() -> datetime:
    """A round deadline an hour away."""
    return datetime.now(UTC) + timedelta(hours=1)


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_a_climb_keeps_a_real_gain_reverts_noise_and_reports_the_final_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """A noise edit stops at screening; a helpful edit is kept; the report confirms it fresh."""
    project, spec, actor = climb_fixture(tmp_path, monkeypatch, kind)
    split = climb_split(project, spec)
    assert len(split.train) == len(split.test) == 2
    assert not set(split.train) & set(split.test)
    noise = PromptEdit(kind="append", text="Be concise.", change_note="Shorter replies.")
    first = run_climb_round(
        project,
        ClimbRound(climb=spec, round=1, edit=noise, deadline=deadline()),
        code_revision="test",
    )
    assert not first.kept and first.confirm is None
    assert first.screen.gain.estimate == 0 and first.screen.noise_floor == 0
    assert first.screen.incumbent_pass_rate == 0.5
    assert first.reason.startswith("screen: train gain")
    test_cases = {f"Case {index}." for index in split.test}
    assert not any(any(name in case for name in test_cases) for case, _ in actor.episodes)
    helpful = PromptEdit(kind="append", text=HELPFUL, change_note="Re-read before editing.")
    second = run_climb_round(
        project,
        ClimbRound(climb=spec, round=2, edit=helpful, deadline=deadline()),
        code_revision="test",
    )
    assert second.kept and second.confirm is not None
    assert second.confirm.gain.estimate == 0.5 and second.confirm.gain.interval == (0.5, 0.5)
    assert second.confirm.unknown == 0 and second.confirm.noise_floor == 0
    assert load_prompt_version(project.artifacts, second.candidate).system_prompt == HELPFUL
    calls = len(actor.requests)
    assert (
        run_climb_round(
            project,
            ClimbRound(climb=spec, round=2, edit=helpful, deadline=deadline()),
            code_revision="test",
        )
        == second
    )
    assert len(actor.requests) == calls
    with pytest.raises(ValueError, match="registered for another edit"):
        run_climb_round(
            project,
            ClimbRound(climb=spec, round=2, edit=noise, deadline=deadline()),
            code_revision="test",
        )
    with pytest.raises(ValueError, match="already measured in round 1"):
        run_climb_round(
            project,
            ClimbRound(
                climb=spec,
                round=3,
                edit=PromptEdit(kind="replace", text="Be concise.", change_note="again"),
                deadline=deadline(),
            ),
            code_revision="test",
        )
    report = run_climb_report(
        project, ClimbReportSpec(climb=spec, rounds=2, deadline=deadline()), code_revision="test"
    )
    assert report.verdict == "merge" and report.best == second.candidate
    assert [row.kept for row in report.rounds] == [False, True]
    assert report.final is not None and report.final.gain.estimate == 0.5
    empty = run_climb_report(
        project, ClimbReportSpec(climb=spec, rounds=1, deadline=deadline()), code_revision="test"
    )
    assert empty.verdict == "no edit kept" and empty.final is None
    assert [row.round for row in load_history(project, spec, 2)] == [1, 2]


def test_a_gain_that_fades_on_fresh_test_repeats_is_within_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirmation keeps the fragile edit, but the final fresh comparison refuses to merge."""
    project, spec, _ = climb_fixture(tmp_path, monkeypatch, "records")
    kept = run_climb_round(
        project,
        ClimbRound(
            climb=spec,
            round=1,
            edit=PromptEdit(kind="append", text=FRAGILE, change_note="Faster first reading."),
            deadline=deadline(),
        ),
        code_revision="test",
    )
    assert kept.kept
    report = run_climb_report(
        project, ClimbReportSpec(climb=spec, rounds=1, deadline=deadline()), code_revision="test"
    )
    assert report.verdict == "within noise, do not merge"
    assert report.final is not None and report.final.gain.estimate == -0.5


def test_a_saturated_train_set_is_refused_before_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A baseline that already passes every train case leaves nothing to climb."""
    project, spec, actor = climb_fixture(tmp_path, monkeypatch, "workbook", system_prompt=HELPFUL)
    with pytest.raises(ValueError, match="saturated"):
        run_climb_round(
            project,
            ClimbRound(
                climb=spec,
                round=1,
                edit=PromptEdit(kind="append", text="Be concise.", change_note="Shorter."),
                deadline=deadline(),
            ),
            code_revision="test",
        )
    test_cases = {f"Case {index}." for index in climb_split(project, spec).test}
    assert not any(any(name in case for name in test_cases) for case, _ in actor.episodes)


def test_too_few_lineages_are_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three lineages cannot give two to each side."""
    project, spec, actor = climb_fixture(tmp_path, monkeypatch, "records", cases=3)
    with pytest.raises(ValueError, match="underpowered: 1 train and 2 test lineages"):
        run_climb_round(
            project,
            ClimbRound(
                climb=spec,
                round=1,
                edit=PromptEdit(kind="append", text=HELPFUL, change_note="Re-read."),
                deadline=deadline(),
            ),
            code_revision="test",
        )
    assert not actor.requests
