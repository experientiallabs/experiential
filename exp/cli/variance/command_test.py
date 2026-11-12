"""``exp variance run`` and ``watch``: wiring, the one consent by digest, detaching and attaching.

The front door itself is replaced by a fake plan and report: these tests cover what the command
does around it (preflight, plan view, consent, the spend ceiling, run records, the live view).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from click.testing import Result
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.variance import command, launch
from exp.cli.variance.fixtures_test import RECORDED, RUN, write_journal, write_ledger
from exp.cli.variance.runs import load_records
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.frontdoor import FrontDoorPlan, FrontDoorReport
from exp.optimize.evaluation.variance.frontdoor_contracts import FrontDoorStage
from exp.optimize.evaluation.variance.frontdoor_states import StageState
from exp.optimize.evaluation.variance.pipeline_spec import CostLine

PLAN = FrontDoorPlan(
    spec_sha256="a" * 64,
    code_revision="rev-1",
    stages=(
        FrontDoorStage(stage="project-build", status=StageState.DONE),
        FrontDoorStage(
            stage="slot tag-banks",
            status=StageState.PLANNED,
            costs=(CostLine(name="clef decisions", attempts=12, usd=0.5),),
            lines=("Author each family's tag bank.",),
        ),
        FrontDoorStage(stage="pipelines", status=StageState.BLOCKED),
    ),
)


@dataclass
class Fakes:
    """What the fake front door saw."""

    planned: int = 0
    runs: list[dict[str, Any]] = field(default_factory=list)
    ceilings: list[dict[str, Any]] = field(default_factory=list)
    spawned: list[list[str]] = field(default_factory=list)
    dependencies: list[dict[str, Any]] = field(default_factory=list)


@pytest.fixture
def fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fakes:
    """Replace the spec, the planner, the runner, the ceiling and the spawner."""
    seen = Fakes()
    spec = object()
    monkeypatch.setattr(launch, "load_spec", lambda path: spec)
    monkeypatch.setattr(
        launch, "check_spec", lambda value: {"slots.tag-banks.decision_alias": "clef"}
    )
    monkeypatch.setattr(launch, "slot_names", lambda value: ["tag-banks", "dense-hybrid"])
    monkeypatch.setattr(launch, "journal_name", lambda store, value: RUN)

    def dependencies(store: ProjectStore, value: object, revision: str, **kwargs: object) -> str:
        """Record how the dependencies were bound."""
        seen.dependencies.append({"revision": revision, **kwargs})
        return "dependencies"

    def plan(store: ProjectStore, value: object, **kwargs: object) -> FrontDoorPlan:
        """Return the fake plan."""
        assert value is spec and kwargs["dependencies"] == "dependencies"
        seen.planned += 1
        return PLAN

    def run(store: ProjectStore, value: object, **kwargs: object) -> FrontDoorReport:
        """Record the consented run."""
        seen.runs.append(kwargs)
        return FrontDoorReport(plan=PLAN)

    def ceiling(store: ProjectStore, value: FrontDoorPlan, **kwargs: object) -> tuple[str, Path]:
        """Record the installed ceiling."""
        seen.ceilings.append({"digest": value.digest, **kwargs})
        return launch.operation_name(value.digest), tmp_path / "receipt.json"

    def spawn(arguments: list[str], log: Path) -> int:
        """Record the detached command."""
        seen.spawned.append([*arguments, str(log)])
        return 424242

    monkeypatch.setattr(launch, "front_door_dependencies", dependencies)
    monkeypatch.setattr(launch, "install_ceiling", ceiling)
    monkeypatch.setattr(command, "plan_front_door", plan)
    monkeypatch.setattr(command, "run_front_door", run)
    monkeypatch.setattr(command, "spawn_detached", spawn)
    monkeypatch.setattr(command, "admission_path", lambda: None)
    monkeypatch.setattr(command, "STORE_EVERY", 0.0)
    return seen


def _invoke(tmp_path: Path, *arguments: str) -> Result:
    """Run ``exp variance run demo SPEC --root ROOT --code-revision rev-1 ARGUMENTS``."""
    spec = tmp_path / "front-door.json"
    spec.write_text("{}")
    return CliRunner().invoke(
        app,
        [
            "variance", "run", "demo", str(spec), "--root", str(tmp_path / "root"),
            "--code-revision", "rev-1", *arguments,
        ],
    )  # fmt: skip


def test_the_dry_run_prints_preflight_plan_and_digest_and_never_runs(
    tmp_path: Path, fakes: Fakes
) -> None:
    """Plan only: the preflight, every stage, the digest and the exact command to run it."""
    result = _invoke(tmp_path, "--dry-run")
    assert result.exit_code == 0, result.output
    output = result.output
    assert "Front-door preflight" in output and "open-weight only" in output
    assert "2. slot tag-banks: up to $0.500000" in output
    assert "3. pipelines: waiting on an earlier stage" in output
    assert f"Plan digest {PLAN.digest}" in output
    assert "State machines" not in output
    assert f"--consent {PLAN.digest} --non-interactive" in output.replace("\n", " ")
    assert fakes.planned == 1 and fakes.runs == [] and fakes.ceilings == []
    assert fakes.dependencies[0]["revision"] == "rev-1"
    assert load_records(ProjectStore(tmp_path / "root", "demo")) == []


def test_consent_by_digest_installs_the_ceiling_runs_and_records(
    tmp_path: Path, fakes: Fakes
) -> None:
    """``--consent`` with the digest's first 12 runs once under the scripts' ceiling."""
    result = _invoke(tmp_path, "--consent", PLAN.digest[:12], "--ceiling-usd", "10")
    assert result.exit_code == 0, result.output
    assert f"Consented to plan {PLAN.digest}" in result.output
    assert fakes.ceilings == [
        {
            "digest": PLAN.digest,
            "ceiling_usd": 10.0,
            "real_attempts": 1_000_000,
            "command": "exp " + " ".join(_options(tmp_path).rerun(PLAN.digest)),
        }
    ]
    (run,) = fakes.runs
    assert run["approval"].plan_sha256 == PLAN.digest
    assert run["approval"].unpriced_attempts == 2_000_000_000
    assert run["approval"].ceiling_usd == 0.5
    assert run["dependencies"] == "dependencies" and run["code_revision"] == "rev-1"
    (record,) = load_records(ProjectStore(tmp_path / "root", "demo"))
    assert record.digest == PLAN.digest and record.journal == RUN
    assert record.state == "finished" and record.operation == launch.operation_name(PLAN.digest)
    assert "Complete" in result.output


def test_a_consent_naming_another_plan_is_refused_before_any_run(
    tmp_path: Path, fakes: Fakes
) -> None:
    """A stale digest means the store or code moved: nothing runs."""
    result = _invoke(tmp_path, "--consent", "b" * 64)
    assert result.exit_code != 0
    assert "does not name this plan" in " ".join(result.output.replace("│", " ").split())
    assert fakes.runs == [] and fakes.ceilings == []


def test_non_interactive_without_consent_stops_at_the_question(
    tmp_path: Path, fakes: Fakes
) -> None:
    """No terminal and neither --yes nor --consent: the command exits before the run."""
    result = _invoke(tmp_path, "--non-interactive")
    assert result.exit_code != 0
    assert fakes.runs == [] and fakes.ceilings == []


def test_yes_consents_to_the_plan_shown(tmp_path: Path, fakes: Fakes) -> None:
    """``--yes`` is the explicit confirmation of the displayed digest."""
    result = _invoke(tmp_path, "--yes", "--non-interactive")
    assert result.exit_code == 0, result.output
    assert f"Consent covers exactly plan {PLAN.digest[:12]}" in result.output
    assert len(fakes.runs) == 1


def test_a_declined_question_runs_nothing(
    tmp_path: Path, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One question; answering no leaves the store untouched."""
    asked: list[dict[str, Any]] = []

    def decline(console: object, **kwargs: object) -> bool:
        """Record the question and decline."""
        asked.append(kwargs)
        return False

    monkeypatch.setattr(command, "require_spend_consent", decline)
    result = _invoke(tmp_path)
    assert result.exit_code == 0, result.output
    assert "Not consented; nothing ran." in result.output
    assert len(asked) == 1 and asked[0]["cost_is_upper_bound"] is False
    assert f"--consent {PLAN.digest}" in asked[0]["command"]
    assert fakes.runs == []


def test_detach_starts_the_exact_consented_command_and_records_it(
    tmp_path: Path, fakes: Fakes
) -> None:
    """The child gets --consent DIGEST and the revision; the record names its PID and log."""
    result = _invoke(tmp_path, "--consent", PLAN.digest, "--detach", "--no-watch")
    assert result.exit_code == 0, result.output
    (spawned,) = fakes.spawned
    assert spawned[:3] == ["variance", "run", "demo"]
    assert spawned[spawned.index("--consent") + 1] == PLAN.digest
    assert spawned[spawned.index("--code-revision") + 1] == "rev-1"
    assert "--non-interactive" in spawned and "--log" in spawned
    assert fakes.runs == []
    (record,) = load_records(ProjectStore(tmp_path / "root", "demo"))
    assert record.pid == 424242 and record.log is not None and record.state == "running"
    assert f"exp variance watch demo {PLAN.digest[:12]}" in result.output


def test_watch_once_reopens_a_recorded_run_by_digest(tmp_path: Path, fakes: Fakes) -> None:
    """A run found by its ledger digest: phases, the running start and calls, as plain lines."""
    runtime = tmp_path / "root" / "projects" / "demo" / "runtime"
    write_journal(runtime / "front-door" / RUN / "transitions.jsonl", RECORDED)
    digest = "0fd3f45f5562beba"
    write_ledger(
        runtime / "spend" / f"front-door-cluster-{digest}" / "ledger.sqlite",
        [("clef", "settled", 2), ("actor-gemma4", "dispatched", 62)],
    )
    result = CliRunner().invoke(
        app,
        ["variance", "watch", "demo", digest[:12], "--root", str(tmp_path / "root"), "--once"],
    )
    assert result.exit_code == 0, result.output
    output = result.output
    assert f"front door {digest[:12]}" in output
    assert "✓ slot tag-banks  done  30m00s" in output
    assert "pipeline task-307a285239d7acd18b5d" in output
    assert "tier gemma4: 1 calls, 0 done, 1 in flight" in output
    assert "2 / 1,000 real attempts" in output


def test_watch_without_a_run_names_the_missing_run(tmp_path: Path, fakes: Fakes) -> None:
    """Nothing to watch is a usage error, not a traceback."""
    result = CliRunner().invoke(
        app, ["variance", "watch", "demo", "--root", str(tmp_path / "root"), "--once"]
    )
    assert result.exit_code != 0
    assert "no front-door run" in result.output


def _options(tmp_path: Path) -> command.RunOptions:
    """The options the consent test invoked."""
    return command.RunOptions(
        project="demo",
        spec=tmp_path / "front-door.json",
        root=tmp_path / "root",
        code_revision="rev-1",
        ceiling_usd=10.0,
    )


def test_records_round_trip(tmp_path: Path) -> None:
    """A saved record reads back newest first."""
    from exp.cli.variance.runs import RunRecord, save_record

    store = ProjectStore(tmp_path, "demo")
    path = save_record(store, RunRecord("c" * 64, RUN, "op", "s.json", "rev", 1, started="t1"))
    assert json.loads(path.read_text())["journal"] == RUN
    assert load_records(store)[0].digest == "c" * 64
