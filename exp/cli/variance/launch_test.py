"""Spec rules, consent matching, the revision and the reporting pipeline executors."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from exp.cli.variance import launch
from exp.common.progress import ProgressEvent
from exp.optimize.evaluation.variance.frontdoor import FrontDoorPlan
from exp.optimize.evaluation.variance.frontdoor_contracts import FrontDoorStage
from exp.optimize.evaluation.variance.frontdoor_states import StageState
from exp.optimize.evaluation.variance.pipeline_spec import CostLine

CLEF = {"build.decision_alias": "clef", "slots.tag-banks.decision_alias": "clef"}


def test_an_astra_alias_is_refused() -> None:
    """Every pipeline role is open-weight."""
    with pytest.raises(launch.SpecRefused, match="astra_alias 'astra'"):
        launch.check_aliases(CLEF, astra_alias="astra", text="{}")


def test_the_retired_jev_alias_is_refused() -> None:
    """A spec still naming jev predates the Clef rename."""
    with pytest.raises(launch.SpecRefused, match="jev"):
        launch.check_aliases(CLEF, astra_alias=None, text='{"alias": "jev"}')


def test_every_decision_must_be_clef() -> None:
    """A tag-banks or state-rag slot answered by another model is named in the refusal."""
    aliases = {**CLEF, "slots.state-rag.decision_alias": "glm"}
    with pytest.raises(launch.SpecRefused, match="slots.state-rag.decision_alias=glm"):
        launch.check_aliases(aliases, astra_alias=None, text="{}")
    launch.check_aliases(CLEF, astra_alias=None, text='{"decision_alias": "clef"}')


def test_consent_matches_the_whole_digest_or_at_least_its_first_twelve() -> None:
    """Short or foreign prefixes never consent."""
    digest = "0123456789abcdef" * 4
    assert launch.matches(digest, digest)
    assert launch.matches(digest[:12].upper(), digest)
    assert not launch.matches(digest[:11], digest)
    assert not launch.matches("f" * 12, digest)


def test_the_priced_bound_and_the_operation_follow_the_scripts() -> None:
    """Every stage's priced lines summed; the operation is front-door-cluster-<digest16>."""
    plan = FrontDoorPlan(
        spec_sha256="a" * 64,
        code_revision="r",
        stages=(
            FrontDoorStage(
                stage="slot tag-banks",
                status=StageState.PLANNED,
                costs=(CostLine(name="a", attempts=3, usd=1.25), CostLine(name="b", attempts=9)),
            ),
            FrontDoorStage(
                stage="slot curriculum",
                status=StageState.PLANNED,
                costs=(CostLine(name="c", attempts=1, usd=0.25),),
            ),
        ),
    )
    assert launch.priced_bound(plan) == 1.5
    assert launch.operation_name(plan.digest) == "front-door-cluster-" + plan.digest[:16]
    approval = launch.approval(plan, 7)
    assert approval.plan_sha256 == plan.digest and approval.unpriced_attempts == 7


def test_the_revision_is_head_and_modified_tracked_files_are_refused(tmp_path: Path) -> None:
    """The revision must name the code that runs."""

    def git(*args: str) -> str:
        """Run git in the scratch checkout."""
        return subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n")
    git("add", "a.py")
    git("commit", "-qm", "one")
    assert launch.source_revision(tmp_path) == git("rev-parse", "HEAD")
    (tmp_path / "untracked.py").write_text("")
    assert launch.source_revision(tmp_path) == git("rev-parse", "HEAD")
    (tmp_path / "a.py").write_text("x = 2\n")
    with pytest.raises(ValueError, match="uncommitted changes"):
        launch.source_revision(tmp_path)
    with pytest.raises(ValueError, match="not a git checkout"):
        launch.source_revision(tmp_path / "missing")


class _Inner:
    """A stage executor that records what it was asked."""

    def __init__(self) -> None:
        """Nothing asked yet."""
        self.asked: list[str] = []

    def keys(self, project: object, spec: object) -> dict[str, str]:
        """Return a key."""
        self.asked.append("keys")
        return {"k": "v"}

    def plan(self, project: object, spec: object, code_revision: str) -> tuple[tuple, tuple]:
        """Return an empty plan."""
        self.asked.append("plan")
        return (), ("line",)

    def run(self, context: object) -> str:
        """Return a result."""
        self.asked.append("run")
        return "ran"


def test_reporting_executors_delegate_and_report_start_and_finish() -> None:
    """Planning and identity delegate unchanged; running reports the stage around the owner."""
    events: list[ProgressEvent] = []
    inner = _Inner()
    wrapped = launch.ReportingExecutor(inner, "traceml", events.append)  # ty: ignore[invalid-argument-type]
    assert wrapped.keys(None, None) == {"k": "v"}  # ty: ignore[invalid-argument-type]
    assert wrapped.plan(None, None, "r") == ((), ("line",))  # ty: ignore[invalid-argument-type]
    context = SimpleNamespace(
        spec=SimpleNamespace(mining=SimpleNamespace(bootstrap=SimpleNamespace(task_id="task-1"))),
        identity="x",
    )
    assert wrapped.run(context) == "ran"  # ty: ignore[invalid-argument-type]
    assert inner.asked == ["keys", "plan", "run"]
    assert events == [
        ProgressEvent("pipeline task-1", completed=2, total=6, detail="traceml"),
        ProgressEvent("pipeline task-1", completed=3, total=6, detail="traceml"),
    ]
    stages = launch.reporting_executors(events.append)
    assert list(stages) == [
        "variance-build",
        "rules",
        "traceml",
        "error-mining",
        "families",
        "records",
    ]
