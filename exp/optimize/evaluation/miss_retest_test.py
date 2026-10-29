"""Miss retest: the build's last step replays the traces it got wrong on its final memory."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.optimize.evaluation.held_out_fidelity_test import (
    PROJECT,
    REVISION,
    USERS,
    SimulatedProvider,
    _import_id,
    build_project,
    build_report,
    episode,
)
from exp.optimize.evaluation.miss_retest import register_miss_retest
from exp.optimize.evaluation.trace_learning import (
    learn_from_traces,
    learning_cases,
    learning_curve,
    learning_plan,
    recorded_plan,
)
from exp.optimize.evaluation.trace_learning_test import memory_delivered

STEPS = 3  # replayed exchanges per fixture trace, after its one-call public prefix


def rebuild(root: Path, store_import: str, *flags: str) -> tuple[int, str]:
    """Run ``exp build`` again on the stored import and return its exit code and output."""
    result = CliRunner().invoke(
        app,
        ["build", PROJECT, "--import-id", store_import, "--root", str(root), "--non-interactive"]
        + list(flags),
    )
    return result.exit_code, " ".join(result.output.split())


def test_the_build_retests_its_miss_on_the_final_memory_and_replays_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the first trace missed (the learnable tool); the final memory reproduces it."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    root, store, output = build_project(tmp_path, monkeypatch, provider, learn=True)
    own = learning_cases(store, "all")[0].task.lineage_group_id  # the trace that missed
    # The terminal gets one result line; the report file keeps the full retest.
    assert (
        "retested 1 trace that had misses · 100% replies exact (3/3) · fixed 1 of 1 missed "
        "turns · 0 regressed"
    ) in output
    report = build_report(store)
    assert f"1 of {USERS} learned traces had a miss · {STEPS} turns" in report
    assert "Miss retest 3/3 replies exact (100%)" in report
    assert "exact before learning 2/3 (67%) → after 3/3 (100%)" in report
    assert "Miss retest" not in output
    # Learning hid each trace's own task; the retest reads what memory learned from it.
    retested = provider.predictions()[STEPS * USERS :]
    assert len(retested) == STEPS
    assert all(own in memory_delivered(prediction) for prediction in retested)

    purchased = len(provider.requests)
    code, replay = rebuild(root, _import_id(store), "--yes")
    assert code == 0, replay
    assert "no new model calls (saved results)" in replay
    report = build_report(store)
    assert report.count("Saved results. No provider calls.") >= 2, "learning and retest replay"
    assert len(provider.requests) == purchased


def test_the_build_skips_learning_without_a_recall_model_and_retest_without_a_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No decision model: a bare simulator and a clear note. No miss: no retest."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, store, output = build_project(tmp_path, monkeypatch, provider)
    assert "built without learning" in output and "retested" not in output
    assert not recorded_plan(store, "all")

    other = tmp_path / "learned"
    other.mkdir()
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, _, output = build_project(other, monkeypatch, provider, learn=True)
    assert f"learned {USERS}/{USERS} traces" in output
    assert "retested" not in output, "every reply was exact, so there is nothing to retest"


@pytest.mark.parametrize(
    "flags",
    [["--models", "a"], ["--fidelity", "spec.json"], ["--resume", "run"]],
)
def test_audit_cannot_combine_with_other_modes(tmp_path: Path, flags: list[str]) -> None:
    """One evaluation mode per command."""
    result = CliRunner().invoke(
        app,
        ["eval", PROJECT, "--root", str(tmp_path / ".exp"), "--audit", "3", *flags],
    )
    assert result.exit_code == 2 and "cannot combine" in " ".join(result.output.split())


@pytest.mark.parametrize("flags", [["--retest-misses", "5"], ["--held-out-fidelity", "3"]])
def test_removed_eval_flags_are_rejected(tmp_path: Path, flags: list[str]) -> None:
    """The retest belongs to the build and the held-out panel to --audit."""
    result = CliRunner().invoke(app, ["eval", PROJECT, "--root", str(tmp_path), *flags])
    assert result.exit_code == 2 and "No such option" in result.output


def test_a_cancelled_preflight_never_hides_learned_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declining the build's learning records nothing; the audit's fit memory stays active."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    root, store, _ = build_project(tmp_path, monkeypatch, provider)
    code, declined = rebuild(root, _import_id(store))
    assert "learned" not in declined, declined
    assert not recorded_plan(store, "all"), "a preflight records no plan"
    outcome = learn_from_traces(
        store,
        recall_alias="jev",
        scope="fit",
        maximum_attempts=10_000,
        code_revision=REVISION,
        clock=lambda: datetime.now(UTC),
    )
    assert learning_plan(store).scope == "fit"
    curve = learning_curve(store)
    assert curve is not None and len(curve.points) == outcome.learned
    with pytest.raises(ValueError, match="no learned memory for this build"):
        register_miss_retest(store, 99, code_revision=REVISION, clock=lambda: datetime.now(UTC))


def test_the_build_retest_reads_the_build_memory_after_an_audit_learned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An audit's fit-only memory publishing later never becomes what the build retests."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    _, store, _ = build_project(tmp_path, monkeypatch, provider, learn=True)
    build_plan = learning_plan(store, scope="all")
    learn_from_traces(
        store,
        recall_alias="jev",
        scope="fit",
        maximum_attempts=10_000,
        code_revision=REVISION,
        clock=lambda: datetime.now(UTC),
    )
    assert learning_plan(store).scope == "fit", "the audit's memory published most recently"
    panel = register_miss_retest(store, 99, code_revision=REVISION, clock=lambda: datetime.now(UTC))
    assert panel.world_model == build_plan.world_model
    assert panel.learned == USERS, "every trace, not only the fit traces"
