"""Staged model optimizer budget and coverage tests."""

# ruff: noqa: F403, F405
from wmo.cli.optimize_model_cmd_fixtures_test import *


def test_the_sweep_persists_the_world_models_own_spend_beside_the_build_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The eval-infrastructure half of a sweep's bill has to land somewhere accountable.

    `route sweep` has always said the simulator's cost is "metered separately" while nothing
    persisted it: the world model opens one metered session per episode and every record died
    with its env. They are rolled into one `kind="sweep"` record in the model's own runs dir,
    beside the build and serve records already there.
    """
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9}, session_usd=0.02)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes")
    assert result.exit_code == 0, result.output

    runs = load_runs(root / "models" / "support" / "runs")
    sweeps = [record for record in runs if record.kind == "sweep"]
    assert len(sweeps) == 1
    swept = sweeps[0]
    # 6 episodes x $0.02, kept split by phase so the serve half and the judge half stay separable.
    assert swept.total.cost_usd == pytest.approx(0.12)
    assert swept.by_phase[Phase.SERVE].cost_usd == pytest.approx(0.09)
    assert swept.by_phase[Phase.JUDGE].cost_usd == pytest.approx(0.03)
    assert swept.total.calls == 18  # 6 sessions x (2 serve + 1 judge)


def test_the_two_sides_of_the_bill_are_reported_but_never_blended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Candidate spend is what serving costs; world-model spend is what measuring costs.

    One number covering both would overstate the price of serving the policy and understate the
    price of producing the evidence, so they are printed as two labeled lines and stored as two
    fields.
    """
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9}, session_usd=0.02)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes")
    assert result.exit_code == 0, result.output
    assert _says(result.output, "measured candidate spend $0.0013")
    assert _says(result.output, "measured world-model spend $0.1200 over 6 session(s)")
    assert _says(result.output, "eval infrastructure, not serving cost")
    assert _says(result.output, 'recorded as kind="sweep"')

    _matrix, _policy, _report, manifest_path = _paths(root)
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    sweep = manifest.record_for(Stage.SWEEP)
    assert sweep is not None
    assert sweep.spend_usd == pytest.approx(0.00132)  # candidate side, unchanged
    assert sweep.world_model_spend_usd == pytest.approx(0.12)  # its own field, never summed in
    assert sweep.total_spend_usd == pytest.approx(0.12132)  # only the cap adds them


def test_the_spend_cap_counts_the_world_model_side_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cap is a question about money leaving the account, and both sides do.

    The candidate side of this sweep projects at $0.33 and measures at $0.0013; the world-model
    side measures $0.12. A cap of $0.50 clears the pre-sweep projection either way, so the only
    thing that can stop the SECOND, freshly-forced sweep is the first run's total, and it only
    exceeds $0.50 once the world-model side is counted.
    """
    world_model = _patch_seams(
        monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9}, session_usd=0.05
    )
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes", "--max-usd", "0.60").exit_code == 0
    after_first = len(world_model.tasks)
    # First run: $0.0013 candidate + $0.30 world model = $0.3013 recorded.
    stopped = _run(tmp_path, root, "--yes", "--max-usd", "0.60", "--force-from", "sweep")
    assert stopped.exit_code == 1, stopped.output
    assert len(world_model.tasks) == after_first  # not one new episode
    assert _says(stopped.output, "stopped at the spend cap")
    assert _says(stopped.output, "$0.30 of its $0.60 cap")


def test_a_candidate_only_cap_would_have_let_that_second_sweep_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative control: with no world-model spend to count, the same cap does not trip."""
    world_model = _patch_seams(
        monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9}, session_usd=0.0
    )
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes", "--max-usd", "0.60").exit_code == 0
    after_first = len(world_model.tasks)
    again = _run(tmp_path, root, "--yes", "--max-usd", "0.60", "--force-from", "sweep")
    assert again.exit_code == 0, again.output
    assert len(world_model.tasks) > after_first


def test_the_first_sweep_says_the_world_model_side_is_not_projectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interactive_stdin: None
) -> None:
    """Before a model's first sweep there is nothing to forecast from, and silence would mislead.

    "Not in this figure" reads as "not much" unless the line says how large that side can get:
    measured 7.0x the candidate side on a real tau corpus. So the caveat states both that it is
    unprojectable and that the printed total is a lower bound.
    """
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    answer = _Answer(False)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    monkeypatch.setattr(optimize_module, "_console", Console(width=240, force_terminal=True))
    result = _run(tmp_path, root)
    assert result.exit_code == 0, result.output
    assert _says(result.output, "not projectable before this model's first sweep")
    assert _says(result.output, "7.0x the candidate side")
    assert _says(result.output, "treat the number above as a lower bound")
    # No forecast is invented, and the run still proceeds to the confirmation.
    assert "projected~$" not in _flat(result.output)
    assert answer.asked


def test_a_second_sweep_forecasts_the_world_model_side_from_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once this model has been swept, its OWN measured ratio is the honest basis for a forecast."""
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9}, session_usd=0.02)
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    # First sweep: $0.00132 candidate, $0.12 world model, a ratio of ~90.9x.
    forced = _run(tmp_path, root, "--yes", "--force-from", "sweep")
    assert forced.exit_code == 0, forced.output
    assert _says(forced.output, "plus a projected ~$30.00 world-model side")
    assert _says(
        forced.output, "measured $0.1200 world-model against $0.0013 projectable candidate"
    )
    assert _says(forced.output, "90.9x")
    assert _says(forced.output, "a forecast from one prior sweep, not arithmetic")


def test_the_forecast_stops_a_sweep_a_candidate_only_cap_would_have_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the allowance: the cap sees the money before it is spent, not after.

    The candidate projection alone is $0.33, so a $5 cap clears it easily and the sweep would
    start. The first sweep measured a 90.9x world-model ratio, which forecasts ~$30 for the same
    grid, so the run stops BEFORE buying any of it and says what the forecast rests on.
    """
    world_model = _patch_seams(
        monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9}, session_usd=0.02
    )
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    after_first = len(world_model.tasks)

    stopped = _run(tmp_path, root, "--yes", "--max-usd", "5", "--force-from", "sweep")
    assert stopped.exit_code == 1, stopped.output
    assert len(world_model.tasks) == after_first  # not one episode bought
    assert _says(stopped.output, "stopped at the spend cap")
    assert _says(stopped.output, "projection basis: the last sweep of this model measured")
    assert _says(stopped.output, "wmo optimize model support --max-usd <more>")


def test_without_the_forecast_that_same_cap_would_not_have_stopped_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative control: with no world-model spend to learn a ratio from, $5 clears $0.33."""
    world_model = _patch_seams(
        monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9}, session_usd=0.0
    )
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    after_first = len(world_model.tasks)
    again = _run(tmp_path, root, "--yes", "--max-usd", "5", "--force-from", "sweep")
    assert again.exit_code == 0, again.output
    assert len(world_model.tasks) > after_first


def test_a_sweep_the_coverage_contract_rejects_is_recorded_and_not_re_bought(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected sweep still cost money, so resume has to preserve it.

    `pricey` is throttled on every call, so its cells go unscored while `cheap` is scored on all
    three scenarios: the two would be ranked on different task sets, and the contract withholds
    the fit. The cells were paid for either way, and the printed message promises re-running will
    not buy them again. It has to be true.
    """
    world_model = _patch_seams(
        monkeypatch, rewards={"cheap-1": 0.4}, throttled_models=frozenset({"pricey-1"})
    )
    root = _project(tmp_path)
    rejected = _run(tmp_path, root, "--yes")
    assert rejected.exit_code == 1, rejected.output
    matrix_path, policy_path, _report, manifest_path = _paths(root)
    assert matrix_path.is_file() and not policy_path.exists()
    assert _says(rejected.output, "DIFFERENT scenarios")
    assert _says(rejected.output, "will not buy these cells again")
    bought = len(world_model.tasks)

    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    sweep = manifest.record_for(Stage.SWEEP)
    assert sweep is not None and sweep.spend_usd > 0.0

    # The documented way forward: accept the bias. It must skip the sweep, not repeat it.
    accepted = _run(tmp_path, root, "--yes", "--allow-uneven-coverage")
    assert accepted.exit_code == 0, accepted.output
    assert len(world_model.tasks) == bought  # not one cell re-bought
    assert _says(accepted.output, "matrix.json is current")
    assert _says(accepted.output, "bias accepted")
    assert policy_path.is_file()  # and the fit finally happened


def test_the_coverage_contract_still_binds_when_the_sweep_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recording a rejected sweep must not become a way to smuggle a biased matrix into a fit.

    This is the hole that opens if the contract is enforced at the end of the sweep instead of in
    front of the fit: the sweep is recorded, the next run skips it, and nothing re-checks the
    evidence. So the gate lives with the fit and binds on the skip path too.
    """
    world_model = _patch_seams(
        monkeypatch, rewards={"cheap-1": 0.4}, throttled_models=frozenset({"pricey-1"})
    )
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 1
    bought = len(world_model.tasks)
    _matrix, policy_path, _report, _manifest = _paths(root)

    # Same inputs, no --allow-uneven-coverage: the sweep is skipped (so nothing is spent) and the
    # fit is STILL withheld rather than quietly proceeding on the biased matrix.
    again = _run(tmp_path, root, "--yes")
    assert again.exit_code == 1, again.output
    assert len(world_model.tasks) == bought
    assert _says(again.output, "matrix.json is current")  # the sweep really was skipped
    assert _says(again.output, "DIFFERENT scenarios")  # and the gate still ran
    assert not policy_path.exists()


def test_accepting_biased_evidence_does_not_stick_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consent to fit on uneven evidence is an input to the fit, so revoking it must be noticed.

    Without `allow_uneven` in the fit's fingerprint the flag is a one-way door: grant it once and
    every later run skips the fit, never reaches the coverage gate, and serves a policy fitted on
    knowingly-biased evidence with nothing in the output saying so.
    """
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4}, throttled_models=frozenset({"pricey-1"}))
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 1  # withheld
    accepted = _run(tmp_path, root, "--yes", "--allow-uneven-coverage")
    assert accepted.exit_code == 0, accepted.output

    # Same project, flag withdrawn: the fit is re-planned because its inputs changed, the gate
    # runs again, and the bias is refused rather than silently inherited.
    revoked = _run(tmp_path, root, "--yes")
    assert revoked.exit_code == 1, revoked.output
    assert _says(revoked.output, "allow_uneven changed")
    assert _says(revoked.output, "DIFFERENT scenarios")


def test_the_cap_refuses_before_asking_rather_than_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interactive_stdin: None
) -> None:
    """Being asked to approve a run and then told it cannot start is the wrong order."""
    world_model = _patch_seams(monkeypatch)
    root = _project(tmp_path)
    answer = _Answer(True)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    monkeypatch.setattr(optimize_module, "_console", Console(width=240, force_terminal=True))
    result = _run(tmp_path, root, "--max-usd", "0.01")
    assert result.exit_code == 1, result.output
    assert answer.asked == []  # never asked
    assert world_model.tasks == []
    assert _says(result.output, "stopped at the spend cap")


def test_a_zero_priced_pool_is_still_confirmed_because_the_simulator_is_not_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interactive_stdin: None
) -> None:
    """Keying the question on the candidate projection skips it exactly when it matters most.

    A pool priced at zero projects $0.00 candidate-side, but the sweep still spends on the world
    model. That is the case where the simulator's cost IS the whole bill, so the question has to
    key on "will this run buy cells", not on a candidate-side number.
    """
    world_model = _patch_seams(monkeypatch, session_usd=0.05)
    root = _project(tmp_path)
    free_pool = tmp_path / "free-pool.toml"
    free_pool.write_text(
        "[[model]]\n"
        'name = "cheap"\n'
        'kind = "openai"\n'
        'model = "cheap-1"\n'
        "input_per_mtok = 0.0\n"
        "output_per_mtok = 0.0\n",
        encoding="utf-8",
    )
    answer = _Answer(False)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    monkeypatch.setattr(optimize_module, "_console", Console(width=240, force_terminal=True))
    result = _run(tmp_path, root, pool=free_pool)
    assert result.exit_code == 0, result.output
    assert len(answer.asked) == 1  # asked, despite a $0.00 candidate projection
    assert world_model.tasks == []  # and declining bought nothing


def test_the_cap_counts_spend_from_runs_whose_records_were_superseded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--max-usd` bounds the optimization, and a re-swept stage's money still left the account.

    The manifest keeps only the LATEST record per stage, so seeding the cap from the stage rows
    forgets every superseded sweep. Three sweeps of $0.30 must read as $0.90 spent, not $0.30.
    """
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9}, session_usd=0.05)
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    assert _run(tmp_path, root, "--yes", "--force-from", "sweep").exit_code == 0

    _matrix, _policy, _report, manifest_path = _paths(root)
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    sweep = manifest.record_for(Stage.SWEEP)
    assert sweep is not None
    # One sweep's record survives, but both sweeps' spend does.
    assert manifest.lifetime_spend_usd == pytest.approx(sweep.total_spend_usd * 2, rel=1e-6)


def test_a_non_tty_spending_run_without_yes_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consent is said, never inferred: no terminal + no --yes + a sweep to buy = exit 2.

    This command briefly shipped the opposite (proceed-and-note), which spent a scripted
    caller's money without agreement; the refusal message names both honest paths out.
    """
    world_model = _patch_seams(monkeypatch)
    root = _project(tmp_path)
    # CliRunner's stdin is not a terminal, and _console is left at its non-terminal default.
    result = _run(tmp_path, root)
    assert result.exit_code == 2, result.output
    assert "cannot ask for spend consent" in result.output
    assert "--yes" in result.output and "--dry-run" in result.output
    assert world_model.tasks == []  # no episode ran
    assert not _paths(root)[0].exists()  # no matrix bought


def test_dry_run_prints_the_plan_and_spends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--dry-run is the read-only view of the plan table: exit 0, no episodes, no artifacts."""
    world_model = _patch_seams(monkeypatch)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--dry-run")
    assert result.exit_code == 0, result.output
    assert "sweep" in result.output  # the plan table rendered
    assert "dry run: nothing was run and nothing was spent" in result.output
    assert world_model.tasks == []
    matrix_path, manifest_path = _paths(root)[0], _paths(root)[1]
    assert not matrix_path.exists()
    assert not manifest_path.exists()  # a dry run leaves no resume state behind
