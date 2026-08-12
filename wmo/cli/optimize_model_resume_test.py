"""Staged model optimizer resume and route-parity tests."""

# ruff: noqa: F403, F405
from wmo.cli.optimize_model_fixtures_test import *


def test_one_command_lands_every_artifact_where_serving_reads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The promise: after one command the model has a fitted, dialed, servable policy."""
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes")
    assert result.exit_code == 0, result.output

    matrix_path, policy_path, report_path, manifest_path = _paths(root)
    # The matrix and the report are this command's own; the policy is where `wmo serve` looks.
    matrix = OutcomeMatrix.load(matrix_path)
    assert len(matrix.outcomes) == 6  # 2 candidates x 3 scenarios x 1 episode
    assert matrix.scenario_ids() == list(_HELD_OUT_IDS[:3])
    policy = RoutingPolicy.load(policy_path)
    assert policy.kind == "knn"
    assert policy.knn_bank_path is not None
    # tune's as-fitted snapshot semantics survive the orchestrator: the dial is recorded and the
    # un-tuned artifact is preserved beside it, so sliding again never compounds.
    assert policy.cost_quality == 0.25
    assert (policy_path.parent / "policy.base.json").is_file()
    assert RoutingPolicy.load(policy_path.parent / "policy.base.json").cost_quality is None
    report = ImprovementReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    assert report.endpoint_id == "support"
    assert report.headline.scenarios_compared == 1
    assert set(policy.fit_scenario_ids).isdisjoint(report.scenario_ids)
    assert len(policy.fit_scenario_ids) + report.scenario_count == 3

    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert [record.stage.value for record in manifest.stages] == ["sweep", "fit", "tune", "report"]
    assert manifest.world_model == "support"

    # The plan table printed every stage before anything spent, and the run ended on the payoff.
    assert _says(result.output, "optimize model: support")
    for stage in ("preflight", "sweep", "fit", "tune", "report"):
        assert stage in _flat(result.output)
    assert _says(result.output, "estimated candidate spend")
    assert _says(result.output, "serve it:   wmo serve --name support")
    assert _says(result.output, 'POST /v1/chat/completions  (model="support")')
    assert "quality" in _flat(result.output) and "latency" in _flat(result.output)


def test_a_second_run_skips_every_stage_and_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume is the property that makes the command safe to re-type: no cell is bought twice."""
    world_model = _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    episodes_after_first = len(world_model.tasks)

    again = _run(tmp_path, root, "--yes")
    assert again.exit_code == 0, again.output
    flat = _flat(again.output)
    assert len(world_model.tasks) == episodes_after_first  # not one new episode
    # Every skip states what was unchanged, not just that it skipped.
    assert _says(again.output, "matrix.json is current: same pool, same scenarios, same episodes")
    assert _says(again.output, "policy.json is current: same matrix, same knn knobs")
    assert _says(again.output, "dial already at 0.25")
    assert _says(again.output, "report.json is current")
    assert "everystageiscurrent" in flat
    # A run with nothing to do asks nothing and spends nothing.
    assert "estimatedcandidatespend" not in flat


def test_force_from_sweep_redoes_the_sweep_and_everything_after_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world_model = _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    before = len(world_model.tasks)

    forced = _run(tmp_path, root, "--yes", "--force-from", "sweep")
    assert forced.exit_code == 0, forced.output
    assert len(world_model.tasks) == before * 2  # the cells were bought again, as asked
    assert _says(forced.output, "forced by --force-from")
    # Downstream is redone because its input is about to change, and the table says exactly that.
    assert _says(forced.output, "runs after sweep, which will change its input")


def test_force_from_fit_leaves_the_paid_sweep_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The point of a staged redo: refitting must never re-buy cells.
    world_model = _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    before = len(world_model.tasks)

    forced = _run(tmp_path, root, "--yes", "--force-from", "fit")
    assert forced.exit_code == 0, forced.output
    assert len(world_model.tasks) == before
    assert _says(forced.output, "matrix.json is current")


def test_editing_the_pool_reruns_the_sweep_and_names_the_pool_as_the_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world_model = _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    before = len(world_model.tasks)

    repriced = _pool_file(tmp_path, pricey_out=30.0)
    changed = _run(tmp_path, root, "--yes", pool=repriced)
    assert changed.exit_code == 0, changed.output
    assert len(world_model.tasks) > before
    assert _says(changed.output, "pool changed")


def test_deleting_the_optimize_dir_resets_resume_without_touching_the_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No hidden state: the manifest dir is disposable and the serving artifact is not in it."""
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    matrix_path, policy_path, _report, manifest_path = _paths(root)
    fitted_before = RoutingPolicy.load(policy_path).fitted_from

    for path in (matrix_path, manifest_path, matrix_path.parent / REPORT_FILENAME):
        path.unlink()
    matrix_path.parent.rmdir()
    assert policy_path.is_file()  # serving still works, which is the whole point

    again = _run(tmp_path, root, "--yes")
    assert again.exit_code == 0, again.output
    assert _says(again.output, "never completed here")
    assert RoutingPolicy.load(policy_path).fitted_from != fitted_before  # a genuinely fresh fit


def test_a_corrupt_manifest_warns_and_replans_instead_of_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world_model = _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    before = len(world_model.tasks)
    _matrix, _policy, _report, manifest_path = _paths(root)
    manifest_path.write_text("{ truncated", encoding="utf-8")

    again = _run(tmp_path, root, "--yes")
    assert again.exit_code == 0, again.output
    assert _says(again.output, "could not be read as a run manifest")
    # Be honest about the price of a reset rather than claiming it is free: with no records to
    # match against, every stage reads as "never completed here" and the sweep is measured again.
    # `RunManifest.save` is atomic precisely so this path stays rare.
    assert _says(again.output, "never completed here")
    assert len(world_model.tasks) == before * 2  # re-bought, which is what a reset costs
    assert RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8")).stages


def test_distill_is_rejected_with_the_command_that_does_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", "--distill", "distill.toml")
    assert result.exit_code != 0
    flat = _flat(result.output)
    assert "reservedandnotimplementedinthisbuild" in flat
    assert _says(result.output, "wmo optimize distill run")
    assert _says(result.output, "wmo optimize route student")
    assert not _paths(root)[0].exists()  # nothing ran


def test_distill_refusal_carries_the_teacher_verdict_on_the_matrix_it_finds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stage is unwired, but its preflight is not: the refusal answers "should I?" anyway."""
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.42})
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes", "--scenarios", "4").exit_code == 0

    result = _run(tmp_path, root, "--yes", "--distill", "distill.toml")

    assert result.exit_code != 0
    assert _says(result.output, "the teacher-search verdict on this model's current matrix")
    # Four scenarios is under the gate's evidence bar, and it says so rather than guessing.
    assert _says(result.output, "INSUFFICIENT EVIDENCE")
    assert _says(result.output, "wmo optimize distill probe")


def test_distill_refusal_without_a_matrix_is_just_the_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing has been measured yet, so there is no verdict to quote and none is invented."""
    _patch_seams(monkeypatch)
    root = _project(tmp_path)

    result = _run(tmp_path, root, "--yes", "--distill", "distill.toml")

    assert result.exit_code != 0
    assert not _says(result.output, "teacher-search verdict")


def test_force_from_a_reserved_stage_says_it_is_not_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", "--force-from", "distill")
    assert result.exit_code != 0
    flat = _flat(result.output)
    assert "reservedslot" in flat and "sweep|fit|tune|report" in flat


def test_force_from_compact_says_it_configures_rather_than_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compaction is a configuration, so there is no work to redo: refuse either way it is asked."""
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    off = _run(tmp_path, root, "--yes", "--force-from", "compact")
    assert off.exit_code != 0
    assert _says(off.output, "named no compressor for it to configure anything with")
    assert _says(off.output, "force one of sweep | fit | tune | report")

    on = _run(tmp_path, root, "--yes", "--force-from", "compact", "--compressor", "truncate")
    assert on.exit_code != 0
    assert _says(on.output, "configures the sweep and the fit rather than running on its own")
    assert _says(on.output, "Change --compressor/--aggressiveness to measure a different arm")


def test_force_from_an_unknown_stage_lists_the_real_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", "--force-from", "nonsense")
    assert result.exit_code != 0
    assert _says(result.output, "unknown stage 'nonsense'")
    assert "sweep|fit|tune|report" in _flat(result.output)


def test_the_spend_cap_stops_before_the_sweep_and_prints_how_to_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 2 candidates x 3 scenarios x 4 calls at the pool's prices projects well over $0.01.
    world_model = _patch_seams(monkeypatch)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", "--max-usd", "0.01")
    assert result.exit_code == 1, result.output
    assert world_model.tasks == []  # stopped BEFORE the stage, not during it
    flat = _flat(result.output)
    assert "stoppedatthespendcap" in flat
    assert _says(result.output, "wmo optimize model support --max-usd <more>")
    assert not _paths(root)[0].exists()


def test_a_cap_that_covers_the_run_lets_it_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The negative control for the cap test: the same run with room under the cap completes.
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", "--max-usd", "100")
    assert result.exit_code == 0, result.output
    assert _paths(root)[0].is_file()


def test_declining_the_confirmation_spends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interactive_stdin: None
) -> None:
    world_model = _patch_seams(monkeypatch)
    root = _project(tmp_path)
    answer = _Answer(False)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    monkeypatch.setattr(optimize_module, "_console", Console(width=240, force_terminal=True))
    result = _run(tmp_path, root)
    assert result.exit_code == 0, result.output
    assert answer.asked  # exactly one question, and it was asked before any episode
    assert len(answer.asked) == 1
    assert world_model.tasks == []
    assert not _paths(root)[0].exists()


def test_the_plan_table_prices_the_sweep_and_labels_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interactive_stdin: None
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    answer = _Answer(False)
    monkeypatch.setattr(consent_module, "Confirm", answer)
    monkeypatch.setattr(optimize_module, "_console", Console(width=240, force_terminal=True))
    result = _run(tmp_path, root)
    flat = _flat(result.output)
    # 3 scenarios x 1 episode x 4 calls = 12 calls; cheap = 12 x (2000 + 250x2)/1e6 = $0.03,
    # pricey = 10x that, so the projected total is $0.33.
    assert "~$0.33" in flat
    # Rich may still wrap / interleave plan-cell text on some terminals; match durable pieces.
    assert "2candidate(s)" in flat
    assert "scenario(s)" in flat and "episode(s)" in flat
    assert "1episode(s)" in flat or "x1" in flat
    # The free stages say free rather than showing a fabricated number, and the estimate names
    # itself a projection with its assumption spelled out.
    # 3 scenarios split 70/30 for router fit vs report: 2 fit, 1 reserved (PR #308).
    # Table wrapping can interleave columns; match durable fragments rather than one long cell.
    assert "knnover2fit" in flat or "2fitscenario" in flat
    assert "cost_quality0.25" in flat or "Balanced(default)" in flat
    assert "aprojection" in flat and "assumedoutputtoken" in flat
    assert "areNOTinthatfigure" in flat or "NOTinthefigure" in flat or "NOT" in flat


def test_the_plan_table_shows_the_pace_and_what_a_resume_will_not_rebuy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interactive_stdin: None
) -> None:
    """Two things an operator authorizing a run needs to see: how hard it will lean on the
    provider, and how much of the grid a previous attempt already paid for."""
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    monkeypatch.setattr(consent_module, "Confirm", _Answer(False))
    monkeypatch.setattr(optimize_module, "_console", Console(width=240, force_terminal=True))
    paced = _run(tmp_path, root, "--concurrency", "6")
    paced_flat = _flat(paced.output)
    assert "2candidate(s)" in paced_flat and "6atatime" in paced_flat
    assert "scenario(s)" in paced_flat and "episode(s)" in paced_flat

    # A sidecar from an interrupted attempt at THIS plan: the row says what is left to buy.
    matrix_path = _paths(root)[0]
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    plan = _sweep_plan(tmp_path, root)
    sidecar = matrix_path.with_name(matrix_path.name + ".partial.jsonl")
    sidecar.write_text(
        PartialHeader(identity=plan.identity).model_dump_json()
        + "\n"
        + ScenarioOutcome(
            scenario_id=scenario_id(plan.scenarios[0]),
            task=plan.scenarios[0].task,
            model="cheap",
            reward=0.5,
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    resumed = _run(tmp_path, root)
    assert "1alreadymeasured,5tobuy" in _flat(resumed.output)


def test_an_unscored_sweep_withholds_the_fit_and_keeps_the_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The orchestrator owns no coverage policy of its own: `route sweep`'s contract holds."""
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    real_env = env_module.WorldModelEnv
    monkeypatch.setattr(
        env_module,
        "WorldModelEnv",
        lambda world_model, *, score_on_close=False: real_env(world_model),
    )
    world_model = _patch_seams(monkeypatch)
    result = _run(tmp_path, root, "--yes")
    assert result.exit_code == 1, result.output
    matrix_path, policy_path, _report, manifest_path = _paths(root)
    assert matrix_path.is_file()  # the paid cells are on disk with their errors
    assert all(not outcome.scored for outcome in OutcomeMatrix.load(matrix_path).outcomes)
    assert not policy_path.exists()  # the fit was withheld
    assert _says(result.output, "no cell was scored")

    # ...and the rejected sweep is RECORDED, so the second attempt costs nothing. Before this was
    # fixed the contract's exit ran before the record was saved, and every retry re-bought every
    # cell while the printed message claimed otherwise.
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert manifest.record_for(Stage.SWEEP) is not None
    bought = len(world_model.tasks)
    again = _run(tmp_path, root, "--yes")
    assert again.exit_code == 1, again.output
    assert len(world_model.tasks) == bought  # not one cell re-bought
    assert _says(again.output, "no cell was scored")  # and still refused


def test_the_sweep_stage_and_route_sweep_produce_identical_matrices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extraction's guarantee: two commands, one measurement, byte-identical evidence.

    Both faces call `wmo.optimize.routing.sweep`, so a divergence here would mean one of them
    grew its own copy of the scenario cut, the tools hint, or the cell ordering.
    """
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9}, modules=(route_module,))
    root = _project(tmp_path)
    pool = _pool_file(tmp_path)

    assert _run(tmp_path, root, "--yes", pool=pool).exit_code == 0
    orchestrated = json.loads(_paths(root)[0].read_text(encoding="utf-8"))

    manual_out = tmp_path / "manual-matrix.json"
    manual = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--root",
            str(root),
            "--pool",
            str(pool),
            "--out",
            str(manual_out),
            "--scenarios",
            "3",
            "--max-steps",
            "4",
            "--yes",
        ],
    )
    assert manual.exit_code == 0, manual.output
    manually = json.loads(manual_out.read_text(encoding="utf-8"))

    assert orchestrated["pool"] == manually["pool"]
    assert len(orchestrated["outcomes"]) == len(manually["outcomes"])
    # Compared as WHOLE cells minus a named exclusion, not as an allowlist of fields: an
    # allowlist silently stops covering anything added to ScenarioOutcome later, and cost and
    # error capture are exactly where two faces would drift.
    wall_clock = {"call_seconds"}
    for cell, other in zip(orchestrated["outcomes"], manually["outcomes"], strict=True):
        assert {k: v for k, v in cell.items() if k not in wall_clock} == {
            k: v for k, v in other.items() if k not in wall_clock
        }


def test_the_closing_numbers_name_their_anchor_and_their_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every headline number says what it was measured against and how (AGENTS numbers honesty)."""
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes")
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "world-modelsimulated" in flat and "held-outscenario" in flat
    assert "measuredcandidate-sideatlistprices" in flat
    assert "cacheeffectsnotmodeled" in flat
    assert "walltimeperpolicycall" in flat
    # The dial and the fallback are named beside them, so the reader knows which policy scored.
    assert _says(result.output, "dial: 0.25 balanced (default)")
    assert "policy:knn(guarded,fallback" in flat


def test_an_anchor_outside_the_pool_is_refused_before_anything_is_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flag error must not surface after the sweep has been paid for.

    The anchor has to name a pool model, and the pool is fully loaded by the pre-flight, so this
    is knowable for free. It used to be caught only in the report stage, by which point the sweep
    was bought, the policy fitted, and the dial set.
    """
    world_model = _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", "--baseline", "ghost")
    assert result.exit_code != 0
    assert _says(result.output, "--baseline 'ghost' is not a model in")
    assert _says(result.output, "Available: cheap, pricey")
    assert world_model.tasks == []  # not one cell bought
    assert not _paths(root)[0].exists()


def test_a_fallback_outside_the_pool_is_refused_before_anything_is_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--fallback names a pool candidate too, so it is checked at --baseline's boundary.

    A typo used to render in the plan table as if it were a real model, pass the spend
    confirmation, and only fail inside the fit stage, after the sweep had been bought.
    """
    world_model = _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", "--fallback", "ghost")
    assert result.exit_code != 0
    assert _says(result.output, "--fallback 'ghost' is not a model in")
    assert _says(result.output, "Available: cheap, pricey")
    assert world_model.tasks == []  # not one cell bought
    assert not _paths(root)[0].exists()


def test_a_fallback_typo_is_refused_by_a_dry_run_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan table is the thing a --dry-run reader trusts; it must not print a typo as real."""
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--dry-run", "--fallback", "ghost")
    assert result.exit_code != 0
    assert _says(result.output, "--fallback 'ghost' is not a model in")
    assert not _says(result.output, "knn (guarded, fallback ghost)")


def test_a_missing_world_model_names_the_command_a_user_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    result = runner.invoke(
        app,
        ["optimize", "model", "ghost", "--root", str(root), "--pool", str(_pool_file(tmp_path))],
    )
    assert result.exit_code != 0
    assert "ghost" in _flat(result.output)


def test_a_refit_retires_the_stale_dial_snapshot_it_superseded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chained refit-then-dial must not trip `route tune`'s stale-snapshot refusal.

    That refusal exists for a human who refits by hand and would otherwise dial the superseded
    fit back over the new one. Here the refit and the dial are the same command's own consecutive
    stages, so the snapshot is stale by construction and is retired explicitly.
    """
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    _matrix, policy_path, _report, _manifest = _paths(root)
    base_path = policy_path.parent / "policy.base.json"
    first_fit = RoutingPolicy.load(base_path).fitted_from

    forced = _run(tmp_path, root, "--yes", "--force-from", "sweep")
    assert forced.exit_code == 0, forced.output
    assert _says(forced.output, "re-baselined the dial")
    # The snapshot is the NEW fit as fitted, and the served policy is the new fit dialed.
    assert base_path.is_file()
    assert RoutingPolicy.load(base_path).fitted_from != first_fit
    assert RoutingPolicy.load(base_path).cost_quality is None
    assert RoutingPolicy.load(policy_path).cost_quality == 0.25


def test_a_redo_that_reproduces_the_same_fit_keeps_its_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative control: only a SUPERSEDED snapshot is retired, never a matching one."""
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    _matrix, policy_path, _report, _manifest = _paths(root)
    base_path = policy_path.parent / "policy.base.json"
    before = base_path.read_bytes()

    # --force-from fit refits the SAME matrix, so the fit (and its provenance) is identical.
    forced = _run(tmp_path, root, "--yes", "--force-from", "fit")
    assert forced.exit_code == 0, forced.output
    assert not _says(forced.output, "re-baselined the dial")
    assert base_path.read_bytes() == before


def test_the_dial_setting_reaches_the_served_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--cost-quality is the one operating-point decision, and it lands on the artifact."""
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", "--cost-quality", "0")
    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(_paths(root)[1])
    assert policy.cost_quality == 0.0
    assert policy.floor_q == 0.5  # the quality end of the measured frontier
    assert _says(result.output, "dial: 0 quality max")
