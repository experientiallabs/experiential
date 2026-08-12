"""Route sweep behavior tests."""

# ruff: noqa: F403, F405
from wmo.cli.route_fixtures_test import *


def test_route_sweep_writes_a_matrix_fit_can_consume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seams = _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "3", "--max-steps", "4", "--yes")
    assert result.exit_code == 0, result.output

    matrix = OutcomeMatrix.load(out)
    # Leak-free AND deterministic: the first three held-out traces by id, never a train task.
    assert matrix.scenario_ids() == list(_HELD_OUT_IDS[:3])
    assert {o.task for o in matrix.outcomes} == {f"task {tid}" for tid in _HELD_OUT_IDS[:3]}
    assert matrix.model_names() == ["cheap", "pricey"]
    assert len(matrix.outcomes) == 6  # 2 candidates x 3 scenarios x 1 episode
    assert all(o.reward == 0.75 for o in matrix.outcomes)
    assert matrix.mean_reward("cheap") == 0.75
    # The candidates saw the corpus's tool surface, summarized from the TRAIN split only: the
    # held-out band's own tool never reaches them (deriving the hint from the measured band would
    # leak, and would make the hint depend on where `--scenarios` cut).
    assert seams.systems
    assert all("ls(path)" in system for system in seams.systems)
    assert not any(_HELD_OUT_TOOL in system for system in seams.systems)
    # Progress streamed per cell, and the printed handoff chains the workflow.
    assert "[1/6]" in result.output and "[6/6]" in result.output
    assert _says(result.output, f"wmo optimize route fit {out} --kind knn")
    # Per-candidate scored counts print even on a clean sweep, so "3 of 3 each" is visible rather
    # than inferred from a total that cannot distinguish 3+3 from 5+1.
    assert _says(result.output, "Scored coverage per candidate")
    assert _flat(result.output).count("cheap30-") == 1  # cheap: 3 scored, 0 unscored, none lost

    # The whole point of the matrix: `fit` consumes it without further preparation.
    policy_file = tmp_path / "policy.json"
    fitted = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(out),
            "--kind",
            "rank",
            "--out",
            str(policy_file),
            "--clusters",
            "2",
            "--top-k-clusters",
            "1",
        ],
    )
    assert fitted.exit_code == 0, fitted.output
    assert RoutingPolicy.load(policy_file).kind == "rank"


def test_route_sweep_scores_every_episode_through_a_scoring_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `WorldModelEnv(..., score_on_close=True)` is the contract `evaluate_pool` needs: the env
    # must judge each episode as it closes, or no cell is evidence.
    seams = _patch_seams(monkeypatch, reward=0.5)
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "2", "--yes")
    assert result.exit_code == 0, result.output
    matrix = OutcomeMatrix.load(out)
    assert len(seams.world_model.scored) == len(matrix.outcomes) == 4
    assert all(o.scored and o.error is None and o.success for o in matrix.outcomes)
    assert [o.reward for o in matrix.outcomes] == [0.5] * 4
    # Every episode ran with index enrichment suspended, so no candidate's predictions can
    # become the next candidate's retrieved demos (which would make scores sweep-order dependent).
    assert seams.world_model.opened_frozen == [True] * 4


def test_route_sweep_at_higher_concurrency_writes_the_same_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `--concurrency` is a speed knob the operator can see in the plan, not a change of evidence:
    # the same cells, the same rows, in the same order.
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(
        tmp_path, root, "support", "--scenarios", "3", "--concurrency", "4", "--yes"
    )
    assert result.exit_code == 0, result.output
    assert _says(result.output, "4 cell(s) run at once")
    matrix = OutcomeMatrix.load(out)
    assert [(o.model, o.scenario_id) for o in matrix.outcomes] == [
        (model, sid) for model in ("cheap", "pricey") for sid in _HELD_OUT_IDS[:3]
    ]
    assert all(o.scored for o in matrix.outcomes)


def test_route_sweep_resumes_the_cells_an_interrupted_run_already_bought(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grid killed mid-flight is finished, not repeated: the filed gap, end to end.

    The first attempt dies inside its fourth cell. The rows it completed are on disk beside the
    matrix, so the second attempt measures only what is missing and the matrix it writes is the
    whole grid.
    """
    seams = _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    real_env = env_module.WorldModelEnv
    cells = itertools.count(1)

    class _DiesOnTheFourthCell:
        def __init__(self, world_model: object, *, score_on_close: bool = False) -> None:
            self._inner = real_env(cast("WorldModel", world_model), score_on_close=score_on_close)
            self._n = next(cells)

        def reset(self, task: str | None = None, seed_state: EnvState | None = None) -> EnvState:
            if self._n == 4:
                raise RuntimeError("simulated transport fault")
            return self._inner.reset(task=task, seed_state=seed_state)

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    monkeypatch.setattr(env_module, "WorldModelEnv", _DiesOnTheFourthCell)
    out, first = _sweep(tmp_path, root, "support", "--scenarios", "3", "--yes")
    assert first.exit_code != 0
    assert not out.exists()  # no matrix: the sweep never finished
    sidecar = out.with_name(out.name + ".partial.jsonl")
    assert len(sidecar.read_text(encoding="utf-8").splitlines()) == 4  # header + 3 paid cells

    monkeypatch.setattr(env_module, "WorldModelEnv", real_env)
    scored_before = len(seams.world_model.scored)
    _, second = _sweep(tmp_path, root, "support", "--scenarios", "3", "--yes")
    assert second.exit_code == 0, second.output
    assert _says(second.output, "RESUMING: 3 of those cell(s) are already measured")
    assert len(seams.world_model.scored) - scored_before == 3  # only the missing cells ran
    assert len(OutcomeMatrix.load(out).outcomes) == 6
    assert not sidecar.exists()


def test_route_sweep_refuses_a_sidecar_that_belongs_to_a_different_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Refused BEFORE the spend question, naming the pin that moved: two arms in one matrix is a
    # comparison nobody measured.
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    out, _first = _sweep(tmp_path, root, "support", "--scenarios", "3", "--yes")
    sidecar = out.with_name(out.name + ".partial.jsonl")
    # Re-create the sidecar a killed run would have left, then change what the sweep measures.
    sidecar.write_text(
        "\n".join(
            [
                PartialHeader(
                    identity=PlanIdentity(
                        pool="stale",
                        task_set_id="task-set-routing",
                        tasks_sha256="0" * 64,
                        task_set_inputs=(),
                        scenarios=tuple(_HELD_OUT_IDS[:3]),
                        episodes=1,
                        max_steps=20,
                        history_chars=DEFAULT_HISTORY_CHARS,
                        compression="raw text (no compression)",
                    )
                ).model_dump_json(),
                *(row.model_dump_json() for row in OutcomeMatrix.load(out).outcomes[:2]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _, blocked = _sweep(tmp_path, root, "support", "--scenarios", "3", "--yes")
    assert blocked.exit_code != 0
    assert _says(blocked.output, "measured under a DIFFERENT plan")
    assert _says(blocked.output, "candidate pool changed")


def test_route_sweep_without_a_scoring_env_produces_no_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The negative control for the test above: drop `score_on_close` and every cell comes back
    # unscored, so the failure the assertion catches is a real behavioral difference.
    seams = _patch_seams(monkeypatch, no_scoring=True)
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "2", "--yes")
    # A sweep that scored nothing exits NON-ZERO, so `sweep && fit` in a script stops instead of
    # fitting on a matrix the command itself says is not evidence.
    assert result.exit_code == 1, result.output
    assert seams.world_model.scored == []
    # ... and the paid rows are still on disk, carrying the `error` that explains them.
    matrix = OutcomeMatrix.load(out)
    assert all(not o.scored for o in matrix.outcomes)
    assert _says(result.output, "no cell was scored")


def test_route_sweep_withholds_the_fit_handoff_on_uneven_scored_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `pricey` is throttled on every call, so its episodes error and its cells go UNSCORED while
    # `cheap` is scored on all three scenarios. Both fitters SKIP unscored rows, so ranking these
    # two against each other would compare 3 scenarios of cheap with 0 of pricey.
    seams = _patch_seams(monkeypatch, throttled_models=frozenset({"pricey-1"}))
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "3", "--yes")
    assert result.exit_code == 1, result.output
    # The artifact is still written: those cells were paid for, and their `error` is the diagnosis.
    matrix = OutcomeMatrix.load(out)
    assert [o.scored for o in matrix.outcomes if o.model == "cheap"] == [True] * 3
    assert [o.scored for o in matrix.outcomes if o.model == "pricey"] == [False] * 3
    assert all("429" in (o.error or "") for o in matrix.outcomes if o.model == "pricey")
    flat = _flat(result.output)
    # The user can see WHICH candidate lost WHICH scenarios, not just a total.
    assert _says(result.output, "Scored coverage per candidate")
    assert _says(result.output, ", ".join(_HELD_OUT_IDS[:3]))
    assert "DIFFERENTscenarios" in flat and "cheap3,pricey0" in flat
    # A candidate the coverage table can only show as all-zero gets its cause quoted from the
    # matrix, named, with what to do: it is the one failure the table itself cannot explain.
    assert _says(result.output, "pricey was never scored; its first cell failed with")
    assert "ratelimitexceeded(429)" in flat
    assert _says(result.output, "fix that entry in the pool file")
    # The handoff is withheld, so `sweep && fit` in a script stops instead of fitting on it, and
    # the message names the one flag that proceeds anyway.
    assert "wmooptimizeroutefit" not in flat
    assert "--allow-uneven-coverage" in flat
    # A real sweep, not an aborted one: every cell ran (the cells are the paid evidence).
    assert len(seams.world_model.tasks) == 6


def test_route_sweep_allow_uneven_coverage_hands_off_and_still_states_the_bias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The opt-out for an operator who knows a candidate's backend was down all sweep and wants the
    # partial data anyway: same coverage table, same warning, but the handoff stands.
    _patch_seams(monkeypatch, throttled_models=frozenset({"pricey-1"}))
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(
        tmp_path, root, "support", "--scenarios", "3", "--yes", "--allow-uneven-coverage"
    )
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "DIFFERENTscenarios" in flat and "biasaccepted" in flat
    assert _says(result.output, f"wmo optimize route fit {out} --kind knn")


def test_route_sweep_withholds_the_fit_handoff_on_uneven_scored_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Uneven EPISODES, not uneven scenario presence: `pricey` keeps episode 0 of every scenario and
    # loses episodes 1 and 2, so both candidates cover BOTH scenarios and a presence-only gate sees
    # nothing wrong. What differs is the scored episode COUNT per (candidate, scenario), which is
    # exactly what the fitters weigh: `fit_rank_policy` averages every surviving episode into its
    # cluster mean (so a scenario counts three times as much for `cheap` as for `pricey`), and
    # `_overall_best` / `best_single_on_fit` pick the default and knn fallback off the same
    # episode-weighted means. Which episodes happened to fail must not decide the policy.
    seams = _patch_seams(monkeypatch, throttled_episodes={"pricey-1": (False, True, True)})
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "2", "--episodes", "3", "--yes")
    assert result.exit_code == 1, result.output
    # Every cell ran and the artifact is on disk: 2 candidates x 2 scenarios x 3 episodes.
    matrix = OutcomeMatrix.load(out)
    assert len(matrix.outcomes) == 12
    assert len(seams.world_model.tasks) == 12
    assert Counter(
        (outcome.model, outcome.scenario_id) for outcome in matrix.outcomes if outcome.scored
    ) == Counter(
        {
            ("cheap", _HELD_OUT_IDS[0]): 3,
            ("cheap", _HELD_OUT_IDS[1]): 3,
            ("pricey", _HELD_OUT_IDS[0]): 1,
            ("pricey", _HELD_OUT_IDS[1]): 1,
        }
    )
    # Presence is identical for both candidates, so nothing but the counts could have caught this.
    scored_scenarios = {
        name: {o.scenario_id for o in matrix.outcomes if o.scored and o.model == name}
        for name in ("cheap", "pricey")
    }
    assert scored_scenarios["cheap"] == scored_scenarios["pricey"] == set(_HELD_OUT_IDS[:2])
    flat = _flat(result.output)
    assert _says(result.output, "DIFFERENT numbers of scored episodes")
    # The table says WHICH candidate thinned WHICH scenario, and by how much.
    assert "pricey24" in flat  # 2 scored cells, 4 unscored
    assert _says(result.output, f"{_HELD_OUT_IDS[0]} 1/3")
    assert _says(result.output, f"{_HELD_OUT_IDS[1]} 1/3")
    # The handoff is withheld, so `sweep && fit` stops here, and the one opt-out is named.
    assert "wmooptimizeroutefit" not in flat
    assert "--allow-uneven-coverage" in flat


def test_route_sweep_allow_uneven_coverage_also_covers_uneven_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same opt-out, same bias statement, for the episode-count case: an operator who knows one
    # candidate was throttled through part of the sweep and wants the partial data anyway.
    _patch_seams(monkeypatch, throttled_episodes={"pricey-1": (False, True, True)})
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(
        tmp_path,
        root,
        "support",
        "--scenarios",
        "2",
        "--episodes",
        "3",
        "--yes",
        "--allow-uneven-coverage",
    )
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert _says(result.output, "DIFFERENT numbers of scored episodes")
    assert "biasaccepted" in flat
    assert _says(result.output, f"wmo optimize route fit {out} --kind knn")


def test_route_sweep_hands_off_when_every_candidate_lost_the_same_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The negative control for the two tests above: EVERY candidate loses episodes 1 and 2 of every
    # scenario, so the per-(candidate, scenario) counts are identical. That is still a comparison,
    # like-for-like on one episode per scenario, so the handoff stands and only the counts show it.
    _patch_seams(
        monkeypatch,
        throttled_episodes={"cheap-1": (False, True, True), "pricey-1": (False, True, True)},
    )
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "2", "--episodes", "3", "--yes")
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "cheap24-" in flat and "pricey24-" in flat  # 2 scored, 4 unscored, nothing thinner
    assert "DIFFERENT" not in flat
    assert _says(result.output, f"wmo optimize route fit {out} --kind knn")


def test_route_sweep_hands_off_when_every_candidate_lost_the_same_scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The judge throttles on ONE scenario, so every candidate loses that scenario and none of the
    # others. Coverage stays like-for-like on what is left, which IS a comparison: the handoff
    # stands, and the counts still show the loss.
    lost = f"task {_HELD_OUT_IDS[1]}"
    _patch_seams(monkeypatch, judge_fails_on=frozenset({lost}))
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "3", "--yes")
    assert result.exit_code == 0, result.output
    matrix = OutcomeMatrix.load(out)
    unscored = [o for o in matrix.outcomes if not o.scored]
    assert {o.model for o in unscored} == {"cheap", "pricey"}
    assert {o.scenario_id for o in unscored} == {_HELD_OUT_IDS[1]}
    flat = _flat(result.output)
    assert f"cheap21{_HELD_OUT_IDS[1]}" in flat  # 2 scored, 1 unscored, and which one it lost
    assert "DIFFERENTscenarios" not in flat
    assert _says(result.output, f"wmo optimize route fit {out} --kind knn")


def test_route_sweep_declining_the_confirmation_spends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interactive_stdin: None
) -> None:
    seams = _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    monkeypatch.setattr(route_sweep_module, "_console", Console(force_terminal=True))
    monkeypatch.setattr(consent_module, "Confirm", _Answer(False))
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "3")
    assert result.exit_code == 0, result.output
    assert not out.exists()  # nothing written
    # The pre-flight DID construct both candidates (that is how an unusable backend becomes a usage
    # error before the cost question), but construction is side-effect free: what proves nothing
    # was spent is that no candidate was ever CALLED and no episode ever opened.
    assert seams.built_providers == ["cheap-1", "pricey-1"]
    assert seams.systems == []
    assert seams.world_model.tasks == []


def test_route_sweep_confirming_at_a_tty_runs_the_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interactive_stdin: None
) -> None:
    seams = _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    monkeypatch.setattr(route_sweep_module, "_console", Console(force_terminal=True))
    monkeypatch.setattr(consent_module, "Confirm", _Answer(True))
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "1")
    assert result.exit_code == 0, result.output
    assert len(OutcomeMatrix.load(out).outcomes) == 2
    # Both candidates constructed twice: once by the pre-flight (before the cost question) and
    # once per cell by `evaluate_pool`, which still owns per-cell provider state.
    assert seams.built_providers == ["cheap-1", "pricey-1", "cheap-1", "pricey-1"]


def test_route_sweep_non_interactive_without_yes_refuses_to_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No TTY to prompt at and no --yes: consent is said, never inferred. This branch used to
    # proceed-and-note; the equivalent branch in `optimize model` spent a scripted caller's
    # real money, so every spend surface now refuses (exit 2) and names the fix.
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "1")
    assert result.exit_code == 2, result.output
    assert _says(result.output, "cannot ask for spend consent")
    assert not Path(out).exists()  # nothing bought


def test_route_sweep_cost_estimate_states_its_assumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    _out, result = _sweep(
        tmp_path,
        root,
        "support",
        "--scenarios",
        "3",
        "--max-steps",
        "20",
        "--assume-input-tokens",
        "2000",
        "--assume-output-tokens",
        "250",
        "--yes",
    )
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    # 3 scenarios x 1 episode x 20 calls = 60 calls; cheap = 60 x (2000x1 + 250x2)/1e6 = $0.15,
    # pricey = 10x that, so the projected total is $1.65.
    assert "0.15" in flat and "1.50" in flat and "1.65" in flat
    assert "ASSUMED" in flat and "ASSUMPTION" in flat
    # The world model's own meter is excluded, and the table says so in plain words (no internal
    # decision id: "D12" means nothing to an operator reading a cost table).
    assert _says(result.output, "meteredseparatelyandareNOTinthisfigure")
    assert "D12" not in flat
    # ... and the measured spend is reported separately from the projection.
    assert "measuredcandidatespend" in flat


def test_route_sweep_rejects_a_missing_pool_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    out = tmp_path / "matrix.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--root",
            str(root),
            "--pool",
            str(tmp_path / "nope.toml"),
            "--out",
            str(out),
            "--yes",
        ],
    )
    assert result.exit_code != 0
    flat = _flat(result.output)
    assert "nope.toml" in flat  # names the file it wanted
    assert "[[model]]" in flat and "input_per_mtok" in flat  # and how to write one
    assert not out.exists()


def test_route_sweep_rejects_a_missing_immutable_task_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=None)
    out, result = _sweep(tmp_path, root, "support", "--yes")
    assert result.exit_code != 0
    flat = _flat(result.output)
    assert "noimmutabletaskset" in flat
    assert "wmobuild" in flat
    assert not out.exists()


def test_route_sweep_rejects_an_unbuilt_world_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "ghost", "--yes")
    assert result.exit_code != 0
    flat = _flat(result.output)
    assert "ghost" in flat and "wmobuild" in flat  # names the model and the command that fixes it
    assert not out.exists()


def test_route_sweep_keeps_the_immutable_held_out_partition_for_small_task_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Even a small task set has an explicit held-out partition. The sweep must not recreate a
    # split from its source traces or fall back to fit tasks.
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus(3))
    out, result = _sweep(tmp_path, root, "--scenarios", "2", "--yes")  # default model resolution
    assert result.exit_code == 0, result.output
    assert "notleak-free" not in _flat(result.output)
    # The direct fixture marks this stable task-ID prefix held out.
    assert OutcomeMatrix.load(out).scenario_ids() == ["tr-000", "tr-001"]


def test_route_sweep_names_the_positional_when_the_model_is_ambiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `WorldModelStore.resolve` says "pass --name", the option `wmo serve`/`play`/`demo` carry.
    # This command takes the model positionally, so following that advice fails; the message has
    # to name what a user of THIS command types.
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    save_config(
        HarnessConfig(
            providers=[ProviderConfig(kind=ProviderKind.ANTHROPIC, model="fake-serve")],
            serve_provider=ProviderKind.ANTHROPIC,
        ),
        root / "models" / "other",
    )
    out, result = _sweep(tmp_path, root, "--yes")  # no MODEL, two models built
    assert result.exit_code != 0
    flat = _flat(result.output)
    assert "MODEL" in flat and "other,support" in flat
    assert "--name" not in flat
    assert "wmooptimizeroutesweepother" in flat  # a command that actually works
    assert not out.exists()
