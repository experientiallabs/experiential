"""Route fit and tuning CLI tests."""

# ruff: noqa: F403, F405
from wmo.cli.route_fixtures_test import *


def test_route_fit_and_report(tmp_path: Path) -> None:
    matrix_file = _matrix_file(tmp_path)
    policy_file = tmp_path / "policy.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(matrix_file),
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
    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(policy_file)
    assert policy.kind == "rank"
    assert len(policy.clusters) == 2

    report_file = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "report",
            str(matrix_file),
            str(policy_file),
            "--baseline",
            "a",
            "--out",
            str(report_file),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(report_file.read_text())
    assert report["baseline"]["model_id"] == "a"
    assert set(policy.fit_scenario_ids).isdisjoint(report["scenario_ids"])
    assert len(policy.fit_scenario_ids) + report["scenario_count"] == 4
    assert report["cost_assumptions"]


def test_the_curve_defaults_to_the_world_model_labels(tmp_path: Path) -> None:
    # Unchanged behavior for every existing caller: a sweep's matrix was scored by the world
    # model's own verifier.
    result, pareto = _fit_then_report(tmp_path)
    assert result.exit_code == 0, result.output
    curve = json.loads(pareto.read_text())
    assert curve["provenance"] == "wm_simulated"
    assert curve["judge"] == "world-model verifier"


def test_a_real_benchmark_matrix_can_label_its_own_curve(tmp_path: Path) -> None:
    # The bench-defaults case: the rewards are real tau2 episodes, and a curve claiming they came
    # out of a world model would present a measurement as a simulation. ParetoCurve.provenance
    # exists to stop exactly that, and until now the CLI hardcoded it.
    result, pareto = _fit_then_report(
        tmp_path, "--provenance", "real_episode", "--judge", "tau2 reward"
    )
    assert result.exit_code == 0, result.output
    curve = json.loads(pareto.read_text())
    assert curve["provenance"] == "real_episode"
    assert curve["judge"] == "tau2 reward"


def test_a_misspelled_provenance_is_refused_not_written(tmp_path: Path) -> None:
    result, pareto = _fit_then_report(tmp_path, "--provenance", "real")
    assert result.exit_code != 0
    assert "real_episode" in result.output
    assert not pareto.exists()


def test_the_report_label_defaults_to_the_world_model_phrasing(tmp_path: Path) -> None:
    result, pareto = _fit_then_report(tmp_path)
    assert result.exit_code == 0, result.output
    report = json.loads((pareto.parent / "report.json").read_text())
    assert "reconstructed from your traces" in report["scenario_label"]


def test_a_real_benchmark_report_can_say_what_it_measured(tmp_path: Path) -> None:
    # scenario_label is the one line of the report a customer actually reads. Telling them their
    # endpoint was measured on scenarios "reconstructed from your traces" when it was measured on
    # a pinned public benchmark is false, and until now the phrasing was hardcoded.
    result, pareto = _fit_then_report(
        tmp_path, "--scenario-label", "on the 20 pinned tau2-bench eval tasks"
    )
    assert result.exit_code == 0, result.output
    report = json.loads((pareto.parent / "report.json").read_text())
    assert report["scenario_label"] == "on the 20 pinned tau2-bench eval tasks"


def test_route_fit_rejects_unknown_embedder(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["optimize", "route", "fit", str(_matrix_file(tmp_path)), "--embedder", "vibes"],
    )
    assert result.exit_code != 0
    assert "hashing, azure or local" in result.output


def test_route_fit_knn_writes_policy_and_sidecar(tmp_path: Path) -> None:
    matrix_file = _knn_matrix_file(tmp_path)
    policy_file = tmp_path / "policy.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(matrix_file),
            "--kind",
            "knn",
            "--fallback",
            "a",
            "--z",
            "0.5",
            "--rag-num",
            "3",
            "--min-pairs",
            "2",
            "--out",
            str(policy_file),
        ],
    )
    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(policy_file)
    assert policy.kind == "knn"
    assert policy.default_model == "a" == policy.guard_model  # the pinned fallback
    # The sidecar is named after --out and recorded in the policy, not resolved by convention.
    assert policy.knn_bank_path == "policy.json.bank.npz"
    assert policy.bank_path() == tmp_path / "policy.json.bank.npz"
    assert policy.bank_path().is_file()  # sidecar beside the policy
    assert len(policy.knn_bank().scenario_ids) == 8
    assert policy.fit_scenario_ids == policy.knn_bank().scenario_ids
    assert "routed away from the fallback" in result.output
    # The prose neighborhoods carry unanimous evidence for b, so that traffic leaves the
    # fallback while the SQL half stays on it.
    matrix = OutcomeMatrix.load(matrix_file)
    prose_ids = [sid for sid in matrix.scenario_ids() if sid.startswith("prose:")]
    assert evaluate_policy(policy, matrix, prose_ids).model_mix == {"b": 1.0}


def test_route_fit_knn_rejects_the_rank_only_cost_knob(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_matrix_file(tmp_path)),
            "--kind",
            "knn",
            "--cost-weight",
            "0.5",
        ],
    )
    assert result.exit_code != 0
    # The message points at the knn cost control that does exist, not just at what is wrong.
    assert "--cost-quality" in result.output


def test_route_fit_rejects_unknown_kind(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["optimize", "route", "fit", str(_matrix_file(tmp_path)), "--kind", "vibes"]
    )
    assert result.exit_code != 0
    assert "knn or rank" in result.output


def test_route_tune_sets_the_dial_and_keeps_the_policy_as_fitted(tmp_path: Path) -> None:
    policy_file = _fitted_knn_policy(tmp_path)
    fitted = RoutingPolicy.load(policy_file)
    result = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.6"]
    )
    assert result.exit_code == 0, result.output
    tuned = RoutingPolicy.load(policy_file)
    assert tuned.cost_quality == 0.6
    assert tuned.pick_lam > 0.0
    assert tuned.guard_mode == "asymmetric"
    # The un-tuned artifact is preserved, so the dial is always re-appliable from the fit.
    base = RoutingPolicy.load(tmp_path / "policy.base.json")
    assert base.model_dump() == fitted.model_dump()
    # The printed anchor table is how an operator learns what the position measured.
    assert "cost_quality=0.6" in result.output
    assert "-46.2%" in result.output


def test_route_tune_twice_equals_tuning_once_from_the_base(tmp_path: Path) -> None:
    policy_file = _fitted_knn_policy(tmp_path)
    runner.invoke(app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "1.0"])
    once = RoutingPolicy.load(policy_file).model_dump()
    for _ in range(2):
        result = runner.invoke(
            app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "1.0"]
        )
        assert result.exit_code == 0, result.output
    assert RoutingPolicy.load(policy_file).model_dump() == once
    # Sliding back down lands exactly where a first-time slide to that position would.
    runner.invoke(app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.25"])
    balanced = RoutingPolicy.load(policy_file)
    assert balanced.cost_quality == 0.25
    assert (balanced.pick_lam, balanced.guard_mode) == (0.0, "symmetric")


def test_route_tune_still_routes_after_the_dial_moves(tmp_path: Path) -> None:
    # The dial must leave a servable policy: same bank, same baseline, still routing.
    matrix_file = _knn_matrix_file(tmp_path)
    policy_file = _fitted_knn_policy(tmp_path)
    runner.invoke(app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "1.0"])
    tuned = RoutingPolicy.load(policy_file)
    matrix = OutcomeMatrix.load(matrix_file)
    prose_ids = [sid for sid in matrix.scenario_ids() if sid.startswith("prose:")]
    assert evaluate_policy(tuned, matrix, prose_ids).model_mix == {"b": 1.0}


def test_route_tune_rejects_a_policy_kind_without_a_dial(tmp_path: Path) -> None:
    policy_file = tmp_path / POLICY_FILENAME
    fit = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_matrix_file(tmp_path)),
            "--kind",
            "rank",
            "--out",
            str(policy_file),
        ],
    )
    assert fit.exit_code == 0, fit.output
    result = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.5"]
    )
    assert result.exit_code != 0
    assert "kind='rank'" in result.output


def test_route_fit_knn_gives_each_policy_its_own_evidence_bank(tmp_path: Path) -> None:
    """Two knn fits into one directory must not share (and overwrite) one sidecar.

    Regression: the bank name used to be hard-coded, so the second fit clobbered the first
    policy's evidence and both policies recorded the same relative path. Policy A then served
    matrix B's rewards, which inverts every routing decision on this pair of matrices.
    """
    a_matrix = _knn_matrix_file(tmp_path, name="matrix_a.json")
    b_matrix = _knn_matrix_file(tmp_path, flip=True, name="matrix_b.json")
    # The third fit shares a STEM with the first: the bank name is appended to the policy
    # filename rather than substituted for its extension, so it still gets its own sidecar.
    fits = (
        ("policy_a.json", a_matrix, "a"),
        ("policy_b.json", b_matrix, "b"),
        ("policy_a.yaml", b_matrix, "b"),
    )
    for name, matrix_file, fallback in fits:
        result = _fit_knn(matrix_file, tmp_path / name, fallback=fallback)
        assert result.exit_code == 0, result.output

    policies = [RoutingPolicy.load(tmp_path / name) for name, _, _ in fits]
    assert [policy.knn_bank_path for policy in policies] == [
        "policy_a.json.bank.npz",
        "policy_b.json.bank.npz",
        "policy_a.yaml.bank.npz",
    ]
    banks = [policy.bank_path() for policy in policies]
    assert len(set(banks)) == len(banks)
    assert all(bank.is_file() for bank in banks)
    # Policy A still routes on ITS evidence: prose is b's half of matrix_a, and matrix_b says
    # the opposite, so this is 1.0 only if the later fits left A's bank alone.
    matrix = OutcomeMatrix.load(a_matrix)
    prose_ids = [sid for sid in matrix.scenario_ids() if sid.startswith("prose:")]
    assert evaluate_policy(policies[0], matrix, prose_ids).model_mix == {"b": 1.0}


def test_route_tune_refuses_a_base_snapshot_from_a_superseded_fit(tmp_path: Path) -> None:
    """fit -> tune -> refit -> tune must not silently dial the pre-refit artifact.

    Regression: `tune` always re-read `<stem>.base.json`, which `fit` never invalidates, so the
    second tune reported success while overwriting the new fit with a dialed copy of the old one.
    """
    policy_file = _fitted_knn_policy(tmp_path)
    tuned = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.6"]
    )
    assert tuned.exit_code == 0, tuned.output

    refit = _fit_knn(
        _knn_matrix_file(tmp_path, flip=True, name="refit_matrix.json"),
        policy_file,
        fallback="b",
    )
    assert refit.exit_code == 0, refit.output
    assert RoutingPolicy.load(policy_file).default_model == "b"

    stale = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.3"]
    )
    assert stale.exit_code != 0
    assert _says(stale.output, "as-fitted snapshot of a different fit")
    assert _says(stale.output, "policy.base.json")  # names the file to delete
    # The refit survives untouched rather than being replaced by a dialed copy of the old fit.
    after = RoutingPolicy.load(policy_file)
    assert after.default_model == "b"
    assert after.cost_quality is None


def test_route_fit_digests_the_bytes_it_actually_fitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recorded digest must describe the matrix the fit SAW, not a later read of the path.

    Regression: `fit` parsed the matrix and then re-read the file to digest it. A corpus rebuilt
    in place between the two stamped the old fit with the new file's digest, so the next fit of
    that new matrix matched its provenance and `tune` accepted the superseded snapshot -- the
    very failure the digest exists to catch. The two now come out of one read.
    """
    matrix_file = _knn_matrix_file(tmp_path)
    fitted_bytes = matrix_file.read_bytes()
    real = {name: getattr(Path, name) for name in ("read_bytes", "read_text")}

    def _rebuilding(name: str):  # noqa: ANN202 - the wrapped reader's own signature
        def _read(self: Path, *args: object, **kwargs: object):  # noqa: ANN202
            payload = real[name](self, *args, **kwargs)
            if self == matrix_file:  # swap the corpus the instant the fit has read it
                monkeypatch.undo()  # ...once, whichever reader the fit happens to use
                _knn_matrix_file(tmp_path, flip=True)
            return payload

        return _read

    for name in real:
        monkeypatch.setattr(Path, name, _rebuilding(name))
    policy_file = tmp_path / POLICY_FILENAME
    assert _fit_knn(matrix_file, policy_file).exit_code == 0
    assert matrix_file.read_bytes() != fitted_bytes  # the file on disk did change under the fit

    # The digest is of the bytes that were fitted, so a later fit of the REPLACEMENT differs.
    fitted_from = RoutingPolicy.load(policy_file).fitted_from or ""
    other = tmp_path / "other.json"
    assert _fit_knn(matrix_file, other).exit_code == 0
    assert fitted_from != (RoutingPolicy.load(other).fitted_from or "")


def test_route_tune_refuses_a_snapshot_after_the_matrix_was_rebuilt_in_place(
    tmp_path: Path,
) -> None:
    """Same matrix path, same flags, different contents: still a different fit.

    A corpus is routinely rebuilt under the filename it already had, so a path alone cannot
    identify a fit. `fitted_from` carries a digest of the matrix, which is what makes the
    snapshot check catch this.
    """
    policy_file = _fitted_knn_policy(tmp_path)
    tuned = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.6"]
    )
    assert tuned.exit_code == 0, tuned.output

    rebuilt = _knn_matrix_file(tmp_path, flip=True)  # same default filename, opposite labels
    refit = _fit_knn(rebuilt, policy_file)  # and the same fit flags as `_fitted_knn_policy`
    assert refit.exit_code == 0, refit.output

    stale = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.3"]
    )
    assert stale.exit_code != 0
    assert _says(stale.output, "as-fitted snapshot of a different fit")
    assert RoutingPolicy.load(policy_file).cost_quality is None  # the refit is untouched


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows rejects '|' in path components used by this regression fixture",
)
def test_route_tune_survives_a_matrix_path_that_looks_like_a_dial_suffix(tmp_path: Path) -> None:
    """An operator-supplied path must not be able to truncate the fit identity it opens.

    Regression: `fit_provenance` split on the FIRST ` | cost_quality=`, and `fitted_from` starts
    with the matrix path. A path containing that substring therefore discarded the digest and
    every fit flag behind it, collapsing unrelated fits onto one identity, and `tune` dialed the
    superseded snapshot over the refit. This name carries a COMPLETE, well-formed dial suffix,
    which is the worst case: the fit flags follow the path, so the real suffix is still the only
    one at the end of the string.
    """
    hostile = "m | cost_quality=0.5 (floor_q=0.05, lam=0, guard=symmetric).json"
    policy_file = tmp_path / POLICY_FILENAME
    assert _fit_knn(_knn_matrix_file(tmp_path, name=hostile), policy_file).exit_code == 0
    for dial in ("0.6", "0.2"):  # no refit between these: the dial must still move freely
        tuned = runner.invoke(
            app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", dial]
        )
        assert tuned.exit_code == 0, tuned.output
    assert RoutingPolicy.load(policy_file).cost_quality == 0.2

    rebuilt = _knn_matrix_file(tmp_path, flip=True, name=hostile)  # same path, opposite labels
    assert _fit_knn(rebuilt, policy_file, fallback="b").exit_code == 0
    stale = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.3"]
    )
    assert stale.exit_code != 0
    assert _says(stale.output, "as-fitted snapshot of a different fit")
    after = RoutingPolicy.load(policy_file)
    assert after.default_model == "b"  # the refit survives, not a dialed copy of the old fit
    assert after.cost_quality is None


def test_route_tune_that_fails_leaves_no_base_snapshot_behind(tmp_path: Path) -> None:
    """A rejected tune must not poison the path for the next fit.

    Regression: the base snapshot was copied before validation, so a failed tune left a stray
    `policy.base.json`. A later `fit --kind knn` into the same path could never be tuned: the
    error reported kind='rank' while the policy on disk was demonstrably kind='knn'.
    """
    policy_file = tmp_path / POLICY_FILENAME
    fit = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_matrix_file(tmp_path)),
            "--kind",
            "rank",
            "--out",
            str(policy_file),
        ],
    )
    assert fit.exit_code == 0, fit.output
    rejected = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.5"]
    )
    assert rejected.exit_code != 0
    assert "kind='rank'" in rejected.output
    assert not (tmp_path / "policy.base.json").exists()

    # The path is still tunable once a knn policy is fitted into it.
    refit = _fit_knn(_knn_matrix_file(tmp_path), policy_file)
    assert refit.exit_code == 0, refit.output
    tuned = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.5"]
    )
    assert tuned.exit_code == 0, tuned.output
    assert RoutingPolicy.load(policy_file).cost_quality == 0.5


def test_route_tune_rejects_a_missing_policy_file(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["optimize", "route", "tune", str(tmp_path / "nope.json"), "--cost-quality", "0.5"]
    )
    assert result.exit_code != 0
    assert "no policy file" in result.output


def test_route_tune_rejects_a_dial_outside_the_range(tmp_path: Path) -> None:
    policy_file = _fitted_knn_policy(tmp_path)
    result = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "2"]
    )
    assert result.exit_code != 0
