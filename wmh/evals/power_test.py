"""Tests for locked paired-simulation manifests, artifacts, and power gates."""

from __future__ import annotations

import math
from decimal import ROUND_CEILING, Decimal, localcontext
from pathlib import Path

import pytest
from pydantic import ValidationError
from scipy.stats import beta

from wmh.evals.paired import (
    BoundedMeanBet,
    PairedEvaluationDesign,
    PairedPanelPlan,
    PairedTaskPlan,
)
from wmh.evals.power import (
    PairedPowerDependenceManifest,
    PairedPowerDgpManifest,
    PairedPowerEffectAtom,
    PairedPowerEffectShapeManifest,
    PairedPowerGateDesign,
    PairedPowerGateReport,
    PairedPowerLaneBaseline,
    PairedPowerReplicationManifest,
    PairedPowerScenarioManifest,
    PairedPowerSeedManifest,
    PairedPowerSimulationManifest,
    PairedPowerTaskProfileEntry,
    PairedPowerTaskProfileManifest,
    PairedPowerTrial,
    PairedPowerTrialArtifact,
    _binomial_lower_bound,
    _binomial_upper_bound,
    evaluate_paired_power_gate,
    load_paired_power_trial_artifact,
    merge_paired_power_chunks,
    resume_paired_power_simulation,
    run_paired_power_chunk,
    write_paired_power_chunk,
    write_paired_power_trial_artifact,
)

_SIMULATION_DIGEST = "sha256:" + "a" * 64
_PAIRED_DESIGN_DIGEST = "sha256:" + "c" * 64
_CANONICAL_TRIAL_EVIDENCE_DIGEST = (
    "sha256:a51e821fb8960fb64d485c9cf8f937dddd94c3bc216a7d3e01f602bfbb87d568"
)


def _design() -> PairedPowerGateDesign:
    # These fixture thresholds exercise the gate only. They are not the study's
    # target MDE or power claim; those must arrive in the locked simulation design.
    return PairedPowerGateDesign(
        simulation_design_digest=_SIMULATION_DIGEST,
        paired_evaluation_design_digest=_PAIRED_DESIGN_DIGEST,
        target_effect=0.2,
        maximum_type_i_error=0.05,
        minimum_power=0.9,
        monte_carlo_alpha=0.01,
        replications_per_scenario=100,
    )


def _synthetic_evaluation_design() -> PairedEvaluationDesign:
    """Return a tiny non-study design containing no benchmark identities."""
    return PairedEvaluationDesign.create(
        tasks=tuple(
            PairedTaskPlan(
                task_id=f"synthetic-task-{index}",
                group_id="synthetic-family" if index < 2 else f"synthetic-group-{index}",
            )
            for index in range(4)
        ),
        panel=tuple(
            PairedPanelPlan(panel_member=lane, attempts=2)
            for lane in ("lane-a", "lane-b", "lane-c")
        ),
        primary_e_value_bets=(
            BoundedMeanBet(fraction=0.25, weight=1 / 16),
            BoundedMeanBet(fraction=0.5, weight=1 / 16),
            BoundedMeanBet(fraction=1.0, weight=7 / 8),
        ),
        schedule_seed="synthetic-schedule",
        analysis_seed="synthetic-analysis",
        randomization_samples=999,
        minimum_equal_task_member_delta=0.03,
        noninferiority_margin=0.02,
    )


def _synthetic_task_profile() -> PairedPowerTaskProfileManifest:
    """Return fixed synthetic rates for schema and resume tests, not a study profile."""
    return PairedPowerTaskProfileManifest(
        tasks=tuple(
            PairedPowerTaskProfileEntry(
                task_id=f"synthetic-task-{index}",
                stratum="synthetic-easy" if index == 0 else "synthetic-hard",
                group_id="synthetic-family" if index < 2 else f"synthetic-group-{index}",
                lane_baselines=tuple(
                    PairedPowerLaneBaseline(panel_member=lane, probability=probability)
                    for lane, probability in (
                        ("lane-a", 0.25 + index * 0.05),
                        ("lane-b", 0.35 + index * 0.05),
                        ("lane-c", 0.45 + index * 0.05),
                    )
                ),
            )
            for index in range(4)
        )
    )


def _synthetic_simulation_manifest(
    design: PairedEvaluationDesign,
    profile: PairedPowerTaskProfileManifest,
) -> PairedPowerSimulationManifest:
    """Freeze a tiny deterministic simulator contract that is explicitly non-study."""
    return PairedPowerSimulationManifest.create(
        evaluation_design=design,
        task_profile=profile,
        dgp=PairedPowerDgpManifest(
            effect_shape=PairedPowerEffectShapeManifest(
                atoms=(
                    PairedPowerEffectAtom(multiplier=-0.5, probability=0.2),
                    PairedPowerEffectAtom(multiplier=1.375, probability=0.8),
                )
            ),
            dependence=PairedPowerDependenceManifest(residual_attempt_intraclass_correlation=0.0),
        ),
        seeds=PairedPowerSeedManifest(root_seed="synthetic-root-seed-v1"),
        replications=PairedPowerReplicationManifest(
            replications_per_scenario=6,
            chunk_size=2,
        ),
        scenarios=(
            PairedPowerScenarioManifest(scenario="weak-null", equal_task_effect=0.0),
            PairedPowerScenarioManifest(
                scenario="target-alternative",
                equal_task_effect=0.1,
            ),
        ),
    )


def _trials(*, null_rejections: int, target_rejections: int) -> tuple[PairedPowerTrial, ...]:
    return tuple(
        PairedPowerTrial(
            simulation_design_digest=_SIMULATION_DIGEST,
            paired_evaluation_design_digest=_PAIRED_DESIGN_DIGEST,
            scenario=scenario,
            replicate=replicate,
            primary_passed=(
                replicate <= null_rejections
                if scenario == "weak-null"
                else replicate <= target_rejections
            ),
        )
        for scenario in ("weak-null", "target-alternative")
        for replicate in range(1, 101)
    )


def test_strong_locked_simulation_fixture_passes_both_power_gates() -> None:
    design = _design()
    report = evaluate_paired_power_gate(
        design,
        _trials(null_rejections=0, target_rejections=100),
    )

    assert report.design == design
    assert report.design.paired_evaluation_design_digest == _PAIRED_DESIGN_DIGEST
    assert report.trial_evidence_digest == _CANONICAL_TRIAL_EVIDENCE_DIGEST
    assert report.digest == report.report_digest
    assert report.empirical_type_i_error == 0.0
    assert report.type_i_error_upper_bound < design.maximum_type_i_error
    assert report.empirical_power == 1.0
    assert report.power_lower_bound > design.minimum_power
    assert report.type_i_error_passed is True
    assert report.power_passed is True
    assert report.passed is True
    assert PairedPowerGateReport.model_validate_json(report.model_dump_json()) == report
    assert (
        evaluate_paired_power_gate(
            design,
            tuple(reversed(_trials(null_rejections=0, target_rejections=100))),
        ).trial_evidence_digest
        == report.trial_evidence_digest
    )


def test_weak_locked_simulation_fixture_fails_without_a_power_claim() -> None:
    design = _design()
    report = evaluate_paired_power_gate(
        design,
        _trials(null_rejections=10, target_rejections=70),
    )

    assert report.type_i_error_upper_bound > design.maximum_type_i_error
    assert report.power_lower_bound < design.minimum_power
    assert report.type_i_error_passed is False
    assert report.power_passed is False
    assert report.passed is False


def test_power_gate_requires_the_exact_locked_replication_matrix() -> None:
    design = _design()
    trials = _trials(null_rejections=0, target_rejections=100)

    with pytest.raises(ValueError, match="exactly fill"):
        evaluate_paired_power_gate(design, trials[:-1])
    with pytest.raises(ValueError, match="duplicate replicate"):
        evaluate_paired_power_gate(design, (*trials, trials[0]))
    with pytest.raises(ValueError, match="digest differs"):
        evaluate_paired_power_gate(
            design,
            (
                trials[0].model_copy(update={"simulation_design_digest": "sha256:" + "b" * 64}),
                *trials[1:],
            ),
        )
    with pytest.raises(ValueError, match="evaluation design digest differs"):
        evaluate_paired_power_gate(
            design,
            (
                trials[0].model_copy(
                    update={"paired_evaluation_design_digest": "sha256:" + "d" * 64}
                ),
                *trials[1:],
            ),
        )


def test_clopper_pearson_bounds_round_outward_at_exact_boundaries() -> None:
    trials = 100
    alpha = 0.01
    raw_upper = float(beta.ppf(1.0 - alpha, 1, trials))
    upper = _binomial_upper_bound(0, trials, alpha=alpha)
    raw_lower = float(beta.ppf(alpha, trials, 1))
    lower = _binomial_lower_bound(trials, trials, alpha=alpha)

    assert upper == math.nextafter(raw_upper, math.inf)
    assert lower == math.nextafter(raw_lower, -math.inf)
    with localcontext() as context:
        context.prec = 80
        exact_alpha = Decimal.from_float(alpha)
        exact_lower = exact_alpha ** (Decimal(1) / Decimal(trials))
        exact_upper = Decimal(1) - exact_lower
        assert Decimal.from_float(upper) >= exact_upper
        assert Decimal.from_float(lower) <= exact_lower
    assert upper < 0.05
    assert _binomial_lower_bound(0, trials, alpha=alpha) == 0.0
    assert _binomial_upper_bound(trials, trials, alpha=alpha) == 1.0


def test_clopper_pearson_upper_moves_past_the_one_ulp_regression() -> None:
    alpha = 0.05
    raw = float(beta.ppf(1.0 - alpha, 2, 2))
    one_ulp = math.nextafter(raw, math.inf)
    certified = _binomial_upper_bound(1, 3, alpha=alpha)

    assert certified > one_ulp
    assert _independent_binomial_tail_upper(
        1,
        3,
        probability=certified,
        lower_tail=True,
    ) <= Decimal.from_float(alpha)


def test_clopper_pearson_exhaustive_independent_root_grid_through_200() -> None:
    alpha = 0.05
    threshold = Decimal.from_float(alpha)
    for trials in range(1, 201):
        for successes in range(trials):
            upper = _binomial_upper_bound(successes, trials, alpha=alpha)
            assert (
                _independent_binomial_tail_upper(
                    successes,
                    trials,
                    probability=upper,
                    lower_tail=True,
                )
                <= threshold
            ), ("upper", successes, trials, upper)
        for successes in range(1, trials + 1):
            lower = _binomial_lower_bound(successes, trials, alpha=alpha)
            assert (
                _independent_binomial_tail_upper(
                    successes,
                    trials,
                    probability=lower,
                    lower_tail=False,
                )
                <= threshold
            ), ("lower", successes, trials, lower)


def test_power_report_rejects_mutated_derived_or_bound_evidence() -> None:
    report = evaluate_paired_power_gate(
        _design(),
        _trials(null_rejections=0, target_rejections=100),
    )

    def reject(update: dict[str, object], match: str) -> None:
        payload = report.model_dump(mode="json")
        payload.update(update)
        with pytest.raises(ValidationError, match=match):
            PairedPowerGateReport.model_validate(payload)

    reject({"null_rejections": 101}, "cannot exceed frozen replications")
    reject({"target_rejections": 99}, "empirical_power differs")
    reject({"empirical_type_i_error": 0.01}, "empirical_type_i_error differs")
    reject({"empirical_power": 0.99}, "empirical_power differs")
    reject({"type_i_error_upper_bound": 0.0}, "type_i_error_upper_bound differs")
    reject({"power_lower_bound": 0.0}, "power_lower_bound differs")
    reject({"type_i_error_passed": False}, "type_i_error_passed differs")
    reject({"power_passed": False}, "power_passed differs")

    changed_design = report.design.model_copy(update={"target_effect": 0.3})
    reject({"design": changed_design.model_dump(mode="json")}, "report digest differs")
    reject({"trial_evidence_digest": "sha256:" + "e" * 64}, "report digest differs")
    reject({"report_digest": "sha256:" + "f" * 64}, "report digest differs")


def _independent_binomial_tail_upper(
    successes: int,
    trials: int,
    *,
    probability: float,
    lower_tail: bool,
) -> Decimal:
    """Direct positive-term Decimal sum independent of the production recurrence."""
    if probability <= 0.0:
        return Decimal(1 if lower_tail else 0)
    if probability >= 1.0:
        return Decimal(0 if lower_tail and successes < trials else 1)
    probability_decimal = Decimal.from_float(probability)
    value_tuple = probability_decimal.as_tuple()
    assert isinstance(value_tuple.exponent, int)
    precision = max(180, 1 - value_tuple.exponent + 12)
    with localcontext() as context:
        context.prec = precision
        context.rounding = ROUND_CEILING
        one = Decimal(1)
        complement = context.subtract(one, probability_decimal)
        assert context.add(probability_decimal, complement) == one
        observed_values = range(successes + 1) if lower_tail else range(successes, trials + 1)
        total = Decimal(0)
        for observed in observed_values:
            term = context.multiply(
                Decimal(math.comb(trials, observed)),
                context.multiply(
                    context.power(probability_decimal, observed),
                    context.power(complement, trials - observed),
                ),
            )
            total = context.add(total, term)
        return +total


def test_simulation_manifest_binds_private_profile_and_exact_primary_matrix() -> None:
    evaluation_design = _synthetic_evaluation_design()
    profile = _synthetic_task_profile()
    manifest = _synthetic_simulation_manifest(evaluation_design, profile)

    assert manifest.paired_evaluation_design_digest == evaluation_design.digest
    assert manifest.task_profile_digest == profile.digest
    assert manifest.task_strata_group_metadata_digest == profile.metadata_digest
    assert manifest.task_count == 4
    assert manifest.lane_set == ("lane-a", "lane-b", "lane-c")
    assert {item.panel_member: item.attempts for item in manifest.attempts_by_lane} == {
        "lane-a": 2,
        "lane-b": 2,
        "lane-c": 2,
    }
    assert manifest.primary_e_value_bets == evaluation_design.primary_e_value_bets
    assert manifest.minimum_equal_task_member_delta == 0.03
    manifest.validate_frozen_inputs(evaluation_design, profile)

    changed_profile = profile.model_copy(
        update={
            "tasks": (
                profile.tasks[0].model_copy(
                    update={
                        "lane_baselines": (
                            profile.tasks[0]
                            .lane_baselines[0]
                            .model_copy(update={"probability": 0.2}),
                            *profile.tasks[0].lane_baselines[1:],
                        )
                    }
                ),
                *profile.tasks[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="task profile digest"):
        manifest.validate_frozen_inputs(evaluation_design, changed_profile)


def test_locked_chunk_is_deterministic_and_artifact_is_public_safe(tmp_path: Path) -> None:
    evaluation_design = _synthetic_evaluation_design()
    profile = _synthetic_task_profile()
    manifest = _synthetic_simulation_manifest(evaluation_design, profile)

    first = run_paired_power_chunk(
        manifest,
        evaluation_design,
        profile,
        scenario="target-alternative",
        first_replicate=1,
        last_replicate=2,
    )
    repeated = run_paired_power_chunk(
        manifest,
        evaluation_design,
        profile,
        scenario="target-alternative",
        first_replicate=1,
        last_replicate=2,
    )
    assert first == repeated
    assert first.digest == repeated.digest

    chunks = tuple(
        run_paired_power_chunk(
            manifest,
            evaluation_design,
            profile,
            scenario=scenario,
            first_replicate=first_replicate,
            last_replicate=last_replicate,
        )
        for scenario in ("weak-null", "target-alternative")
        for first_replicate, last_replicate in ((1, 2), (3, 4), (5, 6))
    )
    artifact = merge_paired_power_chunks(manifest, chunks)
    assert isinstance(artifact, PairedPowerTrialArtifact)
    assert artifact.replications_per_scenario == 6

    artifact_path = tmp_path / "synthetic-power-artifact.json"
    write_paired_power_trial_artifact(artifact_path, artifact)
    loaded = load_paired_power_trial_artifact(artifact_path)
    assert loaded == artifact
    serialized = artifact_path.read_text()
    assert "synthetic-task" not in serialized
    assert "synthetic-family" not in serialized

    gate = PairedPowerGateDesign(
        simulation_design_digest=manifest.digest,
        paired_evaluation_design_digest=evaluation_design.digest,
        target_effect=0.1,
        maximum_type_i_error=0.9,
        minimum_power=0.01,
        monte_carlo_alpha=0.1,
        replications_per_scenario=6,
    )
    report = evaluate_paired_power_gate(gate, loaded)
    expanded_trials = tuple(
        PairedPowerTrial(
            simulation_design_digest=artifact.simulation_design_digest,
            paired_evaluation_design_digest=artifact.paired_evaluation_design_digest,
            scenario=scenario.scenario,
            replicate=replicate,
            primary_passed=primary_passed,
        )
        for scenario in artifact.scenarios
        for replicate, primary_passed in enumerate(scenario.decisions.decisions(), start=1)
    )
    expanded_report = evaluate_paired_power_gate(gate, expanded_trials)
    assert report.null_rejections == artifact.rejection_count("weak-null")
    assert report.target_rejections == artifact.rejection_count("target-alternative")
    assert report.trial_evidence_digest == artifact.trial_evidence_digest
    assert report.trial_evidence_digest == expanded_report.trial_evidence_digest
    assert report.digest == expanded_report.digest
    with pytest.raises(ValueError, match="target effect differs"):
        evaluate_paired_power_gate(gate.model_copy(update={"target_effect": 0.2}), loaded)


def test_chunk_merge_rejects_missing_duplicate_and_drifted_chunks() -> None:
    evaluation_design = _synthetic_evaluation_design()
    profile = _synthetic_task_profile()
    manifest = _synthetic_simulation_manifest(evaluation_design, profile)
    chunks = tuple(
        run_paired_power_chunk(
            manifest,
            evaluation_design,
            profile,
            scenario=scenario,
            first_replicate=first_replicate,
            last_replicate=last_replicate,
        )
        for scenario in ("weak-null", "target-alternative")
        for first_replicate, last_replicate in ((1, 2), (3, 4), (5, 6))
    )

    with pytest.raises(ValueError, match="exactly fill"):
        merge_paired_power_chunks(manifest, chunks[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        merge_paired_power_chunks(manifest, (*chunks, chunks[0]))
    with pytest.raises(ValueError, match="simulation design digest"):
        merge_paired_power_chunks(
            manifest,
            (
                chunks[0].model_copy(update={"simulation_design_digest": "sha256:" + "f" * 64}),
                *chunks[1:],
            ),
        )


def test_resume_reuses_valid_chunks_and_rejects_corruption(tmp_path: Path) -> None:
    evaluation_design = _synthetic_evaluation_design()
    profile = _synthetic_task_profile()
    manifest = _synthetic_simulation_manifest(evaluation_design, profile)
    completed = run_paired_power_chunk(
        manifest,
        evaluation_design,
        profile,
        scenario="weak-null",
        first_replicate=1,
        last_replicate=2,
    )
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    first_path = chunk_dir / "weak-null-000000001-000000002.json"
    write_paired_power_chunk(first_path, completed)
    original_stat = first_path.stat()

    artifact = resume_paired_power_simulation(
        manifest,
        evaluation_design,
        profile,
        chunk_dir,
    )
    assert artifact.replications_per_scenario == 6
    assert first_path.stat().st_mtime_ns == original_stat.st_mtime_ns

    first_path.write_text(first_path.read_text().replace("sha256:", "sha256:0", 1))
    with pytest.raises(ValueError, match="digest|artifact"):
        resume_paired_power_simulation(
            manifest,
            evaluation_design,
            profile,
            chunk_dir,
        )


def test_resume_rejects_foreign_json_but_tolerates_crash_temporary(tmp_path: Path) -> None:
    evaluation_design = _synthetic_evaluation_design()
    profile = _synthetic_task_profile()
    manifest = _synthetic_simulation_manifest(evaluation_design, profile)
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    crash_temporary = chunk_dir / ".weak-null-000000001-000000002.json.crash.tmp"
    crash_temporary.write_text("incomplete")

    artifact = resume_paired_power_simulation(
        manifest,
        evaluation_design,
        profile,
        chunk_dir,
    )
    assert artifact.replications_per_scenario == 6
    assert crash_temporary.read_text() == "incomplete"

    (chunk_dir / "foreign.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected JSON"):
        resume_paired_power_simulation(
            manifest,
            evaluation_design,
            profile,
            chunk_dir,
        )


def test_resume_rejects_drift_before_writing_missing_chunks(tmp_path: Path) -> None:
    evaluation_design = _synthetic_evaluation_design()
    profile = _synthetic_task_profile()
    manifest = _synthetic_simulation_manifest(evaluation_design, profile)
    drifted = run_paired_power_chunk(
        manifest,
        evaluation_design,
        profile,
        scenario="target-alternative",
        first_replicate=5,
        last_replicate=6,
    ).model_copy(update={"simulation_design_digest": "sha256:" + "f" * 64})
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    write_paired_power_chunk(
        chunk_dir / "target-alternative-000000005-000000006.json",
        drifted,
    )

    with pytest.raises(ValueError, match="simulation digest"):
        resume_paired_power_simulation(
            manifest,
            evaluation_design,
            profile,
            chunk_dir,
        )
    assert not (chunk_dir / "weak-null-000000001-000000002.json").exists()
