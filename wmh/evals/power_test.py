"""Tests for locked paired-simulation manifests, artifacts, and power gates."""

from __future__ import annotations

import base64
import hashlib
import math
import os
import shutil
import threading
from decimal import ROUND_CEILING, Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.stats import beta

import wmh.evals.power as power_analysis
from wmh.evals.paired import (
    BoundedMeanBet,
    PairedEvaluationDesign,
    PairedPanelPlan,
    PairedTaskPlan,
    paired_member_primary_decisions,
    paired_primary_decision_passed,
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

_NULL_CONFIGURATIONS = ("lane-a", "lane-b", "lane-c")
_CANONICAL_TRIAL_EVIDENCE_DIGEST = (
    "sha256:4a0e77114b423adcfc4d5282129847c89ead60bbfbcf542b94c451116d0a6b8d"
)


def _design() -> PairedPowerGateDesign:
    # These fixture thresholds exercise the gate only. They are not the study's
    # target MDE or power claim; those must arrive in the locked simulation design.
    evaluation_design = _synthetic_evaluation_design()
    profile = _synthetic_task_profile()
    manifest = _synthetic_simulation_manifest(
        evaluation_design,
        profile,
        replications_per_scenario=200,
        chunk_size=200,
        target_effect=0.2,
    )
    return PairedPowerGateDesign(
        simulation_manifest=manifest,
        simulation_design_digest=manifest.digest,
        paired_evaluation_design_digest=evaluation_design.digest,
        runtime=manifest.runtime,
        null_configurations=_NULL_CONFIGURATIONS,
        target_effect=0.2,
        maximum_type_i_error=0.05,
        minimum_power=0.9,
        monte_carlo_alpha=0.01,
        replications_per_scenario=200,
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
    *,
    replications_per_scenario: int = 6,
    chunk_size: int = 2,
    target_effect: float = 0.1,
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
            replications_per_scenario=replications_per_scenario,
            chunk_size=chunk_size,
        ),
        scenarios=(
            PairedPowerScenarioManifest(scenario="weak-null", equal_task_effect=0.0),
            PairedPowerScenarioManifest(
                scenario="target-alternative",
                equal_task_effect=target_effect,
            ),
        ),
    )


def _trials(*, null_rejections: int, target_rejections: int) -> tuple[PairedPowerTrial, ...]:
    design = _design()
    return tuple(
        PairedPowerTrial(
            simulation_design_digest=design.simulation_design_digest,
            paired_evaluation_design_digest=design.paired_evaluation_design_digest,
            scenario=scenario,
            null_configuration=(null_configuration if scenario == "weak-null" else None),
            replicate=replicate,
            primary_passed=(
                replicate <= null_rejections
                if scenario == "weak-null"
                else replicate <= target_rejections
            ),
        )
        for scenario, null_configuration in (
            *(("weak-null", item) for item in _NULL_CONFIGURATIONS),
            ("target-alternative", None),
        )
        for replicate in range(1, design.replications_per_scenario + 1)
    )


def test_strong_locked_simulation_fixture_passes_both_power_gates() -> None:
    design = _design()
    report = evaluate_paired_power_gate(
        design,
        _trials(null_rejections=0, target_rejections=200),
    )

    assert report.design == design
    assert (
        report.design.paired_evaluation_design_digest
        == report.design.simulation_manifest.paired_evaluation_design_digest
    )
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
            tuple(reversed(_trials(null_rejections=0, target_rejections=200))),
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
    trials = _trials(null_rejections=0, target_rejections=200)

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
        _trials(null_rejections=0, target_rejections=200),
    )

    def reject(update: dict[str, object], match: str) -> None:
        payload = report.model_dump(mode="json")
        payload.update(update)
        with pytest.raises(ValidationError, match=match):
            PairedPowerGateReport.model_validate(payload)

    reject({"null_rejections": 201}, "cannot exceed frozen replications")
    reject({"target_rejections": 99}, "empirical_power differs")
    reject({"empirical_type_i_error": 0.01}, "empirical_type_i_error differs")
    reject({"empirical_power": 0.99}, "empirical_power differs")
    reject({"type_i_error_upper_bound": 0.0}, "type_i_error_upper_bound differs")
    reject({"power_lower_bound": 0.0}, "power_lower_bound differs")
    reject({"type_i_error_passed": False}, "type_i_error_passed differs")
    reject({"power_passed": False}, "power_passed differs")
    reject(
        {"null_configuration_monte_carlo_alpha": report.design.monte_carlo_alpha},
        "null Monte Carlo alpha differs",
    )

    changed_nulls = [item.model_dump(mode="json") for item in report.null_configurations]
    changed_nulls[0]["rejections"] = 1
    reject({"null_configurations": changed_nulls}, "null empirical rate differs")

    changed_design = report.design.model_copy(update={"target_effect": 0.3})
    reject({"design": changed_design.model_dump(mode="json")}, "target effect differs")
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
            null_configuration=null_configuration,
            first_replicate=first_replicate,
            last_replicate=last_replicate,
        )
        for scenario, null_configuration in (
            *(("weak-null", lane) for lane in evaluation_design.panel_members),
            ("target-alternative", None),
        )
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
        simulation_manifest=manifest,
        simulation_design_digest=manifest.digest,
        paired_evaluation_design_digest=evaluation_design.digest,
        runtime=manifest.runtime,
        null_configurations=manifest.lane_set,
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
            null_configuration=scenario.null_configuration,
            replicate=replicate,
            primary_passed=primary_passed,
        )
        for scenario in artifact.scenarios
        for replicate, primary_passed in enumerate(scenario.decisions.decisions(), start=1)
    )
    expanded_report = evaluate_paired_power_gate(gate, expanded_trials)
    assert report.null_rejections == max(
        artifact.rejection_count("weak-null", null_configuration=configuration)
        for configuration in artifact.null_configurations
    )
    assert report.target_rejections == artifact.rejection_count(
        "target-alternative",
        null_configuration=None,
    )
    assert report.trial_evidence_digest == artifact.trial_evidence_digest
    assert report.trial_evidence_digest == expanded_report.trial_evidence_digest
    assert report.digest == expanded_report.digest
    with pytest.raises(ValueError, match="simulator schema digest differs"):
        evaluate_paired_power_gate(
            gate,
            loaded.model_copy(update={"simulator_schema_digest": "sha256:" + "f" * 64}),
        )
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
            null_configuration=null_configuration,
            first_replicate=first_replicate,
            last_replicate=last_replicate,
        )
        for scenario, null_configuration in (
            *(("weak-null", lane) for lane in evaluation_design.panel_members),
            ("target-alternative", None),
        )
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
        null_configuration="lane-a",
        first_replicate=1,
        last_replicate=2,
    )
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir(mode=0o700)
    first_path = chunk_dir / "weak-null-member-001-000000001-000000002.json"
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
    chunk_dir.mkdir(mode=0o700)
    crash_temporary = chunk_dir / ".weak-null-member-001-000000001-000000002.json.crash.tmp"
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
    chunk_dir.mkdir(mode=0o700)
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
    assert not (chunk_dir / "weak-null-member-001-000000001-000000002.json").exists()


def test_signed_effect_calibration_searches_all_clipping_segments() -> None:
    baselines = np.array([0.82612, 0.78988], dtype=np.float64)
    atoms = (
        PairedPowerEffectAtom(multiplier=-2.692513, probability=0.184289),
        PairedPowerEffectAtom(multiplier=1.834231, probability=0.815711),
    )
    target = 0.098021

    scale = power_analysis._calibrated_effect_scale(
        baselines,
        atoms,
        target_effect=target,
    )
    achieved = sum(
        atom.probability
        * float(np.mean(np.clip(baselines + scale * atom.multiplier, 0.0, 1.0) - baselines))
        for atom in atoms
    )

    assert math.isclose(achieved, target, rel_tol=0.0, abs_tol=1e-12)


def test_positive_tiny_effect_cannot_collapse_to_the_null() -> None:
    target = 5e-13
    scale = power_analysis._solve_clipped_additive_scale(
        np.array([0.5], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        target_effect=target,
    )
    achieved = float(np.clip(0.5 + scale, 0.0, 1.0) - 0.5)

    assert scale > 0.0
    assert achieved > 0.0
    assert abs(achieved - target) <= math.ulp(0.5)


def test_target_effect_assignment_is_fixed_and_exact_for_every_roster() -> None:
    evaluation_design = _synthetic_evaluation_design()
    profile = _synthetic_task_profile()
    manifest = _synthetic_simulation_manifest(evaluation_design, profile)

    first = power_analysis._fixed_task_effect_multipliers(manifest, profile)
    second = power_analysis._fixed_task_effect_multipliers(manifest, profile)
    assert np.array_equal(first, second)
    assert first[0] == first[1]

    target = 0.1
    for panel_member in evaluation_design.panel_members:
        baselines = np.array(
            [
                next(
                    item.probability
                    for item in task.lane_baselines
                    if item.panel_member == panel_member
                )
                for task in profile.tasks
            ],
            dtype=np.float64,
        )
        scale = power_analysis._calibrated_fixed_effect_scale(
            baselines,
            first,
            target_effect=target,
        )
        achieved = float(np.mean(np.clip(baselines + scale * first, 0.0, 1.0) - baselines))
        assert math.isclose(achieved, target, rel_tol=0.0, abs_tol=1e-12)


def test_memberwise_null_projection_is_a_conservative_iut_upper_bound() -> None:
    evaluation_design = _synthetic_evaluation_design()
    task_count = len(evaluation_design.task_ids)
    designated_row = tuple(0.25 for _ in range(task_count))
    low_matrix = (
        tuple(-1.0 for _ in range(task_count)),
        designated_row,
        tuple(-1.0 for _ in range(task_count)),
    )
    high_matrix = (
        tuple(1.0 for _ in range(task_count)),
        designated_row,
        tuple(1.0 for _ in range(task_count)),
    )

    low_members = paired_member_primary_decisions(evaluation_design, low_matrix)
    high_members = paired_member_primary_decisions(evaluation_design, high_matrix)
    assert low_members[1] == high_members[1]
    for matrix in (low_matrix, high_matrix):
        member_decisions = paired_member_primary_decisions(evaluation_design, matrix)
        all_member_decision = paired_primary_decision_passed(evaluation_design, matrix)
        assert not all_member_decision or all(member_decisions)
        assert all_member_decision <= member_decisions[1]


def test_weak_null_simulates_only_the_designated_lane_marginal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation_design = _synthetic_evaluation_design()
    profile = _synthetic_task_profile()
    manifest = _synthetic_simulation_manifest(evaluation_design, profile)
    dependence = manifest.dgp.dependence
    assert dependence.iut_null_evaluation == "memberwise-marginal-conservative-upper-bound-v1"
    assert dependence.other_lane_nuisance_bound == "all-other-member-decisions-pass"

    original_sample = power_analysis._sample_attempt_counts
    sample_calls = 0
    sampled_probabilities: list[np.ndarray] = []

    def counted_sample(
        rng: np.random.Generator,
        probability: np.ndarray,
        *,
        attempts: int,
        intraclass_correlation: float,
    ) -> np.ndarray:
        nonlocal sample_calls
        sample_calls += 1
        sampled_probabilities.append(probability.copy())
        return original_sample(
            rng,
            probability,
            attempts=attempts,
            intraclass_correlation=intraclass_correlation,
        )

    monkeypatch.setattr(power_analysis, "_sample_attempt_counts", counted_sample)
    run_paired_power_chunk(
        manifest,
        evaluation_design,
        profile,
        scenario="weak-null",
        null_configuration="lane-b",
        first_replicate=1,
        last_replicate=2,
    )

    assert sample_calls == 2
    expected_lane_b = np.broadcast_to(
        np.array(
            [
                next(
                    lane.probability
                    for lane in task.lane_baselines
                    if lane.panel_member == "lane-b"
                )
                for task in profile.tasks
            ],
            dtype=np.float64,
        ),
        (2, 4),
    )
    assert all(
        np.array_equal(probability, expected_lane_b) for probability in sampled_probabilities
    )


def test_gate_uses_memberwise_iut_nulls_and_worst_simultaneous_bound() -> None:
    evaluation_design = _synthetic_evaluation_design()
    profile = _synthetic_task_profile()
    manifest = _synthetic_simulation_manifest(
        evaluation_design,
        profile,
        target_effect=0.2,
    )
    null_configurations = ("lane-a", "lane-b", "lane-c")
    design = PairedPowerGateDesign(
        simulation_manifest=manifest,
        simulation_design_digest=manifest.digest,
        paired_evaluation_design_digest=evaluation_design.digest,
        runtime=manifest.runtime,
        null_configurations=null_configurations,
        target_effect=0.2,
        maximum_type_i_error=0.9,
        minimum_power=0.01,
        monte_carlo_alpha=0.0031,
        replications_per_scenario=6,
    )
    null_rejections = {"lane-a": 0, "lane-b": 1, "lane-c": 2}
    trials = tuple(
        PairedPowerTrial(
            simulation_design_digest=manifest.digest,
            paired_evaluation_design_digest=evaluation_design.digest,
            scenario="weak-null",
            null_configuration=null_configuration,
            replicate=replicate,
            primary_passed=replicate <= null_rejections[null_configuration],
        )
        for null_configuration in null_configurations
        for replicate in range(1, 7)
    ) + tuple(
        PairedPowerTrial(
            simulation_design_digest=manifest.digest,
            paired_evaluation_design_digest=evaluation_design.digest,
            scenario="target-alternative",
            null_configuration=None,
            replicate=replicate,
            primary_passed=True,
        )
        for replicate in range(1, 7)
    )

    report = evaluate_paired_power_gate(design, trials)

    assert tuple(item.null_configuration for item in report.null_configurations) == (
        null_configurations
    )
    assert tuple(item.rejections for item in report.null_configurations) == (0, 1, 2)
    assert report.null_rejections == 2
    assert Decimal.from_float(report.null_configuration_monte_carlo_alpha) * len(
        null_configurations
    ) <= Decimal.from_float(design.monte_carlo_alpha)
    assert report.type_i_error_upper_bound == max(
        _binomial_upper_bound(
            count,
            6,
            alpha=report.null_configuration_monte_carlo_alpha,
        )
        for count in null_rejections.values()
    )


def test_gate_rejects_omitted_manifest_null_lanes() -> None:
    evaluation_design = _synthetic_evaluation_design()
    profile = _synthetic_task_profile()
    manifest = _synthetic_simulation_manifest(evaluation_design, profile)
    chunks = tuple(
        run_paired_power_chunk(
            manifest,
            evaluation_design,
            profile,
            scenario=scenario,
            null_configuration=null_configuration,
            first_replicate=first,
            last_replicate=last,
        )
        for scenario, null_configuration in (
            ("weak-null", "lane-a"),
            ("target-alternative", None),
        )
        for first, last in manifest.replications.chunk_ranges
    )
    trials = tuple(
        PairedPowerTrial(
            simulation_design_digest=chunk.simulation_design_digest,
            paired_evaluation_design_digest=chunk.paired_evaluation_design_digest,
            scenario=chunk.scenario,
            null_configuration=chunk.null_configuration,
            replicate=replicate,
            primary_passed=decision,
        )
        for chunk in chunks
        for replicate, decision in enumerate(
            chunk.decisions.decisions(),
            start=chunk.first_replicate,
        )
    )
    gate = PairedPowerGateDesign(
        simulation_manifest=manifest,
        simulation_design_digest=manifest.digest,
        paired_evaluation_design_digest=evaluation_design.digest,
        runtime=manifest.runtime,
        null_configurations=manifest.lane_set,
        target_effect=manifest.scenarios[1].equal_task_effect,
        maximum_type_i_error=0.9,
        minimum_power=0.01,
        monte_carlo_alpha=0.1,
        replications_per_scenario=manifest.replications.replications_per_scenario,
    )
    incomplete_gate = gate.model_copy(update={"null_configurations": ("lane-a",)})

    with pytest.raises(ValueError, match="null configurations differ from the manifest"):
        evaluate_paired_power_gate(incomplete_gate, trials)

    for update, match in (
        ({"target_effect": 0.2}, "target effect"),
        ({"replications_per_scenario": 3}, "replication horizon"),
    ):
        with pytest.raises(ValueError, match=match):
            evaluate_paired_power_gate(gate.model_copy(update=update), trials)


def test_simulation_and_gate_reject_runtime_version_drift() -> None:
    evaluation_design = _synthetic_evaluation_design()
    profile = _synthetic_task_profile()
    manifest = _synthetic_simulation_manifest(evaluation_design, profile)
    assert manifest.runtime == power_analysis.PairedPowerRuntimeManifest.current()
    assert manifest.runtime.python_executable_digest.startswith("sha256:")
    assert manifest.runtime.numpy_distribution_record_digest.startswith("sha256:")
    assert manifest.runtime.scipy_distribution_record_digest.startswith("sha256:")
    assert manifest.runtime.pydantic_core_version
    drifted_runtime = manifest.runtime.model_copy(update={"numpy_version": "0.0.invalid"})
    drifted_manifest = manifest.model_copy(update={"runtime": drifted_runtime})

    with pytest.raises(ValueError, match="runtime"):
        run_paired_power_chunk(
            drifted_manifest,
            evaluation_design,
            profile,
            scenario="target-alternative",
            first_replicate=1,
            last_replicate=2,
        )

    gate = PairedPowerGateDesign(
        simulation_manifest=manifest,
        simulation_design_digest=manifest.digest,
        paired_evaluation_design_digest=evaluation_design.digest,
        runtime=manifest.runtime,
        null_configurations=evaluation_design.panel_members,
        target_effect=0.1,
        maximum_type_i_error=0.9,
        minimum_power=0.01,
        monte_carlo_alpha=0.1,
        replications_per_scenario=6,
    )
    with pytest.raises(ValueError, match="runtime"):
        evaluate_paired_power_gate(gate.model_copy(update={"runtime": drifted_runtime}), ())


def test_runtime_manifest_cache_does_not_evict_base_for_subclasses() -> None:
    class DerivedRuntimeManifest(power_analysis.PairedPowerRuntimeManifest):
        pass

    power_analysis.PairedPowerRuntimeManifest.current.cache_clear()
    try:
        base_runtime = power_analysis.PairedPowerRuntimeManifest.current()
        DerivedRuntimeManifest.current()
        assert power_analysis.PairedPowerRuntimeManifest.current() is base_runtime
    finally:
        power_analysis.PairedPowerRuntimeManifest.current.cache_clear()


def test_distribution_identity_is_relocation_invariant_and_content_bound(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-site-packages"
    package = source / "example"
    metadata = source / "example-1.0.dist-info"
    package.mkdir(parents=True)
    metadata.mkdir()
    code = b"VALUE = 1\n"
    package_file = package / "core.py"
    metadata_file = metadata / "METADATA"
    package_file.write_bytes(code)
    metadata_file.write_bytes(b"Name: example\nVersion: 1.0\n")

    def record_line(relative_path: str, payload: bytes) -> str:
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        return f"{relative_path},sha256={digest},{len(payload)}"

    def write_record(
        root: Path,
        *,
        external_hash: str,
        package_payload: bytes,
        package_path: str = "example/core.py",
    ) -> Path:
        record = root / "example-1.0.dist-info" / "RECORD"
        record.write_text(
            "\n".join(
                (
                    record_line(package_path, package_payload),
                    record_line(
                        "example-1.0.dist-info/METADATA",
                        b"Name: example\nVersion: 1.0\n",
                    ),
                    f"../../../bin/example,sha256={external_hash},999",
                    "example-1.0.dist-info/RECORD,,",
                    "",
                )
            )
        )
        return record

    source_record = write_record(source, external_hash="source-path", package_payload=code)
    relocated = tmp_path / "different-prefix" / "site-packages"
    shutil.copytree(source, relocated)
    relocated_record = write_record(
        relocated,
        external_hash="different-generated-script",
        package_payload=code,
        package_path="./example/core.py",
    )

    source_digest = power_analysis._canonical_distribution_record_digest(source_record)
    assert power_analysis._canonical_distribution_record_digest(relocated_record) == source_digest

    changed_code = b"VALUE = 2\n"
    (relocated / "example" / "core.py").write_bytes(changed_code)
    with pytest.raises(RuntimeError, match="does not match RECORD"):
        power_analysis._canonical_distribution_record_digest(relocated_record)
    relocated_record = write_record(
        relocated,
        external_hash="different-generated-script",
        package_payload=changed_code,
        package_path="./example/core.py",
    )
    assert power_analysis._canonical_distribution_record_digest(relocated_record) != source_digest
    relocated_record.write_text(
        relocated_record.read_text() + record_line("example/core.py", changed_code) + "\n"
    )
    with pytest.raises(RuntimeError, match="duplicate paths"):
        power_analysis._canonical_distribution_record_digest(relocated_record)


def test_artifact_writes_are_immutable_and_reject_replacement(tmp_path: Path) -> None:
    evaluation_design = _synthetic_evaluation_design()
    profile = _synthetic_task_profile()
    manifest = _synthetic_simulation_manifest(evaluation_design, profile)
    chunk = run_paired_power_chunk(
        manifest,
        evaluation_design,
        profile,
        scenario="target-alternative",
        first_replicate=1,
        last_replicate=2,
    )
    path = tmp_path / "chunk.json"
    write_paired_power_chunk(path, chunk)

    with pytest.raises(FileExistsError):
        write_paired_power_chunk(path, chunk)


def test_artifact_read_rejects_wrong_mode_and_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation_design = _synthetic_evaluation_design()
    profile = _synthetic_task_profile()
    manifest = _synthetic_simulation_manifest(evaluation_design, profile)
    chunk = run_paired_power_chunk(
        manifest,
        evaluation_design,
        profile,
        scenario="target-alternative",
        first_replicate=1,
        last_replicate=2,
    )
    path = tmp_path / "chunk.json"
    write_paired_power_chunk(path, chunk)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="mode"):
        power_analysis.load_paired_power_chunk(path)

    path.chmod(0o600)
    real_getuid = os.getuid
    monkeypatch.setattr(power_analysis.os, "getuid", lambda: real_getuid() + 1)
    with pytest.raises(ValueError, match="owner"):
        power_analysis.load_paired_power_chunk(path)


def test_resume_rejects_symlinked_or_insecure_directory(tmp_path: Path) -> None:
    evaluation_design = _synthetic_evaluation_design()
    profile = _synthetic_task_profile()
    manifest = _synthetic_simulation_manifest(evaluation_design, profile)
    real_directory = tmp_path / "real"
    real_directory.mkdir(mode=0o700)
    linked_directory = tmp_path / "linked"
    linked_directory.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|directory"):
        resume_paired_power_simulation(
            manifest,
            evaluation_design,
            profile,
            linked_directory,
        )

    nested_directory = real_directory / "nested"
    nested_directory.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|directory"):
        resume_paired_power_simulation(
            manifest,
            evaluation_design,
            profile,
            linked_parent / "nested",
        )

    real_directory.chmod(0o755)
    with pytest.raises(ValueError, match="mode"):
        resume_paired_power_simulation(
            manifest,
            evaluation_design,
            profile,
            real_directory,
        )


def test_artifact_read_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "chunk.json"
    os.mkfifo(fifo, mode=0o600)
    outcome: list[BaseException] = []

    def load() -> None:
        try:
            power_analysis.load_paired_power_chunk(fifo)
        except ValueError as exc:
            outcome.append(exc)

    thread = threading.Thread(target=load, daemon=True)
    thread.start()
    thread.join(timeout=1.0)

    assert not thread.is_alive(), "artifact reader blocked opening a FIFO"
    assert outcome and isinstance(outcome[0], ValueError)
    assert "regular" in str(outcome[0])


def test_artifact_read_rejects_symlink_and_replacement_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation_design = _synthetic_evaluation_design()
    profile = _synthetic_task_profile()
    manifest = _synthetic_simulation_manifest(evaluation_design, profile)
    chunk = run_paired_power_chunk(
        manifest,
        evaluation_design,
        profile,
        scenario="target-alternative",
        first_replicate=1,
        last_replicate=2,
    )
    path = tmp_path / "chunk.json"
    write_paired_power_chunk(path, chunk)
    payload = path.read_bytes()

    symlink = tmp_path / "linked.json"
    symlink.symlink_to(path)
    with pytest.raises(ValueError, match="regular"):
        power_analysis.load_paired_power_chunk(symlink)

    real_open = os.open
    replaced = False

    def replace_before_open(
        target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if target == path.name and dir_fd is not None and not replaced:
            replaced = True
            os.unlink(path.name, dir_fd=dir_fd)
            replacement = real_open(
                path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o600,
                dir_fd=dir_fd,
            )
            try:
                os.write(replacement, payload)
                os.fsync(replacement)
            finally:
                os.close(replacement)
        return real_open(target, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(power_analysis.os, "open", replace_before_open)
    monkeypatch.setattr(power_analysis, "_require_descriptor_relative_filesystem", lambda: None)
    with pytest.raises(ValueError, match="changed while opening"):
        power_analysis.load_paired_power_chunk(path)


def test_artifact_read_rejects_replacement_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation_design = _synthetic_evaluation_design()
    profile = _synthetic_task_profile()
    manifest = _synthetic_simulation_manifest(evaluation_design, profile)
    chunk = run_paired_power_chunk(
        manifest,
        evaluation_design,
        profile,
        scenario="target-alternative",
        first_replicate=1,
        last_replicate=2,
    )
    path = tmp_path / "chunk.json"
    write_paired_power_chunk(path, chunk)
    payload = path.read_bytes()
    real_open = os.open
    real_read = os.read
    replaced = False

    def replace_after_open(descriptor: int, length: int) -> bytes:
        nonlocal replaced
        if not replaced:
            replaced = True
            path.unlink()
            replacement = real_open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o600,
            )
            try:
                os.write(replacement, payload)
                os.fsync(replacement)
            finally:
                os.close(replacement)
        return real_read(descriptor, length)

    monkeypatch.setattr(power_analysis.os, "read", replace_after_open)
    with pytest.raises(ValueError, match="changed|replaced"):
        power_analysis.load_paired_power_chunk(path)


def test_artifact_read_rejects_parent_directory_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation_design = _synthetic_evaluation_design()
    profile = _synthetic_task_profile()
    manifest = _synthetic_simulation_manifest(evaluation_design, profile)
    chunk = run_paired_power_chunk(
        manifest,
        evaluation_design,
        profile,
        scenario="target-alternative",
        first_replicate=1,
        last_replicate=2,
    )
    directory = tmp_path / "chunks"
    directory.mkdir(mode=0o700)
    path = directory / "chunk.json"
    write_paired_power_chunk(path, chunk)
    relocated = tmp_path / "relocated"
    real_read_at = power_analysis._read_bounded_file_at

    def replace_directory(directory_descriptor: int, name: str) -> bytes:
        directory.rename(relocated)
        directory.mkdir(mode=0o700)
        return real_read_at(directory_descriptor, name)

    monkeypatch.setattr(power_analysis, "_read_bounded_file_at", replace_directory)
    with pytest.raises(ValueError, match="directory path was replaced"):
        power_analysis.load_paired_power_chunk(path)


def test_directory_validation_does_not_mask_primary_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("primary write failure")

    def fail_validation(*_args: object, **_kwargs: object) -> None:
        raise ValueError("secondary directory failure")

    monkeypatch.setattr(power_analysis, "_atomic_write_json_at", fail_write)
    monkeypatch.setattr(power_analysis, "_validate_directory_identity", fail_validation)

    with pytest.raises(RuntimeError, match="primary write failure") as error:
        power_analysis._atomic_write_json(tmp_path / "artifact.json", {})

    assert any(
        "secondary directory failure" in note for note in getattr(error.value, "__notes__", ())
    )

    monkeypatch.setattr(power_analysis, "_atomic_write_json_at", lambda *_args, **_kwargs: None)
    try:
        raise LookupError("already handled outside the artifact operation")
    except LookupError:
        with pytest.raises(ValueError, match="secondary directory failure"):
            power_analysis._atomic_write_json(tmp_path / "second.json", {})
