"""Tests for the staged optimizer's engine: skip/rerun decisions, the manifest, and the cap."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.optimize.pipeline import (
    BUILT_STAGES,
    CONFIGURED_STAGES,
    MANIFEST_VERSION,
    RESERVED_STAGES,
    STAGE_ORDER,
    BudgetExceeded,
    RunManifest,
    SpendLedger,
    Stage,
    StageDecision,
    StageRecord,
    StageStatus,
    decide_stage,
    file_sha256,
    forced_stages,
    load_manifest,
    planned_stages,
    project_sweep_spend,
)


def _record(stage: Stage, artifact: Path, **fingerprint: str) -> StageRecord:
    return StageRecord(
        stage=stage,
        fingerprint=fingerprint,
        artifact_path=str(artifact),
        artifact_identity=file_sha256(artifact),
        completed_at="2026-07-27T00:00:00+00:00",
    )


def _manifest(*records: StageRecord) -> RunManifest:
    manifest = RunManifest(world_model="support")
    for record in records:
        manifest = manifest.with_record(record)
    return manifest


def _artifact(tmp_path: Path, text: str = "matrix") -> Path:
    path = tmp_path / "matrix.json"
    path.write_text(text, encoding="utf-8")
    return path


def _decide(
    manifest: RunManifest,
    artifact: Path,
    *,
    forced: bool = False,
    **fingerprint: str,
) -> StageDecision:
    return decide_stage(
        Stage.SWEEP,
        manifest=manifest,
        fingerprint=fingerprint,
        artifact=artifact,
        artifact_identity=file_sha256(artifact),
        forced=forced,
        skip_summary="same pool, same scenarios",
    )


def test_a_stage_with_unchanged_inputs_skips_and_says_what_was_unchanged(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    manifest = _manifest(_record(Stage.SWEEP, artifact, pool="abc", episodes="1"))
    decision = _decide(manifest, artifact, pool="abc", episodes="1")
    assert decision.status is StageStatus.SKIP
    assert "matrix.json is current" in decision.reason
    assert "same pool, same scenarios" in decision.reason


def test_a_changed_input_reruns_and_the_reason_names_which_one(tmp_path: Path) -> None:
    # The whole point of the printed reason: an operator must be able to tell WHY paid work is
    # about to be redone, and "something changed" is not that.
    artifact = _artifact(tmp_path)
    manifest = _manifest(_record(Stage.SWEEP, artifact, pool="abc", episodes="1"))
    decision = _decide(manifest, artifact, pool="abc", episodes="3")
    assert decision.status is StageStatus.RUN
    assert decision.reason.startswith("episodes changed")
    assert "1 -> 3" in decision.reason


def test_a_stage_that_never_ran_here_runs(tmp_path: Path) -> None:
    decision = _decide(_manifest(), _artifact(tmp_path), pool="abc")
    assert decision.status is StageStatus.RUN
    assert decision.reason == "never completed here"


def test_a_deleted_artifact_reruns_even_with_matching_fingerprints(tmp_path: Path) -> None:
    # The manifest is a cache of decisions, never evidence that a file exists. Trusting it over
    # the filesystem would let a run "skip" its way to a fit on a matrix that is not there.
    artifact = _artifact(tmp_path)
    manifest = _manifest(_record(Stage.SWEEP, artifact, pool="abc"))
    artifact.unlink()
    decision = _decide(manifest, artifact, pool="abc")
    assert decision.status is StageStatus.RUN
    assert decision.reason == "matrix.json is no longer on disk"


def test_an_artifact_edited_out_of_band_reruns(tmp_path: Path) -> None:
    # Dropping to a manual command mid-flow is supported, so the artifact it left behind has to
    # be noticed rather than assumed to be the one this run wrote.
    artifact = _artifact(tmp_path)
    manifest = _manifest(_record(Stage.SWEEP, artifact, pool="abc"))
    artifact.write_text("a different matrix entirely", encoding="utf-8")
    decision = _decide(manifest, artifact, pool="abc")
    assert decision.status is StageStatus.RUN
    assert "changed on disk" in decision.reason


def test_force_from_beats_every_matching_fingerprint(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    manifest = _manifest(_record(Stage.SWEEP, artifact, pool="abc"))
    decision = _decide(manifest, artifact, forced=True, pool="abc")
    assert decision.status is StageStatus.RUN
    assert decision.reason == "forced by --force-from"


def test_force_from_selects_that_stage_and_everything_downstream() -> None:
    forced = forced_stages(Stage.FIT)
    assert Stage.FIT in forced
    assert Stage.TUNE in forced and Stage.REPORT in forced
    # ...and nothing upstream, so a redo of the fit never re-buys the sweep's cells.
    assert Stage.SWEEP not in forced and Stage.PREFLIGHT not in forced
    # The reserved slots sit in the order too, so their arrival does not renumber anything.
    assert forced_stages(Stage.SWEEP) >= {Stage.DISTILL, Stage.COMPACT}
    assert forced_stages(None) == frozenset()


def test_a_new_fingerprint_key_reruns_rather_than_silently_ignoring_it(tmp_path: Path) -> None:
    # A build that starts fingerprinting an input the recording never had must not treat the
    # older, less specific record as a match.
    artifact = _artifact(tmp_path)
    manifest = _manifest(_record(Stage.SWEEP, artifact, pool="abc"))
    decision = _decide(manifest, artifact, pool="abc", max_steps="20")
    assert decision.status is StageStatus.RUN
    assert "max_steps is a new input" in decision.reason


def test_a_dropped_fingerprint_key_also_reruns(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    manifest = _manifest(_record(Stage.SWEEP, artifact, pool="abc", episodes="1"))
    decision = _decide(manifest, artifact, pool="abc")
    assert decision.status is StageStatus.RUN
    assert "episodes is no longer an input" in decision.reason


def test_a_corrupt_manifest_resets_cleanly_with_a_warning(tmp_path: Path) -> None:
    # A manifest is a cache, so an unreadable one costs re-running stages and nothing else.
    # Refusing to start over it would strand an operator behind a file they cannot repair.
    path = tmp_path / "optimize-run.json"
    path.write_text("{not json at all", encoding="utf-8")
    read = load_manifest(path, world_model="support")
    assert read.manifest.stages == []
    assert read.warning is not None
    assert "could not be read as a run manifest" in read.warning
    assert "Nothing was deleted" in read.warning
    assert path.is_file()  # the reset is in memory; the file is the operator's to remove


def test_a_manifest_from_another_schema_version_resets(tmp_path: Path) -> None:
    path = tmp_path / "optimize-run.json"
    path.write_text(
        RunManifest(version=MANIFEST_VERSION + 1, world_model="support").model_dump_json(),
        encoding="utf-8",
    )
    read = load_manifest(path, world_model="support")
    assert read.manifest.stages == []
    assert read.warning is not None and "manifest version" in read.warning


def test_a_manifest_for_a_different_world_model_resets(tmp_path: Path) -> None:
    path = tmp_path / "optimize-run.json"
    RunManifest(world_model="other").save(path)
    read = load_manifest(path, world_model="support")
    assert read.manifest.stages == []
    assert read.warning is not None and "'other'" in read.warning


def test_a_missing_manifest_is_an_empty_one_with_nothing_to_say(tmp_path: Path) -> None:
    read = load_manifest(tmp_path / "nope.json", world_model="support")
    assert read.manifest.stages == [] and read.warning is None


def test_recording_a_stage_twice_replaces_it_and_keeps_stage_order(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    manifest = _manifest(
        _record(Stage.REPORT, artifact),
        _record(Stage.SWEEP, artifact, pool="abc"),
        _record(Stage.SWEEP, artifact, pool="def"),
    )
    assert [record.stage for record in manifest.stages] == [Stage.SWEEP, Stage.REPORT]
    sweep = manifest.record_for(Stage.SWEEP)
    assert sweep is not None and sweep.fingerprint == {"pool": "def"}


def test_the_spend_cap_stops_before_a_stage_rather_than_during_it() -> None:
    # Half a sweep is not a cheaper sweep, it is an unusable matrix that was paid for anyway.
    ledger = SpendLedger(max_usd=10.0)
    ledger.record(7.5)
    ledger.check(Stage.SWEEP, 2.0)  # 9.50 total: fits
    with pytest.raises(BudgetExceeded) as caught:
        ledger.check(Stage.SWEEP, 3.0)
    message = str(caught.value)
    assert "$3.00" in message and "$7.50" in message and "$10.00" in message


def test_no_cap_never_stops_anything() -> None:
    ledger = SpendLedger()
    ledger.record(1_000.0)
    ledger.check(Stage.SWEEP, 1_000.0)  # does not raise


def test_file_sha256_of_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert file_sha256(tmp_path / "nope") == ""


def test_the_reserved_slot_is_named_but_not_built() -> None:
    """Distill exists in the ordering so its arrival is additive; compaction has arrived."""
    assert set(RESERVED_STAGES) == {Stage.DISTILL}
    assert set(BUILT_STAGES).isdisjoint(RESERVED_STAGES)
    assert set(BUILT_STAGES) | set(RESERVED_STAGES) | set(CONFIGURED_STAGES) == set(STAGE_ORDER)
    # Compaction sits between sweep and fit, distill after sweep: the slots the design reserved.
    order = list(STAGE_ORDER)
    assert order.index(Stage.SWEEP) < order.index(Stage.COMPACT) < order.index(Stage.FIT)
    assert order.index(Stage.SWEEP) < order.index(Stage.DISTILL)


def test_compaction_is_in_a_run_only_when_a_compressor_is_named() -> None:
    # The row would otherwise advertise a stage with nothing to do, on every uncompressed run.
    assert planned_stages(compacting=False) == BUILT_STAGES
    assert planned_stages(compacting=True) == (
        Stage.PREFLIGHT,
        Stage.SWEEP,
        Stage.COMPACT,
        Stage.FIT,
        Stage.TUNE,
        Stage.REPORT,
    )


def _sweep_record(*, candidate: float, world_model: float, compressor: float = 0.0) -> StageRecord:
    return StageRecord(
        stage=Stage.SWEEP,
        fingerprint={"pool": "abc"},
        completed_at="2026-07-27T00:00:00+00:00",
        spend_usd=candidate,
        compressor_spend_usd=compressor,
        world_model_spend_usd=world_model,
    )


def test_with_no_prior_sweep_the_world_model_side_is_not_projected() -> None:
    # Not projected is not "projected to be zero": the caller prints a caveat instead of adding
    # a number it cannot justify. Nothing predicts the simulator's token use before it runs.
    projection = project_sweep_spend(4.0, None)
    assert projection.candidate_usd == 4.0
    assert projection.world_model_usd == 0.0
    assert projection.basis is None and not projection.projected
    assert projection.total_usd == 4.0


def test_a_prior_sweep_projects_the_world_model_side_from_its_measured_ratio() -> None:
    # The drive's real numbers: $1.8076 world-model against $0.2581 candidate is 7.0x, so a
    # second sweep of the same size is forecast at ~8x the candidate figure in total. A cap that
    # counted only the candidate side would miss almost all of that.
    projection = project_sweep_spend(0.2581, _sweep_record(candidate=0.2581, world_model=1.8076))
    assert projection.world_model_usd == pytest.approx(1.8076)
    assert projection.total_usd == pytest.approx(2.0657)
    assert projection.basis is not None
    assert "7.0x" in projection.basis
    # Four decimals: a sub-cent side must not render as "$0.00" beside a nonzero ratio.
    assert "$1.8076" in projection.basis and "$0.2581 projectable candidate" in projection.basis


def test_the_projection_scales_with_the_size_of_the_sweep_being_planned() -> None:
    # The ratio is the transferable part, not the absolute: a sweep twice the size is forecast
    # at twice the world-model cost.
    prior = _sweep_record(candidate=1.0, world_model=7.0)
    assert project_sweep_spend(2.0, prior).world_model_usd == pytest.approx(14.0)


def test_a_prior_whose_candidate_side_measured_zero_supplies_no_ratio() -> None:
    # Dividing by it is undefined, and carrying its world-model figure over as an absolute would
    # forecast this sweep from another sweep's SIZE rather than its shape.
    projection = project_sweep_spend(4.0, _sweep_record(candidate=0.0, world_model=9.0))
    assert projection.basis is None and projection.world_model_usd == 0.0


def test_a_prior_record_for_another_stage_is_not_a_sweep_basis() -> None:
    fit = StageRecord(
        stage=Stage.FIT, fingerprint={}, completed_at="2026-07-27T00:00:00+00:00", spend_usd=1.0
    )
    assert project_sweep_spend(4.0, fit).basis is None


def test_the_cap_message_names_the_basis_of_a_forecast_it_stopped_on() -> None:
    # An operator told a run cannot start is owed the reasoning, especially when half the figure
    # is a forecast from one prior observation rather than arithmetic.
    ledger = SpendLedger(max_usd=1.0)
    with pytest.raises(BudgetExceeded) as caught:
        ledger.check(Stage.SWEEP, 2.07, basis="the last sweep measured 7.0x")
    assert "projection basis: the last sweep measured 7.0x" in str(caught.value)


def test_a_sub_cent_candidate_side_is_not_rendered_as_zero_in_the_basis() -> None:
    # A basis reading "$0.12 world-model against $0.00 candidate (90.9x)" contradicts itself and
    # reads as the divide-by-zero case this function refuses to do, so an operator would take a
    # working forecast for a bug.
    basis = project_sweep_spend(1.0, _sweep_record(candidate=0.00132, world_model=0.12)).basis
    assert basis is not None
    assert "$0.00 candidate" not in basis
    assert "$0.0013 projectable candidate" in basis


def test_a_compressed_prior_forecasts_on_like_units_not_the_folded_total() -> None:
    """The ratio's denominator has to match the units of the number it multiplies.

    D-COMPRESS folds the compressor's bill into candidate spend, but nothing can project a
    compressor's per-call cost in advance, so the projection this ratio multiplies is the model
    half alone. Dividing by the FOLDED total would shrink every forecast by the compressor's
    share of the last run: systematically short, on the term that dominates the bill.

    Concrete rather than re-derived: asserting the formula the implementation uses would pass
    even if both were wrong together.
    """
    prior = _sweep_record(candidate=1.00, world_model=7.00, compressor=0.30)
    projection = project_sweep_spend(0.70, prior)
    # 7.00 / (1.00 - 0.30) = 10.0x on the projectable part; 10.0 x 0.70 = $7.00 like-for-like.
    # A folded denominator would give 7.00 / 1.00 = 7.0x -> $4.90, a 30% shortfall.
    assert projection.world_model_usd == pytest.approx(7.00)
    assert projection.total_usd == pytest.approx(7.70)
    assert projection.basis is not None
    assert "$0.7000 projectable candidate" in projection.basis and "10.0x" in projection.basis


def test_a_prior_that_was_all_compressor_supplies_no_ratio() -> None:
    # Projectable part is zero, so there is nothing to divide by and nothing honest to forecast.
    projection = project_sweep_spend(
        1.0, _sweep_record(candidate=0.5, world_model=9.0, compressor=0.5)
    )
    assert projection.basis is None and projection.world_model_usd == 0.0


def test_lifetime_split_attributes_presplit_remainder_to_the_candidate_leg() -> None:
    """Manifests written before the split fields reported the combined total as
    the candidate figure; the split keeps doing that for the untracked
    remainder so a resumed run's platform row never jumps."""
    legacy = RunManifest(world_model="tau-bench", lifetime_spend_usd=12.5)

    assert legacy.lifetime_split == (12.5, 0.0)

    tracked = RunManifest(
        world_model="tau-bench",
        lifetime_spend_usd=12.5,
        lifetime_candidate_usd=9.25,
        lifetime_wm_usd=3.25,
    )
    assert tracked.lifetime_split == (9.25, 3.25)


def test_with_record_accumulates_both_lifetime_legs() -> None:
    """Each stage's spend lands on its own lifetime leg, and re-swept stages
    keep their replaced spend counted (money left the account either way)."""
    record = StageRecord(
        stage=Stage.SWEEP,
        fingerprint={},
        artifact_path="matrix.json",
        artifact_identity="a",
        completed_at="2026-07-29T00:00:00+00:00",
        spend_usd=2.0,
        compressor_spend_usd=0.5,
        world_model_spend_usd=7.0,
    )
    manifest = RunManifest(world_model="tau-bench").with_record(record)
    twice = manifest.with_record(record)

    assert manifest.lifetime_candidate_usd == 2.0
    assert manifest.lifetime_wm_usd == 7.0
    assert manifest.lifetime_split == (2.0, 7.0)
    assert twice.lifetime_split == (4.0, 14.0)
    assert twice.lifetime_spend_usd == 18.0
