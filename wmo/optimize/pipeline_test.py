"""Tests for the staged optimizer's engine: skip/rerun decisions, the manifest, and the cap."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.optimize.pipeline import (
    BUILT_STAGES,
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


def test_the_reserved_slots_are_named_but_not_built() -> None:
    """The distill and compaction stages exist in the ordering so their arrival is additive."""
    assert set(RESERVED_STAGES) == {Stage.DISTILL, Stage.COMPACT}
    assert set(BUILT_STAGES).isdisjoint(RESERVED_STAGES)
    assert set(BUILT_STAGES) | set(RESERVED_STAGES) == set(STAGE_ORDER)
    # Compaction sits between sweep and fit, distill after sweep: the slots the design reserved.
    order = list(STAGE_ORDER)
    assert order.index(Stage.SWEEP) < order.index(Stage.COMPACT) < order.index(Stage.FIT)
    assert order.index(Stage.SWEEP) < order.index(Stage.DISTILL)
