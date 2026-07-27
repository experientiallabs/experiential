"""The staged optimizer's engine: what to run, what to skip, and why.

`wmo optimize model` is a sequence of stages that each wrap a manual command. What makes it
resumable rather than merely re-runnable is this module: a manifest at
`<model_dir>/optimize/optimize-run.json` records, per completed stage, the fingerprints of the
inputs it consumed and the hash of the artifact it produced. On the next run a stage is skipped
if and only if every recorded fingerprint still matches the live one and its artifact is still
on disk unchanged, and either way the decision carries a printable reason naming the input that
settled it.

The manifest holds no state anything else depends on. Every artifact lands where the manual
command would have put it, so deleting `optimize/` resets resume without breaking a single
serving path, and dropping to a manual command mid-flow just changes a fingerprint the next run
notices.

Two stages are NAMED here and not implemented in this build: `DISTILL` (train a student, gate it,
add it to the pool, re-sweep it) and `COMPACT` (the compaction slot reserved between sweep and
fit, which activates when the representation-consistency seam lands). They sit in `STAGE_ORDER`
now so their arrival is additive: ordering, force-from arithmetic, and downstream invalidation
already account for them.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wmo.optimize.policy import write_artifact_atomically


class Stage(StrEnum):
    """One step of `wmo optimize model`, in the order the run walks them."""

    PREFLIGHT = "preflight"
    SWEEP = "sweep"
    DISTILL = "distill"
    COMPACT = "compact"
    FIT = "fit"
    TUNE = "tune"
    REPORT = "report"


STAGE_ORDER: tuple[Stage, ...] = (
    Stage.PREFLIGHT,
    Stage.SWEEP,
    Stage.DISTILL,
    Stage.COMPACT,
    Stage.FIT,
    Stage.TUNE,
    Stage.REPORT,
)
"""Every stage the pipeline knows, reserved slots included. Order is the contract."""

BUILT_STAGES: tuple[Stage, ...] = (
    Stage.PREFLIGHT,
    Stage.SWEEP,
    Stage.FIT,
    Stage.TUNE,
    Stage.REPORT,
)
"""The stages this build actually runs; the rest are reserved slots (see the module docstring)."""

RESERVED_STAGES: tuple[Stage, ...] = tuple(
    stage for stage in STAGE_ORDER if stage not in BUILT_STAGES
)

MANIFEST_DIRNAME = "optimize"
"""Where a run's own artifacts live, under the world model's dir."""

MANIFEST_FILENAME = "optimize-run.json"
MATRIX_FILENAME = "matrix.json"
REPORT_FILENAME = "report.json"

MANIFEST_VERSION = 1
"""Bumped when a recorded fingerprint changes meaning; an older manifest resets rather than lies."""


def file_sha256(path: Path) -> str:
    """The file's SHA-256, or an empty string when it is not there."""
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StageRecord(BaseModel):
    """What one completed stage consumed and what it produced.

    `fingerprint` is a flat map of input name to a short string identifying that input's value:
    a file digest, a scenario-set identity, a knob rendered as text. Flat and stringly on purpose,
    since its only job is equality plus a printable "this is what changed" line.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Stage
    fingerprint: dict[str, str]
    artifact_path: str | None = None
    # What the artifact WAS when this stage wrote it. A SHA-256 for a file nothing else touches;
    # for `policy.json`, which `tune` rewrites in place right after `fit` produces it, the fit
    # provenance instead, so a dialed policy still reads as the fit that produced it.
    artifact_identity: str | None = None
    completed_at: str
    # The two sides of a stage's bill, kept apart because they are different kinds of money: what
    # the candidate models charged (the serving cost a customer would pay) and what the world
    # model charged to run the evaluation (eval infrastructure). Both are 0.0 for the free stages.
    spend_usd: float = 0.0
    # The compressor's share of `spend_usd`, which is folded INTO it (D-COMPRESS accounting:
    # every savings number carries the compressor's inference cost). Recorded separately for one
    # reason: the plan table cannot project a compressor's per-call cost, so a forecast built
    # from this stage has to divide the folded total back down to the projectable part or it
    # compares two different quantities (see `project_sweep_spend`).
    compressor_spend_usd: float = 0.0
    world_model_spend_usd: float = 0.0

    @property
    def total_spend_usd(self) -> float:
        """Both sides together. Only the spend CAP adds them; no report ever quotes this."""
        return self.spend_usd + self.world_model_spend_usd


class RunManifest(BaseModel):
    """Every stage this world model has completed, and under what inputs."""

    model_config = ConfigDict(extra="forbid")

    version: int = MANIFEST_VERSION
    world_model: str
    stages: list[StageRecord] = Field(default_factory=list)
    # Every dollar this model's optimization has ever spent, both sides, across every run.
    # `stages` holds only the LATEST record per stage, so a re-swept stage overwrites its
    # predecessor and the stage rows alone cannot answer "what has this cost so far". A spend cap
    # asks exactly that, so the total accumulates here and is never rewritten downward.
    lifetime_spend_usd: float = 0.0

    def record_for(self, stage: Stage) -> StageRecord | None:
        """The recorded run of `stage`, or None when it has never completed."""
        return next((record for record in self.stages if record.stage == stage), None)

    def with_record(self, record: StageRecord) -> RunManifest:
        """A copy with `record` replacing any earlier run of the same stage, in stage order.

        The replaced record's spend is NOT forgotten: that money left the account whether or not
        the artifact it bought is still the current one, so it stays counted in
        `lifetime_spend_usd`.
        """
        kept = [existing for existing in self.stages if existing.stage != record.stage]
        kept.append(record)
        return self.model_copy(
            update={
                "stages": sorted(kept, key=lambda item: STAGE_ORDER.index(item.stage)),
                "lifetime_spend_usd": self.lifetime_spend_usd + record.total_spend_usd,
            }
        )

    def save(self, path: Path) -> None:
        """Write the manifest atomically; a half-written one must not be loadable.

        Non-atomic would be the worst file in this run to tear: an unreadable manifest resets
        resume, and resetting resume re-buys the sweep, which is the entire spend of the command.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        write_artifact_atomically(path, self.model_dump_json(indent=2).encode("utf-8"))


class ManifestRead(BaseModel):
    """A manifest load: what was usable, and what to tell the operator when something was not."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: RunManifest
    warning: str | None = None  # set when a manifest was present but had to be discarded


def load_manifest(path: Path, *, world_model: str) -> ManifestRead:
    """Read the manifest, resetting to an empty one rather than failing on anything unusable.

    A manifest is a cache of decisions, never the source of truth for an artifact, so a corrupt,
    truncated, or version-skewed one is recoverable by re-running stages. Be clear about the
    price of that: the sweep is the command's entire spend, so a reset re-buys it. That is still
    better than refusing to start, which would strand an operator behind a file they have no
    reason to know how to repair, and every reset says so out loud rather than silently redoing
    paid work. `save` is atomic precisely to keep this path rare.
    """
    empty = RunManifest(world_model=world_model)
    if not path.is_file():
        return ManifestRead(manifest=empty)
    try:
        found = RunManifest.model_validate_json(path.read_bytes())
    except (ValidationError, ValueError) as exc:
        return ManifestRead(
            manifest=empty,
            warning=(
                f"{path} could not be read as a run manifest ({type(exc).__name__}), so this run "
                "starts from a clean one: every stage is re-planned from what is on disk. Nothing "
                f"was deleted; remove {path} to silence this."
            ),
        )
    if found.version != MANIFEST_VERSION:
        return ManifestRead(
            manifest=empty,
            warning=(
                f"{path} was written by manifest version {found.version} and this build reads "
                f"version {MANIFEST_VERSION}, so its records cannot be trusted to mean the same "
                "thing; every stage is re-planned from what is on disk."
            ),
        )
    if found.world_model != world_model:
        return ManifestRead(
            manifest=empty,
            warning=(
                f"{path} records a run of world model '{found.world_model}', not "
                f"'{world_model}'; every stage is re-planned from what is on disk."
            ),
        )
    return ManifestRead(manifest=found)


class StageStatus(StrEnum):
    """Whether a stage will do its work on this run."""

    RUN = "run"
    SKIP = "skip"


class StageDecision(BaseModel):
    """One stage's verdict plus the reason to print for it, on both skip and rerun."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Stage
    status: StageStatus
    reason: str

    @property
    def will_run(self) -> bool:
        """Whether this stage does its work on this run."""
        return self.status is StageStatus.RUN


def forced_stages(force_from: Stage | None) -> frozenset[Stage]:
    """`force_from` and everything downstream of it: what a redo invalidates."""
    if force_from is None:
        return frozenset()
    start = STAGE_ORDER.index(force_from)
    return frozenset(STAGE_ORDER[start:])


def decide_stage(
    stage: Stage,
    *,
    manifest: RunManifest,
    fingerprint: dict[str, str],
    artifact: Path | None = None,
    artifact_identity: str | None = None,
    forced: bool,
    skip_summary: str,
) -> StageDecision:
    """Whether `stage` can be skipped, and the sentence explaining either answer.

    Skipped if and only if all three hold: the stage completed before, every input fingerprint
    still matches, and the artifact it produced is still on disk and still the one it wrote. The
    artifact check is what makes the manifest safe to trust: a matrix deleted by hand, or a policy
    replaced by a manual `route fit`, reruns instead of being assumed.

    Args:
        stage: The stage being decided.
        manifest: The run's recorded history.
        fingerprint: This run's live input fingerprints, same keys as the recording.
        artifact: The file the stage produces, or None for a stage that produces none.
        artifact_identity: What that file identifies as RIGHT NOW, compared against what the
            record says the stage wrote (see `StageRecord.artifact_identity`). None skips the
            content half and checks existence only.
        forced: `--force-from` selected this stage or something upstream of it.
        skip_summary: What was unchanged, phrased for the skip line.
    """
    if forced:
        return StageDecision(stage=stage, status=StageStatus.RUN, reason="forced by --force-from")
    record = manifest.record_for(stage)
    if record is None:
        return StageDecision(stage=stage, status=StageStatus.RUN, reason="never completed here")
    changed = _first_difference(record.fingerprint, fingerprint)
    if changed is not None:
        return StageDecision(stage=stage, status=StageStatus.RUN, reason=changed)
    if artifact is not None:
        if not artifact.is_file():
            return StageDecision(
                stage=stage, status=StageStatus.RUN, reason=f"{artifact.name} is no longer on disk"
            )
        if (
            record.artifact_identity is not None
            and artifact_identity is not None
            and artifact_identity != record.artifact_identity
        ):
            return StageDecision(
                stage=stage,
                status=StageStatus.RUN,
                reason=f"{artifact.name} changed on disk since this run wrote it",
            )
    current = f"{artifact.name} is current" if artifact is not None else "already done"
    return StageDecision(stage=stage, status=StageStatus.SKIP, reason=f"{current}: {skip_summary}")


def _first_difference(recorded: dict[str, str], live: dict[str, str]) -> str | None:
    """The first input that differs, phrased for an operator, or None when they agree.

    First rather than all: the point is to name a cause, and a stage reruns whole either way.
    Keys are walked in the live fingerprint's order so the message is stable across runs.
    """
    for key, value in live.items():
        if key not in recorded:
            return f"{key} is a new input in this build"
        if recorded[key] != value:
            return f"{key} changed ({_short(recorded[key])} -> {_short(value)})"
    for key in recorded:
        if key not in live:
            return f"{key} is no longer an input"
    return None


_SHOWN = 12
"""Characters of a fingerprint value shown in a change message: enough to see it moved."""


def _short(value: str) -> str:
    """A fingerprint value truncated to something readable in a one-line change message."""
    return value if len(value) <= _SHOWN else f"{value[:_SHOWN]}…"


class SweepSpendProjection(BaseModel):
    """What a sweep is projected to spend on BOTH sides, and what that projection rests on.

    The candidate side is projectable from arithmetic: real cell and call counts times an assumed
    per-call token budget at each entry's own price. The world-model side is not, because nothing
    predicts how many tokens the simulator and its judge will spend on an episode before the
    episode runs. It is also not small: measured on a real tau corpus it was 7.0x the candidate
    side ($1.81 against $0.26), so treating it as zero makes a spend cap that misses most of the
    money.

    What IS available, once a model has been swept even once, is that model's own measured ratio.
    Projecting this sweep's world-model side as `candidate x (prior world-model / prior candidate)`
    is a forecast from one prior observation, so `basis` says exactly which one and a caller
    quotes it wherever the number appears. With no usable prior the world-model side is 0.0 and
    `basis` is None, which means "not projected", never "projected to be free".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_usd: float
    world_model_usd: float
    basis: str | None

    @property
    def total_usd(self) -> float:
        """Both sides of the projection: what the cap is checked against."""
        return self.candidate_usd + self.world_model_usd

    @property
    def projected(self) -> bool:
        """Whether the world-model side rests on evidence rather than being left out."""
        return self.basis is not None


def project_sweep_spend(candidate_usd: float, prior: StageRecord | None) -> SweepSpendProjection:
    """Project a sweep's total spend, using a prior sweep's measured ratio when there is one.

    `prior` is the manifest's own record of the last completed sweep OF THIS MODEL, which is the
    only place both sides of that sweep were recorded together (the persisted `kind="sweep"` run
    record holds the world-model side alone, so it cannot supply a ratio by itself).

    A prior whose candidate side measured zero cannot supply a ratio at all: dividing by it would
    be undefined, and treating the world-model figure as an absolute carry-over would forecast
    this sweep from another sweep's size rather than its shape. That case reports "not projected".
    """
    if prior is None or prior.stage is not Stage.SWEEP or prior.spend_usd <= 0.0:
        return SweepSpendProjection(candidate_usd=candidate_usd, world_model_usd=0.0, basis=None)
    # Divide by the PROJECTABLE part of the prior candidate spend, not the folded total. The
    # number this ratio multiplies is a projection that excludes the compressor by construction
    # (nothing predicts a compressor's per-call cost in advance), so a folded denominator would
    # shrink the forecast by the compressor's share of the last run: systematically short, on the
    # term that dominates the bill.
    projectable = prior.spend_usd - prior.compressor_spend_usd
    if projectable <= 0.0:
        return SweepSpendProjection(candidate_usd=candidate_usd, world_model_usd=0.0, basis=None)
    ratio = prior.world_model_spend_usd / projectable
    return SweepSpendProjection(
        candidate_usd=candidate_usd,
        world_model_usd=candidate_usd * ratio,
        # Four decimals, not two: a candidate side under a cent rounds to "$0.00" at two, and a
        # line reading "measured $0.12 world-model against $0.00 candidate (90.9x)" contradicts
        # itself and looks exactly like the divide-by-zero this function refuses to do.
        basis=(
            f"the last sweep of this model measured ${prior.world_model_spend_usd:.4f} "
            f"world-model against ${projectable:.4f} projectable candidate ({ratio:.1f}x)"
        ),
    )


class BudgetExceeded(Exception):
    """The run's spend cap would be crossed by the next stage, so the run stops cleanly."""


class SpendLedger(BaseModel):
    """This run's metered spend against `--max-usd`, checked at paid stage boundaries.

    The cap is a stop, not a clamp: a stage either runs whole or does not start, because half a
    sweep is not a cheaper sweep, it is an unusable matrix that was paid for anyway.

    This is the ONE place the candidate side and the world-model side are added together, because
    a cap is a question about total money leaving the account and both sides do. Everywhere a
    number is reported they stay apart (see `StageRecord`). The consequence worth knowing: the
    estimate checked BEFORE a stage is candidate-side only, since the simulator's own token use
    per episode is not projectable, so a sweep can carry the run past the cap and the overrun is
    caught at the next paid boundary (including a later run's, which seeds from the manifest)
    rather than mid-sweep. Free stages are never gated: refusing to run something that costs
    nothing saves nothing.
    """

    model_config = ConfigDict(extra="forbid")

    max_usd: float | None = None
    spent_usd: float = 0.0

    def record(self, amount: float) -> None:
        """Add a completed stage's measured spend to the run's total."""
        self.spent_usd += amount

    def check(self, stage: Stage, estimate_usd: float, *, basis: str | None = None) -> None:
        """Refuse to start `stage` when its projection would carry the run past the cap.

        `basis` names where a projection that is not pure arithmetic came from, and is quoted in
        the stop message: an operator told a run cannot start is owed the reasoning, especially
        when part of the figure is a forecast from one prior observation.

        Raises:
            BudgetExceeded: The cap is set and would be crossed, with the arithmetic quoted.
        """
        if self.max_usd is None:
            return
        if self.spent_usd + estimate_usd > self.max_usd:
            raise BudgetExceeded(
                f"stage '{stage.value}' is projected at ${estimate_usd:.2f} and this run has "
                f"spent ${self.spent_usd:.2f} of its ${self.max_usd:.2f} cap, so starting it "
                "would cross the cap" + (f" (projection basis: {basis})" if basis else "")
            )
