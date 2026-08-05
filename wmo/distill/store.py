"""Distillation artifacts on disk: the per-run directory and the adapter store.

`DistillRunStore` owns one run's artifact directory. Everything a run produces
or needs to resume lives under it:

    <run_dir>/
      config.toml         # exact snapshot of the DistillConfig the run started with
      metrics.jsonl       # one JSON row per warmup/training step, appended as steps finish
      spend.json          # cumulative priced USD across every session, updated per charge
      warmup.json         # the WarmupRecord marker; its existence means warmup is done
      warmup-trials.json  # the warmup collection's TrialRecords (loadable by other runs)
      evals/<name>.json   # interim and final eval payloads, keyed by eval name
      samples/<name>.md   # per-batch sample episode rollouts (step-NNNN, warmup, eval-<name>)
      checkpoints.json    # manifest of saved tinker:// state + sampler paths, plus the
                          # degeneration tripwires' baseline (it must survive --resume)
      gate.json           # the DistillGateRecord verdict
      model_card.json     # the run's DistillModelCard
      handoff.toml        # the [models.agent] serving snippet for the user
      harbor/step-NNNN/   # per-step harbor jobs dirs (harbor source; each trial dir's
                          # result.json carries that trial's token spans)
      tau2/step-NNNN/     # per-step tau2 episode dirs (tau2 source; per-episode results.json
                          # copies plus the spans/ sink dir of recorded token spans)
      warmup-rollouts/    # the warmup phase's isolated rollout root (teacher trials)

`AdapterStore` mirrors `wmo/harness/store.py`'s `HarnessStore` idiom for the
trained adapters themselves: `.wmo/adapters/<name>/` accumulates immutable
`vN/model_card.json` versions, and movable aliases in `aliases.toml` mark
deployment state (promotion and rollback are re-pointing, never rewriting).
"""

from __future__ import annotations

import json
import logging
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wmo.config.store import validate_name
from wmo.core.files import write_text_atomic
from wmo.core.locks import file_write_lock
from wmo.core.types import JsonObject
from wmo.distill.config import DistillConfig, load_distill_config, snapshot_toml
from wmo.distill.gate import DistillGateRecord
from wmo.distill.tokens import TrialRecord
from wmo.distill.tripwire import TripwireBaseline
from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry
from wmo.utils.waterfall import ChatMaxTokensField

logger = logging.getLogger(__name__)


ADAPTERS_DIR = "adapters"
CHAMPION_ALIAS = "champion"

# Tinker's OpenAI-compatible serving endpoint: the production service base URL
# plus the /oai/api/v1 OpenAI-compat prefix (a bare /v1 returns 404; verified
# live 2026-07-23 with a completion from a trained adapter). The endpoint is
# beta, so runs may override it in config; artifacts record the value used.
DEFAULT_TINKER_OPENAI_ENDPOINT = (
    "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
)

_ALIASES_FILE = "aliases.toml"
MODEL_CARD_FILE = "model_card.json"
"""The per-run and per-adapter-version model card filename."""

_CONFIG_FILE = "config.toml"
_METRICS_FILE = "metrics.jsonl"
_SPEND_FILE = "spend.json"
_WARMUP_FILE = "warmup.json"
_WARMUP_TRIALS_FILE = "warmup-trials.json"
_CHECKPOINTS_FILE = "checkpoints.json"
_GATE_FILE = "gate.json"
_HANDOFF_FILE = "handoff.toml"
_EVALS_DIR = "evals"
_SAMPLES_DIR = "samples"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class SpendLedger(BaseModel):
    """The spend.json shape: cumulative priced USD across every session of the run."""

    model_config = ConfigDict(extra="forbid")

    total_usd: float = Field(ge=0.0)


class WarmupRecord(BaseModel):
    """The warmup.json shape: proof one session finished (or skipped) warmup.

    The record exists exactly when the warmup phase reached its terminal
    outcome, so a resumed run never re-runs it. It is NOT a checkpoint
    manifest entry: checkpoint steps drive the resume step count, and warmup
    precedes step 0. `state_path` lets a resume that has no step checkpoint
    yet restore the warmed weights instead of starting cold.
    """

    model_config = ConfigDict(extra="forbid")

    steps: int = Field(ge=0)
    """Warmup optimizer steps actually applied (0 when the phase was skipped)."""

    trials: int = Field(ge=0)
    """Teacher trials collected on the train split."""

    kept_trials: int = Field(ge=0)
    """Trials that survived the `warmup.keep` filter."""

    datums: int = Field(ge=0)
    """SFT datums the kept trials merged into."""

    state_path: str | None = None
    """The post-warmup tinker:// training-state path; None when skipped."""

    sampler_path: str | None = None
    """The post-warmup tinker:// sampler-weights path; None when skipped."""

    skipped_reason: str | None = None
    """Why the phase trained nothing (e.g. zero passing trials); None when it ran."""


class WarmupTrialsManifest(BaseModel):
    """The warmup-trials.json shape: the warmup collection's assembled trials.

    Written when the warmup phase's teacher rollouts finish assembling, BEFORE
    the `warmup.keep` filter, so another run loading the collection through
    `warmup.trajectories_from` may filter differently. The teacher identity is
    recorded so a loading run can refuse trajectories another teacher sampled.
    """

    model_config = ConfigDict(extra="forbid")

    teacher_model: str = Field(min_length=1)
    """The teacher identity (checkpoint or model ref) that sampled the trials."""

    records: list[TrialRecord]
    """Every assembled trial record from the collection, unfiltered."""


class CheckpointRecord(BaseModel):
    """One saved training checkpoint: the exact tinker:// artifacts for a step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(ge=0)
    """0-based training step the checkpoint was saved after."""

    state_path: str = Field(min_length=1)
    """The tinker:// training-state path (feeds load_state on resume)."""

    sampler_path: str = Field(min_length=1)
    """The tinker:// sampler-weights path current at save time."""


class CheckpointManifest(BaseModel):
    """The checkpoints.json shape: every saved checkpoint, plus resume-scoped state."""

    model_config = ConfigDict(extra="forbid")

    checkpoints: list[CheckpointRecord] = Field(default_factory=list)

    tripwire_baseline: TripwireBaseline | None = None
    """The degeneration tripwires' reference, measured at the run's first
    training step (None before that step, or on manifests predating the field).

    It rides the run manifest rather than a metrics row because it must survive
    `--resume`: a resumed session that re-measures its baseline would anchor on
    an already-degenerated policy, making the collapse the new normal and the
    tripwire blind. See `wmo.distill.tripwire.TripwireBaseline`."""


class DistillModelCard(BaseModel):
    """The durable record of one distilled adapter: exact refs plus provenance.

    Tinker's model lineup churns, so the card pins the exact base model,
    teacher, and tinker:// artifact paths rather than relying on names staying
    resolvable. `name` and `version` are stamped by `AdapterStore.save_version`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    version: int = 0
    base_model: str = Field(min_length=1)
    lora_rank: int = Field(ge=1)
    teacher_model: str = Field(min_length=1)
    sampler_path: str = Field(min_length=1)
    """The tinker:// sampler-weights path serving this adapter."""

    state_path: str | None = None
    """The tinker:// training-state path, for training this adapter further."""

    steps_completed: int = Field(ge=0)
    created_at: str = Field(default_factory=_utc_now_iso)
    gate: DistillGateRecord | None = None
    """The promotion verdict that admitted this version, when one was run."""


class DistillRunStore:
    """One distillation run's artifact directory (layout in the module docstring)."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    @property
    def config_path(self) -> Path:
        return self.run_dir / _CONFIG_FILE

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / _METRICS_FILE

    @property
    def spend_path(self) -> Path:
        return self.run_dir / _SPEND_FILE

    @property
    def warmup_path(self) -> Path:
        return self.run_dir / _WARMUP_FILE

    @property
    def warmup_trials_path(self) -> Path:
        return self.run_dir / _WARMUP_TRIALS_FILE

    @property
    def checkpoints_path(self) -> Path:
        return self.run_dir / _CHECKPOINTS_FILE

    @property
    def gate_path(self) -> Path:
        return self.run_dir / _GATE_FILE

    @property
    def model_card_path(self) -> Path:
        return self.run_dir / MODEL_CARD_FILE

    @property
    def handoff_path(self) -> Path:
        return self.run_dir / _HANDOFF_FILE

    @property
    def evals_dir(self) -> Path:
        return self.run_dir / _EVALS_DIR

    @property
    def samples_dir(self) -> Path:
        return self.run_dir / _SAMPLES_DIR

    # -- config snapshot -----------------------------------------------------------------------

    def snapshot_config(self, cfg: DistillConfig) -> Path:
        """Write the exact config the run started with to `config.toml`."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.config_path, snapshot_toml(cfg))
        return self.config_path

    def load_config(self) -> DistillConfig:
        """Read the snapshotted config back (the resume path's source of truth)."""
        return load_distill_config(self.config_path)

    # -- metrics ---------------------------------------------------------------------------------

    def append_metrics(self, step: int, row: BaseModel) -> None:
        """Append one training step's metrics row to `metrics.jsonl`.

        Args:
            step: 0-based training step the row belongs to; written as the
                row's `step` key.
            row: The step's metrics as a pydantic model; its JSON-mode dump
                becomes the rest of the row.

        Raises:
            ValueError: If `step` is negative, or the row itself carries a
                conflicting `step` field.
        """
        if step < 0:
            raise ValueError(f"metrics step must be >= 0, got {step}")
        data = row.model_dump(mode="json")
        if "step" in data and data["step"] != step:
            raise ValueError(
                f"metrics row carries step {data['step']!r} but was appended as step {step}; "
                "drop the row's own step field or pass the matching step"
            )
        payload: JsonObject = {"step": step, **data}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def read_metrics(self, *, tolerate_partial_tail: bool = False) -> list[JsonObject]:
        """Read every metrics row back, in append order.

        Args:
            tolerate_partial_tail: Drop a half-written FINAL line instead of raising. A run
                that died mid-append (ENOSPC, a run dir copied while it was live) leaves one,
                and a read-only reader of an aborted run has to survive it. Off by default:
                everything that writes or resumes a run wants the damage reported, and only
                the last line is ever excusable (a broken row anywhere above it means the
                file lost content, which no reader may quietly skip). Narrow on purpose: it
                excuses only a line that fails to PARSE. `append_metrics` writes one JSON
                object per line, and no truncation of one parses (every strict prefix of
                `{...}` is invalid JSON), so a last line that parses into a non-object is
                something a torn append cannot produce -- real corruption, and still an error.

        Returns:
            One JSON object per non-empty line; an empty list when no metrics
            were written yet.

        Raises:
            ValueError: If a line is not a JSON object; the message names the
                line so the corrupt row can be inspected or removed.
        """
        try:
            text = self.metrics_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        lines = text.splitlines()
        rows: list[JsonObject] = []
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            is_tail = line_number == len(lines)
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                if tolerate_partial_tail and is_tail:
                    continue
                raise ValueError(
                    f"corrupt metrics row on line {line_number} of {self.metrics_path}: {exc}; "
                    "remove the broken line (each line must be one JSON object) and retry"
                ) from exc
            if not isinstance(parsed, dict):
                # Not excused even at the tail, and even under tolerate_partial_tail: a line
                # that parses is a line that was written whole, so this is content damage, not
                # the torn append that flag exists for.
                raise ValueError(
                    f"corrupt metrics row on line {line_number} of {self.metrics_path}: "
                    "expected a JSON object; remove the broken line and retry"
                )
            rows.append(parsed)
        return rows

    def last_step(self) -> int | None:
        """The highest step recorded in `metrics.jsonl`, or None when empty."""
        steps = [row["step"] for row in self.read_metrics() if isinstance(row.get("step"), int)]
        return max(steps) if steps else None

    def budget_spent(self) -> float:
        """Total USD recorded across metrics rows (their `usd` keys).

        This is only the resume fallback for run dirs that predate the spend
        ledger (`read_spend` returning None): metrics rows land once per
        completed step, so spend since the last row (baseline evals before
        step 0, an interim eval after its step's row, the finalize
        student-after eval) is invisible here, and a resumed session that
        re-appends rows for re-run steps double-counts them (conservative
        direction). Rows without a numeric `usd` key contribute nothing; a
        fresh run reports 0.0.
        """
        total = 0.0
        for row in self.read_metrics():
            usd = row.get("usd")
            if isinstance(usd, int | float) and not isinstance(usd, bool):
                total += float(usd)
        return total

    # -- spend ledger ----------------------------------------------------------------------------

    def write_spend(self, total_usd: float) -> None:
        """Persist the run's cumulative priced spend to `spend.json` (atomically).

        The loop calls this on EVERY budget charge, so the ledger (not the
        metrics rows, which only land when a training step completes) is the
        resume path's source of truth for prior spend. Without it, everything
        charged since the last metrics row (holdout baselines before step 0,
        an interim eval after its step's row, the finalize student-after eval)
        would be forgotten and a resumed run could spend `budget.max_usd`
        again.

        Args:
            total_usd: Cumulative priced USD across every session of the run.

        Raises:
            ValueError: If `total_usd` is negative.
        """
        if total_usd < 0:
            raise ValueError(f"cumulative spend must be >= 0, got {total_usd}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = SpendLedger(total_usd=total_usd).model_dump_json(indent=2)
        # The ledger is rewritten on every charge, and a torn write would
        # block the next resume with a corrupt-ledger error.
        write_text_atomic(self.spend_path, payload)

    def read_spend(self) -> float | None:
        """The ledger's cumulative spend, or None when no ledger exists yet.

        None means either a fresh run dir or one from before the ledger
        existed; resume falls back to `budget_spent()` in that case.

        Raises:
            ValueError: If the ledger file exists but does not parse; the
                message says how to recover.
        """
        try:
            text = self.spend_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            return SpendLedger.model_validate_json(text).total_usd
        except ValidationError as exc:
            raise ValueError(
                f"corrupt spend ledger at {self.spend_path}: {exc}; delete the file to "
                "fall back to summing metrics rows (which may miss eval spend, so "
                "consider lowering budget.max_usd accordingly) and resume"
            ) from exc

    # -- warmup ----------------------------------------------------------------------------------

    def write_warmup(self, record: WarmupRecord) -> Path:
        """Persist the warmup phase's terminal record to `warmup.json`.

        Written exactly once per run, when the phase finished training or
        decided to skip; a resumed session that finds the file never re-runs
        warmup.

        Args:
            record: The phase's outcome.

        Returns:
            The written path.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.warmup_path, record.model_dump_json(indent=2))
        return self.warmup_path

    def read_warmup(self) -> WarmupRecord | None:
        """The recorded warmup outcome, or None when the phase never finished.

        Raises:
            ValueError: If the file exists but does not parse; the message
                says how to recover.
        """
        try:
            text = self.warmup_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            return WarmupRecord.model_validate_json(text)
        except ValidationError as exc:
            raise ValueError(
                f"corrupt warmup record at {self.warmup_path}: {exc}; delete the file "
                "so the resumed run re-runs the warmup phase (already-collected teacher "
                "trials under warmup-rollouts/ resume trial-level through harbor)"
            ) from exc

    def write_warmup_trials(self, manifest: WarmupTrialsManifest) -> Path:
        """Persist the warmup collection's trial manifest to `warmup-trials.json`.

        Written when the warmup teacher rollouts finish assembling (before the
        keep filter), so another run can reuse the collection through
        `warmup.trajectories_from` instead of paying for its own.

        Args:
            manifest: The collection's teacher identity and trial records.

        Returns:
            The written path.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.warmup_trials_path, manifest.model_dump_json(indent=2))
        return self.warmup_trials_path

    def read_warmup_trials(self) -> WarmupTrialsManifest | None:
        """The recorded warmup trial manifest, or None when none was written.

        Raises:
            ValueError: If the file exists but does not parse; the message
                says how to recover.
        """
        try:
            text = self.warmup_trials_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            return WarmupTrialsManifest.model_validate_json(text)
        except ValidationError as exc:
            raise ValueError(
                f"corrupt warmup trial manifest at {self.warmup_trials_path}: {exc}; "
                "re-run the source run's warmup collection to rewrite it, or point "
                "warmup.trajectories_from elsewhere"
            ) from exc

    # -- evals -----------------------------------------------------------------------------------

    def write_eval(self, name: str, payload: BaseModel) -> Path:
        """Write one eval payload to `evals/<name>.json`, replacing any prior one.

        Args:
            name: The eval's key (e.g. "baseline-teacher", "step-0010"); must
                be a safe single path segment.
            payload: The eval report model to persist.

        Returns:
            The written path.
        """
        self.evals_dir.mkdir(parents=True, exist_ok=True)
        path = self.evals_dir / f"{validate_name(name)}.json"
        write_text_atomic(path, payload.model_dump_json(indent=2))
        return path

    # -- sample rollouts -------------------------------------------------------------------------

    def write_samples(self, name: str, text: str) -> Path:
        """Write one batch's rendered sample rollouts to `samples/<name>.md`.

        One file per batch, replacing any prior one (a resumed session that
        re-runs a step rewrites its samples): `step-NNNN` for training
        batches, `warmup` for the warmup collection, `eval-<name>` for eval
        batches.

        Args:
            name: The batch's file stem; must be a safe single path segment.
            text: The assembled document (`wmo.distill.samples.samples_markdown`).

        Returns:
            The written path.
        """
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        path = self.samples_dir / f"{validate_name(name)}.md"
        write_text_atomic(path, text)
        return path

    # -- checkpoints -----------------------------------------------------------------------------

    def record_checkpoint(self, step: int, state_path: str, sampler_path: str) -> CheckpointRecord:
        """Record a saved checkpoint in `checkpoints.json`.

        A record for the same step replaces the earlier one (a resumed run
        re-saves its cadence checkpoints); distinct steps append.

        Args:
            step: 0-based training step the checkpoint was saved after.
            state_path: The tinker:// path save_state returned.
            sampler_path: The tinker:// sampler-weights path current at save.

        Returns:
            The recorded checkpoint.
        """
        record = CheckpointRecord(step=step, state_path=state_path, sampler_path=sampler_path)
        manifest = self._read_manifest()
        manifest.checkpoints = [c for c in manifest.checkpoints if c.step != step]
        manifest.checkpoints.append(record)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.checkpoints_path, manifest.model_dump_json(indent=2))
        return record

    def write_tripwire_baseline(self, baseline: TripwireBaseline) -> Path:
        """Persist the degeneration tripwires' baseline into `checkpoints.json`.

        Written the moment the baseline is measured (the run's first training
        step), not at the next checkpoint cadence: a session that dies in
        between must not lose it, or the resumed session would re-baseline
        against whatever policy it restores. `record_checkpoint` reads the
        manifest before rewriting it, so the two never clobber each other.

        Args:
            baseline: The measured baseline.

        Returns:
            The manifest path.
        """
        manifest = self._read_manifest()
        manifest.tripwire_baseline = baseline
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.checkpoints_path, manifest.model_dump_json(indent=2))
        return self.checkpoints_path

    def read_tripwire_baseline(self) -> TripwireBaseline | None:
        """The recorded tripwire baseline, or None when none was measured yet."""
        return self._read_manifest().tripwire_baseline

    def checkpoints(self) -> list[CheckpointRecord]:
        """Every recorded checkpoint, in record order."""
        return list(self._read_manifest().checkpoints)

    def latest_checkpoint(self) -> CheckpointRecord | None:
        """The highest-step checkpoint (the one `--resume` restores), or None."""
        recorded = self._read_manifest().checkpoints
        return max(recorded, key=lambda record: record.step) if recorded else None

    def _read_manifest(self) -> CheckpointManifest:
        try:
            text = self.checkpoints_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return CheckpointManifest()
        try:
            return CheckpointManifest.model_validate_json(text)
        except ValidationError as exc:
            raise ValueError(
                f"corrupt checkpoint manifest at {self.checkpoints_path}: {exc}; "
                "restore it from the run's backup or delete it and re-save a checkpoint "
                "(deleting loses resume points)"
            ) from exc

    # -- terminal artifacts ----------------------------------------------------------------------

    def write_gate(self, record: DistillGateRecord) -> Path:
        """Persist the promotion verdict to `gate.json`."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.gate_path, record.model_dump_json(indent=2))
        return self.gate_path

    def write_model_card(self, card: DistillModelCard) -> Path:
        """Persist the run's model card to `model_card.json`."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.model_card_path, card.model_dump_json(indent=2))
        return self.model_card_path

    def write_handoff(self, toml_text: str) -> Path:
        """Persist the serving handoff snippet to `handoff.toml`.

        Args:
            toml_text: The snippet, normally from `build_handoff_toml`.

        Returns:
            The written path.

        Raises:
            ValueError: If the text is not valid TOML (the snippet must paste
                cleanly into a settings file); build it via `build_handoff_toml`.
        """
        try:
            tomllib.loads(toml_text)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(
                f"handoff snippet is not valid TOML ({exc}); build it with "
                "build_handoff_toml so it pastes cleanly into a settings file"
            ) from exc
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.handoff_path, toml_text)
        return self.handoff_path


class AdapterStore:
    """Named, versioned distilled adapters under `<root>/adapters/<name>/`.

    Mirrors `HarnessStore`: versions are append-only `vN/` directories (each
    holding one `model_card.json`), and deployment state lives in movable
    aliases, so promotion and rollback never rewrite an artifact.
    """

    def __init__(self, root: str | Path = ".wmo") -> None:
        self.root = Path(root)

    @property
    def adapters_dir(self) -> Path:
        return self.root / ADAPTERS_DIR

    def dir_for(self, name: str) -> Path:
        return self.adapters_dir / validate_name(name)

    # -- enumeration -----------------------------------------------------------------------------

    def list_names(self) -> list[str]:
        if not self.adapters_dir.exists():
            return []
        return sorted(
            d.name for d in self.adapters_dir.iterdir() if d.is_dir() and self.versions(d.name)
        )

    def versions(self, name: str) -> list[int]:
        directory = self.dir_for(name)
        if not directory.exists():
            return []
        found: list[int] = []
        for child in directory.iterdir():
            if child.is_dir() and child.name.startswith("v") and child.name[1:].isdigit():
                found.append(int(child.name[1:]))
        return sorted(found)

    def exists(self, name: str) -> bool:
        return bool(self.versions(name))

    # -- aliases ---------------------------------------------------------------------------------

    def aliases(self, name: str) -> dict[str, int]:
        """The alias table for `name`, empty when it has none.

        Names the file on a decode error: `resolve_version(None)` reads this to find the champion,
        so a bare `tomllib` message would reach an operator as a parse error with no path, for a
        file they never edited.
        """
        path = self.dir_for(name) / _ALIASES_FILE
        if not path.exists():
            return {}
        try:
            parsed = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(
                f"adapter alias file {path} is not valid TOML ({exc}); it maps alias names to "
                f"version numbers under [aliases]. Repair it, or delete it to fall back to the "
                f"latest version until the next `wmo optimize distill run` promotion re-points "
                f"{CHAMPION_ALIAS}"
            ) from exc
        data = parsed.get("aliases", {})
        return {k: v for k, v in data.items() if isinstance(v, int)}

    def set_alias(self, name: str, alias: str, version: int) -> None:
        """Point `alias` at `version` (moving it if it exists). Rollback is re-pointing.

        The write was already atomic; the lock covers the rest of the read-modify-write. Two
        promotions of DIFFERENT aliases each read the same table and the later write drops the
        earlier one, with both reporting success.
        """
        if version not in self.versions(name):
            raise ValueError(f"adapter {name!r} has no version v{version}")
        path = self.dir_for(name) / _ALIASES_FILE
        with file_write_lock(path, what="the adapter alias table"):
            current = self.aliases(name)
            current[alias] = version
            write_text_atomic(path, tomli_w.dumps({"aliases": current}))

    # -- load / save -----------------------------------------------------------------------------

    def resolve_version(self, name: str, ref: str | None = None) -> int:
        """Resolve a version ref: `None` -> champion alias, else latest; `"vN"`/`"N"`; an alias."""
        available = self.versions(name)
        if not available:
            raise FileNotFoundError(
                f"no adapter named {name!r} under {self.adapters_dir} "
                f"(have: {', '.join(self.list_names()) or 'none'})"
            )
        aliases = self.aliases(name)
        if ref is None:
            return aliases.get(CHAMPION_ALIAS, available[-1])
        normalized = ref.removeprefix("v")
        if normalized.isdigit():
            version = int(normalized)
            if version not in available:
                raise ValueError(f"adapter {name!r} has no version v{version}")
            return version
        if ref in aliases:
            return aliases[ref]
        raise ValueError(f"adapter {name!r} has no version or alias {ref!r}")

    def resolve(self, name: str, ref: str | None = None) -> DistillModelCard:
        """Load the model card a ref resolves to (see `resolve_version` for refs).

        Args:
            name: The adapter name.
            ref: A version (`"vN"`/`"N"`), an alias (e.g. "champion"), or None
                for the champion alias falling back to latest.

        Returns:
            The version's model card, stamped with its name and version.
        """
        version = self.resolve_version(name, ref)
        path = self.dir_for(name) / f"v{version}" / MODEL_CARD_FILE
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"adapter {name!r} v{version} has no {MODEL_CARD_FILE} at {path}; the version "
                "directory is corrupt, so re-save the adapter or remove the broken version"
            ) from exc
        card = DistillModelCard.model_validate_json(text)
        return card.model_copy(update={"name": name, "version": version})

    def save_version(
        self, name: str, card: DistillModelCard, *, alias: str | None = CHAMPION_ALIAS
    ) -> int:
        """Write `card` as the next version of `name`; optionally point `alias` at it.

        Versions are append-only: this never touches an existing version
        directory.

        Args:
            name: The adapter name (a safe single path segment).
            card: The model card to persist; its `name` and `version` fields
                are stamped, not read.
            alias: The alias to move to the new version; "champion" by
                default, None to save without moving any alias.

        Returns:
            The new version number.
        """
        validate_name(name)
        version = (self.versions(name)[-1] + 1) if self.exists(name) else 1
        stamped = card.model_copy(update={"name": name, "version": version})
        directory = self.dir_for(name) / f"v{version}"
        directory.mkdir(parents=True, exist_ok=False)  # append-only: collision is a bug
        write_text_atomic(directory / MODEL_CARD_FILE, stamped.model_dump_json(indent=2))
        if alias is not None:
            self.set_alias(name, alias, version)
        return version


def build_handoff_toml(sampler_path: str, *, base_model: str, endpoint: str | None = None) -> str:
    """Build the `[models.agent]` snippet pointing an agent at the distilled student.

    The snippet targets wmo's openai provider with a custom endpoint, which is
    how any OpenAI-compatible server is addressed; Tinker's serving endpoint
    authenticates with the Tinker API key via `WMO_ENDPOINT_API_KEY` (the
    openai provider never sends `OPENAI_API_KEY` to a custom endpoint).

    Args:
        sampler_path: The final tinker:// sampler-weights path to serve.
        base_model: The student's base model name, written as `model_type` so
            capability resolution keys on the model family, not the raw
            weights path (the same shape the CLI's `--promote` writes).
            The snippet also pins `chat_max_tokens_field`, because a
            `tinker://` path is outside the built-in catalog: on Tinker's
            endpoint the resolved default would be `max_completion_tokens`,
            which that endpoint answers with a 400 on every call.
        endpoint: The OpenAI-compatible base URL; defaults to
            `DEFAULT_TINKER_OPENAI_ENDPOINT`.

    Returns:
        A valid TOML document: comment lines explaining auth, then the
        `[models.agent]` table.

    Raises:
        ValueError: If the sampler path is not a tinker:// path, or a value
            cannot be embedded in a basic TOML string.
    """
    if not sampler_path.startswith("tinker://"):
        raise ValueError(
            f"sampler path {sampler_path!r} is not a tinker:// weights path; pass the "
            "path returned by save_weights_for_sampler (recorded in the run's "
            "checkpoints.json and model card)"
        )
    # Stripped before it is STORED, not just before it is classified. `is_tinker_endpoint`
    # tolerates surrounding whitespace so a pasted URL still gets Tinker's credential and
    # output-budget defaults, but the entry's endpoint becomes the OpenAI client's base_url
    # verbatim, so persisting the padded string would classify correctly and then fail every
    # routed call on an unusable URL.
    resolved_endpoint = endpoint.strip() if endpoint is not None else DEFAULT_TINKER_OPENAI_ENDPOINT
    budget_field = (
        STUDENT_CHAT_MAX_TOKENS_FIELD
        if is_tinker_endpoint(resolved_endpoint)
        else "max_completion_tokens"
    )
    values = (
        ("sampler path", sampler_path),
        ("base model", base_model),
        ("endpoint", resolved_endpoint),
    )
    for label, value in values:
        if any(ch in value for ch in ('"', "\\", "\n")):
            raise ValueError(
                f"{label} {value!r} contains a quote, backslash, or newline and cannot be "
                "embedded in the handoff snippet; check the value for corruption"
            )
    return (
        "# Distilled student handoff: point your agent at the trained adapter through\n"
        "# Tinker's OpenAI-compatible serving endpoint.\n"
        "# Auth: set WMO_ENDPOINT_API_KEY to your Tinker API key (the TINKER_API_KEY\n"
        "# value); wmo sends WMO_ENDPOINT_API_KEY, never OPENAI_API_KEY, to custom\n"
        "# endpoints.\n"
        "[models.agent]\n"
        'provider = "openai"\n'
        f'model = "{sampler_path}"\n'
        f'model_type = "{base_model}"\n'
        f'endpoint = "{resolved_endpoint}"\n'
        f'chat_max_tokens_field = "{budget_field}"\n'
    )


# Tinker's OpenAI-compatible endpoint takes the classic `max_tokens`; answering with
# `max_completion_tokens` (what the built-in catalog defaults to, and what a `tinker://` weights
# path resolves to since the catalog has never heard of it) is a 400 on every routed call.
STUDENT_CHAT_MAX_TOKENS_FIELD: ChatMaxTokensField = "max_tokens"

# The pool entry reads the Tinker API key straight from its own env var through the pool's
# trusted explicit-credential channel (`PoolEntry.api_key_env` -> `pool_provider` ->
# `get_provider(api_key=...)`). The `[models.agent]` handoff has no such channel and falls back
# to `WMO_ENDPOINT_API_KEY`, so routing a student needs one fewer copy of the key.
STUDENT_API_KEY_ENV = "TINKER_API_KEY"


def is_tinker_endpoint(endpoint: str) -> bool:
    """Whether `endpoint` addresses Tinker's own OpenAI-compatible service.

    This answer decides which credential gets sent and which output-budget parameter is used, so
    it is compared on URL equivalence rather than string equality. An exact match would read a
    pasted `.../oai/api/v1/`, or a host typed in different case, as somebody else's host: it would
    quietly swap `TINKER_API_KEY` for the `WMO_ENDPOINT_API_KEY` fallback and `max_tokens` for
    `max_completion_tokens`, and leave the student 400ing on a URL that is Tinker's.

    Scheme and host are case-insensitive per RFC 3986 section 3.1 and 3.2.2, so they are lowered.
    The scheme's own default port is redundant (section 3.2.3), so `https://...:443/...` is the
    same endpoint as the canonical URL and keeps its defaults. Any OTHER port is a different
    service and stays significant: `:8443` on Tinker's host is somebody's proxy, not Tinker.
    The PATH is not case-insensitive, and is compared as written: `/oai/api/v1` and `/OAI/API/V1`
    are different resources on a server that chooses to distinguish them, and reading one as the
    other would hand a credential to a route the operator did not name. Trailing slashes and
    surrounding whitespace are stripped either way.
    """
    return _normalize_endpoint(endpoint) == _normalize_endpoint(DEFAULT_TINKER_OPENAI_ENDPOINT)


# The port each scheme already implies, which a URL may spell out without changing what it
# addresses (RFC 3986 section 3.2.3).
_DEFAULT_PORTS: dict[str, int] = {"http": 80, "https": 443}


def _normalize_endpoint(endpoint: str) -> tuple[str, str, str]:
    """An OpenAI-compatible base URL as (scheme, authority, path), normalized where URLs let it."""
    parsed = urlsplit(endpoint.strip())
    scheme = parsed.scheme.lower()
    return scheme, _normalize_authority(scheme, parsed.netloc), parsed.path.rstrip("/")


def _normalize_authority(scheme: str, netloc: str) -> str:
    """`netloc` lowercased, with only the port `scheme` already implies dropped.

    Dropping the redundant default port is what keeps a pasted `:443` on Tinker's own defaults.
    Only that exact spelling is dropped, so a non-default port (`:8443`) stays part of the host
    and reads as a custom endpoint, and so does anything unparseable as the default port
    (`:0443`, `:notaport`): comparing those as written keeps a credential decision on the safe
    side rather than guessing what the operator meant.
    """
    authority = netloc.lower()
    default = _DEFAULT_PORTS.get(scheme)
    if default is None:  # a scheme with no implied port: nothing is redundant
        return authority
    # rpartition leaves a bracketed IPv6 literal alone: "[::1]" splits to a "1]" port, which
    # never matches a default port, so only a real trailing ":443"/":80" is removed.
    host, separator, port = authority.rpartition(":")
    if separator and port == str(default):
        return host
    return authority


def student_pool_entry(
    card: DistillModelCard,
    *,
    name: str,
    input_per_mtok: float,
    output_per_mtok: float,
    endpoint: str | None = None,
    api_key_env: str | None = None,
    chat_max_tokens_field: ChatMaxTokensField | None = None,
) -> PoolEntry:
    """Build the routable pool entry for a distilled student, from its model card.

    This is the seam that makes a trained adapter a routing candidate: the card records the
    `tinker://` weights path and the base model, and this turns that pair into a `PoolEntry` the
    router can select and serving can call, with no hand-edited TOML in between. The entry
    addresses the student as any OpenAI-compatible server is addressed (`kind="openai"` plus an
    `endpoint`), which is the same shape `build_handoff_toml` writes for `[models.agent]`.

    Prices are REQUIRED and have no default. A candidate whose cost is unknown reports $0, which
    would make it unconditionally the cheapest model in the pool and let a cost-aware policy route
    everything to it on evidence that does not exist. `PoolEntry` refuses an unpriced non-built-in
    model for the same reason, and a `tinker://` path is never in the built-in price table.

    The Tinker-specific defaults apply ONLY when the entry actually points at Tinker. Reading
    `TINKER_API_KEY` for a host the caller named would send a Tinker bearer token to that host,
    and the pool's whole credential rule is that an operator pairs a key with an endpoint
    deliberately (see the module docstring of `wmo.providers.pool`), never that a helper picks
    the pairing. A custom endpoint therefore gets `api_key_env=None`, which is the documented
    custom-endpoint convention: the openai provider falls back to `WMO_ENDPOINT_API_KEY` and
    never sends a key the caller did not put there. Its output-budget field likewise falls back
    to the repo-wide default rather than inheriting Tinker's `max_tokens`. Pass `api_key_env` or
    `chat_max_tokens_field` explicitly to override either.

    Args:
        card: The run's model card (`<run_dir>/model_card.json`), or an adapter version's.
        name: The pool handle: the stable name policy artifacts and request logs key on.
        input_per_mtok: Prompt-token price, USD per 1M tokens, at the serving endpoint.
        output_per_mtok: Completion-token price, USD per 1M tokens, at the serving endpoint.
        endpoint: The OpenAI-compatible base URL; defaults to
            `DEFAULT_TINKER_OPENAI_ENDPOINT`.
        api_key_env: Env var holding the endpoint's key. Defaults to `TINKER_API_KEY` on
            Tinker's own endpoint and to None (the provider's `WMO_ENDPOINT_API_KEY`
            fallback) on any other.
        chat_max_tokens_field: Output-budget parameter the endpoint accepts. Defaults to
            `max_tokens` on Tinker's own endpoint and to the repo-wide
            `max_completion_tokens` on any other.

    Returns:
        A validated `PoolEntry` ready for `upsert_pool_entry`.

    Raises:
        ValueError: If the card's sampler path is not a `tinker://` weights path, so there is
            nothing servable to point an entry at.
    """
    if not card.sampler_path.startswith("tinker://"):
        raise ValueError(
            f"model card sampler path {card.sampler_path!r} is not a tinker:// weights path; a "
            "routable student needs the path save_weights_for_sampler returned (recorded in the "
            "run's checkpoints.json and model card)"
        )
    # Stripped before it is STORED, not just before it is classified. `is_tinker_endpoint`
    # tolerates surrounding whitespace so a pasted URL still gets Tinker's credential and
    # output-budget defaults, but the entry's endpoint becomes the OpenAI client's base_url
    # verbatim, so persisting the padded string would classify correctly and then fail every
    # routed call on an unusable URL.
    resolved_endpoint = endpoint.strip() if endpoint is not None else DEFAULT_TINKER_OPENAI_ENDPOINT
    on_tinker = is_tinker_endpoint(resolved_endpoint)
    if api_key_env is None and on_tinker:
        api_key_env = STUDENT_API_KEY_ENV
    if chat_max_tokens_field is None:
        chat_max_tokens_field = (
            STUDENT_CHAT_MAX_TOKENS_FIELD if on_tinker else "max_completion_tokens"
        )
    return PoolEntry(
        name=name,
        kind=ProviderKind.OPENAI,
        model=card.sampler_path,
        # The weights path carries no capability information, so the base model is what
        # temperature and output-budget resolution key on.
        model_type=card.base_model,
        chat_max_tokens_field=chat_max_tokens_field,
        endpoint=resolved_endpoint,
        api_key_env=api_key_env,
        # A distilled student is the small open model in the pool, not the frontier anchor: the
        # improvement report's comparison reads `tier` to tell those roles apart.
        tier="open",
        input_per_mtok=input_per_mtok,
        output_per_mtok=output_per_mtok,
    )
