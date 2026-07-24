"""Distillation artifacts on disk: the per-run directory and the adapter store.

`DistillRunStore` owns one run's artifact directory. Everything a run produces
or needs to resume lives under it:

    <run_dir>/
      config.toml         # exact snapshot of the DistillConfig the run started with
      metrics.jsonl       # one JSON row per training step, appended as steps finish
      spend.json          # cumulative priced USD across every session, updated per charge
      evals/<name>.json   # interim and final eval payloads, keyed by eval name
      checkpoints.json    # manifest of saved tinker:// state + sampler paths
      gate.json           # the DistillGateRecord verdict
      model_card.json     # the run's DistillModelCard
      handoff.toml        # the [models.agent] serving snippet for the user
      harbor/step-NNNN/   # per-step harbor jobs dirs (written by the rollout collector)
      tokens/step-NNNN/   # per-step token sinks (written by the rollout collector)

`AdapterStore` mirrors `wmh/harness/store.py`'s `HarnessStore` idiom for the
trained adapters themselves: `.wmh/adapters/<name>/` accumulates immutable
`vN/model_card.json` versions, and movable aliases in `aliases.toml` mark
deployment state (promotion and rollback are re-pointing, never rewriting).
"""

from __future__ import annotations

import json
import logging
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wmh.config.store import validate_name
from wmh.core.types import JsonObject
from wmh.distill.config import DistillConfig, load_distill_config, snapshot_toml
from wmh.distill.gate import DistillGateRecord

logger = logging.getLogger(__name__)

ADAPTERS_DIR = "adapters"
CHAMPION_ALIAS = "champion"

# Tinker's OpenAI-compatible serving endpoint: the SDK's production service
# base URL plus the /v1 OpenAI-compat prefix. The endpoint is beta, so runs
# may override it in config; artifacts always record the value actually used.
DEFAULT_TINKER_OPENAI_ENDPOINT = "https://tinker.thinkingmachines.dev/services/tinker-prod/v1"

_ALIASES_FILE = "aliases.toml"
_CARD_FILE = "model_card.json"

_CONFIG_FILE = "config.toml"
_METRICS_FILE = "metrics.jsonl"
_SPEND_FILE = "spend.json"
_CHECKPOINTS_FILE = "checkpoints.json"
_GATE_FILE = "gate.json"
_MODEL_CARD_FILE = "model_card.json"
_HANDOFF_FILE = "handoff.toml"
_EVALS_DIR = "evals"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class SpendLedger(BaseModel):
    """The spend.json shape: cumulative priced USD across every session of the run."""

    model_config = ConfigDict(extra="forbid")

    total_usd: float = Field(ge=0.0)


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
    """The checkpoints.json shape: every saved checkpoint, in save order."""

    model_config = ConfigDict(extra="forbid")

    checkpoints: list[CheckpointRecord] = Field(default_factory=list)


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
    def checkpoints_path(self) -> Path:
        return self.run_dir / _CHECKPOINTS_FILE

    @property
    def gate_path(self) -> Path:
        return self.run_dir / _GATE_FILE

    @property
    def model_card_path(self) -> Path:
        return self.run_dir / _MODEL_CARD_FILE

    @property
    def handoff_path(self) -> Path:
        return self.run_dir / _HANDOFF_FILE

    @property
    def evals_dir(self) -> Path:
        return self.run_dir / _EVALS_DIR

    # -- config snapshot -----------------------------------------------------------------------

    def snapshot_config(self, cfg: DistillConfig) -> Path:
        """Write the exact config the run started with to `config.toml`."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(snapshot_toml(cfg), encoding="utf-8")
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

    def read_metrics(self) -> list[JsonObject]:
        """Read every metrics row back, in append order.

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
        rows: list[JsonObject] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"corrupt metrics row on line {line_number} of {self.metrics_path}: {exc}; "
                    "remove the broken line (each line must be one JSON object) and retry"
                ) from exc
            if not isinstance(parsed, dict):
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

        The loop calls this on EVERY budget charge, so the ledger — not the
        metrics rows, which only land when a training step completes — is the
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
        # Atomic replace: the ledger is rewritten on every charge, and a torn
        # write would block the next resume with a corrupt-ledger error.
        tmp_path = self.spend_path.with_name(self.spend_path.name + ".tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(self.spend_path)

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
        path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
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
        self.checkpoints_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return record

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
        self.gate_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return self.gate_path

    def write_model_card(self, card: DistillModelCard) -> Path:
        """Persist the run's model card to `model_card.json`."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model_card_path.write_text(card.model_dump_json(indent=2), encoding="utf-8")
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
        self.handoff_path.write_text(toml_text, encoding="utf-8")
        return self.handoff_path


class AdapterStore:
    """Named, versioned distilled adapters under `<root>/adapters/<name>/`.

    Mirrors `HarnessStore`: versions are append-only `vN/` directories (each
    holding one `model_card.json`), and deployment state lives in movable
    aliases, so promotion and rollback never rewrite an artifact.
    """

    def __init__(self, root: str | Path = ".wmh") -> None:
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
        path = self.dir_for(name) / _ALIASES_FILE
        if not path.exists():
            return {}
        data = tomllib.loads(path.read_text(encoding="utf-8")).get("aliases", {})
        return {k: v for k, v in data.items() if isinstance(v, int)}

    def set_alias(self, name: str, alias: str, version: int) -> None:
        """Point `alias` at `version` (moving it if it exists). Rollback is re-pointing."""
        if version not in self.versions(name):
            raise ValueError(f"adapter {name!r} has no version v{version}")
        current = self.aliases(name)
        current[alias] = version
        path = self.dir_for(name) / _ALIASES_FILE
        path.write_text(tomli_w.dumps({"aliases": current}), encoding="utf-8")

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
        path = self.dir_for(name) / f"v{version}" / _CARD_FILE
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"adapter {name!r} v{version} has no {_CARD_FILE} at {path}; the version "
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
        (directory / _CARD_FILE).write_text(stamped.model_dump_json(indent=2), encoding="utf-8")
        if alias is not None:
            self.set_alias(name, alias, version)
        return version


def build_handoff_toml(sampler_path: str, *, endpoint: str | None = None) -> str:
    """Build the `[models.agent]` snippet pointing an agent at the distilled student.

    The snippet targets wmh's openai provider with a custom endpoint, which is
    how any OpenAI-compatible server is addressed; Tinker's serving endpoint
    authenticates with the Tinker API key via `WMH_ENDPOINT_API_KEY` (the
    openai provider never sends `OPENAI_API_KEY` to a custom endpoint).

    Args:
        sampler_path: The final tinker:// sampler-weights path to serve.
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
    resolved_endpoint = endpoint if endpoint is not None else DEFAULT_TINKER_OPENAI_ENDPOINT
    for label, value in (("sampler path", sampler_path), ("endpoint", resolved_endpoint)):
        if any(ch in value for ch in ('"', "\\", "\n")):
            raise ValueError(
                f"{label} {value!r} contains a quote, backslash, or newline and cannot be "
                "embedded in the handoff snippet; check the value for corruption"
            )
    return (
        "# Distilled student handoff: point your agent at the trained adapter through\n"
        "# Tinker's OpenAI-compatible serving endpoint.\n"
        "# Auth: set WMH_ENDPOINT_API_KEY to your Tinker API key (the TINKER_API_KEY\n"
        "# value); wmh sends WMH_ENDPOINT_API_KEY, never OPENAI_API_KEY, to custom\n"
        "# endpoints.\n"
        "[models.agent]\n"
        'provider = "openai"\n'
        f'model = "{sampler_path}"\n'
        f'endpoint = "{resolved_endpoint}"\n'
    )
