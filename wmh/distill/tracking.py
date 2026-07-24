"""Optional Weights & Biases tracking for distillation runs.

`build_tracker` turns the run config's `[wandb]` section into a
`DistillTracker`: the no-op `NullTracker` when tracking is disabled (the
default), or a `WandbTracker` streaming step metrics, eval solve rates, and
the final gate summary to a wandb run. The wandb SDK stays an optional extra
(lazy import, mirroring the tinker SDK in `wmh.providers.tinker`), and
credentials are checked at init so a misconfigured run fails fast BEFORE any
paid rollout. After a successful init the contract inverts: a wandb failure
mid-run (network blip, service outage) logs one warning and every later
tracker call degrades to a no-op, because a dead dashboard must never abort a
paid training run.
"""

from __future__ import annotations

import logging
import netrc
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from wmh.core.types import JsonObject, JsonValue
from wmh.distill.config import DistillConfig

if TYPE_CHECKING:
    from wmh.distill.loop import StepMetrics

logger = logging.getLogger(__name__)

WANDB_API_KEY_ENV = "WANDB_API_KEY"
WANDB_NETRC_MACHINE = "api.wandb.ai"

_MISSING_WANDB_EXTRA = (
    "the wandb SDK is not installed; run `uv sync --extra distill` to enable "
    "[wandb] run tracking, or set wandb.enabled = false in the distill config"
)


class DistillTracker(Protocol):
    """The tracking slice the distillation loop emits to.

    Implementations must never raise from `log_step`, `log_eval`,
    `log_summary`, or `finish` once constructed: the loop calls them inline
    with paid training work.
    """

    def log_step(self, step: int, metrics: StepMetrics) -> None:
        """Record one training step's metrics row."""
        ...

    def log_eval(self, name: str, solve_rate: float, step: int | None) -> None:
        """Record one eval batch's solve rate (None step means pre-training)."""
        ...

    def log_summary(
        self,
        *,
        gate_accepted: bool,
        gate_reason: str,
        teacher_solve_rate: float,
        student_before_solve_rate: float,
        student_after_solve_rate: float,
        total_usd: float,
        steps_completed: int,
    ) -> None:
        """Record the run's terminal outcome (the gate verdict and totals)."""
        ...

    def finish(self) -> None:
        """Flush and close the tracking run (idempotent best effort)."""
        ...


class NullTracker:
    """The disabled tracker: every call is a no-op."""

    def log_step(self, step: int, metrics: StepMetrics) -> None:
        """No-op."""

    def log_eval(self, name: str, solve_rate: float, step: int | None) -> None:
        """No-op."""

    def log_summary(
        self,
        *,
        gate_accepted: bool,
        gate_reason: str,
        teacher_solve_rate: float,
        student_before_solve_rate: float,
        student_after_solve_rate: float,
        total_usd: float,
        steps_completed: int,
    ) -> None:
        """No-op."""

    def finish(self) -> None:
        """No-op."""


# -- the wandb SDK slice (typed so the lazy import stays checkable) -------------------------------


class WandbSummaryLike(Protocol):
    """The run-summary slice: `update` with plain JSON values."""

    def update(self, values: Mapping[str, JsonValue]) -> None:
        """Merge values into the run's summary."""
        ...


class WandbRunLike(Protocol):
    """The active-run slice: only the summary is touched directly."""

    @property
    def summary(self) -> WandbSummaryLike:
        """The run's summary mapping."""
        ...


class WandbModuleLike(Protocol):
    """The module-level wandb surface the tracker drives."""

    def init(
        self,
        *,
        project: str,
        entity: str | None,
        name: str,
        tags: list[str],
        # `dir` shadows the builtin, but it is the wandb SDK's own keyword name.
        dir: str,
        config: JsonObject,
    ) -> WandbRunLike:
        """Start a wandb run."""
        ...

    def log(self, data: Mapping[str, JsonValue], *, step: int) -> None:
        """Log one row of metrics at a step."""
        ...

    def finish(self) -> None:
        """Flush and close the active run."""
        ...


def _import_wandb() -> WandbModuleLike:
    """Lazily import the optional wandb SDK (the distill extra).

    Raises:
        ImportError: If the SDK is not installed; the message names the fix.
    """
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(_MISSING_WANDB_EXTRA) from exc
    return cast("WandbModuleLike", wandb)


def _netrc_has_wandb_login() -> bool:
    """Whether ~/.netrc carries an api.wandb.ai entry (a prior `wandb login`)."""
    try:
        entry = netrc.netrc().authenticators(WANDB_NETRC_MACHINE)
    except (FileNotFoundError, netrc.NetrcParseError):
        return False
    return entry is not None


def _require_wandb_credentials() -> None:
    """Fail fast when no wandb credentials exist (before any paid work).

    Raises:
        ValueError: When WANDB_API_KEY is unset AND no api.wandb.ai entry
            exists in ~/.netrc; the message names both fixes.
    """
    if os.environ.get(WANDB_API_KEY_ENV):
        return
    if _netrc_has_wandb_login():
        return
    raise ValueError(
        f"[wandb] tracking is enabled but no credentials were found: "
        f"{WANDB_API_KEY_ENV} is not set in the environment and ~/.netrc has no "
        f"{WANDB_NETRC_MACHINE} entry. Set {WANDB_API_KEY_ENV} to your API key, or "
        "run `wandb login` once to store it in ~/.netrc"
    )


def _flatten_step_metrics(metrics: StepMetrics) -> dict[str, float | int]:
    """One step's metrics row as flat, namespaced numeric wandb keys.

    Per-meter token counts land under `tokens/`, the step's priced spend
    under `cost/usd`, and every other number under `train/`. Non-numeric
    fields (the sampler path) and unscored values (`reverse_kl_per_token`
    when None) are dropped: wandb charts numbers.
    """
    explicit = {
        "student_prefill_tokens": "tokens/student_prefill",
        "student_sample_tokens": "tokens/student_sample",
        "student_train_tokens": "tokens/student_train",
        "teacher_prefill_tokens": "tokens/teacher_prefill",
        "usd": "cost/usd",
    }
    payload: dict[str, float | int] = {}
    for key, value in metrics.model_dump(mode="json").items():
        if not isinstance(value, int | float) or isinstance(value, bool):
            continue
        payload[explicit.get(key, f"train/{key}")] = value
    return payload


class WandbTracker:
    """Streams a distillation run to Weights & Biases.

    Construction is strict (missing SDK or credentials raise, so a
    misconfigured run fails before spending anything); logging is forgiving
    (after a successful init, the first wandb failure logs one warning and
    every later call becomes a no-op, because a dead dashboard must never
    abort a paid training run).

    Args:
        cfg: The validated run config; its `[wandb]` section names the run
            and its snapshot dump becomes the wandb run config.
        run_dir: The run's artifact directory; wandb files land under it.
        agent_name: The agent being distilled; names the run when
            `wandb.run_name` is unset.

    Raises:
        ImportError: If the wandb SDK is not installed (the distill extra).
        ValueError: If no credentials exist (see `_require_wandb_credentials`).
    """

    def __init__(self, cfg: DistillConfig, run_dir: Path, agent_name: str) -> None:
        wandb = _import_wandb()
        _require_wandb_credentials()
        run_name = cfg.wandb.run_name or f"{agent_name}-{run_dir.name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._wandb = wandb
        self._run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            tags=list(cfg.wandb.tags),
            dir=str(run_dir),
            # The same plain dict snapshot_toml renders, so the wandb run
            # config matches the run dir's config.toml exactly.
            config=cfg.model_dump(mode="json", exclude_none=True),
        )
        self._dead = False
        logger.info("wandb tracking started: project %s, run %s", cfg.wandb.project, run_name)

    def _guarded(self, action: Callable[[], None]) -> None:
        """Run one wandb call, degrading to a no-op if the dashboard dies.

        The first failure logs one warning and marks the tracker dead; every
        later call (including `finish`) is skipped silently. Training must
        keep going: the run's own artifacts (metrics.jsonl, eval reports) are
        unaffected by a lost dashboard.
        """
        if self._dead:
            return
        try:
            action()
        except Exception:  # noqa: BLE001 - any wandb failure degrades, never aborts the run
            self._dead = True
            logger.warning(
                "wandb logging failed; tracking is disabled for the rest of the run "
                "(training continues; the run dir keeps every artifact)",
                exc_info=True,
            )

    def log_step(self, step: int, metrics: StepMetrics) -> None:
        """Log one training step's flattened metrics row."""
        payload = _flatten_step_metrics(metrics)
        self._guarded(lambda: self._wandb.log(cast("Mapping[str, JsonValue]", payload), step=step))

    def log_eval(self, name: str, solve_rate: float, step: int | None) -> None:
        """Log one eval batch's solve rate under `eval/<name>`."""
        at_step = step if step is not None else 0
        self._guarded(lambda: self._wandb.log({f"eval/{name}": solve_rate}, step=at_step))

    def log_summary(
        self,
        *,
        gate_accepted: bool,
        gate_reason: str,
        teacher_solve_rate: float,
        student_before_solve_rate: float,
        student_after_solve_rate: float,
        total_usd: float,
        steps_completed: int,
    ) -> None:
        """Record the run's terminal outcome in the wandb run summary."""
        values: dict[str, JsonValue] = {
            "gate_accepted": gate_accepted,
            "gate_reason": gate_reason,
            "teacher_solve_rate": teacher_solve_rate,
            "student_before_solve_rate": student_before_solve_rate,
            "student_after_solve_rate": student_after_solve_rate,
            "total_usd": total_usd,
            "steps_completed": steps_completed,
        }
        self._guarded(lambda: self._run.summary.update(values))

    def finish(self) -> None:
        """Flush and close the wandb run (skipped once the tracker is dead)."""
        self._guarded(self._wandb.finish)


def build_tracker(cfg: DistillConfig, run_dir: Path, agent_name: str) -> DistillTracker:
    """The tracker for one run: `WandbTracker` when enabled, else `NullTracker`.

    Args:
        cfg: The validated run config (`cfg.wandb.enabled` decides).
        run_dir: The run's artifact directory.
        agent_name: The agent being distilled (names the default wandb run).

    Returns:
        The tracker the loop should emit to.

    Raises:
        ImportError: Tracking enabled but the wandb SDK is not installed.
        ValueError: Tracking enabled but no wandb credentials exist.
    """
    if not cfg.wandb.enabled:
        return NullTracker()
    return WandbTracker(cfg, run_dir, agent_name)
