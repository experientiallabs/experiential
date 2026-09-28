"""Opt-in numeric W&B reporting using the official SDK's local queue and finite shutdown."""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from threading import Lock
from types import TracebackType
from typing import Literal

import wandb

from exp.common.core.artifacts import JsonObject, assert_secret_free, canonical_json_bytes
from exp.common.observability.metrics import MetricRecord

LOGGER = logging.getLogger(__name__)


def _ignore_remote_stop() -> None:
    """Keep a metrics UI stop request from interrupting an independently owned learner."""
    LOGGER.warning("W&B requested a stop; metrics reporting does not control the learning run")


class WandbMetricSink:
    """One explicitly selected W&B run; local training evidence remains authoritative.

    The SDK queues ``record`` locally. Neither success nor ``next_step`` proves remote
    delivery, and resuming cannot promise exactly-once metric delivery after a crash.
    Keep a durable local outbox and its domain receipts when replay matters.

    Attributes:
        next_step: Next global history position for this sink, independent of any
            optimizer/evaluation axis included in metric values.
        url: W&B URL for an online run, or ``None`` for an offline run.
    """

    def __init__(
        self,
        *,
        project: str,
        run_id: str,
        directory: Path,
        entity: str | None = None,
        mode: Literal["online", "offline"] = "online",
        config: JsonObject | None = None,
        initialization_timeout_seconds: float = 30,
        finish_timeout_seconds: float = 30,
    ) -> None:
        """Open an explicit run without collecting console, code, machine, or system data.

        Args:
            project: Explicit destination project; never discovered from a checkout.
            run_id: Stable destination identity. Online runs resume this identity.
            directory: Caller-owned local SDK spool directory, retained after failure.
            entity: Optional W&B account/team, resolved by the SDK if omitted.
            mode: Explicit online upload or provider-free offline recording.
            config: Only caller-selected model/hyperparameter/provenance fields. No
                environment or learner configuration is copied automatically.
            initialization_timeout_seconds: Finite SDK initialization deadline.
            finish_timeout_seconds: Finite upload deadline. Timeout raises while the
                SDK retains unuploaded data locally for explicit recovery.

        Raises:
            ValueError: Configuration is invalid or includes a known credential field.
        """
        if not project.strip() or not run_id.strip():
            raise ValueError("W&B project and run_id must be explicit nonempty names")
        if mode not in {"online", "offline"}:
            raise ValueError("W&B metrics mode must be online or offline")
        for timeout in (initialization_timeout_seconds, finish_timeout_seconds):
            if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= 300:
                raise ValueError("W&B timeouts must be finite and between 0 and 300 seconds")
        selected = {} if config is None else config
        assert_secret_free(selected)
        if len(canonical_json_bytes(selected)) > 65_536:
            raise ValueError("W&B selected config exceeds 65536 bytes; reduce its fields")
        # W&B merges process-level config files and sweep/Launch values before a
        # run's explicit config. Per-run config_paths=[] does not clear that state.
        # Inspect the public setup session instead of mutating another caller's settings.
        inherited = wandb.setup().settings
        if (
            inherited.config_paths
            or inherited.sweep_id
            or inherited.sweep_param_path
            or inherited.launch
            or inherited.launch_config_path
            or "weave" in sys.modules
        ):
            raise ValueError(
                "W&B has inherited config, sweep/Launch, or Weave state; "
                "use a separate metrics reporter process with explicit configuration"
            )
        directory.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._closed = False
        self._offline = mode == "offline"
        if mode == "online" and not wandb.login(
            timeout=math.ceil(initialization_timeout_seconds), force=True
        ):
            raise ValueError("W&B authentication failed; run wandb login before reporting online")
        self._run = wandb.init(
            project=project,
            entity=entity,
            id=run_id,
            dir=str(directory),
            mode=mode,
            reinit="create_new",
            resume="allow" if mode == "online" else None,
            config=selected,
            save_code=False,
            sync_tensorboard=False,
            force=True,
            settings=wandb.Settings(
                config_paths=[],
                capture_loggers={},
                sagemaker_disable=True,
                console="off",
                disable_code=True,
                disable_git=True,
                x_disable_meta=True,
                x_disable_stats=True,
                x_disable_machine_info=True,
                x_save_requirements=False,
                disable_job_creation=True,
                stop_fn=_ignore_remote_stop,
                init_timeout=initialization_timeout_seconds,
                finish_timeout=finish_timeout_seconds,
                finish_timeout_raises=True,
            ),
        )
        try:
            if self._run.disabled or (mode == "online" and self._run.offline):
                raise ValueError(
                    "W&B did not open the requested reporting mode; check authentication"
                )
            # The official property retrieves the resumed next history step with
            # its own finite 30-second SDK wait. Later writes use our local counter.
            self._next_step = self._run.step
        except BaseException:
            try:
                self._run.finish(exit_code=1)
            except Exception as error:  # noqa: BLE001 - retain the initiating SDK failure
                LOGGER.warning("W&B initialization cleanup failed (%s)", type(error).__name__)
            raise

    @property
    def next_step(self) -> int:
        """Return the next locally queued global history step, not a remote delivery cursor."""
        with self._lock:
            return self._next_step

    @property
    def url(self) -> str | None:
        """Return the explicit destination URL when the SDK has an online destination."""
        return None if self._offline else self._run.url

    def record(self, record: MetricRecord) -> None:
        """Queue numeric values at one history step, with domain axes supplied as values.

        Explicit steps must be at least ``next_step``. A durable outbox can use this
        cursor to skip already queued history after resuming the same online run.
        No automatic event-ID deduplication or remote acknowledgement is implied.
        """
        # Frozen Pydantic models still contain mutable dictionaries. Revalidate at
        # the egress boundary rather than trusting a caller's later mutation.
        safe = MetricRecord.model_validate(record.model_dump())
        with self._lock:
            if self._closed:
                raise ValueError("W&B metrics sink is closed; open another explicit run")
            step = self._next_step if safe.step is None else safe.step
            if step < self._next_step:
                raise ValueError("metric step precedes next_step; skip already queued outbox rows")
            self._run.log(dict(safe.values), step=step, commit=True)
            self._next_step = step + 1

    def close(self) -> None:
        """Finish once within the configured SDK upload deadline; preserve local spool on error."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._run.finish()

    def __enter__(self) -> WandbMetricSink:
        """Return this caller-owned sink for explicit lifetime management."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Report delivery failure without replacing an already active caller exception."""
        try:
            self.close()
        except Exception as error:  # noqa: BLE001 - preserve the caller's original failure
            if exception is None:
                raise
            LOGGER.warning(
                "W&B metrics cleanup failed (%s); local spool retained", type(error).__name__
            )
