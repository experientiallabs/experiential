"""Explicit process entrypoint for a finite local or infrastructure-hosted learner."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol, cast

from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings
from exp.optimize.claas.service.configuration import RunReport
from exp.optimize.claas.service.contracts import LearnerRuntimeFactory
from exp.optimize.claas.service.execution import execute_run
from exp.optimize.claas.service.files import MAXIMUM_CONFIGURATION_BYTES, load_configuration
from exp.optimize.claas.service.launch_configuration import RunLaunchConfiguration

logger = logging.getLogger(__name__)


class RuntimeModule(Protocol):
    """Only an explicitly selected runtime extension imports heavyweight GPU packages."""

    ResidentVerlFactory: Callable[[ResidentVerlSettings], LearnerRuntimeFactory]


class PersistenceModule(Protocol):
    """A hosting adapter supplies its own durable mount commit operation."""

    commit_volume: Callable[[], Awaitable[None]]


def run_configuration(configuration: RunLaunchConfiguration) -> RunReport:
    """Validate credentials before selecting veRL, then own the entire finite process run."""
    api_key = os.environ.get(configuration.authentication_env)
    if configuration.run.mode == "run" and (api_key is None or len(api_key) < 16):
        raise ValueError(
            f"set {configuration.authentication_env} to an API key of at least 16 characters"
        )
    runtime = cast(
        RuntimeModule, importlib.import_module("exp.optimize.claas.backends.verl.runtime")
    )
    persist: Callable[[], Awaitable[None]] | None = None
    if configuration.persistence == "modal-volume":
        adapter = cast(
            PersistenceModule,
            importlib.import_module("exp.optimize.claas.backends.modal.persistence"),
        )
        persist = adapter.commit_volume
    return asyncio.run(
        _process(
            configuration, runtime.ResidentVerlFactory(configuration.runtime), api_key, persist
        )
    )


async def _process(
    configuration: RunLaunchConfiguration,
    factory: LearnerRuntimeFactory,
    api_key: str | None,
    persist: Callable[[], Awaitable[None]] | None,
) -> RunReport:
    """Register stop handling before exposing the process ID to a Sandbox owner."""
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    owner = asyncio.current_task()

    def request_stop() -> None:
        """Stop HTTP admission or cancel an owned burst while preserving cleanup ownership."""
        stop.set()
        if configuration.run.mode == "burst" and owner is not None:
            owner.cancel()

    signals = (signal.SIGINT, signal.SIGTERM)
    for item in signals:
        loop.add_signal_handler(item, request_stop)
    pid_path = (
        Path("/tmp/claas-service.pid") if configuration.persistence == "modal-volume" else None
    )
    owns_pid = False
    try:
        if pid_path is not None:
            descriptor = os.open(pid_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            owns_pid = True
            with os.fdopen(descriptor, "w", encoding="ascii") as stream:
                stream.write(str(os.getpid()))
        return await execute_run(
            configuration, factory, api_key=api_key, stop=stop, persist=persist
        )
    finally:
        if pid_path is not None and owns_pid:
            pid_path.unlink(missing_ok=True)
        for item in signals:
            loop.remove_signal_handler(item)


def main() -> int:
    """Read bounded launch JSON and return nonzero for failed or interrupted learning runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--config-stdin", action="store_true")
    arguments = parser.parse_args()
    try:
        if arguments.config_stdin:
            payload = sys.stdin.buffer.read(MAXIMUM_CONFIGURATION_BYTES + 1)
            if len(payload) > MAXIMUM_CONFIGURATION_BYTES:
                raise ValueError("launch configuration exceeds 128 KiB")
            configuration = RunLaunchConfiguration.model_validate_json(payload)
        else:
            configuration = load_configuration(arguments.config)
        report = run_configuration(configuration)
        return 0 if report.status.state == "closed" else 1
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.warning(
            "learning run interrupted; inspect the retained run-report.json before resuming"
        )
        return 130
    except Exception:  # noqa: BLE001 - process entrypoint preserves diagnostic traceback
        logger.exception("learning run failed; inspect retained state before retrying")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
