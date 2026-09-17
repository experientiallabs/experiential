"""One named GPU Sandbox per state Volume, with a finite service-owned lifecycle.

All writers must use this adapter. A fixed environment-wide Modal Dict permanently
binds each Volume ID to one App. Modal's named-Sandbox uniqueness then excludes a
second live writer, including launches through another configured App. Reassigning
ownership or writing to the Volume externally is intentionally unsupported.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path

import modal

from exp.optimize.claas.backends.modal.configuration import ModalLaunch
from exp.optimize.claas.backends.modal.ownership import bind_owner, commit_owner
from exp.optimize.claas.backends.modal.transport import (
    download_artifact,
    upload_import,
)
from exp.optimize.claas.backends.modal.validation import validate_launch
from exp.optimize.claas.service.launch_configuration import RunLaunchConfiguration


class ModalRunHandle:
    """An exact finite Sandbox, never an autoscaling per-request deployment."""

    def __init__(
        self,
        sandbox: modal.Sandbox,
        volume: modal.Volume,
        *,
        cleanup_seconds: float,
        endpoint: str | None = None,
    ) -> None:
        """Retain ownership and the graceful cleanup bound for an allocated run."""
        self._sandbox = sandbox
        self._volume = volume
        self._cleanup_seconds = cleanup_seconds
        self._exit_code: int | None = None
        self.endpoint = endpoint

    @property
    def sandbox_id(self) -> str:
        """Return the exact allocated Sandbox identity for inspection and recovery."""
        return self._sandbox.object_id

    async def wait(self) -> int:
        """Await service exit and reject crashes or Modal's hard lifetime timeout."""
        if self._exit_code is None:
            await self._sandbox.wait.aio()
            code = self._sandbox.returncode
            if code is None:
                raise RuntimeError("Modal did not confirm Sandbox exit; retain its run identity")
            self._exit_code = code
            await self._sandbox.detach.aio()
        code = self._exit_code
        if code != 0:
            raise RuntimeError(
                f"learning Sandbox {self.sandbox_id} exited with code {code}; inspect its logs"
            )
        return code

    async def stop(self) -> None:
        """Ask the service to drain, then force termination only after bounded failure.

        A forced stop raises: it proves container termination, not a checkpoint or
        successful acknowledgement. The durable queue determines recovery next run.
        """
        try:
            async with asyncio.timeout(self._cleanup_seconds):
                await self._drain()
        except BaseException:
            if self._exit_code is None:
                await _terminate(self._sandbox)
            raise

    async def _drain(self) -> None:
        """Keep every poll, signal, and join inside the caller's single cleanup deadline."""
        if self._exit_code is not None or await self._sandbox.poll.aio() is not None:
            await self.wait()
            return
        try:
            value = await self._sandbox.filesystem.read_text.aio("/tmp/claas-service.pid")
            if not re.fullmatch(r"[0-9]{1,10}\s?", value) or int(value) <= 0:
                raise ValueError("invalid learning-service PID file")
            signal = await self._sandbox.exec.aio("kill", "-TERM", str(int(value)))
            await signal.wait.aio()
            if signal.returncode != 0:
                raise RuntimeError("learning-service shutdown signal failed")
        except Exception:  # noqa: BLE001 - a natural exit can remove the service PID file
            if await self._sandbox.poll.aio() is None:
                raise
        await self.wait()

    async def download(
        self, relative_path: str, destination: Path, *, maximum_bytes: int = 16_777_216
    ) -> None:
        """Copy one explicitly named committed artifact without overwriting local files."""
        await download_artifact(
            self._volume, relative_path, destination, maximum_bytes=maximum_bytes
        )


class ModalLauncher:
    """Host the same standalone service used locally, with no training policy."""

    def __init__(self, resources: ModalLaunch) -> None:
        """Validate resource references without contacting Modal or creating resources."""
        self.resources = ModalLaunch.model_validate(resources.model_dump())

    async def start(
        self,
        configuration: RunLaunchConfiguration,
        *,
        run_id: str,
        import_path: Path | None = None,
    ) -> ModalRunHandle:
        """Allocate once, stage optional exact evidence, and start one resident service.

        The App, immutable image, and v2 Volume must already exist. A fresh Sandbox
        mounts the latest committed state; no reload occurs with SQLite open.
        ``import_path`` is transported unchanged and parsed by the shared service.
        """
        configuration = RunLaunchConfiguration.model_validate(configuration.model_dump())
        payload = validate_launch(self.resources, configuration, run_id, import_path)
        resources = self.resources
        app = await modal.App.lookup.aio(
            resources.app_name,
            environment_name=resources.environment_name,
            create_if_missing=False,
        )
        volume = modal.Volume.from_name(
            resources.volume_name,
            environment_name=resources.environment_name,
            create_if_missing=False,
            version=2,
        )
        await volume.hydrate.aio()
        await bind_owner(
            volume, app_name=resources.app_name, environment_name=resources.environment_name
        )
        image = await modal.Image.from_id.aio(resources.image_id)
        secrets = [
            modal.Secret.from_name(name, environment_name=resources.environment_name)
            for name in resources.secret_names
        ]
        if resources.authentication_secret_name:
            secrets.append(
                modal.Secret.from_name(
                    resources.authentication_secret_name,
                    environment_name=resources.environment_name,
                    required_keys=[configuration.authentication_env],
                )
            )
        port = configuration.port
        serves = configuration.run.mode == "run"
        name = "claas-" + hashlib.sha256(volume.object_id.encode()).hexdigest()[:40]
        sandbox = await modal.Sandbox.create.aio(
            "python3",
            "-m",
            "exp.optimize.claas.service.launcher",
            "--config-stdin",
            app=app,
            name=name,
            image=image,
            environment_name=resources.environment_name,
            volumes={"/state": volume},
            secrets=secrets,
            gpu=resources.gpu,
            cpu=resources.cpu,
            memory=resources.memory_mib,
            timeout=resources.timeout_seconds,
            tags={"claas-run": run_id},
            encrypted_ports=[port] if serves else [],
            readiness_probe=modal.Probe.with_tcp(port) if serves else None,
        )
        handle = ModalRunHandle(
            sandbox,
            volume,
            cleanup_seconds=configuration.run.cleanup_timeout_seconds + 30,
        )
        started = False
        try:
            async with asyncio.timeout(resources.startup_timeout_seconds):
                await commit_owner(sandbox, resources.app_name)
                if import_path is not None:
                    await upload_import(
                        sandbox,
                        import_path,
                        run_id,
                        maximum_bytes=min(
                            resources.maximum_upload_bytes, configuration.maximum_import_bytes
                        ),
                    )
                started = True
                sandbox.stdin.write(payload)
                sandbox.stdin.write_eof()
                await sandbox.stdin.drain.aio()
                if serves:
                    await sandbox.wait_until_ready.aio(timeout=resources.startup_timeout_seconds)
                    tunnels = await sandbox.tunnels.aio()
                    handle.endpoint = tunnels[port].url
            return handle
        except BaseException:
            if started:
                try:
                    await handle.stop()
                except Exception:  # noqa: BLE001 - retain the initiating startup failure
                    pass
            else:
                await _terminate(sandbox)
            raise


async def _terminate(sandbox: modal.Sandbox) -> None:
    """Require positive termination within a finite RPC bound before releasing the handle."""
    async with asyncio.timeout(30):
        await sandbox.terminate.aio(wait=True)
        await sandbox.detach.aio()
