"""Provider-free checks of one-container ownership, launch, and finite cleanup."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import modal
import pytest

from exp.optimize.claas.backends.modal import hosting
from exp.optimize.claas.backends.modal.configuration import ModalLaunch
from exp.optimize.claas.backends.modal.configuration_test import launch
from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings
from exp.optimize.claas.service.configuration import RunConfiguration
from exp.optimize.claas.service.contracts import RunMode
from exp.optimize.claas.service.launch_configuration import RunLaunchConfiguration
from exp.optimize.claas.training_contracts_test import spec


def configuration(mode: RunMode = "burst") -> RunLaunchConfiguration:
    """Name bounded existing resources without starting a resident model runtime."""
    return RunLaunchConfiguration(
        directory=Path("/state/learner"),
        spec=spec(),
        run=RunConfiguration(mode=mode, maximum_run_seconds=60, minimum_ready_examples=1),
        runtime=ResidentVerlSettings(
            checkpoint_root=Path("/state/learner/checkpoints"),
            maximum_output_tokens=4,
        ),
        persistence="modal-volume",
        host="0.0.0.0",
        compute_reservation_usd=1,
    )


def fake_sandbox() -> SimpleNamespace:
    """Model only the public Modal lifecycle methods used by the hosting adapter."""
    command = SimpleNamespace(wait=SimpleNamespace(aio=AsyncMock()), returncode=0)
    return SimpleNamespace(
        object_id="sb-fixture",
        returncode=0,
        stdin=SimpleNamespace(
            write=Mock(), write_eof=Mock(), drain=SimpleNamespace(aio=AsyncMock())
        ),
        filesystem=SimpleNamespace(
            write_text=SimpleNamespace(aio=AsyncMock()),
            read_text=SimpleNamespace(aio=AsyncMock(return_value="42\n")),
            make_directory=SimpleNamespace(aio=AsyncMock()),
            list_files=SimpleNamespace(aio=AsyncMock(return_value=[])),
            write_bytes=SimpleNamespace(aio=AsyncMock()),
        ),
        exec=SimpleNamespace(aio=AsyncMock(return_value=command)),
        wait=SimpleNamespace(aio=AsyncMock()),
        poll=SimpleNamespace(aio=AsyncMock(return_value=None)),
        terminate=SimpleNamespace(aio=AsyncMock()),
        detach=SimpleNamespace(aio=AsyncMock()),
        wait_until_ready=SimpleNamespace(aio=AsyncMock()),
        tunnels=SimpleNamespace(
            aio=AsyncMock(
                return_value={
                    8000: SimpleNamespace(url="https://fixture.modal.run"),
                }
            )
        ),
    )


def install_sdk(monkeypatch: pytest.MonkeyPatch, sandbox: SimpleNamespace) -> AsyncMock:
    """Replace provider operations so the tests allocate no App, Volume, or GPU."""
    volume = SimpleNamespace(object_id="vo-fixture", hydrate=SimpleNamespace(aio=AsyncMock()))
    monkeypatch.setattr(modal.App, "lookup", SimpleNamespace(aio=AsyncMock(return_value=object())))
    monkeypatch.setattr(
        modal.Image, "from_id", SimpleNamespace(aio=AsyncMock(return_value=object()))
    )
    monkeypatch.setattr(modal.Volume, "from_name", Mock(return_value=volume))
    monkeypatch.setattr(modal.Secret, "from_name", Mock(return_value=object()))
    create = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(modal.Sandbox, "create", SimpleNamespace(aio=create))
    monkeypatch.setattr(hosting, "bind_owner", AsyncMock())
    return create


def test_burst_starts_one_resident_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """One finite container starts the shared launcher without exposing a serving port."""

    async def scenario() -> None:
        """Exercise the asynchronous hosting contract in one owned event loop."""
        sandbox = fake_sandbox()
        create = install_sdk(monkeypatch, sandbox)
        handle = await hosting.ModalLauncher(launch()).start(configuration(), run_id="run-1")
        assert handle.sandbox_id == "sb-fixture"
        assert handle.endpoint is None
        assert create.await_count == 1
        assert create.call_args.args == (
            "python3",
            "-m",
            "exp.optimize.claas.service.launcher",
            "--config-stdin",
        )
        assert create.call_args.kwargs["encrypted_ports"] == []
        assert create.call_args.kwargs["timeout"] == launch().timeout_seconds
        assert create.call_args.kwargs["volumes"].keys() == {"/state"}
        sandbox.exec.aio.assert_awaited_once_with("/usr/bin/sync", "/state")
        received = RunLaunchConfiguration.model_validate_json(sandbox.stdin.write.call_args.args[0])
        assert received == configuration()
        sandbox.stdin.write_eof.assert_called_once()
        assert await handle.wait() == 0

    asyncio.run(scenario())


def test_live_run_has_one_stable_tls_tunnel(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP requests address the existing Sandbox instead of creating more trainers."""

    async def scenario() -> None:
        """Exercise the asynchronous hosting contract in one owned event loop."""
        sandbox = fake_sandbox()
        create = install_sdk(monkeypatch, sandbox)
        resources = ModalLaunch.model_validate(
            launch().model_dump()
            | {
                "authentication_secret_name": "learner-auth",
            }
        )
        handle = await hosting.ModalLauncher(resources).start(configuration("run"), run_id="run-1")
        assert handle.endpoint == "https://fixture.modal.run"
        assert create.call_args.kwargs["encrypted_ports"] == [8000]
        sandbox.wait_until_ready.aio.assert_awaited_once()
        await handle.stop()
        assert sandbox.exec.aio.call_args.args == ("kill", "-TERM", "42")
        sandbox.terminate.aio.assert_not_called()

    asyncio.run(scenario())


def test_same_volume_uses_same_sandbox_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing run_id cannot bypass Modal's live named-Sandbox uniqueness constraint."""

    async def scenario() -> None:
        """Exercise the asynchronous hosting contract in one owned event loop."""
        create = install_sdk(monkeypatch, fake_sandbox())
        launcher = hosting.ModalLauncher(launch())
        await launcher.start(configuration(), run_id="first")
        first = create.call_args.kwargs["name"]
        await launcher.start(configuration(), run_id="second")
        assert create.call_args.kwargs["name"] == first

    asyncio.run(scenario())


def test_import_is_uploaded_before_service_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the already-owned Sandbox receives input; stdin EOF starts the shared parser."""

    async def scenario() -> None:
        """Exercise the asynchronous hosting contract in one owned event loop."""
        sandbox = fake_sandbox()
        install_sdk(monkeypatch, sandbox)
        path = tmp_path / "examples.jsonl"
        path.write_bytes(b'{"exact":"evidence"}\n')
        configured = RunLaunchConfiguration.model_validate(
            configuration().model_dump()
            | {
                "import_examples_path": "/state/imports/run-1.jsonl",
            }
        )
        await hosting.ModalLauncher(launch()).start(configured, run_id="run-1", import_path=path)
        sandbox.filesystem.write_bytes.aio.assert_awaited_once_with(
            path.read_bytes(),
            "/state/imports/run-1.jsonl",
        )
        sandbox.stdin.write_eof.assert_called_once()

    asyncio.run(scenario())


def test_invalid_configuration_never_constructs_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unmounted durable directory fails before provider clients or paid allocation."""

    async def scenario() -> None:
        """Exercise the asynchronous hosting contract in one owned event loop."""
        provider = Mock(side_effect=AssertionError("provider must not be constructed"))
        monkeypatch.setattr(modal.Volume, "from_name", provider)
        configured = RunLaunchConfiguration.model_validate(
            configuration().model_dump()
            | {
                "directory": "/tmp/learner",
                "runtime": ResidentVerlSettings(
                    checkpoint_root=Path("/tmp/learner/checkpoints"), maximum_output_tokens=4
                ),
            }
        )
        with pytest.raises(ValueError, match="/state"):
            await hosting.ModalLauncher(launch()).start(configured, run_id="run-1")
        provider.assert_not_called()

    asyncio.run(scenario())


def test_failed_staging_terminates_owned_container(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pre-launch persistence failure cannot leak an idle paid Sandbox."""

    async def scenario() -> None:
        """Exercise the asynchronous hosting contract in one owned event loop."""
        sandbox = fake_sandbox()
        install_sdk(monkeypatch, sandbox)
        monkeypatch.setattr(hosting, "commit_owner", AsyncMock(side_effect=OSError("sync")))
        with pytest.raises(OSError, match="sync"):
            await hosting.ModalLauncher(launch()).start(configuration(), run_id="run-1")
        sandbox.terminate.aio.assert_awaited_once_with(wait=True)
        sandbox.stdin.write.assert_not_called()

    asyncio.run(scenario())


def test_forced_stop_is_not_successful_checkpoint() -> None:
    """A hung graceful close ends at the bound and reports a failure after killing compute."""

    async def scenario() -> None:
        """Exercise the asynchronous hosting contract in one owned event loop."""

        async def hang() -> None:
            """Keep service shutdown pending until the adapter's deadline cancels the wait."""
            await asyncio.Future()

        sandbox = fake_sandbox()
        sandbox.wait.aio = AsyncMock(side_effect=hang)
        handle = hosting.ModalRunHandle(
            cast(modal.Sandbox, sandbox),
            cast(modal.Volume, object()),
            cleanup_seconds=0.01,
        )
        with pytest.raises(TimeoutError):
            await handle.stop()
        sandbox.terminate.aio.assert_awaited_once_with(wait=True)
        sandbox.detach.aio.assert_awaited_once()

    asyncio.run(scenario())


def test_nonzero_exit_is_never_reported_as_complete() -> None:
    """An upstream failure remains visible even when the container cleanly terminates."""

    async def scenario() -> None:
        """Exercise the asynchronous hosting contract in one owned event loop."""
        sandbox = fake_sandbox()
        sandbox.returncode = 1
        handle = hosting.ModalRunHandle(
            cast(modal.Sandbox, sandbox),
            cast(modal.Volume, object()),
            cleanup_seconds=1,
        )
        with pytest.raises(RuntimeError, match="exited with code 1"):
            await handle.wait()

    asyncio.run(scenario())


def test_cancelled_readiness_stops_the_exact_started_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling startup after stdin delivery drains the same container before returning."""

    async def scenario() -> None:
        """Cancel after service launch and verify graceful shutdown plus exception identity."""
        sandbox = fake_sandbox()
        install_sdk(monkeypatch, sandbox)
        cancelled = asyncio.CancelledError("caller cancelled")
        sandbox.wait_until_ready.aio.side_effect = cancelled
        resources = ModalLaunch.model_validate(
            launch().model_dump()
            | {
                "authentication_secret_name": "learner-auth",
            }
        )
        with pytest.raises(asyncio.CancelledError) as caught:
            await hosting.ModalLauncher(resources).start(configuration("run"), run_id="cancelled")
        assert caught.value is cancelled
        assert sandbox.exec.aio.call_args.args == ("kill", "-TERM", "42")
        sandbox.wait.aio.assert_awaited_once()
        sandbox.terminate.aio.assert_not_called()

    asyncio.run(scenario())


def test_stop_handles_natural_exit_after_poll() -> None:
    """A naturally completed run need not retain its PID file to prove successful exit."""

    async def scenario() -> None:
        """Make the service exit between the liveness check and the shutdown signal."""
        sandbox = fake_sandbox()
        sandbox.poll.aio.side_effect = [None, 0]
        sandbox.filesystem.read_text.aio.side_effect = FileNotFoundError("PID removed")
        handle = hosting.ModalRunHandle(
            cast(modal.Sandbox, sandbox),
            cast(modal.Volume, object()),
            cleanup_seconds=1,
        )
        await handle.stop()
        sandbox.terminate.aio.assert_not_called()
        assert await handle.wait() == 0
        sandbox.wait.aio.assert_awaited_once()

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["initial", "after_pid_error"])
def test_stop_bounds_every_poll_before_forced_termination(phase: str) -> None:
    """A stalled liveness RPC cannot bypass the cleanup deadline or leak the owned Sandbox."""

    async def scenario() -> None:
        """Exercise both formerly unbounded poll sites with a cancellation-aware SDK fixture."""
        sandbox = fake_sandbox()
        calls = 0

        async def poll() -> int | None:
            """Stall at the selected liveness boundary until its enclosing deadline expires."""
            nonlocal calls
            calls += 1
            if phase == "after_pid_error" and calls == 1:
                return None
            await asyncio.Future()
            return None

        sandbox.poll.aio = AsyncMock(side_effect=poll)
        sandbox.filesystem.read_text.aio.side_effect = FileNotFoundError("PID removed")
        handle = hosting.ModalRunHandle(
            cast(modal.Sandbox, sandbox), cast(modal.Volume, object()), cleanup_seconds=0.01
        )
        with pytest.raises(TimeoutError):
            await handle.stop()
        sandbox.terminate.aio.assert_awaited_once_with(wait=True)
        sandbox.detach.aio.assert_awaited_once()

    asyncio.run(scenario())


def test_declared_import_requires_staging_before_provider_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote path cannot consume old input or allocate a Sandbox without matching upload."""

    async def scenario() -> None:
        """Reject an unstaged import at the SDK-free preflight boundary."""
        lookup = AsyncMock(side_effect=AssertionError("provider must not be contacted"))
        monkeypatch.setattr(modal.App, "lookup", SimpleNamespace(aio=lookup))
        configured = configuration().model_copy(
            update={"import_examples_path": Path("/state/imports/run-1.jsonl")}
        )
        with pytest.raises(ValueError, match="local import_path"):
            await hosting.ModalLauncher(launch()).start(configured, run_id="run-1")
        lookup.assert_not_awaited()

    asyncio.run(scenario())
