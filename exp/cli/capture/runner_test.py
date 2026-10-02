"""The foreground runner retains public CLI arguments and reports recoverable errors."""

import asyncio
import io
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from rich.console import Console
from rich.live import Live

from exp.cli.capture import display as display_module
from exp.cli.capture import runner
from exp.cli.capture.auth import CaptureCredentials
from exp.runtime.capture.certificates import capture_certificate_directory
from exp.runtime.capture.control import CaptureRun, CaptureRunClient, Organization
from exp.runtime.capture.proxy import CaptureBypassReason
from exp.runtime.capture.upload import CaptureUploader, UploadStats


@pytest.mark.parametrize("verbose", [False, True])
def test_runner_receives_public_arguments(monkeypatch: pytest.MonkeyPatch, verbose: bool) -> None:
    """The process entrypoint retains the exact domain set and project root."""
    invocations: list[tuple[tuple[str, ...], Path, bool]] = []

    def capture(console: Console, *, domains: tuple[str, ...], root: Path, verbose: bool) -> None:
        """Record arguments without invoking credentials, trust, or networking."""
        invocations.append((domains, root, verbose))

    monkeypatch.setattr(runner, "_capture", capture)
    arguments = ["--root", "/tmp/project", "--domain", "api.openai.com"]
    if verbose:
        arguments.append("--verbose")
    assert runner.main(arguments) == 0
    assert invocations == [(("api.openai.com",), Path("/tmp/project"), verbose)]


def test_runner_reports_startup_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed foreground capture returns its actionable startup diagnostic."""

    def capture(console: Console, *, domains: tuple[str, ...], root: Path, verbose: bool) -> None:
        """Simulate a content-free setup error before interception begins."""
        raise RuntimeError("Synthetic startup failure")

    monkeypatch.setattr(runner, "_capture", capture)
    assert runner.main(["--root", "/tmp/project", "--domain", "api.openai.com"]) == 1
    output = capsys.readouterr().out
    assert "Synthetic startup failure" in output
    assert "exp capture reset" not in output


def test_preflight_runs_before_login_or_cloud_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing backend prerequisites cannot open login or create a cloud run."""

    def unavailable(*, installation_is_current: Callable[[], bool]) -> None:
        """Reject backend setup before authentication is allowed."""
        raise RuntimeError("Synthetic missing redirector")

    monkeypatch.setattr(runner, "require_local_backend", unavailable)
    with pytest.raises(RuntimeError, match="Synthetic missing redirector"):
        runner._capture(Console(), domains=("api.openai.com",), root=Path("/tmp/project"))


@pytest.mark.parametrize("fail_session", [False, True])
def test_session_ownership_covers_login_and_shutdown(
    monkeypatch: pytest.MonkeyPatch, fail_session: bool
) -> None:
    """The per-user lock prevents competing login or interception until teardown ends."""
    events: list[str] = []
    credentials = CaptureCredentials("https://api.example.com", "https://example.com", "test")

    @contextmanager
    def ownership() -> Iterator[None]:
        """Record the ownership boundary even when shutdown fails."""
        events.append("acquire")
        try:
            yield
        finally:
            events.append("release")

    def authenticate(
        *, console: Console, environment: Mapping[str, str], root: Path
    ) -> CaptureCredentials:
        """Return synthetic login material after the lock has been acquired."""
        assert events == ["preflight", "acquire"]
        events.append("login")
        return credentials

    async def session(
        console: Console,
        *,
        domains: tuple[str, ...],
        credentials: CaptureCredentials,
        verbose: bool,
    ) -> None:
        """Represent the complete asynchronous capture and native shutdown lifetime."""
        assert events == ["preflight", "acquire", "login"]
        events.append("shutdown")
        if fail_session:
            raise RuntimeError("Synthetic shutdown failure")

    monkeypatch.setattr(
        runner, "require_local_backend", lambda installation_is_current: events.append("preflight")
    )
    monkeypatch.setattr(runner, "capture_instance", ownership)
    monkeypatch.setattr(runner, "capture_credentials", authenticate)
    monkeypatch.setattr(runner, "_capture_authenticated", session)
    if fail_session:
        with pytest.raises(RuntimeError, match="Synthetic shutdown failure"):
            runner._capture(Console(), domains=("api.openai.com",), root=Path("/tmp/project"))
    else:
        runner._capture(Console(), domains=("api.openai.com",), root=Path("/tmp/project"))
    assert events == ["preflight", "acquire", "login", "shutdown", "release"]


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("terminal", [False, True])
def test_session_failure_ends_cloud_run_without_reusing_legacy_ca(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verbose: bool, terminal: bool
) -> None:
    """Trust pauses terminal rendering, then startup resumes with the scoped CA."""
    monkeypatch.setenv("TERM", "xterm-256color")
    domains = ("chatgpt.com", "api.openai.com")
    data = tmp_path / "capture"
    legacy = data / "ca" / "mitmproxy-ca-cert.pem"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy CA must remain untouched")
    ca_directory = capture_certificate_directory(data, domains)
    organization = Organization(org_id=uuid4(), org_slug="test", org_name="Test")
    run = CaptureRun(
        id=uuid4(),
        org_id=organization.org_id,
        upload_origin="https://storage.example",
        upload_path_prefix="/storage/v1/object/upload/sign/capture/",
    )
    control = Mock(spec=CaptureRunClient)
    control.whoami.return_value = organization
    control.start.return_value = run
    uploader = Mock(spec=CaptureUploader)
    uploader.stats = UploadStats(0, 0, 0, 0, 0)
    uploader.pending_current_run = 0
    output = io.StringIO()
    console = Console(file=output, width=200, force_terminal=terminal, color_system=None)
    renderers: list[Live] = []
    trust_output_end = 0

    def live(*, console: Console, refresh_per_second: int, transient: bool) -> Live:
        """Track the real renderer that could otherwise overwrite native password prompts."""
        renderer = Live(console=console, refresh_per_second=refresh_per_second, transient=transient)
        renderers.append(renderer)
        return renderer

    def trust_prompt(certificate: Path, *, domains: tuple[str, ...]) -> None:
        """Assert the inherited terminal is idle and its cursor visible during approval."""
        nonlocal trust_output_end
        assert renderers and not renderers[0].is_started
        rendered = output.getvalue()
        assert "Waiting for certificate approval" in rendered
        if terminal:
            assert rendered.rfind("\x1b[?25h") > rendered.rfind("\x1b[?25l") >= 0
        trust_output_end = len(rendered)

    trust = Mock(side_effect=trust_prompt)
    session = AsyncMock(side_effect=RuntimeError("Synthetic session failure"))
    monkeypatch.setattr(runner, "provider_data_dir", lambda: tmp_path)
    monkeypatch.setattr(runner, "CaptureRunClient", lambda **kwargs: control)
    monkeypatch.setattr(runner, "CaptureUploader", lambda **kwargs: uploader)
    monkeypatch.setattr(runner, "certificate_is_trusted", lambda *args, **kwargs: False)
    monkeypatch.setattr(runner, "trust_certificate", trust)
    monkeypatch.setattr(runner, "run_session", session)
    monkeypatch.setattr(display_module, "Live", live)
    with pytest.raises(RuntimeError, match="Synthetic session failure"):
        asyncio.run(
            runner._capture_authenticated(
                console,
                domains=domains,
                verbose=verbose,
                credentials=CaptureCredentials(
                    "https://api.example.com", "https://example.com", "synthetic"
                ),
            )
        )
    assert (session.call_args.kwargs["on_diagnostic"] is not None) == verbose
    certificate = ca_directory / "mitmproxy-ca-cert.pem"
    assert certificate.is_file()
    trust.assert_called_once_with(certificate, domains=domains)
    assert session.call_args.kwargs["ca_directory"] == ca_directory
    assert legacy.read_bytes() == b"legacy CA must remain untouched"
    for detail in (
        "Organization:",
        "Provider domains:",
        "Captures model prompts",
        "Approve Mitmproxy Redirector if macOS requests",
        "DNS settings stay unchanged",
        "First-time setup:",
        "Clients with a custom trust store",
        "Public CA:",
    ):
        assert (detail in output.getvalue()) == verbose
    assert "Capturing to" not in output.getvalue()
    assert "Capture stopped." not in output.getvalue()
    assert "Starting network extension" in output.getvalue()
    resumed = output.getvalue()[trust_output_end:]
    assert "Starting network extension" in resumed
    if terminal:
        assert "\x1b[?25l" in resumed
        assert resumed.rfind("\x1b[?25h") > resumed.rfind("\x1b[?25l")
    assert not renderers[0].is_started
    control.end.assert_awaited_once_with(run, pending_batches=0, upload_errors=0)
    control.close.assert_awaited_once()


@pytest.mark.parametrize("fail_session", [False, True])
def test_runner_reports_pending_startup_then_real_activity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_session: bool
) -> None:
    """The complete runner presents readiness and delivery without exposing setup noise."""
    organization = Organization(org_id=uuid4(), org_slug="test", org_name="Test")
    run = CaptureRun(
        id=uuid4(),
        org_id=organization.org_id,
        upload_origin="https://storage.example",
        upload_path_prefix="/storage/v1/object/upload/sign/capture/",
    )
    control = Mock(spec=CaptureRunClient)
    control.whoami.return_value = organization
    control.start.return_value = run
    uploader = Mock(spec=CaptureUploader)
    uploader.stats = UploadStats(0, 0, 0, 0, 0)
    uploader.pending_current_run = 0
    output = io.StringIO()
    settings_cancelled = asyncio.Event()

    async def pending_settings() -> bool:
        """Represent a slow optional status check that must not outlive readiness."""
        try:
            await asyncio.Event().wait()
            return False
        finally:
            settings_cancelled.set()

    settings = AsyncMock(side_effect=pending_settings)
    monkeypatch.setattr(runner, "open_pending_approval_settings", settings)

    async def session(
        *,
        domains: tuple[str, ...],
        ca_directory: Path,
        uploader: CaptureUploader,
        control: CaptureRunClient,
        run: CaptureRun,
        on_started: Callable[[], None],
        on_progress: Callable[[UploadStats], None],
        on_warning: Callable[[str], None],
        on_waiting: Callable[[], None],
        on_bypass: Callable[[str, str, CaptureBypassReason], None],
        on_diagnostic: Callable[[str], None] | None,
    ) -> UploadStats:
        """Drive the actual console callbacks without activating native interception."""
        assert "Starting network extension" in output.getvalue()
        assert "Capturing to" not in output.getvalue()
        on_waiting()
        await asyncio.sleep(0)
        settings.assert_awaited_once()
        assert "Network Extensions" in output.getvalue()
        assert "Capturing to" not in output.getvalue()
        on_started()
        await asyncio.wait_for(settings_cancelled.wait(), timeout=1.0)
        on_progress(UploadStats(0, 0, 0, 0, 0))
        on_progress(
            UploadStats(
                pending_batches=0,
                upload_errors=0,
                dropped_exchanges=0,
                captured_exchanges=1,
                uploaded_batches=1,
            )
        )
        on_warning("Synthetic upload warning")
        on_bypass("Codex", "api.openai.com", "certificate")
        if fail_session:
            raise RuntimeError("Synthetic running proxy failure")
        return UploadStats(
            pending_batches=1,
            upload_errors=0,
            dropped_exchanges=0,
            captured_exchanges=2,
            uploaded_batches=1,
        )

    monkeypatch.setattr(runner, "provider_data_dir", lambda: tmp_path)
    monkeypatch.setattr(runner, "CaptureRunClient", lambda **kwargs: control)
    monkeypatch.setattr(runner, "CaptureUploader", lambda **kwargs: uploader)
    monkeypatch.setattr(runner, "prepare_certificate", lambda *args: tmp_path / "ca.pem")
    monkeypatch.setattr(runner, "certificate_is_trusted", lambda *args, **kwargs: True)
    monkeypatch.setattr(runner, "run_session", session)
    coroutine = runner._capture_authenticated(
        Console(file=output, width=200),
        domains=("api.openai.com",),
        credentials=CaptureCredentials(
            "https://api.example.com", "https://example.com", "synthetic"
        ),
    )
    if fail_session:
        with pytest.raises(RuntimeError, match="Synthetic running proxy failure"):
            asyncio.run(coroutine)
    else:
        asyncio.run(coroutine)
    rendered = output.getvalue()
    assert "Capturing to Test · Ctrl+C to stop" in rendered
    assert "Telemetry: https://example.com/api-keys?section=capture" in rendered
    assert "0 requests captured · waiting for first completed request" in rendered
    assert "1 request captured" in rendered
    assert "last just now" in rendered
    assert "Synthetic upload warning" in rendered
    assert "Not capturing Codex on api.openai.com: certificate rejected." in rendered
    assert "partial capture: 1 target not captured" in rendered
    assert "Public CA" not in rendered
    assert "batches uploaded" not in rendered
    assert ("Capture stopped. 2 requests captured" in rendered) == (not fail_session)
    assert ("batches pending; the next exp capture retries them." in rendered) == (not fail_session)
    control.end.assert_awaited_once_with(run, pending_batches=0, upload_errors=0)
    control.close.assert_awaited_once()
