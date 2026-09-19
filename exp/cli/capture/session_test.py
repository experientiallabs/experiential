"""Foreground orchestration restores routing when the proxy fails."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

from exp.cli.capture import session
from exp.runtime.capture.control import CaptureRun, CaptureRunClient
from exp.runtime.capture.proxy import CaptureProxy
from exp.runtime.capture.resolver import UpstreamResolver
from exp.runtime.capture.system import CaptureSystemSession
from exp.runtime.capture.upload import CaptureUploader, UploadStats


@pytest.mark.parametrize("fail_before_ready", [False, True])
def test_proxy_failure_restores_networking_and_closes_uploads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_before_ready: bool,
) -> None:
    """A failed listener never installs hosts; a failed active proxy always removes them."""
    events: list[str] = []
    proxy = Mock(spec=CaptureProxy)
    proxy.dropped_exchanges = 0
    resolver = Mock(spec=UpstreamResolver)
    helper = Mock(spec=CaptureSystemSession)
    uploader = Mock(spec=CaptureUploader)
    uploader.stats = UploadStats(0, 0, 0, 0, 0)
    uploader.close.side_effect = lambda **kwargs: events.append("upload-close")
    proxy.shutdown.side_effect = lambda: events.append("proxy-stop")
    helper.close.side_effect = lambda: events.append("hosts-restored")

    async def prime(domains: tuple[str, ...]) -> None:
        """Record resolution without querying a real provider."""
        events.append("dns-ready")

    async def serve(*, port: int, ca_directory: Path, ready: Callable[[], None]) -> None:
        """Fail either before binding or after reporting an active listener."""
        if fail_before_ready:
            raise RuntimeError("synthetic bind failure")
        ready()
        await asyncio.sleep(0.01)
        raise RuntimeError("synthetic active proxy failure")

    def start(*, upstream_port: int, domains: tuple[str, ...]) -> Mock:
        """Record simulated routing activation without modifying the system."""
        events.append("hosts-installed")
        return helper

    resolver.prime.side_effect = prime
    proxy.serve.side_effect = serve
    monkeypatch.setattr(session, "UpstreamResolver", lambda: resolver)
    monkeypatch.setattr(session, "CaptureProxy", lambda **kwargs: proxy)
    monkeypatch.setattr(session.CaptureSystemSession, "start", start)
    with pytest.raises(RuntimeError, match="synthetic"):
        asyncio.run(
            session.run_session(
                domains=("api.openai.com",),
                ca_directory=tmp_path,
                uploader=uploader,
                control=Mock(spec=CaptureRunClient),
                run=CaptureRun(
                    id=uuid4(),
                    org_id=uuid4(),
                    upload_origin="https://storage.example",
                    upload_path_prefix="/storage/v1/object/upload/sign/capture/",
                ),
                on_started=lambda: events.append("active"),
                on_progress=lambda stats: None,
                on_warning=lambda message: None,
            )
        )
    assert events[0] == "dns-ready"
    if fail_before_ready:
        assert "hosts-installed" not in events
        assert "upload-close" not in events
    else:
        assert events[-1] == "upload-close"
        assert events.index("hosts-restored") < events.index("proxy-stop")
