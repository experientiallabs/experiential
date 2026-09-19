"""Capture control derives the organization and never leaks remote error bodies."""

import asyncio
from uuid import uuid4

import httpx
import pytest

from exp.runtime.capture import control
from exp.runtime.capture.control import CaptureCloudError, CaptureRunClient


def test_mismatched_run_is_rejected() -> None:
    """A cloud run from another organization is never accepted."""
    organization = uuid4()
    run_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a controlled cloud response while recording the request contract."""
        assert request.headers["authorization"] == "Bearer xpl_secret"
        return httpx.Response(
            200,
            json={
                "id": str(run_id),
                "org_id": str(uuid4()),
                "upload_origin": "https://storage.example",
                "upload_path_prefix": "/storage/v1/object/upload/sign/capture/",
            },
        )

    client = CaptureRunClient(
        base_url="https://platform.test",
        api_key="xpl_secret",
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        """Run the isolated asynchronous request and always close its client."""
        try:
            with pytest.raises(CaptureCloudError, match="different organization"):
                await client.start(organization, run_id, label="laptop", version="0.1")
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [302, 401, 404, 500])
def test_error_body_is_never_echoed(status: int) -> None:
    """Cloud errors do not expose response bodies or follow redirects."""
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a controlled cloud response while recording the request contract."""
        requests.append(str(request.url))
        return httpx.Response(
            status,
            text="private-upstream-body",
            headers={
                "Location": "https://other-origin.test/",
            },
        )

    client = CaptureRunClient(
        base_url="https://platform.test",
        api_key="xpl_secret",
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        """Run the isolated asynchronous request and always close its client."""
        try:
            with pytest.raises(CaptureCloudError) as caught:
                await client.whoami()
            assert "private-upstream-body" not in str(caught.value)
            assert "xpl_secret" not in str(caught.value)
            assert requests == ["https://platform.test/api/whoami"]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_cloud_wait_has_a_total_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stalled cloud response cannot strand foreground shutdown in a worker thread."""
    monkeypatch.setattr(control, "_REQUEST_TIMEOUT", 0.02)
    cancelled: list[bool] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        """Simulate a peer whose response never completes."""
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        raise AssertionError("deadline failed")

    async def scenario() -> None:
        """Run the isolated asynchronous request and always close its client."""
        client = CaptureRunClient(
            base_url="https://platform.test",
            api_key="xpl_secret",
            transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(CaptureCloudError, match="Cannot reach"):
                await client.whoami()
            assert cancelled == [True]
        finally:
            await client.close()

    asyncio.run(scenario())
