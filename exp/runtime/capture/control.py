"""Normal organization authentication and capture-run lifecycle over HTTP."""

from __future__ import annotations

import asyncio
from uuid import UUID

import httpx
from pydantic import BaseModel, ValidationError

_REQUEST_TIMEOUT = 10.0


class CaptureCloudError(RuntimeError):
    """A capture control request failed without exposing response content or credentials."""


class Organization(BaseModel):
    """Organization identity derived by Platform from the existing API key."""

    org_id: UUID
    org_slug: str
    org_name: str


class CaptureRun(BaseModel):
    """Run identity acknowledged by Platform before networking is changed."""

    id: UUID
    org_id: UUID
    upload_origin: str
    upload_path_prefix: str


class CaptureRunClient:
    """Use one endpoint-bound login for run creation, heartbeat, and completion."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Create an authenticated client that never follows credential redirects."""
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(10.0),
            follow_redirects=False,
            transport=transport,
        )

    async def whoami(self) -> Organization:
        """Resolve the organization from the ordinary Experiential login."""
        try:
            response = await self._request("GET", "/api/whoami")
            return Organization.model_validate(response.json())
        except (ValidationError, ValueError) as exc:
            raise CaptureCloudError("Platform returned an invalid organization identity.") from exc

    async def start(self, org_id: UUID, run_id: UUID, *, label: str, version: str) -> CaptureRun:
        """Create one idempotent run and validate its server-derived organization."""
        response = await self._request(
            "POST",
            f"/api/orgs/{org_id}/capture/runs",
            payload={
                "run_id": str(run_id),
                "label": label[:120],
                "client_version": version[:80],
            },
        )
        try:
            run = CaptureRun.model_validate(response.json())
        except (ValidationError, ValueError) as exc:
            raise CaptureCloudError("Platform returned an invalid capture run.") from exc
        if run.id != run_id or run.org_id != org_id:
            raise CaptureCloudError("Platform returned a capture run for a different organization.")
        return run

    async def heartbeat(self, run: CaptureRun, *, pending_batches: int, upload_errors: int) -> None:
        """Publish absolute queue/error counters without waiting on trace projection."""
        await self._update(run, "heartbeat", pending_batches, upload_errors)

    async def end(self, run: CaptureRun, *, pending_batches: int, upload_errors: int) -> None:
        """Mark the run ended without discarding pending capture batches."""
        await self._update(run, "end", pending_batches, upload_errors)

    async def close(self) -> None:
        """Release the HTTP client without changing the saved login."""
        await self._client.aclose()

    async def _update(self, run: CaptureRun, action: str, pending: int, errors: int) -> None:
        """Send nonnegative absolute counters to one acknowledged run."""
        await self._request(
            "POST",
            f"/api/orgs/{run.org_id}/capture/runs/{run.id}/{action}",
            payload={
                "pending_batches": max(0, pending),
                "upload_errors": max(0, errors),
            },
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, str | int] | None = None,
    ) -> httpx.Response:
        """Reject failures with a content-free message safe for terminal presentation."""
        try:
            async with asyncio.timeout(_REQUEST_TIMEOUT):
                response = await self._client.request(method, path, json=payload)
        except (httpx.HTTPError, TimeoutError) as exc:
            raise CaptureCloudError(
                "Cannot reach Platform. Check your connection and retry."
            ) from exc
        if response.status_code in {401, 403}:
            raise CaptureCloudError("Platform rejected the login. Run exp login and retry.")
        if response.status_code == 404:
            raise CaptureCloudError(
                "This Platform endpoint does not support Capture yet. Use the capture preview."
            )
        if not response.is_success:
            raise CaptureCloudError(
                f"Platform capture request failed (HTTP {response.status_code})."
            )
        return response
