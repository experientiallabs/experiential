"""Typed HTTP client for the platform's CLI registry surface.

Every call carries the org API key as a bearer credential; the platform scopes
reads and writes to that key's organization at member strength. Error payloads
are the platform's uniform ``{"error": message}`` shape, surfaced as
:class:`PlatformError` with the HTTP status attached.
"""

from __future__ import annotations

import hashlib
import json
from importlib import metadata

import httpx
from pydantic import BaseModel

from wmh.core.types import JsonValue

_TIMEOUT_SECONDS = 120.0


class PlatformError(RuntimeError):
    """A platform request failed; carries the HTTP status when one exists."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ActorInfo(BaseModel):
    """Who the platform resolved the credential to."""

    kind: str  # "api_key" | "user"
    id: str


class OrgInfo(BaseModel):
    """One organization visible to the credential."""

    id: str
    slug: str
    name: str


class ProjectInfo(BaseModel):
    """One project visible to the credential."""

    id: str
    org_id: str
    slug: str
    name: str


class WhoAmI(BaseModel):
    """Response of ``GET /api/whoami``."""

    actor: ActorInfo
    orgs: list[OrgInfo]
    projects: list[ProjectInfo]


class RemoteWorldModel(BaseModel):
    """The slice of a world-model row the CLI presents."""

    id: str
    name: str
    display_name: str | None = None
    status: str
    updated_at: str | None = None


class RemoteHarness(BaseModel):
    """The slice of a registry harness row the CLI presents."""

    id: str
    name: str
    latest_version: int
    updated_at: str | None = None


class RemoteHarnessVersion(BaseModel):
    """One doc-less entry of a harness's version lineage."""

    version: int
    doc_hash: str
    created_at: str | None = None


class HarnessVersionDoc(BaseModel):
    """One full harness version, doc included."""

    version: int
    doc: dict[str, JsonValue]
    doc_hash: str


class PushedHarnessVersion(BaseModel):
    """Response of a harness push: the version the doc landed as."""

    name: str
    version: int
    doc_hash: str
    created: bool  # False when the push was an idempotent repeat of the tip


def fetch_cli_config(web_url: str, *, transport: httpx.BaseTransport | None = None) -> str | None:
    """Ask the web app which backend host the CLI should call.

    ``GET {web_url}/api/cli/config`` is public: the backend URL is not a
    secret (every Endpoints page shows it) and everything behind it is
    bearer-gated.
    """
    with httpx.Client(timeout=30.0, transport=transport) as client:
        response = client.get(f"{web_url.rstrip('/')}/api/cli/config")
        if response.status_code != 200:
            msg = f"platform discovery failed with HTTP {response.status_code} at {web_url}"
            raise PlatformError(msg, status_code=response.status_code)
        api_url = response.json().get("apiUrl")
        return str(api_url).rstrip("/") if api_url else None


class PlatformClient:
    """Requests against the platform registry, authenticated with an org API key."""

    def __init__(
        self,
        api_url: str,
        token: str,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        try:
            version = metadata.version("world-model-harness")
        except metadata.PackageNotFoundError:
            version = "dev"
        self._client = httpx.Client(
            base_url=api_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": f"wmh/{version}",
            },
            timeout=_TIMEOUT_SECONDS,
            transport=transport,
        )

    def __enter__(self) -> PlatformClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # -- identity ------------------------------------------------------------------------------

    def whoami(self) -> WhoAmI:
        response = self._client.get("/api/whoami")
        self._raise_for_error(response)
        return WhoAmI.model_validate(response.json())

    # -- world models --------------------------------------------------------------------------

    def list_world_models(self, project_id: str) -> list[RemoteWorldModel]:
        response = self._client.get(f"/api/projects/{project_id}/world-models")
        self._raise_for_error(response)
        rows = response.json().get("world_models", [])
        return [RemoteWorldModel.model_validate(row) for row in rows]

    def push_model_bundle(
        self,
        project_id: str,
        name: str,
        content: bytes,
        meta: dict[str, JsonValue],
    ) -> RemoteWorldModel:
        response = self._client.post(
            f"/api/projects/{project_id}/world-models/{name}/bundle",
            files={"file": (f"{name}.tar.gz", content, "application/gzip")},
            data={"meta": json.dumps(meta)},
        )
        self._raise_for_error(response)
        return RemoteWorldModel.model_validate(response.json())

    def download_model_bundle(self, project_id: str, name: str) -> bytes:
        """Fetch a model's bundle bytes, verifying the declared digest."""
        response = self._client.get(f"/api/projects/{project_id}/world-models/{name}/bundle")
        self._raise_for_error(response)
        content = response.content
        declared = response.headers.get("X-Bundle-Sha256")
        actual = hashlib.sha256(content).hexdigest()
        if declared is not None and declared != actual:
            msg = f"bundle digest mismatch for {name}: expected {declared}, got {actual}"
            raise PlatformError(msg)
        return content

    # -- harnesses -----------------------------------------------------------------------------

    def list_harnesses(self, project_id: str) -> list[RemoteHarness]:
        response = self._client.get(f"/api/projects/{project_id}/harnesses")
        self._raise_for_error(response)
        rows = response.json().get("harnesses", [])
        return [RemoteHarness.model_validate(row) for row in rows]

    def get_harness(
        self, project_id: str, name: str
    ) -> tuple[RemoteHarness, list[RemoteHarnessVersion]]:
        response = self._client.get(f"/api/projects/{project_id}/harnesses/{name}")
        self._raise_for_error(response)
        payload = response.json()
        harness = RemoteHarness.model_validate(payload["harness"])
        versions = [RemoteHarnessVersion.model_validate(row) for row in payload["versions"]]
        return harness, versions

    def get_harness_version(self, project_id: str, name: str, version: int) -> HarnessVersionDoc:
        response = self._client.get(
            f"/api/projects/{project_id}/harnesses/{name}/versions/{version}"
        )
        self._raise_for_error(response)
        return HarnessVersionDoc.model_validate(response.json())

    def push_harness_version(
        self,
        project_id: str,
        name: str,
        doc: dict[str, JsonValue],
        doc_hash: str,
    ) -> PushedHarnessVersion:
        response = self._client.post(
            f"/api/projects/{project_id}/harnesses/{name}/versions",
            json={"doc": doc, "doc_hash": doc_hash},
        )
        self._raise_for_error(response)
        return PushedHarnessVersion.model_validate(response.json())

    # -- internals -----------------------------------------------------------------------------

    def _raise_for_error(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            message = response.json().get("error", response.text)
        except (json.JSONDecodeError, ValueError):
            message = response.text or f"HTTP {response.status_code}"
        if response.status_code == 401:
            message = f"{message} — run `wmh login` (or check WMH_PLATFORM_TOKEN)"
        raise PlatformError(str(message), status_code=response.status_code)
