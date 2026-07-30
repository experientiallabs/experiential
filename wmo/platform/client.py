"""Typed HTTP client for the platform's CLI registry surface.

Every call carries the org API key as a bearer credential; the platform scopes
reads and writes to that key's organization at member strength. Error payloads
are the platform's uniform ``{"error": message}`` shape, surfaced as
:class:`PlatformError` with the HTTP status attached.

Every failure leaves this module as a :class:`PlatformError`: a host that
answers nothing becomes :class:`PlatformUnreachable`, and a host that answers
something other than JSON becomes a plain :class:`PlatformError`. Callers
therefore handle one exception type instead of httpx internals, which is what
keeps ``wmo login``/``status``/``run``/``push``/``pull`` from printing
tracebacks when the platform is down or the URL is not a platform at all.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Literal

import httpx
from llm_waterfall import ChatRequest, ChatResponse
from pydantic import BaseModel

from wmo.core.types import Action, JsonObject, JsonValue, Observation

_TIMEOUT_SECONDS = 120.0
_WORKSPACE_TIMEOUT_SECONDS = 300.0


class PlatformError(RuntimeError):
    """A platform request failed; carries the HTTP status when one exists."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class PlatformUnreachable(PlatformError):
    """No response at all: DNS failure, refused connection, timeout, bad scheme.

    Distinct from an HTTP error because the remedy is different — the host or
    the network is wrong, not the credential — and `status_code` is None.
    """


class _GuardedTransport(httpx.BaseTransport):
    """Maps httpx transport failures to :class:`PlatformUnreachable`.

    Every request in this module goes through one transport, so this is the
    single place a connection failure has to be translated; wrapping here also
    covers the injected test transports unchanged.
    """

    def __init__(self, inner: httpx.BaseTransport) -> None:
        self._inner = inner

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        try:
            return self._inner.handle_request(request)
        except httpx.RequestError as error:
            # Connect timeouts stringify to "", so name the class instead. The
            # remedy is the same wherever this surfaces, so it travels with the
            # message: every caller re-raises it as its own clean CLI error.
            detail = str(error) or type(error).__name__
            raise PlatformUnreachable(
                f"cannot reach {_shown(request.url)}: {detail}; check your connection, "
                "or re-run `wmo login --url <platform url>`"
            ) from error

    def close(self) -> None:
        self._inner.close()


def _guarded(transport: httpx.BaseTransport | None) -> _GuardedTransport:
    """Wrap an injected transport, or the default one, in the failure mapping."""
    return _GuardedTransport(transport if transport is not None else httpx.HTTPTransport())


def _shown(url: httpx.URL) -> str:
    """The part of a URL safe to print: bundle transfers sign with query parameters.

    Storage hands back capability URLs whose token lives in the query string, and
    these messages land in terminals and CI logs. The host and path are what tell
    the user which hop failed; the signature never is.
    """
    return str(url.copy_with(query=None, fragment=None))


def _decode_json(response: httpx.Response) -> JsonObject:
    """Read a JSON object body, reporting anything else as a platform error.

    A 200 carrying HTML is how a marketing site, a login-walled preview, or a
    captive portal answers: "this is not a platform", not a crash. Every
    successful response in this module is decoded here so that answer can never
    reach the user as a `JSONDecodeError`.
    """
    where = _shown(response.request.url)
    try:
        payload = response.json()
    except ValueError as error:
        content_type = response.headers.get("content-type", "no content type").split(";")[0]
        msg = f"{where} answered HTTP {response.status_code} with {content_type}, not JSON"
        raise PlatformError(msg, status_code=response.status_code) from error
    if not isinstance(payload, dict):
        msg = f"{where} answered with a JSON {type(payload).__name__}, not an object"
        raise PlatformError(msg, status_code=response.status_code)
    return payload


def _run_path(external_id: str) -> str:
    """A run name as a URL path segment, keeping its slashes.

    Run names are hierarchical (`jt/grid-c2/identity`) and the route takes them as a path parameter,
    so the slashes are structural and must survive. Everything else is data an operator chose and
    has to be escaped: a name with a `?` or a `#` would otherwise silently truncate the path into a
    query or a fragment and read a different run.
    """
    return urllib.parse.quote(external_id, safe="/")


def _sse_frames(response: httpx.Response) -> Iterator[JsonObject]:
    """Decode an SSE response into one JSON body per `data:` frame.

    Only `data:` lines carry payload. `id:` lines restate the frame's `pos`, which
    the body already holds, and comment lines (`:`) are the server's keep-alives.
    Both are skipped rather than surfaced, so a caller sees events and nothing else.
    """
    for line in response.iter_lines():
        if not line.startswith("data:"):
            continue
        body = line[len("data:") :].strip()
        if not body:
            continue
        try:
            decoded = json.loads(body)
        except ValueError:
            # Skipped rather than raised, deliberately: this generator feeds `wmo runs tail`, which
            # follows a run for hours, and one frame mangled by a proxy must not end the follow with
            # a decode traceback. A frame nobody can read is a frame nobody can miss either.
            continue
        if isinstance(decoded, dict):
            yield decoded


def _rows(payload: JsonObject, key: str) -> list[JsonValue]:
    """Read a list of rows out of a decoded payload, or report the wrong shape."""
    rows = payload.get(key, [])
    if not isinstance(rows, list):
        msg = f"platform answered with {key!r} as a JSON {type(rows).__name__}, not a list"
        raise PlatformError(msg)
    return rows


class ActorInfo(BaseModel):
    """Who the platform resolved the credential to."""

    kind: str  # "api_key" | "user"
    id: str


class OrgInfo(BaseModel):
    """One organization visible to the credential."""

    id: str
    slug: str
    name: str


class WhoAmI(BaseModel):
    """Response of ``GET /api/whoami``."""

    actor: ActorInfo
    orgs: list[OrgInfo]


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


class RunTarget(BaseModel):
    """A platform id resolved to one executable resource kind."""

    id: str
    kind: Literal["world_model", "agent"]
    org_id: str
    name: str
    display_name: str | None = None
    status: str


class RemoteWorldModelSession(BaseModel):
    """The slice of a hosted world-model session needed by ``wmo run``."""

    id: str
    world_model_id: str
    status: str


class RemoteAgentSession(BaseModel):
    """Hosted E2B agent session state needed by the CLI driver."""

    id: str
    agent_id: str
    status: str
    workspace_sync: bool
    launched_from: str
    starting_detail: str | None = None
    ended_reason: str | None = None
    error: str | None = None


class RemoteAgentSessionEvent(BaseModel):
    """One durable event from a hosted agent session transcript."""

    seq: int
    kind: Literal[
        "user_message",
        "assistant_message",
        "tool_call",
        "tool_output",
        "tool_result",
        "submit",
        "state",
        "status",
        "error",
        "workspace_patch",
    ]
    payload: dict[str, JsonValue]


class RemoteAgentEventPage(BaseModel):
    """One poll page of hosted transcript events and current session status."""

    events: list[RemoteAgentSessionEvent]
    last_seq: int
    status: str


class WorkspacePatchResult(BaseModel):
    """Paths accepted or rejected while applying a live workspace patch."""

    applied: list[str]
    conflicts: list[str]


class LocalPiRunInfo(BaseModel):
    """An org-scoped platform usage record for the built-in local pi harness."""

    id: str
    org_id: str
    status: str
    worker_provider: str
    worker_model: str


def fetch_cli_config(web_url: str, *, transport: httpx.BaseTransport | None = None) -> str | None:
    """Ask the web app which backend host the CLI should call.

    ``GET {web_url}/api/cli/config`` is public: the backend URL is not a
    secret (every Endpoints page shows it) and everything behind it is
    bearer-gated.
    """
    with httpx.Client(timeout=30.0, transport=_guarded(transport)) as client:
        response = client.get(f"{web_url.rstrip('/')}/api/cli/config")
        if response.status_code != 200:
            msg = f"platform discovery failed with HTTP {response.status_code} at {web_url}"
            raise PlatformError(msg, status_code=response.status_code)
        api_url = _decode_json(response).get("apiUrl")
        return str(api_url).rstrip("/") if api_url else None


class PlatformClient:
    """Requests against the platform registry, authenticated with an org API key."""

    def __init__(
        self,
        api_url: str,
        token: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float | None = None,
    ) -> None:
        """Open an authenticated client.

        Args:
            api_url: Backend host.
            token: Org API key.
            transport: Test transport override.
            timeout: Request timeout; defaults to the registry's patient bound.
                Callers on a run's critical path should pass a short one (see
                `wmo.runs.client.TELEMETRY_TIMEOUT_SECONDS`), because a
                black-holing platform would otherwise stall the run itself.
        """
        try:
            version = metadata.version("world-model-optimizer")
        except metadata.PackageNotFoundError:
            version = "dev"
        self._client = httpx.Client(
            base_url=api_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": f"wmo/{version}",
            },
            timeout=timeout if timeout is not None else _TIMEOUT_SECONDS,
            transport=_guarded(transport),
        )
        # Bundle bytes move directly against storage's signed URLs; that
        # client carries no platform credential.
        self._transfer = httpx.Client(
            headers={"User-Agent": f"wmo/{version}"},
            timeout=_TIMEOUT_SECONDS,
            transport=_guarded(transport),
        )

    def __enter__(self) -> PlatformClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()
        self._transfer.close()

    # -- identity ------------------------------------------------------------------------------

    def whoami(self) -> WhoAmI:
        response = self._client.get("/api/whoami")
        self._raise_for_error(response)
        # The identity call doubles as "is this host a platform?": `login
        # --api-url` and `status` reach it before anything else, so a non-JSON
        # 200 has to read as a verdict rather than a decoder traceback.
        return WhoAmI.model_validate(_decode_json(response))

    # -- unified runs --------------------------------------------------------------------------

    def resolve_run_target(self, target_id: str) -> RunTarget:
        """Resolve an opaque platform id without guessing from failed requests."""
        response = self._client.get(f"/api/run-targets/{target_id}")
        self._raise_for_error(response)
        return RunTarget.model_validate(_decode_json(response))

    def create_world_model_session(
        self, world_model_id: str, *, task: str | None = None
    ) -> RemoteWorldModelSession:
        """Open a hosted session for a platform world model."""
        response = self._client.post(
            f"/api/world-models/{world_model_id}/sessions", json={"task": task}
        )
        self._raise_for_error(response)
        return RemoteWorldModelSession.model_validate(_decode_json(response))

    def step_world_model_session(self, session_id: str, action: Action) -> Observation:
        """Advance a hosted world-model session by one action."""
        response = self._client.post(
            f"/api/sessions/{session_id}/step",
            json={"action": action.model_dump(mode="json")},
        )
        self._raise_for_error(response)
        return Observation.model_validate(_decode_json(response)["observation"])

    def create_agent_session(
        self,
        agent_id: str,
        *,
        workspace: bytes | None,
        instruction: str | None = None,
    ) -> RemoteAgentSession:
        """Create a hosted agent session, optionally staging a local snapshot."""
        payload: dict[str, JsonValue] = {"instruction": instruction}
        if workspace is not None:
            upload = self._client.post(
                f"/api/agents/{agent_id}/workspace-uploads",
                files={"workspace": ("workspace.tar.gz", workspace, "application/gzip")},
                timeout=_WORKSPACE_TIMEOUT_SECONDS,
            )
            self._raise_for_error(upload)
            payload["workspace_upload_id"] = str(_decode_json(upload)["id"])
        response = self._client.post(
            f"/api/agents/{agent_id}/sessions",
            json=payload,
        )
        self._raise_for_error(response)
        return RemoteAgentSession.model_validate(_decode_json(response))

    def get_agent_session(self, agent_id: str, session_id: str) -> RemoteAgentSession:
        """Read current hosted agent session state."""
        response = self._client.get(f"/api/agents/{agent_id}/sessions/{session_id}")
        self._raise_for_error(response)
        return RemoteAgentSession.model_validate(_decode_json(response))

    def resolve_agent_session(self, session_id: str) -> RemoteAgentSession:
        """Resolve a bare session id to its owning agent and current state."""
        response = self._client.get(f"/api/agent-sessions/{session_id}")
        self._raise_for_error(response)
        return RemoteAgentSession.model_validate(_decode_json(response))

    def end_agent_session(self, agent_id: str, session_id: str) -> RemoteAgentSession:
        """Request an end, reconciling directly when the hosted driver is gone."""
        response = self._client.post(f"/api/agents/{agent_id}/sessions/{session_id}/end")
        self._raise_for_error(response)
        return RemoteAgentSession.model_validate(_decode_json(response))

    def list_agent_session_events(
        self, agent_id: str, session_id: str, *, after: int
    ) -> RemoteAgentEventPage:
        """Poll hosted transcript events after one durable sequence cursor."""
        response = self._client.get(
            f"/api/agents/{agent_id}/sessions/{session_id}/events",
            params={"after": after},
        )
        self._raise_for_error(response)
        return RemoteAgentEventPage.model_validate(_decode_json(response))

    def post_agent_session_command(
        self, agent_id: str, session_id: str, kind: str, *, text: str | None = None
    ) -> None:
        """Steer, interrupt, or end one hosted agent session."""
        response = self._client.post(
            f"/api/agents/{agent_id}/sessions/{session_id}/commands",
            json={"kind": kind, "text": text},
        )
        self._raise_for_error(response)

    def upload_agent_workspace_patch(
        self, agent_id: str, session_id: str, content: bytes
    ) -> WorkspacePatchResult:
        """Apply local changes conditionally to a running hosted workspace."""
        response = self._client.post(
            f"/api/agents/{agent_id}/sessions/{session_id}/workspace/patches",
            files={"patch": ("workspace-patch.tar.gz", content, "application/gzip")},
            timeout=_WORKSPACE_TIMEOUT_SECONDS,
        )
        self._raise_for_error(response)
        return WorkspacePatchResult.model_validate(_decode_json(response))

    def download_agent_workspace_patch(
        self, agent_id: str, session_id: str, revision: str
    ) -> bytes:
        """Download one remote-to-local live workspace patch."""
        response = self._client.get(
            f"/api/agents/{agent_id}/sessions/{session_id}/workspace/patches/{revision}",
            timeout=_WORKSPACE_TIMEOUT_SECONDS,
        )
        self._raise_for_error(response)
        return response.content

    def acknowledge_agent_workspace_patch(
        self, agent_id: str, session_id: str, revision: str
    ) -> None:
        """Remove a remote patch after it is safely reflected or reported locally."""
        response = self._client.post(
            f"/api/agents/{agent_id}/sessions/{session_id}/workspace/patches/{revision}/ack"
        )
        self._raise_for_error(response)

    def download_agent_workspace(self, agent_id: str, session_id: str) -> bytes:
        """Download a terminal hosted session's final E2B workspace snapshot."""
        response = self._client.get(
            f"/api/agents/{agent_id}/sessions/{session_id}/workspace",
            timeout=_WORKSPACE_TIMEOUT_SECONDS,
        )
        self._raise_for_error(response)
        return response.content

    def acknowledge_agent_workspace(self, agent_id: str, session_id: str) -> None:
        """Confirm the final archive is safe locally so platform objects can be removed."""
        response = self._client.post(f"/api/agents/{agent_id}/sessions/{session_id}/workspace/ack")
        self._raise_for_error(response)

    # -- world models --------------------------------------------------------------------------

    def list_world_models(self, org_id: str) -> list[RemoteWorldModel]:
        response = self._client.get(f"/api/orgs/{org_id}/world-models")
        self._raise_for_error(response)
        rows = _rows(_decode_json(response), "world_models")
        return [RemoteWorldModel.model_validate(row) for row in rows]

    def push_model_bundle(
        self,
        org_id: str,
        name: str,
        bundle_path: Path,
        sha256: str,
        byte_size: int,
        meta: dict[str, JsonValue],
    ) -> RemoteWorldModel:
        """Push a packed bundle file: ticket, direct PUT to storage, finalize.

        The bundle bytes stream from disk straight to the signed staging URL;
        only the finalize declaration (digest + size + serve metadata) goes
        through the API.
        """
        ticket_response = self._client.post(
            f"/api/orgs/{org_id}/world-models/{name}/bundle/uploads"
        )
        self._raise_for_error(ticket_response)
        ticket = _decode_json(ticket_response)
        upload_url = str(ticket["upload_url"])
        with bundle_path.open("rb") as fh:
            upload_response = self._transfer.put(
                upload_url,
                content=fh,
                headers={
                    "Content-Type": "application/gzip",
                    "x-upsert": "false",
                    "Authorization": f"Bearer {ticket.get('token', '')}",
                },
            )
        if not upload_response.is_success:
            msg = (
                f"bundle upload to storage failed with HTTP {upload_response.status_code}: "
                f"{upload_response.text[:200]}"
            )
            raise PlatformError(msg, status_code=upload_response.status_code)

        finalize = self._client.post(
            f"/api/orgs/{org_id}/world-models/{name}/bundle",
            json={
                "staging_path": ticket["staging_path"],
                "sha256": sha256,
                "byte_size": byte_size,
                "meta": meta,
            },
        )
        self._raise_for_error(finalize)
        return RemoteWorldModel.model_validate(_decode_json(finalize))

    def get_endpoint(self, org_id: str, name: str) -> JsonObject | None:
        """One endpoint's detail, or ``None`` when the org has no endpoint by that name.

        A 404 is an answer here, not a failure: the push flow asks this question
        to decide between creating the endpoint and updating it in place.
        """
        response = self._client.get(f"/api/orgs/{org_id}/endpoints/{name}")
        if response.status_code == 404:
            return None
        self._raise_for_error(response)
        decoded = _decode_json(response)
        if not isinstance(decoded, dict):
            msg = f"endpoint detail returned a {type(decoded).__name__}, expected an object"
            raise PlatformError(msg)
        return decoded

    def create_endpoint(
        self,
        org_id: str,
        name: str,
        *,
        world_model_id: str | None = None,
        model: str | None = None,
    ) -> JsonObject:
        """Create one endpoint, optionally tied to a world model.

        Args:
            org_id: Organization to create under.
            name: Endpoint slug.
            world_model_id: Simulation the endpoint links to, or ``None`` for an
                endpoint whose evidence comes from a real benchmark.
            model: Pool entry name for the day-one static policy; the platform
                default serves when omitted. Deployments differ in which models
                they hold credentials for, so the platform's refusal names the
                serveable set when this pick (or its default) cannot serve.

        Returns:
            The created endpoint summary.
        """
        body: dict[str, JsonValue] = {"name": name, "world_model_id": world_model_id}
        if model is not None:
            body["model"] = model
        response = self._client.post(f"/api/orgs/{org_id}/endpoints", json=body)
        self._raise_for_error(response)
        decoded = _decode_json(response)
        if not isinstance(decoded, dict):
            msg = f"endpoint create returned a {type(decoded).__name__}, expected an object"
            raise PlatformError(msg)
        return decoded

    def install_endpoint_artifacts(
        self, org_id: str, name: str, *, policy: JsonObject, report: JsonObject
    ) -> JsonObject:
        """Install a measured policy and its improvement report on an endpoint.

        The JSON body twin of :meth:`install_endpoint_policy`, for policies with
        no evidence bank (static, rank): the pair replaces the endpoint's day-one
        policy and null report wholesale. A knn policy is two artifacts and must
        go through the multipart installer instead; the platform refuses it here.

        Returns:
            The endpoint detail the platform returns.
        """
        response = self._client.put(
            f"/api/orgs/{org_id}/endpoints/{name}/artifacts",
            json={"policy": policy, "report": report},
        )
        self._raise_for_error(response)
        decoded = _decode_json(response)
        if not isinstance(decoded, dict):
            msg = f"artifacts install returned a {type(decoded).__name__}, expected an object"
            raise PlatformError(msg)
        return decoded

    def install_endpoint_policy(
        self,
        org_id: str,
        endpoint: str,
        policy_path: Path,
        bank_path: Path | None,
        report_path: Path | None = None,
    ) -> JsonValue:
        """Install a fitted routing policy on a hosted endpoint.

        Multipart rather than a signed-URL handoff like `push_model_bundle`: a policy
        bank is a few hundred KB to a few MB, not a whole model bundle, and the server
        has to validate the bank AGAINST the policy before storing it. Splitting them
        across two requests would mean accepting bytes it cannot yet check.

        Args:
            org_id: Organization that owns the endpoint.
            endpoint: Endpoint slug (the `model` a customer's client sends).
            policy_path: Fitted `policy.json`.
            bank_path: The `.npz` evidence sidecar; required for a knn policy, and
                ``None`` for static or rank, which have none.
            report_path: Optional improvement-report JSON to publish alongside.

        Returns:
            The endpoint summary the platform returns.
        """
        # File reads are this client's own failure surface: a path that vanished or
        # became unreadable between the CLI's validation and this call must surface
        # as the PlatformError every caller already renders, not a raw traceback.
        try:
            files: dict[str, tuple[str, bytes, str]] = {}
            if bank_path is not None:
                files["bank"] = (
                    bank_path.name,
                    bank_path.read_bytes(),
                    "application/octet-stream",
                )
            data = {"policy": policy_path.read_text(encoding="utf-8")}
            if report_path is not None:
                data["report"] = report_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            msg = f"could not read the policy artifacts to install: {error}"
            raise PlatformError(msg) from error
        response = self._client.put(
            f"/api/orgs/{org_id}/endpoints/{endpoint}/policy",
            data=data,
            files=files or None,
        )
        self._raise_for_error(response)
        return _decode_json(response)

    def download_model_bundle(self, org_id: str, name: str, dest: Path) -> str:
        """Stream a model's bundle from storage to ``dest``, verifying its digest.

        The API hands back an expiring signed URL plus the recorded sha256;
        the bytes come straight from storage's CDN and are hashed as they
        stream to disk.

        Returns:
            The verified sha256 hex digest.
        """
        response = self._client.get(f"/api/orgs/{org_id}/world-models/{name}/bundle")
        self._raise_for_error(response)
        payload = _decode_json(response)
        declared = str(payload["sha256"])

        digest = hashlib.sha256()
        part_path = dest.with_name(f"{dest.name}.part")
        with self._transfer.stream("GET", str(payload["url"])) as stream:
            if not stream.is_success:
                msg = f"bundle download failed with HTTP {stream.status_code}"
                raise PlatformError(msg, status_code=stream.status_code)
            with part_path.open("wb") as fh:
                for chunk in stream.iter_bytes():
                    digest.update(chunk)
                    fh.write(chunk)
        actual = digest.hexdigest()
        if actual != declared:
            part_path.unlink(missing_ok=True)
            msg = f"bundle digest mismatch for {name}: expected {declared}, got {actual}"
            raise PlatformError(msg)
        part_path.replace(dest)
        return actual

    # -- harnesses -----------------------------------------------------------------------------

    def list_harnesses(self, org_id: str) -> list[RemoteHarness]:
        response = self._client.get(f"/api/orgs/{org_id}/harnesses")
        self._raise_for_error(response)
        rows = _rows(_decode_json(response), "harnesses")
        return [RemoteHarness.model_validate(row) for row in rows]

    def get_harness(
        self, org_id: str, name: str
    ) -> tuple[RemoteHarness, list[RemoteHarnessVersion]]:
        response = self._client.get(f"/api/orgs/{org_id}/harnesses/{name}")
        self._raise_for_error(response)
        payload = _decode_json(response)
        harness = RemoteHarness.model_validate(payload["harness"])
        rows = _rows(payload, "versions")
        versions = [RemoteHarnessVersion.model_validate(row) for row in rows]
        return harness, versions

    def get_harness_version(self, org_id: str, name: str, version: int) -> HarnessVersionDoc:
        response = self._client.get(f"/api/orgs/{org_id}/harnesses/{name}/versions/{version}")
        self._raise_for_error(response)
        return HarnessVersionDoc.model_validate(_decode_json(response))

    def push_harness_version(
        self,
        org_id: str,
        name: str,
        doc: dict[str, JsonValue],
        doc_hash: str,
    ) -> PushedHarnessVersion:
        response = self._client.post(
            f"/api/orgs/{org_id}/harnesses/{name}/versions",
            json={"doc": doc, "doc_hash": doc_hash},
        )
        self._raise_for_error(response)
        return PushedHarnessVersion.model_validate(_decode_json(response))

    # -- built-in local pi runs ---------------------------------------------------------------

    def create_local_pi_run(self, org_id: str) -> LocalPiRunInfo:
        """Open a metered platform run for WMO's built-in local pi harness."""
        response = self._client.post(f"/api/orgs/{org_id}/local-pi-runs")
        self._raise_for_error(response)
        return LocalPiRunInfo.model_validate(_decode_json(response))

    def complete_local_pi_worker(
        self, org_id: str, run_id: str, request: ChatRequest
    ) -> ChatResponse:
        """Answer one built-in pi worker turn through the platform."""
        response = self._client.post(
            f"/api/orgs/{org_id}/local-pi-runs/{run_id}/worker-completion",
            json=request.model_dump(mode="json", exclude_none=True),
        )
        self._raise_for_error(response)
        return ChatResponse.model_validate(_decode_json(response))

    def finish_local_pi_run(
        self,
        org_id: str,
        run_id: str,
        *,
        status: str,
        ended_reason: str,
        error: str | None = None,
    ) -> None:
        """Report the terminal transition of a built-in local pi run."""
        response = self._client.post(
            f"/api/orgs/{org_id}/local-pi-runs/{run_id}/finish",
            json={"status": status, "ended_reason": ended_reason, "error": error},
        )
        self._raise_for_error(response)

    # -- run telemetry -------------------------------------------------------------------------

    def push_run_events(
        self,
        org_id: str,
        external_id: str,
        *,
        emitter_id: str,
        events: Sequence[JsonObject],
    ) -> JsonObject:
        """Push one batch of run telemetry.

        The run is named in the path (its name carries slashes, which the
        platform routes as a path parameter) and the body is the feeding
        process plus its events. The response carries the newly-accepted count,
        the run's resume mark, and any pending control commands.
        """
        response = self._client.post(
            f"/api/orgs/{org_id}/runs/{_run_path(external_id)}/events",
            json={"emitter_id": emitter_id, "events": list(events)},
        )
        self._raise_for_error(response)
        return _decode_json(response)

    def ack_run_control(
        self,
        org_id: str,
        external_id: str,
        control_id: str,
        *,
        status: str,
        note: str | None = None,
    ) -> JsonObject:
        """Answer one pulled control command; `rejected` with a note is legal."""
        body: JsonObject = {"status": status}
        if note is not None:
            body["note"] = note
        response = self._client.post(
            f"/api/orgs/{org_id}/runs/{_run_path(external_id)}/control/{control_id}/ack",
            json=body,
        )
        self._raise_for_error(response)
        return _decode_json(response)

    # -- run telemetry reads -------------------------------------------------------------------

    def list_org_runs(
        self,
        org_id: str,
        *,
        status: str | None = None,
        kind: str | None = None,
        cursor_ts: str | None = None,
        cursor_id: str | None = None,
        limit: int = 50,
    ) -> JsonObject:
        """One keyset page of an organization's runs, newest first.

        `next_cursor` in the reply is two fields (`ts`, `id`) which echo back as `cursor_ts` and
        `cursor_id`; there is no opaque cursor string.
        """
        params: dict[str, str | int] = {"limit": limit}
        for key, value in (
            ("status", status),
            ("kind", kind),
            ("cursor_ts", cursor_ts),
            ("cursor_id", cursor_id),
        ):
            if value is not None:
                params[key] = value
        return self._read(f"/api/orgs/{org_id}/runs", params)

    def get_org_run(self, org_id: str, external_id: str) -> JsonObject:
        """One run with its stages, per-model cell rollup, and pending control commands."""
        return self._read(f"/api/orgs/{org_id}/runs/{_run_path(external_id)}", {})

    def list_org_run_cells(
        self,
        org_id: str,
        external_id: str,
        *,
        model: str | None = None,
        scored: bool | None = None,
        error: bool | None = None,
        cursor_key: str | None = None,
        limit: int = 100,
    ) -> JsonObject:
        """One keyset page of a run's measured cells, `cell_key` ascending."""
        params: dict[str, str | int | bool] = {"limit": limit}
        for key, value in (("model", model), ("cursor_key", cursor_key)):
            if value is not None:
                params[key] = value
        for key, flag in (("scored", scored), ("error", error)):
            if flag is not None:
                params[key] = flag
        return self._read(f"/api/orgs/{org_id}/runs/{_run_path(external_id)}/cells", params)

    def list_org_run_events(
        self,
        org_id: str,
        external_id: str,
        *,
        after_pos: int = 0,
        limit: int = 500,
        tail: bool = False,
        event_type: str | None = None,
    ) -> JsonObject:
        """One page of a run's raw event log in arrival order.

        The cursor is `pos` (the server's arrival position), never the emitter's `seq`: writers
        feeding one run own disjoint seq ranges, so a seq cursor would hide a late low-band write.
        """
        params: dict[str, str | int | bool] = {
            "after_pos": after_pos,
            "limit": limit,
            "tail": tail,
        }
        if event_type is not None:
            params["type"] = event_type
        return self._read(f"/api/orgs/{org_id}/runs/{_run_path(external_id)}/events", params)

    @contextmanager
    def stream_org_run_events(
        self, org_id: str, external_id: str, *, after_pos: int = 0, timeout_s: float = 300.0
    ) -> Iterator[Iterator[JsonObject]]:
        """Open the resumable SSE tail of a run's log, yielding one decoded frame body at a time.

        `Last-Event-ID` travels beside the query cursor because the server prefers the header on a
        reconnect; sending both is what keeps a resume exact. The stream closes on its own once a
        terminal run is drained, which is what lets a tail exit instead of hanging.

        A context manager, so the connection is always released; the caller owns the reconnect
        policy (see `wmo.runs.reader`).
        """
        headers = {"Accept": "text/event-stream"}
        if after_pos:
            headers["Last-Event-ID"] = str(after_pos)
        with self._client.stream(
            "GET",
            f"/api/orgs/{org_id}/runs/{_run_path(external_id)}/stream",
            params={"after_pos": after_pos},
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=30.0),
        ) as response:
            if not response.is_success:
                response.read()
                self._raise_for_error(response)
            yield _sse_frames(response)

    def request_org_run_control(
        self, org_id: str, external_id: str, *, command: str, args: JsonObject | None = None
    ) -> JsonObject:
        """Queue one control command for whichever process is feeding a run.

        Delivery is pull-based and queueing does not change the run's status: a stop that never
        reaches its machine must not make the panel claim the run stopped.
        """
        response = self._client.post(
            f"/api/orgs/{org_id}/runs/{_run_path(external_id)}/control",
            json={"command": command, "args": args or {}},
        )
        self._raise_for_error(response)
        return _decode_json(response)

    # -- internals -----------------------------------------------------------------------------

    def _read(self, path: str, params: dict[str, str | int | bool]) -> JsonObject:
        """One authenticated GET returning a JSON object, or a `PlatformError`."""
        response = self._client.get(path, params=params)
        self._raise_for_error(response)
        return _decode_json(response)

    def _raise_for_error(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            message = response.json().get("error", response.text)
        except (json.JSONDecodeError, ValueError):
            message = response.text or f"HTTP {response.status_code}"
        if response.status_code == 401:
            message = f"{message} — run `wmo login` (or check WMO_PLATFORM_TOKEN)"
        raise PlatformError(str(message), status_code=response.status_code)
