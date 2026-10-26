"""Bridge-callable ``/v1/videos`` methods for the native control plane.

The native routes call these by name. A composition that installs a video job
store (``components.video_jobs``) gets a :class:`VideoEngine` over the same
authority, routes, catalogs, write ledger, and guardrails as every other
surface, plus the background poller that settles jobs nobody polls; a
composition without one answers every video route with a uniform 404. An
optional ``components.video_outputs``
(:class:`~exp.runtime.gateway.video.output.VideoOutputStore`) makes xAI upload
each rendered video into host storage (required for Zero Data Retention teams).

Every read authenticates the bearer key into its organization and identity
first and is scoped to them: another identity's id answers exactly like an
absent id.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Protocol

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.native_accounting_errors import NativeBridgeError, authority_error
from exp.runtime.gateway.native_components import NativeGatewayComponents, SyncWriteLedger
from exp.runtime.gateway.video.contracts import render_deleted_object, render_video_object
from exp.runtime.gateway.video.engine import VideoEngine
from exp.runtime.gateway.video.output import VideoOutputStore
from exp.runtime.gateway.video.store import ListOrder, VideoJobStore
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.video_requests import decode_video_create

_CLIENT_HEADER_KEYS = ("user_agent", "originator", "app_title", "app_referer")


class _VideoPlane(Protocol):
    """The control-plane state the video methods read."""

    _components: NativeGatewayComponents
    _write_ledger: SyncWriteLedger
    _guardrails: GuardrailEngine | None
    _videos: VideoEngine | None


def _disabled() -> NativeBridgeError:
    """The uniform answer on gateways without a video job store."""
    return NativeBridgeError(
        OpenAIProtocolError(
            status_code=404,
            code="not_found",
            message="Video generation is not enabled on this gateway.",
        )
    )


def _string(data: JsonObject, key: str) -> str:
    """Read one required non-empty string argument from the data plane.

    Raises:
        NativeBridgeError: The argument is absent or empty (a 400 on its name).
    """
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise NativeBridgeError(
            OpenAIProtocolError(
                status_code=400, code="invalid_value", message=f"{key} is required.", param=key
            )
        )
    return value


def _video_owner(plane: _VideoPlane, data: JsonObject) -> tuple[VideoEngine, str, str]:
    """Return the engine and the bearer key's organization and identity.

    Raises:
        NativeBridgeError: Videos are disabled, or the key is invalid.
    """
    if plane._videos is None:
        raise _disabled()
    raw_key = _string(data, "bearer_key")
    try:
        plane._components.store.authenticate_key(raw_key=raw_key)
        organization_id, identity_id = plane._components.store.authenticated_identity(
            raw_key=raw_key
        )
    except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
        raise authority_error(exc) from exc
    return plane._videos, organization_id, identity_id


class NativeVideosMixin:
    """The ``video_*`` boundary methods."""

    _videos: VideoEngine | None

    def _bind_videos(self: _VideoPlane, *, start_poller: bool = True) -> None:
        """Build the video engine when the composition installs a job store.

        Args:
            start_poller: Whether to start the background settlement poller.
        """
        jobs: VideoJobStore | None = getattr(self._components, "video_jobs", None)
        outputs: VideoOutputStore | None = getattr(self._components, "video_outputs", None)
        self._videos = (
            None
            if jobs is None
            else VideoEngine(
                components=self._components,
                ledger=self._write_ledger,
                jobs=jobs,
                guardrails=self._guardrails,
                outputs=outputs,
            )
        )
        if self._videos is not None and start_poller:
            self._videos.start_poller()

    def video_create(self: _VideoPlane, argument: str) -> str:
        """Admit and submit one video job.

        Args:
            argument: JSON with ``bearer_key``, ``fields`` (the JSON body or the
                multipart text fields), an optional ``reference`` holding the
                uploaded file part's ``content_type`` and base64 ``data``, and
                the forwarded calling-application headers.

        Returns:
            The OpenAI ``video`` object as JSON.

        Raises:
            NativeBridgeError: Any refusal, with its public error.
        """
        if self._videos is None:
            raise _disabled()
        data = json.loads(argument)
        raw_key = _string(data, "bearer_key")
        fields = data.get("fields")
        if not isinstance(fields, dict):
            raise NativeBridgeError(
                OpenAIProtocolError(
                    status_code=400,
                    code="invalid_request",
                    message="The request body must be a JSON object or multipart form data.",
                )
            )
        uploaded: tuple[str, bytes] | None = None
        reference = data.get("reference")
        if isinstance(reference, dict):
            try:
                uploaded = (
                    str(reference.get("content_type") or ""),
                    base64.b64decode(str(reference.get("data") or ""), validate=True),
                )
            except (binascii.Error, ValueError) as exc:  # pragma: no cover - the plane encodes it.
                raise NativeBridgeError(
                    OpenAIProtocolError(
                        status_code=400,
                        code="invalid_value",
                        message="input_reference could not be read.",
                        param="input_reference",
                    )
                ) from exc
        try:
            decoded = decode_video_create(fields, uploaded_reference=uploaded)
        except OpenAIProtocolError as exc:
            raise NativeBridgeError(exc) from exc
        job = self._videos.create(
            raw_key=raw_key,
            alias=decoded.alias,
            request=decoded.request,
            reference_image=decoded.reference_image,
            client_headers={key: data[key] for key in _CLIENT_HEADER_KEYS if key in data},
        )
        return json.dumps(render_video_object(job), separators=(",", ":"))

    def video_retrieve(self: _VideoPlane, argument: str) -> str:
        """Return one owned video object, refreshed from its provider.

        Args:
            argument: JSON with ``bearer_key`` and ``video_id``.

        Returns:
            The OpenAI ``video`` object as JSON.
        """
        data = json.loads(argument)
        engine, organization_id, identity_id = _video_owner(self, data)
        job = engine.owned(
            organization_id=organization_id,
            identity_id=identity_id,
            video_id=_string(data, "video_id"),
        )
        return json.dumps(render_video_object(job), separators=(",", ":"))

    def video_list(self: _VideoPlane, argument: str) -> str:
        """List the caller's videos.

        Args:
            argument: JSON with ``bearer_key`` and optional ``limit``, ``after``,
                and ``order`` (``asc`` or ``desc``, default ``desc``).

        Returns:
            The OpenAI list object as JSON.
        """
        data = json.loads(argument)
        engine, organization_id, identity_id = _video_owner(self, data)
        raw_limit = data.get("limit")
        limit = (
            int(raw_limit)
            if isinstance(raw_limit, str | int) and str(raw_limit).isdecimal()
            else 20
        )
        order: ListOrder = "asc" if data.get("order") == "asc" else "desc"
        after = data.get("after")
        jobs, has_more = engine.list_jobs(
            organization_id=organization_id,
            identity_id=identity_id,
            limit=limit,
            after=after if isinstance(after, str) and after else None,
            order=order,
        )
        rendered = [render_video_object(job) for job in jobs]
        page: JsonObject = {
            "object": "list",
            "data": rendered,
            "first_id": rendered[0]["id"] if rendered else None,
            "last_id": rendered[-1]["id"] if rendered else None,
            "has_more": has_more,
        }
        return json.dumps(page, separators=(",", ":"))

    def video_delete(self: _VideoPlane, argument: str) -> str:
        """Delete one owned, finished video from reads and listings.

        Args:
            argument: JSON with ``bearer_key`` and ``video_id``.

        Returns:
            The OpenAI ``video.deleted`` object as JSON.
        """
        data = json.loads(argument)
        engine, organization_id, identity_id = _video_owner(self, data)
        video_id = _string(data, "video_id")
        engine.delete(organization_id=organization_id, identity_id=identity_id, video_id=video_id)
        return json.dumps(render_deleted_object(video_id), separators=(",", ":"))

    def video_content(self: _VideoPlane, argument: str) -> str:
        """Describe how the data plane downloads one owned, completed video.

        Args:
            argument: JSON with ``bearer_key``, ``video_id``, and optional ``variant``.

        Returns:
            JSON ``{"url", "headers", "content_type"}``; the data plane sends the
            headers to that URL only and never across a redirect.
        """
        data = json.loads(argument)
        engine, organization_id, identity_id = _video_owner(self, data)
        variant = data.get("variant")
        fetch = engine.content(
            organization_id=organization_id,
            identity_id=identity_id,
            video_id=_string(data, "video_id"),
            variant=variant if isinstance(variant, str) and variant else "video",
        )
        return json.dumps(
            {"url": fetch.url, "headers": fetch.headers, "content_type": fetch.content_type},
            separators=(",", ":"),
        )
