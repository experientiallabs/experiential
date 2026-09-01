"""String-JSON control-plane methods for the batch routes.

The native data plane calls these methods by name through its bridge, one
JSON-string argument in and one JSON-string result out, mirroring the
synchronous lane's control-plane convention. Authentication reuses the host's
gateway control store, so a batch caller holds exactly the same virtual-key
authority as a synchronous caller.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.batch.contracts import BatchSubmitError
from exp.runtime.gateway.batch.engine import BatchEngine
from exp.runtime.gateway.interfaces import GatewayControlStore

_LOGGER = logging.getLogger(__name__)

_ERROR_STATUS: dict[str, int] = {
    "invalid_request_error": 400,
    "insufficient_quota": 402,
    "not_found": 404,
    "cancel_unsupported": 409,
    "provider_error": 502,
}


class BatchControlPlane:
    """Bridge-callable batch methods over the engine and the host's identity."""

    def __init__(self, *, engine: BatchEngine, control: GatewayControlStore) -> None:
        """Bind the engine and the identity authority."""
        self._engine = engine
        self._control = control

    def _identity(self, argument: JsonObject) -> tuple[str, str]:
        """Authenticate the carried bearer key into an owning identity.

        Raises:
            BatchSubmitError: When the key is absent or invalid.
        """
        raw_key = argument.get("bearer_key")
        if not isinstance(raw_key, str) or not raw_key:
            raise BatchSubmitError("missing bearer key", code="not_found")
        try:
            self._control.authenticate_key(raw_key=raw_key)
            return self._control.authenticated_identity(raw_key=raw_key)
        except Exception as exc:
            raise BatchSubmitError("invalid or revoked key", code="not_found") from exc

    @staticmethod
    def _decode(argument: str) -> JsonObject:
        """Parse the bridge's JSON-string argument into an object.

        Raises:
            BatchSubmitError: When the argument is not a JSON object.
        """
        try:
            parsed = json.loads(argument)
        except json.JSONDecodeError as exc:
            raise BatchSubmitError("malformed control payload") from exc
        if not isinstance(parsed, dict):
            raise BatchSubmitError("control payload must be an object")
        return parsed

    @staticmethod
    def _envelope(payload: JsonObject, *, status: int = 200) -> str:
        """Render one route result envelope for the data plane."""
        return json.dumps({"status": status, "body": payload})

    @staticmethod
    def _error(error: BatchSubmitError) -> str:
        """Render one OpenAI-envelope error for the data plane."""
        status = _ERROR_STATUS.get(error.code, 400)
        return json.dumps(
            {
                "status": status,
                "body": {
                    "error": {
                        "message": error.message,
                        "type": "invalid_request_error" if status < 500 else "api_error",
                        "code": error.code,
                    }
                },
            }
        )

    def file_create(self, argument: str) -> str:
        """Store one uploaded batch input file: {bearer_key, filename, purpose, content_b64}."""
        try:
            payload = self._decode(argument)
            organization_id, _ = self._identity(payload)
            content_b64 = payload.get("content_b64")
            if not isinstance(content_b64, str):
                raise BatchSubmitError("file content is missing")
            try:
                content = base64.b64decode(content_b64, validate=True)
            except (ValueError, TypeError) as exc:
                raise BatchSubmitError("file content is not valid base64") from exc
            record = self._engine.upload_file(
                organization_id=organization_id,
                filename=str(payload.get("filename") or "batch.jsonl"),
                purpose=str(payload.get("purpose") or ""),
                content=content,
            )
            return self._envelope(record.public_object())
        except BatchSubmitError as error:
            return self._error(error)

    def file_retrieve(self, argument: str) -> str:
        """Return one owned file's metadata object: {bearer_key, file_id}."""
        try:
            payload = self._decode(argument)
            organization_id, _ = self._identity(payload)
            record = self._engine.file_metadata(
                organization_id=organization_id, file_id=str(payload.get("file_id") or "")
            )
            if record is None:
                raise BatchSubmitError("file does not exist", code="not_found")
            return self._envelope(record.public_object())
        except BatchSubmitError as error:
            return self._error(error)

    def file_content(self, argument: str) -> str:
        """Return one owned file's content as base64: {bearer_key, file_id}."""
        try:
            payload = self._decode(argument)
            organization_id, _ = self._identity(payload)
            content = self._engine.file_content(
                organization_id=organization_id, file_id=str(payload.get("file_id") or "")
            )
            if content is None:
                raise BatchSubmitError("file does not exist", code="not_found")
            return self._envelope({"content_b64": base64.b64encode(content).decode("ascii")})
        except BatchSubmitError as error:
            return self._error(error)

    def batch_create(self, argument: str) -> str:
        """Submit one batch: {bearer_key, input_file_id, endpoint, metadata?}."""
        try:
            payload = self._decode(argument)
            organization_id, identity_id = self._identity(payload)
            metadata = payload.get("metadata")
            job = self._engine.submit(
                organization_id=organization_id,
                identity_id=identity_id,
                input_file_id=str(payload.get("input_file_id") or ""),
                endpoint=str(payload.get("endpoint") or ""),
                metadata=(
                    {str(key): str(value) for key, value in metadata.items()}
                    if isinstance(metadata, dict)
                    else None
                ),
            )
            return self._envelope(job.public_object())
        except BatchSubmitError as error:
            return self._error(error)

    def batch_retrieve(self, argument: str) -> str:
        """Return one owned batch object: {bearer_key, batch_id}."""
        try:
            payload = self._decode(argument)
            organization_id, _ = self._identity(payload)
            job = self._engine.retrieve(
                organization_id=organization_id, batch_id=str(payload.get("batch_id") or "")
            )
            if job is None:
                raise BatchSubmitError("batch does not exist", code="not_found")
            return self._envelope(job.public_object())
        except BatchSubmitError as error:
            return self._error(error)

    def batch_list(self, argument: str) -> str:
        """Return the caller's batches page: {bearer_key, limit?, after?}."""
        try:
            payload = self._decode(argument)
            organization_id, _ = self._identity(payload)
            limit = payload.get("limit")
            after = payload.get("after")
            jobs = self._engine.list_jobs(
                organization_id=organization_id,
                limit=limit if isinstance(limit, int) else 20,
                after=after if isinstance(after, str) and after else None,
            )
            return self._envelope(
                {
                    "object": "list",
                    "data": [job.public_object() for job in jobs],
                    "has_more": False,
                }
            )
        except BatchSubmitError as error:
            return self._error(error)

    def batch_cancel(self, argument: str) -> str:
        """Request cancellation of one owned batch: {bearer_key, batch_id}."""
        try:
            payload = self._decode(argument)
            organization_id, _ = self._identity(payload)
            job = asyncio.run(
                self._engine.cancel(
                    organization_id=organization_id,
                    batch_id=str(payload.get("batch_id") or ""),
                )
            )
            return self._envelope(job.public_object())
        except BatchSubmitError as error:
            return self._error(error)
