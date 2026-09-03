"""Control-plane envelope tests: auth, shapes, and error status mapping."""

from __future__ import annotations

import base64
import json
from typing import NoReturn

from exp.runtime.gateway.batch.engine import BatchEngine
from exp.runtime.gateway.batch.engine_test import (
    MemoryCatalog,
    MemoryFiles,
    MemoryLedger,
    MemorySecrets,
    MemoryStore,
    _chat_line,
)
from exp.runtime.gateway.batch.plane import BatchControlPlane
from exp.runtime.gateway.sqlite.store import InvalidVirtualKeyError


class MemoryControl:
    """GatewayControlStore double accepting exactly one key."""

    def authenticate_key(self, *, raw_key: str) -> None:
        """Accept only the fixture key; a lookup outage is a distinct failure."""
        if raw_key == "xpl_outage":
            raise RuntimeError("store unreachable")
        if raw_key != "xpl_good":
            raise InvalidVirtualKeyError("unknown key")

    def authenticated_identity(self, *, raw_key: str) -> tuple[str, str]:
        """Return the fixture identity."""
        return ("org_a", "id_a")

    def authorize_request(self, **_: object) -> NoReturn:
        """Unused by the batch plane."""
        raise NotImplementedError

    def granted_aliases(self, **_: object) -> NoReturn:
        """Unused by the batch plane."""
        raise NotImplementedError

    def granted_alias_authorities(self, **_: object) -> NoReturn:
        """Unused by the batch plane."""
        raise NotImplementedError


def _plane() -> BatchControlPlane:
    """Compose one plane over fresh in-memory seams."""
    engine = BatchEngine(
        store=MemoryStore(),
        files=MemoryFiles(),
        catalog=MemoryCatalog(),
        ledger=MemoryLedger(),
        secrets_resolver=MemorySecrets(),
    )
    return BatchControlPlane(engine=engine, control=MemoryControl())


def _call(plane: BatchControlPlane, method: str, **payload: object) -> tuple[int, dict]:
    """Invoke one plane method and split its envelope."""
    rendered = getattr(plane, method)(json.dumps({"bearer_key": "xpl_good", **payload}))
    parsed = json.loads(rendered)
    return parsed["status"], parsed["body"]


def test_invalid_key_maps_to_the_uniform_401() -> None:
    """A bad or missing bearer key produces the synchronous lane's 401 envelope."""
    plane = _plane()
    for argument in ({"bearer_key": "xpl_bad"}, {"bearer_key": ""}, {}):
        parsed = json.loads(plane.batch_list(json.dumps(argument)))
        assert parsed["status"] == 401, argument
        assert parsed["body"]["error"] == {
            "message": parsed["body"]["error"]["message"],
            "type": "authentication_error",
            "code": "invalid_key",
        }


def test_key_lookup_outage_is_an_internal_error_not_a_key_verdict() -> None:
    """A failing control store answers 500, never 401: the key was not judged."""
    parsed = json.loads(_plane().batch_list(json.dumps({"bearer_key": "xpl_outage"})))
    assert parsed["status"] == 500
    assert parsed["body"]["error"] == {
        "message": "the gateway request failed",
        "type": "api_error",
        "code": "internal_error",
    }


def test_file_roundtrip_and_batch_lifecycle_through_the_plane() -> None:
    """Upload, submit, retrieve, and list all shape OpenAI-compatible objects."""
    plane = _plane()
    content = "\n".join([_chat_line("a"), _chat_line("b")]).encode("utf-8")
    status, file_object = _call(
        plane,
        "file_create",
        filename="input.jsonl",
        purpose="batch",
        content_b64=base64.b64encode(content).decode("ascii"),
    )
    assert status == 200 and file_object["object"] == "file"
    status, batch_object = _call(
        plane,
        "batch_create",
        input_file_id=file_object["id"],
        endpoint="/v1/chat/completions",
        metadata={"job": "nightly"},
    )
    assert status == 200
    assert batch_object["object"] == "batch"
    assert batch_object["status"] == "validating"
    assert batch_object["request_counts"]["total"] == 2
    status, fetched = _call(plane, "batch_retrieve", batch_id=batch_object["id"])
    assert status == 200 and fetched["id"] == batch_object["id"]
    status, listing = _call(plane, "batch_list")
    assert status == 200 and len(listing["data"]) == 1
    status, roundtrip = _call(plane, "file_content", file_id=file_object["id"])
    assert status == 200
    assert base64.b64decode(roundtrip["content_b64"]) == content


def test_submit_rejection_maps_to_the_openai_error_envelope() -> None:
    """A validation refusal renders status 400 with the message."""
    plane = _plane()
    status, body = _call(
        plane, "batch_create", input_file_id="file_missing", endpoint="/v1/chat/completions"
    )
    assert status == 400
    assert body["error"]["type"] == "invalid_request_error"
    assert "does not exist" in body["error"]["message"]


def test_malformed_payloads_and_bad_base64_are_client_errors() -> None:
    """Envelope parsing failures never raise; they render 400s."""
    plane = _plane()
    parsed = json.loads(plane.batch_retrieve("not json"))
    assert parsed["status"] == 400
    status, body = _call(plane, "file_create", purpose="batch", content_b64="@@@")
    assert status == 400 and "base64" in body["error"]["message"]


def test_unknown_ids_map_to_404() -> None:
    """Missing files and batches produce not_found envelopes."""
    plane = _plane()
    for method, key in (("batch_retrieve", "batch_id"), ("file_retrieve", "file_id")):
        status, body = _call(plane, method, **{key: "missing"})
        assert status == 404 and body["error"]["code"] == "not_found"
