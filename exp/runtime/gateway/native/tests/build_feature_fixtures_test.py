"""Generate signing and authenticated-carrier cases from the existing Python contracts."""

from __future__ import annotations

import base64
import json
import random
from dataclasses import fields
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote, urlsplit

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.reasoning_carrier import seal_reasoning_content
from exp.runtime.gateway.reasoning_carrier_test import _authority, _tool_calls


def signing_cases() -> list[JsonObject]:
    """Cover encoded model IDs, session tokens, regions and exact Unicode bodies."""
    from hashlib import sha256  # noqa: PLC0415 - generator-local fixture utility

    cases: list[JsonObject] = []
    for index in range(60):
        region = ["us-east-1", "eu-west-2", "us-west-2"][index % 3]
        model = ["anthropic.claude:0", "arn:aws:bedrock:model/a b", "x~y/z%25"][index % 3]
        path = "converse-stream" if index % 2 else "invoke"
        url = (
            f"https://bedrock-runtime.{region}.amazonaws.com/model/{quote(model, safe='/~')}/{path}"
        )
        body = json.dumps({"input": "Hello 世界", "n": index}, ensure_ascii=index % 2 == 0)
        token = "synthetic-session-token" if index % 2 else None
        accept = "application/vnd.amazon.eventstream" if index % 2 else "application/json"
        now = datetime(2026, 10, 31, 23, 59, tzinfo=UTC) + timedelta(minutes=index)
        credentials = Credentials("SYNTHETICKEY", "synthetic-signing-secret", token)
        request = AWSRequest(
            method="POST",
            url=url,
            data=body.encode(),
            headers={"content-type": "application/json", "accept": accept},
        )
        with patch("botocore.auth.get_current_datetime", return_value=now):
            SigV4Auth(credentials, "bedrock", region).add_auth(request)
        endpoint = urlsplit(url)
        cases.append(
            {
                "binding": {
                    "url": url,
                    "body_sha256": sha256(body.encode()).hexdigest(),
                    "authority": {
                        "kind": "sigv4",
                        "access_key": credentials.access_key,
                        "secret_key": credentials.secret_key,
                        "session_token": token,
                        "region": region,
                        "service": "bedrock",
                        "host": endpoint.netloc,
                        "path": quote(endpoint.path, safe="/~"),
                        "query": "",
                        "accept": accept,
                    },
                },
                "input": {"url": url, "body": body},
                "now": now.isoformat(),
                "expected": {
                    "headers": {str(k).lower(): str(v) for k, v in request.headers.items()}
                },
            }
        )
    return cases


def carrier_cases() -> list[JsonObject]:
    """Freeze decrypted claims for semantic arguments including decimal edge cases."""
    rng = random.Random(90571)
    values = [
        0.0,
        -0.0,
        1e-7,
        1e-4,
        1e16,
        1e18,
        9007199254740991.0,
        9007199254740992.0,
        0.00001,
        -1e50,
        123456789012345678901234567890,
        *[rng.uniform(-1, 1) * 10 ** rng.randrange(-25, 24) for _ in range(256)],
    ]
    result: list[JsonObject] = []
    authority = _authority()
    claims = {
        field.name: getattr(authority, field.name)
        for field in fields(authority)
        if field.name not in {"aead_key", "scheme"}
    }
    claims["schema_version"] = 3
    for index, value in enumerate(values):
        calls = _tool_calls(
            "call-one", raw_arguments=json.dumps({"z": value, "a": ["世界", True, None]})
        )
        assistant = [None, "\n ", "Visible text 世界"][index % 3]
        carrier = seal_reasoning_content(
            authority,
            issuing_request_id="request",
            issuing_route_depth=0,
            issuing_history_sha256="e" * 64,
            assistant_content=assistant,
            tool_calls=calls,
            content="reasoning 世界",
        )
        # Authenticate using the established reader, including the visible-turn binding.
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: PLC0415

        public, encoded = carrier.rsplit(":", 1)
        envelope = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        expected = json.loads(
            AESGCM(authority.aead_key).decrypt(envelope[:12], envelope[12:], public.encode())
        )
        result.append(
            {
                "authority": {
                    "claims": claims,
                    "key": base64.b64encode(authority.aead_key).decode(),
                    "prefix": authority.scheme.prefix,
                    "history_sha256": "e" * 64,
                },
                "input": {
                    "route_depth": 0,
                    "route_sha256": authority.reasoning_route_sha256,
                    "assistant_content": assistant,
                    "content": "reasoning 世界",
                    "tool_calls": [
                        {"call_id": c.call_id, "name": c.name, "raw_arguments": c.raw_arguments}
                        for c in calls
                    ],
                },
                "expected": expected,
                "python_carrier": carrier,
            }
        )
    return result


if __name__ == "__main__":
    directory = Path(__file__).with_name("generated")
    for name, cases in (("signing", signing_cases()), ("carriers", carrier_cases())):
        directory.joinpath(f"{name}.jsonl").write_text(
            "".join(json.dumps(case, separators=(",", ":")) + "\n" for case in cases)
        )
