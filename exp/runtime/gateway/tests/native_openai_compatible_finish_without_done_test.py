"""A finish_reason chunk followed by EOF (no ``[DONE]``) settles through the served gateway."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.tests.launch_test import _provider_frame, _ServedGateway, _unused_port


@pytest.mark.parametrize("finish", ["content_filter", "stop"])
def test_finish_reason_without_done_sentinel_is_a_complete_ending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finish: str
) -> None:
    """Azure Foundry DeepSeek lanes close a filtered stream after ``finish_reason``, no ``[DONE]``.

    Live 2026-09-15 (deepseek-v4-flash on azure_openai): the provider streamed
    content, sent ``finish_reason: "content_filter"``, and closed the connection
    without the sentinel; the gateway filed ~900 such turns a day as
    "provider stream ended without a terminal event" 502s. The finish reason is
    the provider's declared ending, so a filtered turn now answers as the
    refusal it named and a ``stop`` ending completes like a ``[DONE]`` one.
    """

    class Provider(BaseHTTPRequestHandler):
        """Serve synthetic chunks and close without the ``[DONE]`` sentinel."""

        def do_POST(self) -> None:  # noqa: N802
            """Return text, the finish chunk, trailing usage, then EOF."""
            self.rfile.read(int(self.headers["content-length"]))
            frames = (
                _provider_frame({"choices": [], "prompt_filter_results": []})
                + _provider_frame({"choices": [{"index": 0, "delta": {"content": "OK"}}]})
                + _provider_frame({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
                + _provider_frame(
                    {"choices": [], "usage": {"prompt_tokens": 13, "completion_tokens": 1}}
                )
            )
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(frames)))
            self.end_headers()
            self.wfile.write(frames)

        def log_message(self, format: str, *args: object) -> None:
            """Keep synthetic HTTP traffic out of test logs."""
            del format, args

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    _manager, raw_key = _configured_gateway(
        tmp_path, base_url=f"http://127.0.0.1:{provider.server_port}/v1"
    )
    gateway = _ServedGateway(tmp_path, _unused_port())
    try:
        gateway.start()
        base_url = f"http://127.0.0.1:{gateway.port}/v1"
        with OpenAI(api_key=raw_key, base_url=base_url, max_retries=0) as client:
            # Both endings answer exactly as their `[DONE]`-terminated twins do:
            # the content already streamed is delivered and the turn completes
            # (a filtered turn after visible content is the caller's answer,
            # not a 502; the ledger still records the provider's refusal).
            chat = client.chat.completions.create(
                model="coding", messages=[{"role": "user", "content": "Reply OK"}]
            )
            assert chat.choices[0].message.content == "OK"
            assert chat.choices[0].finish_reason == "stop"
            assert chat.usage is not None
            assert (chat.usage.prompt_tokens, chat.usage.completion_tokens) == (13, 1)
        # The streamed shape: the filtered turn ends with the refusal delta and a
        # finish chunk, never the mid-stream all_routes_failed error frame.
        with httpx.stream(
            "POST",
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "coding",
                "stream": True,
                "messages": [{"role": "user", "content": "Reply OK"}],
            },
            timeout=30.0,
        ) as response:
            assert response.status_code == 200
            frames = [
                json.loads(line[len("data: ") :])
                for line in response.iter_lines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
        assert all("error" not in frame for frame in frames), frames
        finishes = [
            choice.get("finish_reason")
            for frame in frames
            for choice in frame.get("choices", [])
            if choice.get("finish_reason")
        ]
        assert finishes == ["stop"], frames
        refusal_deltas = [
            choice["delta"]["refusal"]
            for frame in frames
            for choice in frame.get("choices", [])
            if "refusal" in choice.get("delta", {})
        ]
        assert (refusal_deltas == [""]) is (finish == "content_filter"), frames
    finally:
        gateway.stop()
        provider.shutdown()
        provider.server_close()
        thread.join(timeout=5)
