"""Smoke-test driver: capture a trace corpus from a locally routed wmo endpoint.

Part of the local-model smoke tests (C): sends real chat traffic through a served
endpoint whose static policy pins a local Ollama model, reads back each request's
metering row from the endpoint's own request log, and writes one OTel-GenAI trace
per exchange whose `wmo.attribution` records the local model identity and its
zero-cost accounting. The output feeds `wmo build --file` for smoke test (D).

Usage:
    uv run python .agents/scripts/capture_local_endpoint_traces.py \
        --endpoint http://localhost:8765/v1 --model tau-jt-toy \
        --log /tmp/wmo-local-smoke/serving/requests.jsonl \
        --out /tmp/wmo-local-smoke/captured.otel.jsonl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import httpx

from wmo.core.types import Action, ActionKind, Observation, Step, StepAttribution, Trace
from wmo.ingest.otel_writer import write_traces_jsonl
from wmo.serving.chat import RequestLogRecord

logger = logging.getLogger(__name__)

# Short airline-support-flavored tasks (the toy model's domain), tool-free so a
# 4B model answers in one turn and the corpus stays a plumbing fixture.
TASKS = [
    "A customer asks how early they can check in for a domestic flight.",
    "A customer wants to know the checked-bag weight limit in economy.",
    "A customer asks whether they can change a basic-economy ticket.",
    "A customer asks how to add a lap infant to an existing reservation.",
    "A customer wants the refund policy for a cancelled award ticket.",
    "A customer asks whether their small dog can fly in the cabin.",
    "A customer asks how many miles a one-way upgrade to business costs.",
    "A customer wants to know if seat selection is free at check-in.",
    "A customer asks what happens if they miss a connecting flight.",
    "A customer asks how to request a wheelchair at the departure gate.",
    "A customer wants to know when online check-in closes before departure.",
    "A customer asks whether travel insurance covers weather cancellations.",
]


def _one_exchange(client: httpx.Client, endpoint: str, model: str, task: str) -> str:
    response = client.post(
        f"{endpoint}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": task}],
            "max_tokens": 2000,
        },
        timeout=180.0,
    )
    response.raise_for_status()
    body = response.json()
    return body["choices"][0]["message"]["content"]


def _last_log_row(log_path: Path) -> RequestLogRecord:
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return RequestLogRecord.model_validate_json(lines[-1])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="Served /v1 base URL")
    parser.add_argument("--model", required=True, help="Endpoint name (the served world model)")
    parser.add_argument("--log", required=True, help="The endpoint's requests.jsonl")
    parser.add_argument("--out", required=True, help="OTel-GenAI JSONL to write")
    args = parser.parse_args()

    log_path = Path(args.log)
    traces: list[Trace] = []
    with httpx.Client() as client:
        for index, task in enumerate(TASKS):
            reply = _one_exchange(client, args.endpoint, args.model, task)
            row = _last_log_row(log_path)
            # The whole point of (C): the corpus must record WHICH model served
            # and that it billed nothing, straight from the endpoint's own log.
            attribution = StepAttribution(
                model=row.provider_model,
                provider="openai",
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                cost_usd=row.cost_usd,
                latency_ms=row.latency_ms,
                provenance="local-endpoint-smoke-v1",
            )
            traces.append(
                Trace(
                    trace_id=f"local-smoke-{index:04d}",
                    source="local-endpoint-smoke",
                    metadata={"endpoint": args.model, "routed_pool_entry": row.model},
                    steps=[
                        Step(
                            action=Action(kind=ActionKind.MESSAGE, content=reply),
                            observation=Observation(content=""),
                            task=task,
                            attribution=attribution,
                        )
                    ],
                )
            )
            logger.info(
                "%02d/%d routed=%s cost=$%.6f out_tokens=%d",
                index + 1,
                len(TASKS),
                row.model,
                row.cost_usd,
                row.output_tokens,
            )
    written = write_traces_jsonl(traces, Path(args.out))
    logger.info("wrote %d spans -> %s", written, args.out)


if __name__ == "__main__":
    main()
