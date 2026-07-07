"""Minimal OpenAI-compatible /v1/chat/completions shim over a wmh provider (Bedrock).

Lets Qwen-AgentWorld's `eval.py judge` stage run unmodified against a Bedrock Anthropic
model via `--judge-base-url http://127.0.0.1:8765/v1 --judge-api-key EMPTY`.

STAND-IN JUDGE ONLY: AgentWorldBench's pinned judge is OpenAI `gpt-5.2-2025-12-11`.
Scores produced through this shim are NOT comparable to the paper's table (D12: never
compare across judges) — they only prove the pipeline end-to-end until a live
OPENAI_API_KEY is available.

Usage:
    uv run python .agents/scripts/agentworldbench/judge_shim.py --model us.anthropic.claude-opus-4-8
    curl http://127.0.0.1:8765/usage   # metered judge tokens + cost so far
"""

from __future__ import annotations

import argparse
import time
import uuid

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from wmh.providers import get_provider
from wmh.providers.base import Message, ProviderConfig, ProviderKind
from wmh.tracking.metered import MeteredProvider
from wmh.tracking.tracker import Phase, RunTracker

MAX_OUTPUT_TOKENS = 8192  # judge emits reasoning + one JSON block; clamp their 32768 default


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = MAX_OUTPUT_TOKENS
    temperature: float = 0.0


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage


class UsageSummary(BaseModel):
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    judge_model: str = Field(description="the actual backing model, not the requested alias")


def create_app(model: str, region: str) -> FastAPI:
    tracker = RunTracker(run_id="awb-judge-shim", kind="eval")
    tracker.start()
    provider = MeteredProvider(
        get_provider(ProviderConfig(kind=ProviderKind.BEDROCK, model=model, region=region)),
        tracker,
        base_phase=Phase.JUDGE,
    )
    app = FastAPI(title="wmh AgentWorldBench judge shim")
    calls = {"n": 0}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest) -> ChatResponse:
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        turns = [
            Message(role="assistant" if m.role == "assistant" else "user", content=m.content)
            for m in request.messages
            if m.role != "system"
        ]
        completion = provider.complete(
            system,
            turns,
            temperature=request.temperature,
            max_tokens=min(request.max_tokens, MAX_OUTPUT_TOKENS),
        )
        calls["n"] += 1
        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model,
            choices=[ChatChoice(message=ChatMessage(role="assistant", content=completion.text))],
            usage=ChatUsage(
                prompt_tokens=completion.usage.input_tokens,
                completion_tokens=completion.usage.output_tokens,
                total_tokens=completion.usage.input_tokens + completion.usage.output_tokens,
            ),
        )

    @app.get("/usage")
    def usage() -> UsageSummary:
        total = tracker.record_summary().total
        return UsageSummary(
            calls=calls["n"],
            input_tokens=total.input_tokens,
            output_tokens=total.output_tokens,
            cost_usd=total.cost_usd,
            judge_model=model,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="us.anthropic.claude-opus-4-8")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(create_app(args.model, args.region), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
