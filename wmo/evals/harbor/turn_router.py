"""Per-turn model routing for matched Harbor evaluations.

This is an evaluation bridge, not a new fitting algorithm. It reuses a persisted WMO profile
policy and chooses a provider for every structured agent request. ``task`` mode routes from the
initial task text and is therefore the per-task comparison. ``transcript`` mode routes from the
growing conversation and exercises the cache-aware per-turn switch gate.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from llm_waterfall import ChatRequest, ChatResponse

from wmo.evals.harbor.agent import WmoHarborAgent
from wmo.optimize.policy import RoutingDecision, RoutingPolicy, select_model
from wmo.providers.base import (
    Completion,
    Message,
    Provider,
    ProviderConfig,
    VerifyResult,
)
from wmo.providers.registry import get_provider
from wmo.providers.retry import wrap_provider_with_retries

RouterMode = Literal["task", "transcript"]


def _text(value: object) -> str:
    """Flatten provider-neutral message content into routing text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_text(item) for item in value)
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
        content = value.get("content")
        if content is not None:
            return _text(content)
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return "" if value is None else str(value)


def _message_text(message: object) -> str:
    """Include content and tool-call arguments in the growing transcript length."""
    parts = [_text(getattr(message, "content", ""))]
    for call in getattr(message, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        parts.append(str(getattr(function, "name", "")))
        parts.append(str(getattr(function, "arguments", "")))
    return " ".join(part for part in parts if part)


class TurnRoutingProvider:
    """Route each structured request through providers named by a profile policy."""

    def __init__(
        self,
        policy: RoutingPolicy,
        *,
        mode: RouterMode,
        log_path: Path,
        config: ProviderConfig,
    ) -> None:
        if policy.kind != "profile":
            raise ValueError(f"turn evaluation expects a profile policy, got {policy.kind!r}")
        if mode not in ("task", "transcript"):
            raise ValueError(f"unknown turn routing mode {mode!r}")
        self.config = config
        self._policy = policy
        self._mode = mode
        self._log_path = log_path
        self._log_lock = threading.Lock()
        self._incumbent: str | None = None
        self._calls = 0
        names = set(policy.profile_models) | {policy.default_model}
        self._providers: dict[str, Provider] = {}
        for entry in policy.pool:
            if entry.name in names:
                self._providers[entry.name] = wrap_provider_with_retries(
                    get_provider(entry.provider_config())
                )

    def _route_text(self, request: ChatRequest) -> tuple[str, int]:
        messages = list(request.messages)
        task_text = next(
            (_message_text(message) for message in messages if message.role == "user"),
            "",
        )
        transcript = "\n".join(
            f"{message.role}: {_message_text(message)}" for message in messages
        )
        if self._mode == "task":
            return task_text, len(task_text)
        return transcript, len(transcript)

    def _decision(self, request: ChatRequest) -> tuple[RoutingDecision, str, int, str | None]:
        route_text, route_chars = self._route_text(request)
        previous = self._incumbent
        decision = select_model(
            self._policy,
            route_text,
            incumbent=previous,
            conversation_chars=route_chars if previous is not None else 0,
        )
        return decision, route_text, route_chars, previous

    def _record(
        self,
        *,
        decision: RoutingDecision,
        route_text: str,
        route_chars: int,
        previous: str | None,
        response: ChatResponse,
    ) -> None:
        usage = response.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        entry = next(item for item in self._policy.pool if item.name == decision.model)
        uncached_cost = (
            prompt_tokens * entry.price().input_per_mtok
            + output_tokens * entry.price().output_per_mtok
        ) / 1_000_000
        evidence = decision.evidence
        row: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(),
            "turn": self._calls,
            "mode": self._mode,
            "model": decision.model,
            "incumbent_before": previous,
            "switched": previous is not None and previous != decision.model,
            "route_text_chars": len(route_text),
            "conversation_chars": route_chars,
            "reason": decision.reason,
            "gate": evidence.gate if evidence else None,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "uncached_cost_usd": uncached_cost,
        }
        with self._log_lock:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        decision, route_text, route_chars, previous = self._decision(request)
        provider = self._providers.get(decision.model)
        if provider is None:
            raise RuntimeError(f"profile selected unconfigured model {decision.model!r}")
        response = provider.complete_chat(request)  # type: ignore[attr-defined]
        self._calls += 1
        self._incumbent = decision.model
        self._record(
            decision=decision,
            route_text=route_text,
            route_chars=route_chars,
            previous=previous,
            response=response,
        )
        return response

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        """Keep the Provider protocol complete for non-pi callers."""
        del system, temperature
        if not self._providers:
            raise RuntimeError("turn router has no configured providers")
        model = self._incumbent or self._policy.default_model
        return self._providers[model].complete("", messages, max_tokens=max_tokens)  # type: ignore[attr-defined]

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("the DeepSWE profile router does not use embeddings")

    def verify(self) -> VerifyResult:
        return VerifyResult(ok=True, kind=self.config.kind, model=self.config.model)

    def context_window(self) -> int:
        return 200_000


class TurnRoutingHarborAgent(WmoHarborAgent):
    """Harbor agent bridge that swaps providers between agent turns."""

    def __init__(
        self,
        logs_dir: Path,
        *,
        router_policy_path: str,
        router_mode: RouterMode = "transcript",
        **kwargs: object,
    ) -> None:
        self._router_policy_path = Path(router_policy_path)
        self._router_mode = router_mode
        super().__init__(logs_dir=logs_dir, **kwargs)

    def _build_provider(self, config: ProviderConfig) -> Provider:
        policy = RoutingPolicy.load(self._router_policy_path)
        return TurnRoutingProvider(
            policy,
            mode=self._router_mode,
            log_path=self.logs_dir / "turn-router.jsonl",
            config=config,
        )
