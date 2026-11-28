"""Deterministic user-only conversation packing under an exact classifier budget."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from exp.runtime.gateway.contracts import GatewayMessage, GatewayRequest
from exp.runtime.gateway.guardrails.classification_window import message_text


@dataclass(frozen=True)
class UserConversationWindow:
    """Pack recent user turns without retaining content between requests.

    Attributes:
        count_tokens: Exact state encoder used by the classifier, without framing.
        max_tokens: Available state tokens after template, questions and output reserve.
        max_bytes: Available UTF-8 JSON state bytes after the remaining request fields.
        identity: Immutable tokenizer, template and budget identity for verdict caches.
    """

    count_tokens: Callable[[str], int]
    max_tokens: int
    max_bytes: int
    identity: str

    def __post_init__(self) -> None:
        """Refuse a budget that cannot even represent an empty user turn."""
        if not self.identity or not self._fits(("",)):
            raise ValueError("classifier budget cannot hold a user turn and its framing")

    def _fits(self, turns: tuple[str, ...]) -> bool:
        """Check the exact rendered state against both independent wire limits."""
        state = render_user_turns(turns)
        return (
            len(json.dumps(state, ensure_ascii=False).encode("utf-8")) <= self.max_bytes
            and self.count_tokens(state) <= self.max_tokens
        )

    def project_input(self, request: GatewayRequest) -> GatewayRequest:
        """Keep the latest user turn and the newest preceding user text that fits.

        Native agent messages and function outputs are excluded even when a
        protocol adapter carries them with a user role. An oversized current
        turn keeps its head and tail, marked explicitly. An older boundary turn
        keeps its tail. Every chosen candidate is checked with the real encoder;
        binary search only chooses deterministic character cuts, never estimates
        a token count. No conversation lookup or durable write occurs here.
        """
        turns = tuple(
            message_text(message)
            for message in request.messages
            if message.role == "user" and message.provider_native_item is None
        )
        # Most conversations fit. Count them once instead of repeatedly encoding
        # each growing suffix, which would be quadratic in the number of turns.
        if turns and self._fits(turns):
            return self._request(request, turns)
        # Find the largest whole-turn suffix in logarithmically many exact
        # encodes. This bounds preparation for thousands of short old turns.
        low, high = 0, len(turns)
        count = 0
        while low <= high:
            middle = (low + high) // 2
            candidate = turns[-middle:] if middle else ()
            if self._fits(candidate):
                count = middle
                low = middle + 1
            else:
                high = middle - 1
        kept = turns[-count:] if count else ()
        if count < len(turns):
            text = turns[-count - 1]
            current = not kept
            low, high = 0, len(text)
            best = kept
            while low <= high:
                size = (low + high) // 2
                if current:
                    tail = size // 2
                    excerpt = text[: size - tail] + "\n[...]\n" + (text[-tail:] if tail else "")
                else:
                    excerpt = "[...]\n" + (text[-size:] if size else "")
                candidate = (excerpt, *kept)
                if self._fits(candidate):
                    best = candidate
                    low = size + 1
                else:
                    high = size - 1
            kept = best
            if current and not kept:
                raise ValueError("classifier budget cannot hold a marked current user turn")
        return self._request(request, kept)

    @staticmethod
    def _request(request: GatewayRequest, kept: tuple[str, ...]) -> GatewayRequest:
        """Carry only selected plain user text into request-local coverage accounting."""
        return GatewayRequest(
            surface=request.surface,
            messages=tuple(GatewayMessage(role="user", content=text) for text in kept)
            or (GatewayMessage(role="user", content=""),),
        )

    def state(self, request: GatewayRequest) -> str:
        """Render precisely the user content selected by this window."""
        projected = self.project_input(request)
        turns = tuple(message.content or "" for message in projected.messages)
        return render_user_turns(turns) if any(turns) else ""


def render_user_turns(turns: tuple[str, ...]) -> str:
    """Preserve chronology and unambiguous turn boundaries inside user-supplied text."""
    return json.dumps(
        {"previous_user_requests": turns[:-1], "current_user_request": turns[-1] if turns else ""},
        ensure_ascii=False,
        separators=(",", ":"),
    )
