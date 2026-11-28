"""Deterministic user-only conversation packing with a byte-based token heuristic."""

from __future__ import annotations

import json
from dataclasses import dataclass

from exp.runtime.gateway.contracts import GatewayMessage, GatewayRequest
from exp.runtime.gateway.guardrails.classification_window import message_text

_JSON_LINE_ESCAPES = str.maketrans({"\u0085": r"\u0085", "\u2028": r"\u2028", "\u2029": r"\u2029"})


@dataclass(frozen=True)
class UserConversationWindow:
    """Pack recent user turns without retaining content between requests.

    Attributes:
        max_tokens: Estimated state token budget after template and output reserve.
        bytes_per_token: Heuristic UTF-8 bytes per token; no tokenizer runs here.
        max_bytes: Available UTF-8 JSON state bytes after the remaining request fields.
        identity: Immutable heuristic, template and budget identity for verdict caches.
    """

    max_tokens: int
    max_bytes: int
    identity: str
    bytes_per_token: int = 3

    def __post_init__(self) -> None:
        """Refuse a budget that cannot represent the mandatory truncated current turn."""
        if (
            self.max_tokens < 1
            or self.max_bytes < 1
            or self.bytes_per_token < 1
            or not self.identity
            or not self._fits(("\n[...]\n",))
        ):
            raise ValueError("classifier budget cannot hold a user turn and its framing")

    def _fits(self, turns: tuple[str, ...]) -> bool:
        """Apply the heuristic state budget and exact serialized request-byte cap."""
        state = render_user_turns(turns)
        return (
            len(state.encode("utf-8")) <= self.max_tokens * self.bytes_per_token
            and len(json.dumps(state, ensure_ascii=False).encode("utf-8")) <= self.max_bytes
        )

    def project_input(self, request: GatewayRequest) -> GatewayRequest:
        """Keep the latest user turn and the newest preceding user text that fits.

        Native agent messages and function outputs are excluded even when a
        protocol adapter carries them with a user role. An oversized current
        turn keeps its head and tail, marked explicitly. An older boundary turn
        keeps its tail. UTF-8 size estimates token use; no tokenization occurs
        during packing or truncation. The model may tokenize unusual text above
        the estimate. No conversation lookup or durable write occurs here.
        """
        turns = tuple(
            message_text(message)
            for message in request.messages
            if message.role == "user" and message.provider_native_item is None
        )
        if turns:
            # Blank padding contains no classifiable request. Keep the current
            # boundary even when it is blank, without spending history on it.
            turns = tuple(text for text in turns[:-1] if text.strip()) + (
                turns[-1] if turns[-1].strip() else "",
            )
        # Most conversations fit. Check once instead of repeatedly serializing
        # each growing suffix, which would be quadratic in the number of turns.
        if turns and self._fits(turns):
            return self._request(request, turns)
        # Find the largest whole-turn suffix in logarithmically many byte
        # checks. This bounds preparation for thousands of short old turns.
        low, high = 0, len(turns) - 1
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

            def excerpt(size: int) -> str:
                """Mark a deterministic head/tail or tail-only character cut."""
                if current:
                    tail = size // 2
                    return text[: size - tail] + "\n[...]\n" + (text[-tail:] if tail else "")
                return "[...]\n" + (text[-size:] if size else "")

            low, high = 0, len(text)
            best = kept
            while low <= high:
                size = (low + high) // 2
                candidate = (excerpt(size), *kept)
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
    """Place all selected user turns inside the classifier's request boundary.

    Existing classifier instructions assess ``[latest user]``. Its input is the
    whole user-only conversation, not just the follow-up. JSON array entries
    preserve chronological turn boundaries without introducing new role labels.
    """
    payload = json.dumps({"user_turns": turns or ("",)}, ensure_ascii=False)
    return "[latest user]\n" + payload.translate(_JSON_LINE_ESCAPES)
