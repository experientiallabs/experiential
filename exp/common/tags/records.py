"""Tag results for one captured exchange, in the exact JSON shape tagging runs return.

Scores are probabilities in ``[0, 1]`` ordered like ``TagBank.for_scope`` for their scope. A turn
can span several windows, so every window keeps its own scores and ``summarize`` reduces them to
the per-tag maximum (true in any window) and minimum (true in every window) that searches read.
Message ranges are zero-based and half-open over the exchange's request messages followed by
its response messages.
"""

from __future__ import annotations

import math

from pydantic import AwareDatetime, Field, field_validator, model_validator

from exp.common.core.artifacts import ArtifactId, ContractModel, Sha256


def _check_probabilities(values: tuple[float, ...]) -> tuple[float, ...]:
    """Require every value to be a finite probability.

    Args:
        values: Candidate scores.

    Returns:
        The unchanged scores.

    Raises:
        ValueError: A score is not finite or lies outside ``[0, 1]``.
    """
    for value in values:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"tag scores must be probabilities in [0, 1], got {value!r}")
    return values


class WindowScores(ContractModel):
    """One window's tag probabilities.

    Attributes:
        window_index: Zero-based position of the window within its scope.
        start_message: First message the window covers.
        end_message: One past the last message the window covers; greater than
            ``start_message``.
        scores: One probability per tag of the scope, in bank scope order.
    """

    window_index: int = Field(ge=0)
    start_message: int = Field(ge=0)
    end_message: int = Field(ge=1)
    scores: tuple[float, ...]

    @field_validator("scores")
    @classmethod
    def _probabilities(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        """Reject scores outside ``[0, 1]``.

        Args:
            value: Candidate scores.

        Returns:
            The unchanged scores.
        """
        return _check_probabilities(value)

    @model_validator(mode="after")
    def _ordered_range(self) -> WindowScores:
        """Require a non-empty message range.

        Returns:
            The validated window.

        Raises:
            ValueError: ``end_message`` does not exceed ``start_message``.
        """
        if self.end_message <= self.start_message:
            raise ValueError("a window must cover at least one message (end_message > start)")
        return self


class TagFacts(ContractModel):
    """Exchange facts read directly from the traffic, with no model call.

    Attributes:
        tool_calls: Number of tool calls in the response.
        turns: Number of user messages in the request, the current one included.
        truncated: Whether the response stopped because it reached its length limit.
        has_images: Whether any request message carries an image part.
        structured_output: Whether the request asked for JSON or schema-constrained output.
    """

    tool_calls: int = Field(ge=0)
    turns: int = Field(ge=0)
    truncated: bool
    has_images: bool
    structured_output: bool


class TagSummary(ContractModel):
    """Per-tag reduction of an exchange's turn windows, the form searches read.

    Attributes:
        bank_id: Bank that produced the scores.
        bank_version: Version of that bank.
        turn_max: Highest score per turn tag across windows.
        turn_min: Lowest score per turn tag across windows.
        turn_windows: Number of turn windows reduced.
    """

    bank_id: ArtifactId
    bank_version: int = Field(ge=1)
    turn_max: tuple[float, ...]
    turn_min: tuple[float, ...]
    turn_windows: int = Field(ge=1)

    @field_validator("turn_max", "turn_min")
    @classmethod
    def _probabilities(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        """Reject scores outside ``[0, 1]``.

        Args:
            value: Candidate scores.

        Returns:
            The unchanged scores.
        """
        return _check_probabilities(value)

    @model_validator(mode="after")
    def _aligned(self) -> TagSummary:
        """Require aligned arrays with each minimum at most its maximum.

        Returns:
            The validated summary.

        Raises:
            ValueError: The arrays differ in length or a minimum exceeds its maximum.
        """
        if len(self.turn_max) != len(self.turn_min):
            raise ValueError("turn_max and turn_min must have one score per turn tag")
        if any(low > high for low, high in zip(self.turn_min, self.turn_max, strict=True)):
            raise ValueError("a turn_min score cannot exceed its turn_max score")
        return self


class ExchangeTags(ContractModel):
    """Every tag result for one exchange, the wire format a tagging run returns.

    Attributes:
        exchange_id: Identifier of the tagged exchange, from 1 through 256 characters.
        conversation_key: Key grouping the exchange with the rest of its conversation.
        bank_id: Bank that produced the scores.
        bank_version: Version of that bank.
        bank_digest: ``TagBank.digest`` of the exact bank used.
        model: Served decision model that answered the questions.
        tagged_at: When tagging finished, with a timezone.
        conversation: Opening window scores, or None when the conversation was already tagged.
        turn: Turn window scores in window order; at least one window.
        facts: Facts read directly from the exchange.
    """

    exchange_id: str = Field(min_length=1, max_length=256)
    conversation_key: str = Field(min_length=1, max_length=256)
    bank_id: ArtifactId
    bank_version: int = Field(ge=1)
    bank_digest: Sha256
    model: str = Field(min_length=1, max_length=256)
    tagged_at: AwareDatetime
    conversation: WindowScores | None
    turn: tuple[WindowScores, ...] = Field(min_length=1)
    facts: TagFacts

    @model_validator(mode="after")
    def _consistent_windows(self) -> ExchangeTags:
        """Require ordered turn windows with one score per turn tag each.

        Returns:
            The validated record.

        Raises:
            ValueError: Window indexes are not ``0..n-1``, the opening is not window 0, or turn
                windows disagree on their score count.
        """
        if tuple(window.window_index for window in self.turn) != tuple(range(len(self.turn))):
            raise ValueError("turn windows must be numbered 0, 1, 2, ... in order")
        if self.conversation is not None and self.conversation.window_index != 0:
            raise ValueError("the conversation opening is a single window with index 0")
        if len({len(window.scores) for window in self.turn}) != 1:
            raise ValueError("every turn window must have one score per turn tag")
        return self


def summarize(tags: ExchangeTags) -> TagSummary:
    """Reduce an exchange's turn windows to per-tag maximum and minimum scores.

    A bare tag in a rule reads the maximum (true in any window); ``all(tag)`` reads the
    minimum (true in every window).

    Args:
        tags: The exchange's tag results.

    Returns:
        The summary for the exchange's turn tags.
    """
    columns = tuple(zip(*(window.scores for window in tags.turn), strict=True))
    return TagSummary(
        bank_id=tags.bank_id,
        bank_version=tags.bank_version,
        turn_max=tuple(max(column) for column in columns),
        turn_min=tuple(min(column) for column in columns),
        turn_windows=len(tags.turn),
    )


__all__ = ["ExchangeTags", "TagFacts", "TagSummary", "WindowScores", "summarize"]
