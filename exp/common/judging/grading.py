"""One-shot answer grading: one candidate answer against one scenario task.

The judge scores how well a single answer accomplishes a single scenario's
task on a ``[0, 1]`` rubric with a one-line rationale, judging task
completion only, never style, length, or authorship. Unlike the rubric
judges, it grades plain text rather than persisted rollout evidence, so an
eval loop can score arbitrary (scenario, answer) cells. The caller decides
what a failed grade means; the Experiential Labs platform, the first
consumer of this seam, treats it as an absent cell because an unscored
answer proves nothing.
"""

from __future__ import annotations

from pydantic import Field, ValidationError, field_validator

from exp.common.core.artifacts import ContractModel
from exp.common.models import (
    ModelClient,
    ModelMessage,
    ModelRequest,
    StructuredReplyError,
    structured_reply_json,
)

# Reasoning models spend hidden reasoning tokens (hundreds per call on
# current frontier models) from the same output budget as the visible JSON
# verdict, and an exhausted budget is rejected as a truncated reply. The
# default leaves room for both; a longer reply costs nothing unless produced.
DEFAULT_GRADING_OUTPUT_TOKENS = 2_000

GRADING_PROMPT = (
    "You are grading ONE candidate answer to ONE scenario drawn from this "
    "organization's real traffic.\n"
    "Score how well the answer accomplishes the scenario's task on a 0..1 "
    "rubric: 1.0 = fully accomplishes the task, correct and complete; "
    "0.5 = partially useful; 0.0 = wrong or useless.\n"
    "Judge task completion only: never the answer's style, its length, or "
    "which model might have written it.\n"
    "Respond with ONLY a JSON object, no prose and no code fences: "
    '{"score": 0.0, "rationale": "..."}. rationale is one line '
    "(at most 300 characters)."
)


class AnswerGradeError(ValueError):
    """The judge's reply broke the answer-grading output contract."""


class AnswerGrade(ContractModel):
    """One rubric grade for one (scenario, answer) pair.

    Args:
        score: Task-completion score in ``[0, 1]``.
        rationale: One line supporting the score.
    """

    score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=300)

    @field_validator("rationale")
    @classmethod
    def _require_visible_text(cls, value: str) -> str:
        """Trim edge whitespace and reject a blank-looking rationale."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must contain non-whitespace text")
        return stripped


class AnswerGradeJudge:
    """Grades one candidate answer against one scenario task on a [0, 1] rubric."""

    def __init__(
        self,
        client: ModelClient,
        *,
        maximum_output_tokens: int = DEFAULT_GRADING_OUTPUT_TOKENS,
    ) -> None:
        """Bind the judge to one configured model client.

        Args:
            client: Configured judge model client.
            maximum_output_tokens: Upper bound for each verdict reply.

        Raises:
            ValueError: ``maximum_output_tokens`` is not positive.
        """
        if maximum_output_tokens <= 0:
            raise ValueError("maximum_output_tokens must be positive")
        self._client = client
        self._maximum_output_tokens = maximum_output_tokens

    def grade(self, *, task: str, answer: str) -> AnswerGrade:
        """One model call grading one answer against one scenario task.

        Args:
            task: The scenario task text, sent verbatim.
            answer: The candidate answer text, sent verbatim.

        Returns:
            The parsed rubric grade.

        Raises:
            ValueError: ``task`` or ``answer`` is empty.
            AnswerGradeError: The reply was truncated at the token limit,
                carried no text, or broke the JSON contract.
        """
        if not task.strip():
            raise ValueError("answer grading needs non-empty scenario task text")
        if not answer.strip():
            raise ValueError("answer grading needs non-empty candidate answer text")
        response = self._client.complete(
            ModelRequest(
                messages=(
                    ModelMessage(role="system", content=GRADING_PROMPT),
                    ModelMessage(
                        role="user",
                        content=f"SCENARIO TASK:\n{task}\n\nCANDIDATE ANSWER:\n{answer}",
                    ),
                ),
                maximum_output_tokens=self._maximum_output_tokens,
            )
        )
        try:
            raw = structured_reply_json(response)
        except StructuredReplyError as error:
            raise AnswerGradeError(str(error)) from error
        try:
            return AnswerGrade.model_validate(raw)
        except ValidationError as error:
            msg = "the judge returned a grade outside the contracted shape; rerun the grading"
            raise AnswerGradeError(msg) from error
