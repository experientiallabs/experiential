"""Two-axis realism judging over scenario text: likelihood and feasibility.

Models conflate "unlikely" with "impossible". This judge scores one scenario
on two separate axes, likelihood and feasibility, each in ``[0, 1]`` with a
one-line rationale, and nothing here ever combines the axes: a rare-but-real
situation is low likelihood and high feasibility, and a scenario judged
near-infeasible despite being observed in real traffic contradicts reality
and deserves human review first. Unlike the rubric judges, this judge scores
plain scenario text rather than persisted rollout evidence. The Experiential
Labs platform is the first consumer of this seam.
"""

from __future__ import annotations

import json

from pydantic import Field, ValidationError

from exp.common.core.artifacts import ContractModel
from exp.common.models import (
    ModelClient,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    structured_json_text,
)

DEFAULT_REALISM_OUTPUT_TOKENS = 400

REALISM_PROMPT = (
    "You assess ONE scenario for an AI agent, on two SEPARATE axes:\n"
    "- likelihood: how probable is it that this situation occurs in this "
    "organization's real traffic? 0 = essentially never, 1 = routine.\n"
    "- feasibility: is the situation possible AT ALL for this agent and its "
    "users? 0 = actually impossible, 1 = clearly possible.\n"
    "These are different questions. A rare-but-real situation is LOW "
    "likelihood and HIGH feasibility; never call it infeasible because it "
    "is rare.\n"
    "Respond with ONLY a JSON object, no prose and no code fences: "
    '{"likelihood": 0.0, "feasibility": 0.0, "rationale": "..."}. '
    "rationale is one line (at most 300 characters)."
)


class RealismJudgmentError(ValueError):
    """The judge's reply broke the realism output contract."""


class RealismAssessment(ContractModel):
    """One two-axis realism verdict; the axes are never combined.

    Args:
        likelihood: How probable the scenario is in real traffic, in
            ``[0, 1]``.
        feasibility: Whether the scenario is possible at all, in ``[0, 1]``.
        rationale: One line supporting both scores.
    """

    likelihood: float = Field(ge=0.0, le=1.0)
    feasibility: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=300)


class RealismJudge:
    """Scores scenario text on separate likelihood and feasibility axes."""

    def __init__(
        self,
        client: ModelClient,
        *,
        maximum_output_tokens: int = DEFAULT_REALISM_OUTPUT_TOKENS,
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

    def assess(self, scenario: str) -> RealismAssessment:
        """One model call judging one scenario on both axes.

        Args:
            scenario: The scenario task text to judge, sent verbatim.

        Returns:
            The parsed two-axis assessment.

        Raises:
            ValueError: ``scenario`` is empty.
            RealismJudgmentError: The reply was truncated at the token limit,
                carried no text, or broke the JSON contract.
        """
        if not scenario:
            raise ValueError("realism assessment needs non-empty scenario text")
        response = self._client.complete(
            ModelRequest(
                messages=(
                    ModelMessage(role="system", content=REALISM_PROMPT),
                    ModelMessage(role="user", content=scenario),
                ),
                maximum_output_tokens=self._maximum_output_tokens,
            )
        )
        if response.finish_reason is ModelFinishReason.LENGTH:
            msg = "the judge stopped at its output-token limit; raise maximum_output_tokens"
            raise RealismJudgmentError(msg)
        content = response.output.content
        if content is None:
            msg = "the judge returned no text output; rerun the assessment"
            raise RealismJudgmentError(msg)
        try:
            raw = json.loads(structured_json_text(content))
        except json.JSONDecodeError as error:
            msg = "the judge returned non-JSON output; rerun the assessment"
            raise RealismJudgmentError(msg) from error
        try:
            return RealismAssessment.model_validate(raw)
        except ValidationError as error:
            msg = (
                "the judge returned an assessment outside the contracted shape; "
                "rerun the assessment"
            )
            raise RealismJudgmentError(msg) from error
