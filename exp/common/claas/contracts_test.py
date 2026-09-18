"""Regression coverage for truthful sampled-token and capture-consent contracts."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from exp.common.claas import (
    CapturePolicy,
    ClaasScope,
    ExactTokenEvidence,
    Experience,
    ExperienceProvenance,
)


def _tokens() -> ExactTokenEvidence:
    """Build exact sampled-token evidence for one pinned test policy."""
    return ExactTokenEvidence(
        model_id="model",
        model_revision="base-revision",
        policy_revision="step-1",
        tokenizer_id="tokenizer",
        tokenizer_revision="revision-1",
        sampling_temperature=1.0,
        sampling_top_p=1.0,
        sampling_top_k=None,
        prompt_token_ids=(1, 2),
        response_token_ids=(3,),
        response_logprobs=(-0.2,),
    )


def test_capture_defaults_off_and_retention_is_bounded() -> None:
    """Content capture requires explicit consent and consistent positive bounds."""
    scope = ClaasScope(user_id="local-user", application_id="claims-agent")
    assert not CapturePolicy(scope=scope).enabled
    with pytest.raises(ValidationError):
        CapturePolicy(scope=scope, maximum_storage_bytes=1)


@pytest.mark.parametrize("probabilities", [(), (-0.1, -0.2), (float("nan"),), (0.1,)])
def test_invalid_sampled_probabilities_are_rejected(probabilities: tuple[float, ...]) -> None:
    """Malformed token evidence cannot masquerade as a valid policy rollout."""
    with pytest.raises(ValidationError):
        ExactTokenEvidence.model_validate(
            {**_tokens().model_dump(), "response_logprobs": probabilities}
        )


def test_sampled_tokens_require_matching_generation_provenance() -> None:
    """An arbitrary provider capture has no exact tokens and rejects misbound evidence."""
    experience = Experience(
        experience_id="experience-1",
        response_id="response-1",
        scope=ClaasScope(user_id="local-user", application_id="claims-agent"),
        protocol="chat_completions",
        captured_at=datetime.now(UTC),
        request={"messages": []},
        response={"choices": []},
        provenance=ExperienceProvenance(
            source_kind="traffic", source_id="request-1", model_id="model"
        ),
    )
    assert experience.exact_tokens is None
    with pytest.raises(ValidationError):
        Experience.model_validate({**experience.model_dump(), "exact_tokens": _tokens()})
