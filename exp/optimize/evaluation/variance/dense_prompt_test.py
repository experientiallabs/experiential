"""The dense judge prompt variants and profile binding."""

from __future__ import annotations

import pytest

from exp.optimize.evaluation.variance.dense_judge_test import MODEL
from exp.optimize.evaluation.variance.dense_prompt import (
    DEFAULT_DENSE_VARIANTS,
    DENSE_PROMPT_VARIANTS,
    dense_model_calls,
    dense_profile,
    dense_prompt_sha256,
)


def test_variants_differ_and_each_states_the_observation_rule() -> None:
    """Three distinct variants share the rule that observing a change is not establishing it."""
    assert len(set(DENSE_PROMPT_VARIANTS.values())) == len(DEFAULT_DENSE_VARIANTS) == 3
    for text in DENSE_PROMPT_VARIANTS.values():
        assert "OBSERVING IS NOT ESTABLISHING" in text and "never scored" in text


def test_the_profile_pins_its_variants() -> None:
    """Choosing other variants is another profile; unknown variants are refused."""
    profile = dense_profile(model_alias="judge", model=MODEL, maximum_output_tokens=512)
    single = dense_profile(
        model_alias="judge",
        model=MODEL,
        maximum_output_tokens=512,
        variants=DEFAULT_DENSE_VARIANTS[:1],
    )
    assert profile.prompt_sha256 != single.prompt_sha256
    assert dense_model_calls(profile) == 3 * (profile.evidence_tool_budget + 3)
    with pytest.raises(ValueError, match="no dense prompt variants"):
        dense_prompt_sha256(("unknown",))
