"""Tests for router judgment contracts."""

import pytest
from pydantic import ValidationError

from wmo.common.judging import PromptDefinition
from wmo.common.judging.lm import PORTABLE_RATIONALE_JSON_SCHEMA
from wmo.common.judging.lm_test import _axis_schema
from wmo.common.models import PricingSource
from wmo.optimize.router.judging.contracts import (
    JudgeCalibrationBudget,
    JudgePromptTemplate,
    judge_feedback_schema,
)


def test_stored_budget_without_pricing_source_stays_unknown() -> None:
    """Older audit payloads remain readable without inventing a pricing source."""
    budget = JudgeCalibrationBudget.model_validate(
        {
            "input_usd_per_million_tokens": 1.0,
            "output_usd_per_million_tokens": 2.0,
            "maximum_input_tokens_per_call": 4096,
            "maximum_attempts_per_call": 3,
            "call_count": 1,
            "estimated_cost_usd": 0.01,
            "maximum_cost_usd": 1.0,
        }
    )

    assert budget.pricing_source is PricingSource.UNKNOWN


def test_scalar_feedback_schema_matches_the_portable_rationale_contract() -> None:
    """Scalar setup schemas keep a required score and an optional nullable rationale."""
    schema = judge_feedback_schema("scalar")
    properties, required = _axis_schema(schema)

    assert properties["rationale"] == PORTABLE_RATIONALE_JSON_SCHEMA
    assert required == ["dimension_id", "raw_score"]
    assert "evidence_span_ids" not in properties


def test_template_version_one_requires_re_setup() -> None:
    """Citation-era setups fail closed with an actionable recreate message."""
    template = JudgePromptTemplate(
        prompt=PromptDefinition.from_text("custom-judge-v1", "Follow the saved contract exactly."),
        variable_mapping={"rubric": "RULES_CUSTOM", "rollout": "TRACE_CUSTOM"},
        response_schema=judge_feedback_schema("scalar"),
    )
    payload = template.model_dump(mode="json")
    payload["template_version"] = "1"

    with pytest.raises(ValidationError, match="retired citation contract"):
        JudgePromptTemplate.model_validate(payload)
