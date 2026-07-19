"""Provider-neutral reasoning effort values and backend capability validation."""

from __future__ import annotations

from typing import Literal, cast, get_args

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

REASONING_EFFORTS = cast("tuple[ReasoningEffort, ...]", get_args(ReasoningEffort))

_BEDROCK_ADAPTIVE_MODELS = frozenset(
    {
        "anthropic.claude-opus-4-7",
        "anthropic.claude-opus-4-6-v1",
        "anthropic.claude-sonnet-4-6",
    }
)
_BEDROCK_MAX_EFFORT_MODEL = "anthropic.claude-opus-4-6-v1"


def validate_backend_reasoning_effort(
    provider: str,
    model: str,
    effort: str | None,
) -> ReasoningEffort | None:
    """Validate one immutable backend reasoning setting before any request can run."""
    if effort is None:
        return None
    if effort not in REASONING_EFFORTS:
        expected = ", ".join(REASONING_EFFORTS)
        raise ValueError(f"unknown reasoning effort {effort!r}; expected one of: {expected}")
    validated = cast("ReasoningEffort", effort)
    if provider != "bedrock":
        raise ValueError(
            "llm-waterfall reasoning effort is only supported for Bedrock adaptive reasoning"
        )
    base_model = bedrock_base_model_id(model)
    if base_model not in _BEDROCK_ADAPTIVE_MODELS:
        raise ValueError(f"Bedrock model {model!r} does not support adaptive reasoning")
    if validated == "max" and base_model != _BEDROCK_MAX_EFFORT_MODEL:
        raise ValueError("reasoning effort 'max' is supported only by Claude Opus 4.6")
    if validated in ("none", "minimal", "xhigh"):
        raise ValueError(
            f"Bedrock adaptive reasoning does not support effort {validated!r}; "
            "choose low, medium, high, or max"
        )
    return validated


def bedrock_base_model_id(model: str) -> str:
    """Return the foundation-model id encoded by a Bedrock runtime identifier.

    Bedrock accepts a foundation-model id, a system inference-profile id such as
    ``au.anthropic...``, or the ARN form of either resource.  Geography prefixes are an AWS
    data value rather than a closed enum, so normalize the structural ``.<vendor>.`` boundary
    instead of maintaining a list that goes stale when AWS adds a geography.

    Application inference-profile ARNs can have arbitrary names and do not encode their backing
    model.  Those intentionally remain opaque so capability validation fails closed rather than
    guessing which model an account-local alias routes to.
    """
    resource_id = model.rsplit("/", maxsplit=1)[-1] if model.startswith("arn:") else model
    vendor_marker = ".anthropic."
    if vendor_marker in resource_id:
        _, suffix = resource_id.split(vendor_marker, maxsplit=1)
        resource_id = f"anthropic.{suffix}"
    return resource_id.removesuffix(":0")
