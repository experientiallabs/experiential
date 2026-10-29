"""Validate and translate exact numeric reasoning controls on qualified provider wires."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderParameterError

_QWEN_BUDGET_HOSTS = frozenset(
    {
        "dashscope.aliyuncs.com",
        "dashscope-intl.aliyuncs.com",
        "dashscope-us.aliyuncs.com",
        "maas.qwencloudapi.com",
        "token-plan.ap-southeast-1.maas.aliyuncs.com",
    }
)
# Deliberate model contracts, not arbitrary future generations or instruct-only variants.
_QWEN_BUDGET_MODELS = frozenset(
    {
        "qwen3.8-max",
        "qwen3.8-flash",
        "qwen3.8-2.4t-a95b",
        "qwen3.8-27b",
        "qwen3.8-omni-flash",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.7-flash",
        "qwen3.6-max-preview",
        "qwen3.6-plus",
        "qwen3.6-flash",
        "qwen3.5-plus",
        "qwen3.5-flash",
        "qwen3.5-397b-a17b",
        "qwen3.5-122b-a10b",
        "qwen3.5-35b-a3b",
        "qwen3.5-27b",
        "qwen3.5-9b",
        "qwen3.5-4b",
        "qwen3-0.6b",
        "qwen3-1.7b",
        "qwen3-4b",
        "qwen3-8b",
        "qwen3-14b",
        "qwen3-32b",
        "qwen3-30b-a3b",
        "qwen3-235b-a22b",
        "qwen3-235b-a22b-thinking-2507",
        "qwen3-30b-a3b-thinking-2507",
        "qwen3-max-preview",
        "qwen3-omni-flash",
        "qwen3-vl-plus",
        "qwen3-vl-flash",
        "qwen3-vl-235b-a22b-thinking",
        "qwen3-vl-30b-a3b-thinking",
        "qwen3-vl-32b-thinking",
        "qwen3-vl-8b-thinking",
        "glm-4.7",
        "glm-5",
        "glm-5.1",
        "glm-5.2",
        "kimi-k2-thinking",
        "kimi-k2.5",
        "kimi-k2.6",
        "kimi-k2.7-code",
    }
)
_QWEN_TOTAL_CAP_MODELS = frozenset(
    {
        "qwen3.8-max",
        "qwen3.8-flash",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.7-flash",
        "qwen3.6-plus",
        "qwen3.6-flash",
        "qwen3.5-plus",
        "qwen3.5-flash",
    }
)
_ANTHROPIC_BUDGET_MODELS = frozenset(
    {
        "claude-3-7-sonnet",
        "claude-sonnet-4",
        "claude-sonnet-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4",
        "claude-opus-4-1",
        "claude-opus-4-5",
        "claude-opus-4-6",
        "claude-haiku-4-5",
        "claude-mythos-preview",
    }
)
_GEMINI_BUDGET_RANGES = {
    "gemini-2.5-pro": (128, 32768, False),
    "gemini-2.5-flash": (0, 24576, True),
    "gemini-2.5-flash-lite": (512, 24576, True),
    "gemini-2.5-flash-preview-04-17": (0, 24576, True),
    "gemini-2.5-flash-preview-05-20": (0, 24576, True),
    "gemini-2.5-flash-preview-09-2025": (0, 24576, True),
    "gemini-2.5-flash-lite-preview-06-17": (512, 24576, True),
    "gemini-2.5-flash-lite-preview-09-2025": (512, 24576, True),
    "gemini-2.5-pro-preview-06-05": (128, 32768, False),
}


def _dated_model(model: str, models: frozenset[str]) -> str | None:
    """Match explicit model identities or their provider's dated snapshots."""
    if model in models:
        return model
    return next(
        (
            root
            for root in models
            if re.fullmatch(re.escape(root) + r"-(?:\d{4}|\d{8}|\d{4}-\d{2}-\d{2})", model)
        ),
        None,
    )


def thinking_budget_value(request: GatewayRequest) -> int | None:
    """Read the caller's numeric control without changing its public spelling."""
    if request.thinking_budget is not None:
        return request.thinking_budget
    config = request.provider_thinking_config
    value = config.get("budget_tokens") if config is not None else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def thinking_budget_parameter(request: GatewayRequest) -> str:
    """Name the original caller field in any budget refusal or disclosure."""
    return "thinking_budget" if request.thinking_budget is not None else "thinking.budget_tokens"


def qwen_uses_total_budget_cap(profile: GatewayWireProfile) -> bool:
    """Whether this Qwen Cloud model documents a combined max_completion_tokens ceiling."""
    return _dated_model(profile.model_id, _QWEN_TOTAL_CAP_MODELS) is not None


def thinking_budget_wire_field(profile: GatewayWireProfile) -> str:
    """Return the exact provider control used after compatibility validation."""
    if profile.dialect == "anthropic_messages":
        return "thinking.budget_tokens"
    if profile.dialect == "gemini_generate_content":
        return "generationConfig.thinkingConfig.thinkingBudget"
    return "thinking_budget"


def require_thinking_budget_support(profile: GatewayWireProfile, request: GatewayRequest) -> None:
    """Validate numeric controls before admission and again at frozen dispatch.

    Budgets are provider targets, not promises about observed reasoning length.
    The output ceiling independently bounds financial reservation. Unknown
    models, relays and future families fail closed instead of silently ignoring
    a control. Zero and -1 are retained only where their documented semantics apply.
    """
    budget = thinking_budget_value(request)
    if budget is None:
        return
    param = thinking_budget_parameter(request)

    def reject(message: str, *, invalid: bool = False) -> None:
        """Refuse with the original field and an actionable explanation."""
        raise ProviderParameterError(
            message=message,
            param=param,
            code="invalid_parameter" if invalid else "unsupported_parameter",
        )

    config = request.provider_thinking_config
    if request.thinking_budget is not None and config is not None:
        reject(
            "Choose one numeric thinking control; remove the other thinking configuration.",
            invalid=True,
        )
    if config is not None and config.get("type") != "enabled":
        reject("A nested thinking budget requires type 'enabled'.", invalid=True)
    if (
        request.reasoning_effort is not None
        or (request.provider_output_config or {}).get("effort") is not None
    ):
        reject(
            "A numeric thinking budget cannot be combined with "
            "reasoning_effort/output_config.effort "
            "effort. Choose one depth control.",
            invalid=True,
        )
    if not profile.supports_reasoning or profile.reasoning_effort_required:
        reject(
            "This route cannot preserve a numeric thinking budget. Choose a "
            "budget-capable route or remove the budget."
        )
    if (
        profile.dialect != "anthropic_messages"
        and config is not None
        and set(config) - {"type", "budget_tokens"}
    ):
        reject(
            "This route cannot preserve the complete thinking "
            "configuration. Remove extra thinking controls or choose Anthropic."
        )
    if profile.dialect == "anthropic_messages":
        model = profile.model_id.lower().replace(".", "-").replace("_", "-")
        if _dated_model(model, _ANTHROPIC_BUDGET_MODELS) is None:
            reject(
                "This Anthropic model does not support a numeric thinking "
                "budget. Choose a budget-capable model or remove the budget "
                "and use adaptive thinking."
            )
        if budget < 1024:
            reject(
                "Anthropic thinking budgets must be at least 1024. Increase the budget.",
                invalid=True,
            )
        maximum = request.maximum_output_tokens or profile.maximum_output_tokens
        if maximum is not None and budget >= maximum:
            reject(
                "The thinking budget must be below the output limit. Raise "
                "the output limit or lower the budget.",
                invalid=True,
            )
        return
    if profile.dialect == "gemini_generate_content":
        bounds = _GEMINI_BUDGET_RANGES.get(profile.model_id.removeprefix("models/"))
        if bounds is None:
            reject(
                "This Gemini model is not qualified for numeric budgets. "
                "Choose a supported Gemini 2.5 model or use reasoning_effort."
            )
            return
        minimum, maximum, can_disable = bounds
        if budget != -1 and not (budget == 0 and can_disable) and not minimum <= budget <= maximum:
            reject(
                f"This Gemini model requires a budget from {minimum} to "
                f"{maximum}, or -1 for dynamic thinking"
                + (
                    ", or 0 to disable thinking."
                    if can_disable
                    else "; thinking cannot be disabled."
                ),
                invalid=True,
            )
        return
    origin = urlsplit(profile.url)
    qwen_host = (
        origin.hostname in _QWEN_BUDGET_HOSTS
        or re.fullmatch(
            r"[a-zA-Z0-9-]+\.ap-southeast-1\.maas\.aliyuncs\.com", origin.hostname or ""
        )
        is not None
    )
    if (
        profile.dialect != "openai_compatible"
        or origin.scheme != "https"
        or not qwen_host
        or _dated_model(profile.model_id, _QWEN_BUDGET_MODELS) is None
    ):
        reject(
            "This route cannot preserve a numeric thinking budget. Choose a "
            "qualified Anthropic, Gemini 2.5 or Qwen Cloud model, or remove the budget."
        )
    if budget < 0:
        reject(
            "Dynamic thinking (-1) is only supported on qualified Gemini "
            "routes. Supply a nonnegative Qwen Cloud budget.",
            invalid=True,
        )
    if not qwen_uses_total_budget_cap(profile):
        maximum = request.maximum_output_tokens or profile.maximum_output_tokens
        if maximum is not None and budget >= maximum:
            reject(
                "This route reserves thinking and answer tokens separately. "
                "Raise the total output limit above the thinking budget.",
                invalid=True,
            )


def budgeted_provider_request(
    profile: GatewayWireProfile, request: GatewayRequest
) -> GatewayRequest:
    """Translate only the numeric carrier for one already-validated provider wire."""
    budget = thinking_budget_value(request)
    if budget is None:
        return request
    if profile.dialect == "anthropic_messages":
        config: JsonObject = request.provider_thinking_config or {
            "type": "enabled",
            "budget_tokens": budget,
        }
        return request.model_copy(
            update={"thinking_budget": None, "provider_thinking_config": config}
        )
    return request.model_copy(update={"thinking_budget": budget, "provider_thinking_config": None})


def qwen_budget_payload(
    profile: GatewayWireProfile, request: GatewayRequest, payload: JsonObject
) -> None:
    """Bind Qwen thinking plus answer ceilings to the reserved total output bound."""
    budget = thinking_budget_value(request)
    if budget is None:
        return
    payload["thinking_budget"] = budget
    # Always-thinking Kimi K2 exposes no enable switch.
    if _dated_model(profile.model_id, frozenset({"kimi-k2-thinking"})) is None:
        payload["enable_thinking"] = True
    if not qwen_uses_total_budget_cap(profile):
        maximum = request.maximum_output_tokens or profile.maximum_output_tokens
        if maximum is None:
            raise ProviderParameterError(
                message="Supply an output limit to bound the separate thinking and answer budgets.",
                param=thinking_budget_parameter(request),
                code="invalid_parameter",
            )
        payload.pop("max_completion_tokens", None)
        payload["max_tokens"] = maximum - budget
