"""Interactive setup prompts for the identity-scoped standard guardrail pack."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from exp.cli.gateway.guardrail_setup_store import (
    GUARDRAILS_CUSTOM,
    GUARDRAILS_OFF,
    GUARDRAILS_STANDARD,
    GuardrailInspection,
    GuardrailSetupMode,
    GuardrailSetupPlan,
    GuardrailSetupSelection,
    apply_setup_guardrails,
    guardrail_config_path,
    guardrail_setup_compensation,
    inspect_setup_guardrails,
    setup_adapter_id,
    setup_policy_id,
)
from exp.cli.providers.provider_picker import ask_text
from exp.runtime.gateway.guardrails.http_json import (
    validate_bearer_env_name,
    validate_classifier_url,
)

_ENABLED_ANSWERS = frozenset({"on", "yes", "y", "true", "standard"})
_DISABLED_ANSWERS = frozenset({"off", "no", "n", "false"})
_CUSTOM_KEEP_ANSWERS = frozenset({"keep", "k"})
_CUSTOM_REPLACE_ANSWERS = frozenset({"replace"})

__all__ = (
    "GUARDRAILS_CUSTOM",
    "GUARDRAILS_OFF",
    "GUARDRAILS_STANDARD",
    "GuardrailInspection",
    "GuardrailSetupMode",
    "GuardrailSetupPlan",
    "GuardrailSetupSelection",
    "apply_setup_guardrails",
    "collect_guardrail_setup",
    "guardrail_config_path",
    "guardrail_setup_compensation",
    "inspect_setup_guardrails",
    "setup_adapter_id",
    "setup_policy_id",
)


def collect_guardrail_setup(
    *,
    console: Console,
    root: Path,
    organization_id: str,
    identity_id: str,
    edit: bool,
) -> GuardrailSetupSelection:
    """Collect a preserve, disable, or standard-pack choice for one identity.

    Enter on the defaults screen preserves the current file. Edit is the only
    path that can author or remove setup-owned artifacts.

    Args:
        console: Terminal used for the short opt-in prompts.
        root: EXP root that may already contain a guardrail file.
        organization_id: Local organization written on a standard policy.
        identity_id: Identity selected on the defaults screen.
        edit: Whether the operator chose to edit the displayed defaults.

    Returns:
        A mutation plan, or ``None`` when the current file must be preserved.

    Raises:
        ValueError: The existing file is malformed, or an answer is ambiguous.
    """
    inspection = inspect_setup_guardrails(root, organization_id, identity_id)
    if not edit:
        return GuardrailSetupSelection(plan=None, display=inspection.display)
    if inspection.mode is GuardrailSetupMode.CUSTOM:
        return _collect_custom_choice(
            console=console,
            organization_id=organization_id,
            identity_id=identity_id,
            inspection=inspection,
        )
    return _collect_standard_choice(
        console=console,
        organization_id=organization_id,
        identity_id=identity_id,
        inspection=inspection,
    )


def _collect_custom_choice(
    *,
    console: Console,
    organization_id: str,
    identity_id: str,
    inspection: GuardrailInspection,
) -> GuardrailSetupSelection:
    """Keep a hand-authored policy unless the operator types replace.

    Args:
        console: Terminal used for the keep-or-replace prompt.
        organization_id: Local organization for a replacement pack.
        identity_id: Identity whose custom policy is on disk.
        inspection: Current custom/preserved classification.

    Returns:
        A preserve selection, or a standard-pack replacement plan.

    Raises:
        ValueError: The answer is not an explicit keep or replace.
    """
    answer = ask_text(
        "Existing guardrails are custom/preserved. Type keep or replace",
        console=console,
        default="keep",
    ).lower()
    if answer in _CUSTOM_KEEP_ANSWERS:
        return GuardrailSetupSelection(plan=None, display=inspection.display)
    if answer not in _CUSTOM_REPLACE_ANSWERS:
        raise ValueError(
            "type keep to preserve the custom policy, or replace to author the standard pack"
        )
    return GuardrailSetupSelection(
        plan=_collect_standard_plan(
            console=console,
            organization_id=organization_id,
            identity_id=identity_id,
            inspection=inspection,
            replace_custom=True,
        ),
        display=GUARDRAILS_STANDARD,
    )


def _collect_standard_choice(
    *,
    console: Console,
    organization_id: str,
    identity_id: str,
    inspection: GuardrailInspection,
) -> GuardrailSetupSelection:
    """Opt the selected identity into or out of the setup-owned standard pack.

    Args:
        console: Terminal used for the short standard-pack prompts.
        organization_id: Local organization written on the policy.
        identity_id: Identity selected by the operator.
        inspection: Current Off or Standard classification.

    Returns:
        A disable plan or a standard-pack plan.

    Raises:
        ValueError: The answer is ambiguous or the classifier URL is missing.
    """
    default = "on" if inspection.mode is GuardrailSetupMode.STANDARD else "off"
    answer = ask_text("Standard guardrails", console=console, default=default).lower()
    if answer in _DISABLED_ANSWERS:
        return GuardrailSetupSelection(
            plan=GuardrailSetupPlan(
                action="off",
                organization_id=organization_id,
                identity_id=identity_id,
            ),
            display=GUARDRAILS_OFF,
        )
    if answer not in _ENABLED_ANSWERS:
        raise ValueError(
            "type on to enable the standard pack, or off to leave this identity unguarded"
        )
    return GuardrailSetupSelection(
        plan=_collect_standard_plan(
            console=console,
            organization_id=organization_id,
            identity_id=identity_id,
            inspection=inspection,
            replace_custom=False,
        ),
        display=GUARDRAILS_STANDARD,
    )


def _collect_standard_plan(
    *,
    console: Console,
    organization_id: str,
    identity_id: str,
    inspection: GuardrailInspection,
    replace_custom: bool,
) -> GuardrailSetupPlan:
    """Collect the dedicated classifier endpoint and optional bearer name.

    Args:
        console: Terminal used for the two classifier prompts.
        organization_id: Local organization written on the policy.
        identity_id: Identity selected by the operator.
        inspection: Current file, used only to default an existing URL.
        replace_custom: Whether a custom policy for this identity may be replaced.

    Returns:
        A validated standard-pack plan.

    Raises:
        ValueError: The URL or bearer environment name is invalid.
    """
    url_default = inspection.classifier_url or ""
    classifier_url = ask_text(
        "Classifier URL",
        console=console,
        default=url_default if url_default else None,
    )
    if not classifier_url:
        raise ValueError("Classifier URL is required when standard guardrails are enabled")
    validate_classifier_url(classifier_url)
    bearer_default = inspection.bearer_env or ""
    bearer_answer = ask_text(
        "Bearer credential environment variable",
        console=console,
        default=bearer_default,
    )
    bearer_env = validate_bearer_env_name(bearer_answer) if bearer_answer else None
    return GuardrailSetupPlan(
        action="standard",
        organization_id=organization_id,
        identity_id=identity_id,
        classifier_url=classifier_url,
        bearer_env=bearer_env,
        replace_custom=replace_custom,
    )
