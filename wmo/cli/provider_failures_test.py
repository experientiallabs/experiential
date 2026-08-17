"""Tests for shared CLI rendering of sanitized provider failures."""

from __future__ import annotations

from click import unstyle
from rich.console import Console

from wmo.cli.provider_failures import (
    judge_calibration_retry_command,
    render_provider_failure,
    router_optimization_retry_command,
    sanitized_stack,
)
from wmo.runtime.models.providers.errors import ProviderError

_SECRET = "sk-secret-live-key-1234567890"


def test_provider_failure_render_is_actionable_and_secret_free() -> None:
    """CLI output names the safe diagnostic, saved progress, and exact retry command."""
    console = Console(record=True, width=120)
    error = ProviderError(
        f"Unsupported parameter {_SECRET}",
        provider="openai",
        endpoint_class="responses",
        status_code=400,
        error_code="unsupported_parameter",
        error_type="invalid_request_error",
        rejected_parameter="temperature",
        request_id="req_safe_1",
    )

    render_provider_failure(
        console,
        error,
        saved_progress=(
            "10 human labels saved for this trace sample",
            "the failed provider attempt was not recorded as completed evidence",
        ),
        retry_command="wmo config judge calibrate demo --root /tmp/.wmo --yes",
    )
    printed = unstyle(console.export_text())

    assert "Provider call failed" in printed
    assert "openai responses HTTP 400" in printed
    assert "unsupported_parameter" in printed
    assert "rejected parameter: temperature" in printed
    assert "request id: req_safe_1" in printed
    assert "not retryable" in printed
    assert "10 human labels saved" in printed
    assert "wmo config judge calibrate demo --root /tmp/.wmo --yes" in printed
    assert "Traceback" not in printed
    assert _SECRET not in printed


def test_debug_stack_is_sanitized_and_opt_in() -> None:
    """Stack frames appear only for explicit debug output and never include secrets."""
    error = ProviderError(
        f"Authorization: Bearer {_SECRET}",
        provider="openai",
        endpoint_class="responses",
        status_code=401,
    )
    try:
        raise error
    except ProviderError as raised:
        lines = sanitized_stack(raised)

    assert lines
    assert all(_SECRET not in line for line in lines)
    assert any("ProviderError" in line or "errors.py" in line for line in lines)


def test_judge_retry_command_reuses_saved_labels_without_recollecting_them() -> None:
    """The printed retry command is the exact consented calibration invocation."""
    command = judge_calibration_retry_command(
        "support",
        root="/tmp/.wmo",
        sample_size=10,
        input_price=1.0,
        output_price=2.0,
        maximum_input_tokens=4096,
        maximum_cost_usd=5.0,
        accept_insufficient_labels=True,
    )

    assert command == (
        "wmo config judge calibrate support --root /tmp/.wmo --sample-size 10 "
        "--maximum-input-tokens 4096 --input-usd-per-million 1.0 "
        "--output-usd-per-million 2.0 --maximum-cost-usd 5.0 --yes "
        "--accept-insufficient-labels"
    )


def test_judge_retry_command_omits_optional_catalog_price_overrides() -> None:
    """Catalog-priced runs retry without inventing advanced price or ceiling flags."""
    command = judge_calibration_retry_command(
        "support",
        root="/tmp/.wmo",
        sample_size=10,
        input_price=None,
        output_price=None,
        maximum_input_tokens=32768,
        maximum_cost_usd=None,
        accept_insufficient_labels=False,
    )

    assert command == (
        "wmo config judge calibrate support --root /tmp/.wmo --sample-size 10 "
        "--maximum-input-tokens 32768 --yes"
    )


def test_router_retry_command_preserves_candidates_limits_and_fidelity_approval() -> None:
    """A failed optimize run reprints the same preflight inputs, not a bare --yes."""
    command = router_optimization_retry_command(
        "support",
        root="/tmp/.wmo",
        candidates=("candidate-a", "candidate-b"),
        candidate_models=('{"alias":"candidate-c","provider":"openai","model":"gpt"}',),
        incumbent="candidate-a",
        maximum_provider_cost_usd=12.5,
        maximum_judgments=40,
        preferred_fidelity_overlaps=6,
        maximum_model_calls=4,
        maximum_router_feature_tokens=4096,
        maximum_retrieval_query_tokens=16384,
        simulation_maximum_output_tokens=8000,
        maximum_concurrency=2,
        approve_fidelity=True,
        non_interactive=True,
    )

    assert command == (
        "wmo optimize router support --root /tmp/.wmo --maximum-provider-cost-usd 12.5 "
        "--maximum-judgments 40 --preferred-fidelity-overlaps 6 --maximum-model-calls 4 "
        "--maximum-router-feature-tokens 4096 --maximum-retrieval-query-tokens 16384 "
        "--simulation-maximum-output-tokens 8000 --maximum-concurrency 2 --yes "
        "--candidate candidate-a --candidate candidate-b "
        '--candidate-model \'{"alias":"candidate-c","provider":"openai","model":"gpt"}\' '
        "--incumbent candidate-a --approve-fidelity --non-interactive"
    )
