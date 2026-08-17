"""Shared CLI rendering for sanitized provider-call failures."""

from __future__ import annotations

import os
import shlex
import traceback
from collections.abc import Sequence
from typing import NoReturn

import typer
from rich.console import Console

from wmo.runtime.models.providers.errors import ProviderError, sanitize_provider_text

_DEBUG_ENV = "WMO_DEBUG"


def debug_output_enabled(*flags: bool) -> bool:
    """Return whether sanitized stack diagnostics should be printed.

    Args:
        flags: Explicit command ``--debug`` values. Any true value enables debug output.

    Returns:
        ``True`` when a caller asked for debug output or ``WMO_DEBUG`` is a truthy env value.
    """
    if any(flags):
        return True
    return os.environ.get(_DEBUG_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def render_provider_failure(
    console: Console,
    error: ProviderError,
    *,
    saved_progress: Sequence[str],
    retry_command: str,
    debug: bool = False,
) -> None:
    """Print a concise actionable provider failure without a traceback.

    Args:
        console: Command-owned console used for operator output.
        error: Typed sanitized provider failure.
        saved_progress: Durable work that remains after the failed call.
        retry_command: Exact command that resumes without repeating completed work.
        debug: Whether to append sanitized stack frames after the operator summary.
    """
    console.print(f"Provider call failed: {error}")
    console.print(
        f"  {error.provider} {error.endpoint_class}"
        + (f" HTTP {error.status_code}" if error.status_code is not None else "")
    )
    if error.error_code or error.error_type:
        console.print(
            "  "
            + " ".join(part for part in (error.error_type, error.error_code) if part is not None)
        )
    if error.rejected_parameter is not None:
        console.print(f"  rejected parameter: {error.rejected_parameter}")
    if error.request_id is not None:
        console.print(f"  request id: {error.request_id}")
    console.print("  retryable" if error.retryable else "  not retryable")
    if saved_progress:
        console.print("Saved progress:")
        for item in saved_progress:
            console.print(f"  {item}")
    console.print("Retry:")
    console.print(f"  {retry_command}")
    if debug_output_enabled(debug):
        console.print("Debug:")
        for line in sanitized_stack(error):
            console.print(f"  {line}")


def exit_provider_failure(
    console: Console,
    error: ProviderError,
    *,
    saved_progress: Sequence[str],
    retry_command: str,
    debug: bool = False,
) -> NoReturn:
    """Render one provider failure and exit without a traceback.

    Args:
        console: Command-owned console used for operator output.
        error: Typed sanitized provider failure.
        saved_progress: Durable work that remains after the failed call.
        retry_command: Exact command that resumes without repeating completed work.
        debug: Whether to append sanitized stack frames after the operator summary.

    Raises:
        typer.Exit: Always exits with status 1 after rendering.
    """
    render_provider_failure(
        console,
        error,
        saved_progress=saved_progress,
        retry_command=retry_command,
        debug=debug,
    )
    raise typer.Exit(1)


def sanitized_stack(error: ProviderError) -> tuple[str, ...]:
    """Return traceback lines after redacting secret-shaped tokens.

    Args:
        error: Typed sanitized provider failure.

    Returns:
        Frame lines that never include credentials, headers, or raw request bodies.
    """
    lines: list[str] = []
    for raw in traceback.format_exception(type(error), error, error.__traceback__):
        cleaned = sanitize_provider_text(raw.rstrip())
        if cleaned:
            lines.extend(cleaned.splitlines())
    return tuple(lines)


def judge_calibration_retry_command(
    project: str,
    *,
    root: str,
    sample_size: int,
    input_price: float | None,
    output_price: float | None,
    maximum_input_tokens: int,
    maximum_cost_usd: float | None,
    accept_insufficient_labels: bool,
) -> str:
    """Return the exact command that resumes judge calibration after a provider failure.

    Args:
        project: Local project ID.
        root: Local project root.
        sample_size: Frozen calibration sample size.
        input_price: Advanced input-price override, or ``None`` to keep catalog pricing.
        output_price: Advanced output-price override, or ``None`` to keep catalog pricing.
        maximum_input_tokens: Conservative input bound for every call attempt.
        maximum_cost_usd: Explicit spend ceiling, or ``None`` to keep the shared command budget.
        accept_insufficient_labels: Whether the original run accepted fewer than ten labels.

    Returns:
        A copy-paste command that reuses saved labels and completed probes.
    """
    command = (
        f"wmo config judge calibrate {project} --root {root} "
        f"--sample-size {sample_size} --maximum-input-tokens {maximum_input_tokens}"
    )
    if input_price is not None:
        command += f" --input-usd-per-million {input_price}"
    if output_price is not None:
        command += f" --output-usd-per-million {output_price}"
    if maximum_cost_usd is not None:
        command += f" --maximum-cost-usd {maximum_cost_usd}"
    command += " --yes"
    if accept_insufficient_labels:
        command += " --accept-insufficient-labels"
    return command


def router_optimization_retry_command(
    project: str,
    *,
    root: str,
    candidates: tuple[str, ...],
    candidate_models: tuple[str, ...],
    incumbent: str | None,
    maximum_provider_cost_usd: float,
    maximum_judgments: int,
    preferred_fidelity_overlaps: int,
    maximum_model_calls: int,
    maximum_router_feature_tokens: int,
    maximum_retrieval_query_tokens: int,
    simulation_maximum_output_tokens: int,
    maximum_concurrency: int,
    approve_fidelity: bool,
    non_interactive: bool,
) -> str:
    """Return the exact command that resumes router optimization after a provider failure.

    Args:
        project: Local project ID.
        root: Local project root.
        candidates: Confirmed completion aliases from the failed run.
        candidate_models: JSON ``ProviderModelSelection`` objects supplied for new aliases.
        incumbent: Confirmed quality baseline alias, when one was selected.
        maximum_provider_cost_usd: Shared provider-spend ceiling.
        maximum_judgments: Judgment-call ceiling.
        preferred_fidelity_overlaps: Preferred real-overlap denominator.
        maximum_model_calls: Candidate turns admitted per simulation episode.
        maximum_router_feature_tokens: Input ceiling for each router feature embedding.
        maximum_retrieval_query_tokens: Input ceiling for each grounded retrieval query.
        simulation_maximum_output_tokens: Candidate and world-model output ceiling per turn.
        maximum_concurrency: Maximum simulation workers.
        approve_fidelity: Whether the original run approved measured fidelity.
        non_interactive: Whether the original run forbade prompts.

    Returns:
        A copy-paste command that reuses completed artifacts and the same preflight inputs.
    """
    command = [
        "wmo",
        "optimize",
        "router",
        project,
        "--root",
        root,
        "--maximum-provider-cost-usd",
        f"{maximum_provider_cost_usd:.6f}".rstrip("0").rstrip("."),
        "--maximum-judgments",
        str(maximum_judgments),
        "--preferred-fidelity-overlaps",
        str(preferred_fidelity_overlaps),
        "--maximum-model-calls",
        str(maximum_model_calls),
        "--maximum-router-feature-tokens",
        str(maximum_router_feature_tokens),
        "--maximum-retrieval-query-tokens",
        str(maximum_retrieval_query_tokens),
        "--simulation-maximum-output-tokens",
        str(simulation_maximum_output_tokens),
        "--maximum-concurrency",
        str(maximum_concurrency),
        "--yes",
    ]
    for alias in candidates:
        command.extend(["--candidate", alias])
    for definition in candidate_models:
        command.extend(["--candidate-model", definition])
    if incumbent is not None:
        command.extend(["--incumbent", incumbent])
    if approve_fidelity:
        command.append("--approve-fidelity")
    if non_interactive:
        command.append("--non-interactive")
    return shlex.join(command)
