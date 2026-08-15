"""Interactive and structured collection of router candidates without provider calls."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt

from wmo.common.models import (
    ModelCapabilities,
    ModelCatalog,
    ProviderModelSelection,
    RouterCandidateSelection,
    catalog_state_sha256,
    completion_candidate_aliases,
    validate_router_candidate_selection,
)
from wmo.optimize.router.composition import RouterCandidateSetupPlan


def collect_router_candidate_setup(
    path: Path,
    catalog: ModelCatalog,
    *,
    candidates: tuple[str, ...],
    candidate_models: tuple[ProviderModelSelection, ...] = (),
    incumbent: str | None,
    non_interactive: bool,
    console: Console,
) -> RouterCandidateSetupPlan:
    """Collect and validate candidate roles without mutating the shared catalog.

    Args:
        path: Shared ``models.toml`` path whose state is being confirmed.
        catalog: Already validated catalog shown during collection.
        candidates: Repeatable explicit candidate aliases supplied by the caller.
        candidate_models: Repeatable complete candidate definitions supplied by the caller.
        incumbent: Optional explicit quality incumbent.
        non_interactive: Whether absent values must be reported instead of prompted.
        console: Rich terminal used for prompts and the final summary.

    Returns:
        Confirmed selection and exact catalog digest for a later atomic write.

    Raises:
        typer.BadParameter: The selection is missing, ambiguous, or ineligible.
        typer.Abort: The operator rejects the interactive summary.
    """
    state_sha256 = catalog_state_sha256(path)
    collected_models = candidate_models
    if not non_interactive:
        collected_models = _interactive_candidate_models(catalog, candidate_models, console=console)
    prospective = _catalog_with_candidate_models(catalog, collected_models)
    if non_interactive:
        selected = candidates or tuple(item.alias for item in collected_models)
        selection = _noninteractive_selection(prospective, selected, incumbent)
    else:
        selection = _interactive_selection(prospective, candidates, incumbent, console=console)
    problems = validate_router_candidate_selection(prospective, selection)
    if problems:
        raise typer.BadParameter("router candidate setup is incomplete: " + "; ".join(problems))
    prospective = prospective.model_copy(
        update={
            "roles": prospective.roles.model_copy(
                update={
                    "candidates": selection.candidates,
                    "incumbent": selection.incumbent,
                }
            )
        }
    )
    console.print("[bold]Router candidate summary[/bold]")
    console.print("candidates: " + ", ".join(selection.candidates))
    console.print(f"incumbent: {selection.incumbent}")
    for model in collected_models:
        caps = model.capabilities
        console.print(
            f"{model.alias}: {model.connection}/{model.model}, "
            f"context {caps.context_window_tokens}, output {caps.maximum_output_tokens}"
        )
    if not non_interactive and not Confirm.ask(
        "Save these router candidates?", default=False, console=console
    ):
        raise typer.Abort()
    return RouterCandidateSetupPlan(
        selection=selection,
        candidate_models=collected_models,
        prospective_catalog=prospective,
        expected_catalog_sha256=state_sha256,
    )


def _catalog_with_candidate_models(
    catalog: ModelCatalog,
    candidate_models: tuple[ProviderModelSelection, ...],
) -> ModelCatalog:
    """Merge explicit candidate definitions into an in-memory catalog.

    Args:
        catalog: Current catalog that remains unchanged on disk.
        candidate_models: Complete candidate definitions collected for this optimize run.

    Returns:
        Prospective catalog used for aggregate validation.

    Raises:
        typer.BadParameter: An alias repeats or names an unknown connection.
    """
    aliases = tuple(item.alias for item in candidate_models)
    if len(set(aliases)) != len(aliases):
        raise typer.BadParameter("candidate model definitions must use unique aliases")
    unknown_connections = sorted(
        {item.connection for item in candidate_models}.difference(catalog.connections)
    )
    if unknown_connections:
        raise typer.BadParameter(
            "candidate models name unknown provider connections: " + ", ".join(unknown_connections)
        )
    models = dict(catalog.models)
    for item in candidate_models:
        candidate_record = item.catalog_record()
        existing = models.get(item.alias)
        if existing is not None and existing != candidate_record:
            raise typer.BadParameter(
                f"candidate alias {item.alias!r} already names different model metadata; "
                "use a new alias"
            )
        models[item.alias] = candidate_record
    return catalog.model_copy(update={"models": models})


def _interactive_candidate_models(
    catalog: ModelCatalog,
    candidate_models: tuple[ProviderModelSelection, ...],
    *,
    console: Console,
) -> tuple[ProviderModelSelection, ...]:
    """Collect complete candidate definitions until at least two aliases are eligible.

    Args:
        catalog: Existing provider connections and model aliases.
        candidate_models: Complete definitions already supplied on the command line.
        console: Rich terminal used for explicit model metadata prompts.

    Returns:
        Supplied and interactively collected candidate definitions.

    Raises:
        typer.BadParameter: Entered model metadata is invalid.
    """
    collected = list(candidate_models)
    prospective = _catalog_with_candidate_models(catalog, tuple(collected))
    while len(completion_candidate_aliases(prospective)) < 2:
        console.print(
            "Router optimization needs at least two completion candidates. "
            "Add a model using one configured provider connection."
        )
        try:
            selection = ProviderModelSelection(
                alias=Prompt.ask("Candidate alias", console=console),
                connection=Prompt.ask(
                    "Provider connection",
                    choices=sorted(catalog.connections),
                    console=console,
                ),
                model=Prompt.ask("Provider model ID", console=console),
                capabilities=ModelCapabilities(
                    supports_tools=Confirm.ask("Supports tools?", default=False, console=console),
                    supports_completions=True,
                    context_window_tokens=int(Prompt.ask("Context window tokens", console=console)),
                    maximum_output_tokens=int(Prompt.ask("Maximum output tokens", console=console)),
                    input_cost_per_million_tokens_usd=float(
                        Prompt.ask("Input USD per million tokens", console=console)
                    ),
                    output_cost_per_million_tokens_usd=float(
                        Prompt.ask("Output USD per million tokens", console=console)
                    ),
                    cached_input_cost_per_million_tokens_usd=float(
                        Prompt.ask("Cached input USD per million tokens", console=console)
                    ),
                    cache_write_cost_per_million_tokens_usd=float(
                        Prompt.ask("Cache write USD per million tokens", console=console)
                    ),
                ),
            )
        except ValueError as exc:
            raise typer.BadParameter(f"candidate model definition is invalid: {exc}") from None
        collected.append(selection)
        prospective = _catalog_with_candidate_models(catalog, tuple(collected))
    return tuple(collected)


def _noninteractive_selection(
    catalog: ModelCatalog,
    candidates: tuple[str, ...],
    incumbent: str | None,
) -> RouterCandidateSelection:
    """Resolve repeatable structured input or one complete persisted selection.

    Args:
        catalog: Current local catalog.
        candidates: Explicit repeatable aliases, possibly empty.
        incumbent: Explicit incumbent, possibly absent.

    Returns:
        Unambiguous selected roles.

    Raises:
        typer.BadParameter: Inputs are partial or no complete persisted selection exists.
    """
    selected = candidates or catalog.roles.candidates
    selected_incumbent = incumbent or (catalog.roles.incumbent if not candidates else None)
    missing = []
    if not selected:
        missing.append("at least two repeatable --candidate ALIAS values")
    elif len(selected) < 2:
        missing.append("a second distinct --candidate ALIAS value")
    if selected_incumbent is None:
        missing.append("--incumbent ALIAS")
    if missing:
        raise typer.BadParameter("noninteractive router setup is missing: " + "; ".join(missing))
    if selected_incumbent is None:
        raise AssertionError("validated noninteractive selection has an incumbent")
    try:
        return RouterCandidateSelection(candidates=selected, incumbent=selected_incumbent)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None


def _interactive_selection(
    catalog: ModelCatalog,
    candidates: tuple[str, ...],
    incumbent: str | None,
    *,
    console: Console,
) -> RouterCandidateSelection:
    """Prompt for multiple eligible aliases and one incumbent with no inferred model choice.

    Args:
        catalog: Current local catalog.
        candidates: Optional explicit aliases that skip only the candidate prompt.
        incumbent: Optional explicit incumbent that skips only the incumbent prompt.
        console: Rich terminal used for choices.

    Returns:
        Explicit selection ready for summary confirmation.

    Raises:
        typer.BadParameter: Fewer than two eligible aliases exist or entered aliases are invalid.
    """
    eligible = completion_candidate_aliases(catalog)
    if len(eligible) < 2:
        raise typer.BadParameter(
            "router optimization needs at least two configured aliases with "
            "supports_completions=true and complete input, output, cached input, and cache write "
            "pricing"
        )
    selected = candidates
    if not selected:
        default = (
            ",".join(catalog.roles.candidates)
            if len(catalog.roles.candidates) >= 2
            and set(catalog.roles.candidates).issubset(eligible)
            else None
        )
        answer = (
            Prompt.ask(
                "Candidate aliases (comma separated)",
                default=default,
                console=console,
            )
            if default is not None
            else Prompt.ask("Candidate aliases (comma separated)", console=console)
        )
        selected = tuple(item.strip() for item in answer.split(",") if item.strip())
    unknown = tuple(alias for alias in selected if alias not in eligible)
    if unknown:
        raise typer.BadParameter(
            "candidate aliases are not eligible: " + ", ".join(sorted(set(unknown)))
        )
    default_incumbent = catalog.roles.incumbent if catalog.roles.incumbent in selected else None
    selected_incumbent = incumbent or Prompt.ask(
        "Incumbent alias",
        choices=list(selected),
        default=default_incumbent,
        console=console,
    )
    if selected_incumbent is None:
        raise AssertionError("interactive incumbent prompt returned no alias")
    try:
        return RouterCandidateSelection(candidates=selected, incumbent=selected_incumbent)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
