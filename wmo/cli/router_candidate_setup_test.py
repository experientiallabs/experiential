"""Router candidate collection tests."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
import typer
from rich.console import Console
from rich.prompt import Confirm

from wmo.cli import router_candidate_setup
from wmo.cli.picker import PickerAction, PickerOption, PickerResult
from wmo.cli.provider_setup import RouterCandidatePickerResult
from wmo.cli.router_candidate_setup import collect_router_candidate_setup
from wmo.common.models import (
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelRoles,
    ProviderConnection,
    ProviderModelSelection,
    RouterCandidateSelection,
    configure_router_candidates,
    write_model_catalog,
)


def test_noninteractive_requires_two_candidates_and_incumbent_without_writing(
    tmp_path: Path,
) -> None:
    """Structured automation reports all missing role inputs before catalog mutation.

    Args:
        tmp_path: Temporary root containing the shared catalog.
    """
    path = tmp_path / "models.toml"
    catalog = _catalog()
    write_model_catalog(path, catalog)
    before = path.read_bytes()

    with pytest.raises(typer.BadParameter) as error:
        collect_router_candidate_setup(
            path,
            catalog,
            candidates=("candidate-a",),
            incumbent=None,
            non_interactive=True,
            console=_console(),
            interactive_command="wmo optimize router support --root /tmp/.wmo",
        )

    message = str(error.value)
    assert "second distinct" in message
    assert "--incumbent" in message
    assert "Run `wmo optimize router support --root /tmp/.wmo`" in message
    assert path.read_bytes() == before


def test_noninteractive_reuses_one_complete_persisted_selection(tmp_path: Path) -> None:
    """An exact saved selection is unambiguous automation input on replay.

    Args:
        tmp_path: Temporary root containing persisted candidate roles.
    """
    path = tmp_path / "models.toml"
    catalog = _catalog().model_copy(
        update={
            "roles": ModelRoles(candidates=("candidate-a", "candidate-b"), incumbent="candidate-a")
        }
    )
    write_model_catalog(path, catalog)

    plan = collect_router_candidate_setup(
        path,
        catalog,
        candidates=(),
        incumbent=None,
        non_interactive=True,
        console=_console(),
    )

    assert plan.selection.candidates == ("candidate-a", "candidate-b")
    assert plan.selection.incumbent == "candidate-a"


def test_interactive_collection_requires_final_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New candidate roles remain in memory when the operator rejects the summary.

    Args:
        tmp_path: Temporary root containing an unchanged catalog.
        monkeypatch: Patch selection input without a terminal.
    """
    path = tmp_path / "models.toml"
    catalog = _catalog()
    write_model_catalog(path, catalog)
    before = path.read_bytes()
    answers = iter(("1,2", "", "2"))

    def picker_answer(*args: object, **kwargs: object) -> str:
        """Return the next deterministic line answer for a non-terminal picker."""
        del args, kwargs
        return next(answers)

    def reject_summary(*args: object, **kwargs: object) -> bool:
        """Reject the final persistence summary."""
        del args, kwargs
        return False

    monkeypatch.setattr(Console, "input", picker_answer)
    monkeypatch.setattr(Confirm, "ask", reject_summary)

    with pytest.raises(typer.Abort):
        collect_router_candidate_setup(
            path,
            catalog,
            candidates=(),
            incumbent=None,
            non_interactive=False,
            console=_console(),
        )

    assert path.read_bytes() == before


def test_back_from_incumbent_keeps_the_candidates_just_chosen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reopening the candidate screen preselects the in-session choices, not persisted roles.

    Args:
        tmp_path: Temporary root containing persisted candidate roles.
        monkeypatch: Replace both picker screens with scripted results.
    """
    path = tmp_path / "models.toml"
    catalog = _catalog().model_copy(
        update={"roles": ModelRoles(candidates=("candidate-a", "world"), incumbent="candidate-a")}
    )
    write_model_catalog(path, catalog)
    seen: list[tuple[str, ...]] = []

    def scripted_candidates(
        console: Console,
        *,
        title: str,
        options: list[PickerOption],
        preselected: tuple[str, ...] = (),
        minimum: int = 1,
    ) -> PickerResult:
        """Record the offered preselection and choose both eligible aliases."""
        del console, title, options, minimum
        seen.append(tuple(preselected))
        return PickerResult(values=("candidate-b", "candidate-a"))

    incumbent_results = iter(
        (PickerResult(action=PickerAction.BACK), PickerResult(values=("candidate-b",)))
    )

    def scripted_incumbent(*args: object, **kwargs: object) -> PickerResult:
        """Go back once, then confirm one incumbent."""
        del args, kwargs
        return next(incumbent_results)

    def reject_summary(*args: object, **kwargs: object) -> bool:
        """Reject the final persistence summary."""
        del args, kwargs
        return False

    monkeypatch.setattr(router_candidate_setup, "choose_many", scripted_candidates)
    monkeypatch.setattr(router_candidate_setup, "choose_one", scripted_incumbent)
    monkeypatch.setattr(Confirm, "ask", reject_summary)

    with pytest.raises(typer.Abort):
        collect_router_candidate_setup(
            path,
            catalog,
            candidates=(),
            incumbent=None,
            non_interactive=False,
            console=_console(),
        )

    assert seen == [(), ("candidate-b", "candidate-a")]


def test_first_optimize_can_define_candidates_from_existing_connections(tmp_path: Path) -> None:
    """Structured first optimize adds candidate metadata only after complete validation.

    Args:
        tmp_path: Temporary root containing only build-time model roles.
    """
    path = tmp_path / "models.toml"
    catalog = ModelCatalog(
        connections={"provider": ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY")},
        models={"world": ModelRecord(connection="provider", model="world")},
        roles=ModelRoles(world_model="world"),
    )
    write_model_catalog(path, catalog)
    before = path.read_bytes()
    definitions = (_candidate_model("candidate-a"), _candidate_model("candidate-b"))

    plan = collect_router_candidate_setup(
        path,
        catalog,
        candidates=(),
        candidate_models=definitions,
        incumbent="candidate-a",
        non_interactive=True,
        console=_console(),
    )

    assert path.read_bytes() == before
    configured = configure_router_candidates(
        path,
        plan.selection,
        candidate_models=plan.candidate_models,
        expected_state_sha256=plan.expected_catalog_sha256,
    )
    assert configured.roles.candidates == ("candidate-a", "candidate-b")
    assert configured.roles.world_model == "world"
    assert configured.models["world"] == catalog.models["world"]


def test_missing_candidates_use_configured_provider_picker_without_raw_prompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing candidate role delegates discovery and selection to provider setup ownership.

    Args:
        tmp_path: Temporary root containing the shared catalog.
        monkeypatch: Replace provider discovery with a deterministic picker result.
    """
    path = tmp_path / "models.toml"
    catalog = _catalog().model_copy(
        update={
            "models": {
                "candidate-a": _catalog().models["candidate-a"],
                "world": _catalog().models["world"],
            }
        }
    )
    write_model_catalog(path, catalog)
    new_connection = ProviderConnection(
        name="new",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
    )
    new_model = _candidate_model("candidate-b").model_copy(update={"connection": "new"})

    def picker(*args: object, **kwargs: object) -> RouterCandidatePickerResult:
        """Return one configured and one newly discovered candidate without prompts."""
        del args
        assert kwargs["candidates"] == ("candidate-a",)
        return RouterCandidatePickerResult(
            selection=RouterCandidateSelection(
                candidates=("candidate-a", "candidate-b"), incumbent="candidate-a"
            ),
            candidate_models=(new_model,),
            connections=(new_connection,),
        )

    monkeypatch.setattr(router_candidate_setup, "run_router_candidate_picker", picker)
    monkeypatch.setattr(Confirm, "ask", lambda *args, **kwargs: True)

    plan = collect_router_candidate_setup(
        path,
        catalog,
        candidates=("candidate-a",),
        incumbent=None,
        non_interactive=False,
        console=_console(),
    )

    assert plan.selection.candidates == ("candidate-a", "candidate-b")
    assert plan.candidate_connections == (new_connection,)
    assert plan.prospective_catalog.connections["new"].provider == "openai"
    assert "candidate-b" in plan.prospective_catalog.models


def test_interactive_confirmation_cannot_retarget_an_existing_alias(tmp_path: Path) -> None:
    """Interactive collection rejects replacement metadata before presenting a summary.

    Args:
        tmp_path: Temporary root containing existing candidate aliases.
    """
    path = tmp_path / "models.toml"
    catalog = _catalog()
    write_model_catalog(path, catalog)
    replacement = _candidate_model("candidate-a").model_copy(update={"model": "different-model"})
    before = path.read_bytes()

    with pytest.raises(typer.BadParameter, match="use a new alias"):
        collect_router_candidate_setup(
            path,
            catalog,
            candidates=("candidate-a", "candidate-b"),
            candidate_models=(replacement,),
            incumbent="candidate-a",
            non_interactive=False,
            console=_console(),
        )

    assert path.read_bytes() == before


def _candidate_model(alias: str) -> ProviderModelSelection:
    """Return one complete explicit router candidate definition.

    Args:
        alias: Local candidate alias.

    Returns:
        Complete model and economics selection.
    """
    return ProviderModelSelection(
        alias=alias,
        connection="provider",
        model=alias,
        capabilities=ModelCapabilities(
            supports_completions=True,
            context_window_tokens=32_000,
            maximum_output_tokens=4_000,
            input_cost_per_million_tokens_usd=1,
            output_cost_per_million_tokens_usd=2,
            cached_input_cost_per_million_tokens_usd=0.5,
            cache_write_cost_per_million_tokens_usd=1.5,
        ),
    )


def _catalog() -> ModelCatalog:
    """Return two fully priced completion aliases and one unrelated model."""
    capabilities = ModelCapabilities(
        supports_completions=True,
        context_window_tokens=32_000,
        maximum_output_tokens=4_000,
        input_cost_per_million_tokens_usd=1,
        output_cost_per_million_tokens_usd=2,
        cached_input_cost_per_million_tokens_usd=0.5,
        cache_write_cost_per_million_tokens_usd=1.5,
    )
    return ModelCatalog(
        connections={"provider": ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY")},
        models={
            "candidate-a": ModelRecord(
                connection="provider", model="candidate-a", capabilities=capabilities
            ),
            "candidate-b": ModelRecord(
                connection="provider", model="candidate-b", capabilities=capabilities
            ),
            "world": ModelRecord(connection="provider", model="world"),
        },
    )


def _console() -> Console:
    """Return a deterministic non-color Rich console for prompt tests.

    Returns:
        Console with isolated non-color output.
    """
    return Console(
        file=StringIO(),
        color_system=None,
        force_terminal=False,
    )
