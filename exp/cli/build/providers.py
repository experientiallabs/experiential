"""Project-aware provider and model confirmation shared by interactive build paths."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from exp.cli.providers import setup as provider_setup
from exp.common.models import ModelCatalog, ProviderSetup
from exp.common.project import ProjectStore


def require_replay_role_overrides(
    root: Path,
    project: str,
    *,
    world_model: str | None,
    judge: str | None,
    embedder: str | None,
) -> None:
    """Reject role overrides that differ from a selected completed build.

    Args:
        root: Local EXP root.
        project: Existing project identifier.
        world_model: Optional requested world-model alias.
        judge: Optional requested judge alias.
        embedder: Optional requested embedder alias.

    Raises:
        ValueError: A supplied override differs from the selected completed-build role.
    """
    store = ProjectStore(root, project)
    if not store.paths.project_toml.exists():
        return
    config = store.load_project()
    if config.build is None or config.models is None:
        return
    requested = {
        "world_model": world_model,
        "judge": judge,
        "embedder": embedder,
    }
    mismatches = tuple(
        f"{role}={alias!r} (selected {getattr(config.models, role)!r})"
        for role, alias in requested.items()
        if alias is not None and alias != getattr(config.models, role)
    )
    if mismatches:
        raise ValueError(
            "role overrides differ from the selected completed build: "
            + ", ".join(mismatches)
            + ". Build a new project to use different models."
        )


def configure_build_providers(
    root: Path,
    project: str,
    *,
    providers: tuple[str, ...],
    world_model: str | None,
    judge: str | None,
    embedder: str | None,
    console: Console,
) -> ModelCatalog:
    """Always present provider and model choices, defaulting to this project's roles.

    Shared catalog roles are defaults for new projects. Existing projects retain their frozen
    role identity, even when another project's setup changed the shared catalog defaults.
    Confirming those same choices preserves completed artifacts and their paid-work reuse.

    Args:
        root: Local EXP root containing the shared model catalog.
        project: Project whose saved role choices should be preselected.
        providers: Explicit provider choices, or an empty tuple to open the provider picker.
        world_model: Optional initial world-model choice.
        judge: Optional initial judge choice.
        embedder: Optional initial embedder choice.
        console: Terminal used for all provider and model screens.

    Returns:
        The catalog saved after the operator confirms the chosen models and roles.

    Raises:
        ValueError: Confirmed roles conflict with an existing immutable build.
    """
    store = ProjectStore(root, project)
    saved = store.load_project().models if store.paths.project_toml.exists() else None

    def validate_roles(setup: ProviderSetup) -> None:
        """Reject incompatible project roles before writing the shared catalog."""
        require_replay_role_overrides(
            root,
            project,
            world_model=setup.world_model,
            judge=setup.judge,
            embedder=setup.embedder,
        )

    return provider_setup.run_provider_setup(
        root,
        provider_setup.ProviderSetupOptions(
            providers=providers,
            world_model=world_model or (saved.world_model if saved else None),
            judge=judge or (saved.judge if saved else None),
            embedder=embedder or (saved.embedder if saved else None),
        ),
        non_interactive=False,
        replace=False,
        console=console,
        validate_setup=validate_roles,
    )
