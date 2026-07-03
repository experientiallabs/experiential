"""Shared CLI helper for the agent commands: resolve + load a world model by name.

Mirrors `wmh.cli.app._load_model` but returns just the `(WorldModel, Provider)` pair the agent
commands need. Kept out of `agent_app` so the resolution logic is unit-testable without the typer
command wrappers.
"""

from __future__ import annotations

import typer

from wmh.config import WorldModelStore
from wmh.engine import WorldModel
from wmh.providers.base import Provider


def load_world_model_for_agent(name: str | None, root: str) -> tuple[WorldModel, Provider]:
    """Resolve a world model by `--name` (or the sole built one); load it with its provider."""
    from wmh.engine import load_world_model

    store = WorldModelStore(root)
    try:
        model_dir = store.resolve(name)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    return load_world_model(model_dir)
