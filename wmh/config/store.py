"""Named world models on disk.

The selected project root (`.wmh/` by default) holds named world models under `models/<name>/`.
Each model directory is a self-contained artifact in the layout `ArtifactPaths` already understands
(config.toml, prompts/, index/, metrics.json). The store turns names into directories, lists what
is available, and reads a small summary for `wmh list`.

    .wmh/                <- writable: where `wmh build` writes
      models/
        tau2-airline/    <- one artifact (config.toml, prompts/, index/, metrics.json)
        retail-bench/    <- another

"Filesystem as DB": loading a model is just reading its folder. The default root is the writable
`.wmh/` directory used by `wmh build`; callers can pass another root such as `examples/<task>` to
read intentional prebuilt example artifacts from that root's `models/` directory.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from pydantic import BaseModel, JsonValue

from wmh.config.card import ModelCard, load_card
from wmh.config.config import ARTIFACT_DIR, ArtifactPaths, HarnessConfig

# The implicit model name used when the user does not pass `--name`.
DEFAULT_MODEL_NAME = "default"

# A safe, filesystem-friendly model name: no path separators, traversal, or leading dot.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_name(name: str) -> str:
    """Return `name` if it is a safe single path segment, else raise a friendly ValueError."""
    if not _NAME_RE.match(name) or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(
            f"invalid world model name {name!r}: use letters, digits, '.', '_', '-' "
            "(must start with a letter or digit, no path separators)"
        )
    return name


def normalize_name(name: str) -> str:
    """Dash-join whitespace: 'tau bench' -> 'tau-bench'. Validity stays validate_name's call."""
    return re.sub(r"\s+", "-", name.strip())


class ModelInfo(BaseModel):
    """A one-line summary of a built world model, for `wmh list`."""

    name: str
    serve_provider: str
    serve_model: str
    held_out_accuracy: float | None = None
    rollouts_used: int | None = None
    frontier_size: int | None = None


class WorldModelStore:
    """Resolves and enumerates named world models under a selected project root."""

    def __init__(self, root: str | Path = ARTIFACT_DIR) -> None:
        self.root = Path(root)

    @property
    def models_dir(self) -> Path:
        """The selected root's models dir (`.wmh/models/` by default)."""
        return self.root / "models"

    def model_dir(self, name: str) -> Path:
        """The artifact directory for `name` (where a build writes; may not exist)."""
        return self.models_dir / validate_name(name)

    def dir_for(self, name: str) -> Path | None:
        """The artifact dir holding `name`'s config, or None."""
        model_dir = self.model_dir(name)
        if ArtifactPaths(model_dir).config.exists():
            return model_dir
        return None

    def exists(self, name: str) -> bool:
        return self.dir_for(name) is not None

    def _names_in(self, models_dir: Path) -> set[str]:
        """Names of every artifact (a dir containing config.toml) directly under `models_dir`."""
        if not models_dir.exists():
            return set()
        return {d.name for d in models_dir.iterdir() if d.is_dir() and (d / "config.toml").exists()}

    def list_names(self) -> list[str]:
        """Sorted names of every locally built model."""
        return sorted(self._names_in(self.models_dir))

    def resolve(self, name: str | None) -> Path:
        """Resolve `name` to a model's artifact dir for read commands (serve/demo/play).

        With an explicit `name`, require it to exist. With `name=None`, fall back to the single
        available model if there is exactly one; else raise, listing the choices.
        """
        if name is not None:
            found = self.dir_for(name)
            if found is None:
                available = self.list_names()
                hint = f" (have: {', '.join(available)})" if available else ""
                raise FileNotFoundError(
                    f"no world model named {name!r} under {self.models_dir}{hint}; "
                    "run `wmh build --name <name>` first"
                )
            return found

        names = self.list_names()
        if not names:
            raise FileNotFoundError(
                f"no world models built under {self.models_dir}; "
                "run `wmh build --name <name>` first"
            )
        if len(names) > 1:
            raise ValueError(
                f"multiple world models built ({', '.join(names)}); pass --name to choose one"
            )
        resolved = self.dir_for(names[0])
        assert resolved is not None  # name came from list_names(), so it resolves
        return resolved

    def card(self, name: str) -> ModelCard | None:
        """The model's `card.json` metadata, or None when the model carries no card."""
        model_dir = self.dir_for(name)
        if model_dir is None:
            raise FileNotFoundError(f"no world model named {name!r} under {self.models_dir}")
        return load_card(model_dir)

    def info(self, name: str) -> ModelInfo:
        """Read a model's config + metrics into a summary (for `wmh list`)."""
        model_dir = self.dir_for(name)
        if model_dir is None:
            raise FileNotFoundError(f"no world model named {name!r}")
        paths = ArtifactPaths(model_dir)
        with paths.config.open("rb") as fh:
            config = HarnessConfig.model_validate(tomllib.load(fh))
        accuracy: float | None = None
        rollouts: int | None = None
        if paths.metrics.exists():
            metrics = json.loads(paths.metrics.read_text(encoding="utf-8"))
            accuracy = _as_float(metrics.get("held_out_accuracy"))
            rollouts = _as_int(metrics.get("rollouts_used"))
        frontier_size: int | None = None
        if paths.frontier.exists():
            frontier = json.loads(paths.frontier.read_text(encoding="utf-8"))
            if isinstance(frontier, list):
                frontier_size = len(frontier)
        serve = config.serve_provider_config()
        return ModelInfo(
            name=name,
            serve_provider=serve.kind.value,
            serve_model=serve.model,
            held_out_accuracy=accuracy,
            rollouts_used=rollouts,
            frontier_size=frontier_size,
        )

    def list_info(self) -> list[ModelInfo]:
        return [self.info(name) for name in self.list_names()]


def _as_float(value: JsonValue) -> float | None:
    # bool is an int subclass; exclude it so a stray `true` doesn't read as 1.0.
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _as_int(value: JsonValue) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
