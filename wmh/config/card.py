"""The v0 ModelCard: descriptive metadata a built world model carries in its artifact.

`card.json` sits in the artifact root next to `config.toml` and is what distribution surfaces
(the website gallery, `GET /world_models`, future `wmh export`/`pull`) render. The shape follows
the registry contract sketch in the coordination plan (PLAN.md §2.1); WS-A4 owns and extends it.
A model without a card still loads everywhere — cards are additive metadata, never required.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

CARD_FILENAME = "card.json"


class CardCorpus(BaseModel):
    """Size and origin of the trace corpus the model was built from.

    `traces` is optional: for a model whose build predates card support the trace count may not
    be reconstructable, while the indexed step count always is.
    """

    traces: int | None = None
    steps: int
    source: str | None = None


class CardFidelity(BaseModel):
    """Headline fidelity: which eval suite produced it and the run it came from."""

    suite: str
    score: float
    std: float | None = None
    run_id: str | None = None


class ModelCard(BaseModel):
    """Machine-readable description of one built world model (see module docstring)."""

    schema_version: int = 1
    name: str
    title: str
    description: str = ""
    task: str | None = None
    corpus: CardCorpus
    provider: str
    model_id: str
    fidelity: CardFidelity | None = None
    cost_per_step_usd: float | None = None
    latency_per_step_s: float | None = None
    built_at: str | None = None  # ISO-8601 UTC
    license: str | None = None
    tags: list[str] = Field(default_factory=list)


def make_build_card(
    *,
    name: str,
    provider: str,
    model_id: str,
    traces: int | None,
    steps: int,
    built_at: str,
    source: str | None = None,
    title: str = "",
    description: str = "",
    tags: list[str] | None = None,
) -> ModelCard:
    """Assemble the card a completed build writes.

    The single card-construction site for both build paths (`wmh build` and serve-side builds),
    so their cards never drift. `fidelity`/cost/latency stay unset here — they are stamped later
    from eval results, not known at build time.
    """
    return ModelCard(
        name=name,
        title=title or name,
        description=description,
        corpus=CardCorpus(traces=traces, steps=steps, source=source),
        provider=provider,
        model_id=model_id,
        built_at=built_at,
        tags=tags or [],
    )


def card_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / CARD_FILENAME


def save_card(card: ModelCard, model_dir: str | Path) -> Path:
    """Write `card.json` into `model_dir`, returning the path written."""
    path = card_path(model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_card(model_dir: str | Path) -> ModelCard | None:
    """Read a model dir's `card.json`, or None when the model has no card.

    A present-but-broken card raises (with the offending path) rather than silently hiding the
    model's metadata: the card was written intentionally, so corruption is a real error.
    """
    path = card_path(model_dir)
    if not path.exists():
        return None
    try:
        return ModelCard.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"malformed card.json at {path}: {exc}") from exc
