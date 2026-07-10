"""Tests for the v0 ModelCard: round-trip, store loading, and graceful absence."""

from pathlib import Path

import pytest

from wmh.config.card import CardCorpus, CardFidelity, ModelCard, load_card, save_card
from wmh.config.store import WorldModelStore


def _card(name: str = "tau-bench") -> ModelCard:
    return ModelCard(
        name=name,
        title="Tau-bench world model",
        description="Airline/retail/telecom tool-call environment.",
        task="tau-bench",
        corpus=CardCorpus(traces=1033, steps=10578, source="traces.otel.jsonl"),
        provider="bedrock",
        model_id="us.anthropic.claude-opus-4-8",
        fidelity=CardFidelity(suite="default", score=0.90, run_id="r1"),
        built_at="2026-07-01T00:00:00Z",
        tags=["tool-calls", "example"],
    )


def test_card_round_trips_through_disk(tmp_path: Path) -> None:
    card = _card()
    save_card(card, tmp_path)
    loaded = load_card(tmp_path)
    assert loaded == card


def test_load_card_returns_none_when_absent(tmp_path: Path) -> None:
    assert load_card(tmp_path) is None


def test_load_card_rejects_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "card.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="card.json"):
        load_card(tmp_path)


def test_store_card_reads_model_card(tmp_path: Path) -> None:
    store = WorldModelStore(tmp_path)
    model_dir = store.model_dir("m1")
    model_dir.mkdir(parents=True)
    (model_dir / "config.toml").write_text("", encoding="utf-8")
    assert store.card("m1") is None  # model exists, no card yet: degrade, don't raise
    save_card(_card("m1"), model_dir)
    card = store.card("m1")
    assert card is not None
    assert card.name == "m1"


def test_store_card_missing_model_raises(tmp_path: Path) -> None:
    store = WorldModelStore(tmp_path)
    with pytest.raises(FileNotFoundError, match="m1"):
        store.card("m1")
