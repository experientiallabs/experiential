"""Contract tests for the in-process local embedder (stubbed backend + gated live check)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from wmo.common.providers.local_embed import (
    DEFAULT_LOCAL_EMBED_MODEL,
    LOCAL_EMBED_DIM,
    MLX_DEFAULT_MODEL,
    LocalEmbedder,
    _mlx_available,
    default_model_cached,
    pick_backend,
)


def test_pick_backend_prefers_mlx_on_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("wmo.common.providers.local_embed._mlx_available", lambda: True)
    assert pick_backend() == "mlx"


def test_pick_backend_falls_back_to_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    # torch is a dev dependency, so the fallback leg is exercised for real; whether it says
    # cuda or cpu is the machine's to answer, and both are torch.
    monkeypatch.setattr("wmo.common.providers.local_embed._mlx_available", lambda: False)
    assert pick_backend() in ("torch-cuda", "torch-cpu")


def test_pick_backend_names_the_extra_when_nothing_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("wmo.common.providers.local_embed._mlx_available", lambda: False)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="local"):
        pick_backend()


def test_default_model_cached_is_false_without_a_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_backend() -> str:
        raise RuntimeError("nothing installed")

    monkeypatch.setattr("wmo.common.providers.local_embed.pick_backend", _no_backend)
    assert default_model_cached() is False


def _stub(width: int) -> LocalEmbedder:
    """An embedder whose backend is a deterministic stub of the given output width."""

    def encode(texts: list[str]) -> np.ndarray:
        return np.array([[float(len(text) + col) for col in range(width)] for text in texts])

    return LocalEmbedder(backend="torch-cpu", dim=LOCAL_EMBED_DIM, encode=encode)


def test_embed_normalizes_and_preserves_order() -> None:
    embedder = _stub(LOCAL_EMBED_DIM)
    vectors = np.array(embedder.embed(["a", "bbbb"]))
    assert vectors.shape == (2, LOCAL_EMBED_DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    # Different texts produce different (stub) vectors, in input order.
    assert not np.allclose(vectors[0], vectors[1])


def test_embed_of_nothing_is_nothing() -> None:
    assert _stub(LOCAL_EMBED_DIM).embed([]) == []


def test_a_width_mismatch_names_both_widths_and_the_fix() -> None:
    with pytest.raises(ValueError, match="1024"):
        _stub(8).embed(["text"])


def test_the_default_model_maps_to_its_mlx_conversion_on_mlx_only() -> None:
    assert LocalEmbedder(backend="mlx").resolved_model() == MLX_DEFAULT_MODEL
    assert LocalEmbedder(backend="torch-cpu").resolved_model() == DEFAULT_LOCAL_EMBED_MODEL
    # A user-pinned id is honored verbatim, even on mlx.
    assert LocalEmbedder("me/mine", backend="mlx").resolved_model() == "me/mine"


def test_sizes_must_be_positive() -> None:
    with pytest.raises(ValueError, match="dim"):
        LocalEmbedder(dim=0)
    with pytest.raises(ValueError, match="batch"):
        LocalEmbedder(batch=0)


@pytest.mark.skipif(
    not (
        _mlx_available()
        and importlib.util.find_spec("huggingface_hub") is not None
        and default_model_cached("mlx")
    ),
    reason="needs Apple silicon with mlx-lm installed and the default model already cached",
)
def test_live_mlx_embeds_semantically() -> None:
    """The real model, only where its weights are already on disk (never downloads in tests)."""
    embedder = LocalEmbedder()
    vectors = np.array(
        embedder.embed(
            [
                "Fix the race condition in the connection pool.",
                "Fix a race condition in the connection pooling code.",
                "Bake a chocolate cake for the birthday party.",
            ]
        )
    )
    assert vectors.shape == (3, LOCAL_EMBED_DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    near = float(vectors[0] @ vectors[1])
    far = float(vectors[0] @ vectors[2])
    assert near > far, "paraphrases must sit closer than an unrelated task"
