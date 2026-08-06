"""In-process text embeddings: Qwen3-Embedding run locally via MLX or torch, no API.

This is `EmbedderKind.LOCAL` / `EmbedderSpec(kind="local")`: the semantic counterpart to the
hashing embedder that still needs no credential and no embedding API. Model weights download
from Hugging Face on FIRST use (they are public model weights, not our artifact) and land in
the shared HF cache; every embed after that is fully offline. The routing path built on it
(embed, retrieve neighbors, decide) therefore runs on the user's machine end to end.

Backend selection (`pick_backend`) is automatic and explicit in errors:

1. `mlx` on Apple silicon when `mlx_lm` is importable (the `local` extra installs it there).
   The default model id maps to the validated 4-bit MLX conversion of the same model
   (`MLX_DEFAULT_MODEL`), which is what the shipped DeepSWE artifact's vectors were produced
   with.
2. `torch-cuda` when torch is importable and reports a CUDA device.
3. `torch-cpu` when torch is importable.

Otherwise construction raises, naming the `local` extra to install. `mlx_lm`/`mlx` are imported
via `importlib` rather than statically because they exist only on darwin/arm64 (the extra is
platform-markered), and a static import would break the type gate everywhere else; torch and
transformers install on every platform, so their deferred imports stay static.

Pooling: the final hidden state of the last token of the PLAIN text (no appended EOS),
L2-normalized. The model card's example appends an EOS token; the recorded vectors the shipped
DeepSWE artifact was fitted on were produced without one (verified against that cache: no-EOS
last-token pooling reproduces its vectors at median cosine 1.0 across sampled tasks, appending
EOS drops to ~0.82), so this module matches the artifact rather than the card. Quantized (MLX
4-bit) and full-precision (torch) weights produce slightly different vectors of the same
geometry; anything that must be bit-exact (a published reproduction) serves recorded vectors
through `wmo.optimize.routing.embedding_cache.CachedTaskEmbedder` instead of re-embedding.
"""

from __future__ import annotations

import importlib
import importlib.util
import platform
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_LOCAL_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
LOCAL_EMBED_DIM = 1024  # Qwen3-Embedding-0.6B's native output width
# The validated MLX conversion of the default model (4-bit DWQ). Substituted for the default id
# on the mlx backend only, and only for the DEFAULT id: a user-pinned model id is honored
# verbatim on every backend.
MLX_DEFAULT_MODEL = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
# Qwen3-Embedding truncation budget, matching the model card's usage example.
MAX_INPUT_TOKENS = 8192

LocalBackend = Literal["mlx", "torch-cuda", "torch-cpu"]

_INSTALL_HINT = (
    "no local embedding backend is available: install the `local` extra "
    "(`pip install 'world-model-optimizer[local]'`), which provides mlx-lm on Apple silicon "
    "and torch elsewhere"
)


def _mlx_available() -> bool:
    """MLX runs on Apple-silicon macs only, and only when the platform-markered extra landed."""
    return (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_lm") is not None
    )


def pick_backend() -> LocalBackend:
    """Choose the best local backend this machine can run (see module docstring for the order).

    Raises:
        RuntimeError: Neither mlx nor torch is importable; the message names the extra.
    """
    if _mlx_available():
        return "mlx"
    if importlib.util.find_spec("torch") is not None:
        import torch

        return "torch-cuda" if torch.cuda.is_available() else "torch-cpu"
    raise RuntimeError(_INSTALL_HINT)


def default_model_cached(backend: LocalBackend | None = None) -> bool:
    """Whether the default model's weights are already in the local Hugging Face cache.

    What lets `--embedder auto` prefer the local backend without ever triggering a surprise
    download: cached weights mean the local embedder is free to construct, so auto picks it;
    uncached means the operator opts in once with an explicit `--embedder local`.
    """
    try:
        resolved = backend or pick_backend()
    except RuntimeError:
        return False
    model = MLX_DEFAULT_MODEL if resolved == "mlx" else DEFAULT_LOCAL_EMBED_MODEL
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(model, "config.json"), str)


class LocalEmbedder:
    """Embedder over an in-process Qwen3-Embedding model (the `Embedder` protocol).

    The model loads lazily on the first `embed` call (loading is seconds, downloading on first
    ever use is hundreds of MB), and the produced width is checked against `dim` so a policy
    fitted at 1024 dimensions can never silently receive vectors of another geometry.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        dim: int = LOCAL_EMBED_DIM,
        batch: int = 16,
        backend: LocalBackend | None = None,
        encode: Callable[[list[str]], np.ndarray] | None = None,
    ) -> None:
        """Configure the embedder; nothing heavy happens until the first `embed`.

        Args:
            model: Hugging Face model id; None means `DEFAULT_LOCAL_EMBED_MODEL` (which the
                mlx backend serves through its validated 4-bit conversion, `MLX_DEFAULT_MODEL`).
            dim: Expected output width; produced vectors are checked against it.
            batch: Texts per forward pass on the torch backend (mlx encodes per text).
            backend: Pin a backend instead of `pick_backend()`'s auto-selection.
            encode: Test seam: a stub returning an (n, dim) array in place of a real model.

        Raises:
            ValueError: `dim` or `batch` is not positive.
        """
        if dim <= 0:
            raise ValueError(f"embedding dim must be positive, got {dim}")
        if batch <= 0:
            raise ValueError(f"batch must be positive, got {batch}")
        self.model = model or DEFAULT_LOCAL_EMBED_MODEL
        self.dim = dim
        self._batch = batch
        self._backend = backend
        self._encode = encode

    def resolved_backend(self) -> LocalBackend:
        """The backend this embedder runs on (pinned, or picked for this machine)."""
        if self._backend is None:
            self._backend = pick_backend()
        return self._backend

    def resolved_model(self) -> str:
        """The model id actually loaded: the default maps to its MLX conversion on mlx."""
        if self.model == DEFAULT_LOCAL_EMBED_MODEL and self.resolved_backend() == "mlx":
            return MLX_DEFAULT_MODEL
        return self.model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed `texts` into L2-normalized `dim`-wide vectors.

        Raises:
            ValueError: The model produced vectors of a width other than `dim` (names both
                widths and the model, since this means the spec and the model disagree).
        """
        if not texts:
            return []
        if self._encode is None:
            self._encode = self._build_encode()
        vectors = np.asarray(self._encode(list(texts)), dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape != (len(texts), self.dim):
            raise ValueError(
                f"local embedder '{self.resolved_model()}' produced shape {vectors.shape}, "
                f"expected ({len(texts)}, {self.dim}); set the spec's dim to the model's "
                "native width (Qwen3-Embedding-0.6B is 1024)"
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.where(norms > 0.0, norms, 1.0)
        return [row.tolist() for row in vectors]

    # ------------------------------------------------------------------ backends
    def _build_encode(self) -> Callable[[list[str]], np.ndarray]:
        backend = self.resolved_backend()
        if backend == "mlx":
            return self._mlx_encode()
        return self._torch_encode(device="cuda" if backend == "torch-cuda" else "cpu")

    def _mlx_encode(self) -> Callable[[list[str]], np.ndarray]:
        """Last-token pooling through mlx_lm's inner model (hidden states, not logits).

        One text per forward pass: no padding, so position -1 is the text's own last token,
        the recipe the shipped artifact's vectors were produced with (module docstring).
        """
        mlx_lm = importlib.import_module("mlx_lm")  # importlib: platform-conditional extra
        mx = importlib.import_module("mlx.core")
        # Indexed rather than unpacked: `mlx_lm.load` is typed as returning a third (config)
        # element in some versions, and only the model and tokenizer are wanted on any of them.
        loaded = mlx_lm.load(self.resolved_model())
        model, tokenizer = loaded[0], loaded[1]

        def encode(texts: list[str]) -> np.ndarray:
            rows = []
            for text in texts:
                ids = tokenizer.encode(text)[:MAX_INPUT_TOKENS]
                hidden = model.model(mx.array([ids]))
                # bfloat16 activations have no numpy buffer equivalent; cast in mlx first.
                rows.append(np.asarray(hidden[0, -1, :].astype(mx.float32), dtype=np.float64))
            return np.stack(rows)

        return encode

    def _torch_encode(self, *, device: str) -> Callable[[list[str]], np.ndarray]:
        """Last-token pooling via transformers, left-padded so the last position is real."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.resolved_model(), padding_side="left")
        if tokenizer is None:  # transformers types this Optional; embedding needs one
            raise RuntimeError(f"no tokenizer ships with {self.resolved_model()}")
        model = AutoModel.from_pretrained(self.resolved_model()).to(device).eval()

        def encode(texts: list[str]) -> np.ndarray:
            rows: list[np.ndarray] = []
            for start in range(0, len(texts), self._batch):
                batch = tokenizer(
                    texts[start : start + self._batch],
                    padding=True,
                    truncation=True,
                    max_length=MAX_INPUT_TOKENS,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    hidden = model(**batch).last_hidden_state
                # Left padding puts every sequence's last real token at position -1 (no EOS
                # appended; see the module docstring's pooling note).
                rows.extend(hidden[:, -1, :].float().cpu().numpy())
            return np.stack(rows)

        return encode
