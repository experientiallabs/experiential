"""METHOD C step 1: extract PREFILL ACTIVATIONS from an open-weight model per DeepSWE task.

Why: our off-the-shelf text embeddings (text-embedding-3-large) carry no per-task routing signal
(kNN Spearman rho flat at 0.12-0.17 while a task-BLIND per-arm-mean predictor reached 0.205 and
overtook it). NVIDIA's "LLM Router: prefill activations" (arXiv 2603.20895v2) reports 0.8560 mean
per-model AUC from a model's own hidden states vs 0.8040 for the best of 1,300+ semantic-embedding
configs, and their encoder-target decoupling result says an OPEN-weight encoder's activations can
predict a CLOSED model's correctness. That is the exact substitution this script sets up.

What it does: one forward pass per task statement, no generation, hidden states from every layer,
MEAN-POOLED over non-pad tokens (their pooling). All layers are stored so the consumer can slice
the upper half without a re-run; last-token pooling is stored too because it is free once the
forward pass is done. Costs $0 of API spend: local weights, local compute.

Runs in a standalone venv because the world-model-optimizer venv deliberately has no torch:
    uv venv /tmp/router-act-venv --python 3.12 && VIRTUAL_ENV=/tmp/router-act-venv uv pip install torch transformers
    /tmp/router-act-venv/bin/python .agents/scripts/router_activations.py Qwen/Qwen3-0.6B

Output goes beside the cached OpenAI embeddings, OUTSIDE this repo, because it is data not code:
    ~/Documents/experientiallabs/coding-router/results/deepswe_acts_<model>.npz
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

CODING_ROUTER = pathlib.Path.home() / "Documents/experientiallabs/coding-router"
OUT_DIR = CODING_ROUTER / "results"
# Long enough for every DeepSWE instruction.md (max 5,385 chars ~ 1.4k tokens), so nothing is cut.
MAX_TOKENS = 4096


def task_texts() -> dict[str, str]:
    """Task statements via the project loader, so the boilerplate stripping matches every other run."""
    sys.path.insert(0, str(CODING_ROUTER))
    from loaders import deepswe  # noqa: PLC0415  (needs the sys.path line above)

    return deepswe.load()["text"]


def device_of() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
    texts = task_texts()
    ids = sorted(texts)
    dev = device_of()
    # bf16 for the 8B (memory), fp32 for the 0.6B (free, and avoids any MPS bf16 kernel doubt).
    dtype = torch.bfloat16 if "0.6B" not in model_id and dev != "cpu" else torch.float32
    print(f"{model_id} on {dev} dtype={dtype} over {len(ids)} tasks", flush=True)

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=dtype).to(dev).eval()
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    print(f"  {n_layers} layers x {hidden} hidden", flush=True)

    mean_pool = np.zeros((len(ids), n_layers + 1, hidden), dtype=np.float32)
    last_tok = np.zeros((len(ids), n_layers + 1, hidden), dtype=np.float32)
    n_tokens = np.zeros(len(ids), dtype=np.int32)
    t0 = time.time()
    for i, tid in enumerate(ids):
        enc = tok(texts[tid], return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
        enc = {k: v.to(dev) for k, v in enc.items()}
        n_tokens[i] = int(enc["input_ids"].shape[1])
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        # hidden_states is (embeddings, layer_1, ..., layer_n); one sequence per call so no padding.
        for li, h in enumerate(out.hidden_states):
            h32 = h[0].float()
            mean_pool[i, li] = h32.mean(dim=0).cpu().numpy()
            last_tok[i, li] = h32[-1].cpu().numpy()
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(ids)}  {time.time() - t0:.0f}s", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / f"deepswe_acts_{model_id.split('/')[-1].replace('.', '_')}.npz"
    np.savez_compressed(dest, task_ids=np.array(ids), mean_pool=mean_pool.astype(np.float16),
                        last_token=last_tok.astype(np.float16), n_tokens=n_tokens,
                        model_id=np.array(model_id))
    print(f"wrote {dest}  {dest.stat().st_size / 1e6:.1f} MB  in {time.time() - t0:.0f}s")
    print(f"prompt tokens min={n_tokens.min()} median={int(np.median(n_tokens))} max={n_tokens.max()}"
          f" (cap {MAX_TOKENS}, truncated={int((n_tokens >= MAX_TOKENS).sum())})")


if __name__ == "__main__":
    main()
