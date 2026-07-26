"""H100 latency leg (track C1): batched GPU throughput for the surviving learned/scored compressors.

Measures llmlingua2-fixed-threshold and selective-context-absolute end to end (score +
select + rebuild) on the round 0 audit transcripts, at request batch sizes 1/4/16, on
one GPU. Reports s/10k raw tokens, p50 per-request latency, throughput, and amortized
$/10k tokens at the box's retail hourly rate. Also checks GPU determinism (same batch
twice) and fp16-vs-fp32 keep/drop agreement, the lit review's flagged risk.

Runs on the GPU box (self-contained: needs torch + transformers + the compression
module file + transcripts JSONL next to it):

    python bench_compressor_gpu.py --device cuda:0 --out gpu-bench.json

The vLLM tenant on the box owns ~90GB/GPU; this bench needs <3GB and must not disturb
it (we cap batch chunks accordingly).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import statistics
import sys
import time
from pathlib import Path

log = logging.getLogger("gpu_bench")

HERE = Path(__file__).resolve().parent
LLMLINGUA2_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
BOX_USD_PER_HOUR = 19.544  # Standard_NC80adis_H100_v5, centralindia, retail consumption
GPUS_ON_BOX = 2


def _load_compression_module():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("compression", HERE / "compression.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compression"] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load_compression_module()


class BatchedLLMLingua2:
    """LLMLingua-2 keep-probs with cross-request chunk batching (fixed threshold 0.5)."""

    def __init__(self, device: str, dtype_name: str) -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        self.dtype = torch.float16 if dtype_name == "fp16" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(LLMLINGUA2_MODEL)
        self.model = (
            AutoModelForTokenClassification.from_pretrained(LLMLINGUA2_MODEL)
            .to(device=device, dtype=self.dtype)
            .eval()
        )
        id2label = self.model.config.id2label
        keep_ids = [i for i, lab in id2label.items() if str(lab).lower() in ("1", "preserve")]
        self.keep_idx = keep_ids[0] if keep_ids else 1

    def compress_requests(self, requests: list[list[str]], threshold: float = 0.5) -> list[str]:
        """Compress a batch of transcripts; all their chunks share forward passes."""
        window = 400
        chunks: list[tuple[int, int, int, list[str]]] = []  # (req, turn, start, words)
        words_per_turn: dict[tuple[int, int], list[str]] = {}
        for ri, turns in enumerate(requests):
            for ti, turn in enumerate(turns):
                ws = C.split_words(turn)
                words_per_turn[(ri, ti)] = ws
                stripped = [w.strip() for w in ws]
                for start in range(0, len(stripped), window):
                    chunks.append((ri, ti, start, stripped[start : start + window]))
        probs: dict[tuple[int, int], dict[int, float]] = {k: {} for k in words_per_turn}
        sub = 32  # chunks per forward; keeps activations <3GB next to the vLLM tenant
        for i in range(0, len(chunks), sub):
            batch = chunks[i : i + sub]
            enc = self.tokenizer(
                [c[3] for c in batch],
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits
            p_keep = self.torch.softmax(logits.float(), dim=-1)[:, :, self.keep_idx].cpu()
            for bi, (ri, ti, start, words) in enumerate(batch):
                word_ids = enc.word_ids(bi)
                sums = [0.0] * len(words)
                counts = [0] * len(words)
                for pos, wid in enumerate(word_ids):
                    if wid is None:
                        continue
                    sums[wid] += float(p_keep[bi, pos])
                    counts[wid] += 1
                for wi in range(len(words)):
                    probs[(ri, ti)][start + wi] = sums[wi] / counts[wi] if counts[wi] else 0.0
        out: list[str] = []
        for ri, turns in enumerate(requests):
            kept_turns = []
            for ti in range(len(turns)):
                ws = words_per_turn[(ri, ti)]
                p = probs[(ri, ti)]
                kept_turns.append("".join(w for wi, w in enumerate(ws) if p.get(wi, 0.0) >= threshold))
            out.append(C.join_turns(kept_turns))
        return out


class BatchedGpt2SelfInfo:
    """Selective-Context absolute threshold with cross-request unit batching."""

    def __init__(self, device: str, dtype_name: str) -> None:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        self.torch = torch
        self.device = device
        self.dtype = torch.float16 if dtype_name == "fp16" else torch.float32
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            GPT2LMHeadModel.from_pretrained("gpt2").to(device=device, dtype=self.dtype).eval()
        )

    def unit_bits(self, units: list[str]) -> list[float]:
        import torch.nn.functional as functional

        results = [0.0] * len(units)
        todo = [(i, u.strip()) for i, u in enumerate(units) if u.strip()]
        # Small sub-batches + fused cross_entropy: the vLLM tenant leaves <5GB free, and a
        # materialized [B, T, vocab] log_softmax at B=64 needs ~6.6GB on its own.
        sub = 8
        for s in range(0, len(todo), sub):
            batch = todo[s : s + sub]
            enc = self.tokenizer(
                [t for _, t in batch],
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            ids, mask = enc["input_ids"], enc["attention_mask"]
            with self.torch.no_grad():
                logits = self.model(input_ids=ids, attention_mask=mask).logits
            tgt = ids[:, 1:]
            vocab = logits.shape[-1]
            nll = functional.cross_entropy(
                logits[:, :-1].reshape(-1, vocab).float(),
                tgt.reshape(-1),
                reduction="none",
            ).view(tgt.shape)
            valid = mask[:, 1:].bool()
            for bi, (orig_i, _) in enumerate(batch):
                v = nll[bi][valid[bi]]
                results[orig_i] = (
                    20.0 if v.numel() == 0 else float(v.mean().item() / 0.6931471805599453)
                )
            del logits, nll
        return results

    def compress_requests(self, requests: list[list[str]], threshold: float) -> list[str]:
        per_req_units = [C.split_units(C.join_turns(turns)) for turns in requests]
        flat = [u for units in per_req_units for u in units]
        bits = self.unit_bits(flat)
        out: list[str] = []
        pos = 0
        for units in per_req_units:
            scores = bits[pos : pos + len(units)]
            pos += len(units)
            out.append("".join(u for u, b in zip(units, scores) if b >= threshold))
        return out


def bench(  # noqa: ANN201
    name: str,
    compress_requests,  # noqa: ANN001
    transcripts: list[list[str]],
    raw_tokens: list[int],
    batch_size: int,
    torch,  # noqa: ANN001
    device: str,
):
    # Warm up (kernel selection, allocator) on the first batch, untimed.
    compress_requests(transcripts[:batch_size])
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    lat: list[float] = []
    outputs: list[str] = []
    total_tokens = 0
    t_all = time.perf_counter()
    for i in range(0, len(transcripts), batch_size):
        group = transcripts[i : i + batch_size]
        t0 = time.perf_counter()
        outs = compress_requests(group)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        lat.extend([dt] * len(group))  # every request in the group waits for the group
        outputs.extend(outs)
        total_tokens += sum(raw_tokens[i : i + batch_size])
    wall = time.perf_counter() - t_all
    s_per_10k = wall / total_tokens * 10_000
    usd_hr_gpu = BOX_USD_PER_HOUR / GPUS_ON_BOX
    return {
        "method": name,
        "batch_size": batch_size,
        "wall_s": round(wall, 3),
        "raw_tokens": total_tokens,
        "s_per_10k_tok": round(s_per_10k, 4),
        "request_latency_p50_s": round(statistics.median(lat), 4),
        "throughput_tok_s": round(total_tokens / wall),
        "usd_per_10k_tok_fullbox": round(s_per_10k * BOX_USD_PER_HOUR / 3600, 6),
        "usd_per_10k_tok_per_gpu": round(s_per_10k * usd_hr_gpu / 3600, 6),
        "outputs_sha": __import__("hashlib").sha256("".join(outputs).encode()).hexdigest()[:12],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--transcripts", default=str(HERE / "audit-transcripts.jsonl"))
    ap.add_argument("--out", default=str(HERE / "gpu-bench.json"))
    args = ap.parse_args()
    import torch
    from transformers import GPT2TokenizerFast

    transcripts = [json.loads(ln)["turns"] for ln in Path(args.transcripts).open()]
    counter = GPT2TokenizerFast.from_pretrained("gpt2")
    raw_tokens = [len(counter(C.join_turns(t))["input_ids"]) for t in transcripts]
    log.info("%d transcripts, %d raw tokens total", len(transcripts), sum(raw_tokens))

    results = []
    for dtype_name in ("fp32", "fp16"):
        ll2 = BatchedLLMLingua2(args.device, dtype_name)
        si = BatchedGpt2SelfInfo(args.device, dtype_name)
        si_threshold = 6.5  # representative absolute bits threshold; ratio is not the point here
        for batch_size in (1, 4, 16):
            r1 = bench(
                f"llmlingua2-fixed-threshold-{dtype_name}",
                lambda g: ll2.compress_requests(g),
                transcripts,
                raw_tokens,
                batch_size,
                torch,
                args.device,
            )
            r2 = bench(
                f"selective-context-absolute-{dtype_name}",
                lambda g: si.compress_requests(g, si_threshold),
                transcripts,
                raw_tokens,
                batch_size,
                torch,
                args.device,
            )
            results.extend([r1, r2])
            log.info("%s", r1)
            log.info("%s", r2)
        del ll2, si
        torch.cuda.empty_cache()

    # Determinism: same batch twice, byte-identical (fp32 and fp16 separately).
    for dtype_name in ("fp32", "fp16"):
        ll2 = BatchedLLMLingua2(args.device, dtype_name)
        a = ll2.compress_requests(transcripts[:8])
        b = ll2.compress_requests(transcripts[:8])
        results.append(
            {
                "check": f"gpu-determinism-llmlingua2-{dtype_name}",
                "identical": a == b,
            }
        )
        del ll2
        torch.cuda.empty_cache()
    # fp16 vs fp32 agreement (keep/drop flips near the threshold).
    a32 = BatchedLLMLingua2(args.device, "fp32").compress_requests(transcripts[:16])
    torch.cuda.empty_cache()
    a16 = BatchedLLMLingua2(args.device, "fp16").compress_requests(transcripts[:16])
    same = sum(1 for x, y in zip(a32, a16) if x == y)
    results.append({"check": "fp16-vs-fp32-llmlingua2", "identical_outputs": f"{same}/16"})

    Path(args.out).write_text(json.dumps(results, indent=2))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
