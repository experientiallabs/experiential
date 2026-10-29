#!/usr/bin/env python3
"""Exact word times by forced alignment (torchaudio MMS_FA) of a KNOWN transcript.

Use this instead of whisper word timestamps, which drift 0.3–0.8 s.

Usage: fa_align.py lines.json out.json
  lines.json: {"L1": {"src": "path/to/media", "a": 61.5, "b": 64.3,
                      "text": "how do i go off the beaten path"}, ...}
    a, b: a generous window (±0.3 s) around the line, in source seconds.
    text: the verified words, lowercase, no punctuation, numbers spelled out.
  out.json: {"L1": {"src": ..., "words": [[word, t0, t1], ...]}, ...} in source seconds.

Verify any disputed boundary by transcribing the gap alone. A word the aligner forced into
silence is a sign the text doesn't match the audio in that window.
Needs: ffmpeg, numpy, torch, torchaudio (the MMS_FA model downloads on first use).
"""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

np: Any = importlib.import_module("numpy")
torch: Any = importlib.import_module("torch")
torchaudio: Any = importlib.import_module("torchaudio")


def load(src: str, a: float, b: float, sr: int) -> bytes:
    """Decode [a, b] seconds of src to mono float32 PCM bytes at sr."""
    cmd = ["ffmpeg", "-v", "error", "-ss", str(a), "-to", str(b), "-i", src]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    return subprocess.run(cmd, capture_output=True, check=False).stdout


def main() -> None:
    """Align every line in lines.json and write word times."""
    if len(sys.argv) != 3:
        sys.stdout.write(str(__doc__))
        sys.exit(1)
    bundle = torchaudio.pipelines.MMS_FA
    model, tok, aligner = bundle.get_model(), bundle.get_tokenizer(), bundle.get_aligner()
    sr = bundle.sample_rate
    lines = json.loads(Path(sys.argv[1]).read_text())
    out: dict[str, Any] = {}
    for key, line in lines.items():
        pcm = np.frombuffer(load(line["src"], line["a"], line["b"], sr), np.float32).copy()
        wav = torch.from_numpy(pcm)[None]
        words = line["text"].split()
        with torch.inference_mode():
            em, _ = model(wav)
            spans = aligner(em[0], tok(words))
        ratio = wav.shape[1] / em.shape[1] / sr
        times = [
            [w, round(line["a"] + sp[0].start * ratio, 3), round(line["a"] + sp[-1].end * ratio, 3)]
            for w, sp in zip(words, spans, strict=True)
        ]
        out[key] = {"src": line["src"], "words": times}
        sys.stdout.write(key + " " + " ".join(f"{w}@{t0:.2f}" for w, t0, _ in times) + "\n")
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
