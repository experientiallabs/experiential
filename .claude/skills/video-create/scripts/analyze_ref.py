#!/usr/bin/env python3
"""Measure a reference video so the brief is built on numbers, not impressions.

Usage: analyze_ref.py <video> <out_dir> [--whisper-model PATH] [--label NAME]

Writes to <out_dir>/<label>/:
  probe.json         fps, resolution, duration
  contact.jpg        1 frame every 2s, 6 columns
  cuts.txt           cut times (mean-abs-diff spikes on downscaled grey frames)
  loudness.txt       momentary LUFS every 2s + integrated LUFS
  report.md          cuts per 10s, silence windows, BPM estimate, loudness arc,
                     longest uncut takes, transcript with timestamps (if whisper found)
Needs: ffmpeg/ffprobe, numpy. Optional: whisper-cli + a ggml model.
"""

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

np: Any = importlib.import_module("numpy")


def sh(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command and capture its text output."""
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def probe(path: str) -> dict[str, Any]:
    """Return fps, resolution, duration and whether the file has audio."""
    r = sh(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height,r_frame_rate",
            "-of",
            "json",
            path,
        ]
    )
    j = json.loads(r.stdout)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    num, den = map(int, v["r_frame_rate"].split("/"))
    return {
        "width": v["width"],
        "height": v["height"],
        "fps": round(num / den, 3),
        "duration": float(j["format"]["duration"]),
        "has_audio": any(s["codec_type"] == "audio" for s in j["streams"]),
    }


def cuts(path: str, fps: float) -> list[float]:
    """Detect cuts as mean-abs-diff spikes (3x local median) on 64x36 grey frames."""
    w, h = 64, 36
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:{h},format=gray"]
        + ["-f", "rawvideo", "-"],
        capture_output=True,
        check=False,
    )
    fr = np.frombuffer(r.stdout, np.uint8)
    n = len(fr) // (w * h)
    fr = fr[: n * w * h].reshape(n, h, w).astype(np.float32)
    d = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))
    out: list[float] = []
    for i in range(len(d)):
        lo, hi = max(0, i - 8), min(len(d), i + 9)
        med = float(np.median(np.r_[d[lo:i], d[i + 1 : hi]])) if hi - lo > 1 else 0.0
        if d[i] > 12 and d[i] > 3 * max(med, 1.0):
            t = (i + 1) / fps
            if not out or t - out[-1] > 0.25:
                out.append(round(t, 3))
    return out


def loudness(wav: str) -> tuple[dict[int, float], float | None]:
    """Return momentary LUFS per 2 s bucket and the integrated LUFS."""
    r = sh(
        ["ffmpeg", "-nostats", "-i", wav, "-af"]
        + ["ebur128=metadata=1,ametadata=print:key=lavfi.r128.M", "-f", "null", "-"]
    )
    arc: dict[int, float] = {}
    t = 0.0
    for line in r.stderr.splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            t = float(m.group(1))
        m = re.search(r"lavfi\.r128\.M=(-?[\d.inf]+)", line)
        if m:
            b = int(t // 2) * 2
            if b not in arc:
                try:
                    arc[b] = float(m.group(1))
                except ValueError:
                    arc[b] = -120.0
    r2 = sh(["ffmpeg", "-nostats", "-i", wav, "-af", "ebur128", "-f", "null", "-"])
    ii = re.findall(r"I:\s+(-?[\d.]+) LUFS", r2.stderr)
    return arc, (float(ii[-1]) if ii else None)


def bpm(wav: str) -> float:
    """Estimate tempo by onset autocorrelation (may be half or double; confirm from cut spacing)."""
    with wave.open(wav) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)
        sr, hop = w.getframerate(), 512
    n = len(x) // hop
    e = np.log1p((x[: n * hop].reshape(n, hop) ** 2).sum(1))
    on = np.maximum(0, np.diff(e))
    on -= on.mean()
    ac = np.correlate(on, on, "full")[len(on) - 1 :]
    lags = np.arange(1, len(ac)) / (sr / hop)
    b = 60 / lags
    m = (b > 70) & (b < 170)
    return round(float(b[m][np.argmax(ac[1:][m])]), 1)


def main() -> None:
    """Measure one reference and write its report."""
    a = sys.argv[1:]
    if len(a) < 2:
        sys.stdout.write(str(__doc__))
        sys.exit(1)
    src, outroot = a[0], a[1]
    model = a[a.index("--whisper-model") + 1] if "--whisper-model" in a else None
    label = a[a.index("--label") + 1] if "--label" in a else Path(src).stem[:40]
    out = Path(outroot) / label
    out.mkdir(parents=True, exist_ok=True)

    p = probe(src)
    (out / "probe.json").write_text(json.dumps(p, indent=2))
    rows = max(1, int(p["duration"] / 12) + 1)
    sh(
        ["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", f"fps=1/2,scale=480:-1,tile=6x{rows}"]
        + ["-frames:v", "1", str(out / "contact.jpg")]
    )

    c = cuts(src, p["fps"])
    (out / "cuts.txt").write_text("\n".join(map(str, c)))
    buckets = [0] * (int(p["duration"] // 10) + 1)
    for t in c:
        buckets[int(t // 10)] += 1
    bounds = [0.0, *c, p["duration"]]
    spans = ((bounds[i + 1] - bounds[i], bounds[i]) for i in range(len(bounds) - 1))
    takes = sorted(spans, reverse=True)[:5]

    rep = [
        f"# Reference analysis: {label}",
        "",
        f"- {p['width']}x{p['height']} @ {p['fps']}fps, {p['duration']:.2f}s, "
        f"audio={p['has_audio']}",
        f"- cuts detected: {len(c)} (mean shot {p['duration'] / max(1, len(c) + 1):.2f}s)",
        f"- cuts per 10s: `{','.join(map(str, buckets))}`",
        "- longest uncut takes: " + ", ".join(f"{d:.1f}s @ {s:.1f}s" for d, s in takes),
    ]

    if p["has_audio"]:
        wav = str(out / "audio16k.wav")
        sh(["ffmpeg", "-v", "error", "-y", "-i", src, "-ar", "16000", "-ac", "1", wav])
        arc, integ = loudness(wav)
        arc_txt = "\n".join(f"{k}\t{v:.1f}" for k, v in sorted(arc.items()))
        (out / "loudness.txt").write_text(arc_txt + f"\nintegrated\t{integ}\n")
        sil = [k for k, v in sorted(arc.items()) if v < -35 and k > 0]
        rep += [
            f"- integrated loudness: {integ} LUFS",
            f"- estimated tempo: ~{bpm(wav)} BPM (onset autocorrelation; may be half/double)",
            f"- quiet windows (< -35 LUFS momentary): {', '.join(f'{s}s' for s in sil) or 'none'}",
            "- loudness arc (t:LUFS every 2s): `"
            + " ".join(f"{k}:{v:.0f}" for k, v in sorted(arc.items()))
            + "`",
        ]
        wc = shutil.which("whisper-cli")
        if wc and model and os.path.exists(model):
            r = sh([wc, "-m", model, "-f", wav, "-np"])
            lines = [ln for ln in r.stdout.splitlines() if ln.startswith("[")]
            (out / "transcript.txt").write_text("\n".join(lines))
            rep += ["", "## Transcript (label each line narrator / archival / on-screen)"]
            rep += ["```", *lines, "```"]
        else:
            rep += ["", "_No transcript: pass --whisper-model path/to/ggml-*.bin._"]
    (out / "report.md").write_text("\n".join(rep) + "\n")
    sys.stdout.write("\n".join(rep[:12]) + f"\n\n-> {out}/report.md\n")


if __name__ == "__main__":
    main()
