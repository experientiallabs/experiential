"""Print one plain-English status line for the active distill run.

Format:
  training step 4 of 6 | 6 of 8 complete, 2 CUT OFF | 26k tokens of reasoning each
  | student-vs-teacher gap 0.125 (want it falling) | only 80% of tokens scored
  | spent $1.41 | ~26m to go  <wandb url>

printed on one line. Every field is read from the run's own artifacts: the
metrics row for counts and the gap, the spend ledger for cost, the harbor step
directory mtimes for the ETA, and the console log for the wandb run URL.

Usage: uv run python .agents/distill/status_line.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

RUNS_DIR = Path(".wmh/distill-runs")
CONFIG_DIR = Path(".agents/distill")


def _running_runs() -> list[tuple[str, Path]]:
    """(config name, run dir) for every live distill process, read from its OWN argv.

    The run dir MUST come from the process's `--run-dir`, not from the config name. Deriving it
    from the name silently reported a DEAD run as live: a fresh `super-anchor-v2` run using
    `distill-super-anchor.toml` resolved to the stale `.wmh/distill-runs/super-anchor` directory
    (which exists, so the old lookup preferred it) and the status line published that finished
    run's final failing step -- "step 11 of 45 | 32 CUT OFF" -- as if it were happening now.
    A status line that invents a state is worse than no status line.
    """
    # `pgrep -af` prints argv on Linux but NOT on macOS (`-a` is a GNU extension; BSD pgrep
    # silently prints bare pids), so resolve argv per pid with ps instead of trusting pgrep.
    found = subprocess.run(
        ["pgrep", "-f", "--", "--distill-config"], capture_output=True, text=True, check=False
    )
    runs: list[tuple[str, Path]] = []
    for pid in found.stdout.split():
        described = subprocess.run(
            ["ps", "-o", "command=", "-p", pid], capture_output=True, text=True, check=False
        )
        argv = described.stdout.split()
        name, run_dir = None, None
        for i, token in enumerate(argv[:-1]):
            if token == "--distill-config":
                name = Path(argv[i + 1]).stem.replace("distill-", "")
            elif token == "--run-dir":
                run_dir = Path(argv[i + 1])
        # Other lanes share this machine and this CLI: a sibling agent's run (e.g. under
        # `.wmh/xtoken-runs/`) matches the same pgrep and would be reported here as though it
        # were ours. Reporting another lane's numbers as our own is worse than reporting
        # nothing, so only runs under OUR runs dir count.
        if run_dir is not None and RUNS_DIR not in run_dir.parents:
            continue
        # One launch shows up as several pids (the `uv` wrapper and the python child both carry
        # the same argv), so key on the run dir: one run, one line.
        if name and run_dir and all(run_dir != seen for _, seen in runs):
            runs.append((name, run_dir))
    return runs


def _spend(run_dir: Path) -> float:
    try:
        return float(json.loads((run_dir / "spend.json").read_text()).get("total_usd", 0.0))
    except (OSError, ValueError):
        return 0.0


def _last_row(run_dir: Path) -> dict[str, object]:
    try:
        lines = [
            line for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()
        ]
    except OSError:
        return {}
    for line in reversed(lines):
        row = json.loads(line)
        if row.get("phase") != "warmup":
            return row
    return {}


def _total_steps(name: str) -> int:
    try:
        text = (CONFIG_DIR / f"distill-{name}.toml").read_text()
    except OSError:
        return 0
    match = re.search(r"^steps = (\d+)", text, re.M)
    return int(match.group(1)) if match else 0


def _eta_minutes(run_dir: Path, done: int, total: int) -> float | None:
    dirs = sorted(glob.glob(str(run_dir / "harbor" / "step-*")))
    if len(dirs) < 2 or done >= total:
        return None
    stamps = sorted(os.path.getmtime(d) for d in dirs)
    per_step = (stamps[-1] - stamps[0]) / 60 / max(len(dirs) - 1, 1)
    return (total - done) * per_step


def _stall_warning(run_dir: Path) -> str:
    """`STALLED Nm` when the run dir has stopped gaining files for long enough to be suspicious.

    Catches the SILENT class of failure, and only that class. A wedged Tinker session stops
    writing metrics rows entirely, so the status line keeps republishing the last one and a dead
    run is indistinguishable from a slow one -- that cost 30+ minutes of a live run before anyone
    noticed. Freshness is the missing signal because it is orthogonal to every layer that can
    lie: the SDK reported no error, E2B reported 1000+ free sandboxes, and the process stayed
    alive.

    Verified against both recorded failures rather than assumed: the wedged run reports
    `STALLED 165m (budget 54m)`, and the shim-overload run correctly reports NOTHING -- it was
    emitting artifacts continuously while losing 92% of episodes to `Server disconnected`. That
    is the LOUD class, and it belongs to `scaffold_loss_rate` and the stop-reason breakdown, not
    here. Do not widen this to try to catch it; a freshness check that fires on a busy-but-failing
    run is just noise.

    Threshold is 2.5x the observed median step duration (not a constant): steps here range from
    ~5 to ~45 minutes depending on model, concurrency and task mix, so any fixed minute count is
    either deafening or useless. Falls back to 45 minutes before two steps exist to time.
    """
    newest = 0.0
    for path in run_dir.rglob("*"):
        try:
            if path.is_file():
                newest = max(newest, path.stat().st_mtime)
        except OSError:  # noqa: PERF203 - a file vanishing mid-walk is not an error here
            continue
    if not newest:
        return ""
    idle_min = (time.time() - newest) / 60

    dirs = sorted(glob.glob(str(run_dir / "harbor" / "step-*")))
    stamps = sorted(os.path.getmtime(d) for d in dirs)
    gaps = [(b - a) / 60 for a, b in zip(stamps, stamps[1:], strict=False)]
    budget = 2.5 * sorted(gaps)[len(gaps) // 2] if gaps else 45.0
    if idle_min <= budget:
        return ""
    return f" | STALLED {int(idle_min)}m no new files (budget {int(budget)}m)"


def _wandb_url(run_dir: Path) -> str:
    """The run's wandb URL, taken from the run dir itself rather than a console log.

    wandb names its local directory `wandb/run-<stamp>-<run_id>`, which is present from the moment
    tracking starts and does not depend on where the launcher happened to redirect stdout. The old
    version scraped `<run_dir>-console.log`, a path nothing actually writes, so the URL was always
    missing.
    """
    stamped = sorted(run_dir.glob("wandb/run-*-*"))
    if stamped:
        run_id = stamped[-1].name.rsplit("-", 1)[-1]
        return f"https://wandb.ai/kfallah/wmh-distill/runs/{run_id}"
    log = run_dir.parent / f"{run_dir.name}-console.log"
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return ""
    urls = re.findall(r"https://wandb\.ai/\S+/runs/\w+", text)
    return urls[-1] if urls else ""


def _pre_training_phase(run_dir: Path) -> str:
    """What a run is doing before its first training step writes a metrics row.

    Baselines take a while (the Ultra teacher over 17 tasks x 3 attempts), and reporting
    "step 0 of 45 ... finishing" through that window is worse than saying nothing: it reads as a
    stalled or nearly-done run. Name the phase and the evidence instead.
    """
    reports = sorted((run_dir / "evals").glob("*.json")) if (run_dir / "evals").is_dir() else []
    done = [r.stem for r in reports]
    if not done:
        return "measuring the teacher baseline (Ultra 550B, 17 tasks x 3) - no training step yet"
    return f"baselines done: {', '.join(done)} - waiting on the first training step"


def _tokens(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count // 1000}k"
    return str(count)


def _duration(minutes: float) -> str:
    if minutes >= 60:
        return f"{int(minutes // 60)}h{int(minutes % 60):02d}m"
    return f"{int(minutes)}m"


def _line(name: str, run_dir: Path) -> str:
    row = _last_row(run_dir)
    total = _total_steps(name)
    if not row:
        url = _wandb_url(run_dir)
        line = (
            f"{_pre_training_phase(run_dir)} | spent ${_spend(run_dir):.2f}"
            f"{_stall_warning(run_dir)}"
        )
        return f"{line}  {url}" if url else line
    done = (int(row.get("step") or 0) + 1) if row else 0

    trials = int(row.get("trials") or 0)
    empty = int(row.get("empty_span_trials") or 0)
    finished = trials - empty
    episodes = f"{finished} of {trials} complete"
    if empty:
        episodes += f", {empty} CUT OFF"

    loss_tokens = int(row.get("loss_tokens") or 0)
    context_tokens = int(row.get("context_tokens") or 0)
    per_episode = loss_tokens // finished if finished else 0
    reasoning = f"{_tokens(per_episode)} tokens of reasoning each"

    total_tokens = loss_tokens + context_tokens
    scored_pct = (100 * loss_tokens // total_tokens) if total_tokens else 0
    scored = f"only {scored_pct}% of tokens scored"

    kl = row.get("reverse_kl_per_token")
    gap = (
        f"student-vs-teacher gap {float(kl):.3f} (want it falling)"
        if isinstance(kl, (int, float))
        else "gap not measured yet"
    )

    eta = _eta_minutes(run_dir, done, total)
    left = f"~{_duration(eta)} to go" if eta else "finishing"
    url = _wandb_url(run_dir)

    parts = [
        f"training step {done} of {total}",
        episodes,
        reasoning,
        gap,
        scored,
        f"spent ${_spend(run_dir):.2f}",
        left,
    ]
    line = " | ".join(parts) + _stall_warning(run_dir)
    return f"{line}  {url}" if url else line


def main() -> None:
    alive = _running_runs()
    if not alive:
        spent = sum(_spend(d) for d in RUNS_DIR.glob("*") if d.is_dir())
        print(f"no training running | spent ${spent:.2f} total across all runs")
        return
    for name, run_dir in alive:
        prefix = f"[{run_dir.name.replace('super-', '')}] " if len(alive) > 1 else ""
        print(prefix + _line(name, run_dir))


if __name__ == "__main__":
    main()
