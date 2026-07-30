"""METHOD A collector: run the FORK experiment that measures P(success | prefix, arm).

The question: does the agent's execution context after K turns predict which arm should finish
the episode better than the task statement does? Our measured baseline is that the task statement
carries almost no per-task signal (kNN Spearman rho flat at 0.12-0.17 across an 8x data sweep,
while a task-BLIND per-arm-mean predictor reached 0.205 and OVERTOOK it). SWE-Router
(arXiv:2607.00053) independently derives the same failure -- kNN on text-embedding-3-large is the
WORST baseline in their table at Route-AUC 0.371 -- and their fix is a trajectory prefix, which
lifts Route-AUC 0.627 (K=0) -> 0.768 (K=1) -> 0.780 (K=2) and then DECLINES (0.750 at K=3, 0.718
at K=4). All the gain arrives in one turn, so this script runs K=1 and K=2 and nothing deeper.

WHY A FORK AND NOT A CORRELATION. Routing needs P(success | prefix, top) versus
P(success | prefix, cheap) on the SAME prefix. Two independent episodes cannot give that: they
diverge at turn 1, so any difference confounds "which arm is better here" with "which trajectory
did it happen to take". So every prefix is run once and then CONTINUED on each candidate arm from
byte-identical state.

Two probe directions share this harness:
  CHEAP-PROBE  run the cheap arm for K turns, then fork. (SWE-Router's setting.)
  TOP-PROBE    run the TOP arm for K turns, then fork. Novel, and cheap for a non-obvious reason:
               we measured episode cost to be superlinear in step index (log-log slope 1.26 over
               22,417 trials), so the top model's EARLY turns are its cheap ones.

Per task this collects 2 probes (2 turns each, snapshotted after turn 1 AND turn 2) x 3 fork arms
= 12 continuations, plus one static mid-arm episode. The two remaining statics are free: a
TOP-PROBE snapshot continued on the top arm IS a full top-arm episode, and likewise for cheap.

THE FORK IS A REAL FILESYSTEM CLONE, NOT A REPLAY. `docker commit` on the paused probe container
produces an image; each fork is a fresh container from that image. Byte-identical /app, no
re-execution of the probe's tool calls, no hash-matching needed. (The alternative the spec allowed
-- deterministically replaying the probe's tool calls into a fresh sandbox and asserting file
hashes match -- was not needed and is not used.)

E2B could not host this: DeepSWE ships one pinned Docker image per task
(public.ecr.aws/d3j8x8q7/swe-bench-202605:<ext_id>-v1.1) and E2B runs its own template registry,
so it would mean building 113 templates. Docker on the devbox also gives exact `commit` cloning,
which is the whole fork mechanism. harness/e2b_exec.py's INVARIANTS are carried over even though
its code is not: bounded concurrency, capacity retry, and a guaranteed kill in a finally.

GRADING is the publisher's own verifier, unmodified. task.toml declares
`environment_mode = "separate"`, and tests/grader.py reads exactly one input --
/logs/artifacts/model.patch -- so grading needs no image build: run a fresh container from the
task image, copy tests/ in, drop the patch at /logs/artifacts/model.patch, run /tests/test.sh,
read /logs/verifier/reward.json. Reward is GRADED f2p_passed/f2p_total, never binary: binary
overstates the arm gap ~3.5x on this very data.

Usage (on the box, inside tmux):
    python3 router_methods_a_collect.py --manifest manifest.json --out runs/ --budget 3900
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import json
import os
import pathlib
import re
import shlex
import subprocess
import threading
import time
import uuid

# --------------------------------------------------------------------------- price table
# Fetched live 2026-07-28 (OpenAI) / verified against platform.claude.com 2026-07-29 for
# claude-opus-5, which is newer than most cached docs and IS live (confirmed via models.list).
# USD per 1M tokens. Reasoning/thinking tokens bill as OUTPUT on both providers.
#
# claude-opus-5 at 5/0.50/25/6.25 is corroborated by the published DeepSWE trial table: trial
# abs-module-cache-flags__MPuAwjS reports 2,695,796 input (2,608,297 cached) + 40,951 output at
# $2.87464975, which these rates reproduce to within a plausible ~19k cache-write split.


@dataclasses.dataclass(frozen=True)
class Price:
    inp: float          # uncached input, $/1M
    cache_read: float   # cached input read, $/1M
    out: float          # output (incl. reasoning/thinking), $/1M
    cache_write: float  # 5-min cache write, $/1M


PRICES: dict[str, dict[str, Price]] = {
    "anthropic": {
        "claude-opus-5": Price(5.00, 0.50, 25.00, 6.25),
    },
    "openai": {
        # The GPT-5.6 family is the first to charge for cache writes (1.25x input).
        "gpt-5.6-luna": Price(1.00, 0.10, 6.00, 1.25),
        "gpt-5.6-terra": Price(2.50, 0.25, 15.00, 3.125),
    },
}

# Measured live 2026-07-29: all three arms accept these and emit a tool call. Note `max` IS a
# valid OpenAI effort on the gpt-5.6 family -- router/pricing.py's OPENAI_EFFORTS list predates
# that family being probed and omits it.
EFFORTS_OK = {
    ("anthropic", "claude-opus-5"): ("low", "medium", "high", "xhigh", "max"),
    ("openai", "gpt-5.6-luna"): ("none", "low", "medium", "high", "xhigh", "max"),
    ("openai", "gpt-5.6-terra"): ("none", "low", "medium", "high", "xhigh", "max"),
}


@dataclasses.dataclass(frozen=True)
class Arm:
    provider: str
    model: str
    effort: str

    @property
    def id(self) -> str:
        return f"{self.model}@{self.effort}"

    def request_kwargs(self) -> dict:
        assert self.effort in EFFORTS_OK[(self.provider, self.model)], f"bad effort {self!r}"
        if self.provider == "anthropic":
            # Opus 5 runs WITHOUT thinking if `thinking` is omitted -- it must be set explicitly.
            return {"model": self.model, "thinking": {"type": "adaptive"},
                    "output_config": {"effort": self.effort}}
        return {"model": self.model, "reasoning": {"effort": self.effort}}


TOP = Arm("anthropic", "claude-opus-5", "high")     # measured best single static, graded 0.955
MID = Arm("openai", "gpt-5.6-luna", "max")          # published mean $3.03/task, f2p 0.946
CHEAP = Arm("openai", "gpt-5.6-terra", "high")      # published mean $1.13/task, f2p 0.917
FORK_ARMS = (TOP, MID, CHEAP)
BY_ID = {a.id: a for a in FORK_ARMS}


@dataclasses.dataclass
class Usage:
    inp: int = 0
    cache_read: int = 0
    cache_write: int = 0
    out: int = 0
    reasoning: int = 0
    requests: int = 0

    def add(self, **kw: int) -> None:
        for k, v in kw.items():
            setattr(self, k, getattr(self, k) + (v or 0))

    def cost(self, arm: Arm) -> float:
        p = PRICES[arm.provider][arm.model]
        # OpenAI reports only `cached_tokens` (reads), never cache WRITES, so uncached OpenAI input
        # is billed at the cache-write rate. This is not a guess: solving the published opus-5
        # trial above for its write split gives 87,385 of 87,499 uncached tokens = 99.9% writes,
        # and reproduces $2.87464975 EXACTLY. In an agentic loop essentially every uncached input
        # token is a first touch that gets written to cache, so the write rate is the right rate.
        # It also errs in the direction that does not flatter the router: it can only over-state
        # the cheap arms' cost, i.e. under-state the saving we would claim. Anthropic reports
        # writes explicitly and is billed exactly.
        inp_rate = p.cache_write if arm.provider == "openai" else p.inp
        return (self.inp * inp_rate + self.cache_read * p.cache_read
                + self.cache_write * p.cache_write + self.out * p.out) / 1_000_000


# --------------------------------------------------------------------------- hard-won constants
# A tight max_tokens turns reasoning DEPTH into a scored TASK failure. At max_tokens=8000 we lost
# 69 of 139 runs to stop=max_tokens and it faked an entire "effort inversion" result (higher
# effort looked worse purely because it hit the wall more often). 32k was still not enough:
# adaptive thinking has no declared budget to size against, and one opus run returned
# stop=max_tokens at out_tok=32000, turns=1, with nothing written. An unused cap costs nothing --
# only generated tokens bill -- so the floor is generous on purpose. DO NOT LOWER.
MIN_MAX_TOKENS = 64_000
# 16 turns made Opus finish with no patch written. DO NOT LOWER.
MAX_TURNS = 60
K_VALUES = (1, 2)          # SWE-Router's Route-AUC peaks at K=2 and DECLINES from K=3. Never >2.
TOOL_TIMEOUT_S = 120.0
EPISODE_WALL_S = 3600.0
VERIFY_TIMEOUT_S = 1800.0  # task.toml verifier.timeout_sec
CONTAINER_CPUS = "2"       # task.toml environment.cpus
CONTAINER_MEM = "8g"       # task.toml environment.memory_mb

SYSTEM = """You are a software engineer working in a checked-out git repository at /app.

You have one tool: bash. Use it to explore the repository, edit files, and run tests.
The repository is at a base commit with no future history; there is no network access.

Work until the task is fully implemented. Prefer reading the existing code and tests before
editing. When you are done, make sure every change is saved to disk in /app.

Be efficient: batch independent shell commands into one call where you can."""

BASH_SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string", "description": "bash command, run in /app"}},
    "required": ["command"],
}
BASH_DESC = "Run a bash command in the repository at /app and return its combined stdout+stderr."
TOOL_OUT_CAP = 30_000


# --------------------------------------------------------------------------- docker layer
_sem: threading.Semaphore | None = None
_sem_lock = threading.Lock()
RUN_TAG = os.environ.get("ROUTER_A_TAG", "rtra")  # label for orphan reaping

# public.ecr.aws rate-limits anonymous pulls per IP and WILL reject a stampede: measured
# `toomanyrequests: Rate exceeded` on the 2nd of 2 back-to-back pulls. Each task's image is ~2.7GB
# and is pulled once then reused by all 15 of that task's jobs, so pulls are throttled hard and
# retried with long backoff rather than run at task concurrency.
PULL_SEM = threading.Semaphore(3)
_PULL_DELAYS = (5.0, 15.0, 45.0, 120.0, 300.0, 600.0)
_RETRYABLE_PULL = ("toomanyrequests", "rate exceeded", "429", "timeout", "temporary",
                   "connection reset", "i/o timeout", "unexpected eof")


def semaphore(limit: int) -> threading.Semaphore:
    global _sem
    with _sem_lock:
        if _sem is None:
            _sem = threading.Semaphore(limit)
    return _sem


def pull_image(image: str, log=None) -> None:
    """Pull once, retrying registry rate limits. Same shape as the E2B capacity retry this
    harness inherits its invariants from."""
    last = ""
    for delay in (*_PULL_DELAYS, None):
        with PULL_SEM:
            p = _docker("pull", "--quiet", image, timeout=3600.0, check=False)
        if p.returncode == 0:
            return
        last = (p.stderr or p.stdout or "")[:300]
        if delay is None or not any(s in last.lower() for s in _RETRYABLE_PULL):
            raise RuntimeError(f"docker pull {image} failed: {last}")
        if log:
            log(f"pull retry in {delay:.0f}s ({image[-24:]}): {last.strip()[:90]}")
        time.sleep(delay)
    raise RuntimeError(f"docker pull {image} exhausted retries: {last}")


def _docker(*args: str, timeout: float = 300.0, check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"docker {args[0]} failed rc={p.returncode}: {p.stderr[:400]}")
    return p


class Container:
    """One container for one episode. Always removed, even on error (orphans starve later runs)."""

    def __init__(self, image: str, tag: str, limit: int):
        self.image, self.tag, self.limit = image, tag, limit
        self.cid: str | None = None
        self._held = False

    def __enter__(self) -> Container:
        semaphore(self.limit).acquire()
        self._held = True
        try:
            # --network none matches task.toml (agent+verifier both no-network) and means the
            # model-authored commands cannot reach anything.
            p = _docker("run", "-d", "--rm=false", "--network", "none",
                        "--cpus", CONTAINER_CPUS, "--memory", CONTAINER_MEM,
                        "--label", f"{RUN_TAG}=1", "--label", f"{RUN_TAG}_job={self.tag}",
                        "-w", "/app", self.image, "sleep", "infinity")
            self.cid = p.stdout.strip()
        except Exception:
            semaphore(self.limit).release()
            self._held = False
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            if self.cid:
                _docker("rm", "-f", self.cid, timeout=120.0, check=False)
        except Exception:  # noqa: BLE001,S110 -- teardown must not mask the real error
            pass
        finally:
            if self._held:
                semaphore(self.limit).release()
                self._held = False

    def bash(self, command: str, timeout: float = TOOL_TIMEOUT_S) -> str:
        try:
            p = _docker("exec", "-w", "/app", self.cid, "bash", "-lc", command,
                        timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return f"[command timed out after {timeout:.0f}s and was killed]"
        out = (p.stdout or "") + (p.stderr or "")
        if p.returncode != 0:
            out += f"\n[exit code {p.returncode}]"
        return out[:TOOL_OUT_CAP] if out else "[no output]"

    def commit(self, tag: str) -> str:
        """Exact filesystem snapshot -> image. This IS the fork mechanism."""
        _docker("commit", self.cid, tag, timeout=600.0)
        return tag

    def model_patch(self, base_commit: str) -> str:
        """Diff vs base_commit with untracked files staged, which is the preimage grader.py resets
        to. Works whether or not the agent committed its work."""
        return self.bash(
            "git add -A >/dev/null 2>&1; "
            f"git diff --binary --cached {shlex.quote(base_commit)}", timeout=180.0)


# --------------------------------------------------------------------------- agent loop
@dataclasses.dataclass
class Step:
    """One provider-neutral turn. This is what makes a CROSS-MODEL fork possible at all."""
    turn: int
    text: str
    calls: list[dict]          # [{id, name, input}]
    results: list[str]         # aligned with calls


@dataclasses.dataclass
class RunOut:
    arm_id: str
    turns: int
    usage: Usage
    cost_usd: float
    wall_s: float
    stop: str                  # end_turn | max_turns | max_tokens | error | refusal | probe_done
    error: str | None
    steps: list[Step]
    native: list               # provider-native history, for a SAME-model continuation


def _thinking_stripped(native: list) -> list:
    """Drop provider-native reasoning artifacts. Used only when the continuing model differs from
    the producing model.

    Provider fact, verified against primary docs 2026-07-29: reasoning state is tied to the model
    that produced it. Anthropic thinking blocks replayed to a DIFFERENT model are dropped from the
    prompt (before pricing -- they lower usage.input_tokens rather than billing), and OpenAI
    reasoning items carry response-scoped ids that another model will reject. The same docs warn
    that stripping thinking blocks WITHIN one model's own turn causes ordering/signature 400s,
    which is why this is never applied to a same-model continuation -- there we replay verbatim.
    In practice every cross-model fork here is also cross-PROVIDER or cross-model, so the prefix
    is rebuilt from the neutral Step list instead and this is a belt-and-braces guard.
    """
    out = []
    for m in native:
        c = m.get("content")
        if isinstance(c, list):
            c = [b for b in c
                 if (b.get("type") if isinstance(b, dict) else getattr(b, "type", None))
                 not in ("thinking", "redacted_thinking")]
            if not c:
                continue
            m = {**m, "content": c}
        if isinstance(m, dict) and m.get("type") == "reasoning":
            continue
        out.append(m)
    return out


class Runner:
    """Runs an episode under ONE arm, optionally stopping after K turns, optionally resuming from
    a prefix. The scaffold, tool set, system prompt and stopping rule are identical across arms --
    that invariant is the whole experiment."""

    def __init__(self, arm: Arm, container: Container, timeout_s: float = 1800.0):
        self.arm, self.box = arm, container
        self.max_tokens = MIN_MAX_TOKENS
        if arm.provider == "anthropic":
            import anthropic
            self._cl = anthropic.Anthropic(max_retries=5, timeout=timeout_s)
        else:
            import openai
            self._cl = openai.OpenAI(max_retries=5, timeout=timeout_s)

    # ---- prefix construction -------------------------------------------------
    def _seed(self, task: str, steps: list[Step], native: list, same_model: bool) -> list:
        if not steps:
            return ([{"role": "user", "content": task}] if self.arm.provider == "anthropic"
                    else [{"role": "user", "content": task}])
        if same_model and native:
            return list(native)  # MUST be verbatim within one model's own tool-use turns
        return self._rebuild(task, steps)

    def _rebuild(self, task: str, steps: list[Step]) -> list:
        """Re-express the prefix in this provider's shape from the neutral Step list. Reasoning
        artifacts are absent by construction -- Step never carries them."""
        if self.arm.provider == "anthropic":
            msgs: list = [{"role": "user", "content": task}]
            for s in steps:
                blocks: list = []
                if s.text:
                    blocks.append({"type": "text", "text": s.text})
                for c in s.calls:
                    blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"],
                                   "input": c["input"]})
                if not blocks:
                    blocks = [{"type": "text", "text": "(continuing)"}]
                msgs.append({"role": "assistant", "content": blocks})
                if s.calls:
                    msgs.append({"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": c["id"],
                         "content": r[:TOOL_OUT_CAP]}
                        for c, r in zip(s.calls, s.results)]})
            return msgs
        hist: list = [{"role": "user", "content": task}]
        for s in steps:
            if s.text:
                hist.append({"role": "assistant", "content": s.text})
            for c in s.calls:
                hist.append({"type": "function_call", "call_id": c["id"], "name": c["name"],
                             "arguments": json.dumps(c["input"])})
            for c, r in zip(s.calls, s.results):
                hist.append({"type": "function_call_output", "call_id": c["id"],
                             "output": r[:TOOL_OUT_CAP]})
        return hist

    # ---- the loop ------------------------------------------------------------
    def run(self, task: str, *, prefix: list[Step] | None = None, native: list | None = None,
            same_model: bool = False, stop_after: int | None = None,
            deadline: float | None = None) -> RunOut:
        steps = list(prefix or [])
        msgs = self._seed(task, steps, _thinking_stripped(native or []) if not same_model
                          else (native or []), same_model)
        u = Usage()
        t0 = time.time()
        stop, err = "max_turns", None
        turn = len(steps)
        limit = MAX_TURNS if stop_after is None else min(MAX_TURNS, len(steps) + stop_after)

        while turn < limit:
            if deadline and time.time() > deadline:
                stop = "timeout"
                break
            turn += 1
            try:
                if self.arm.provider == "anthropic":
                    calls, text, sr, msgs = self._one_anthropic(msgs, u)
                else:
                    calls, text, sr, msgs = self._one_openai(msgs, u)
            except Exception as e:  # noqa: BLE001 -- an infra failure must be distinguishable
                stop, err = "error", f"{type(e).__name__}: {e}"[:600]
                turn -= 1
                break

            if sr == "refusal":
                stop = "refusal"
                break
            step = Step(turn=turn, text=text, calls=[], results=[])
            if not calls:
                steps.append(step)
                stop = "max_tokens" if sr in ("max_tokens", "incomplete") else "end_turn"
                break
            for c in calls:
                step.calls.append(c)
                step.results.append(self.box.bash(c["input"].get("command", "")))
            steps.append(step)
            msgs = self._feed(msgs, step)
            if stop_after is not None and turn >= limit:
                stop = "probe_done"
                break
        else:
            if stop_after is not None:
                stop = "probe_done"

        return RunOut(self.arm.id, turn, u, u.cost(self.arm), round(time.time() - t0, 2),
                      stop, err, steps, msgs)

    # ---- providers -----------------------------------------------------------
    def _one_anthropic(self, msgs: list, u: Usage):
        system = [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]
        r = self._cl.messages.create(
            max_tokens=self.max_tokens, system=system, messages=msgs,
            tools=[{"name": "bash", "description": BASH_DESC, "input_schema": BASH_SCHEMA}],
            **self.arm.request_kwargs())
        ru = r.usage
        u.add(inp=ru.input_tokens, out=ru.output_tokens, requests=1,
              cache_read=getattr(ru, "cache_read_input_tokens", 0) or 0,
              cache_write=getattr(ru, "cache_creation_input_tokens", 0) or 0)
        # Echo assistant content back verbatim -- thinking blocks must not be edited within a
        # model's own turn or the next request 400s.
        msgs = msgs + [{"role": "assistant", "content": r.content}]
        calls = [{"id": b.id, "name": b.name, "input": dict(b.input)}
                 for b in r.content if b.type == "tool_use"]
        text = " ".join(b.text for b in r.content if b.type == "text")[:4000]
        return calls, text, r.stop_reason, msgs

    def _one_openai(self, hist: list, u: Usage):
        r = self._cl.responses.create(
            instructions=SYSTEM, input=hist, max_output_tokens=self.max_tokens,
            tools=[{"type": "function", "name": "bash", "description": BASH_DESC,
                    "parameters": BASH_SCHEMA}],
            **self.arm.request_kwargs())
        ru = r.usage
        det = getattr(ru, "input_tokens_details", None)
        cached = getattr(det, "cached_tokens", 0) or 0
        odet = getattr(ru, "output_tokens_details", None)
        u.add(inp=max(0, ru.input_tokens - cached), cache_read=cached, out=ru.output_tokens,
              reasoning=getattr(odet, "reasoning_tokens", 0) or 0, requests=1)
        # Reasoning items are preserved verbatim; dropping them breaks the NEXT turn of the same
        # model. They are removed only when handing the prefix to a different model.
        hist = hist + [it.model_dump(exclude_none=True) for it in r.output]
        calls = []
        for c in r.output:
            if getattr(c, "type", "") == "function_call":
                try:
                    args = json.loads(c.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append({"id": c.call_id, "name": c.name, "input": args})
        return calls, (r.output_text or "")[:4000], r.status, hist

    def _feed(self, msgs: list, step: Step) -> list:
        if self.arm.provider == "anthropic":
            # ALL results for one assistant turn go back in a SINGLE user message, else the model
            # learns to stop making parallel calls.
            return msgs + [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": c["id"], "content": r[:TOOL_OUT_CAP]}
                for c, r in zip(step.calls, step.results)]}]
        return msgs + [{"type": "function_call_output", "call_id": c["id"],
                        "output": r[:TOOL_OUT_CAP]}
                       for c, r in zip(step.calls, step.results)]


# --------------------------------------------------------------------------- grading
REWARD_RE = re.compile(r"\{.*?\"f2p_total\".*?\}", re.S)


def grade(image: str, tests_dir: pathlib.Path, patch: str, tag: str, limit: int) -> dict:
    """Run the publisher's verifier unmodified in a FRESH container from the task image."""
    with Container(image, tag, limit) as box:
        box.bash("mkdir -p /tests /logs/artifacts /logs/verifier")
        for name in ("test.sh", "test.patch", "grader.py", "config.json"):
            _docker("cp", str(tests_dir / name), f"{box.cid}:/tests/{name}", timeout=120.0)
        if patch.strip():
            pf = pathlib.Path(f"/tmp/{tag}.patch")
            pf.write_text(patch)
            try:
                _docker("cp", str(pf), f"{box.cid}:/logs/artifacts/model.patch", timeout=120.0)
            finally:
                pf.unlink(missing_ok=True)
        box.bash("chmod +x /tests/test.sh")
        out = box.bash("bash /tests/test.sh 2>&1 | tail -c 4000; "
                       "echo '---REWARD---'; cat /logs/verifier/reward.json 2>/dev/null",
                       timeout=VERIFY_TIMEOUT_S)
        blob = out.rsplit("---REWARD---", 1)[-1]
        m = REWARD_RE.search(blob)
        if not m:
            return {"verifier_ok": False, "tail": out[-1500:]}
        d = json.loads(m.group(0))
        d["verifier_ok"] = True
        return d


def outcome_of(run: RunOut, patch: str, g: dict) -> str:
    """Infra failures must be excluded from the gradeable denominator; agent failures stay scored."""
    if run.stop == "error":
        return "infra_error"
    if not g.get("verifier_ok"):
        return "infra_error"
    if g.get("apply_failed"):
        return "apply_failed"
    if run.stop == "timeout":
        return "timeout"
    if run.stop == "max_tokens":
        return "cap_hit"
    if not patch.strip():
        return "no_patch"
    return "graded"


# --------------------------------------------------------------------------- spend governor
class Governor:
    """Wall-clock is usually the constraint, but this method has a hard $ cap, so cost is tracked
    centrally and jobs stop being ISSUED once the soft cap is crossed. Never kills work in flight:
    a half-finished task is wasted money."""

    def __init__(self, cap: float):
        self.cap, self.spent = cap, 0.0
        self._lk = threading.Lock()

    def add(self, usd: float) -> float:
        with self._lk:
            self.spent += usd
            return self.spent

    def ok(self) -> bool:
        with self._lk:
            return self.spent < self.cap


# --------------------------------------------------------------------------- job graph
def atomic_write(path: pathlib.Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(obj))
    tmp.rename(path)  # rename is atomic on the same fs: never a half-written checkpoint


def probe_key(task: str, direction: str) -> str:
    return f"{task}__probe_{direction}"


def fork_key(task: str, direction: str, k: int, arm_id: str) -> str:
    return f"{task}__{direction}_K{k}__{arm_id.replace('/', '_')}"


def static_key(task: str, arm_id: str) -> str:
    return f"{task}__static__{arm_id.replace('/', '_')}"


def run_task(t: dict, tests_root: pathlib.Path, out: pathlib.Path, gov: Governor,
             limit: int, log) -> None:
    """Probe both directions, snapshot at K=1 and K=2, fork every snapshot onto every arm."""
    image, task_id, prompt, base = t["image"], t["task_id"], t["prompt"], t["base_commit"]
    tests = tests_root / task_id / "tests"
    snaps: dict[tuple[str, int], str] = {}
    prefixes: dict[tuple[str, int], tuple[list[Step], list, str]] = {}
    made: list[str] = []
    # CUMULATIVE probe cost to reach turn k. This is charged to the router: the probe is part of
    # the routing decision, and the literature's omission of it is the thing we are correcting.
    probe_cost: dict[tuple[str, int], float] = {}

    try:
        pull_image(image, log)   # once per task; all 15 jobs reuse it
        for direction, arm in (("cheap", CHEAP), ("top", TOP)):
            ck = out / f"{probe_key(task_id, direction)}.json"
            tagbase = f"{RUN_TAG}/{task_id}-{direction}-{uuid.uuid4().hex[:6]}".lower()
            with Container(image, f"probe-{task_id}-{direction}", limit) as box:
                r = Runner(arm, box)
                steps: list[Step] = []
                native: list = []
                cum = 0.0
                for k in K_VALUES:
                    if not gov.ok():
                        log(f"BUDGET stop before probe {task_id}/{direction} K={k}")
                        return
                    rr = r.run(prompt, prefix=steps, native=native, same_model=True,
                               stop_after=1, deadline=time.time() + EPISODE_WALL_S)
                    gov.add(rr.cost_usd)
                    cum += rr.cost_usd
                    steps, native = rr.steps, rr.native
                    if rr.stop == "error":
                        log(f"PROBE-ERR {task_id}/{direction} K={k}: {rr.error}")
                        return
                    img = box.commit(f"{tagbase}-k{k}")
                    made.append(img)
                    snaps[(direction, k)] = img
                    probe_cost[(direction, k)] = cum
                    prefixes[(direction, k)] = ([dataclasses.replace(s) for s in steps],
                                                list(native), arm.id)
                    log(f"snap {task_id}/{direction} K={k} turns={len(steps)} "
                        f"${rr.cost_usd:.3f} cum=${cum:.3f} spent=${gov.spent:.0f}")
                atomic_write(ck, {
                    "task": task_id, "direction": direction, "arm": arm.id,
                    "probe_cost_usd_by_k": {str(k): probe_cost.get((direction, k))
                                            for k in K_VALUES},
                    "steps": [dataclasses.asdict(s) for s in steps]})

        jobs: list[tuple] = []
        for (direction, k), img in snaps.items():
            pre, native, producer = prefixes[(direction, k)]
            for arm in FORK_ARMS:
                jobs.append((fork_key(task_id, direction, k, arm.id), arm, img,
                             pre[:k], native, producer, direction, k))
        jobs.append((static_key(task_id, MID.id), MID, image, [], [], None, "static", 0))

        for key, arm, img, pre, native, producer, direction, k in jobs:
            ck = out / f"{key}.json"
            if ck.exists():
                continue
            if not gov.ok():
                log(f"BUDGET stop before {key}")
                return
            same = (producer == arm.id)
            with Container(img, key, limit) as box:
                rr = Runner(arm, box).run(prompt, prefix=pre, native=native if same else [],
                                          same_model=same,
                                          deadline=time.time() + EPISODE_WALL_S)
                gov.add(rr.cost_usd)
                patch = "" if rr.stop == "error" else box.model_patch(base)
            g = ({} if rr.stop == "error"
                 else grade(image, tests, patch, f"v-{key}"[:60], limit))
            rec = {
                "key": key, "task": task_id, "variant": direction, "k": k, "arm": arm.id,
                "probe_arm": producer, "fork_same_model": same,
                "turns": rr.turns, "stop": rr.stop, "error": rr.error,
                "outcome": outcome_of(rr, patch, g),
                "cost_usd": rr.cost_usd, "wall_s": rr.wall_s,
                # Cumulative probe spend to reach this prefix. total_cost_usd is what the router
                # is actually charged for choosing this arm on this task.
                "probe_cost_usd": probe_cost.get((direction, k), 0.0),
                "total_cost_usd": rr.cost_usd + probe_cost.get((direction, k), 0.0),
                "usage": dataclasses.asdict(rr.usage),
                "f2p_passed": g.get("f2p_passed"), "f2p_total": g.get("f2p_total"),
                "graded": (g["f2p_passed"] / g["f2p_total"]
                           if g.get("f2p_total") else None),
                "reward_binary": g.get("reward"), "apply_failed": g.get("apply_failed", 0),
                "patch_bytes": len(patch), "verifier_ok": g.get("verifier_ok", False),
                "verifier_tail": g.get("tail"),
                "spent_after": round(gov.spent, 4), "ts": time.time(),
            }
            atomic_write(ck, rec)
            log(f"done {key} outcome={rec['outcome']} graded={rec['graded']} "
                f"turns={rr.turns} ${rr.cost_usd:.3f} spent=${gov.spent:.0f}")
    finally:
        for img in made:
            _docker("rmi", "-f", img, timeout=180.0, check=False)
        _docker("rmi", "-f", image, timeout=300.0, check=False)  # reclaim ~2.7GB/task


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tests-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=3900.0)
    ap.add_argument("--tasks", type=int, default=0, help="0 = walk the whole manifest")
    ap.add_argument("--concurrent-tasks", type=int, default=8)
    ap.add_argument("--max-containers", type=int, default=26)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(pathlib.Path(a.manifest).read_text())
    if a.tasks:
        manifest = manifest[:a.tasks]
    tests_root = pathlib.Path(a.tests_root)

    # Resume: re-bank what previous runs already spent so the cap is global, not per-process.
    gov = Governor(a.budget)
    for f in out.glob("*.json"):
        try:
            gov.add(json.loads(f.read_text()).get("cost_usd", 0.0) or 0.0)
        except Exception:  # noqa: BLE001,S112
            continue
    lk = threading.Lock()

    def log(msg: str) -> None:
        with lk:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"METHOD A collector | {len(manifest)} tasks | budget ${a.budget:.0f} "
        f"| already banked ${gov.spent:.2f} | arms top={TOP.id} mid={MID.id} cheap={CHEAP.id}")
    log(f"K={K_VALUES} max_turns={MAX_TURNS} max_tokens={MIN_MAX_TOKENS} "
        f"containers<={a.max_containers} tasks<={a.concurrent_tasks}")
    semaphore(a.max_containers)

    done = 0
    with cf.ThreadPoolExecutor(max_workers=a.concurrent_tasks) as ex:
        futs = {ex.submit(run_task, t, tests_root, out, gov, a.max_containers, log): t
                for t in manifest}
        for fu in cf.as_completed(futs):
            done += 1
            tid = futs[fu]["task_id"]
            try:
                fu.result()
            except Exception as e:  # noqa: BLE001 -- one bad task must not end the run
                log(f"TASK-FAIL {tid}: {type(e).__name__}: {e}")
            log(f"PROGRESS tasks={done}/{len(manifest)} "
                f"episodes={len(list(out.glob('*.json')))} spent=${gov.spent:.2f}")
    log(f"FINISHED spent=${gov.spent:.2f} episodes={len(list(out.glob('*.json')))}")


if __name__ == "__main__":
    main()
