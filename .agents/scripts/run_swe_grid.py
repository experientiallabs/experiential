"""Resumable runner for the REAL SWE-bench grid behind the product's swe-bench default.

The bench-defaults program buys one real-episode outcome matrix per benchmark. This is the
SWE-bench half: the pinned instance set x the candidate pool x N episodes, every cell a real
mini-swe-agent episode inside that instance's real SWE-bench Docker image, scored by the
official SWE-bench test-suite verifier. There is no world-model leg here, by design: the
matrix is `real_episode` provenance end to end.

WHY A SCRIPT AND NOT `wmo optimize route sweep`: the sweep path measures candidates against a
built world model (`WorldModelEnv`), which is the wrong environment for this run. SWE-bench's
environment is a real repo container, and its judge is a deterministic test suite, neither of
which the sweep path can reach. What this script owns is therefore only the outer loop and
durability; every model call goes through the PUBLIC provider layer (`wmo.providers.pool`) and
every price comes from the pool file, never from a harness cost field.

HOW THE HARNESS REACHES OUR MODELS. mini-swe-agent talks to an OpenAI-compatible base URL
through litellm, so this runner starts `wmo.distill.tau2_proxy.EpisodeProxy` (already generic
over a provider, already OpenAI-shaped) and registers ONE ALIAS PER CELL. The alias is what
carries cell identity across the HTTP boundary, so concurrent cells never share usage, and the
recorded tokens are exactly the tokens that cell's episode sampled. Two consequences worth
stating plainly:

- The agent protocol is the benchmark's CANONICAL one: mini-swe-agent's own `swebench.yaml`,
  with the `bash` tool called as a tool. The textbased backticks variant was tried first and
  abandoned on measurement: without a stop sequence (which `Provider.complete` cannot express)
  haiku-4-5 hallucinated a whole multi-turn transcript in one reply, ran to the 8192-token
  output cap on step 1, and produced no parseable action. Tool calls stop natively, and the
  canonical protocol is the one published SWE-bench numbers use, so the only thing standing
  between these solve rates and that literature is this program's step pin (see
  `PROGRAM_STEP_LIMIT`), which is stated rather than absorbed.
- That protocol needs `complete_chat` from every candidate, which Anthropic direct did not have
  (the same missing capability that makes the serving endpoint refuse Anthropic tool calls). It
  was implemented for this run: `wmo/providers/_anthropic_chat.py` plus
  `AnthropicProvider.complete_chat`, with prompt-cache breakpoints, so the `fable-5` anchor is
  priced with the cache credit a production deployment would get instead of paying full input
  rate for a replayed transcript on every step.

DURABILITY, which is the only thing here that is not a library call:

- PER-CELL PERSISTENCE. Every cell owns a directory holding its agent run, its verifier report
  and its `outcome.json`. A cell whose `outcome.json` loads clean is skipped, so re-running
  continues where the last process stopped and several processes can share one grid directory
  (disjoint `--models` never touch the same file, and no lock is needed).
- APPEND-ONLY LEDGER. One newline-terminated JSON line per finished cell, written with a
  single `open(..., "a")`, so concurrent processes cannot lose each other's bills.
- COHORT. The grid directory carries the tip sha, the pool file, the pin, the step limit and
  the episode count the cells were bought under. Cells measured under two harnesses are not
  one matrix, so a new cohort gets a new directory.
- UNSCORED IS EVIDENCE. A provider fault, a proxy fault or a cell that outran
  `--cell-deadline-s` lands as `reward=None` plus `error`, never as reward 0: an infrastructure
  failure is not a verdict from the test suite. An agent that finished and submitted nothing IS
  a verdict (empty patch, reward 0), and the two are recorded differently.

A DEADLINE EXISTS HERE, unlike the tau grid runner, which deliberately has none. The reason is
this box: SWE-bench publishes x86_64 images and this machine is arm64, so every container step
runs under emulation and a single pathological cell can eat a whole night's wall clock. The
deadline is wall-clock only, it is recorded, and it never converts into a reward.

    # validation smoke: 2 pinned instances, 1 episode, 2 cheap candidates
    uv run python .agents/scripts/run_swe_grid.py --smoke --models haiku-4-5,qwen3.5-9b

    # the buy (staggered behind the sibling lanes on shared provider limits)
    uv run python .agents/scripts/run_swe_grid.py --episodes 2 --concurrency 6

Credentials come from the ambient environment exactly as the product reads them (the pool
entry's `api_key_env` per candidate). Nothing here prints a secret.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import threading
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from llm_waterfall.types import ChatRequest, ChatResponse
from pydantic import BaseModel, JsonValue

from wmo.distill.tau2_proxy import EpisodeProxy
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import TokenUsage, ToolCallingProvider
from wmo.providers.pool import ModelPool, PoolEntry, load_pool, pool_provider

logger = logging.getLogger("run_swe_grid")

MAIN_CHECKOUT = Path("/Users/silen/Desktop/Projects/world-model-harness")
"""The main checkout: the corpus, the harness venv and the artifact root all live there."""

PIN_SALT = "swe-defaults-v1:"
"""Salt for the deterministic instance pin. Documented in the findings; never change silently."""

DATASET = "princeton-nlp/SWE-Bench_Verified"
SPLIT = "test"

SHIPPED_STEP_LIMIT = 250
"""The step limit in mini-swe-agent's own `swebench.yaml`, recorded for the record."""

PROGRAM_STEP_LIMIT = 75
"""This program's step pin, and a documented DEVIATION from the shipped 250.

Measured reason, not a budget guess: the validation probe (haiku-4-5 on django__django-15280)
spent 217 API calls and 15.4 minutes of wall clock to submit one patch, because a weak model
thrashes until the limit stops it. At 640 cells on an emulated box that tail alone is more than
a night of wall clock. 75 keeps every episode that solves its instance in a normal number of
steps intact and truncates only the thrash, it is applied UNIFORMLY to all candidates, and it is
the reason these solve rates are not comparable to published SWE-bench numbers (which is stated
in the findings, not buried here).
"""

CANONICAL_TEMPERATURE = 0.0
"""Sampling pin, passed to the harness so every candidate is asked for the same thing.

Two backends do not honor it and that is deliberate upstream, not a gap here: Anthropic's
provider never forwards sampling params (Claude 4.8+/5 reject them, adaptive thinking is the
default), and any candidate whose API refuses the field has it dropped by the harness's
`drop_params`. So this is the pin where it is honored and the recorded caveat where it is not.
"""


class CellRecord(BaseModel):
    """Everything the proxy observed for one cell, accumulated across its calls."""

    usage: TokenUsage = TokenUsage()
    call_seconds: list[float] = []
    replies: list[str] = []
    provider_error: str | None = None


@dataclass(frozen=True)
class CellKey:
    """One (candidate, instance, episode) cell of the grid."""

    model: str
    instance_id: str
    episode: int

    @property
    def slug(self) -> str:
        """Filesystem- and litellm-safe handle for this cell."""
        return f"{self.model}--{self.instance_id}--ep{self.episode}"


class _RecordingChatProvider:
    """One cell's provider: the pool candidate's own `complete_chat`, metered per call.

    Satisfies `wmo.providers.base.ToolCallingProvider`, which is all `EpisodeProxy` requires.
    Everything about the request shape belongs to the backend; this only records what the cell
    sampled (tokens including the cache split, wall seconds, reply text) so the row can be priced
    by the pool entry rather than by a harness cost field.
    """

    def __init__(
        self,
        entry: PoolEntry,
        provider: ToolCallingProvider,
        record: CellRecord,
        *,
        cell_label: str = "",
    ) -> None:
        self._entry = entry
        self._provider = provider
        self._record = record
        self._label = cell_label or entry.name
        self._lock = threading.Lock()

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """One agent step: sample the candidate, record what it cost, pass the response back."""
        started = time.monotonic()
        try:
            response = self._provider.complete_chat(request)
        except Exception as exc:
            with self._lock:
                # First fault wins: it is the one that ended the episode.
                if self._record.provider_error is None:
                    self._record.provider_error = f"{type(exc).__name__}: {exc}"
            raise
        elapsed = time.monotonic() - started
        call_usage = _usage_of(response)
        text = _reply_text(response)
        with self._lock:
            total = self._record.usage
            self._record.usage = TokenUsage(
                input_tokens=total.input_tokens + call_usage.input_tokens,
                output_tokens=total.output_tokens + call_usage.output_tokens,
                cached_input_tokens=total.cached_input_tokens + call_usage.cached_input_tokens,
                cache_write_input_tokens=(
                    total.cache_write_input_tokens + call_usage.cache_write_input_tokens
                ),
            )
            self._record.call_seconds.append(elapsed)
            self._record.replies.append(text)
            calls = len(self._record.call_seconds)
            running = self._entry.cost_usd(self._record.usage)
        # A step-level heartbeat, because an emulated SWE-bench episode is long and otherwise
        # silent: without it a stalled cell and a working cell look identical for many minutes.
        logger.info(
            "%s step %d: in=%d (cached %d) out=%d %.1fs cell_cost=$%.4f",
            self._label,
            calls,
            call_usage.input_tokens,
            call_usage.cached_input_tokens,
            call_usage.output_tokens,
            elapsed,
            running,
        )
        return response


def _usage_of(response: ChatResponse) -> TokenUsage:
    """The call's usage on `TokenUsage`'s cached-as-subset contract.

    `ChatResponse.token_usage()` projects only the two totals, and the cache split is exactly
    what cache-adjusted pricing needs, so the provider-carried cache counters are read here.
    Backends that report no cache tokens leave the split at zero, which prices the whole prompt
    at the full input rate (never silently free).
    """
    if response.usage is None:
        return TokenUsage()
    extra = response.usage.model_dump()
    read = extra.get("cache_read_input_tokens")
    write = extra.get("cache_creation_input_tokens")
    if not isinstance(read, int):
        read = extra.get("cached_tokens") if isinstance(extra.get("cached_tokens"), int) else 0
    if not isinstance(write, int):
        write = 0
    return TokenUsage(
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        cached_input_tokens=int(read or 0),
        cache_write_input_tokens=int(write),
    )


def _reply_text(response: ChatResponse) -> str:
    """The assistant text of a response, tool calls rendered so a reader can see the action."""
    if not response.choices:
        return ""
    message = response.choices[0].message
    parts: list[str] = []
    content = message.content
    if isinstance(content, str) and content:
        parts.append(content)
    for call in message.tool_calls or []:
        parts.append(f"{call.function.name}({call.function.arguments})")
    return "\n".join(parts)


def pinned_instances(corpus: Path, count: int) -> list[str]:
    """The deterministic instance pin: `count` instances of `corpus`, chosen by salted hash.

    Ordering by `sha256(PIN_SALT + instance_id)` makes the cut reproducible on any machine and
    independent of corpus order, and unstratified so nothing can be quietly cherry-picked. The
    resulting repo mix is reported in the findings rather than engineered here.
    """
    instance_ids: set[str] = set()
    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            span = json.loads(line)
            for attribute in span.get("attributes", []):
                if attribute.get("key") != "wmh.trace.metadata":
                    continue
                metadata = json.loads(attribute.get("value", {}).get("stringValue", "{}"))
                instance_id = metadata.get("instance_id")
                if isinstance(instance_id, str) and instance_id:
                    instance_ids.add(instance_id)
    ranked = sorted(
        instance_ids,
        key=lambda iid: (hashlib.sha256((PIN_SALT + iid).encode("utf-8")).hexdigest(), iid),
    )
    if len(ranked) < count:
        raise ValueError(f"{corpus} carries only {len(ranked)} instances; {count} were pinned")
    return ranked[:count]


def _read_patch(agent_dir: Path, instance_id: str) -> str | None:
    """The patch mini-swe-agent recorded for `instance_id`, or None if it recorded no row."""
    preds = agent_dir / "preds.json"
    if not preds.exists():
        return None
    try:
        rows = json.loads(preds.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    row = rows.get(instance_id)
    if not isinstance(row, dict):
        return None
    patch = row.get("model_patch")
    return patch if isinstance(patch, str) else ""


def _read_trajectory(agent_dir: Path, instance_id: str) -> dict[str, JsonValue]:
    """The instance's trajectory JSON, or an empty mapping when the run never wrote one."""
    candidates = sorted(agent_dir.rglob(f"{instance_id}.traj.json"))
    if not candidates:
        return {}
    try:
        loaded = json.loads(candidates[0].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _agent_steps(trajectory: dict[str, JsonValue]) -> int:
    """Shell commands the episode issued, counted from its recorded messages."""
    messages = trajectory.get("messages")
    if not isinstance(messages, list):
        return 0
    return sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")


def _stop_reason(trajectory: dict[str, JsonValue]) -> str:
    """The harness's own exit status for the episode, verbatim when it recorded one."""
    info = trajectory.get("info")
    if isinstance(info, dict):
        status = info.get("exit_status")
        if isinstance(status, str) and status:
            return status
    return "unknown"


def run_agent(
    cell: CellKey,
    *,
    alias: str,
    cell_dir: Path,
    proxy_base_url: str,
    swe_dir: Path,
    step_limit: int,
    deadline_s: float,
) -> tuple[str, str | None]:
    """Run one real mini-swe-agent episode in the instance's container.

    Returns:
        The episode's stdout+stderr tail and, when the run could not produce an episode at all,
        the reason. A nonzero exit with a recorded patch is NOT an error: the harness exits
        nonzero for an unsubmitted episode, which is a real outcome the verifier still scores.
    """
    agent_dir = cell_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(swe_dir / ".venv" / "bin" / "python"),
        "-m",
        "minisweagent.run.benchmarks.swebench",
        "--subset",
        "verified",
        "--split",
        SPLIT,
        "--filter",
        f"^{cell.instance_id}$",
        "--redo-existing",
        # The benchmark's canonical agent config, unedited apart from the pins below.
        "-c",
        "swebench.yaml",
        "-c",
        f"agent.step_limit={step_limit}",
        # litellm's own cost meter cannot price a proxy alias, so the cost limit is disarmed
        # here and the run's spend is metered where it is real: the recorded provider usage,
        # priced by the pool entry.
        "-c",
        "agent.cost_limit=0",
        "-c",
        f"model.model_kwargs.api_base={proxy_base_url}",
        "-c",
        "model.model_kwargs.api_key=proxy-local",
        "-c",
        f"model.model_kwargs.temperature={CANONICAL_TEMPERATURE}",
        "-m",
        f"openai/{alias}",
        "-o",
        str(agent_dir),
        "-w",
        "1",
    ]
    environment = dict(os.environ)
    environment["MSWEA_SILENT_STARTUP"] = "1"
    # A global cost limit would abort on litellm's zero-priced proxy calls; metering is ours.
    environment.pop("MSWEA_GLOBAL_COST_LIMIT", None)
    # litellm cannot price a per-cell proxy alias, and the harness turns that lookup failure
    # into a RuntimeError that ends the episode after step 1 (measured). Cost tracking here is
    # provider-side by design (recorded usage x the pool entry's row), so the harness's own
    # meter is switched off rather than fed a fake price that could reach a report.
    environment["MSWEA_COST_TRACKING"] = "ignore_errors"
    try:
        finished = subprocess.run(
            command,
            cwd=swe_dir,
            env=environment,
            capture_output=True,
            text=True,
            timeout=deadline_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "", f"cell outran its {deadline_s:.0f}s wall deadline"
    tail = (finished.stdout or "")[-4000:] + (finished.stderr or "")[-4000:]
    (cell_dir / "agent.log").write_text(tail, encoding="utf-8")
    return tail, None


def verify_patch(
    cell: CellKey,
    *,
    alias: str,
    cell_dir: Path,
    patch: str,
    swe_dir: Path,
    timeout_s: float,
) -> tuple[bool | None, str]:
    """Score one patch with the official SWE-bench verifier (a deterministic test suite).

    Returns:
        `(resolved, detail)`. `resolved` is None when the verifier itself could not reach a
        verdict, which is an infrastructure outcome and must not be read as a failed patch.
    """
    verify_dir = cell_dir / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    predictions = verify_dir / "preds.json"
    predictions.write_text(
        json.dumps(
            {
                cell.instance_id: {
                    "instance_id": cell.instance_id,
                    "model_name_or_path": alias,
                    "model_patch": patch,
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    run_id = f"v{uuid.uuid4().hex[:8]}"
    command = [
        str(swe_dir / ".venv" / "bin" / "python"),
        "-m",
        "swebench.harness.run_evaluation",
        "-d",
        DATASET,
        "-s",
        SPLIT,
        "-i",
        cell.instance_id,
        "-p",
        str(predictions),
        "-id",
        run_id,
        "--max_workers",
        "1",
        # `env` (the harness default) DELETES the instance images this grid pulled once and
        # reuses for every cell; `instance` keeps them.
        "--cache_level",
        "instance",
        "--namespace",
        "swebench",
    ]
    try:
        finished = subprocess.run(
            command,
            cwd=verify_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, f"verifier outran its {timeout_s:.0f}s deadline"
    (verify_dir / "verify.log").write_text(
        (finished.stdout or "")[-8000:] + (finished.stderr or "")[-8000:], encoding="utf-8"
    )
    reports = sorted(verify_dir.glob(f"*.{run_id}.json"))
    if not reports:
        return None, "verifier wrote no report"
    try:
        report = json.loads(reports[0].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, "verifier report was not readable JSON"

    def named_in(bucket: str) -> bool:
        ids = report.get(bucket)
        return isinstance(ids, list) and cell.instance_id in ids

    # Order matters, and the buckets are NOT nested: an empty patch appears in
    # `empty_patch_ids` ONLY, never in `unresolved_ids` (measured on swebench 4.1.0, schema
    # version 2). Reading it as "named in no outcome list" is what made a model that exhausted
    # its step budget look UNSCORED instead of failed, which would have quietly deleted the
    # weakest candidates' worst rows and flattered exactly the models a savings headline leans
    # on. An infrastructure error and an incomplete evaluation stay unscored; everything the
    # verifier actually adjudicated becomes a reward.
    if named_in("error_ids"):
        return None, "the verifier reported an evaluation error for this instance"
    if named_in("incomplete_ids"):
        return None, "the verifier could not complete this instance's evaluation"
    if named_in("resolved_ids"):
        return True, "resolved by the test suite"
    if named_in("empty_patch_ids"):
        return False, "no diff submitted (the verifier's empty-patch bucket)"
    if named_in("unresolved_ids"):
        return False, "unresolved by the test suite"
    return None, "the verifier report named this instance in no outcome list"


def run_cell(
    cell: CellKey,
    *,
    entry: PoolEntry,
    task: str,
    grid_dir: Path,
    proxy: EpisodeProxy,
    swe_dir: Path,
    step_limit: int,
    deadline_s: float,
    verify_timeout_s: float,
) -> ScenarioOutcome:
    """Measure one cell end to end: real episode, real verifier, priced by the pool entry."""
    cell_dir = grid_dir / "cells" / cell.model / cell.instance_id / f"ep{cell.episode}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    alias = f"swecell-{uuid.uuid4().hex[:10]}"
    record = CellRecord()
    candidate = pool_provider(entry)
    if not isinstance(candidate, ToolCallingProvider):
        raise TypeError(
            f"pool model '{entry.name}' (kind={entry.kind.value}) has no complete_chat, so it "
            "cannot run a tool-calling benchmark harness"
        )
    provider = _RecordingChatProvider(entry, candidate, record, cell_label=cell.slug)
    proxy.register(alias, provider)
    started = time.monotonic()
    try:
        _, agent_error = run_agent(
            cell,
            alias=alias,
            cell_dir=cell_dir,
            proxy_base_url=proxy.base_url,
            swe_dir=swe_dir,
            step_limit=step_limit,
            deadline_s=deadline_s,
        )
    finally:
        proxy.release(alias)
    agent_seconds = time.monotonic() - started
    trajectory = _read_trajectory(cell_dir / "agent", cell.instance_id)
    patch = _read_patch(cell_dir / "agent", cell.instance_id)

    outcome = ScenarioOutcome(
        scenario_id=cell.instance_id,
        task=task,
        model=cell.model,
        episode=cell.episode,
        steps=_agent_steps(trajectory),
        stop_reason=_stop_reason(trajectory),
        usage=record.usage,
        cost_usd=entry.cost_usd(record.usage),
        call_seconds=record.call_seconds,
        replies=record.replies,
    )

    infra_error = record.provider_error or agent_error
    if infra_error is not None:
        # The episode never completed for reasons that are not the model's capability.
        outcome.error = infra_error
        outcome.critique = "unscored: the episode did not complete"
    elif patch is None:
        outcome.error = "the harness recorded no prediction row for this instance"
        outcome.critique = "unscored: no prediction row"
    elif not patch.strip():
        # An episode that ended with no diff cannot resolve anything, so the verifier is not
        # worth 26 seconds of emulated container per cell to be told so. This is the ordinary
        # shape of a candidate that exhausted the uniform step budget (stop_reason
        # LimitsExceeded), and it is a FAILED attempt, not an infrastructure fault.
        outcome.reward = 0.0
        outcome.success = False
        outcome.critique = f"no diff submitted (harness exit: {outcome.stop_reason})"
    else:
        resolved, detail = verify_patch(
            cell,
            alias=alias,
            cell_dir=cell_dir,
            patch=patch,
            swe_dir=swe_dir,
            timeout_s=verify_timeout_s,
        )
        if resolved is None:
            outcome.error = f"verifier: {detail}"
            outcome.critique = "unscored: the verifier reached no verdict"
        else:
            outcome.reward = 1.0 if resolved else 0.0
            outcome.success = resolved
            outcome.critique = f"swe-bench test suite: {detail}"

    (cell_dir / "outcome.json").write_text(outcome.model_dump_json(indent=2), encoding="utf-8")
    (cell_dir / "cell.json").write_text(
        json.dumps(
            {
                "model": cell.model,
                "instance_id": cell.instance_id,
                "episode": cell.episode,
                "alias": alias,
                "agent_seconds": agent_seconds,
                "patch_chars": len(patch or ""),
                "provenance": "real_episode",
                "judge": "swe-bench test suite (deterministic verifier)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return outcome


def load_cell(grid_dir: Path, cell: CellKey) -> ScenarioOutcome | None:
    """A previously measured cell, or None when it must still be bought."""
    path = grid_dir / "cells" / cell.model / cell.instance_id / f"ep{cell.episode}" / "outcome.json"
    if not path.exists():
        return None
    try:
        return ScenarioOutcome.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        logger.warning("cell %s has an unreadable outcome.json; re-measuring it", cell.slug)
        return None


def append_ledger(grid_dir: Path, payload: dict[str, JsonValue]) -> None:
    """Append one bill line. One open-append-close per line, so concurrent writers are safe."""
    with (grid_dir / "ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _tip_sha() -> str:
    """The main checkout's current commit, the cohort's harness stamp."""
    finished = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=MAIN_CHECKOUT,
        capture_output=True,
        text=True,
        check=False,
    )
    return finished.stdout.strip()[:12] or "unknown"


def write_matrix(grid_dir: Path, pool: ModelPool, outcomes: list[ScenarioOutcome]) -> Path:
    """Merge every measured cell into one `OutcomeMatrix` on disk."""
    matrix = OutcomeMatrix(pool=pool.models, outcomes=outcomes)
    path = grid_dir / "matrix.json"
    matrix.save(path)
    return path


def collect_outcomes(grid_dir: Path, cells: Sequence[CellKey]) -> list[ScenarioOutcome]:
    """Every cell of `cells` that is already on disk, in cell order."""
    found: list[ScenarioOutcome] = []
    for cell in cells:
        outcome = load_cell(grid_dir, cell)
        if outcome is not None:
            found.append(outcome)
    return found


def problem_statements(swe_dir: Path, instance_ids: Sequence[str]) -> dict[str, str]:
    """Each pinned instance's problem statement, read through the harness venv's dataset copy.

    The statement is the scenario's `task`, and the routing fitters read it, so it comes from
    the dataset the episodes actually ran against rather than from the trace corpus.
    """
    script = (
        "import json,sys\n"
        "from datasets import load_dataset\n"
        f"ds=load_dataset({DATASET!r},split={SPLIT!r})\n"
        "want=set(json.loads(sys.argv[1]))\n"
        "out={r['instance_id']:r['problem_statement'] for r in ds "
        "if r['instance_id'] in want}\n"
        "sys.stdout.write(json.dumps(out))\n"
    )
    finished = subprocess.run(
        [str(swe_dir / ".venv" / "bin" / "python"), "-c", script, json.dumps(list(instance_ids))],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = json.loads(finished.stdout)
    return {str(k): str(v) for k, v in loaded.items()}


def main(argv: Sequence[str] | None = None) -> int:
    """Run (or resume) the grid and merge every measured cell into one matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="", help="grid directory name (default: swe-<tip>)")
    parser.add_argument("--models", default="", help="comma-separated pool names (default: all)")
    parser.add_argument("--scenarios", type=int, default=20, help="pinned instances to measure")
    parser.add_argument("--episodes", type=int, default=2, help="episodes per (instance, model)")
    parser.add_argument("--concurrency", type=int, default=6, help="cells in flight")
    parser.add_argument("--step-limit", type=int, default=PROGRAM_STEP_LIMIT)
    parser.add_argument("--cell-deadline-s", type=float, default=3600.0)
    parser.add_argument("--verify-timeout-s", type=float, default=1800.0)
    parser.add_argument("--pool", default=str(MAIN_CHECKOUT / ".wmo" / "jt" / "pool-17.toml"))
    parser.add_argument(
        "--swe-dir", default=str(MAIN_CHECKOUT / "packages" / "environment-capture" / "swe-bench")
    )
    parser.add_argument(
        "--grid-root", default=str(MAIN_CHECKOUT / ".wmo" / "jt" / "bench-defaults")
    )
    parser.add_argument("--smoke", action="store_true", help="2 instances x 1 episode")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    swe_dir = Path(args.swe_dir)
    pool_path = Path(args.pool)
    full_pool = load_pool(pool_path)
    enabled = full_pool.enabled_models()
    if args.models:
        wanted = [name.strip() for name in args.models.split(",") if name.strip()]
        entries = [full_pool.entry(name) for name in wanted]
    else:
        entries = enabled
    pool = ModelPool(models=entries)

    scenarios = 2 if args.smoke else args.scenarios
    episodes = 1 if args.smoke else args.episodes
    instance_ids = pinned_instances(swe_dir / "traces.otel.jsonl", scenarios)
    tasks = problem_statements(swe_dir, instance_ids)

    tip = _tip_sha()
    cohort = args.cohort or (f"swe-smoke-{tip}" if args.smoke else f"swe-{tip}")
    grid_dir = Path(args.grid_root) / "swe" / cohort
    (grid_dir / "cells").mkdir(parents=True, exist_ok=True)
    (grid_dir / "pool.toml").write_text(pool_path.read_text(encoding="utf-8"), encoding="utf-8")
    (grid_dir / "cohort.json").write_text(
        json.dumps(
            {
                "cohort": cohort,
                "tip": tip,
                "pool_file": str(pool_path),
                "models": [entry.name for entry in pool.models],
                "instance_ids": instance_ids,
                "pin_rule": f"sha256({PIN_SALT} + instance_id), ascending, first {scenarios}",
                "episodes": episodes,
                "step_limit": args.step_limit,
                "shipped_step_limit": SHIPPED_STEP_LIMIT,
                "temperature": CANONICAL_TEMPERATURE,
                "dataset": DATASET,
                "split": SPLIT,
                "agent_config": "swebench.yaml (canonical, tool-calling)",
                "harness": "mini-swe-agent 2.4.6, docker environment (x86_64 under emulation)",
                "judge": "swe-bench test suite (deterministic verifier)",
                "provenance": "real_episode",
                "started_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Episode-major, then instance, then model: a run that is cut short still leaves COMPLETE
    # paired rows (every candidate measured on the instances it reached) rather than a few
    # candidates measured on everything, which no paired statistic could use.
    cells = [
        CellKey(model=entry.name, instance_id=instance_id, episode=episode)
        for episode in range(episodes)
        for instance_id in instance_ids
        for entry in pool.models
    ]
    todo = [cell for cell in cells if load_cell(grid_dir, cell) is None]
    logger.info(
        "cohort %s: %d cells total, %d already measured, %d to buy",
        cohort,
        len(cells),
        len(cells) - len(todo),
        len(todo),
    )

    proxy = EpisodeProxy()
    proxy.start()
    spent = 0.0
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pooled:
            futures = {
                pooled.submit(
                    run_cell,
                    cell,
                    entry=full_pool.entry(cell.model),
                    task=tasks.get(cell.instance_id, ""),
                    grid_dir=grid_dir,
                    proxy=proxy,
                    swe_dir=swe_dir,
                    step_limit=args.step_limit,
                    deadline_s=args.cell_deadline_s,
                    verify_timeout_s=args.verify_timeout_s,
                ): cell
                for cell in todo
            }
            done = 0
            for future in as_completed(futures):
                cell = futures[future]
                done += 1
                try:
                    outcome = future.result()
                except Exception as exc:  # noqa: BLE001 - one cell must not end the grid
                    logger.error("cell %s raised: %s", cell.slug, exc)
                    append_ledger(
                        grid_dir,
                        {
                            "ts": datetime.now(UTC).isoformat(),
                            "cell": cell.slug,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    continue
                spent += outcome.cost_usd
                if not outcome.scored:
                    # Loud at the moment it happens, not at analysis time: unscored cells are
                    # excluded from every statistic, so a systematic reason (one backend
                    # faulting, the verifier wedged) has to be visible while the grid can still
                    # be stopped and fixed rather than discovered as a hole in the matrix.
                    logger.warning("UNSCORED %s: %s", cell.slug, outcome.error)
                append_ledger(
                    grid_dir,
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "cell": cell.slug,
                        "model": cell.model,
                        "instance_id": cell.instance_id,
                        "episode": cell.episode,
                        "reward": outcome.reward,
                        "steps": outcome.steps,
                        "stop_reason": outcome.stop_reason,
                        "input_tokens": outcome.usage.input_tokens,
                        "cached_input_tokens": outcome.usage.cached_input_tokens,
                        "output_tokens": outcome.usage.output_tokens,
                        "cost_usd": outcome.cost_usd,
                        "model_seconds": sum(outcome.call_seconds),
                        "error": outcome.error,
                    },
                )
                logger.info(
                    "[%d/%d] %s reward=%s steps=%d cost=$%.4f cohort_spend=$%.2f",
                    done,
                    len(todo),
                    cell.slug,
                    outcome.reward,
                    outcome.steps,
                    outcome.cost_usd,
                    spent,
                )
    finally:
        proxy.stop()

    measured = collect_outcomes(grid_dir, cells)
    path = write_matrix(grid_dir, pool, measured)
    scored = [o for o in measured if o.scored]
    logger.info(
        "matrix %s: %d cells on disk, %d scored, %d unscored, cohort spend $%.2f",
        path,
        len(measured),
        len(scored),
        len(measured) - len(scored),
        sum(o.cost_usd for o in measured),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
