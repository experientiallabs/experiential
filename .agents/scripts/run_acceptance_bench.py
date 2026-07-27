"""Compaction acceptance benchmark: (compressor config, model) grid through the production seam.

Runs each arm through `evaluate_pool(compression=...)`, the D-COMPRESS integration seam on main,
so accuracy numbers are measured through the same wrapper stack production serves through (the
#259 lesson). One arm = one CompressionConfig applied to the pool models over the financebench-s80
scenario set (the exact 80 scenario ids of the routing cohort matrix), 2 episodes each, max_steps
16, judge pinned to Opus 4.8.

Arms: off / llmlingua2-endpoint / truncate-protect-task / control-random / control-truncate. The
learned arm is the team's H100 endpoint (`wmo.optimize.compression_endpoint`), so nothing here
loads a model and torch is not needed anywhere.

Modes:
    --dry-run   No episodes. Registers every arm, replays the s80 matrix transcripts through each
                arm's compressor offline, and prints achieved token ratios + the matched control
                aggressiveness + the cost projection table (the master-approval artifact). The
                endpoint arm does hit the live endpoint here (pennies).
    --probe N   Measures what the projection guesses. Runs N episodes of the `off` arm only,
                spread across the models, and reports measured candidate $/ep per model, measured
                env+judge $/ep, and the RECOMPUTED full-grid total from those measurements.
                Writes to its own rows file, never to a per-arm one.
    (live)      The grid. Requires --cap-usd (the master-approved ALL-IN cap). Meters candidate,
                compressor, and env+judge spend as it goes and halts at the cap. Resumable: rows
                JSONL per arm, existing (scenario, model, episode) cells skipped.

    uv run python .agents/scripts/run_acceptance_bench.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from wmo.config import load_config, load_env_file
from wmo.engine.world_model import WorldModel
from wmo.env.base import Env, WorldModelEnv
from wmo.env.closed_loop import evaluate_pool, scenario_id
from wmo.env.llm_agent import DEFAULT_HISTORY_CHARS
from wmo.env.scenarios import Scenario, scenarios_from_traces, tools_hint_from_traces
from wmo.ingest import get_adapter
from wmo.optimize.compression import (
    CompressionConfig,
    get_compressor,
    register_compressor,
    servable_compressor,
)
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import Provider, ProviderConfig, ProviderKind
from wmo.providers.pool import (
    DEFAULT_POOL_PATH,
    ModelPool,
    PoolEntry,
    load_pool,
    pool_provider,
)
from wmo.providers.registry import get_provider
from wmo.research.compressors import RandomRemovalCompressor, TruncateProtectTaskCompressor
from wmo.tracking import Phase, RunRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for _noisy in ("httpx", "httpcore", "urllib3", "botocore", "anthropic", "openai"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
log = logging.getLogger("acceptance_bench")

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path.home() / "Desktop/Projects/wmh-compression-data"
ROUTING_MATRICES = Path.home() / "Desktop/Projects/wmh-routing-data/matrices"
OUT_DIR = DATA_ROOT / "matrices"
BUNDLE = REPO / "packages/environment-capture/financebench"
MODEL_DIR = BUNDLE / "models/financebench"
TRACES = BUNDLE / "traces.otel.jsonl"
COHORT_MATRIX = ROUTING_MATRICES / "financebench-s80_matrix.json"

EPISODES = 2
MAX_STEPS = 16  # s80 cohort convention
# How much of each observation the agent sees before its next turn. Part of a matrix's
# capture cohort (it changes what the candidates are measured on), so it is pinned
# explicitly here rather than left to `evaluate_pool`'s default drifting under us.
HISTORY_CHARS = DEFAULT_HISTORY_CHARS
COHORT_SCENARIOS = 80
DEFAULT_MODELS = "gpt-5.4-mini,sonnet-5,gpt-5.5"
DRY_RUN_TRANSCRIPTS = 40
PROBE_ROWS_STEM = "financebench-s80-probe"
PROBE_DEFAULT_CAP_USD = 25.0

# The judge is pinned for the whole grid: every arm's rewards must come from the same grader, or
# the arm-to-arm deltas the grid exists to measure are partly judge drift.
JUDGE = ProviderConfig(
    kind=ProviderKind.BEDROCK, model="us.anthropic.claude-opus-4-8", region="us-east-1"
)

# Arm name -> the compressor id it runs. `off` has no compressor (compression=None, today's rows
# exactly). `truncate` and `identity` are the seam's built-ins; the two controls come from
# wmo.research.compressors; the learned arm is the endpoint, registered lazily by
# `wmo.optimize` as a factory.
ARM_COMPRESSORS: dict[str, str] = {
    "llmlingua2-endpoint": "llmlingua2-endpoint",
    "truncate-protect-task": "truncate-protect-task",
    "control-random": "random-removal",
    "control-truncate": "truncate",
}
ARM_NAMES = ("off", *ARM_COMPRESSORS)
ENDPOINT_ARM = "llmlingua2-endpoint"

# The endpoint reads `aggressiveness` as the absolute keep-probability bar a word must clear.
# 0.5 is round 0's pinned threshold, so the learned arm here is the method round 0 measured.
ENDPOINT_THRESHOLD = 0.5

# Module-level singletons: `register_compressor` refuses to rebind an id to a DIFFERENT object,
# so registering fresh instances twice in one process would raise.
_LOCAL_COMPRESSORS = (RandomRemovalCompressor(), TruncateProtectTaskCompressor())


class CapExceeded(BaseException):
    """Raised from the per-cell callback when the all-in cap is breached.

    Deliberately a BaseException rather than an Exception. `evaluate_pool` wraps its `on_outcome`
    call in `except Exception` on purpose (a broken progress pipe must never throw away cells
    that were already paid for), so an ordinary exception raised there is logged and the sweep
    keeps spending. The cell loop in `run_cells` ALSO checks the budget before scheduling the
    next cell, so the halt never depends on this class escaping; this only bounds the overshoot
    to the episode that was already in flight.
    """


def register_arms() -> None:
    """Register the local control compressors; the built-ins and the endpoint need nothing."""
    for compressor in _LOCAL_COMPRESSORS:
        register_compressor(compressor)


def compression_for(compressor_id: str, aggressiveness: float) -> CompressionConfig:
    """The config for one arm, stamped with the version that will actually RUN.

    Mirrors `_compression_for` in `wmo/cli/route_app.py`: the version is read off the resolved
    implementation rather than defaulted, so the row's provenance names the bytes that produced
    it, and `servable_compressor` refuses an arm that could never be mounted (an unknown id, a
    version mismatch, or a churny compressor) before a cent is spent measuring it.
    """
    resolved = get_compressor(compressor_id)
    config = CompressionConfig(
        compressor_id=compressor_id,
        compressor_version=resolved.version,
        aggressiveness=aggressiveness,
    )
    servable_compressor(config)
    return config


def build_arms(names: list[str], control_aggressiveness: float) -> dict[str, CompressionConfig | None]:
    """Arm name -> config (None = compression off), built only for the arms asked for.

    Built lazily per name because resolving the endpoint arm constructs its client, which reads
    credentials and handshakes with the box: a run of the local arms only should not need the
    endpoint to be up.
    """
    arms: dict[str, CompressionConfig | None] = {}
    for name in names:
        if name == "off":
            arms[name] = None
            continue
        aggressiveness = (
            ENDPOINT_THRESHOLD if name == ENDPOINT_ARM else control_aggressiveness
        )
        arms[name] = compression_for(ARM_COMPRESSORS[name], aggressiveness)
    return arms


# ---------------------------------------------------------------------------
# Spend: all-in, because the approved cap is all-in.
# ---------------------------------------------------------------------------


@dataclass
class Spend:
    """Running all-in spend, which is what the cap is set against.

    Three sides, metered separately because they are billed separately and the findings quote
    them separately: the candidate model's own calls (priced by its pool entry), the compressor's
    own inference (the endpoint's per-call cost, which the seam puts on the outcome), and the
    evaluation infrastructure (world-model serve + judge, metered by the world model itself).
    """

    cap_usd: float
    candidate_usd: float = 0.0
    compressor_usd: float = 0.0
    serve_usd: float = 0.0
    judge_usd: float = 0.0
    episodes: int = 0
    env_episodes_metered: int = 0
    env_episodes_unmetered: int = 0
    per_model_usd: dict[str, float] = field(default_factory=dict)
    per_model_episodes: dict[str, int] = field(default_factory=dict)

    @property
    def env_judge_usd(self) -> float:
        return self.serve_usd + self.judge_usd

    @property
    def total_usd(self) -> float:
        return self.candidate_usd + self.compressor_usd + self.env_judge_usd

    @property
    def exceeded(self) -> bool:
        return self.total_usd > self.cap_usd

    def add_outcome(self, outcome: ScenarioOutcome) -> None:
        """Bank one cell's candidate-side and compressor-side cost."""
        self.candidate_usd += outcome.cost_usd
        self.compressor_usd += outcome.compressor_cost_usd
        self.episodes += 1
        self.per_model_usd[outcome.model] = (
            self.per_model_usd.get(outcome.model, 0.0) + outcome.cost_usd
        )
        self.per_model_episodes[outcome.model] = self.per_model_episodes.get(outcome.model, 0) + 1

    def add_env_record(self, record: RunRecord) -> None:
        """Bank one episode's world-model record: serve and judge, split by phase."""
        by_phase = record.by_phase
        judge = by_phase.get(Phase.JUDGE)
        judge_usd = judge.cost_usd if judge is not None else 0.0
        self.judge_usd += judge_usd
        # Everything the session metered that was not the judge is the simulator's own cost
        # (serve steps, plus zero-cost embed events), so nothing is silently dropped.
        self.serve_usd += record.total.cost_usd - judge_usd
        self.env_episodes_metered += 1

    def report(self) -> str:
        return (
            f"all-in ${self.total_usd:.2f} of ${self.cap_usd:.2f} cap "
            f"(candidate ${self.candidate_usd:.2f}, compressor ${self.compressor_usd:.4f}, "
            f"serve ${self.serve_usd:.2f}, judge ${self.judge_usd:.2f}, "
            f"{self.episodes} episodes)"
        )


class EnvUsageMeter:
    """An `env_factory` wrapper that banks each episode's world-model usage as it closes.

    The env+judge half of the bill. `WorldModel` meters its own serve calls and its judge call
    onto a per-session `RunTracker`, and `WorldModelEnv.close` leaves that session's final
    `RunRecord` on the env as `usage`; this harvests it. Same mechanism `wmo.optimize.sweep`
    uses (`_SessionHarvest`), lifted here because this script does not go through `run_sweep`.

    Banking is driven by the caller right after each cell completes (`run_episode` closes the env
    inside its own `finally`, so the record is already final by then) and again when the next env
    is built, which is what keeps the cap's view of spend current instead of one episode stale.
    An env that exposes no record is COUNTED, never guessed at.
    """

    def __init__(self, inner: Callable[[], Env], spend: Spend) -> None:
        self._inner = inner
        self._spend = spend
        self._open: Env | None = None

    def __call__(self) -> Env:
        self.bank()
        self._open = self._inner()
        return self._open

    def bank(self) -> None:
        """Take the closed env's record, once; a banked env is dropped so it cannot double count."""
        if self._open is None:
            return
        env, self._open = self._open, None
        record = getattr(env, "usage", None)
        if isinstance(record, RunRecord):
            self._spend.add_env_record(record)
        else:
            self._spend.env_episodes_unmetered += 1


# ---------------------------------------------------------------------------
# The cell loop: one (scenario, model) cell per `evaluate_pool` call.
# ---------------------------------------------------------------------------

Cell = tuple[Scenario, PoolEntry]


def run_cells(
    cells: list[Cell],
    *,
    compression: CompressionConfig | None,
    meter: EnvUsageMeter,
    tools_hint: str,
    spend: Spend,
    on_row: Callable[[ScenarioOutcome], None],
    label: str,
    history_chars: int = HISTORY_CHARS,
    episode_limit: int | None = None,
    provider_factory: Callable[[PoolEntry], Provider] = pool_provider,
) -> list[ScenarioOutcome]:
    """Run `cells` one at a time, checking the budget BEFORE scheduling each one.

    Driving `evaluate_pool` per (scenario, model) cell rather than once per arm is what makes the
    cap reliable. `evaluate_pool` runs its cells in one sequential loop and has no cancel hook, so
    the only in-flight way to stop it is to raise out of `on_outcome`, and that path is guarded by
    an `except Exception` that exists to protect paid cells from a broken progress pipe. Deciding
    between cells instead needs no exception semantics at all: nothing is scheduled after the
    budget is gone. `CapExceeded` (a BaseException, so the guard cannot swallow it) is the
    backstop that also stops the cell already running, which bounds the overshoot to one episode.

    Rows are persisted by `on_row` as each episode completes, so a halt (or a crash) always
    leaves every completed episode on disk and the arm resumes from there.
    """
    collected: list[ScenarioOutcome] = []
    started = time.monotonic()

    def record(outcome: ScenarioOutcome) -> None:
        meter.bank()  # this episode's env is closed; its record is final
        spend.add_outcome(outcome)
        collected.append(outcome)
        on_row(outcome)
        log.info(
            "%s: %s on %s ep%d reward=%s | %s",
            label,
            outcome.model,
            outcome.scenario_id,
            outcome.episode,
            "unscored" if outcome.reward is None else f"{outcome.reward:.2f}",
            spend.report(),
        )
        if spend.exceeded:
            raise CapExceeded(
                f"{label}: all-in spend ${spend.total_usd:.2f} exceeded cap "
                f"${spend.cap_usd:.2f}; halting (rows persisted, rerun with a fresh cap to resume)"
            )

    for index, (scenario, entry) in enumerate(cells, start=1):
        if spend.exceeded:
            log.warning(
                "%s: budget spent, %d of %d cells left unscheduled (%s)",
                label,
                len(cells) - index + 1,
                len(cells),
                spend.report(),
            )
            break
        if episode_limit is not None and len(collected) >= episode_limit:
            break
        try:
            evaluate_pool(
                meter,
                ModelPool(models=[entry]),
                [scenario],
                episodes_per_scenario=EPISODES,
                max_steps=MAX_STEPS,
                tools_hint=tools_hint,
                history_chars=history_chars,
                on_outcome=record,
                compression=compression,
                provider_factory=provider_factory,
            )
        except CapExceeded as halt:
            log.error("%s", halt)
            break
        finally:
            meter.bank()  # nothing should be left open, but an unbanked env is unaccounted spend
    log.info("%s: %d episodes in %.0fs, %s", label, len(collected), time.monotonic() - started, spend.report())
    return collected


# ---------------------------------------------------------------------------
# Shared setup.
# ---------------------------------------------------------------------------


def cohort_ids() -> set[str]:
    """The 80 scenario ids the grid is pinned to (the routing cohort matrix's own rows)."""
    outcomes = json.loads(COHORT_MATRIX.read_text())["outcomes"]
    return {str(outcome["scenario_id"]) for outcome in outcomes}


def cohort_scenarios(traces_path: Path, *, allow_partial: bool) -> tuple[list[Scenario], str]:
    """The cohort's scenarios (in corpus order) plus the corpus tools hint.

    Fails closed on a partial match by default: the 80-scenario cohort is the pin that makes
    these rows comparable with the routing matrix and with every other arm, and a corpus that
    resolves 77 of them silently measures a different grid.
    """
    if not traces_path.is_file():
        raise SystemExit(
            f"no trace corpus at {traces_path}. It is a Hub-hosted bundle (gitignored here): "
            "fetch it with `uv run python -m environment_capture.hub financebench`, or pass "
            "--traces pointing at a local copy."
        )
    traces = get_adapter("otel-genai").from_file(str(traces_path))
    wanted = cohort_ids()
    scenarios = [s for s in scenarios_from_traces(traces) if scenario_id(s) in wanted]
    log.info(
        "scenarios: %d of %d cohort ids matched from %d traces in %s",
        len(scenarios),
        len(wanted),
        len(traces),
        traces_path,
    )
    if len(scenarios) != len(wanted) and not allow_partial:
        raise SystemExit(
            f"{traces_path} resolves {len(scenarios)} of the {len(wanted)} pinned cohort "
            "scenarios; a partial cohort is a different grid. Point --traces at the corpus the "
            "s80 matrix was captured from, or pass --allow-partial-cohort to measure the subset "
            "deliberately."
        )
    return scenarios, tools_hint_from_traces(traces)


def pool_entries(pool_path: Path, models: list[str]) -> list[PoolEntry]:
    """The pool entries for `models`, in the order given (the grid's candidate side)."""
    if not pool_path.is_file():
        raise SystemExit(
            f"no candidate pool at {pool_path}; pass --pool, or register the candidates with "
            "`uv run wmo providers set`. The grid needs the same handles the s80 matrix used: "
            f"{DEFAULT_MODELS}."
        )
    pool = load_pool(pool_path)
    try:
        return [pool.entry(name) for name in models]
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc


def env_meter(spend: Spend) -> EnvUsageMeter:
    """One metered env per episode: the pinned world model, judged by the pinned Opus 4.8."""
    config = load_config(str(MODEL_DIR))
    serve_config = config.serve_provider_config()
    judge_provider = get_provider(JUDGE)

    def build() -> Env:
        world_model = WorldModel.load(
            str(MODEL_DIR), get_provider(serve_config), reward_provider=judge_provider
        )
        return WorldModelEnv(world_model, score_on_close=True)

    return EnvUsageMeter(build, spend)


def rows_path_for(stem: str) -> Path:
    return OUT_DIR / f"{stem}_rows.jsonl"


def done_cells(rows_path: Path) -> set[tuple[str, str, int]]:
    """The (scenario, model, episode) cells already on disk, so a rerun resumes."""
    if not rows_path.exists():
        return set()
    done: set[tuple[str, str, int]] = set()
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done.add((str(row["scenario_id"]), str(row["model"]), int(row["episode"])))
    return done


# ---------------------------------------------------------------------------
# Dry run: achieved ratios + cost projection, the approval artifact.
# ---------------------------------------------------------------------------


def dry_run(models: list[str], arms: list[str], control_aggressiveness: float) -> None:
    """Replay cohort transcripts through each arm offline and print the projection table."""
    cohort = json.loads(COHORT_MATRIX.read_text())["outcomes"]
    # Rebuild segment lists the way the seam sees them: the compressing provider hands over the
    # user-role message contents, which are the task plus the env observations. The matrix stores
    # agent replies, not observations, so this proxy uses task + replies (the round 0 convention;
    # observation-light, stated in the findings).
    sample: list[list[str]] = []
    seen: set[str] = set()
    for outcome in cohort:
        if outcome["scenario_id"] in seen or not outcome.get("replies"):
            continue
        seen.add(outcome["scenario_id"])
        sample.append([str(outcome["task"])] + [str(reply) for reply in outcome["replies"]])
        if len(sample) >= DRY_RUN_TRANSCRIPTS:
            break

    register_arms()
    configs = build_arms(arms, control_aggressiveness)
    log.info("achieved token ratios on %d cohort transcripts (chars/4 proxy):", len(sample))
    ratios: dict[str, float] = {}
    endpoint_cost = 0.0
    endpoint_raw = 0
    for name, config in configs.items():
        if config is None:
            continue
        compressor = get_compressor(config.compressor_id)
        raw = compressed = 0
        cost = 0.0
        for segments in sample:
            result = compressor.compress(segments, config)
            raw += result.tokens_in_raw
            compressed += result.tokens_in_compressed
            cost += result.cost_usd
        ratios[name] = compressed / raw
        log.info(
            "  %-24s ratio=%.3f (removes %.1f%%) at aggressiveness %.2f",
            name,
            ratios[name],
            (1 - ratios[name]) * 100,
            config.aggressiveness,
        )
        if name == ENDPOINT_ARM:
            endpoint_cost, endpoint_raw = cost, raw

    if ENDPOINT_ARM in ratios:
        matched = 1 - ratios[ENDPOINT_ARM]
        log.info(
            "matched control aggressiveness (the ENDPOINT arm's own mean removal): %.3f "
            "(rerun live with --control-aggressiveness %.2f)",
            matched,
            matched,
        )
        if endpoint_raw:
            log.info(
                "  endpoint compressor side: $%.6f over %d raw proxy tokens "
                "($%.4f per 10k), metered per call and counted against the cap live",
                endpoint_cost,
                endpoint_raw,
                endpoint_cost / endpoint_raw * 10_000,
            )
    else:
        log.info("endpoint arm not in --arms; matched control aggressiveness not measured")

    # Cost projection from the cohort's own measured numbers.
    per_model_cost: dict[str, float] = {}
    for model in models:
        rows = [
            outcome
            for outcome in cohort
            if outcome["model"] == model and outcome.get("cost_usd") is not None
        ]
        per_model_cost[model] = sum(row["cost_usd"] for row in rows) / len(rows)
    arms_n = len(configs)
    episodes_per_arm_model = COHORT_SCENARIOS * EPISODES
    candidate = sum(per_model_cost.values()) * episodes_per_arm_model * arms_n
    # Env + judge, token arithmetic only (stated +-2x). `--probe` replaces this estimate with a
    # measurement, which is the whole reason that mode exists.
    env_ep = 12_000 * 5 / 1e6 + 1_600 * 25 / 1e6
    judge_ep = 4_000 * 5 / 1e6 + 400 * 25 / 1e6
    grid_episodes = episodes_per_arm_model * arms_n * len(models)
    env_judge = (env_ep + judge_ep) * grid_episodes
    log.info(
        "cost projection (%d arms x %d models x %d scenarios x %d episodes = %d episodes):",
        arms_n,
        len(models),
        COHORT_SCENARIOS,
        EPISODES,
        grid_episodes,
    )
    for model, cost_per_ep in per_model_cost.items():
        log.info(
            "  candidate %-14s $%.4f/ep -> $%.2f",
            model,
            cost_per_ep,
            cost_per_ep * episodes_per_arm_model * arms_n,
        )
    log.info("  candidate total  $%.0f", candidate)
    log.info(
        "  env+judge (ESTIMATE, +-2x) $%.0f  (env $%.3f/ep + judge $%.3f/ep; measure it with "
        "--probe)",
        env_judge,
        env_ep,
        judge_ep,
    )
    log.info(
        "  TOTAL $%.0f (range $%.0f-$%.0f on the env uncertainty)",
        candidate + env_judge,
        candidate + env_judge / 2,
        candidate + env_judge * 2,
    )


# ---------------------------------------------------------------------------
# Probe: measure the two numbers the projection guesses at.
# ---------------------------------------------------------------------------


def probe_run(
    models: list[str],
    episodes: int,
    cap_usd: float,
    pool_path: Path,
    traces_path: Path,
    *,
    allow_partial: bool,
    history_chars: int = HISTORY_CHARS,
) -> None:
    """Run `episodes` uncompressed episodes and recompute the grid total from what they cost."""
    scenarios, hint = cohort_scenarios(traces_path, allow_partial=allow_partial)
    entries = pool_entries(pool_path, models)
    cells_needed = math.ceil(episodes / EPISODES)
    scenario_count = math.ceil(episodes / (len(entries) * EPISODES))
    cells: list[Cell] = [
        (scenario, entry) for scenario in scenarios[:scenario_count] for entry in entries
    ][:cells_needed]
    log.info(
        "probe: %d episodes over %d cells (%d scenarios x %d models, %d episodes each), "
        "arm 'off' only, cap $%.2f",
        episodes,
        len(cells),
        scenario_count,
        len(entries),
        EPISODES,
        cap_usd,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = rows_path_for(PROBE_ROWS_STEM)
    spend = Spend(cap_usd=cap_usd)
    meter = env_meter(spend)
    with rows_path.open("a", encoding="utf-8") as handle:

        def persist(outcome: ScenarioOutcome) -> None:
            handle.write(outcome.model_dump_json() + "\n")
            handle.flush()

        outcomes = run_cells(
            cells,
            compression=None,
            meter=meter,
            tools_hint=hint,
            history_chars=history_chars,
            spend=spend,
            on_row=persist,
            label="probe",
            episode_limit=episodes,
        )

    if not outcomes:
        raise SystemExit("probe ran no episodes; nothing to project from")
    log.info("probe rows appended to %s", rows_path)
    log.info("MEASURED candidate cost per episode:")
    for model in models:
        ran = spend.per_model_episodes.get(model, 0)
        if not ran:
            log.info("  %-14s no episodes ran (raise --probe)", model)
            continue
        log.info("  %-14s $%.4f/ep over %d episodes", model, spend.per_model_usd[model] / ran, ran)
    metered = spend.env_episodes_metered
    if not metered:
        raise SystemExit(
            "no world-model usage was metered, so env+judge cost cannot be measured; the env "
            "factory must build a WorldModelEnv (its `usage` record is the source)"
        )
    env_judge_ep = spend.env_judge_usd / metered
    serve_ep = spend.serve_usd / metered
    judge_ep = spend.judge_usd / metered
    log.info(
        "  env+judge      $%.4f/ep (serve $%.4f + judge $%.4f) over %d metered episodes"
        "%s",
        env_judge_ep,
        serve_ep,
        judge_ep,
        metered,
        f", {spend.env_episodes_unmetered} unmetered" if spend.env_episodes_unmetered else "",
    )
    unscored = [o for o in outcomes if o.reward is None]
    if unscored:
        log.warning(
            "probe: %d of %d episodes were unscored (first: %s); a grid at this rate loses that "
            "share of its cells",
            len(unscored),
            len(outcomes),
            unscored[0].error,
        )

    arms_n = len(ARM_NAMES)
    episodes_per_arm_model = COHORT_SCENARIOS * EPISODES
    grid_episodes = episodes_per_arm_model * arms_n * len(models)
    candidate = 0.0
    for model in models:
        ran = spend.per_model_episodes.get(model, 0)
        if ran:
            candidate += spend.per_model_usd[model] / ran * episodes_per_arm_model * arms_n
    env_judge = env_judge_ep * grid_episodes
    log.info(
        "RECOMPUTED full grid from measurements (%d arms x %d models x %d scenarios x %d "
        "episodes = %d episodes):",
        arms_n,
        len(models),
        COHORT_SCENARIOS,
        EPISODES,
        grid_episodes,
    )
    log.info("  candidate  $%.0f", candidate)
    log.info("  env+judge  $%.0f", env_judge)
    log.info(
        "  TOTAL      $%.0f (compressor side excluded: it is the endpoint arm only, metered per "
        "call, and the dry run prints its measured rate)",
        candidate + env_judge,
    )
    if len(models) > len([m for m in models if spend.per_model_episodes.get(m)]):
        log.warning(
            "some models ran no probe episodes, so the candidate projection above is incomplete"
        )


# ---------------------------------------------------------------------------
# Live grid.
# ---------------------------------------------------------------------------


def live_run(
    models: list[str],
    cap_usd: float,
    control_aggressiveness: float,
    arms: list[str],
    pool_path: Path,
    traces_path: Path,
    *,
    allow_partial: bool,
    history_chars: int = HISTORY_CHARS,
) -> None:
    """Run the arms in order against the pinned cohort, metering all-in spend against `cap_usd`."""
    register_arms()
    configs = build_arms(arms, control_aggressiveness)
    scenarios, hint = cohort_scenarios(traces_path, allow_partial=allow_partial)
    entries = pool_entries(pool_path, models)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spend = Spend(cap_usd=cap_usd)
    meter = env_meter(spend)
    log.info(
        "grid: %d arms x %d models x %d scenarios x %d episodes, all-in cap $%.2f",
        len(configs),
        len(entries),
        len(scenarios),
        EPISODES,
        cap_usd,
    )

    for arm, compression in configs.items():
        stem = f"financebench-s80-{arm}"
        rows_path = rows_path_for(stem)
        done = done_cells(rows_path)
        if done:
            log.info("%s: resuming, %d rows already on disk", arm, len(done))
        cells = [
            (scenario, entry)
            for scenario in scenarios
            for entry in entries
            if any(
                (scenario_id(scenario), entry.name, episode) not in done
                for episode in range(EPISODES)
            )
        ]
        if not cells:
            log.info("%s: complete, nothing to run", arm)
        with rows_path.open("a", encoding="utf-8") as handle:

            def persist(outcome: ScenarioOutcome, seen: set[tuple[str, str, int]] = done) -> None:
                key = (outcome.scenario_id, outcome.model, outcome.episode)
                if key in seen:
                    # Only reachable on resume, where a cell with one of its two episodes on disk
                    # has to rerun both (`evaluate_pool` has no episode offset). The rerun really
                    # was paid for, so its spend counts; the duplicate row does not get written.
                    log.info("%s: %s already on disk, row not duplicated", arm, key)
                    return
                seen.add(key)
                handle.write(outcome.model_dump_json() + "\n")
                handle.flush()

            run_cells(
                cells,
                compression=compression,
                meter=meter,
                tools_hint=hint,
                history_chars=history_chars,
                spend=spend,
                on_row=persist,
                label=arm,
            )

        remaining = [
            cell
            for cell in cells
            if any(
                (scenario_id(cell[0]), cell[1].name, episode) not in done
                for episode in range(EPISODES)
            )
        ]
        if remaining:
            log.warning(
                "%s: %d cells still missing, matrix NOT written (a partial matrix reads as a "
                "complete arm); rerun to resume",
                arm,
                len(remaining),
            )
        else:
            matrix_path = OUT_DIR / f"{stem}_matrix.json"
            rows = [
                ScenarioOutcome.model_validate_json(line)
                for line in rows_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            matrix = OutcomeMatrix(pool=entries, outcomes=rows)
            matrix_path.write_text(matrix.model_dump_json(), encoding="utf-8")
            log.info("%s: wrote %d rows to %s", arm, len(rows), matrix_path)
        if spend.exceeded:
            log.error("grid halted on the cap after arm '%s'; %s", arm, spend.report())
            break
    log.info("grid finished: %s", spend.report())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="no episodes; ratios + projection")
    parser.add_argument(
        "--probe",
        type=int,
        nargs="?",
        const=10,
        default=None,
        help="measure env+judge and candidate $/ep on N uncompressed episodes (default 10)",
    )
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument(
        "--arms",
        default=",".join(ARM_NAMES),
        help=f"comma list; known arms: {', '.join(ARM_NAMES)}",
    )
    parser.add_argument(
        "--cap-usd",
        type=float,
        default=None,
        help="master-approved ALL-IN cap (candidate + compressor + env/judge); required live",
    )
    parser.add_argument("--control-aggressiveness", type=float, default=0.4)
    parser.add_argument(
        "--history-chars",
        type=int,
        default=HISTORY_CHARS,
        help="observation chars the agent sees per turn; part of the capture cohort",
    )
    parser.add_argument("--pool", default=str(REPO / DEFAULT_POOL_PATH))
    parser.add_argument("--traces", default=str(TRACES))
    parser.add_argument(
        "--allow-partial-cohort",
        action="store_true",
        help="proceed when the corpus resolves fewer than the 80 pinned cohort scenarios",
    )
    args = parser.parse_args()

    # The compressor endpoint (WMO_COMPRESSOR_*) and the candidate accounts (AZURE_*) live in the
    # worktree's .env, exactly as the CLI's own entry point reads them.
    load_env_file(REPO / ".env")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARM_NAMES]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {list(ARM_NAMES)}")

    if args.dry_run:
        dry_run(models, arms, args.control_aggressiveness)
        return
    if args.probe is not None:
        probe_run(
            models,
            args.probe,
            args.cap_usd if args.cap_usd is not None else PROBE_DEFAULT_CAP_USD,
            Path(args.pool),
            Path(args.traces),
            allow_partial=args.allow_partial_cohort,
            history_chars=args.history_chars,
        )
        return
    if args.cap_usd is None:
        raise SystemExit(
            "live runs need --cap-usd (the master-approved all-in cap); or use --dry-run/--probe"
        )
    live_run(
        models,
        args.cap_usd,
        args.control_aggressiveness,
        arms,
        Path(args.pool),
        Path(args.traces),
        allow_partial=args.allow_partial_cohort,
        history_chars=args.history_chars,
    )


if __name__ == "__main__":
    main()
