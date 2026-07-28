"""Real tau2-bench episodes for the pinned eval split: the sim-to-real leg of the tau benchmark.

Every other matrix in this project scores candidates INSIDE a world model with an LLM judge. This
one runs Sierra's actual benchmark and takes tau2's own reward (DB state check plus action,
communicate, and env-assertion checks). Fit on the world-model matrices, validate here.

What makes this the validation leg rather than a second, differently-shaped experiment:

- **Same scenarios.** Selection comes from `scenarios_eval.jsonl`, the pinned eval split every
  tau arm evaluates on (`pin_scenarios.py`, seed 4405, ALL of the test split). Each pinned
  scenario carries the tau2 `user_scenario.instructions` blob verbatim, so it resolves back to a
  tau2 task id by exact match against the domain's `tasks.json`. An unmatched scenario is a hard
  error: silently running a different task set is exactly the failure this leg exists to rule out.
- **Same ids.** `scenario_id` is `"<domain>:<task_id>"`, never the bare tau2 id. Airline and
  retail both number their tasks from "0" and all 50 airline ids collide with retail ids, so bare
  ids would merge two different tasks into one matrix cell.
- **One environment for every candidate.** The user simulator is part of the environment, so it is
  pinned to one cheap model (`--user-sim`, default `gpt-5.4-mini`) for every run. Letting it vary
  per candidate would change the environment per candidate and make the rewards incomparable.
  Neither stream is given a temperature: several pool models reject sampling params, and dropping
  it everywhere removes a source of cross-model variation rather than only working around the
  strict ones.

Protocol pins and capture cohort: what an episode IS here is fixed by five values, adopted as
canonical for every real-tau2 leg (joint-tau master, 2026-07-27 "cross-lane weave" ack 2):
`max_turns=100`, `episode_timeout_s=1800`, `max_tokens=8192`, user simulator `azure/gpt-5.4-mini`,
and tau2's own retries OFF with retry owned by this runner. All five are forwarded explicitly
because tau2's defaults are different (200 steps, no per-episode timeout, no token cap, 3 retries),
so inheriting them would quietly produce rows from a different environment than the training lane's
runs. Every row records the pin set as a `cohort` label, and `sim_to_real.py` refuses to pair rows
whose labels differ. The flags exist so the pins can be moved deliberately; any move is a NEW
cohort, not a variation of this one. The one knowing difference from the training lane
(`wmo/distill/tau2.py`, which shells the same clone with the same pins) is sampling temperature:
that lane pins its Tinker student to 1.0, while pool candidates here run at each provider's own
default for the reason above. Temperature is not part of the canonical pin set.

Reward provenance: this leg NEVER runs a wmo judge. It reads `reward_info.reward` out of tau2's
`results.json` and nothing else. Note that tau2's own reward is not uniformly deterministic:
7 of the 20 pinned eval tasks carry `NL_ASSERTION` in their `reward_basis`, which tau2 scores with
its built-in NL-assertion judge. Every row records `nl_assertion_reward` so analysis can split
the fully deterministic rows (DB / ACTION / COMMUNICATE / ENV_ASSERTION) from the rest.

Cost: `cost_usd_pool` is computed from recorded token usage at OUR pool prices, not taken from
tau2's `agent_cost`. tau2 gets its number from litellm's price table, which does not know our
Azure MaaS deployments (measured: it priced Kimi-K2.6 at $0.0197 where the published eastus2
meters give $0.0351). tau2's figures are kept alongside ours for audit.

Isolation: `wmo` never imports `tau2` (see ../README.md). This shells out to the tau2 CLI in its
own venv. `--capture-dir` points at the directory holding `tau2-bench/` and `.venv/`; it defaults
to this benchmark dir, so the README's setup block is all that is needed.

Resumable and budgeted: rows are appended to `rows.jsonl` after every batch and keyed by
(scenario, model, episode), so a stop or a crash leaves a usable partial grid and a rerun picks up
exactly what is missing. Every attempt at a cell gets its own tau2 save directory, because tau2
treats an existing one as a checkpoint and refuses a run whose task list is a subset of it, which
is precisely what a resumed cell asks for. `--retry-failed` re-buys the cells whose latest row
carries no reward: with tau2's own retries pinned off, that is where a transiently dead episode
gets its second chance. `--budget-usd` is checked between batches, and models run cheapest-first
for the same reason.

Run from the repo root:

    uv run python packages/environment-capture/tau-bench/rl/real_episodes.py --dry-run
    uv run python packages/environment-capture/tau-bench/rl/real_episodes.py --episodes 2
    uv run python packages/environment-capture/tau-bench/rl/real_episodes.py --write-matrix-only
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

from wmo.core.types import JsonObject, JsonValue
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import DEFAULT_POOL_PATH, PoolEntry, load_pool

logger = logging.getLogger("tau-real")

_HERE = Path(__file__).resolve().parent
EVAL_SCENARIOS = _HERE / "scenarios_eval.jsonl"
DEFAULT_CAPTURE_DIR = _HERE.parent
DEFAULT_OUT_DIR = Path(".wmo/evals/tau-bench-real")

# The user simulator is environment, not candidate: one cheap model, identical for every run.
DEFAULT_USER_SIM = "gpt-5.4-mini"
DEFAULT_BUDGET_USD = 140.0
DEFAULT_MAX_CONCURRENCY = 4

# The CANONICAL real-tau2 protocol pins (joint-tau master, DECISIONS.md 2026-07-27 "cross-lane
# weave" ack 2). Every real-tau2 leg in the project runs these exact values; the training lane
# carries the same numbers as its `rollout`/`sampling` config defaults.
CANONICAL_MAX_TURNS = 100
CANONICAL_EPISODE_TIMEOUT_S = 1800.0
CANONICAL_MAX_TOKENS = 8192
CANONICAL_TAU2_MAX_RETRIES = 0

# Rows written before the pins were forwarded carry no label. They are their own cohort: their
# episodes ran on tau2's defaults, which is a different environment, not a noisier sample of this
# one.
UNLABELED_COHORT = "unlabeled"

# Wall clock granted past the batch's own graceful per-episode budget before the subprocess is
# killed. A kill forfeits every episode tau2 had not yet written, so this covers tau2's startup,
# its evaluation pass, and its save, not a second wave of episodes.
BATCH_KILL_MARGIN_S = 300.0

# How many attempts one cell may make before its save-directory search gives up. Reaching this
# means something is failing every time, and silently piling up directories would hide it.
MAX_ATTEMPTS_PER_CELL = 100

# A realistic agent-side episode mix, used only to order the pool cheapest-first. Ordering by
# input price alone would let a model with a cheap input rate and an expensive output rate look
# cheaper than it is.
_EPISODE_MIX = TokenUsage(input_tokens=30_000, output_tokens=1_000)

# Microsoft fronts two different Azure services and litellm addresses them with DIFFERENT env
# prefixes (AZURE_* vs AZURE_AI_*). That split is what lets a google-sheets user simulator run
# against a silen-resource candidate inside one tau2 process without either clobbering the other.
_AZURE_OPENAI_HOST = ".openai.azure.com"
_AZURE_AI_HOST = ".services.ai.azure.com"


class ProtocolPins(BaseModel):
    """The five values that define what a real-tau2 episode is, plus the cohort label they name.

    Defaults are the canonical pins. Constructing this with anything else is legal and explicit:
    it produces a different `label`, which is how every downstream consumer learns that those rows
    are a separate capture cohort.
    """

    max_turns: int = Field(default=CANONICAL_MAX_TURNS, ge=1)
    episode_timeout_s: float = Field(default=CANONICAL_EPISODE_TIMEOUT_S, gt=0)
    # 0 means "send no cap at all"; see agent_llm_args.
    max_tokens: int = Field(default=CANONICAL_MAX_TOKENS, ge=0)
    tau2_max_retries: int = Field(default=CANONICAL_TAU2_MAX_RETRIES, ge=0)
    user_sim: str = DEFAULT_USER_SIM

    @property
    def label(self) -> str:
        """The cohort label recorded on every row: short, greppable, and complete."""
        return (
            f"turns{self.max_turns}-t{self.episode_timeout_s:.0f}-tok{self.max_tokens}"
            f"-r{self.tau2_max_retries}-sim-{self.user_sim}"
        )

    @property
    def is_canonical(self) -> bool:
        """Whether these are the canonical pins every other real-tau2 leg runs."""
        return self.label == ProtocolPins().label


def agent_llm_args(pins: ProtocolPins) -> str:
    """The `--agent-llm-args` JSON for the candidate stream: the token cap, and nothing else.

    `max_tokens=0` drops the key entirely. That escape hatch is deliberate: litellm rewrites
    `max_tokens` to `max_completion_tokens` only for the reasoning models its table knows, so a
    deployment name it has never seen is rejected outright by Azure, and a grid that cannot start
    at all is worse than a second cohort that says so in its label.
    """
    args: JsonObject = {} if pins.max_tokens <= 0 else {"max_tokens": pins.max_tokens}
    return json.dumps(args)


def batch_timeout_s(task_count: int, max_concurrency: int, episode_timeout_s: float) -> float:
    """Hard-kill deadline for one batch subprocess, derived from the per-episode pin.

    tau2's own `--timeout` ends each episode gracefully and still scores it, so this deadline is
    only for a wedged runner, and firing it forfeits every episode in the batch tau2 had not
    written yet. It therefore has to cover the whole batch: as many waves of the per-episode
    budget as the concurrency needs, plus the margin.

    Args:
        task_count: Tasks in this batch.
        max_concurrency: Episodes tau2 runs at once.
        episode_timeout_s: The per-episode pin handed to tau2.

    Returns:
        Seconds to allow the subprocess before killing it.
    """
    waves = max(1, math.ceil(task_count / max(1, max_concurrency)))
    return waves * episode_timeout_s + BATCH_KILL_MARGIN_S


def _zero_if_null(value: JsonValue) -> JsonValue:
    """Treat an explicitly null meter as zero: providers omit usage on errored/streamed turns."""
    return 0 if value is None else value


class Tau2Usage(BaseModel):
    """Per-message token counts as tau2 records them."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    _null_is_zero = field_validator("prompt_tokens", "completion_tokens", mode="before")(
        _zero_if_null
    )


class Tau2Message(BaseModel):
    """The fields of a tau2 transcript message this leg meters. Everything else is ignored."""

    role: str = ""
    # Multimodal/structured replies arrive as a list of content blocks rather than a string;
    # kept as raw JSON so one such message cannot invalidate the episode around it.
    content: JsonValue = None
    generation_time_seconds: float | None = None
    tool_calls: list[JsonValue] | None = None
    usage: Tau2Usage | None = None

    def text(self) -> str:
        """The reply as text, or "" when this message carries no plain-string content."""
        return self.content if isinstance(self.content, str) else ""


class Tau2RewardInfo(BaseModel):
    """tau2's verdict. `reward` is None only when tau2 could not score the episode."""

    reward: float | None = None


class Tau2Simulation(BaseModel):
    """One tau2 episode. Lenient by design: tau2 owns this schema and may extend it."""

    task_id: str = ""
    duration: float = 0.0
    termination_reason: str = ""
    agent_cost: float | None = None
    user_cost: float | None = None
    reward_info: Tau2RewardInfo | None = None
    messages: list[Tau2Message] = []

    _null_is_zero = field_validator("duration", mode="before")(_zero_if_null)


class Tau2Results(BaseModel):
    """The subset of tau2's `results.json` this leg reads."""

    simulations: list[Tau2Simulation] = []


class RealScenario(BaseModel):
    """One pinned eval scenario, resolved to the tau2 task it was captured from."""

    scenario_id: str  # "<domain>:<task_id>"
    domain: str
    task_id: str
    task: str  # the tau2 user_scenario.instructions blob, verbatim from the pinned split
    provenance: list[str]  # source trace ids, the pinned split's identity key
    nl_assertion_reward: bool  # tau2 scores part of this task with its NL-assertion judge


class RealEpisodeRow(BaseModel):
    """One (scenario, candidate, episode) real-benchmark episode, with its meter readings."""

    scenario_id: str
    domain: str
    task_id: str
    task: str
    provenance: list[str]
    model: str  # pool entry name
    route: str  # the litellm route tau2 was given
    episode: int
    reward: float | None  # None = unscored (infrastructure failure), never treated as 0
    nl_assertion_reward: bool
    termination_reason: str
    duration_s: float
    agent_input_tokens: int
    agent_output_tokens: int
    user_input_tokens: int
    user_output_tokens: int
    cost_usd_pool: float  # authoritative: our pool prices
    cost_usd_tau2_agent: float | None  # litellm's guess, kept for audit
    cost_usd_tau2_user: float | None
    steps: int
    call_seconds: list[float]
    replies: list[str]
    user_sim: str
    # The protocol pins this episode ran under (`ProtocolPins.label`). Rows from before the pins
    # were forwarded have no label and default to UNLABELED_COHORT rather than failing to load:
    # they are readable evidence of a different cohort, and consumers refuse to pair across them.
    cohort: str = UNLABELED_COHORT

    @property
    def key(self) -> tuple[str, str, int]:
        """The resume key: one row per scenario, candidate, and episode index."""
        return (self.scenario_id, self.model, self.episode)


def domains_dir(capture_dir: Path) -> Path:
    """The tau2 domain data root inside a capture directory."""
    return capture_dir / "tau2-bench" / "data" / "tau2" / "domains"


def _instructions_key(instructions: JsonValue) -> str:
    return json.dumps(instructions, sort_keys=True, ensure_ascii=False)


def _task_index(capture_dir: Path, domain: str) -> dict[str, JsonObject]:
    """Canonical instructions blob -> tau2 task, for one domain's `tasks.json`."""
    path = domains_dir(capture_dir) / domain / "tasks.json"
    if not path.is_file():
        raise SystemExit(
            f"no tau2 task file at {path}. Clone tau2-bench into the capture dir first "
            "(see packages/environment-capture/tau-bench/README.md § Setup), or pass "
            "--capture-dir pointing at an existing clone."
        )
    index: dict[str, JsonObject] = {}
    for task in json.loads(path.read_text(encoding="utf-8")):
        scenario = task.get("user_scenario") or {}
        index[_instructions_key(scenario.get("instructions") or {})] = task
    return index


def _has_nl_assertion(task: JsonObject) -> bool:
    criteria = task.get("evaluation_criteria")
    if not isinstance(criteria, dict):
        return False
    basis = criteria.get("reward_basis")
    return isinstance(basis, list) and "NL_ASSERTION" in basis


def load_pinned_scenarios(
    capture_dir: Path, scenarios_path: Path = EVAL_SCENARIOS
) -> list[RealScenario]:
    """Resolve every pinned eval scenario to its tau2 task id.

    The pinned split stores the tau2 `user_scenario.instructions` blob verbatim, so the match is
    exact on canonicalized JSON rather than fuzzy on prose. Any scenario that fails to resolve
    aborts the run: quietly dropping it would run this leg on a different task set than the world
    model was evaluated on, which is the one thing a sim-to-real comparison cannot survive.

    Args:
        capture_dir: Directory holding the `tau2-bench/` clone.
        scenarios_path: The pinned eval split JSONL.

    Returns:
        Resolved scenarios, ordered by scenario id.

    Raises:
        SystemExit: When the task data is missing or a scenario does not resolve.
    """
    if not scenarios_path.is_file():
        raise SystemExit(
            f"no pinned eval split at {scenarios_path}. Regenerate it with "
            "`uv run python packages/environment-capture/tau-bench/rl/pin_scenarios.py`."
        )
    lines = [
        line for line in scenarios_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    indexes: dict[str, dict[str, JsonObject]] = {}
    resolved: list[RealScenario] = []
    unmatched: list[str] = []
    for line in lines:
        pinned = json.loads(line)
        domain = str(pinned["domain"])
        if domain not in indexes:
            indexes[domain] = _task_index(capture_dir, domain)
        task_text = str(pinned["task"])
        try:
            instructions = json.loads(task_text)
        except ValueError as error:
            raise SystemExit(
                f"pinned scenario {pinned.get('provenance')} has a `task` field that is not the "
                f"tau2 instructions JSON ({error}). Re-pin the split with pin_scenarios.py."
            ) from error
        task = indexes[domain].get(_instructions_key(instructions))
        if task is None:
            unmatched.append(f"{domain}/{pinned['provenance'][0]}")
            continue
        task_id = str(task["id"])
        resolved.append(
            RealScenario(
                scenario_id=f"{domain}:{task_id}",
                domain=domain,
                task_id=task_id,
                task=task_text,
                provenance=[str(p) for p in pinned["provenance"]],
                nl_assertion_reward=_has_nl_assertion(task),
            )
        )
    if unmatched:
        raise SystemExit(
            f"{len(unmatched)} pinned eval scenario(s) do not match any tau2 task: "
            f"{unmatched[:5]}. The tau2 clone's tasks.json has drifted from the corpus the "
            "split was pinned on; re-pin against the same tau2 revision, or pass --capture-dir "
            "pointing at the clone the corpus was captured with."
        )
    return sorted(resolved, key=lambda s: s.scenario_id)


def litellm_route(entry: PoolEntry) -> str:
    """The litellm model string tau2 needs for a pool entry.

    Raises:
        SystemExit: When the entry's backend has no tau2/litellm equivalent.
    """
    if entry.kind is ProviderKind.ANTHROPIC:
        return f"anthropic/{entry.model}"
    if entry.kind is ProviderKind.OPENAI or entry.kind is ProviderKind.OPENAI_RESPONSES:
        return f"openai/{entry.model}"
    if entry.kind is ProviderKind.OPENROUTER:
        return f"openrouter/{entry.model}"
    if entry.kind is ProviderKind.BEDROCK:
        return f"bedrock/{entry.model}"
    if entry.kind is ProviderKind.AZURE_OPENAI:
        deployment = entry.deployment or entry.model
        endpoint = entry.endpoint or ""
        if _AZURE_AI_HOST in endpoint:
            return f"azure_ai/{deployment}"
        if _AZURE_OPENAI_HOST in endpoint:
            return f"azure/{deployment}"
        # Guessing here would send the request to the wrong service with the wrong credential
        # variable and surface as an opaque 401 inside tau2.
        raise SystemExit(
            f"pool model '{entry.name}' has azure endpoint '{endpoint}', which is neither "
            f"'*{_AZURE_OPENAI_HOST}' (litellm azure/) nor '*{_AZURE_AI_HOST}' (litellm "
            "azure_ai/). Set `endpoint` to the account URL so the route can be resolved."
        )
    raise SystemExit(
        f"pool model '{entry.name}' has kind '{entry.kind}', which tau2 cannot call. "
        "Drop it from the run with --only, or give it an OpenAI-compatible pool entry."
    )


def _credentials(entry: PoolEntry, environ: dict[str, str]) -> dict[str, str]:
    """The env vars litellm needs to reach one pool entry's account.

    Raises:
        SystemExit: When the entry names an API key variable that is not set.
    """
    route = litellm_route(entry)
    family = route.split("/", 1)[0]
    if family == "anthropic":
        key = environ.get(entry.api_key_env or "ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit(
                f"pool model '{entry.name}' needs {entry.api_key_env or 'ANTHROPIC_API_KEY'} "
                "in the environment"
            )
        return {"ANTHROPIC_API_KEY": key}
    if family in ("azure", "azure_ai"):
        variable = entry.api_key_env or (
            "AZURE_API_KEY" if family == "azure" else "AZURE_AI_API_KEY"
        )
        key = environ.get(variable)
        if not key:
            raise SystemExit(f"pool model '{entry.name}' needs {variable} in the environment")
        endpoint = entry.endpoint or ""
        if family == "azure":
            return {
                "AZURE_API_KEY": key,
                "AZURE_API_BASE": endpoint,
                "AZURE_API_VERSION": entry.api_version or "2024-10-21",
            }
        # The Azure AI inference endpoint litellm calls is the account URL plus /models; pool
        # entries record the account URL because that is what the wmo provider wants.
        base = endpoint if endpoint.endswith("/models") else f"{endpoint}/models"
        return {"AZURE_AI_API_KEY": key, "AZURE_AI_API_BASE": base}
    if family == "openai":
        key = environ.get(entry.api_key_env or "OPENAI_API_KEY")
        if not key:
            raise SystemExit(
                f"pool model '{entry.name}' needs {entry.api_key_env or 'OPENAI_API_KEY'} "
                "in the environment"
            )
        credentials = {"OPENAI_API_KEY": key}
        if entry.endpoint:
            # A self-hosted OpenAI-compatible server (a Tinker-served student, a local vLLM).
            # Dropping its endpoint would send this candidate's episodes to api.openai.com
            # under the wrong key and quietly measure a different model.
            credentials["OPENAI_API_BASE"] = entry.endpoint
            credentials["OPENAI_BASE_URL"] = entry.endpoint
        return credentials
    if family == "openrouter":
        key = environ.get(entry.api_key_env or "OPENROUTER_API_KEY")
        if not key:
            raise SystemExit(
                f"pool model '{entry.name}' needs {entry.api_key_env or 'OPENROUTER_API_KEY'} "
                "in the environment"
            )
        return {"OPENROUTER_API_KEY": key}
    return {}  # bedrock reads the ambient AWS profile/role


def build_env(
    capture_dir: Path, agent: PoolEntry, user_sim: PoolEntry, environ: dict[str, str]
) -> dict[str, str]:
    """The subprocess environment for one batch: tau2's data dir plus both streams' credentials.

    Raises:
        SystemExit: When the two streams need the same credential variable with different values
            (one tau2 process cannot hold two accounts on one Azure family), or when a key is
            missing.
    """
    env = dict(environ)
    env["TAU2_DATA_DIR"] = str(capture_dir / "tau2-bench" / "data")
    agent_credentials = _credentials(agent, environ)
    user_credentials = _credentials(user_sim, environ)
    for name, value in agent_credentials.items():
        clash = user_credentials.get(name)
        if clash is not None and clash != value:
            raise SystemExit(
                f"candidate '{agent.name}' and user simulator '{user_sim.name}' both need "
                f"{name} but with different values. One tau2 process holds one account per "
                "credential family; pin the user simulator to a model on the candidate's "
                "account, or run that candidate separately."
            )
    env.update(user_credentials)
    env.update(agent_credentials)
    return env


def batch_command(
    capture_dir: Path,
    entry: PoolEntry,
    user_sim: PoolEntry,
    domain: str,
    task_ids: Sequence[str],
    save_to: str,
    max_concurrency: int,
    pins: ProtocolPins,
) -> list[str]:
    """The exact tau2 CLI invocation for one (candidate, domain) batch, pins included."""
    # The pinned telecom scenarios were captured from telecom's 2285-task "full" split;
    # tau2's default task set for the domain does not contain them, so every telecom batch
    # died with "Not all tasks were found" until this forwarded the same override the
    # training lane pins (wmo/distill/tau2.py TASK_SPLIT_OVERRIDES). A property of the
    # corpus, not a knob.
    split_override = ["--task-split-name", "full"] if domain == "telecom" else []
    return [
        str(capture_dir / ".venv" / "bin" / "tau2"),
        "run",
        "--domain",
        domain,
        *split_override,
        "--agent-llm",
        litellm_route(entry),
        "--agent-llm-args",
        agent_llm_args(pins),
        "--user-llm",
        litellm_route(user_sim),
        # The user stream carries no args at all, matching the training lane's `user_llm_args`
        # default, so both legs' user simulators are the same environment.
        "--user-llm-args",
        "{}",
        "--num-trials",
        "1",
        "--task-ids",
        *task_ids,
        "--max-concurrency",
        str(max_concurrency),
        "--save-to",
        save_to,
        # The canonical pins. tau2's own defaults are 200 steps, no per-episode timeout, and 3
        # retries, so every one of these has to be argv: inheriting a default would silently
        # capture a different environment than the training lane measures.
        "--max-steps",
        str(pins.max_turns),
        "--timeout",
        f"{pins.episode_timeout_s:.0f}",
        # tau2's internal retry re-runs the whole simulation inside this subprocess, which
        # multiplies the batch wall clock past its deadline and appends the abandoned attempt's
        # episodes to the same results.json. Retry is the runner's job instead (--retry-failed),
        # one fresh save directory per attempt.
        "--max-retries",
        str(pins.tau2_max_retries),
        # A headless batch must never block on tau2's interactive resume prompt: it reads the
        # parent's stdin and would hang until the hard-kill deadline. `next_save_to` already keeps
        # every attempt on an unused directory, so there should be nothing to resume.
        "--auto-resume",
    ]


def save_to_name(entry: PoolEntry, domain: str, episode: int, attempt: int = 0) -> str:
    """tau2 `--save-to` slug, unique per candidate, domain, episode index, and attempt."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in entry.name)
    return f"real_{safe}_{domain}_e{episode}_a{attempt}"


def next_save_to(capture_dir: Path, entry: PoolEntry, domain: str, episode: int) -> str:
    """The first unused `--save-to` slug for a cell, so an attempt never lands on a checkpoint.

    tau2 reads an existing save directory as a checkpoint to resume: it prompts on stdin, and then
    refuses the run outright when the new task list is a subset of the checkpointed one. A resumed
    cell asks for exactly that subset (only the tasks with no row yet), so sharing a directory
    across attempts makes the second attempt impossible, which matters much more now that tau2's
    own retries are pinned off. Each attempt therefore gets its own directory and its own
    results.json, and the previous attempt's evidence stays on disk.

    Args:
        capture_dir: Directory holding the `tau2-bench/` clone.
        entry: The candidate this batch measures.
        domain: tau2 domain for this batch.
        episode: Episode index.

    Returns:
        A slug whose simulations directory does not exist yet.

    Raises:
        SystemExit: When the cell has already used every attempt slot.
    """
    simulations = capture_dir / "tau2-bench" / "data" / "simulations"
    for attempt in range(MAX_ATTEMPTS_PER_CELL):
        name = save_to_name(entry, domain, episode, attempt)
        if not (simulations / name).exists():
            return name
    raise SystemExit(
        f"candidate '{entry.name}' has used all {MAX_ATTEMPTS_PER_CELL} attempt slots for "
        f"{domain} episode {episode}. Something is failing on every attempt: read the newest "
        f"{simulations / save_to_name(entry, domain, episode, MAX_ATTEMPTS_PER_CELL - 1)} before "
        "clearing the older directories."
    )


def _stream_tokens(messages: Sequence[Tau2Message], role: str) -> tuple[int, int]:
    prompt = completion = 0
    for message in messages:
        if message.role != role or message.usage is None:
            continue
        prompt += message.usage.prompt_tokens
        completion += message.usage.completion_tokens
    return prompt, completion


def rows_from_results(
    results: Tau2Results,
    entry: PoolEntry,
    episode: int,
    by_task_id: dict[str, RealScenario],
    user_sim: PoolEntry,
    cohort: str = UNLABELED_COHORT,
) -> list[RealEpisodeRow]:
    """Turn one tau2 `results.json` into metered sidecar rows.

    Simulations of tasks outside the pinned split are dropped rather than recorded: tau2 filters
    by task id already, so their presence would mean the run drifted off the split.

    Args:
        results: The batch's parsed `results.json`.
        entry: The candidate this batch measured.
        episode: Episode index.
        by_task_id: The pinned scenarios this batch requested, keyed by tau2 task id.
        user_sim: The pinned user simulator, part of the environment.
        cohort: `ProtocolPins.label` for the pins the batch ran under.

    Returns:
        One row per on-split simulation.
    """
    rows: list[RealEpisodeRow] = []
    for sim in results.simulations:
        scenario = by_task_id.get(sim.task_id)
        if scenario is None:
            logger.warning("  dropping off-split simulation for task %s", sim.task_id)
            continue
        agent_in, agent_out = _stream_tokens(sim.messages, "assistant")
        user_in, user_out = _stream_tokens(sim.messages, "user")
        assistant = [m for m in sim.messages if m.role == "assistant"]
        rows.append(
            RealEpisodeRow(
                scenario_id=scenario.scenario_id,
                domain=scenario.domain,
                task_id=sim.task_id,
                task=scenario.task,
                provenance=scenario.provenance,
                model=entry.name,
                route=litellm_route(entry),
                episode=episode,
                reward=sim.reward_info.reward if sim.reward_info is not None else None,
                nl_assertion_reward=scenario.nl_assertion_reward,
                termination_reason=sim.termination_reason,
                duration_s=sim.duration,
                agent_input_tokens=agent_in,
                agent_output_tokens=agent_out,
                user_input_tokens=user_in,
                user_output_tokens=user_out,
                cost_usd_pool=entry.cost_usd(
                    TokenUsage(input_tokens=agent_in, output_tokens=agent_out)
                ),
                cost_usd_tau2_agent=sim.agent_cost,
                cost_usd_tau2_user=sim.user_cost,
                steps=sum(1 for m in sim.messages if m.tool_calls),
                call_seconds=[
                    m.generation_time_seconds
                    for m in assistant
                    if m.generation_time_seconds is not None
                ],
                replies=[m.text() for m in assistant if m.text()],
                user_sim=user_sim.name,
                cohort=cohort,
            )
        )
    return rows


def load_rows(path: Path) -> list[RealEpisodeRow]:
    """Read the resumable sidecar, or an empty list when nothing has run yet.

    A kill during the append leaves a torn final line. Refusing to load it would brick both the
    resume and `--write-matrix-only`, losing every episode already bought, so an unreadable line
    is dropped with a warning instead: its cell simply looks unrun and gets bought again, which
    costs one episode rather than the whole grid.
    """
    if not path.exists():
        return []
    rows: list[RealEpisodeRow] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(RealEpisodeRow.model_validate_json(line))
        except ValidationError:
            logger.warning(
                "  %s line %d is unreadable (torn write?); dropping it, so its cell will be "
                "treated as unrun and bought again",
                path,
                number,
            )
    return rows


def append_rows(path: Path, rows: Iterable[RealEpisodeRow]) -> None:
    """Append a batch as ONE write, then fsync, so a kill lands between batches not mid-line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(row.model_dump_json() + "\n" for row in rows)
    if not payload:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def spend_usd(rows: Iterable[RealEpisodeRow]) -> float:
    """Total run cost: our priced candidate side plus tau2's figure for the user simulator.

    Counts EVERY row, including an attempt a later retry superseded: both were bought.
    """
    return sum(row.cost_usd_pool + (row.cost_usd_tau2_user or 0.0) for row in rows)


def latest_per_cell(rows: Sequence[RealEpisodeRow]) -> list[RealEpisodeRow]:
    """One row per (scenario, candidate, episode): the last one written wins.

    Only a `--retry-failed` pass writes a second row for a cell, and the retry is what supersedes
    the attempt that failed. Collapsing here is what keeps a retried cell from being counted as two
    episodes; `spend_usd` still counts both attempts, because both were paid for.
    """
    latest: dict[tuple[str, str, int], RealEpisodeRow] = {}
    for row in rows:
        latest[row.key] = row
    return list(latest.values())


def resume_keys(rows: Sequence[RealEpisodeRow], retry_failed: bool) -> set[tuple[str, str, int]]:
    """The cells resume treats as already bought.

    With `retry_failed`, a cell whose latest row has no reward (tau2 could not score it: an
    infrastructure failure, a killed batch) counts as unrun, so this pass buys it once more. One
    pass, one retry: a cell that fails again simply stays unscored, and the operator decides
    whether to run the flag again.
    """
    return {
        row.key for row in latest_per_cell(rows) if not (retry_failed and row.reward is None)
    }


def to_matrix(rows: Sequence[RealEpisodeRow], pool: Sequence[PoolEntry]) -> OutcomeMatrix:
    """Sidecar rows -> OutcomeMatrix. Rows with no reward stay unscored, never zeroed."""
    named = {entry.name for entry in pool}
    rows = latest_per_cell(rows)
    outcomes = [
        ScenarioOutcome(
            scenario_id=row.scenario_id,
            task=row.task,
            model=row.model,
            episode=row.episode,
            reward=row.reward,
            success=row.reward is not None and row.reward >= 1.0,
            steps=row.steps,
            stop_reason=row.termination_reason,
            usage=TokenUsage(
                input_tokens=row.agent_input_tokens, output_tokens=row.agent_output_tokens
            ),
            cost_usd=row.cost_usd_pool,
            call_seconds=row.call_seconds,
            replies=row.replies,
        )
        for row in rows
        if row.model in named
    ]
    return OutcomeMatrix(pool=list(pool), outcomes=outcomes)


def price_order(pool: Sequence[PoolEntry]) -> list[PoolEntry]:
    """Cheapest candidate first, so a budget stop leaves the widest usable grid."""
    return sorted(pool, key=lambda entry: entry.cost_usd(_EPISODE_MIX))


def _run_batch(
    command: Sequence[str],
    capture_dir: Path,
    save_to: str,
    env: dict[str, str],
    timeout_s: float,
) -> Tau2Results | None:
    """Run one tau2 batch and read its results.json back, or None when it produced none.

    A failed batch is reported and skipped rather than raised: earlier batches are already on
    disk, and the rerun that resumes them will retry exactly this cell.

    `--save-to` is deterministic per cell, so a previous attempt may have left a `results.json`
    at this path. Accepting that file when tau2 dies before rewriting it would append episodes
    from an OLD run as though this run had just produced them, and the grid would report
    coverage it never bought. The modification time before the run is therefore the guard: a
    file that did not change is a leftover, not a result.
    """
    results = capture_dir / "tau2-bench" / "data" / "simulations" / save_to / "results.json"
    before = results.stat().st_mtime_ns if results.is_file() else None
    try:
        completed = subprocess.run(
            list(command),
            cwd=capture_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.error("  tau2 exceeded %.0fs and was killed; skipping this batch", timeout_s)
        return None
    except OSError as error:
        raise SystemExit(
            f"could not launch tau2 ({error}). Expected the CLI at {command[0]}; create the "
            "venv per packages/environment-capture/tau-bench/README.md § Setup, or pass "
            "--capture-dir pointing at an existing clone."
        ) from error

    if not results.is_file():
        logger.error("  no results.json (exit %d); stderr tail:", completed.returncode)
        logger.error("  %s", (completed.stderr or "")[-600:])
        return None
    if before is not None and results.stat().st_mtime_ns == before:
        logger.error(
            "  tau2 exited %d without rewriting %s; refusing to reuse the previous run's "
            "results. stderr tail:",
            completed.returncode,
            results,
        )
        logger.error("  %s", (completed.stderr or "")[-600:])
        return None
    if completed.returncode != 0:
        # Fresh output from a nonzero exit is a PARTIAL batch: the episodes it holds were paid
        # for and are worth keeping, and resume will retry whatever is still missing.
        logger.warning("  tau2 exited %d; keeping the partial batch it wrote", completed.returncode)
    return _parse_results(results)


def _parse_results(results: Path) -> Tau2Results | None:
    """Read `results.json`, keeping the simulations that validate and naming the ones that fail.

    tau2 owns this schema. Rejecting the whole file over one unexpected field would discard a
    batch of episodes already paid for, and because `--save-to` is deterministic the rerun would
    overwrite the evidence, so per-simulation validation is what keeps a schema surprise cheap.
    """
    try:
        payload = json.loads(results.read_text(encoding="utf-8"))
    except ValueError as error:
        logger.error("  %s is not valid JSON: %s", results, error)
        return None
    simulations = payload.get("simulations") if isinstance(payload, dict) else None
    if not isinstance(simulations, list):
        logger.error("  %s has no 'simulations' list", results)
        return None
    kept: list[Tau2Simulation] = []
    for index, raw in enumerate(simulations):
        try:
            kept.append(Tau2Simulation.model_validate(raw))
        except ValidationError as error:
            logger.error("  dropping simulation %d of %s: %s", index, results, error)
    return Tau2Results(simulations=kept)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--episodes", type=int, default=1, help="episode indices 0..N-1")
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these pool names")
    parser.add_argument(
        "--scenario",
        nargs="*",
        default=None,
        help="restrict to these '<domain>:<task_id>' scenario ids (smokes and single-cell reruns)",
    )
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="re-run the cells whose latest row has no reward, once (tau2's own retries are off)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="resolve the split and print the tau2 commands"
    )
    parser.add_argument(
        "--write-matrix-only", action="store_true", help="rebuild the matrix from rows.jsonl"
    )
    pins = parser.add_argument_group(
        "protocol pins",
        "The canonical real-tau2 protocol, shared with the training lane. Changing ANY of these "
        "defines a NEW capture cohort: the rows get a different cohort label, and sim_to_real.py "
        "refuses to pair them with canonical rows.",
    )
    pins.add_argument(
        "--max-turns",
        type=int,
        default=CANONICAL_MAX_TURNS,
        help=f"tau2 --max-steps; PIN {CANONICAL_MAX_TURNS} (tau2's own default is 200). "
        "Changing it is a new capture cohort.",
    )
    pins.add_argument(
        "--episode-timeout-s",
        type=float,
        default=CANONICAL_EPISODE_TIMEOUT_S,
        help=f"tau2 --timeout, per episode; PIN {CANONICAL_EPISODE_TIMEOUT_S:.0f} (tau2 has no "
        "timeout by default). Changing it is a new capture cohort.",
    )
    pins.add_argument(
        "--max-tokens",
        type=int,
        default=CANONICAL_MAX_TOKENS,
        help=f"candidate completion cap; PIN {CANONICAL_MAX_TOKENS}. 0 omits the key entirely, "
        "for a deployment that rejects max_tokens. Changing it is a new capture cohort.",
    )
    pins.add_argument(
        "--tau2-max-retries",
        type=int,
        default=CANONICAL_TAU2_MAX_RETRIES,
        help=f"tau2 --max-retries; PIN {CANONICAL_TAU2_MAX_RETRIES} (tau2's own default is 3), "
        "because retry belongs to --retry-failed. Changing it is a new capture cohort.",
    )
    pins.add_argument(
        "--user-sim",
        default=DEFAULT_USER_SIM,
        help=f"pool name of the user simulator; PIN {DEFAULT_USER_SIM}. It is the environment, "
        "not a candidate. Changing it is a new capture cohort.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    rows_path = args.out_dir / "rows.jsonl"
    matrix_path = args.out_dir / "matrix.json"

    pool = load_pool(args.pool).models
    by_name = {entry.name: entry for entry in pool}
    rows = load_rows(rows_path)

    if args.write_matrix_only:
        # A run refuses to append across cohorts, so a mixed file here was merged by hand. Salvage
        # is still the right default (the episodes were bought), but the matrix must say so.
        present = sorted({row.cohort for row in rows})
        if len(present) > 1:
            logger.warning(
                "these %d rows span %d capture cohorts %s, which measure different environments. "
                "The matrix will pool them; split rows.jsonl by cohort if that is not what you "
                "want.",
                len(rows),
                len(present),
                present,
            )
        to_matrix(rows, pool).save(matrix_path)
        logger.info("wrote %s from %d rows", matrix_path, len(rows))
        return 0

    if args.user_sim not in by_name:
        logger.error("user simulator '%s' is not in the pool %s", args.user_sim, sorted(by_name))
        return 2
    user_sim = by_name[args.user_sim]

    try:
        pins = ProtocolPins(
            max_turns=args.max_turns,
            episode_timeout_s=args.episode_timeout_s,
            max_tokens=args.max_tokens,
            tau2_max_retries=args.tau2_max_retries,
            user_sim=args.user_sim,
        )
    except ValidationError as error:
        logger.error("protocol pins are out of range: %s", error)
        return 2
    if pins.is_canonical:
        logger.info("protocol pins: cohort '%s' (canonical)", pins.label)
    else:
        logger.warning(
            "NON-CANONICAL protocol pins: cohort '%s', canonical is '%s'. These rows are a "
            "separate capture cohort; sim_to_real.py will refuse to pair them with canonical rows.",
            pins.label,
            ProtocolPins().label,
        )
    foreign = sorted({row.cohort for row in rows} - {pins.label})
    if foreign:
        logger.error(
            "%s already holds rows from cohort(s) %s, but this run is cohort '%s'. One rows.jsonl "
            "holds one cohort: point --out-dir at a new directory for this pin set, or match the "
            "pins that produced the existing rows.",
            rows_path,
            foreign,
            pins.label,
        )
        return 2

    # A typo in --only would otherwise be silently ignored and quietly buy a smaller grid than
    # the operator asked for, which is only discovered after the spend.
    unknown_models = sorted(set(args.only or []) - set(by_name))
    if unknown_models:
        logger.error("--only names models that are not in the pool: %s", unknown_models)
        return 2

    scenarios = load_pinned_scenarios(args.capture_dir)
    if args.scenario:
        wanted = set(args.scenario)
        unknown = sorted(wanted - {s.scenario_id for s in scenarios})
        if unknown:
            logger.error("--scenario ids are not in the pinned eval split: %s", unknown)
            return 2
        scenarios = [s for s in scenarios if s.scenario_id in wanted]
    by_domain: dict[str, list[RealScenario]] = {}
    for scenario in scenarios:
        by_domain.setdefault(scenario.domain, []).append(scenario)
    logger.info(
        "pinned eval split: %d scenarios over %s (%d NL-assertion scored)",
        len(scenarios),
        ", ".join(f"{d} {len(v)}" for d, v in sorted(by_domain.items())),
        sum(1 for s in scenarios if s.nl_assertion_reward),
    )

    done = resume_keys(rows, args.retry_failed)
    spent = spend_usd(rows)
    logger.info(
        "resume: %d rows on disk, $%.2f already spent, budget stop $%.0f",
        len(rows),
        spent,
        args.budget_usd,
    )
    if args.retry_failed:
        logger.info(
            "--retry-failed: %d cell(s) whose latest row is unscored will be bought again",
            len(latest_per_cell(rows)) - len(done),
        )

    # The user simulator is the environment, so it is never also measured as a candidate.
    candidates = [
        entry
        for entry in price_order(pool)
        if entry.name != user_sim.name and (not args.only or entry.name in args.only)
    ]
    if not candidates:
        logger.error(
            "no candidates left to run: --only %s selects nothing outside the pinned user "
            "simulator '%s'. Pool models are %s.",
            args.only,
            user_sim.name,
            sorted(by_name),
        )
        return 2

    for episode in range(args.episodes):
        for entry in candidates:
            for domain, domain_scenarios in sorted(by_domain.items()):
                missing = [
                    s for s in domain_scenarios if (s.scenario_id, entry.name, episode) not in done
                ]
                if not missing:
                    continue
                save_to = next_save_to(args.capture_dir, entry, domain, episode)
                command = batch_command(
                    args.capture_dir,
                    entry,
                    user_sim,
                    domain,
                    [s.task_id for s in missing],
                    save_to,
                    args.max_concurrency,
                    pins,
                )
                if args.dry_run:
                    logger.info("  would run: %s", " ".join(command))
                    continue
                if spent >= args.budget_usd:
                    logger.warning("BUDGET STOP at $%.2f; leaving the grid partial", spent)
                    to_matrix(load_rows(rows_path), pool).save(matrix_path)
                    return 0
                logger.info(
                    "  %s / %s e%d: %d tasks -> %s",
                    entry.name,
                    domain,
                    episode,
                    len(missing),
                    save_to,
                )
                env = build_env(args.capture_dir, entry, user_sim, dict(os.environ))
                payload = _run_batch(
                    command,
                    args.capture_dir,
                    save_to,
                    env,
                    batch_timeout_s(len(missing), args.max_concurrency, pins.episode_timeout_s),
                )
                if payload is None:
                    continue
                batch = rows_from_results(
                    payload,
                    entry,
                    episode,
                    {s.task_id: s for s in missing},
                    user_sim,
                    pins.label,
                )
                append_rows(rows_path, batch)
                done.update(row.key for row in batch)
                spent += spend_usd(batch)
                scored = [row.reward for row in batch if row.reward is not None]
                logger.info(
                    "  -> %d sims, mean reward %s, running total $%.2f",
                    len(batch),
                    f"{sum(scored) / len(scored):.3f}" if scored else "n/a",
                    spent,
                )

    if args.dry_run:
        logger.info("dry run: nothing executed, nothing spent")
        return 0
    to_matrix(load_rows(rows_path), pool).save(matrix_path)
    logger.info("wrote %s; total spend $%.2f", matrix_path, spent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
