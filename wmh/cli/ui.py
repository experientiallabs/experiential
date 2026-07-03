"""Terminal UX for the `wmh` CLI: guided creation/selection flows, an animated build pipeline, and
the interactive play REPL.

Everything that talks to `rich` lives here so the engine stays headless. Responsibilities:

- `run_build_wizard` interactively fills in any missing `wmh build` inputs (name, traces, provider,
  region, budget), so a bare `wmh build` becomes a guided creation flow.
- `select_model` shows a numbered picker so a user can choose which built world model to run when
  `--name` is omitted and several exist.
- `RichBuildReporter` implements `wmh.engine.reporting.BuildReporter`, turning build events into a
  guided, animated pipeline (stage lines + a live GEPA rollout progress bar) on a TTY, and into
  plain one-line-per-event output when piped (non-TTY), so logs stay legible.
- `run_play_repl` drives the human-in-the-loop demo: the user types actions, the world model
  answers, and the evolving session state (scratchpad + history) is rendered each turn.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

import click
import typer
from pydantic import BaseModel
from rich.console import Console
from rich.control import Control
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.segment import ControlType
from rich.table import Table

from wmh.config import (
    PROVIDER_ENV_VARS,
    ModelInfo,
    normalize_name,
    upsert_env_var,
    validate_name,
)
from wmh.core.types import Action, ActionKind, Session
from wmh.engine.play import PlayTurn, parse_action, play_turn
from wmh.engine.world_model import WorldModel
from wmh.providers import verify_all, verify_embedder
from wmh.providers.base import ProviderConfig, ProviderKind, VerifyResult

# A reader takes a fully-rendered prompt string and returns the user's typed line.
PromptReader = Callable[[str], str]

# Stage glyphs reused by the animated and plain reporters.
_CHECK = "[green]✓[/green]"

# Serve providers offered in the wizard picker, with the model ids each supports. The first model
# in each list is the suggested default. Keep these in sync with the provider backends.
_PROVIDER_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-5.5", "gpt-5.5-pro", "gpt-5.4"],
    "anthropic": ["claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"],
    "bedrock": [
        "us.anthropic.claude-opus-4-8",
        "us.anthropic.claude-opus-4-7",
        # haiku needs the dated inference-profile id; the undated alias is rejected by Bedrock
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    ],
    # openai_responses (the Responses API) stays flag-only (`wmh build --provider
    # openai_responses`); the wizard list keeps to the four everyday backends.
    "azure": ["gpt-5.5", "gpt-5.4"],
}
_DEFAULT_REGIONS: dict[str, str] = {"bedrock": "us-east-1"}

# Embedders offered in the wizard, with the embeddings-model ids each provider-backed one supports
# (None = the offline hashing embedder, no model). First entry is the suggested default.
_EMBEDDERS: dict[str, list[str] | None] = {
    "hashing": None,
    "bedrock": ["amazon.titan-embed-text-v2:0"],
    "openai": ["text-embedding-3-small", "text-embedding-3-large"],
    "azure": ["text-embedding-3-small", "text-embedding-3-large"],
}


class BuildParams(BaseModel):
    """The fully-resolved inputs for a build, as collected by the creation wizard."""

    name: str
    file: str | None = None
    vendor: str | None = None
    # None = not chosen yet: the wizard suggests the first provider with credentials present,
    # and the non-interactive path falls back to bedrock.
    provider: str | None = None
    model: str = "us.anthropic.claude-opus-4-8"
    region: str | None = None
    gepa_budget: int = 50
    train_split: float = 0.8
    embed_provider: str = "hashing"
    embed_model: str | None = None
    embed_dim: int = 512


def run_build_wizard(
    console: Console,
    defaults: BuildParams,
    reader: PromptReader | None = None,
    verify: Callable[[ProviderConfig], VerifyResult] | None = None,
    verify_embed: Callable[[ProviderConfig], VerifyResult] | None = None,
) -> BuildParams:
    """Guided creation flow: prompt for each build input, pre-filled with `defaults`.

    Returns a resolved `BuildParams`. Any value already set in `defaults` (i.e. passed as a flag)
    becomes the suggested default the user can accept with Enter. The serve provider and any
    provider-backed embedder are live-pinged right after their model pick — a failure loops back
    to the picker instead of surfacing after the wizard. `verify`/`verify_embed` exist so tests
    can stub the pings. Raises `ValueError` if no trace source (file or vendor) is provided.
    """
    interactive = reader is None
    check = verify if verify is not None else (lambda cfg: verify_all([cfg])[0])
    check_embed = verify_embed if verify_embed is not None else verify_embedder
    base_ask = reader if reader is not None else (lambda text: console.input(text))
    base_secret = (
        reader if reader is not None else (lambda text: console.input(text, password=True))
    )

    def _eof_aborts(read: PromptReader) -> PromptReader:
        # Exhausted piped stdin (or Ctrl-D) aborts the wizard cleanly ("Aborted.", exit 1)
        # instead of leaking an EOFError traceback from whichever prompt was being read.
        def wrapped(text: str) -> str:
            try:
                return read(text)
            except EOFError:
                raise typer.Abort() from None

        return wrapped

    ask = _eof_aborts(base_ask)
    ask_secret = _eof_aborts(base_secret)

    console.print(
        Panel(
            "Let's create a world model. Press Enter to accept the [dim]default[/dim] in brackets.",
            title="[bold cyan]wmh build[/bold cyan]",
            border_style="cyan",
        )
    )

    # Whitespace is dash-joined ('tau bench' -> 'tau-bench') rather than rejected; a name
    # that is still unsafe (path separators, ...) re-prompts with the validation message
    # instead of escaping as a ValueError traceback. An invalid name arriving via --name is
    # dropped from the suggested default — otherwise Enter would re-offer it forever.
    try:
        name_default = validate_name(normalize_name(defaults.name))
    except ValueError:
        name_default = None
    while True:
        raw_name = _prompt_text(console, ask, "Name this world model", name_default)
        name = normalize_name(raw_name)
        try:
            validate_name(name)
        except ValueError as err:
            console.print(f"[red]{escape(str(err))}[/red]")
            continue
        if name != raw_name:
            console.print(f"  [dim]using[/dim] {escape(name)}")
        break

    file = defaults.file
    vendor = defaults.vendor
    if not file and not vendor:
        # A trace source is required, so re-prompt on empty input rather than erroring out.
        while not file:
            file = _prompt_text(
                console,
                ask,
                "Path to exported traces (OTLP-JSON / JSONL)",
                None,
                example="examples/tau-bench/traces.otel.jsonl",
            )
            if not file:
                console.print("[red]a traces path is required (or pass --vendor)[/red]")

    # Serve provider: providers with credentials already present are annotated, and the first
    # of them is the suggested default (no default when none have creds). After the pick, any
    # missing credential is prompted for and persisted to .env rather than failing mid-build.
    providers = list(_PROVIDER_MODELS)
    with_creds = [p for p in providers if _has_credentials(p)]
    # Name the actual variable so a key inherited from the shell (e.g. exported in ~/.zshrc)
    # is traceable — "api key exists" alone reads as a mystery when .env doesn't have it.
    notes = {p: _creds_note(p) for p in with_creds}
    provider_default = defaults.provider or (with_creds[0] if with_creds else None)
    while True:
        provider = _select(
            console,
            ask,
            "Serve provider",
            providers,
            provider_default,
            interactive=interactive,
            notes=notes,
        )
        _ensure_credentials(console, ask_secret, provider)
        model = _select(
            console,
            ask,
            "Serve model id",
            _PROVIDER_MODELS[provider],
            defaults.model,
            interactive=interactive,
        )
        region = None
        if provider == "bedrock":
            region_default = defaults.region or _DEFAULT_REGIONS.get(provider)
            region = _prompt_text(console, ask, "AWS region", region_default) or None
        # Live ping now, not after the whole wizard: a bad key or model id loops straight
        # back to the picker (the failed pick becomes the suggested retry default).
        console.print(f"verifying {provider}…")
        ping = check(ProviderConfig(kind=ProviderKind(provider), model=model, region=region))
        if ping.ok:
            console.print(f"  {_CHECK} {provider} ({escape(model)}) reachable")
            break
        console.print(
            f"  [red]✗ {provider} ({escape(model)}) failed[/red]: {escape(ping.detail or '')}"
        )
        console.print("  [yellow]fix the credentials or pick a different provider/model[/yellow]")
        provider_default = provider

    gepa_budget = _prompt_int(
        console,
        ask,
        "GEPA rollout budget (more rollouts = better prompt, higher cost/time)",
        defaults.gepa_budget,
    )

    # Retrieval phi embedder: default offline hashing (no creds). A provider-backed embedder is
    # picked from the list and prompts for its embeddings-model id; phi dimensionality keeps its
    # default (the index and query embedders must agree, so it is not a wizard knob).
    embed_default = defaults.embed_provider
    while True:
        embed_provider = _select(
            console,
            ask,
            "Embedder",
            list(_EMBEDDERS),
            embed_default,
            interactive=interactive,
        )
        embed_model = defaults.embed_model
        embed_models = _EMBEDDERS[embed_provider]
        if embed_models is None:
            break  # offline hashing embedder: nothing to verify
        if embed_provider != provider:
            _ensure_credentials(console, ask_secret, embed_provider)
        embed_model = _select(
            console,
            ask,
            "Embeddings model id",
            embed_models,
            embed_model or embed_models[0],
            interactive=interactive,
        )
        # Same inline ping for the embed path, with the embeddings model and phi dimension
        # stamped on (mirrors HarnessConfig.embed_provider_config).
        console.print(f"verifying embed:{embed_provider}…")
        ping = check_embed(
            ProviderConfig(
                kind=ProviderKind(embed_provider),
                model=embed_model,
                embed_model=embed_model,
                embed_dim=defaults.embed_dim,
                region=region if embed_provider == "bedrock" else None,
            )
        )
        if ping.ok:
            console.print(f"  {_CHECK} embed:{embed_provider} ({escape(embed_model)}) reachable")
            break
        console.print(
            f"  [red]✗ embed:{embed_provider} ({escape(embed_model)}) failed[/red]: "
            f"{escape(ping.detail or '')}"
        )
        console.print("  [yellow]fix the credentials or pick a different embedder[/yellow]")
        embed_default = embed_provider

    return BuildParams(
        name=name,
        file=file,
        vendor=vendor,
        provider=provider,
        model=model,
        region=region,
        gepa_budget=gepa_budget,
        train_split=defaults.train_split,
        embed_provider=embed_provider,
        embed_model=embed_model,
        embed_dim=defaults.embed_dim,
    )


def _picker_fits(console: Console, row_count: int) -> bool:
    """Whether the arrow-key picker can run: a real TTY and every row fits on screen at once
    (the repaint moves the cursor up over the block, which breaks if the block scrolled)."""
    return console.is_terminal and sys.stdin.isatty() and row_count + 2 <= console.size.height


def _decode_key(seq: str) -> str:
    """Map one click.getchar() sequence to a picker key.

    getchar returns a whole escape sequence per call ('\x1b[A'), so nothing can desync: plain
    and application-mode arrows decode to 'up'/'down', any other escape sequence (modified
    arrows, PgUp, Delete, bare ESC) is the inert 'esc', and a plain character passes through.
    """
    if seq in ("\x1b[A", "\x1bOA"):
        return "up"
    if seq in ("\x1b[B", "\x1bOB"):
        return "down"
    if seq.startswith("\x1b"):
        return "esc"
    return seq


def _step_selection(key: str, index: int, count: int) -> tuple[int, bool]:
    """Picker key reducer: next highlighted index and whether the selection was accepted.

    Arrows (and vi j/k) move with wraparound, Enter accepts the highlight, a digit jump-selects
    that option; anything else is inert.
    """
    if key in ("up", "k"):
        return (index - 1) % count, False
    if key in ("down", "j"):
        return (index + 1) % count, False
    if key in ("\r", "\n"):
        return index, True
    # ASCII-decimal only: '²'.isdigit() is True but int('²') raises.
    if key.isascii() and key.isdecimal() and 1 <= int(key) <= count:
        return int(key) - 1, True
    return index, False


def _arrow_select(console: Console, rows: list[str], index: int) -> int:
    """Drive an up/down picker over pre-rendered `rows`; return the chosen index.

    Keys come from click.getchar(): raw mode handled portably (termios on Unix, msvcrt on
    Windows), one whole escape sequence per call, Ctrl-C raised as KeyboardInterrupt for
    click's usual clean abort. EOF (closed stdin) aborts rather than spinning.
    """
    painted = False

    def paint(current: int) -> None:
        nonlocal painted
        if painted:
            console.control(Control.move(y=-len(rows)))
        for i, row in enumerate(rows):
            console.control(Control((ControlType.ERASE_IN_LINE, 2)))
            pointer = "[bold cyan]\u276f[/bold cyan]" if i == current else " "
            console.print(f" {pointer} {row}", highlight=False, no_wrap=True, overflow="ellipsis")
        painted = True

    while True:
        paint(index)
        try:
            seq = click.getchar()
        except EOFError:
            raise typer.Abort() from None
        if seq == "":
            raise typer.Abort()
        index, accepted = _step_selection(_decode_key(seq), index, len(rows))
        if accepted:
            paint(index)
            return index


def _select(
    console: Console,
    ask: PromptReader,
    label: str,
    options: list[str],
    default: str | None,
    *,
    interactive: bool = False,
    notes: dict[str, str] | None = None,
) -> str:
    """Pick one of `options`: an arrow-key picker on a real TTY, else a numbered prompt.

    The Enter-default is `default` when present in `options`; a non-matching default falls
    back to the first option, and None means no Enter-default at all (the provider picker: a
    default exists only when creds do; the model picker: the choice is always explicit).
    `notes` adds a dim annotation after an option's name (e.g. "api key exists").
    """
    if default in options:
        chosen_default = default
    elif default is not None:
        chosen_default = options[0]
    else:
        chosen_default = None
    notes = notes or {}

    def row(opt: str) -> str:
        note = f"  [dim]({notes[opt]})[/dim]" if opt in notes else ""
        marker = "  [dim](default)[/dim]" if opt == chosen_default else ""
        return f"{escape(opt)}{note}{marker}"

    if interactive and _picker_fits(console, len(options)):
        console.print(f"[bold]{label}[/bold] [dim](up/down + Enter)[/dim]:")
        start = options.index(chosen_default) if chosen_default is not None else 0
        return options[_arrow_select(console, [row(opt) for opt in options], start)]

    console.print(f"[bold]{label}[/bold]:")
    for i, opt in enumerate(options, start=1):
        console.print(f"  [cyan]{i}[/cyan]. {row(opt)}")
    prompt = f"[dim]\\[{escape(chosen_default)}][/dim] > " if chosen_default is not None else "> "
    while True:
        raw = ask(prompt).strip()
        if not raw and chosen_default is not None:
            return chosen_default
        choice = _parse_int(raw)
        if choice is not None and 1 <= choice <= len(options):
            return options[choice - 1]
        if raw in options:  # allow typing the option name directly
            return raw
        console.print(f"[red]pick 1-{len(options)} or an option name[/red]")


def _provider_env_vars(provider: str) -> list[str]:
    """The env vars `provider` reads its credentials from ([] for unknown/offline kinds)."""
    try:
        return PROVIDER_ENV_VARS[ProviderKind(provider)]
    except (ValueError, KeyError):
        return []


def _creds_note(provider: str) -> str:
    """Picker annotation for a provider whose credentials are present, naming what was found."""
    env_vars = _provider_env_vars(provider)
    return f"{env_vars[0]} set" if len(env_vars) == 1 else "creds set"


def _has_credentials(provider: str) -> bool:
    """Offline presence check: every credential env var for `provider` is set (not validated)."""
    env_vars = _provider_env_vars(provider)
    return bool(env_vars) and all(os.environ.get(var) for var in env_vars)


def _ensure_credentials(console: Console, ask_secret: PromptReader, provider: str) -> None:
    """Prompt for any missing credential env vars and persist entered values to `.env`.

    Presence only — the live ping that confirms the creds actually work happens once before
    the build (see `wmh build`). Enter skips a var, leaving it to the shell environment.
    """
    for var in _provider_env_vars(provider):
        if os.environ.get(var):
            console.print(f"  {_CHECK} {var} is set")
            continue
        prompt = f"  [bold]{var}[/bold] [dim](saved to .env; Enter to skip)[/dim]: "
        value = ask_secret(prompt).strip()
        if value:
            try:
                upsert_env_var(var, value)
            except (ValueError, OSError) as err:
                # Persistence refused (symlinked .env, O_NOFOLLOW's ELOOP on a swapped link,
                # unwritable dir, ...): the session still gets the credential.
                os.environ[var] = value
                console.print(f"  [yellow]{var} not saved: {escape(str(err))}[/yellow]")
            else:
                console.print(f"  {_CHECK} {var} saved to .env")
        else:
            console.print(f"  [yellow]{var} still unset[/yellow]")


def select_model(
    console: Console, infos: list[ModelInfo], reader: PromptReader | None = None
) -> str:
    """Show a numbered picker and return the chosen model name.

    Re-prompts on invalid input. With a single model it returns that name without prompting.
    """
    if len(infos) == 1:
        return infos[0].name
    ask = reader if reader is not None else (lambda text: console.input(text))
    notes = {
        info.name: f"held-out {info.held_out_accuracy:.2f}"
        for info in infos
        if info.held_out_accuracy is not None
    }
    return _select(
        console,
        ask,
        "Select a world model",
        [info.name for info in infos],
        None,
        interactive=reader is None,
        notes=notes,
    )


def _prompt_text(
    console: Console,
    ask: PromptReader,
    label: str,
    default: str | None,
    *,
    example: str | None = None,
) -> str:
    # Escape interpolated values: "default" or anything with [...] is valid rich markup and would
    # otherwise be swallowed (rendered invisibly) instead of shown. A prompt with no default can
    # carry a grey `example` hint so the user sees the expected shape of the answer.
    if default:
        suffix = f" [dim]\\[{escape(default)}][/dim]"
    elif example:
        suffix = f" [dim](e.g. {escape(example)})[/dim]"
    else:
        suffix = ""
    value = ask(f"[bold]{label}[/bold]{suffix}: ").strip()
    return value or (default or "")


def _prompt_int(console: Console, ask: PromptReader, label: str, default: int) -> int:
    while True:
        raw = ask(f"[bold]{label}[/bold] [dim]\\[{default}][/dim]: ").strip()
        if not raw:
            return default
        value = _parse_int(raw)
        if value is not None and value >= 0:
            return value
        console.print("[red]enter a non-negative whole number[/red]")


def _parse_int(raw: str) -> int | None:
    """Parse a base-10 integer, or None. Unlike `str.isdigit`, this rejects unicode digit
    characters (e.g. superscripts) that `isdigit()` accepts but `int()` rejects with ValueError."""
    try:
        return int(raw)
    except ValueError:
        return None


class RichBuildReporter:
    """A `BuildReporter` that renders the build as a guided pipeline.

    On a TTY it shows stage lines and a live progress bar for GEPA rollouts (with the running
    held-out score). When output is piped (`console.is_terminal` is false) it degrades to a single
    plain line per event — no spinners, no carriage returns — so captured logs stay readable.
    """

    def __init__(self, console: Console, model_name: str) -> None:
        self._console = console
        self._name = model_name
        self._tty = console.is_terminal
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None

    def ingest_done(self, traces: int, steps: int) -> None:
        self._stage(f"ingested {traces} traces → normalized {steps} steps")

    def split_done(self, train: int, test: int) -> None:
        self._stage(f"split {train} train / {test} held-out traces")

    def index_done(self, steps: int) -> None:
        self._stage(f"indexed {steps} steps into the replay buffer")

    def optimize_start(self, budget: int) -> None:
        self._stage(f"optimizing env prompt with GEPA (budget {budget} metric calls)")
        if self._tty and budget > 0:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("{task.fields[score]}"),
                TimeElapsedColumn(),
                console=self._console,
                # Not transient: the finished bar stays in scrollback, snapped to 100% with the
                # true final call count and elapsed time (see optimize_done).
                transient=False,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                "GEPA metric calls", total=budget, score="score n/a"
            )

    def rollout(self, done: int, budget: int, score: float | None) -> None:
        label = f"avg fidelity {score:.3f}" if score is not None else "score n/a"
        if self._progress is not None and self._task_id is not None:
            # The budget is an estimate (GEPA treats max_metric_calls as a soft cap and can
            # overshoot). Never pin a live run at 100% — rich freezes the elapsed clock once
            # completed == total, which reads as "stuck". Grow the total instead.
            total = budget if done < budget else done + 1
            self._progress.update(self._task_id, completed=done, total=total, score=label)
        elif not self._tty:
            # Non-TTY: emit a sparse heartbeat so long runs still show life without flooding logs.
            if done == 1 or done % 10 == 0 or done >= budget:
                progress = (
                    f"{done}/{budget}" if done <= budget else f"{done} (budget target {budget})"
                )
                self._console.print(f"  GEPA metric call {progress} ({label})")

    def optimize_done(self, held_out_accuracy: float, frontier_size: int, rollouts: int) -> None:
        if self._progress is not None:
            if self._task_id is not None and rollouts > 0:
                # 100% exactly when GEPA is actually done: the endpoint can't be predicted (soft
                # cap + per-iteration costs decided at runtime), so the bar snaps to the ACTUAL
                # final call count on the completion event instead of guessing during the run.
                self._progress.update(self._task_id, completed=rollouts, total=rollouts)
            self._progress.stop()
            self._progress = None
            self._task_id = None
        self._stage(
            f"GEPA done: held-out {held_out_accuracy:.3f}, "
            f"{frontier_size} frontier candidates, {rollouts} rollouts used"
        )

    def _stage(self, message: str) -> None:
        self._console.print(f"{_CHECK} {message}")

    def close(self) -> None:
        """Stop the live progress bar if it is still running (e.g. the build raised mid-GEPA)."""
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None

    def __enter__(self) -> RichBuildReporter:
        return self

    def __exit__(self, *exc: object) -> None:
        # Always tear down the live Progress so an exception during the build doesn't leave a
        # spinning bar that corrupts the terminal.
        self.close()


def build_summary_panel(info: ModelInfo, root: str) -> Panel:
    """A tidy panel summarizing a freshly built world model (shown after `wmh build`)."""
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="bold")
    table.add_column()
    table.add_row("name", info.name)
    table.add_row("artifact", root)
    table.add_row("serve provider", f"{info.serve_provider} ({info.serve_model})")
    if info.held_out_accuracy is not None:
        table.add_row("held-out accuracy", f"{info.held_out_accuracy:.3f}")
    if info.rollouts_used is not None:
        table.add_row("rollouts used", str(info.rollouts_used))
    if info.frontier_size is not None:
        table.add_row("frontier candidates", str(info.frontier_size))
    return Panel(
        table,
        title=f"[bold green]world model ready: {info.name}[/bold green]",
        subtitle="serve it with `wmh serve` or step into it with `wmh play`",
        border_style="green",
    )


def models_table(infos: list[ModelInfo]) -> Table:
    """A table of every built world model (for `wmh list`)."""
    table = Table(title="world models")
    table.add_column("name", style="bold")
    table.add_column("serve provider")
    table.add_column("held-out", justify="right")
    table.add_column("rollouts", justify="right")
    table.add_column("frontier", justify="right")
    for info in infos:
        table.add_row(
            info.name,
            f"{info.serve_provider} ({info.serve_model})",
            "-" if info.held_out_accuracy is None else f"{info.held_out_accuracy:.3f}",
            "-" if info.rollouts_used is None else str(info.rollouts_used),
            "-" if info.frontier_size is None else str(info.frontier_size),
        )
    return table


# --- interactive play REPL -----------------------------------------------------------------------

_PLAY_HELP = (
    "[bold]You are the agent.[/bold] Type an action and the world model answers:\n"
    '  [cyan]get_user {"id": "u1"}[/cyan]   a tool call with JSON arguments\n'
    "  [cyan]list_flights[/cyan]            a tool call with no arguments\n"
    "  [cyan]say I am stuck[/cyan]          a free-text message to the environment\n"
    "Commands: [cyan]:state[/cyan] show session state  ·  [cyan]:help[/cyan]  ·  "
    "[cyan]:quit[/cyan] (or Ctrl-D) to exit"
)


_AGENT_PROMPT = "[bold]agent>[/bold] "


def run_play_repl(
    console: Console,
    world_model: WorldModel,
    model_name: str,
    task: str | None,
    reader: PromptReader | None = None,
) -> None:
    """Run the human-in-the-loop demo against `world_model`.

    `reader` is an optional `PromptReader` (`(prompt_text) -> line`) used to source input — injected
    in tests, defaults to the console's prompt. The loop ends on `:quit`, EOF, or KeyboardInterrupt.
    """
    ask = reader if reader is not None else console.input
    session = world_model.new_session(task=task)
    console.print(
        Panel(
            _PLAY_HELP,
            title=f"[bold]playing[/bold] {model_name}",
            subtitle=f"task: {task}" if task else "no task set",
            border_style="cyan",
        )
    )

    while True:
        try:
            line = ask(_AGENT_PROMPT)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return
        line = line.strip()
        if not line:
            continue
        if line in {":quit", ":q", ":exit"}:
            console.print("[dim]bye[/dim]")
            return
        if line in {":help", ":h"}:
            console.print(_PLAY_HELP)
            continue
        if line == ":state":
            _render_state(console, world_model.get_session(session.id))
            continue
        _handle_action(console, world_model, session.id, line)


def _handle_action(console: Console, world_model: WorldModel, session_id: str, line: str) -> None:
    """Parse + step one typed action, rendering the observation (or a friendly error).

    A failed step (e.g. a provider/network error) is reported and swallowed so the REPL keeps the
    session alive instead of crashing the whole interactive run.
    """
    try:
        action = parse_action(line)
    except ValueError as exc:
        console.print(f"[red]parse error[/red]: {exc}")
        return
    try:
        with console.status("[dim]world model thinking…[/dim]", spinner="dots"):
            turn = play_turn(world_model, session_id, action)
    except Exception as exc:  # noqa: BLE001 - keep the REPL alive; surface the failure to the user
        console.print(f"[red]step failed[/red]: {exc}")
        return
    _render_turn(console, turn)


def _render_turn(console: Console, turn: PlayTurn) -> None:
    console.print(f"[bold cyan]→ you[/bold cyan]: {_action_text(turn.action)}")
    style = "red" if turn.observation.is_error else "green"
    label = "error" if turn.observation.is_error else "observation"
    console.print(
        Panel(
            turn.observation.content or "[dim](empty)[/dim]",
            title=f"[bold]{label}[/bold]",
            border_style=style,
        )
    )


def _render_state(console: Console, session: Session) -> None:
    scratchpad = session.state.scratchpad or "[dim](empty)[/dim]"
    body = f"[bold]task[/bold]: {session.task or '(none)'}\n"
    body += f"[bold]turns[/bold]: {len(session.history)}\n\n"
    body += f"[bold]scratchpad[/bold]:\n{scratchpad}"
    console.print(Panel(body, title="session state", border_style="blue"))


def _action_text(action: Action) -> str:
    if action.kind == ActionKind.TOOL_CALL:
        return f"{action.name}({action.arguments})"
    return f'message: "{action.content}"'
