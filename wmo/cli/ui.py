"""Shared terminal UX for the `wmo` CLI."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

import click
import typer
from rich.console import Console
from rich.control import Control
from rich.markup import escape
from rich.segment import ControlType

from wmo.cli.ui_tables import models_table  # noqa: F401
from wmo.common.config import (
    PROVIDER_ENV_VARS,
    ModelInfo,
    upsert_env_var,
)
from wmo.common.providers.base import ProviderConfig, ProviderKind, VerifyResult

# A reader takes a fully-rendered prompt string and returns the user's typed line.
PromptReader = Callable[[str], str]

# Stage glyphs reused by the animated and plain reporters.
_CHECK = "[green]✓[/green]"

# Serve providers offered in the wizard picker, with the model ids each supports. The first model
# in each list is the suggested default. Keep these in sync with the provider backends.
_PROVIDER_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-5.5", "gpt-5.5-pro", "gpt-5.4", "gpt-5.4-mini"],
    "anthropic": [
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "claude-opus-5",
    ],
    # Keep this picker curated to inference profiles verified by the normal interactive flow.
    # The full canonical registry remains available to flags and programmatic callers.
    "bedrock": ["claude-opus-4-8", "claude-opus-4-7", "claude-haiku-4-5", "claude-opus-5"],
    # openai_responses (the Responses API) stays flag-only (`wmo build --provider
    # openai_responses`); the wizard list keeps to the four everyday backends.
    "azure": ["gpt-5.5", "gpt-5.4"],
}
_DEFAULT_REGIONS: dict[str, str] = {"bedrock": "us-east-1"}


def select_provider_and_model(
    console: Console,
    ask: PromptReader,
    ask_secret: PromptReader,
    *,
    default_provider: str | None,
    default_model: str | None,
    default_region: str | None,
    interactive: bool,
    check: Callable[[ProviderConfig], VerifyResult],
) -> tuple[str, str, str | None]:
    """The wizard's serve-provider block: pick provider + model (+ region), creds, live verify.

    Providers with credentials present are annotated and the first becomes the suggested default
    (none otherwise); a failed live ping loops back to the picker with the failed pick as the
    retry default. Returns (provider, model, region).
    """
    from wmo.common.providers.models import resolve_provider_model

    providers = list(_PROVIDER_MODELS)
    with_creds = [p for p in providers if has_credentials(p)]
    # Name the actual variable so a key inherited from the shell (e.g. exported in ~/.zshrc)
    # is traceable — "api key exists" alone reads as a mystery when .env doesn't have it.
    notes = {p: creds_note(p) for p in with_creds}
    provider_default = default_provider or (with_creds[0] if with_creds else None)
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
        ensure_credentials(console, ask_secret, provider)
        model = _select(
            console,
            ask,
            "Serve model type",
            _PROVIDER_MODELS[provider],
            default_model,
            interactive=interactive,
            collapsed=2,
        )
        region = None
        if provider == "bedrock":
            region_default = default_region or _DEFAULT_REGIONS.get(provider)
            region = _prompt_text(console, ask, "AWS region", region_default) or None
        # Live ping now, not at the end: a bad key or model id loops straight back to the
        # picker (the failed pick becomes the suggested retry default).
        console.print(f"verifying {provider}…")
        model_spec = resolve_provider_model(ProviderKind(provider), model)
        ping = check(
            ProviderConfig(
                kind=ProviderKind(provider),
                model_type=model_spec.model_type,
                model=model_spec.model_id,
                region=region,
            )
        )
        if ping.ok:
            console.print(f"  {_CHECK} {provider} ({escape(model)}) reachable")
            return provider, model, region
        console.print(
            f"  [red]✗ {provider} ({escape(model)}) failed[/red]: {escape(ping.detail or '')}"
        )
        console.print("  [yellow]fix the credentials or pick a different provider/model[/yellow]")
        provider_default = provider


def _picker_fits(console: Console, row_count: int) -> bool:
    """Whether the arrow-key picker can run: a real TTY and every row fits on screen at once
    (the repaint moves the cursor up over the block, which breaks if the block scrolled)."""
    return console.is_terminal and sys.stdin.isatty() and row_count + 2 <= console.size.height


def _split_keys(raw: str) -> list[str]:
    """Split one getchar() read into individual key sequences.

    Fast key repeat (holding an arrow) can deliver several escape sequences in a single raw
    read; treating the batch as one unknown sequence would drop them all as inert.
    """
    keys: list[str] = []
    i = 0
    while i < len(raw):
        if raw[i] != "\x1b":
            keys.append(raw[i])
            i += 1
            continue
        if raw[i : i + 2] == "\x1b[":
            j = i + 2
            while j < len(raw) and not ("@" <= raw[j] <= "~"):
                j += 1
            keys.append(raw[i : j + 1])
            i = j + 1
        elif raw[i : i + 2] == "\x1bO":
            keys.append(raw[i : i + 3])
            i += 3
        else:
            keys.append("\x1b")
            i += 1
    return keys


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


def _arrow_select(
    console: Console, rows: list[str], index: int, hidden_rows: list[str] | None = None
) -> int:
    """Drive an up/down picker over pre-rendered `rows`; return the chosen index.

    Keys come from click.getchar(): raw mode handled portably (termios on Unix, msvcrt on
    Windows), one whole escape sequence per call, Ctrl-C raised as KeyboardInterrupt for
    click's usual clean abort. EOF (closed stdin) aborts rather than spinning.

    `hidden_rows` collapse behind a dim "… N more" row; navigating (or jump-selecting) onto it
    reveals the rest in place. The returned index is into rows + hidden_rows.
    """
    visible = list(rows)
    hidden = list(hidden_rows or [])
    painted_height = 0

    def current_rows() -> list[str]:
        more = [f"[dim]… {len(hidden)} more[/dim]"] if hidden else []
        return visible + more

    def paint(current: int) -> None:
        nonlocal painted_height
        shown = current_rows()
        if painted_height:
            console.control(Control.move(y=-painted_height))
        for i, row in enumerate(shown):
            console.control(Control((ControlType.ERASE_IN_LINE, 2)))
            pointer = "[bold cyan]\u276f[/bold cyan]" if i == current else " "
            console.print(f" {pointer} {row}", highlight=False, no_wrap=True, overflow="ellipsis")
        painted_height = len(shown)

    while True:
        paint(index)
        try:
            seq = click.getchar()
        except EOFError:
            raise typer.Abort() from None
        if seq == "":
            raise typer.Abort()
        for key in _split_keys(seq):
            index, accepted = _step_selection(_decode_key(key), index, len(current_rows()))
            if hidden and index == len(visible):
                # Landed on the "more" row: reveal the rest in place; the highlight stays
                # put, now on the first revealed option.
                visible.extend(hidden)
                hidden = []
                accepted = False
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
    collapsed: int | None = None,
) -> str:
    """Pick one of `options`: an arrow-key picker on a real TTY, else a numbered prompt.

    The Enter-default is `default` when present in `options`; a non-matching default falls
    back to the first option, and None means no Enter-default at all (the provider picker: a
    default exists only when creds do; the model picker: the choice is always explicit).
    `notes` adds a dim annotation after an option's name (e.g. "api key exists"). `collapsed`
    shows only the first N options in the arrow picker behind a "… more" row (the numbered
    fallback always lists everything, so scripted input is unaffected).
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
        rows = [row(opt) for opt in options]
        if collapsed is not None and start < collapsed < len(options):
            return options[_arrow_select(console, rows[:collapsed], start, rows[collapsed:])]
        return options[_arrow_select(console, rows, start)]

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


def creds_note(provider: str) -> str:
    """Picker annotation for a provider whose credentials are present, naming what was found."""
    env_vars = _provider_env_vars(provider)
    return f"{env_vars[0]} set" if len(env_vars) == 1 else "creds set"


def has_credentials(provider: str) -> bool:
    """Offline presence check: every credential env var for `provider` is set (not validated)."""
    env_vars = _provider_env_vars(provider)
    return bool(env_vars) and all(os.environ.get(var) for var in env_vars)


def ensure_credentials(console: Console, ask_secret: PromptReader, provider: str) -> None:
    """Prompt for any missing credential env vars and persist entered values to `.env`.

    Presence only — the live ping that confirms the creds actually work happens once before
    the build (see `wmo build`). Enter skips a var, leaving it to the shell environment.
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


def select_option(
    console: Console,
    label: str,
    options: list[str],
    *,
    notes: dict[str, str] | None = None,
    default: str | None = None,
    reader: PromptReader | None = None,
) -> str:
    """Pick one of `options` (arrow-key picker on a TTY, numbered prompt otherwise).

    `default` is the Enter-default; None keeps the historic behavior of requiring an explicit
    pick. `notes` annotates an option with a dim suffix (e.g. which credential was found).
    """
    ask = reader if reader is not None else (lambda text: console.input(text))
    return _select(console, ask, label, options, default, interactive=True, notes=notes)


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


def _parse_int(raw: str) -> int | None:
    """Parse a base-10 integer, or None. Unlike `str.isdigit`, this rejects unicode digit
    characters (e.g. superscripts) that `isdigit()` accepts but `int()` rejects with ValueError."""
    try:
        return int(raw)
    except ValueError:
        return None


def explicit_param(ctx: typer.Context, param: str) -> bool:
    """Whether `param` was explicitly passed on the command line.

    Compared by enum NAME: typer vendors click, so its ParameterSource enum is not
    click.core's class and an identity check would silently never match.
    """
    source = ctx.get_parameter_source(param)
    return source is not None and source.name == "COMMANDLINE"
