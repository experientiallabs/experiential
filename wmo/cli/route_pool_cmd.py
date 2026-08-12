"""Route student registration and static-policy commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm

from wmo.common.config import ARTIFACT_DIR, WorldModelStore

if TYPE_CHECKING:
    # Type-only: real imports are local to the commands and helpers that construct or inspect
    # these values, so importing this module never pulls the optimize/engine/env/distill/pool
    # bodies behind it.
    from wmo.common.providers.pool import PoolEntry
    from wmo.common.vendor.waterfall import ChatMaxTokensField

from wmo.cli.route_constants import (
    _DEFAULT_POOL_PATH,
    _MAX_TOKENS_FIELDS,
    _POLICY_FILENAME,
)

_console = Console()


def student(
    card_dir: str = typer.Argument(
        ...,
        help="The distillation run dir, or an adapter version dir: whichever holds "
        "model_card.json.",
    ),
    input_per_mtok: float = typer.Option(
        ...,
        "--input-per-mtok",
        min=0.0,
        help="Prompt-token price at the serving endpoint, USD per 1M tokens. Required: an "
        "unpriced candidate reports $0 and a cost-aware policy would route everything to it.",
    ),
    output_per_mtok: float = typer.Option(
        ...,
        "--output-per-mtok",
        min=0.0,
        help="Completion-token price at the serving endpoint, USD per 1M tokens.",
    ),
    name: str = typer.Option(
        "student", "--name", help="Pool handle: what policy artifacts and request logs call it."
    ),
    pool: str = typer.Option(
        _DEFAULT_POOL_PATH, "--pool", help="Candidate pool TOML to add the entry to."
    ),
    endpoint: str = typer.Option(
        None,
        "--endpoint",
        help="OpenAI-compatible base URL. Default: Tinker's serving endpoint.",
    ),
    api_key_env: str = typer.Option(
        None,
        "--api-key-env",
        help="Env var holding the endpoint's API key. Default: TINKER_API_KEY on Tinker's own "
        "endpoint; on any other --endpoint the provider's WMO_ENDPOINT_API_KEY fallback is used, "
        "so a Tinker key is never sent to a host you named.",
    ),
    chat_max_tokens_field: str = typer.Option(
        None,
        "--chat-max-tokens-field",
        help="Output-budget parameter the endpoint accepts: max_tokens | max_completion_tokens. "
        "Default: max_tokens on Tinker's endpoint, max_completion_tokens on any other.",
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Skip the confirmation when an entry of this name already exists."
    ),
) -> None:
    r"""Add a distilled student to the candidate pool, so the router can select it.

    The keystone step between training and serving: a run produces a `tinker://` adapter, and this
    turns it into a `\[\[model]]` entry the sweep measures, the fitter routes to, and the endpoint
    calls, with no hand-edited TOML in between:

        wmo optimize route student .wmo/distill/support --input-per-mtok 0.1 --output-per-mtok 0.4

    On Tinker's own endpoint the entry reads its credential from `TINKER_API_KEY`, so export that
    before serving. Point `--endpoint` somewhere else and the Tinker defaults do NOT follow: the
    entry falls back to `WMO_ENDPOINT_API_KEY` and to `max_completion_tokens`, so a Tinker key is
    never sent to a host you named. `--api-key-env` and `--chat-max-tokens-field` set either
    explicitly.

    To serve the student on its own with no measurement at all, follow this with
    `wmo optimize route pin <world-model> --model student`; to have the router CHOOSE between the
    student and the rest of the roster, run `wmo optimize route fit` on a matrix that covers both.

    Args:
        card_dir: Distillation run or adapter directory containing `model_card.json`.
        input_per_mtok: Prompt-token serving price in USD per million tokens.
        output_per_mtok: Completion-token serving price in USD per million tokens.
        name: Candidate-pool handle for the student.
        pool: Candidate pool TOML to update.
        endpoint: Optional OpenAI-compatible serving endpoint.
        api_key_env: Optional environment variable holding that endpoint's credential.
        chat_max_tokens_field: Output-token parameter accepted by the endpoint.
        yes: Whether to replace an existing candidate without prompting.

    Raises:
        typer.BadParameter: The card, endpoint, pricing, or pool entry is invalid.
    """
    from wmo.common.core.locks import FileLockTimeout
    from wmo.common.providers.pool import upsert_pool_entry
    from wmo.optimize.model.store import MODEL_CARD_FILE, DistillModelCard, student_pool_entry

    card_path = Path(card_dir) / MODEL_CARD_FILE
    if not card_path.is_file():
        raise typer.BadParameter(
            f"no {MODEL_CARD_FILE} at {card_path}; pass a distillation run directory (the one "
            "holding config.toml and metrics.jsonl) or an adapter version directory "
            "(.wmo/adapters/<name>/vN)"
        )
    if endpoint is not None and not endpoint.strip():
        # `--endpoint "$UNSET_VAR"` is the way this happens. Falling back to Tinker's endpoint
        # would silently serve a different host than the script meant to name.
        raise typer.BadParameter(
            "--endpoint is empty; give the OpenAI-compatible base URL, or drop the flag to use "
            "Tinker's serving endpoint"
        )
    if api_key_env is not None and not api_key_env.strip():
        # Same accident as an empty --endpoint. An empty string reaches `pool_provider` as a
        # falsy api_key_env, which it reads as "no explicit credential" and skips its own
        # unset-variable check, so the misconfiguration would only surface as a 401 at request
        # time with no hint.
        raise typer.BadParameter(
            "--api-key-env is empty; name the environment variable holding the endpoint's key, "
            "or drop the flag to use the provider's default credentials"
        )
    if chat_max_tokens_field is not None and chat_max_tokens_field not in _MAX_TOKENS_FIELDS:
        raise typer.BadParameter(
            f"unknown --chat-max-tokens-field {chat_max_tokens_field!r}; use "
            f"{' or '.join(_MAX_TOKENS_FIELDS)}"
        )
    try:
        card = DistillModelCard.model_validate_json(card_path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise typer.BadParameter(f"cannot read the model card at {card_path}: {exc}") from exc
    try:
        entry = student_pool_entry(
            card,
            name=name,
            input_per_mtok=input_per_mtok,
            output_per_mtok=output_per_mtok,
            endpoint=endpoint,
            api_key_env=api_key_env,
            chat_max_tokens_field=cast("ChatMaxTokensField | None", chat_max_tokens_field),
        )
    except ValidationError as exc:
        raise typer.BadParameter(f"cannot build a pool entry for '{name}': {exc}") from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    pool_path = Path(pool)
    if _pool_has(pool_path, name) and not yes and not _confirm_replace(pool_path, name):
        _console.print(
            f"left {pool_path} unchanged; pass --yes to replace '{name}' (comments in the file "
            "are not preserved by a replacement), or --name <other> to keep both"
        )
        raise typer.Exit(0)
    if _pool_disabled(pool_path, name):
        # Same rule as the registry writer: `enabled = false` is an explicit operator edit,
        # and replacing the entry must not silently put the candidate back into selection.
        entry = entry.model_copy(update={"enabled": False})
        _console.print(
            f"[dim]'{name}' is disabled in the roster (enabled = false); keeping it disabled[/dim]"
        )
    try:
        written = upsert_pool_entry(entry, pool_path)
    except FileLockTimeout as exc:
        # Nothing is wrong with the flags, so this is not a BadParameter: another writer is in the
        # way. Exit non-zero (and say to retry) so a script does not read it as a registration.
        _console.print(f"[red]pool busy[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    verb = "replaced" if written.replaced else "added"
    rewrite_note = (
        "\n  the roster was rewritten, so its comments are gone" if written.rewritten else ""
    )
    _console.print(
        f"[green]✓[/green] {verb} pool candidate [bold]{name}[/bold]{rewrite_note} -> {pool_path}\n"
        f"  {card.base_model} adapter at {entry.model}\n"
        f"  ${input_per_mtok:g}/${output_per_mtok:g} per 1M in/out tokens, "
        f"{_credential_note(entry)}\n"
        f"  serve it directly: wmo optimize route pin <world-model> --model {name}",
        soft_wrap=True,
    )


def _credential_note(entry: PoolEntry) -> str:
    """How this entry authenticates, so the summary never names a key it will not send."""
    if entry.api_key_env is not None:
        return f"credential from {entry.api_key_env}"
    return "credential from WMO_ENDPOINT_API_KEY (the custom-endpoint fallback)"


def _pool_has(path: Path, name: str) -> bool:
    """Whether `path` already carries an entry called `name` (False when there is no pool yet)."""
    from wmo.common.providers.pool import load_pool

    if not path.is_file():
        return False
    try:
        return any(entry.name == name for entry in load_pool(path).models)
    except (ValueError, FileNotFoundError):
        # An unreadable pool is upsert_pool_entry's error to raise, with its own message; do not
        # pre-empt it here with a confirmation prompt about an entry we cannot see.
        return False


def _pool_disabled(path: Path, name: str) -> bool:
    """Whether `path` carries an entry called `name` with `enabled = false` (else False)."""
    from wmo.common.providers.pool import load_pool

    if not path.is_file():
        return False
    try:
        return any(entry.name == name and not entry.enabled for entry in load_pool(path).models)
    except (ValueError, FileNotFoundError):
        return False


def _confirm_replace(path: Path, name: str) -> bool:
    """Confirm repointing an existing pool handle; a non-interactive run declines."""
    try:
        return Confirm.ask(f"Replace the existing '{name}' entry in {path}?", default=False)
    except EOFError:
        return False


def pin(
    world_model: str = typer.Argument(
        None, help="Built world model whose endpoint serves this policy. Default: the only one."
    ),
    model: str = typer.Option(
        ...,
        "--model",
        help="Pool entry every request goes to (a `wmo optimize route student` name).",
    ),
    pool: str = typer.Option(
        _DEFAULT_POOL_PATH, "--pool", help="Candidate pool TOML to snapshot into the policy."
    ),
    root: str = typer.Option(ARTIFACT_DIR, "--root", help="Project dir holding the built models."),
    out: str = typer.Option(
        None, "--out", help="Override where the policy JSON lands (default: the model's own dir)."
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Skip the confirmation when a policy is already installed."
    ),
) -> None:
    """Serve one pool model as an endpoint, with no matrix and no fit.

    A `kind="static"` policy sends every request to `--model`, which is all a single distilled
    student needs to be reachable through the OpenAI-compatible endpoint:

        wmo optimize route student .wmo/distill/support --input-per-mtok 0.1 --output-per-mtok 0.4
        wmo optimize route pin support --model student
        wmo serve --name support

    The policy is written to the world model's artifact dir, because that is where `wmo serve`
    looks for one. This is the honest "before" state the routing story is told against: a static
    endpoint has learned nothing and saves nothing, and `GET /v1/endpoints/<name>/savings` will
    say so. Replace it with `wmo optimize route fit` on a real outcome matrix to let the router
    choose per request.

    Args:
        world_model: Built world model whose endpoint receives the static policy.
        model: Enabled pool entry selected for every request.
        pool: Candidate pool TOML to snapshot into the policy.
        root: Project artifact directory containing built models.
        out: Optional policy destination, defaulting to the model directory.
        yes: Whether to replace an installed policy without prompting.

    Raises:
        typer.BadParameter: The model, pool, or policy destination is invalid.
    """
    from wmo.common.providers.pool import load_pool
    from wmo.optimize.routing.policy import RoutingPolicy

    store = WorldModelStore(root)
    try:
        model_dir = store.resolve(world_model)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        # `resolve` says "pass --name", the option `wmo serve`/`play`/`demo` carry. This command
        # has no --name (its --model is the POOL entry), so say what a user of it actually types.
        names = store.list_names()
        raise typer.BadParameter(
            f"multiple world models built ({', '.join(names)}); name one as the WORLD_MODEL "
            f"argument, e.g. `wmo optimize route pin {names[0]} --model {model}`"
        ) from exc
    pool_path = Path(pool)
    try:
        roster = load_pool(pool_path)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        raise typer.BadParameter(f"cannot read the pool at {pool_path}: {exc}") from exc
    active = roster.enabled_models()
    if all(entry.name != model for entry in active):
        if any(entry.name == model for entry in roster.models):
            raise typer.BadParameter(
                f"pool model '{model}' is disabled (enabled = false) in {pool_path}; flip it "
                "back on to pin it"
            )
        available = ", ".join(entry.name for entry in active)
        raise typer.BadParameter(
            f"no pool model named '{model}' in {pool_path}; available: {available}"
        )
    out_path = Path(out) if out else model_dir / _POLICY_FILENAME
    if out and out_path.resolve() != (model_dir / _POLICY_FILENAME).resolve():
        # The foot-gun that bit both bench-defaults lanes (2026-07-29): an --out
        # anywhere but <model dir>/policy.json succeeds, prints the same cheerful
        # line, and leaves the file serving actually reads holding whatever policy
        # it held before (a different FILENAME in the right dir misses identically).
        # The pin still lands where asked; the operator is told serving will not
        # see it.
        _console.print(
            f"[yellow]![/yellow] --out is outside {model_dir}; `wmo serve --name "
            f"{model_dir.name}` and GET /config read {model_dir / _POLICY_FILENAME}, "
            "which this pin does NOT update"
        )
    if out_path.is_file() and not yes and not _confirm_overwrite(out_path):
        _console.print(f"left {out_path} in place")
        raise typer.Exit(0)
    policy = RoutingPolicy(
        kind="static",
        default_model=model,
        # Only the enabled roster travels: the policy's pool is what serving may construct
        # providers for, and a turned-off candidate must not become reachable through an
        # endpoint pinned after the operator turned it off.
        pool=active,
        fitted_from=f"pinned to {model} from {pool_path} (no outcome matrix)",
    )
    policy.save(out_path)
    _console.print(
        f"[green]✓[/green] pinned endpoint [bold]{model_dir.name}[/bold] to "
        f"[bold]{model}[/bold] -> {out_path}\n"
        f"  every request goes to {model}; nothing is measured and nothing is saved yet\n"
        f"  serve it: wmo serve --name {model_dir.name}\n"
        "  to let the router choose per request instead, replace this with "
        "`wmo optimize route fit <matrix.json>`",
        soft_wrap=True,
    )


def _confirm_overwrite(path: Path) -> bool:
    """Confirm replacing an installed policy; a non-interactive run declines.

    Worth asking about: the file being replaced may be a fitted knn policy, whose evidence bank
    sidecar this static policy will not use and does not remove.
    """
    try:
        return Confirm.ask(f"Replace the policy already at {path}?", default=False)
    except EOFError:
        return False


def register(app: typer.Typer) -> None:
    """Register route pool-management commands on their parent Typer app.

    Args:
        app: Parent Typer application that owns the route command group.
    """
    app.command("student", help="Add a distilled student to the routing candidate pool.")(student)
    app.command("pin", help="Install a static policy that always uses one pool model.")(pin)
