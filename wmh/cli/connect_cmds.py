"""Context connector commands: `wmh connect` and the `wmh context` family.

`wmh connect` lists every registered connector with its credential status; `wmh connect
<service>` runs that connector's interactive auth flow (browser OAuth, device code, or a pasted
token) and saves the credential to the user-global connector store. `wmh context pull` turns a
connected service into a named, replayable bundle under `<dir>/.wmh/context/`; `list`/`show`
inspect bundles; `attach` renders one into a world model's knowledge base.
"""

from __future__ import annotations

import webbrowser
from collections import Counter
from datetime import UTC, datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from wmh.config import ARTIFACT_DIR, ArtifactPaths, WorldModelStore, load_config
from wmh.connect import (
    BundleManifest,
    ConnectError,
    ConnectorAuth,
    ConnectUI,
    ContextItem,
    ContextStore,
    PullQuery,
    delete_connector_auth,
    get_connector,
    list_connectors,
    load_connector_auth,
    render_markdown,
    resolve_env_token,
    save_connector_auth,
    token_env_vars,
)
from wmh.engine.knowledge import DEFAULT_RENDER_BUDGET, KnowledgeBase

_console = Console()
_CHECK = "[green]✓[/green]"

context_app = typer.Typer(
    help="Pull context bundles from connected services and attach them to world models.",
    no_args_is_help=True,
)

# Module-level singletons: a typer.Argument call can't be a default inline (ruff B008).
_CONNECT_SERVICE = typer.Argument(
    None, help="Connector to run (see `wmh connect` for the list); omit to show them all."
)
_PULL_SERVICE = typer.Argument(..., help="Connected service to pull from.")
_BUNDLE_NAME = typer.Argument(..., help="Context bundle name (see `wmh context list`).")


def connect(
    service: str = _CONNECT_SERVICE,
    remove: bool = typer.Option(False, "--remove", help="Delete the stored credential instead."),
) -> None:
    """Connect a context service (bare invocation lists every connector and its status)."""
    if service is None:
        _print_connector_table()
        return
    if service not in list_connectors():
        known = ", ".join(list_connectors())
        raise typer.BadParameter(f"no connector named {service!r}; available: {known}")
    if remove:
        if delete_connector_auth(service):
            _console.print(f"{_CHECK} Removed the stored {service} credential.")
        else:
            _console.print(f"Nothing stored for {service}; nothing to remove.")
        return
    connector = get_connector(service)
    try:
        auth = connector.connect(_connect_ui())
        identity = auth.account or connector.verify(auth)
    except ConnectError as error:
        _console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error
    if auth.account is None:
        auth = auth.model_copy(update={"account": identity})
    resolved = resolve_env_token(service)
    if auth.kind == "token" and resolved is not None and auth.access_token == resolved[1].strip():
        # The documented env-token contract: verified and used as-is, never persisted.
        _console.print(
            f"{_CHECK} Connected {connector.label} as [bold]{identity}[/bold] "
            f"using ${resolved[0]}; nothing written to disk"
        )
        return
    path = save_connector_auth(service, auth)
    _console.print(f"{_CHECK} Connected {connector.label} as [bold]{identity}[/bold] ({path})")


@context_app.command("pull")
def context_pull(
    service: str = _PULL_SERVICE,
    target: str = typer.Option(
        None,
        "--target",
        help="Repo 'owner/name', channel, calendar id, folder id, or site domain.",
    ),
    query: str = typer.Option(None, "--query", help="Free text or service search syntax."),
    since: str = typer.Option(None, "--since", help="ISO-8601 lower bound on item time."),
    until: str = typer.Option(None, "--until", help="ISO-8601 upper bound on item time."),
    limit: int = typer.Option(100, "--limit", help="Maximum items to pull."),
    name: str = typer.Option(
        None, "--name", help="Bundle name (default: <service>-<UTC timestamp>)."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace an existing bundle with this name."
    ),
    directory: str = typer.Option(
        ".", "--dir", help="Project dir; bundles land under <dir>/.wmh/context/."
    ),
) -> None:
    """Pull content from a connected service into a named context bundle."""
    if service not in list_connectors():
        known = ", ".join(list_connectors())
        raise typer.BadParameter(f"no connector named {service!r}; available: {known}")
    auth = _auth_or_fail(service)
    connector = get_connector(service)
    pull_query = PullQuery(target=target, query=query, since=since, until=until, limit=limit)
    try:
        items = connector.pull(auth, pull_query)
    except ConnectError as error:
        _console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error
    bundle_name = name or f"{service}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    manifest = BundleManifest(
        name=bundle_name,
        connector=service,
        query=pull_query,
        pulled_at=datetime.now(UTC).isoformat(timespec="seconds"),
        item_count=len(items),
        account=auth.account,
    )
    store = ContextStore(directory)
    try:
        bundle_dir = store.save(manifest, items, overwrite=overwrite)
    except FileExistsError as error:
        raise typer.BadParameter(
            f"context bundle {bundle_name!r} already exists at {store.bundle_dir(bundle_name)}; "
            "pass --overwrite to replace it, or --name to pick another name"
        ) from error
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    _console.print(
        f"{_CHECK} Pulled {len(items)} items from {service} into [bold]{bundle_dir}[/bold]"
    )
    if items:
        table = Table(show_header=True, header_style="bold")
        table.add_column("kind")
        table.add_column("items", justify="right")
        for kind, count in sorted(Counter(item.kind.value for item in items).items()):
            table.add_row(kind, str(count))
        _console.print(table)
    _console.print(f"attach it: `wmh context attach {bundle_name} --model <name>`")


@context_app.command("list")
def context_list(
    directory: str = typer.Option(
        ".", "--dir", help="Project dir; bundles live under <dir>/.wmh/context/."
    ),
) -> None:
    """List every saved context bundle with its provenance."""
    manifests = ContextStore(directory).list_bundles()
    if not manifests:
        _console.print("no context bundles yet; run `wmh context pull <service>` to create one")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("bundle")
    table.add_column("connector")
    table.add_column("items", justify="right")
    table.add_column("pulled at")
    table.add_column("account")
    for manifest in manifests:
        table.add_row(
            manifest.name,
            manifest.connector,
            str(manifest.item_count),
            manifest.pulled_at,
            manifest.account or "",
        )
    _console.print(table)


@context_app.command("show")
def context_show(
    name: str = _BUNDLE_NAME,
    limit: int = typer.Option(20, "--limit", help="Maximum items to render."),
    directory: str = typer.Option(
        ".", "--dir", help="Project dir; bundles live under <dir>/.wmh/context/."
    ),
) -> None:
    """Show one bundle: its manifest and the pulled items (titles, kinds, dates)."""
    manifest, items = _load_bundle(directory, name)
    _console.print(f"[bold]{manifest.name}[/bold]: {manifest.item_count} items")
    _console.print(f"  connector: {manifest.connector}")
    _console.print(f"  pulled at: {manifest.pulled_at}")
    if manifest.account:
        _console.print(f"  account: {manifest.account}")
    table = Table(show_header=True, header_style="bold")
    table.add_column("title")
    table.add_column("kind")
    table.add_column("created")
    table.add_column("updated")
    for item in items[:limit]:
        table.add_row(item.title, item.kind.value, item.created_at or "", item.updated_at or "")
    _console.print(table)
    if len(items) > limit:
        _console.print(f"... and {len(items) - limit} more (raise --limit to see them)")


@context_app.command("attach")
def context_attach(
    name: str = _BUNDLE_NAME,
    model: str = typer.Option(
        ..., "--model", help="World model whose knowledge base receives the bundle."
    ),
    max_chars: int = typer.Option(
        None,
        "--max-chars",
        help="Cap the rendered markdown; whole items are dropped from the tail, visibly.",
    ),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir holding the world models."),
    directory: str = typer.Option(
        ".", "--dir", help="Project dir; bundles live under <dir>/.wmh/context/."
    ),
) -> None:
    """Render a bundle to markdown and write it into a model's knowledge base.

    The bundle lands as `knowledge/context-<bundle>.md`, so it renders into the env prompt's
    KNOWLEDGE BASE section on the next serve. Models without knowledge support (built without
    `--knowledge` and holding no `knowledge/` dir) are refused with instructions to opt in.
    """
    manifest, items = _load_bundle(directory, name)
    try:
        model_dir = WorldModelStore(root).resolve(model)
    except (FileNotFoundError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    paths = ArtifactPaths(model_dir)
    if not load_config(model_dir).knowledge and not paths.knowledge.is_dir():
        _console.print(
            f"[red]world model {model!r} has no knowledge base: rebuild with "
            f"`wmh build --knowledge`, or create {paths.knowledge}/ to opt in, "
            "then re-run this command[/red]"
        )
        raise typer.Exit(code=1)
    text = render_markdown(manifest, items, max_chars=max_chars)
    kb = KnowledgeBase(paths.knowledge)
    file_name = f"context-{name}.md"
    kb.write_file(file_name, text)
    _console.print(
        f"{_CHECK} Wrote [bold]{paths.knowledge / file_name}[/bold] ({len(text):,} chars)"
    )
    _print_budget_state(kb)


# -- helpers ---------------------------------------------------------------------------------


def _print_connector_table() -> None:
    """The bare `wmh connect` view: every registered connector and its credential status."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("connector")
    table.add_column("service")
    table.add_column("status")
    for name in list_connectors():
        table.add_row(name, get_connector(name).label, _status(name))
    _console.print(table)
    _console.print(
        "connect one: [bold]wmh connect <name>[/bold]; "
        "then pull: [bold]wmh context pull <name>[/bold]"
    )


def _status(name: str) -> str:
    """One connector's credential status line for the table."""
    resolved = resolve_env_token(name)
    if resolved is not None:
        return f"env token (${resolved[0]})"
    try:
        auth = load_connector_auth(name)
    except ConnectError:
        return f"[red]corrupt entry; re-run `wmh connect {name}`[/red]"
    if auth is None:
        return "[dim]not connected[/dim]"
    if not auth.access_token:
        # An aborted OAuth flow can persist a placeholder entry before tokens arrive.
        return f"[yellow]incomplete; re-run `wmh connect {name}`[/yellow]"
    return f"connected as {auth.account}" if auth.account else f"connected ({auth.kind})"


def _connect_ui() -> ConnectUI:
    """The rich-console ConnectUI the CLI hands to connectors (they never print themselves)."""

    def open_url(url: str) -> None:
        _console.print(f"Approve access in your browser:\n  [bold]{url}[/bold]")
        webbrowser.open(url)

    def present_code(verification_uri: str, user_code: str) -> None:
        _console.print(Panel(f"[bold]{user_code}[/bold]", title="enter this code", expand=False))
        _console.print(f"at [bold]{verification_uri}[/bold]")

    def prompt_secret(label: str) -> str:
        return typer.prompt(label, hide_input=True, default="", show_default=False)

    def info(message: str) -> None:
        _console.print(message)

    return ConnectUI(
        open_url=open_url, present_code=present_code, prompt_secret=prompt_secret, info=info
    )


def _auth_or_fail(service: str) -> ConnectorAuth:
    """The stored credential for `service`, or a usage error saying how to connect."""
    try:
        auth = load_connector_auth(service)
    except ConnectError as error:
        raise typer.BadParameter(str(error)) from error
    if auth is None:
        env_hint = " or ".join(f"${var}" for var in token_env_vars(service))
        raise typer.BadParameter(
            f"not connected to {service}; run `wmh connect {service}` first (or set {env_hint})"
        )
    return auth


def _load_bundle(directory: str, name: str) -> tuple[BundleManifest, list[ContextItem]]:
    """Load one bundle, turning store errors into usage errors."""
    try:
        return ContextStore(directory).load(name)
    except (FileNotFoundError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error


def _print_budget_state(kb: KnowledgeBase) -> None:
    """Say how the serve-time render budget treats the knowledge base after this write."""
    sections = [content.strip() for content in kb.files().values() if content.strip()]
    total = sum(len(section) for section in sections) + 4 * len(sections)  # headers + joins
    if total > DEFAULT_RENDER_BUDGET:
        _console.print(
            f"[yellow]the knowledge base now holds ~{total:,} chars, over its "
            f"{DEFAULT_RENDER_BUDGET:,}-char render budget: the prompt render will be truncated "
            "with a loud marker; curate the files or re-attach with a smaller --max-chars[/yellow]"
        )
    else:
        _console.print(
            f"renders into the KNOWLEDGE BASE prompt section within its "
            f"{DEFAULT_RENDER_BUDGET:,}-char budget (~{total:,} chars used)"
        )


def register(app: typer.Typer) -> None:
    """Attach `wmh connect` and the `wmh context` sub-app to the root CLI."""
    app.command("connect")(connect)
    app.add_typer(context_app, name="context")
