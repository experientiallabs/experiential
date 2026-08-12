"""Artifact listing, download, serving, and model-resolution commands."""

from __future__ import annotations

import time
import urllib.error
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.filesize import decimal
from rich.markup import escape
from rich.progress import BarColumn, Progress, TextColumn

from wmo.common.config import (
    ARTIFACT_DIR,
    WorldModelStore,
    load_config,
    validate_name,
)

if TYPE_CHECKING:
    from wmo.common.providers import ProviderConfig
    from wmo.common.providers.base import Provider

from wmo.cli.command_common import _credential_hint

_console = Console()
_CHECK = "[green]✓[/green]"
_DOWNLOAD_BENCHMARKS = typer.Argument(
    None, help="Benchmark bundles to download, or 'all'. Omit for a picker."
)


def list_models(root: str = typer.Option(ARTIFACT_DIR, help="Project dir to list.")) -> None:
    """List every world model built under the project dir.

    An empty listing names the directory it searched, because `--root` defaults to a
    cwd-relative `.wmo` and "nothing here" and "wrong directory" look identical otherwise.
    An artifact that cannot be read is listed as `unreadable` with its reason, so one broken
    `config.toml` costs you that one row instead of the whole listing.

    Args:
        root: Project artifact directory to inspect.

    Raises:
        typer.BadParameter: The requested project root is a file instead of a directory.
    """
    from wmo.cli.ui import models_table

    if Path(root).is_file():
        raise typer.BadParameter(
            f"--root {root} is a file, not a project dir; pass the dir holding models/ "
            f"(the default is `{ARTIFACT_DIR}`)"
        )
    store = WorldModelStore(root)
    infos = store.list_info()
    if not infos:
        # Name the trace export too: `wmo build --name <name>` alone has no corpus to build from.
        _console.print(
            f"[yellow]no world models built under {store.models_dir}[/yellow]; run "
            "`wmo build --name <name> --file <traces export>`"
        )
        return
    _console.print(models_table(infos))
    for info in infos:
        if info.error is not None:
            _console.print(f"  [red]✗ {info.name}[/red]: {escape(info.error)}")


def download(
    benchmarks: list[str] = _DOWNLOAD_BENCHMARKS,
    force: bool = typer.Option(False, "--force", help="Overwrite existing local files."),
) -> None:
    """Download benchmark data bundles (traces, task data, prebuilt models) from the Hub.

    With no arguments, lists the org's published datasets (live, via the Hub API) and offers a
    picker. Bundles land in `environment-capture-data/<benchmark>/` under the current directory;
    set `ENVCAP_DATA_ROOT` to put them somewhere else. Existing local files are kept unless
    `--force`.

    A bundle arrives ready to use, not just ready to build from: its `models/` are prebuilt world
    models available to the local server and closed-loop evaluation.

    Args:
        benchmarks: Named benchmark bundles, `all`, or no value for an interactive picker.
        force: Whether an existing local bundle may be overwritten.

    Raises:
        typer.BadParameter: The Hub cannot provide the requested bundle or picker data.
    """
    from wmo.cli.ui import select_option
    from wmo.simulation.hub import corpus_path, published_corpora

    selected = list(benchmarks or [])
    if selected == ["all"]:
        selected = _all_downloadable()
    if not selected:
        try:
            published = published_corpora()
        except urllib.error.URLError as exc:
            raise typer.BadParameter(
                f"could not list the Hub's published datasets ({exc.reason}); check the "
                "connection, or pass benchmark names directly, e.g. `wmo download bird-sql`"
            ) from exc
        if not published:
            raise typer.BadParameter(
                "no published corpora found on the Hub; "
                "pass benchmark names directly, e.g. `wmo download bird-sql`"
            )
        notes = {}
        for corpus in published:
            local = (corpus_path(corpus.benchmark)).exists()
            state = "local copy present" if local else "not downloaded"
            when = f", updated {corpus.last_modified}" if corpus.last_modified else ""
            notes[corpus.benchmark] = f"{state}{when}"
        choices = [corpus.benchmark for corpus in published]
        picked = select_option(
            _console, "Download which data bundle?", [*choices, "all"], notes=notes
        )
        selected = choices if picked == "all" else [picked]
    failures: list[str] = []
    for name in selected:
        existing = corpus_path(name).exists()
        try:
            path = _fetch_with_progress(name, force=force)
        except urllib.error.HTTPError as exc:
            # One unpublished/broken dataset must not abort the REST of a multi-download:
            # record it, keep fetching, and fail (with every name) at the end. The reason is
            # quoted rather than summarized here: a fetch tries more than one dataset repo id
            # and only the error it raises knows which ones the Hub refused.
            reason = f"the Hub answered {exc.code} for {exc.url} ({exc.reason})"
            note = f"Hub answered {exc.code}"
        except urllib.error.URLError as exc:
            # The connection itself is down, which is NOT a verdict on one bundle: everything
            # queued behind it would fail identically, so stop instead of printing the same
            # reason once per benchmark. (Checked before the OSError branch below, which it
            # would otherwise be swallowed by - URLError is an OSError.)
            raise typer.BadParameter(
                f"{name}: could not reach the Hub ({exc.reason}); check the connection and re-run"
                " - fetches resume file-by-file"
            ) from exc
        except ValueError as exc:
            # An unknown name, decided offline before the network is touched. Asked for on its
            # own it stays a plain usage error, because wrapping `wmo download nope` in "some
            # datasets could not be downloaded" buries the answer to the common typo.
            if len(selected) == 1:
                raise typer.BadParameter(str(exc)) from exc
            reason = note = str(exc)
        except OSError as exc:
            # A transfer still truncated after `fetch_corpus`'s own per-file retries. It says
            # nothing about the bundles queued behind it, so in a list it joins the end-of-run
            # report rather than stranding them; alone it is a runtime failure, not a usage
            # error, so it exits 1 with the reason instead of `Invalid value:`.
            if len(selected) == 1:
                _console.print(f"[red]✗ could not download {name}[/red]: {escape(str(exc))}")
                raise typer.Exit(1) from exc
            reason = note = str(exc)
        else:
            state = "kept local" if existing and not force else "fetched"
            _console.print(f"{_CHECK} {state} [bold]{name}[/bold] -> {path}")
            continue
        failures.append(f"{name}: {reason}")
        _console.print(f"[yellow]skipping {name}: {escape(note)}[/yellow]")
    if failures:
        # No cause is asserted here: the list now collects unknown names, Hub refusals and
        # broken transfers alike, and each line carries the reason it actually failed for.
        raise typer.BadParameter(
            "some datasets could not be downloaded (`wmo download` with no arguments lists "
            "what the Hub publishes):\n  " + "\n  ".join(failures)
        )


def _all_downloadable() -> list[str]:
    """The bundles `wmo download all` should fetch, live Hub list preferred.

    The Hub's own listing is authoritative, so it is tried first. Offline the local registry
    answers instead - but only the entries it marks as published. The whole registry is the
    wrong answer: it names bundles registered here so the write side knows how to publish them,
    which the Hub can only answer 401 for, and one of those turns an otherwise complete
    `wmo download all` into a failed command over something the user cannot act on.

    Both narrowings are announced. A quiet substitution of a stale local list for the live one,
    or a quiet drop of a registered benchmark, reads afterwards as "everything was fetched".
    """
    from wmo.simulation.hub import CORPORA, downloadable_benchmarks, published_corpora

    try:
        return sorted(corpus.benchmark for corpus in published_corpora())
    except urllib.error.URLError as exc:
        selected = downloadable_benchmarks()
        _console.print(
            f"[yellow]could not list the Hub's published datasets ({exc.reason}); falling back "
            "to the bundles this release knows about[/yellow]"
        )
        skipped = sorted(set(CORPORA) - set(selected))
        if skipped:
            _console.print(
                f"[yellow]not downloading {', '.join(skipped)}: registered here but never "
                "pushed to the Hub, so there is nothing to fetch[/yellow]"
            )
        return selected


def _fetch_with_progress(name: str, *, force: bool) -> Path:
    """fetch_corpus with a live per-file progress bar (hidden when nothing needs downloading).

    The bar counts FILES because that is what the wait is made of: a bundle is one request per
    file, so a byte-weighted bar sat at 97% for 98% of the download. The bundle's size stays on
    screen as description text, which is where a constant belongs.
    """
    from wmo.simulation.hub import fetch_corpus

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed:.0f}/{task.total:.0f} files"),
        console=_console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(f"downloading {name}", total=None, visible=False)

        def on_progress(done: float, total: float, byte_total: int) -> None:
            progress.update(
                task_id,
                completed=done,
                total=total or None,
                description=f"downloading {name} ({decimal(byte_total)})",
                visible=True,
            )

        return fetch_corpus(name, force=force, on_progress=on_progress)


def _prepare_out_path(out: str | None) -> None:
    """Validate `--out` and create its parent directory BEFORE any (paid) eval work runs.

    Reports are written last, so a `--out` under a missing directory used to surface as a raw
    FileNotFoundError that discarded a finished run. Creating the parent here makes every eval
    flow behave the same and fail before it spends anything.
    """
    if out is None:
        return
    path = Path(out)
    if path.is_dir():
        raise typer.BadParameter(
            f"--out {path} is a directory; pass the file to write (e.g. `--out {path}/report.json`)"
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise typer.BadParameter(f"cannot create --out directory {path.parent}: {err}") from None


def _model_candidates(root: str) -> list[tuple[str, Path, str]]:
    """Every reachable artifact as `(label, store_root, name)`, local builds first.

    The label disambiguates same-named artifacts (`tau-bench (local)` vs `tau-bench (tau-bench
    example)`), so every message that enumerates choices must print labels, not bare names.
    """
    candidates: list[tuple[str, Path, str]] = []
    candidates.extend((f"{n} (local)", Path(root), n) for n in WorldModelStore(root).list_names())
    for example_dir in _discover_examples():
        example_store = WorldModelStore(example_dir)
        candidates.extend(
            (f"{n} ({example_dir.name} example)", example_dir, n)
            for n in example_store.list_names()
        )
    return candidates


def _is_default_project_dir(root: str) -> bool:
    """Whether `--root` still points at the default project dir, however it was spelled.

    Comparing the raw string against `.wmo` made `--root ./.wmo` and `--root .wmo/` (what shell
    tab-completion types) silently mean something different from `--root .wmo`, so resolve both.
    """
    return Path(root).resolve() == Path(ARTIFACT_DIR).resolve()


def _resolve_model_any(name: str | None, root: str) -> tuple[Path, str]:
    """Which artifact a read command should open, as `(store_root, resolved_name)`.

    A `--root` pointing somewhere other than the default project dir keeps single-root behavior.
    Otherwise the search spans `<root>/models/*` plus the downloaded `<data root>/*/models/*`.
    """
    if not _is_default_project_dir(root):
        return Path(root), _resolve_name(WorldModelStore(root), name)

    candidates = _model_candidates(root)
    if name is not None:
        matched = [c for c in candidates if c[2] == name]
        if not matched:
            have = ", ".join(c[0] for c in candidates) or "none built"
            raise typer.BadParameter(f"no world model named {name!r} (have: {have})")
        # Prefer the local build over a same-named example artifact.
        _label, store_root, resolved = matched[0]
    elif not candidates:
        raise typer.BadParameter(
            "no world models found; build one with `wmo build --file <traces> --name <name>`, "
            "or fetch a published benchmark corpus first with `wmo download tau-bench`"
        )
    elif len(candidates) == 1:
        _label, store_root, resolved = candidates[0]
    elif _console.is_terminal:
        labels = [c[0] for c in candidates]
        chosen = _select_from(labels)
        _label, store_root, resolved = candidates[labels.index(chosen)]
    else:
        have = ", ".join(c[0] for c in candidates)
        raise typer.BadParameter(
            f"multiple world models ({have}); pass --name{_shadow_hint(candidates)}"
        )
    return store_root, resolved


def _shadow_hint(candidates: list[tuple[str, Path, str]]) -> str:
    """Name the flag that reaches a shipped example a same-named local build shadows.

    `--name` cannot separate the two (the local build always wins), so a listing that shows the
    name twice has to say which flag can: `--root <the example dir>`.
    """
    names = [c[2] for c in candidates]
    shadowed = [c for c in candidates if names.count(c[2]) > 1 and not c[0].endswith("(local)")]
    if not shadowed:
        return ""
    label, store_root, shadowed_name = shadowed[0]
    return (
        f" (--name {shadowed_name} takes the local build; for '{label}' pass --root {store_root})"
    )


def _select_from(labels: list[str]) -> str:
    """Interactive picker over pre-rendered labels (arrow keys on a TTY)."""
    from wmo.cli import ui as _ui  # package-internal: reuse the wizard's picker machinery

    return _ui._select(
        _console,
        lambda text: _console.input(text),
        "Select a world model",
        labels,
        None,
        interactive=True,
    )


def _resolve_name(store: WorldModelStore, name: str | None) -> str:
    """Resolve which model to run: explicit `--name`, an interactive picker, or the sole model.

    With `--name`, validate it exists. Otherwise, when several models are built on an interactive
    terminal, show a numbered picker; on a non-TTY (or a single model) defer to `store.resolve`,
    which returns the lone model or raises a helpful "pass --name" error. Store errors
    (unknown/ambiguous name) are turned into a clean `typer.BadParameter` rather than a traceback.
    """
    from wmo.cli.ui import select_model

    try:
        if name is not None:
            store.resolve(name)  # validates existence, raising a friendly error if missing
            return name
        # Only enumerate full model summaries when we actually need the picker (>1 model on a TTY).
        # `list_names` is cheap (a dir scan); `list_info` reads every config/metrics/frontier file.
        if _console.is_terminal and len(store.list_names()) > 1:
            # An artifact `list_info` reports as unreadable cannot be run, so keep it off the
            # menu; `wmo list` is where its reason is printed.
            readable = [info for info in store.list_info() if info.error is None]
            if not readable:
                raise ValueError(
                    f"no readable world model under {store.models_dir}; "
                    "run `wmo list` to see what is wrong with each one"
                )
            return select_model(_console, readable)
        return store.resolve(None).name
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _benchmark_roots() -> tuple[Path, ...]:
    """Every root holding self-contained task dirs.

    Benchmark data is not vendored in this repo: `wmo download` writes bundles through
    `wmo.simulation.hub`, which owns where they land (`$ENVCAP_DATA_ROOT` if set, else
    `environment-capture-data/` under the working directory). Deriving the root from
    `corpus_path` instead of hardcoding one keeps discovery pointed wherever download wrote,
    including when the override moves it.
    """
    # Imported here per this module's deferred-import rule (#373): the CLI's startup
    # latency budget forbids eager imports, and this function runs only on discovery.
    from wmo.simulation.hub import CORPORA, corpus_path

    # corpus_path is `<root>/<benchmark>/traces.otel.jsonl`, so its grandparent is the root.
    return (corpus_path(next(iter(CORPORA))).parent.parent,)


def _discover_examples() -> list[Path]:
    found: list[Path] = []
    for root in _benchmark_roots():
        if not root.exists():
            continue
        found.extend(
            path
            for path in root.iterdir()
            if path.is_dir()
            and _is_safe_example_name(path.name)
            and ((path / "traces.otel.jsonl").exists() or (path / "run.sh").exists())
        )
    return sorted(found)


def _is_safe_example_name(name: str) -> bool:
    """Whether `name` is resolvable at all - discovery must not surface what lookup would reject.

    A downloaded dir whose name `validate_name` rejects can never be named on a command line, so
    listing it as a model candidate or in an "available:" hint would only offer a dead end.
    """
    try:
        validate_name(name)
    except ValueError:
        return False
    return True


def _short_error(exc: Exception) -> str:
    """The error's code + service message, without transport chatter.

    botocore's text ("... (reached max retries: 1) ...") reads as OUR retry state and confuses
    the narration; the structured code + message is what the user needs.
    """
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error", {})
        code, message = error.get("Code"), error.get("Message") or ""
        if code:
            return f"{code}: {message}".rstrip(": ")[:110]
    return str(exc).splitlines()[0][:110]


class _RetryNarrator:
    """Console narration for RetryingProvider: hiccup lines + an inline countdown.

    The hiccup line prints only when the failure CHANGES (a stream of identical throttles says
    it once); while a rich status is attached (demo), the wait counts down in place as
    "retry k/3 - waiting Ns…" and then hands the spinner back to the busy text.
    """

    def __init__(self, console: Console) -> None:
        self._console = console
        self._status = None  # rich Status while a spinner context is active
        self.busy = ""
        self._last_error: str | None = None
        self._attempt = 0
        self._total = 0

    def attach(self, status, busy: str) -> None:  # noqa: ANN001 - rich Status
        self._status = status
        self.busy = busy

    def detach(self) -> None:
        self._status = None
        self._last_error = None

    def on_retry(self, attempt: int, total: int, delay: float, exc: Exception) -> None:
        detail = _short_error(exc)
        if detail != self._last_error:
            self._console.print(f"  [yellow]provider hiccup: {escape(detail)}[/yellow]")
            self._last_error = detail
        self._attempt, self._total = attempt, total
        if self._status is None:
            self._console.print(f"  [yellow]retry {attempt}/{total} in {delay:.0f}s…[/yellow]")

    def sleep(self, delay: float) -> None:
        remaining = int(delay)
        while remaining > 0:
            if self._status is not None:
                self._status.update(
                    f"[yellow]retry {self._attempt}/{self._total} - waiting {remaining}s…[/yellow]"
                )
            time.sleep(1)
            remaining -= 1
        if self._status is not None:
            self._status.update(self.busy)


_NARRATOR = _RetryNarrator(_console)


def _load_model(name: str | None, root: str, *, max_fidelity: bool = False):  # noqa: ANN202
    """Resolve + load a named world model (or the single built one) with its serve provider.

    The serve provider comes from the MODEL'S OWN config (the one it was built to serve on),
    wrapped so transient capacity errors retry with narrated exponential backoff instead of
    dying. `max_fidelity` = the online extras (see `WorldModel.load`); default is pure RAG.
    Returns `(world_model, resolved_name, provider)`.
    """
    import wmo.common.providers as providers
    from wmo.common.providers.retry import wrap_provider_with_retries
    from wmo.simulation.model.world_model import WorldModel

    store = WorldModelStore(root)
    resolved_name = _resolve_name(store, name)
    model_dir = store.resolve(resolved_name)
    config = load_config(model_dir)
    serve_config = config.serve_provider_config()
    backend = providers.get_provider(serve_config)
    _prepare_serve_provider_or_exit(backend, serve_config)
    provider = wrap_provider_with_retries(
        backend,
        on_retry=_NARRATOR.on_retry,
        sleep=_NARRATOR.sleep,
    )
    world_model = WorldModel.load(
        str(model_dir), provider, telemetry_root=store.root, max_fidelity=max_fidelity
    )
    return world_model, resolved_name, provider


def _prepare_serve_provider_or_exit(provider: Provider, config: ProviderConfig) -> None:
    """Resolve the serve backend's local prerequisites before the first step, or exit cleanly.

    Every backend builds its SDK client lazily, so a missing SDK or an unset credential otherwise
    surfaces as the SDK's own exception mid-rollout, which Typer renders as a traceback.
    `PreparableProvider.prepare` is the free, offline seam `wmo optimize route sweep` already
    pre-flights with (no backend's `prepare` touches the network), so the interactive commands
    fail here with the same hint `wmo providers verify` prints. Bedrock and tinker document a
    residual gap they cannot close locally; those still fail on the first call.
    """
    from wmo.common.providers.base import PreparableProvider

    if not isinstance(provider, PreparableProvider):
        return
    try:
        provider.prepare()
    except Exception as exc:  # noqa: BLE001 - every backend raises its own SDK's type here
        _console.print(
            f"[red]✗ {config.kind.value} ({config.model}) unusable[/red]: {escape(str(exc))}"
        )
        _console.print(f"  [yellow]{escape(_credential_hint(config.kind, str(exc)))}[/yellow]")
        _console.print("  [yellow]then re-check with `wmo providers verify`[/yellow]")
        raise typer.Exit(1) from exc
