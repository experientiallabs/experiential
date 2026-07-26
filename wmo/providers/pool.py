"""Candidate model pool: the editable roster the routing optimizer selects over.

The pool is one operator-owned TOML file (default `.wmo/pool.toml`, like `.wmo/fallback.toml`),
one `[[model]]` table per candidate. Swapping the roster is editing that file; nothing else in
the harness hardcodes candidate models.

Trust note: the pool file is trusted local config, unlike a model bundle's `config.toml`. That is
why entries may name `api_key_env` (which environment variable holds that account's API key) and
`pool_provider` resolves it and hands the key to `get_provider` as an explicit argument: the
explicit-argument channel is unreachable from bundle-controlled config, so a bundle can never
choose which env var gets read or where its value gets sent.

Pricing: entries for models in the built-in `wmo.tracking.pricing` table need no price fields;
anything else must declare `input_per_mtok`/`output_per_mtok` so downstream cost numbers stay
honest (an unpriced candidate would silently report $0). `cached_input_per_mtok` is the provider
cache-READ price and `cache_write_per_mtok` the cache-WRITE price, carried for cache-aware
routing and compression costs.

`kind = "openrouter"` entries are the one exception, because OpenRouter publishes prices for
every model it fronts: an entry that declares none resolves them from that catalog at load
(`wmo.providers.openrouter_pricing`) and keeps them, so a pool entry needs only a model id and
downstream artifacts still record exact numbers. Offline with nothing cached, the entry falls
back to the same "declare the prices" error, with the catalog failure named.
"""

from __future__ import annotations

import fcntl
import os
import time
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import tomli_w
from llm_waterfall import ChatMaxTokensField
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from wmo.core.types import JsonObject
from wmo.providers.base import (
    PreparableProvider,
    Provider,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
)
from wmo.providers.openrouter_pricing import resolve_price as resolve_openrouter_price
from wmo.providers.registry import get_provider
from wmo.tracking.pricing import ModelPrice, price_for

DEFAULT_POOL_PATH = Path(".wmo/pool.toml")

# How long an upsert waits for another process to finish its write. A write is a few file
# operations, so two racing `wmo optimize route student` runs never come close to this; the bound
# exists so a hung holder is REPORTED instead of hanging the terminal forever.
POOL_LOCK_TIMEOUT_S = 10.0
_LOCK_POLL_S = 0.01

# D-REPORT ModelRef vocabulary: "frontier" anchors the improvement report's comparison; "open"
# models carry the run-10x-more-for-the-same-budget story.
Tier = Literal["frontier", "open"]


class PoolEntry(BaseModel):
    """One candidate model. `name` is the stable handle policy artifacts and request logs key on.

    `extra="forbid"`: a typo like `api_key_evn` must fail at load, not surface as a 401 at
    request time with no hint (same policy as `.wmo/fallback.toml`'s rungs).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: ProviderKind
    model: str = Field(min_length=1)  # provider runtime id (on Azure: the base model id)
    # Canonical model identity for capability resolution, when `model` is a runtime id that does
    # not carry it: a distilled student's `model` is a `tinker://` weights path, and only its base
    # model determines which request shape and sampling params the backend accepts.
    model_type: str | None = None
    # The output-budget parameter this backend accepts. Built-in models resolve it from the
    # catalog; a self-hosted or beta OpenAI-compatible server that wants the classic `max_tokens`
    # (Tinker's serving endpoint does) declares it here, because the catalog has never heard of it.
    chat_max_tokens_field: ChatMaxTokensField = "max_completion_tokens"
    endpoint: str | None = None
    deployment: str | None = None  # Azure deployment name
    api_version: str | None = None  # Azure api-version
    region: str | None = None  # AWS Bedrock region (bedrock entries only)
    api_key_env: str | None = None  # env var holding this entry's API key (multi-account pools)
    tier: Tier = "frontier"
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    cached_input_per_mtok: float | None = None  # provider cache-read price, USD per 1M tokens
    cache_write_per_mtok: float | None = None  # provider cache-write price, USD per 1M tokens

    @model_validator(mode="after")
    def _validate_price(self) -> PoolEntry:
        if (self.input_per_mtok is None) != (self.output_per_mtok is None):
            raise ValueError(
                f"pool model '{self.name}': set both input_per_mtok and output_per_mtok, or neither"
            )
        catalog_note = ""
        if self.input_per_mtok is None and self.kind is ProviderKind.OPENROUTER:
            catalog_note = self._resolve_openrouter_price()
        if self.input_per_mtok is None and price_for(self.model) is None:
            raise ValueError(
                f"pool model '{self.name}': '{self.model}' has no built-in price;{catalog_note} "
                "add input_per_mtok and output_per_mtok (USD per 1M tokens) to its pool entry"
            )
        if self.kind is ProviderKind.AZURE_OPENAI and self.deployment is None:
            # Without this the entry loads fine and the first request routed to it 500s
            # from AzureOpenAIProvider._deployment(); load is the validation boundary.
            raise ValueError(
                f"pool model '{self.name}': azure entries need `deployment` (the Azure "
                "deployment name to call)"
            )
        if self.kind is ProviderKind.BEDROCK and self.api_key_env is not None:
            # Same boundary: `BedrockProvider.__init__` refuses an explicit key, and a sweep
            # constructs providers lazily per cell, so this would abort mid-run after the
            # candidates ahead of it had already been paid for.
            raise ValueError(
                f"pool model '{self.name}': bedrock authenticates with AWS credentials "
                "(profile/role), not an API key; drop api_key_env from this entry"
            )
        return self

    def _resolve_openrouter_price(self) -> str:
        """Stamp this entry with OpenRouter's published price, returning "" or why it could not.

        Reached only from `_validate_price`, and only for an OpenRouter entry that declared no
        price: no other provider's config validation touches the catalog, and nothing fetches
        at import time.

        Stamping (rather than looking the price up per call) is what makes a fitted policy
        stable. `RoutingPolicy.pool` and `OutcomeMatrix.pool` serialize these entries verbatim,
        so the numbers a policy was fitted under travel with it and re-validating that artifact
        finds the prices already set, which short-circuits this method entirely. A later vendor
        price change cannot silently re-price a policy already in production; refitting or a
        fresh `load_pool` is what picks it up.

        Returns:
            An empty string on success, else a clause naming the catalog failure, ready to be
            spliced into the caller's "declare the prices explicitly" error.
        """
        resolution = resolve_openrouter_price(self.model)
        if resolution.price is None:
            return f" {resolution.detail};"
        self.input_per_mtok = resolution.price.input_per_mtok
        self.output_per_mtok = resolution.price.output_per_mtok
        # Cache tiers are published per model too, but an explicit entry value stays the
        # operator's (a negotiated rate must not be overwritten by the public list price).
        if self.cached_input_per_mtok is None:
            self.cached_input_per_mtok = resolution.price.cache_read_per_mtok
        if self.cache_write_per_mtok is None:
            self.cache_write_per_mtok = resolution.price.cache_write_per_mtok
        return ""

    def price(self) -> ModelPrice:
        """This entry's price row: the explicit override, else the built-in pricing table."""
        if self.input_per_mtok is not None and self.output_per_mtok is not None:
            return ModelPrice(
                input_per_mtok=self.input_per_mtok, output_per_mtok=self.output_per_mtok
            )
        price = price_for(self.model)
        if price is None:  # unreachable after validation; keep the failure loud, not $0
            raise ValueError(f"pool model '{self.name}': no price available for '{self.model}'")
        return price

    def cost_usd(self, usage: TokenUsage) -> float:
        """Effective USD cost of `usage` priced by THIS entry's row (overrides included).

        Cache-adjusted on both tiers: cache-read tokens (`usage.cached_input_tokens`) bill at
        `cached_input_per_mtok` and cache-write tokens (`usage.cache_write_input_tokens`) at
        `cache_write_per_mtok`; each tier falls back to the built-in price row's rate when the
        entry carries no override, and to the full input rate when the row has no tier either
        (never silently free). The global `wmo.tracking.pricing.cost_usd` only knows the
        built-in table; pool entries with explicit prices must be costed here or they would
        silently read $0.
        """
        price = self.price()
        read = min(usage.cached_input_tokens, usage.input_tokens)
        write = min(usage.cache_write_input_tokens, usage.input_tokens - read)
        read_rate = self.cached_input_per_mtok
        if read_rate is None:
            read_rate = (
                price.cache_read_per_mtok
                if price.cache_read_per_mtok is not None
                else price.input_per_mtok
            )
        write_rate = self.cache_write_per_mtok
        if write_rate is None:
            write_rate = (
                price.cache_write_per_mtok
                if price.cache_write_per_mtok is not None
                else price.input_per_mtok
            )
        return (
            (usage.input_tokens - read - write) * price.input_per_mtok
            + read * read_rate
            + write * write_rate
            + usage.output_tokens * price.output_per_mtok
        ) / 1_000_000

    def provider_config(self) -> ProviderConfig:
        return ProviderConfig(
            kind=self.kind,
            model=self.model,
            model_type=self.model_type,
            chat_max_tokens_field=self.chat_max_tokens_field,
            endpoint=self.endpoint,
            deployment=self.deployment,
            api_version=self.api_version,
            region=self.region,
        )


class ModelPool(BaseModel):
    """The full candidate roster, as loaded from one pool TOML."""

    models: list[PoolEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> ModelPool:
        seen: set[str] = set()
        for entry in self.models:
            if entry.name in seen:
                raise ValueError(f"pool model '{entry.name}' is declared twice; names are handles")
            seen.add(entry.name)
        return self

    def entry(self, name: str) -> PoolEntry:
        for candidate in self.models:
            if candidate.name == name:
                return candidate
        available = ", ".join(m.name for m in self.models)
        raise KeyError(f"no pool model named '{name}'; available: {available}")


def load_pool(path: Path = DEFAULT_POOL_PATH) -> ModelPool:
    """Load and validate the pool file at `path`."""
    if not path.is_file():
        raise FileNotFoundError(
            f"no model pool file at {path}; create it with one [[model]] table per candidate "
            "(fields: name, kind, model, and for non-built-in models input_per_mtok/"
            "output_per_mtok; endpoint/deployment/api_version/api_key_env as the backend needs)"
        )
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return ModelPool.model_validate({"models": data.get("model", [])})


def pool_api_key(entry: PoolEntry) -> str | None:
    """One entry's own API key, read from its `api_key_env`; None means backend defaults.

    Separate from `pool_provider` so a caller about to spend money on a WHOLE pool can resolve
    every candidate's credentials up front (`wmo optimize route sweep` does, before its cost
    confirmation) instead of discovering an unset variable at the first cell of a candidate it
    already paid to reach. The lookup is local `os.environ`, so it costs nothing to run early.

    Raises:
        ValueError: `api_key_env` names a variable that is unset or empty.
    """
    if not entry.api_key_env:
        return None
    api_key = os.environ.get(entry.api_key_env)
    if not api_key:
        raise ValueError(
            f"pool model '{entry.name}': environment variable {entry.api_key_env} is unset "
            "or empty; export that account's API key or drop api_key_env to use the "
            "backend's default credentials"
        )
    return api_key


class PoolLockTimeout(RuntimeError):
    """Another writer held the roster's lock for longer than the bounded wait."""


def upsert_pool_entry(
    entry: PoolEntry,
    path: Path = DEFAULT_POOL_PATH,
    *,
    lock_timeout_s: float | None = None,
) -> bool:
    """Add `entry` to the pool roster at `path`, replacing any entry with the same name.

    The one write path for the roster, so a trained model becomes routable without hand-editing
    TOML. Entries already in the file are carried through as the exact tables they were written
    as: re-serializing them through `PoolEntry` would stamp every default (`tier`,
    `chat_max_tokens_field`) into a file an operator maintains by hand.

    Adding a NEW entry appends its one table and touches nothing else, so comments, key order,
    and spacing all survive byte for byte. REPLACING an entry has to remove the old table, which
    means re-rendering the file through `tomllib` -> `tomli_w`: that keeps every entry's fields
    but DROPS comments, because neither library round-trips them. Callers that can prompt should
    say so before replacing (`wmo optimize route student` does).

    The merged roster is validated as a whole `ModelPool` BEFORE anything is written, and the
    write itself goes through a temp file, so a rejected entry can never leave the pool
    unloadable. That matters more than it looks: `load_pool` is what serving and the routing
    optimizer both call, so a broken pool file takes every endpoint down, not just the candidate
    being added.

    The whole read-validate-write cycle runs under an exclusive cross-process lock (see
    `_pool_write_lock`), so two registrations racing each other both land. Without it each reads
    the same roster and the later write erases the earlier entry, while both commands report
    success: a model an operator registered is simply not in the pool, and nothing says so.

    Args:
        entry: The candidate to add, or to replace an existing entry of the same name with.
        path: The pool TOML. Created, with its parent directory, when absent.
        lock_timeout_s: Seconds to wait for another writer's lock before giving up; the default
            is `POOL_LOCK_TIMEOUT_S`.

    Returns:
        True when an entry of the same name was replaced, False when `entry` was appended.

    Raises:
        ValueError: If `path` exists but is not a readable pool file, if it is already an invalid
            roster before this entry is added, or if adding `entry` would make it one.
        PoolLockTimeout: If another writer holds the roster's lock for the whole wait.
    """
    timeout_s = POOL_LOCK_TIMEOUT_S if lock_timeout_s is None else lock_timeout_s
    path.parent.mkdir(parents=True, exist_ok=True)  # the lock file lives beside the roster
    with _pool_write_lock(path, timeout_s=timeout_s):
        return _upsert_locked(entry, path)


@contextmanager
def _pool_write_lock(path: Path, *, timeout_s: float) -> Iterator[None]:
    """Hold the exclusive cross-process write lock for the roster at `path`.

    The lock sits in a sibling `<pool>.lock` file, not in the roster: writing goes through an
    atomic rename, which swaps the roster's inode, so a lock taken on the file itself would stop
    protecting anything the moment the first writer landed.

    `flock` belongs to the open file description, so the kernel drops it when the holder exits,
    crashes, or is killed. A leftover `.lock` FILE is therefore never a held lock and can never
    wedge a later run (which is also why it is left in place: unlinking it would let a waiter
    block on an inode nobody else can reach). A live holder that hangs still could wedge us, so
    the wait is bounded and reports what to do instead of blocking forever.

    Args:
        path: The pool TOML being written.
        timeout_s: Seconds to keep retrying the lock before raising.

    Yields:
        None, with the lock held; it is released when the block exits, on any path.

    Raises:
        PoolLockTimeout: If the lock is still held after `timeout_s`.
    """
    lock_path = path.with_name(f"{path.name}.lock")
    deadline = time.monotonic() + timeout_s
    fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_CLOEXEC, 0o600)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise PoolLockTimeout(
                        f"another process has been writing the model pool at {path} for more "
                        f"than {timeout_s:g}s (lock file {lock_path}); a registration takes "
                        "milliseconds, so retry, and if it keeps failing look for a stuck "
                        "process holding that lock (the lock is released automatically when "
                        "that process exits, so the lock file itself is never the problem)"
                    ) from None
                time.sleep(_LOCK_POLL_S)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _upsert_locked(entry: PoolEntry, path: Path) -> bool:
    """The read-validate-write cycle of `upsert_pool_entry`, with the roster's lock held."""
    tables = _raw_tables(path)
    if tables:
        # Validate what is ALREADY there first, so a pre-existing bad row is reported as a
        # pre-existing bad row. Otherwise a typo an operator left in some other entry surfaces as
        # "adding 'student' would make this an invalid pool", pointing at the flags they just got
        # right instead of at the line that is actually wrong.
        try:
            ModelPool.model_validate({"models": tables})
        except ValidationError as exc:
            raise ValueError(
                f"pool file {path} is already invalid, before adding '{entry.name}': {exc}"
            ) from exc
    kept = [table for table in tables if table.get("name") != entry.name]
    replaced = len(kept) != len(tables)
    # exclude_defaults keeps the written table as small as the hand-written ones around it, and
    # exclude_none drops the backend knobs this kind does not use. mode="json" turns the
    # ProviderKind enum into the plain string TOML can hold.
    table = entry.model_dump(mode="json", exclude_defaults=True, exclude_none=True)
    merged = [*kept, table]
    try:
        ModelPool.model_validate({"models": merged})
    except ValidationError as exc:
        raise ValueError(f"adding '{entry.name}' would make {path} an invalid pool: {exc}") from exc
    rendered = tomli_w.dumps({"model": [table]})
    if replaced or not path.is_file():
        # A replacement has to remove the old table, which means re-rendering the whole file.
        body = tomli_w.dumps({"model": merged})
    else:
        # The common case is a first registration, and appending the one new table byte-preserves
        # everything already in the file: an operator's comments, key order, and spacing. Rewriting
        # would silently destroy the comments that say which account a row bills to, and nothing
        # would tell them it happened.
        existing = path.read_text(encoding="utf-8")
        if existing.endswith("\n\n"):
            separator = ""
        else:
            separator = "\n" if existing.endswith("\n") else "\n\n"
        body = f"{existing}{separator}{rendered}"
    # The temp name carries this process's pid. The lock already keeps writers off each other, so
    # this is defense in depth for anything that writes the roster without taking it: a shared
    # fixed `.tmp` path would let each writer rename the other's half-written file into place,
    # turning a lost update into a CORRUPT roster.
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(body, encoding="utf-8")
        tmp_path.replace(path)
    except OSError:
        tmp_path.unlink(missing_ok=True)  # never leave a stray temp beside the roster
        raise
    return replaced


def _raw_tables(path: Path) -> list[JsonObject]:
    """The `[[model]]` tables in `path`, exactly as written; empty when the file does not exist."""
    if not path.is_file():
        return []
    try:
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(
            f"pool file {path} is not valid TOML ({exc}); fix the file, or move it aside to "
            "start a fresh roster"
        ) from exc
    tables = parsed.get("model", [])
    if not isinstance(tables, list) or not all(isinstance(table, dict) for table in tables):
        raise ValueError(
            f"pool file {path} does not hold an array of [[model]] tables; every candidate is its "
            "own [[model]] table (fields: name, kind, model, and prices for non-built-in models)"
        )
    return [dict(table) for table in tables]


def pool_provider(entry: PoolEntry) -> Provider:
    """Construct the provider for one pool entry, resolving its per-account API key.

    Construction is side-effect free for every backend (`wmo.providers.registry`): each one only
    stores its config and defers the SDK import, the credential read, and the client to its first
    request. Which is also why construction alone is a weak pre-flight: use
    `prepare_pool_provider` when the point is to learn whether a candidate can be CALLED.

    A construction failure is re-raised naming the entry and its kind: a backend that refuses to
    be built ("Bedrock authenticates with AWS credentials, not an API key") knows nothing about
    the pool it came from, and the caller is usually looping over candidates.

    Raises:
        ValueError: The entry names an unset `api_key_env`, or its backend refuses this config.
    """
    api_key = pool_api_key(entry)
    try:
        return get_provider(entry.provider_config(), api_key=api_key)
    except ValueError as exc:
        raise ValueError(f"pool model '{entry.name}' (kind={entry.kind.value}): {exc}") from exc


def static_requirements(entry: PoolEntry) -> list[str]:
    """What one entry's KIND needs from the entry alone, worded for the file it is edited in.

    Read before any SDK is imported and before any client is built, so an entry that could never
    be called fails on its own config. Each item is derived from the backend's request path, not
    guessed:

    - azure needs `deployment` (on the wire, Azure's `model` IS the deployment name, so
      `AzureOpenAIProvider._deployment` refuses without it) and `api_version` (its client cannot be
      built without one). `endpoint` is NOT required here: it legitimately comes from
      AZURE_OPENAI_ENDPOINT, which the provider resolves.
    - tinker needs a base model name, because `TinkerChatProvider` resolves its renderer and
      tokenizer from `ProviderConfig.model_type` and a pool entry has no field that fills it, so a
      `tinker://` weights path in `model` can never render a prompt.
    - bedrock, openai, openai_responses, anthropic and openrouter need nothing from the entry: a
      region (for bedrock) or a credential (for the rest) is what they need, and those resolve
      from the local environment, not from this file. openrouter also needs no price, because an
      entry that declares none resolves one from its published catalog at load.

    Kept out of `PoolEntry`'s own validation on purpose, unlike the two rules that are there: a
    saved `OutcomeMatrix` carries its pool inline, so tightening load-time validation would
    retroactively refuse matrices whose entries only ever supplied prices. Being callable matters
    where a candidate is about to be called.

    Returns:
        One human-readable complaint per unmet requirement; empty when the entry is complete.
    """
    match entry.kind:
        case ProviderKind.AZURE_OPENAI:
            missing: list[str] = []
            if entry.deployment is None:
                missing.append(
                    "`deployment` (the Azure deployment name every request names as its model)"
                )
            if entry.api_version is None:
                missing.append(
                    "`api_version` (AzureOpenAIProvider cannot build its client without one)"
                )
            return [f"azure entries need {item}" for item in missing]
        case ProviderKind.TINKER:
            if entry.model.startswith("tinker://"):
                return [
                    "a tinker entry's `model` cannot be a tinker:// weights path: the renderer and "
                    "tokenizer resolve from the BASE model name, which a pool entry has no field "
                    "to carry, so name the base model (e.g. 'Qwen/Qwen3-8B') instead"
                ]
            return []
        case (
            ProviderKind.BEDROCK
            | ProviderKind.OPENAI
            | ProviderKind.OPENAI_RESPONSES
            | ProviderKind.ANTHROPIC
            | ProviderKind.OPENROUTER
        ):
            return []


def prepare_pool_provider(entry: PoolEntry) -> Provider:
    """Construct one entry's provider AND resolve every prerequisite that needs no request.

    The pre-flight seam for a caller about to spend money on a whole roster (`wmo optimize route
    sweep`). `pool_provider` alone is too weak for that: every backend builds its SDK client
    lazily, so an uninstalled SDK extra, an unset credential, or a region that resolves nowhere
    still lands at that candidate's FIRST CALL, after the candidates ahead of it have been paid
    for. This runs the kind's `static_requirements` and then `PreparableProvider.prepare`, which
    forces the lazy client to be built where that is free.

    Free, and provably so: no backend's `prepare` issues a request (see each one's docstring).
    Verifying a candidate over the wire is deliberately NOT done here, because `wmo providers
    verify` bills a real call per model, which a pre-flight that runs before spend is authorized
    may not do. Two backends therefore keep a documented residual gap, and callers should say so:
    bedrock cannot resolve AWS CREDENTIALS locally (building the client walks a chain that reaches
    the instance-metadata endpoint over the network, and succeeds with no credentials anyway), and
    tinker cannot resolve SERVICE REACHABILITY (constructing the client connects and pins a
    server-side session). Both stay first-call failures.

    Returns:
        The prepared provider. Callers that only wanted the check may discard it; the sweep does,
        because `evaluate_pool` builds its own per cell to keep per-episode provider state fresh.

    Raises:
        ValueError: This entry cannot be used, named with its kind and what to do about it.
    """
    problems = static_requirements(entry)
    if problems:
        raise ValueError(
            f"pool model '{entry.name}' (kind={entry.kind.value}): {'; '.join(problems)}"
        )
    provider = pool_provider(entry)
    if isinstance(provider, PreparableProvider):
        try:
            provider.prepare()
        except Exception as exc:  # noqa: BLE001 - every backend raises its own SDK's type here
            # Re-raised as one usage error naming the entry: the SDK's own message says what is
            # missing, and the caller is looping over candidates in a file it wants to edit.
            raise ValueError(f"pool model '{entry.name}' (kind={entry.kind.value}): {exc}") from exc
    return provider
