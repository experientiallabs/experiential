"""Tests for the interactive model registry that fills `.wmo/pool.toml`.

Every prompt is driven by a scripted input stream and every assertion is made against the roster
the run actually wrote, because the point of this surface is the file, not the transcript. The
OpenRouter catalog is faked through the same disk cache the resolver reads, so nothing here
reaches the network.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from io import StringIO
from pathlib import Path

import pytest
import typer
from pydantic import ValidationError
from rich.console import Console

from wmo.cli import pool_registry
from wmo.cli.pool_registry import (
    EntryOptions,
    build_pool_entry,
    needs_price,
    pool_path_for,
    read_pool_entries,
    register_model_ids,
    run_pool_registry,
)
from wmo.common.config import PROVIDER_ENV_VARS
from wmo.common.observability.pricing import ModelPrice
from wmo.common.providers.base import ProviderKind, VerifyResult
from wmo.common.providers.catalog import (
    CatalogModel,
    CatalogSource,
    ProviderCatalog,
    list_provider_models,
)
from wmo.common.providers.openrouter_pricing import CATALOG_PATH_ENV, PriceCatalog
from wmo.common.providers.pool import PoolEntry, load_pool

_SONNET = "anthropic/claude-sonnet-4.5"
_DEEPSEEK = "deepseek/deepseek-v3.2"


def _fake_prices() -> dict[str, ModelPrice]:
    """A published catalog big enough that the picker has to filter rather than list."""
    prices = {
        _SONNET: ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0, cache_read_per_mtok=0.3),
        _DEEPSEEK: ModelPrice(input_per_mtok=0.27, output_per_mtok=0.4),
        "anthropic/claude-haiku-4.5": ModelPrice(input_per_mtok=1.0, output_per_mtok=5.0),
    }
    for index in range(25):
        prices[f"filler/model-{index:02d}"] = ModelPrice(
            input_per_mtok=float(index), output_per_mtok=float(index * 2)
        )
    return prices


class _Script:
    """A scripted stdin: each read pops the next line, and running out raises EOFError.

    EOFError rather than StopIteration on purpose: that is what `rich`'s `Console.input` raises
    on closed stdin, so a test that under-feeds the prompts exercises the real abort path.
    """

    def __init__(self, lines: list[str]) -> None:
        self.remaining = list(lines)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.remaining:
            raise EOFError(prompt)
        return self.remaining.pop(0)


@pytest.fixture(autouse=True)
def _catalog_and_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Seed the OpenRouter cache, fake every credential, and cut the live ping off the network.

    Credentials matter because picking a backend other than the one just configured runs the
    wizard's credential prompt, which would otherwise consume script lines and write a `.env`
    into the working directory. Replacing `verify_pool_entry` is the same rule `wmo/conftest.py`
    applies to the price catalog: a test that forgets to inject a ping must fail loudly rather
    than quietly calling openrouter.ai. Tests that want a ping to FAIL pass their own.
    """
    cache = tmp_path / "openrouter-prices.json"
    catalog = PriceCatalog(fetched_at=time.time(), source="test fixture", prices=_fake_prices())
    cache.write_text(catalog.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv(CATALOG_PATH_ENV, str(cache))
    for env_vars in PROVIDER_ENV_VARS.values():
        for var in env_vars:
            monkeypatch.setenv(var, "test-value")
    monkeypatch.setattr(pool_registry, "verify_pool_entry", _refuse_network)


def _refuse_network(entry: PoolEntry) -> VerifyResult:
    """Stand-in for the live ping: the suite never calls a provider."""
    raise AssertionError(f"the test suite does not ping providers (tried {entry.model})")


def _ok(entry: PoolEntry) -> VerifyResult:
    """A live ping that succeeds. Every test injects one; nothing here reaches a backend."""
    return VerifyResult(ok=True, kind=entry.kind, model=entry.model)


def _drive(
    pool: Path,
    lines: list[str],
    *,
    kind: ProviderKind = ProviderKind.OPENROUTER,
    options: EntryOptions | None = None,
    verify: Callable[[PoolEntry], VerifyResult] = _ok,
) -> tuple[int, str, _Script]:
    """Run the registry against `pool` with `lines` as the user's answers."""
    console = Console(file=StringIO(), width=120, no_color=True, highlight=False)
    script = _Script(lines)
    written = run_pool_registry(
        console,
        script,
        script,
        pool_path=pool,
        default_kind=kind,
        options=options or EntryOptions(),
        verify=verify,
    )
    assert isinstance(console.file, StringIO)
    return written, console.file.getvalue(), script


def _openrouter_pass(*models: str, tier: str = "", api_key_env: str = "") -> list[str]:
    """The answers that register `models` from OpenRouter in one pass (handles defaulted)."""
    return [*models, "", tier, api_key_env, *["" for _ in models]]


def test_registers_openrouter_models_and_never_asks_them_for_a_price(tmp_path: Path) -> None:
    # OpenRouter self-prices from its published catalog (#284), so a price prompt here would be
    # asking for a number the entry is about to resolve anyway.
    pool = tmp_path / "pool.toml"
    written, output, script = _drive(
        pool,
        ["y", "openrouter", *_openrouter_pass(_SONNET, _DEEPSEEK, tier="open"), "n"],
    )

    assert written == 2
    entries = {entry.name: entry for entry in read_pool_entries(pool)}
    assert set(entries) == {"claude-sonnet-4.5", "deepseek-v3.2"}
    sonnet = entries["claude-sonnet-4.5"]
    assert sonnet.kind is ProviderKind.OPENROUTER
    assert sonnet.model == _SONNET
    assert sonnet.tier == "open"
    # Stamped from the catalog, so the roster records exact numbers rather than deferring.
    assert (sonnet.input_per_mtok, sonnet.output_per_mtok) == (3.0, 15.0)
    assert sonnet.cached_input_per_mtok == 0.3
    assert sonnet.deployment is None and sonnet.api_version is None
    assert not any("price" in prompt.lower() for prompt in script.prompts)
    assert "USD per 1M tokens" not in output


def test_a_second_pass_adds_a_second_provider_without_dropping_the_first(tmp_path: Path) -> None:
    # The whole point of a re-runnable registry: a user accumulates a pool across several
    # providers in several passes, and pass two must not clobber pass one.
    pool = tmp_path / "pool.toml"
    _drive(pool, ["y", "openrouter", *_openrouter_pass(_SONNET, tier="open"), "n"])
    assert [entry.name for entry in read_pool_entries(pool)] == ["claude-sonnet-4.5"]

    written, _, _ = _drive(
        pool,
        [
            "y",
            "azure",
            "gpt-5.4",  # an exact catalog id picks that model directly
            "",
            "frontier",
            "",  # api key env: the backend default
            "prod-gpt54",  # deployment
            "2025-01-01-preview",  # api-version
            "",  # handle
            "n",
        ],
        kind=ProviderKind.OPENROUTER,
    )

    assert written == 1
    entries = {entry.name: entry for entry in read_pool_entries(pool)}
    assert set(entries) == {"claude-sonnet-4.5", "prod-gpt54"}
    assert entries["claude-sonnet-4.5"].kind is ProviderKind.OPENROUTER
    azure = entries["prod-gpt54"]
    assert azure.kind is ProviderKind.AZURE_OPENAI
    assert azure.model == "gpt-5.4"
    assert azure.deployment == "prod-gpt54"
    assert azure.api_version == "2025-01-01-preview"


def test_re_registering_the_same_model_leaves_the_file_byte_identical(tmp_path: Path) -> None:
    # Idempotence has to be at the FILE level: writing the same entry again would go through
    # upsert's replacement path, which re-renders the roster and drops its comments.
    pool = tmp_path / "pool.toml"
    _drive(pool, ["y", "openrouter", *_openrouter_pass(_SONNET, _DEEPSEEK), "n"])
    before = pool.read_text(encoding="utf-8")

    written, output, _ = _drive(
        pool, ["y", "openrouter", *_openrouter_pass(_SONNET, _DEEPSEEK), "n"]
    )

    assert written == 0
    assert pool.read_text(encoding="utf-8") == before
    assert "already registered, unchanged" in output


def test_the_picker_shows_what_is_already_in_the_pool(tmp_path: Path) -> None:
    # A second pass starts from a visible roster, and an already-registered model is annotated
    # with the handle it carries so re-picking it is an informed choice.
    pool = tmp_path / "pool.toml"
    _drive(pool, ["y", "openrouter", *_openrouter_pass(_SONNET), "n"])

    _, output, _ = _drive(pool, ["y", "openrouter", "sonnet", "", "n"])

    assert "routing pool" in output
    assert "in pool as claude-sonnet-4.5" in output


def test_search_narrows_the_catalog_and_numbers_pick_from_the_matches(tmp_path: Path) -> None:
    # 338 models are not a scrollable list. An unfiltered listing is capped and says so, and a
    # search is what makes numbered picking meaningful.
    pool = tmp_path / "pool.toml"
    written, output, _ = _drive(
        pool, ["y", "openrouter", "claude", "1 2", "", "frontier", "", "", "", "n"]
    )

    assert "more; refine the search" in output
    assert written == 2
    assert {entry.model for entry in read_pool_entries(pool)} == {
        _SONNET,
        "anthropic/claude-haiku-4.5",
    }


def test_a_number_outside_the_listing_selects_nothing_at_all(tmp_path: Path) -> None:
    # "1 99" against one match is a typo, not a half-selection: toggling the valid half would
    # register a model the user never chose while looking like it worked.
    pool = tmp_path / "pool.toml"
    _, output, _ = _drive(pool, ["y", "openrouter", "claude-haiku", "1 99", "", "n"])

    assert "pick 1-1 from the rows above" in output
    assert read_pool_entries(pool) == []


def test_a_typed_id_outside_the_catalog_is_registered_after_confirmation(tmp_path: Path) -> None:
    # A model published after this release, or a self-hosted OpenAI-compatible server, must be
    # reachable; a catalog is suggestions, never a whitelist.
    pool = tmp_path / "pool.toml"
    written, _, _ = _drive(
        pool,
        [
            "y",
            "openai",
            "",  # endpoint URL: blank = OpenAI's own API
            "my-vllm/qwen3-32b",
            "y",  # yes, take it as a literal id
            "",
            "open",
            "WMO_ENDPOINT_API_KEY",
            "0.1",  # no built-in price, so both tiers are asked for
            "0.4",
            "",
            "n",
        ],
    )

    assert written == 1
    entry = read_pool_entries(pool)[0]
    assert entry.name == "qwen3-32b"
    assert entry.kind is ProviderKind.OPENAI
    assert entry.model == "my-vllm/qwen3-32b"
    assert entry.api_key_env == "WMO_ENDPOINT_API_KEY"
    assert (entry.input_per_mtok, entry.output_per_mtok) == (0.1, 0.4)


def test_declining_a_literal_id_registers_nothing(tmp_path: Path) -> None:
    pool = tmp_path / "pool.toml"
    written, _, _ = _drive(pool, ["y", "openai", "", "typo-model", "n", "", "n"])

    assert written == 0
    assert not pool.exists()


def test_a_price_is_re_asked_until_it_is_a_non_negative_number(tmp_path: Path) -> None:
    # There is no safe default: a candidate priced at $0 wins every cost-aware routing decision.
    pool = tmp_path / "pool.toml"
    written, output, _ = _drive(
        pool,
        ["y", "openai", "", "self-hosted", "y", "", "open", "", "free", "-2", "0", "1.5", "", "n"],
    )

    assert "non-negative" in output
    assert written == 1
    entry = read_pool_entries(pool)[0]
    assert (entry.input_per_mtok, entry.output_per_mtok) == (0.0, 1.5)


def test_a_built_in_priced_model_is_not_asked_for_a_price(tmp_path: Path) -> None:
    # gpt-5.4 is in `wmo.common.observability.pricing`, so the entry is honest without a declared
    # price and stays as small as a hand-written one.
    pool = tmp_path / "pool.toml"
    written, _, _ = _drive(
        pool,
        ["y", "openai", "", "gpt-5.4", "", "frontier", "", "", "n"],
        kind=ProviderKind.OPENAI,
    )

    assert written == 1
    entry = read_pool_entries(pool)[0]
    assert entry.input_per_mtok is None
    assert entry.price().input_per_mtok == 2.5


def test_bedrock_is_asked_for_a_region_and_never_for_an_api_key_env(tmp_path: Path) -> None:
    # Bedrock authenticates with AWS credentials; `PoolEntry` rejects an api_key_env outright,
    # so asking for one would collect an answer the roster refuses.
    pool = tmp_path / "pool.toml"
    written, _, script = _drive(
        pool,
        ["y", "bedrock", "claude-haiku-4-5", "", "frontier", "eu-central-1", "", "n"],
        kind=ProviderKind.BEDROCK,
    )

    assert written == 1
    entry = read_pool_entries(pool)[0]
    assert entry.region == "eu-central-1"
    assert entry.api_key_env is None
    assert not any("API key" in prompt for prompt in script.prompts)


def test_an_openrouter_pass_is_never_asked_azure_questions(tmp_path: Path) -> None:
    pool = tmp_path / "pool.toml"
    _, _, script = _drive(pool, ["y", "openrouter", *_openrouter_pass(_SONNET), "n"])

    assert not any("Azure" in prompt for prompt in script.prompts)


def test_the_same_model_under_a_second_account_becomes_a_second_entry(tmp_path: Path) -> None:
    # Multi-account pools deliberately carry one model twice under two credentials; collapsing
    # them onto one handle would silently delete an account's candidate.
    pool = tmp_path / "pool.toml"
    _drive(pool, ["y", "openrouter", *_openrouter_pass(_SONNET, api_key_env="ACCOUNT_A"), "n"])
    written, _, _ = _drive(
        pool, ["y", "openrouter", *_openrouter_pass(_SONNET, api_key_env="ACCOUNT_B"), "n"]
    )

    assert written == 1
    entries = read_pool_entries(pool)
    assert [entry.name for entry in entries] == ["claude-sonnet-4.5", "claude-sonnet-4.5-2"]
    assert [entry.api_key_env for entry in entries] == ["ACCOUNT_A", "ACCOUNT_B"]


def test_a_handle_that_would_replace_another_entry_is_confirmed_first(tmp_path: Path) -> None:
    # Reusing a handle repoints everything keyed on it and rewrites the file; declining loops
    # back to the prompt instead of doing it quietly.
    pool = tmp_path / "pool.toml"
    _drive(pool, ["y", "openrouter", *_openrouter_pass(_SONNET), "n"])

    written, output, _ = _drive(
        pool,
        [
            "y",
            "openrouter",
            _DEEPSEEK,
            "",
            "frontier",
            "",
            "claude-sonnet-4.5",  # a handle that already names a different model
            "n",  # do not replace it
            "deepseek",  # a fresh handle instead
            "n",
        ],
    )

    assert "already names" in output
    assert written == 1
    entries = {entry.name: entry.model for entry in read_pool_entries(pool)}
    assert entries == {"claude-sonnet-4.5": _SONNET, "deepseek": _DEEPSEEK}


def test_declining_the_offer_writes_no_roster_at_all(tmp_path: Path) -> None:
    pool = tmp_path / "pool.toml"
    written, output, script = _drive(pool, ["n"])

    assert written == 0
    assert not pool.exists()
    assert "skipped" in output
    assert script.remaining == []


def test_selecting_nothing_writes_no_roster(tmp_path: Path) -> None:
    pool = tmp_path / "pool.toml"
    written, output, _ = _drive(pool, ["y", "openrouter", "", "n"])

    assert written == 0
    assert not pool.exists()
    assert "nothing selected" in output


def test_exhausted_input_aborts_instead_of_leaking_an_eoferror(tmp_path: Path) -> None:
    # Piped input that runs out, or Ctrl-D, must end the command cleanly rather than printing a
    # traceback over a half-written roster.
    pool = tmp_path / "pool.toml"
    with pytest.raises(typer.Abort):
        _drive(pool, ["y", "openrouter", _SONNET])


def test_an_unreadable_roster_is_left_to_the_writer_to_report(tmp_path: Path) -> None:
    # Display and duplicate detection degrade quietly; `upsert_pool_entry` owns the actionable
    # "fix the file, or move it aside" error and must not be pre-empted with a worse one.
    pool = tmp_path / "pool.toml"
    pool.write_text("this is not toml [[[", encoding="utf-8")

    written, output, _ = _drive(pool, ["y", "openrouter", *_openrouter_pass(_SONNET), "n"])

    assert written == 0
    assert "not valid TOML" in output
    assert pool.read_text(encoding="utf-8") == "this is not toml [[["


def test_flag_supplied_knobs_are_dropped_when_the_first_pass_switches_backend(
    tmp_path: Path,
) -> None:
    # The switch can happen on the FIRST pass too: the user configures an Azure worker and then
    # registers OpenRouter candidates. The Azure knobs must not follow them there.
    pool = tmp_path / "pool.toml"
    written, _, _ = _drive(
        pool,
        ["y", "openrouter", *_openrouter_pass(_SONNET), "n"],
        kind=ProviderKind.AZURE_OPENAI,
        options=EntryOptions(deployment="prod-gpt55", api_version="2030-01-01"),
    )

    assert written == 1
    entry = read_pool_entries(pool)[0]
    assert entry.kind is ProviderKind.OPENROUTER
    assert entry.deployment is None
    assert entry.api_version is None


def test_flag_supplied_backend_knobs_seed_the_matching_backends_pass(tmp_path: Path) -> None:
    # `--deployment`/`--api-version` described the provider being SET; letting them follow the
    # user to a different backend would stamp an Azure api-version onto an OpenRouter entry.
    pool = tmp_path / "pool.toml"
    options = EntryOptions(deployment="prod-gpt55", api_version="2030-01-01")
    written, _, _ = _drive(
        pool,
        [
            "y",
            "azure",
            "gpt-5.5",
            "",
            "frontier",
            "",
            "",  # accept the flag-supplied deployment
            "",  # accept the flag-supplied api-version
            "",  # handle
            "y",
            "openrouter",
            *_openrouter_pass(_SONNET),
            "n",
        ],
        kind=ProviderKind.AZURE_OPENAI,
        options=options,
    )

    assert written == 2
    entries = {entry.name: entry for entry in read_pool_entries(pool)}
    assert entries["prod-gpt55"].deployment == "prod-gpt55"
    assert entries["prod-gpt55"].api_version == "2030-01-01"
    assert entries["claude-sonnet-4.5"].deployment is None
    assert entries["claude-sonnet-4.5"].api_version is None


def test_register_model_ids_writes_without_prompting(tmp_path: Path) -> None:
    # The scripted path: same roster a person would build by hand, no terminal required.
    pool = tmp_path / "pool.toml"
    console = Console(file=StringIO(), width=120, no_color=True)

    written = register_model_ids(
        console,
        pool_path=pool,
        kind=ProviderKind.OPENROUTER,
        model_ids=[_SONNET, _DEEPSEEK],
        options=EntryOptions(tier="open"),
        verify=_ok,
    )

    assert written == 2
    entries = read_pool_entries(pool)
    assert [entry.name for entry in entries] == ["claude-sonnet-4.5", "deepseek-v3.2"]
    assert all(entry.tier == "open" for entry in entries)


def test_register_model_ids_writes_a_loadable_roster_for_two_built_in_priced_ids(
    tmp_path: Path,
) -> None:
    """Two ids in one pass must leave a pool `load_pool` can read.

    `--pool-model` writes one entry per id through `upsert_pool_entry`, so it is not a workaround
    for the duplicate-key corruption: entry #2 takes the same append path. The ids here are
    deliberately built-in priced ones, whose entries carry no price fields and so render short.
    The OpenRouter batch above cannot catch this: its entries are stamped with resolved prices,
    which pushes them past the length heuristic that decided the rendering.
    """
    pool = tmp_path / "pool.toml"
    console = Console(file=StringIO(), width=120, no_color=True)

    written = register_model_ids(
        console,
        pool_path=pool,
        kind=ProviderKind.OPENAI,
        model_ids=["gpt-5.5", "gpt-5.4-mini"],
        options=EntryOptions(),
        verify=_ok,
    )

    assert written == 2
    assert [entry.name for entry in load_pool(pool).models] == ["gpt-5.5", "gpt-5.4-mini"]


def test_register_model_ids_is_idempotent(tmp_path: Path) -> None:
    pool = tmp_path / "pool.toml"
    console = Console(file=StringIO(), width=120, no_color=True)
    register_model_ids(
        console,
        pool_path=pool,
        kind=ProviderKind.OPENROUTER,
        model_ids=[_SONNET],
        options=EntryOptions(),
        verify=_ok,
    )
    before = pool.read_text(encoding="utf-8")

    written = register_model_ids(
        console,
        pool_path=pool,
        kind=ProviderKind.OPENROUTER,
        model_ids=[_SONNET],
        options=EntryOptions(),
        verify=_ok,
    )

    assert written == 0
    assert pool.read_text(encoding="utf-8") == before


def test_register_model_ids_pings_every_candidate_on_its_own_route(tmp_path: Path) -> None:
    # A --pool-model can differ from the verified worker in model, endpoint, deployment and
    # credential, so the worker's ping proves nothing about it, and a script is exactly where
    # nobody is watching for the 401 that would otherwise surface inside a paid sweep.
    pool = tmp_path / "pool.toml"
    console = Console(file=StringIO(), width=120, no_color=True)
    pinged: list[str] = []

    def record(entry: PoolEntry) -> VerifyResult:
        pinged.append(entry.model)
        return _ok(entry)

    register_model_ids(
        console,
        pool_path=pool,
        kind=ProviderKind.OPENROUTER,
        model_ids=[_SONNET, _DEEPSEEK],
        options=EntryOptions(),
        verify=record,
    )

    assert pinged == [_SONNET, _DEEPSEEK]


def test_register_model_ids_writes_nothing_when_a_candidate_is_not_callable(
    tmp_path: Path,
) -> None:
    # Built and proved before anything is written, so a bad second id cannot leave half a roster
    # behind for a script that then reports failure.
    pool = tmp_path / "pool.toml"
    console = Console(file=StringIO(), width=120, no_color=True)

    def refuse_second(entry: PoolEntry) -> VerifyResult:
        if entry.model == _DEEPSEEK:
            return VerifyResult(ok=False, kind=entry.kind, model=entry.model, detail="401")
        return _ok(entry)

    with pytest.raises(typer.BadParameter, match="is not callable"):
        register_model_ids(
            console,
            pool_path=pool,
            kind=ProviderKind.OPENROUTER,
            model_ids=[_SONNET, _DEEPSEEK],
            options=EntryOptions(),
            verify=refuse_second,
        )
    assert not pool.exists()


def test_register_model_ids_refuses_a_model_it_cannot_price(tmp_path: Path) -> None:
    # Non-interactively there is nobody to ask, and a silently unpriced candidate reports $0.
    pool = tmp_path / "pool.toml"
    console = Console(file=StringIO(), width=120, no_color=True)

    with pytest.raises(typer.BadParameter, match="no built-in price"):
        register_model_ids(
            console,
            pool_path=pool,
            kind=ProviderKind.OPENAI,
            model_ids=["mystery-model"],
            options=EntryOptions(),
        )
    assert not pool.exists()


def test_register_model_ids_resolves_a_bedrock_id_from_the_built_in_registry(
    tmp_path: Path,
) -> None:
    # A scripted registration must pick up the same runtime id and canonical type the picker
    # would, so `--pool-model claude-opus-4-8` is callable rather than a literal that is not.
    pool = tmp_path / "pool.toml"
    console = Console(file=StringIO(), width=120, no_color=True)

    register_model_ids(
        console,
        pool_path=pool,
        kind=ProviderKind.BEDROCK,
        model_ids=["claude-opus-4-8"],
        options=EntryOptions(region="us-east-1"),
        verify=_ok,
    )

    entry = read_pool_entries(pool)[0]
    assert entry.name == "claude-opus-4-8"
    assert entry.model == "us.anthropic.claude-opus-4-8"
    assert entry.model_type == "claude-opus-4-8"
    assert entry.region == "us-east-1"


def test_every_pass_is_live_verified_including_a_switched_backend(tmp_path: Path) -> None:
    # Presence of an environment variable is not proof it holds a working key. A backend the
    # command never verified must not silently contribute candidates that 401 at sweep time,
    # after the sweep has paid for every candidate ahead of them.
    pool = tmp_path / "pool.toml"
    seen: list[PoolEntry] = []

    def record(entry: PoolEntry) -> VerifyResult:
        seen.append(entry)
        return _ok(entry)

    _drive(
        pool,
        [
            "y",
            "openrouter",
            *_openrouter_pass(_SONNET),
            "y",
            "azure",
            "gpt-5.4",
            "",
            "frontier",
            "",
            "prod-gpt54",
            "2025-01-01-preview",
            "",
            "n",
        ],
        verify=record,
    )

    assert [(entry.kind, entry.model) for entry in seen] == [
        (ProviderKind.OPENROUTER, _SONNET),
        (ProviderKind.AZURE_OPENAI, "gpt-5.4"),
    ]
    # The ping goes out with the connection details the entry will be written with.
    assert seen[1].deployment == "prod-gpt54"
    assert seen[1].api_version == "2025-01-01-preview"


def test_every_candidate_is_pinged_on_its_own_route(tmp_path: Path) -> None:
    # A pass is not one route. Every Azure candidate names a different operator-chosen
    # deployment, so a first deployment that answers proves nothing about the second, and a
    # once-per-pass ping would wave the rest through.
    pool = tmp_path / "pool.toml"
    pinged: list[str] = []

    def record(entry: PoolEntry) -> VerifyResult:
        pinged.append(entry.deployment or entry.model)
        return _ok(entry)

    written, _, _ = _drive(
        pool,
        [
            "y",
            "azure",
            "gpt-5.4",
            "gpt-5.5",
            "",
            "frontier",
            "",  # api key env
            "chat-prod",
            "2025-01-01-preview",
            "",  # gpt-5.4: deployment, api-version, handle
            "chat-preview",
            "2025-01-01-preview",
            "",  # gpt-5.5
            "n",
        ],
        kind=ProviderKind.AZURE_OPENAI,
        verify=record,
    )

    assert written == 2
    assert pinged == ["chat-prod", "chat-preview"]


def test_a_candidate_that_fails_its_ping_does_not_take_the_others_down(tmp_path: Path) -> None:
    # Per-candidate verification means a per-candidate decision: one unreachable deployment is
    # declined on its own, and the rest of the selection still lands.
    pool = tmp_path / "pool.toml"

    def refuse_second(entry: PoolEntry) -> VerifyResult:
        if entry.model == _DEEPSEEK:
            return VerifyResult(ok=False, kind=entry.kind, model=entry.model, detail="404 model")
        return _ok(entry)

    written, _, _ = _drive(
        pool,
        ["y", "openrouter", _SONNET, _DEEPSEEK, "", "frontier", "", "", "n", "n"],
        verify=refuse_second,
    )

    assert written == 1
    assert [entry.model for entry in read_pool_entries(pool)] == [_SONNET]


def test_a_failed_ping_registers_nothing_unless_it_is_overridden(tmp_path: Path) -> None:
    pool = tmp_path / "pool.toml"

    def refuse(entry: PoolEntry) -> VerifyResult:
        return VerifyResult(ok=False, kind=entry.kind, model=entry.model, detail="401 bad key")

    written, output, _ = _drive(
        pool,
        ["y", "openrouter", _SONNET, "", "frontier", "", "n", "n"],
        verify=refuse,
    )

    assert written == 0
    assert not pool.exists()
    assert "401 bad key" in output


def test_a_failed_ping_can_be_overridden_for_a_machine_without_the_key(tmp_path: Path) -> None:
    # A roster is legitimately built where the credential is not, so the failure is reported and
    # the choice is the user's rather than a silent discard of everything they just picked.
    pool = tmp_path / "pool.toml"

    def refuse(entry: PoolEntry) -> VerifyResult:
        return VerifyResult(ok=False, kind=entry.kind, model=entry.model, detail="offline")

    written, _, _ = _drive(
        pool,
        ["y", "openrouter", _SONNET, "", "frontier", "", "y", "", "n"],
        verify=refuse,
    )

    assert written == 1
    assert read_pool_entries(pool)[0].model == _SONNET


def test_half_a_price_pair_still_prompts_for_both(tmp_path: Path) -> None:
    # `PoolEntry` takes both prices or neither, so treating a supplied input price as "already
    # priced" would skip the model with a validation error instead of asking for the other half.
    pool = tmp_path / "pool.toml"
    written, _, _ = _drive(
        pool,
        ["y", "openai", "", "self-hosted", "y", "", "open", "", "0.2", "0.8", "", "n"],
        options=EntryOptions(input_per_mtok=0.1),
    )

    assert written == 1
    entry = read_pool_entries(pool)[0]
    assert (entry.input_per_mtok, entry.output_per_mtok) == (0.2, 0.8)


def test_one_azure_deployment_cannot_describe_several_pool_models(tmp_path: Path) -> None:
    # Azure sends the deployment as the request's model id, so reusing one deployment across
    # several ids registers several candidates that all call the same deployment.
    pool = tmp_path / "pool.toml"
    console = Console(file=StringIO(), width=120, no_color=True)

    with pytest.raises(typer.BadParameter, match="once per deployment"):
        register_model_ids(
            console,
            pool_path=pool,
            kind=ProviderKind.AZURE_OPENAI,
            model_ids=["gpt-5.4", "gpt-5.5"],
            options=EntryOptions(deployment="prod"),
        )
    assert not pool.exists()


def test_a_scripted_azure_registration_must_name_its_deployment(tmp_path: Path) -> None:
    # Deployment names are chosen by whoever created them. Defaulting to the model id would
    # write a candidate addressing a route that does not exist, and a script has nobody to ask.
    pool = tmp_path / "pool.toml"
    console = Console(file=StringIO(), width=120, no_color=True)

    with pytest.raises(typer.BadParameter, match="azure needs --deployment"):
        register_model_ids(
            console,
            pool_path=pool,
            kind=ProviderKind.AZURE_OPENAI,
            model_ids=["gpt-5.4"],
            options=EntryOptions(),
        )
    assert not pool.exists()


def test_one_azure_deployment_registers_cleanly(tmp_path: Path) -> None:
    pool = tmp_path / "pool.toml"
    console = Console(file=StringIO(), width=120, no_color=True)

    register_model_ids(
        console,
        pool_path=pool,
        kind=ProviderKind.AZURE_OPENAI,
        model_ids=["gpt-5.4"],
        options=EntryOptions(deployment="chat-prod", api_version="2025-01-01-preview"),
        verify=_ok,
    )

    entry = read_pool_entries(pool)[0]
    assert (entry.name, entry.model, entry.deployment) == ("chat-prod", "gpt-5.4", "chat-prod")
    assert entry.api_version == "2025-01-01-preview"


@pytest.mark.parametrize("kind", list(ProviderKind))
def test_a_price_is_prompted_exactly_when_the_entry_would_be_rejected(kind: ProviderKind) -> None:
    # The invariant behind "ask only what the kind needs": `needs_price` has to agree with
    # `PoolEntry`'s own validation on every row of every catalog, or the flow either asks for a
    # number it will not use or skips a model it could have registered.
    for model in list_provider_models(kind).models:
        try:
            build_pool_entry(name="probe", kind=kind, model=model, options=EntryOptions())
        except ValidationError:
            accepted = False
        else:
            accepted = True
        assert accepted is not needs_price(kind, model), model.id


def test_no_openrouter_model_needs_a_price_while_the_catalog_is_reachable() -> None:
    # The #284 exception, pinned: OpenRouter self-prices, so its whole catalog registers with
    # zero price prompts.
    catalog = list_provider_models(ProviderKind.OPENROUTER)
    assert catalog.models
    assert not any(needs_price(ProviderKind.OPENROUTER, model) for model in catalog.models)


def test_pool_path_follows_the_root_unless_it_is_overridden(tmp_path: Path) -> None:
    assert pool_path_for(tmp_path / ".wmo") == tmp_path / ".wmo" / "pool.toml"
    assert pool_path_for(tmp_path, "elsewhere/roster.toml") == Path("elsewhere/roster.toml")
    # The default root resolves to exactly what the routing commands read.
    assert pool_path_for(".wmo") == Path(".wmo/pool.toml")


def _local_catalog(endpoint: str) -> ProviderCatalog:
    """What a live Ollama would list, without the network."""
    return ProviderCatalog(
        kind=ProviderKind.OPENAI,
        source=CatalogSource.PUBLISHED,
        models=[CatalogModel(id="qwen3:4b")],
        detail=f"1 models served at {endpoint.rstrip('/')}/models",
    )


def test_an_openai_endpoint_pass_lists_the_server_and_defaults_the_price_to_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The whole local-model flow in one interactive pass: URL, the server's own model list,
    # explicit $0 stamped (blank answers take the self-hosted default), plain openai on the wire.
    monkeypatch.setattr(pool_registry, "endpoint_catalog", _local_catalog)
    pool = tmp_path / "pool.toml"
    written, output, _ = _drive(
        pool,
        [
            "y",
            "openai",
            "http://localhost:11434/v1",  # the endpoint prompt
            "1",  # pick qwen3:4b off the server's list
            "",  # done choosing
            "open",
            "",  # api_key_env: none, a local server needs no key
            "",  # input price: blank takes the self-hosted default 0
            "",  # output price: same
            "",  # handle: default qwen3-4b
            "n",
        ],
        kind=ProviderKind.OPENAI,
    )

    assert written == 1
    assert "reads as a locally hosted server" in output
    entry = read_pool_entries(pool)[0]
    assert entry.name == "qwen3-4b"
    assert entry.kind is ProviderKind.OPENAI
    assert entry.model == "qwen3:4b"
    assert entry.endpoint == "http://localhost:11434/v1"
    assert entry.api_key_env is None
    # Explicit zeros, not absent prices: declared free is not the unpriced accident.
    assert (entry.input_per_mtok, entry.output_per_mtok) == (0.0, 0.0)


def test_a_self_hosted_id_shadowing_a_built_in_model_still_takes_an_explicit_price(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Someone serving "gpt-5.4" locally must not inherit OpenAI's hosted rate: the price a
    # candidate bills at is a property of the SERVER, and this one is the operator's own.
    monkeypatch.setattr(
        pool_registry,
        "endpoint_catalog",
        lambda endpoint: ProviderCatalog(
            kind=ProviderKind.OPENAI,
            source=CatalogSource.PUBLISHED,
            models=[CatalogModel(id="gpt-5.4")],
        ),
    )
    pool = tmp_path / "pool.toml"
    written, _, _ = _drive(
        pool,
        ["y", "openai", "http://localhost:8001/v1", "1", "", "open", "", "", "", "", "n"],
        kind=ProviderKind.OPENAI,
    )

    assert written == 1
    entry = read_pool_entries(pool)[0]
    assert (entry.input_per_mtok, entry.output_per_mtok) == (0.0, 0.0)
    assert entry.price().input_per_mtok == 0.0


def test_register_model_ids_prices_a_self_hosted_candidate_at_zero_without_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The scripted twin: `wmo providers set --provider openai --endpoint <url> --pool-model ...`
    # with no price flags lands as an explicit $0 pair, never as an unpriced entry.
    monkeypatch.setattr(pool_registry, "endpoint_catalog", _local_catalog)
    pool = tmp_path / "pool.toml"
    console = Console(file=StringIO(), width=120, no_color=True)

    written = register_model_ids(
        console,
        pool_path=pool,
        kind=ProviderKind.OPENAI,
        model_ids=["qwen3:4b"],
        options=EntryOptions(endpoint="http://localhost:11434/v1", tier="open"),
        verify=_ok,
    )

    assert written == 1
    entry = read_pool_entries(pool)[0]
    assert entry.endpoint == "http://localhost:11434/v1"
    assert (entry.input_per_mtok, entry.output_per_mtok) == (0.0, 0.0)
    assert isinstance(console.file, StringIO)
    assert "pricing at $0/Mtok" in console.file.getvalue()


def test_the_same_model_on_two_endpoints_is_two_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The endpoint is part of candidate identity, like the account: collapsing the two would
    # silently delete one server's entry on re-registration.
    monkeypatch.setattr(pool_registry, "endpoint_catalog", _local_catalog)
    pool = tmp_path / "pool.toml"
    console = Console(file=StringIO(), width=120, no_color=True)
    for endpoint in ("http://localhost:11434/v1", "http://silens-mac.local:8001/v1"):
        register_model_ids(
            console,
            pool_path=pool,
            kind=ProviderKind.OPENAI,
            model_ids=["qwen3:4b"],
            options=EntryOptions(endpoint=endpoint),
            verify=_ok,
        )

    entries = read_pool_entries(pool)
    assert [entry.name for entry in entries] == ["qwen3-4b", "qwen3-4b-2"]
    assert {entry.endpoint for entry in entries} == {
        "http://localhost:11434/v1",
        "http://silens-mac.local:8001/v1",
    }


def test_re_registration_keeps_a_disabled_entry_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `enabled = false` is an explicit operator edit; a registration re-run must not silently
    # put the candidate back into selection.
    monkeypatch.setattr(pool_registry, "endpoint_catalog", _local_catalog)
    pool = tmp_path / "pool.toml"
    console = Console(file=StringIO(), width=120, no_color=True)
    options = EntryOptions(endpoint="http://localhost:11434/v1")
    register_model_ids(
        console,
        pool_path=pool,
        kind=ProviderKind.OPENAI,
        model_ids=["qwen3:4b"],
        options=options,
        verify=_ok,
    )
    pool.write_text(
        pool.read_text(encoding="utf-8").replace(
            'name = "qwen3-4b"', 'name = "qwen3-4b"\nenabled = false'
        ),
        encoding="utf-8",
    )

    register_model_ids(
        console,
        pool_path=pool,
        kind=ProviderKind.OPENAI,
        model_ids=["qwen3:4b"],
        options=options,
        verify=_ok,
    )

    entries = read_pool_entries(pool)
    assert [entry.name for entry in entries] == ["qwen3-4b"]
    assert entries[0].enabled is False
    assert isinstance(console.file, StringIO)
    assert "keeping it disabled" in console.file.getvalue()


def test_equivalent_endpoint_spellings_are_one_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Scheme/host case and a default port are the same route; a sweep must not pay for the
    # same backend twice under two handles.
    monkeypatch.setattr(pool_registry, "endpoint_catalog", _local_catalog)
    pool = tmp_path / "pool.toml"
    console = Console(file=StringIO(), width=120, no_color=True)
    for endpoint in ("http://localhost:80/v1", "HTTP://LocalHost/v1"):
        register_model_ids(
            console,
            pool_path=pool,
            kind=ProviderKind.OPENAI,
            model_ids=["qwen3:4b"],
            options=EntryOptions(endpoint=endpoint),
            verify=_ok,
        )

    assert [entry.name for entry in read_pool_entries(pool)] == ["qwen3-4b"]


def test_a_seeded_endpoint_can_be_cleared_back_to_openais_own_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A session started with --endpoint must still be able to register hosted candidates:
    # '-' clears the seed, and the prompt says so instead of promising a blank that cannot work.
    monkeypatch.setattr(
        pool_registry,
        "endpoint_catalog",
        lambda endpoint: (_ for _ in ()).throw(AssertionError("hosted pass must not probe")),
    )
    pool = tmp_path / "pool.toml"
    written, output, script = _drive(
        pool,
        ["y", "openai", "-", "gpt-5.4", "", "frontier", "", "", "n"],
        kind=ProviderKind.OPENAI,
        options=EntryOptions(endpoint="http://localhost:11434/v1"),
    )

    assert written == 1
    entry = read_pool_entries(pool)[0]
    assert entry.endpoint is None
    assert entry.input_per_mtok is None  # hosted gpt-5.4 keeps its built-in price
    assert any("'-' = OpenAI's own API" in prompt for prompt in script.prompts)


def test_a_remote_paid_endpoint_is_never_defaulted_to_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Together/Groq/a proxy bill real money: the scripted path demands the price loudly,
    # exactly as main did for any unpriced candidate.
    monkeypatch.setattr(
        pool_registry,
        "endpoint_catalog",
        lambda endpoint: ProviderCatalog(
            kind=ProviderKind.OPENAI,
            source=CatalogSource.PUBLISHED,
            models=[CatalogModel(id="llama-3.3-70b")],
        ),
    )
    pool = tmp_path / "pool.toml"
    console = Console(file=StringIO(), width=120, no_color=True)
    with pytest.raises(typer.BadParameter, match="cannot be assumed"):
        register_model_ids(
            console,
            pool_path=pool,
            kind=ProviderKind.OPENAI,
            model_ids=["llama-3.3-70b"],
            options=EntryOptions(endpoint="https://api.together.xyz/v1"),
            verify=_ok,
        )
    assert not pool.exists()


def test_a_half_supplied_price_pair_is_refused_not_zero_filled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only output_per_mtok given: zero-filling the pair would silently discard the declared
    # rate; the loud both-or-neither rule the CLI flags already have applies here too.
    monkeypatch.setattr(pool_registry, "endpoint_catalog", _local_catalog)
    console = Console(file=StringIO(), width=120, no_color=True)
    with pytest.raises(typer.BadParameter, match="both"):
        register_model_ids(
            console,
            pool_path=tmp_path / "pool.toml",
            kind=ProviderKind.OPENAI,
            model_ids=["qwen3:4b"],
            options=EntryOptions(endpoint="http://localhost:11434/v1", output_per_mtok=12.0),
            verify=_ok,
        )


def test_endpoint_identity_survives_bad_ports_and_keeps_queries_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A non-numeric port must not crash candidate comparison, and two proxy endpoints
    # differing only by a routing query parameter are two candidates, not one.
    monkeypatch.setattr(pool_registry, "endpoint_catalog", _local_catalog)
    pool = tmp_path / "pool.toml"
    console = Console(file=StringIO(), width=120, no_color=True)
    for endpoint in (
        "http://localhost:port/v1",  # unparsable port: raw-text identity, no crash
        "http://localhost:8000/v1?tenant=a",
        "http://localhost:8000/v1?tenant=b",
    ):
        register_model_ids(
            console,
            pool_path=pool,
            kind=ProviderKind.OPENAI,
            model_ids=["qwen3:4b"],
            options=EntryOptions(endpoint=endpoint),
            verify=_ok,
        )

    entries = read_pool_entries(pool)
    assert len(entries) == 3
    assert {entry.endpoint for entry in entries} == {
        "http://localhost:port/v1",
        "http://localhost:8000/v1?tenant=a",
        "http://localhost:8000/v1?tenant=b",
    }


def test_endpoints_differing_only_by_userinfo_are_two_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Embedded credentials route differently (two proxy tenants); the second registration
    # must not replace the first credentialed candidate.
    monkeypatch.setattr(pool_registry, "endpoint_catalog", _local_catalog)
    pool = tmp_path / "pool.toml"
    console = Console(file=StringIO(), width=120, no_color=True)
    for endpoint in ("http://alice@localhost:8000/v1", "http://bob@localhost:8000/v1"):
        register_model_ids(
            console,
            pool_path=pool,
            kind=ProviderKind.OPENAI,
            model_ids=["qwen3:4b"],
            options=EntryOptions(endpoint=endpoint),
            verify=_ok,
        )

    entries = read_pool_entries(pool)
    assert len(entries) == 2
    assert {entry.endpoint for entry in entries} == {
        "http://alice@localhost:8000/v1",
        "http://bob@localhost:8000/v1",
    }
