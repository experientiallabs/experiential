"""Tests for the terminal UX: the non-TTY build reporter and the play REPL (injected I/O)."""

from __future__ import annotations

import importlib
import os

import pytest
import typer
from rich.console import Console

from wmh.cli.ui import (
    BuildParams,
    RichBuildReporter,
    _decode_key,
    _step_selection,
    models_table,
    run_build_wizard,
    run_play_repl,
    select_model,
)
from wmh.config import PROVIDER_ENV_VARS, ModelInfo
from wmh.core.types import Action, ActionKind, Observation, Step, Trace
from wmh.engine.world_model import WorldModel
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder

ui_module = importlib.import_module("wmh.cli.ui")


@pytest.fixture(autouse=True)
def _all_provider_creds(monkeypatch) -> None:  # noqa: ANN001 - pytest fixture
    """Deterministic creds baseline: every provider env var set, so wizard tests never hit the
    interactive missing-credential prompts unless a test clears vars explicitly."""
    for env_vars in PROVIDER_ENV_VARS.values():
        for var in env_vars:
            monkeypatch.setenv(var, "test-cred")


def _scripted_reader(answers: list[str]):  # noqa: ANN202 - returns a PromptReader
    """A PromptReader that returns successive `answers`, ignoring the rendered prompt text."""
    it = iter(answers)
    return lambda _prompt: next(it)


class FakeProvider:
    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        return Completion(
            text='{"output": "found u1", "is_error": false, "state_note": "looked up u1"}'
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def _world_model() -> WorldModel:
    retriever = EmbeddingRetriever(HashingEmbedder(dim=32))
    retriever.index(
        [
            Trace(
                trace_id="t",
                steps=[
                    Step(
                        action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={}),
                        observation=Observation(content="found u0"),
                    )
                ],
            )
        ]
    )
    return WorldModel(FakeProvider(), retriever, top_k=3)


def test_reporter_degrades_to_plain_lines_when_not_a_tty() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    reporter = RichBuildReporter(console, "airline")
    with console.capture() as cap:
        reporter.ingest_done(3, 9)
        reporter.split_done(2, 1)
        reporter.index_done(9)
        reporter.optimize_start(20)
        reporter.rollout(1, 20, 0.4)
        reporter.rollout(10, 20, 0.6)
        reporter.optimize_done(0.6, 2, 20)
    out = cap.get()
    assert "ingested 3 traces" in out
    assert "normalized 9 steps" in out
    assert "2 train / 1 held-out" in out
    assert "GEPA metric call 1/20" in out  # non-TTY heartbeat
    assert "GEPA metric call 10/20" in out
    assert "held-out 0.600" in out


def test_reporter_does_not_show_impossible_progress_denominator() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    reporter = RichBuildReporter(console, "airline")
    with console.capture() as cap:
        reporter.optimize_start(1)
        reporter.rollout(10, 1, 0.6)
    out = cap.get()
    assert "GEPA metric call 10 (budget target 1)" in out
    assert "10/1" not in out


def test_models_table_renders_names() -> None:
    console = Console(force_terminal=False, no_color=True, width=120)
    table = models_table([ModelInfo(name="airline", serve_provider="bedrock", serve_model="opus")])
    with console.capture() as cap:
        console.print(table)
    assert "airline" in cap.get()


def test_play_repl_renders_observation_and_state_then_quits() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    reader = _scripted_reader(['get_user {"id": "u1"}', ":state", ":quit"])
    with console.capture() as cap:
        run_play_repl(console, _world_model(), "airline", task="look up users", reader=reader)
    out = cap.get()
    assert "found u1" in out  # observation rendered
    assert "looked up u1" in out  # scratchpad updated and shown by :state
    assert "bye" in out


def test_play_repl_reports_parse_errors_without_crashing() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    reader = _scripted_reader(['get_user ["bad"]', ":quit"])
    with console.capture() as cap:
        run_play_repl(console, _world_model(), "airline", task=None, reader=reader)
    assert "parse error" in cap.get()


def test_play_repl_exits_cleanly_on_eof() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)

    def eof(_prompt: str) -> str:
        raise EOFError

    with console.capture() as cap:
        run_play_repl(console, _world_model(), "airline", task=None, reader=eof)
    assert "bye" in cap.get()


# --- creation wizard -----------------------------------------------------------------------------


def test_decode_key_maps_arrows_and_passes_plain_chars() -> None:
    assert _decode_key("\x1b[A") == "up"
    assert _decode_key("\x1bOA") == "up"  # application cursor mode
    assert _decode_key("\x1b[B") == "down"
    assert _decode_key("\x1b[1;5A") == "esc"  # modified arrows are inert, not a stray '5'
    assert _decode_key("\x1b[5~") == "esc"  # PgUp is inert
    assert _decode_key("\x1b") == "esc"
    assert _decode_key("j") == "j"
    assert _decode_key("\r") == "\r"


def test_arrow_select_moves_pointer_and_accepts(monkeypatch) -> None:  # noqa: ANN001
    keys = iter(["\x1b[B", "\x1b[1;5A", "\r"])  # down, inert modified arrow, Enter
    monkeypatch.setattr(ui_module.click, "getchar", lambda: next(keys))
    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    assert ui_module._arrow_select(console, ["a", "b", "c"], 0) == 1
    assert "\u276f b" in console.export_text()  # pointer painted on the accepted row


def test_arrow_select_aborts_on_eof(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(ui_module.click, "getchar", lambda: "")
    console = Console(force_terminal=False, no_color=True, width=100)
    with pytest.raises(typer.Abort):  # closed stdin must abort, not busy-loop
        ui_module._arrow_select(console, ["a", "b"], 0)


def test_step_selection_navigates_wraps_and_accepts() -> None:
    assert _step_selection("down", 0, 3) == (1, False)
    assert _step_selection("j", 1, 3) == (2, False)
    assert _step_selection("down", 2, 3) == (0, False)  # wraps
    assert _step_selection("up", 0, 3) == (2, False)  # wraps backwards
    assert _step_selection("k", 2, 3) == (1, False)
    assert _step_selection("\r", 1, 3) == (1, True)  # Enter accepts the highlight
    assert _step_selection("2", 0, 3) == (1, True)  # digits jump-select
    assert _step_selection("9", 0, 3) == (0, False)  # out-of-range digit is inert
    assert _step_selection("x", 1, 3) == (1, False)  # unknown keys are inert


def test_build_wizard_collects_all_inputs() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    # Prompts in order: name, file, provider (select), model (select), region (bedrock only),
    # budget, embedder (select). No embed-model (hashing default) and no phi-dim prompt.
    reader = _scripted_reader(
        [
            "tau2-airline",
            "/tmp/traces.jsonl",
            "bedrock",
            "us.anthropic.claude-opus-4-8",
            "us-east-1",
            "8",
            "hashing",
        ]
    )
    # train_split has no wizard prompt; it must carry through from the flag-supplied defaults.
    params = run_build_wizard(console, BuildParams(name="default", train_split=0.5), reader=reader)
    assert params.name == "tau2-airline"
    assert params.file == "/tmp/traces.jsonl"
    assert params.provider == "bedrock"
    assert params.region == "us-east-1"
    assert params.gepa_budget == 8
    assert params.embed_provider == "hashing"
    assert params.train_split == 0.5


def test_build_wizard_select_by_number() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    # Provider/model/embedder are numbered pickers; choosing by index must work. Pick anthropic (2),
    # its second model, no region prompt (not bedrock), budget 8, hashing embedder (1).
    reader = _scripted_reader(["m", "/tmp/t.jsonl", "2", "2", "8", "1"])
    params = run_build_wizard(console, BuildParams(name="default"), reader=reader)
    assert params.provider == "anthropic"
    assert params.model == "claude-opus-4-7"  # second anthropic model
    assert params.region is None  # region only prompted for bedrock
    assert params.embed_provider == "hashing"


def test_build_wizard_collects_provider_embedder() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    # A provider-backed embedder adds an embeddings-model picker; phi dim keeps its default.
    reader = _scripted_reader(
        ["m", "/tmp/t.jsonl", "openai", "gpt-5.5", "8", "openai", "text-embedding-3-large"]
    )
    params = run_build_wizard(console, BuildParams(name="default"), reader=reader)
    assert params.embed_provider == "openai"
    assert params.embed_model == "text-embedding-3-large"
    assert params.embed_dim == 512  # default, no longer prompted


def test_build_wizard_accepts_defaults_with_blank_input() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    # File provided (so that prompt is skipped); press Enter (blank) for every remaining prompt
    # (name/provider/model/region/budget/embedder) to accept the suggested defaults.
    reader = _scripted_reader(["", "", "", "", "", ""])
    defaults = BuildParams(name="seeded", file="/tmp/t.jsonl", provider="bedrock", gepa_budget=50)
    params = run_build_wizard(console, defaults, reader=reader)
    assert params.name == "seeded"  # blank kept the default
    assert params.provider == "bedrock"
    assert params.gepa_budget == 50
    assert params.region == "us-east-1"  # bedrock default suggested + accepted
    assert params.embed_provider == "hashing"  # default embedder, no embed-model prompt


def test_build_wizard_dashes_spaces_in_name() -> None:
    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    # Whitespace is dash-joined rather than rejected: typing "tau bench" quietly becomes
    # "tau-bench", with a dim note showing the name actually used.
    reader = _scripted_reader(["tau bench", "/tmp/t.jsonl", "", "", "", "", ""])
    params = run_build_wizard(console, BuildParams(name="default"), reader=reader)
    assert params.name == "tau-bench"
    assert "using tau-bench" in console.export_text()


def test_build_wizard_reprompts_on_invalid_name() -> None:
    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    # A name normalization can't rescue ("tau/bench" has a path separator) must re-prompt with
    # the friendly validation message, not escape as a ValueError traceback.
    reader = _scripted_reader(["tau/bench", "tau-bench", "/tmp/t.jsonl", "", "", "", "", ""])
    params = run_build_wizard(console, BuildParams(name="default"), reader=reader)
    assert params.name == "tau-bench"
    assert "invalid world model name" in console.export_text()


def test_build_wizard_normalizes_flag_name_in_default() -> None:
    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    # A whitespace-y --name flag becomes the normalized bracketed default, acceptable via Enter
    # on the first prompt (no validation error, no re-prompt).
    reader = _scripted_reader(["", "/tmp/t.jsonl", "", "", "", "", ""])
    params = run_build_wizard(console, BuildParams(name="tau bench"), reader=reader)
    assert params.name == "tau-bench"
    assert "invalid world model name" not in console.export_text()


def test_build_wizard_drops_invalid_flag_name_from_default() -> None:
    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    # An unrescuable --name flag must not become the bracketed Enter-default (it could never be
    # accepted); blank input then means "no name yet" and re-prompts until a valid one arrives.
    reader = _scripted_reader(["", "tau-bench", "/tmp/t.jsonl", "", "", "", "", ""])
    params = run_build_wizard(console, BuildParams(name="tau/bench"), reader=reader)
    assert params.name == "tau-bench"
    assert "[tau/bench]" not in console.export_text()


def test_build_wizard_aborts_cleanly_on_eof() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)

    def eof_reader(_prompt: str) -> str:
        raise EOFError

    # Exhausted piped stdin (or Ctrl-D) must abort the wizard cleanly, not leak an EOFError
    # traceback. typer.Abort is handled by click's runner ("Aborted.", exit 1).
    with pytest.raises(typer.Abort):
        run_build_wizard(console, BuildParams(name="default"), reader=eof_reader)


def test_build_wizard_reprompts_on_blank_trace_source() -> None:
    # A blank traces path must re-ask, not crash. After a name, blank the file once, then give a
    # real path; remaining prompts (provider/model/region/budget/embedder) take defaults.
    console = Console(force_terminal=False, no_color=True, width=100)
    reader = _scripted_reader(["mymodel", "", "/tmp/t.jsonl", "", "", "", "", ""])
    params = run_build_wizard(console, BuildParams(name="default"), reader=reader)
    assert params.name == "mymodel"
    assert params.file == "/tmp/t.jsonl"


def test_build_wizard_prompts_for_missing_credentials_and_saves(
    tmp_path,  # noqa: ANN001 - pytest fixture
    monkeypatch,  # noqa: ANN001 - pytest fixture
) -> None:
    # Picking a provider with unset creds prompts for each env var; entered values land in
    # os.environ AND .env (so the next session has them), Enter skips a var with a warning.
    for var in ("AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    reader = _scripted_reader(
        # name, file, provider, AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (skip),
        # model, region, budget, embedder
        ["m", "/tmp/t.jsonl", "bedrock", "us-east-1", "test-key-id", "", "1", "", "8", "1"]
    )
    params = run_build_wizard(console, BuildParams(name="default"), reader=reader)
    out = console.export_text()
    assert params.provider == "bedrock"
    assert os.environ["AWS_ACCESS_KEY_ID"] == "test-key-id"
    env_file = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "AWS_REGION=us-east-1" in env_file
    assert "AWS_ACCESS_KEY_ID=test-key-id" in env_file
    assert "AWS_SECRET_ACCESS_KEY" not in env_file  # skipped with Enter
    assert "AWS_SECRET_ACCESS_KEY still unset" in out


def test_build_wizard_keeps_session_creds_when_persistence_fails(
    monkeypatch,  # noqa: ANN001 - pytest fixture
) -> None:
    # A refused .env write (symlink, ELOOP, read-only dir) must not crash the wizard: the
    # credential still applies to the running session, with a visible warning.
    for var in ("AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(var, raising=False)

    def refuse(var: str, value: str) -> None:
        raise OSError("too many levels of symbolic links")

    monkeypatch.setattr(ui_module, "upsert_env_var", refuse)
    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    reader = _scripted_reader(
        ["m", "/tmp/t.jsonl", "bedrock", "us-east-1", "key-id", "secret", "1", "", "8", "1"]
    )
    params = run_build_wizard(console, BuildParams(name="default"), reader=reader)
    assert params.provider == "bedrock"
    assert os.environ["AWS_ACCESS_KEY_ID"] == "key-id"  # session env applied anyway
    assert "not saved" in console.export_text()


def test_build_wizard_defaults_to_first_provider_with_creds(monkeypatch) -> None:  # noqa: ANN001
    # No --provider flag: the suggested default is the first provider (in openai, anthropic,
    # bedrock, azure order) whose creds are present — here anthropic, once openai's key is gone.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    console = Console(force_terminal=False, no_color=True, width=100)
    reader = _scripted_reader(["m", "/tmp/t.jsonl", "", "", "", ""])
    params = run_build_wizard(console, BuildParams(name="default"), reader=reader)
    assert params.provider == "anthropic"
    assert params.model == "claude-opus-4-8"  # model default follows the provider


def test_build_wizard_annotates_providers_that_have_keys(monkeypatch) -> None:  # noqa: ANN001
    # Providers whose creds are present are labeled in the picker; cleared ones are not.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    reader = _scripted_reader(["m", "/tmp/t.jsonl", "openai", "", "", ""])
    run_build_wizard(console, BuildParams(name="default"), reader=reader)
    out = console.export_text()
    assert "openai  (api key exists)" in out
    assert "anthropic  (api key exists)" not in out


# --- selection picker ----------------------------------------------------------------------------


def test_select_model_single_returns_without_prompting() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    info = ModelInfo(name="only", serve_provider="bedrock", serve_model="opus")
    # No reader needed: a single model is returned directly.
    assert select_model(console, [info]) == "only"


def test_select_model_picks_by_number() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    infos = [
        ModelInfo(name="airline", serve_provider="bedrock", serve_model="opus"),
        ModelInfo(name="retail", serve_provider="bedrock", serve_model="opus"),
    ]
    assert select_model(console, infos, reader=_scripted_reader(["2"])) == "retail"


def test_select_model_reprompts_then_accepts_name() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    infos = [
        ModelInfo(name="airline", serve_provider="bedrock", serve_model="opus"),
        ModelInfo(name="retail", serve_provider="bedrock", serve_model="opus"),
    ]
    # First an out-of-range number (re-prompts), then the model name directly.
    chosen = select_model(console, infos, reader=_scripted_reader(["9", "airline"]))
    assert chosen == "airline"


def test_select_model_survives_unicode_digit_input() -> None:
    # '²' (superscript two): str.isdigit() is True but int() raises ValueError. The picker must
    # treat it as invalid and re-prompt, not crash.
    console = Console(force_terminal=False, no_color=True, width=100)
    infos = [
        ModelInfo(name="airline", serve_provider="bedrock", serve_model="opus"),
        ModelInfo(name="retail", serve_provider="bedrock", serve_model="opus"),
    ]
    chosen = select_model(console, infos, reader=_scripted_reader(["²", "1"]))
    assert chosen == "airline"


def test_build_wizard_reprompts_on_unicode_digit_budget() -> None:
    # Same unicode-digit footgun in the int prompt: must re-ask, not crash. With file provided the
    # prompts are name/provider/model/budget/embedder (no region: the creds-default provider is
    # openai); blanks accept defaults, '²' is a bad budget that must be rejected and re-asked.
    console = Console(force_terminal=False, no_color=True, width=100)
    reader = _scripted_reader(["", "", "", "²", "7", ""])
    params = run_build_wizard(console, BuildParams(name="m", file="/tmp/t.jsonl"), reader=reader)
    assert params.gepa_budget == 7
