"""Tests for the terminal UX: the non-TTY build reporter and the play REPL (injected I/O)."""

from __future__ import annotations

import importlib
import io
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
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind, VerifyResult
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder

ui_module = importlib.import_module("wmh.cli.ui")


def _ok_verify(cfg: ProviderConfig) -> VerifyResult:
    """Stub inline-wizard verifier: every provider/embedder pings ok, no network."""
    return VerifyResult(ok=True, kind=cfg.kind, model=cfg.model)


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
        reporter.split_done(2, 1, 1)
        reporter.index_done(9)
        reporter.optimize_start(20)
        reporter.rollout(1, 20, 0.4)
        reporter.rollout(10, 20, 0.6)
        reporter.optimize_done(0.6, 2, 20)
    out = cap.get()
    assert "ingested 3 traces" in out
    assert "normalized 9 steps" in out
    assert "2 train / 1 val / 1 test" in out
    assert "GEPA metric call 1/20" in out  # non-TTY heartbeat
    assert "GEPA metric call 10/20" in out
    assert "val 0.600" in out


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


def test_split_keys_separates_batched_sequences() -> None:
    assert ui_module._split_keys("\x1b[B\x1b[B") == ["\x1b[B", "\x1b[B"]
    assert ui_module._split_keys("\x1b[1;5A") == ["\x1b[1;5A"]
    assert ui_module._split_keys("jk\r") == ["j", "k", "\r"]
    assert ui_module._split_keys("\x1bOA5") == ["\x1bOA", "5"]
    assert ui_module._split_keys("\x1b") == ["\x1b"]


def test_arrow_select_reveals_hidden_rows_on_navigation(monkeypatch) -> None:  # noqa: ANN001
    # Collapsed picker: two rows + a "… N more" row; arrowing down onto it expands in place
    # and the highlight lands on the first revealed option.
    keys = iter(["\x1b[B", "\x1b[B", "\x1b[B", "\r"])  # down to "more", auto-expand, down, Enter
    monkeypatch.setattr(ui_module.click, "getchar", lambda: next(keys))
    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    chosen = ui_module._arrow_select(console, ["a", "b"], 0, ["c", "d"])
    assert chosen == 3  # a -> b -> (more: expands, highlight on c) -> d -> Enter
    out = console.export_text()
    assert "… 2 more" in out  # the collapsed affordance rendered
    assert "\u276f d" in out  # and the final highlight reached a previously hidden row


def test_select_collapsed_keeps_numbered_fallback_complete() -> None:
    # Non-TTY: collapsed is an arrow-picker affordance only; scripted input sees every option.
    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    chosen = ui_module._select(
        console,
        _scripted_reader(["4"]),
        "Pick",
        ["a", "b", "c", "d"],
        "a",
        interactive=False,
        collapsed=2,
    )
    assert chosen == "d"
    assert "4." in console.export_text()  # all four options listed


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
    # Prompts in order: name, trace source (select), file, provider (select), model (select),
    # region (bedrock only), budget, embedder (select). No embed-model (hashing) or phi-dim prompt.
    reader = _scripted_reader(
        [
            "tau2-airline",
            "",  # trace source: accept the default (otel-genai)
            "/tmp/traces.jsonl",
            "bedrock",
            "claude-opus-4-8",
            "us-east-1",
            "",  # judge model: accept the bedrock default (dated haiku)
            "high",
            "hashing",
        ]
    )
    # train_split has no wizard prompt; it must carry through from the flag-supplied defaults.
    params = run_build_wizard(
        console,
        BuildParams(name="default", train_split=0.5),
        reader=reader,
        verify=_ok_verify,
        verify_embed=_ok_verify,
    )
    assert params.name == "tau2-airline"
    assert params.file == "/tmp/traces.jsonl"
    assert params.provider == "bedrock"
    assert params.region == "us-east-1"
    assert params.judge_model == "claude-haiku-4-5"
    assert params.fidelity == "high"
    assert params.embed_provider == "hashing"
    assert params.train_split == 0.5


def test_build_wizard_select_by_number() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    # Provider/model/embedder are numbered pickers; choosing by index must work. Pick anthropic (2),
    # its second model, no region prompt (not bedrock), budget 8, hashing embedder (1).
    reader = _scripted_reader(["m", "", "/tmp/t.jsonl", "2", "2", "", "1", "1"])
    params = run_build_wizard(
        console,
        BuildParams(name="default"),
        reader=reader,
        verify=_ok_verify,
        verify_embed=_ok_verify,
    )
    assert params.provider == "anthropic"
    assert params.model == "claude-opus-4-7"  # second anthropic model
    assert params.judge_model == "claude-haiku-4-5"  # blank accepted the anthropic judge default
    assert params.region is None  # region only prompted for bedrock
    assert params.embed_provider == "hashing"


def test_build_wizard_collects_provider_embedder() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    # A provider-backed embedder adds an embeddings-model picker; phi dim keeps its default.
    reader = _scripted_reader(
        [
            "m",
            "",
            "/tmp/t.jsonl",
            "openai",
            "gpt-5.5",
            "",
            "max",
            "openai",
            "text-embedding-3-large",
        ]
    )
    params = run_build_wizard(
        console,
        BuildParams(name="default"),
        reader=reader,
        verify=_ok_verify,
        verify_embed=_ok_verify,
    )
    assert params.embed_provider == "openai"
    assert params.embed_model == "text-embedding-3-large"
    assert params.embed_dim == 512  # default, no longer prompted


def test_build_wizard_accepts_defaults_with_blank_input() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    # File provided (so that prompt is skipped); press Enter (blank) for every remaining prompt
    # (name/provider/model/region/budget/embedder) to accept the suggested defaults.
    reader = _scripted_reader(["", "", "", "", "", "", ""])
    defaults = BuildParams(
        name="seeded", file="/tmp/t.jsonl", provider="bedrock", fidelity="medium"
    )
    params = run_build_wizard(
        console, defaults, reader=reader, verify=_ok_verify, verify_embed=_ok_verify
    )
    assert params.name == "seeded"  # blank kept the default
    assert params.provider == "bedrock"
    assert params.fidelity == "medium"
    assert params.region == "us-east-1"  # bedrock default suggested + accepted
    assert params.embed_provider == "hashing"  # default embedder, no embed-model prompt


def test_build_reporter_bar_never_pins_at_100_while_running() -> None:
    # The metric-call budget is a soft cap GEPA can overshoot; a live run must never read
    # completed == total (rich freezes the elapsed clock and the bar looks stuck at 100%).
    console = Console(force_terminal=True, no_color=True, width=100, file=io.StringIO())
    reporter = RichBuildReporter(console, "demo")
    reporter.optimize_start(10)
    assert reporter._progress is not None
    task = reporter._progress.tasks[0]

    reporter.rollout(9, 10, 0.5)
    assert (task.completed, task.total) == (9, 10)
    reporter.rollout(10, 10, 0.5)  # reaching the estimate: still not "finished"
    assert task.total is not None and task.completed < task.total
    reporter.rollout(14, 10, 0.6)  # overshoot: the total grows with reality
    assert (task.completed, task.total) == (14, 15)
    reporter.optimize_done(0.6, 1, 14)
    # 100% happens exactly at completion: the bar snaps to the actual final call count.
    assert (task.completed, task.total) == (14, 14)


def test_build_reporter_activity_window_streams_within_fixed_height() -> None:
    console = Console(force_terminal=True, no_color=True, width=100, file=io.StringIO())
    reporter = RichBuildReporter(console, "demo")
    reporter.optimize_start(10)
    for i in range(20):
        reporter.activity(f"Iteration {i}: note")
    # The window keeps only the newest lines (fixed height, no terminal scroll).
    assert len(reporter._activity) == 8
    assert reporter._activity[-1] == "Iteration 19: note"
    assert reporter._activity[0] == "Iteration 12: note"
    reporter.optimize_done(0.5, 1, 12)
    assert reporter._live is None  # display released on completion


def test_build_reporter_activity_lines_are_width_safe() -> None:
    # Judge critiques can contain combining marks / double-width glyphs whose cell width the
    # terminal and rich disagree on; one such line in the live region desyncs every repaint
    # (orphaned frame headers). Lines are reduced to printable ASCII before rendering.
    console = Console(force_terminal=True, no_color=True, width=100, file=io.StringIO())
    reporter = RichBuildReporter(console, "demo")
    reporter.optimize_start(10)
    reporter.activity("matches the \u0935\u093e\u0938\u094d\u0924\u0935\u093f\u0915 obs\te\u0301")
    line = reporter._activity[-1]
    assert all(" " <= ch <= "~" for ch in line)
    assert line.startswith("matches the ")
    reporter.optimize_done(0.5, 1, 10)


def test_build_reporter_activity_is_quiet_when_piped() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    reporter = RichBuildReporter(console, "demo")
    reporter.optimize_start(10)
    with console.capture() as cap:
        reporter.activity("Iteration 1: noisy inner-loop line")
    assert cap.get() == ""  # non-TTY logs keep the sparse heartbeat only


def test_build_wizard_dashes_spaces_in_name() -> None:
    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    # Whitespace is dash-joined rather than rejected: typing "tau bench" quietly becomes
    # "tau-bench", with a dim note showing the name actually used.
    reader = _scripted_reader(["tau bench", "", "/tmp/t.jsonl", "", "", "", "", ""])
    params = run_build_wizard(
        console,
        BuildParams(name="default"),
        reader=reader,
        verify=_ok_verify,
        verify_embed=_ok_verify,
    )
    assert params.name == "tau-bench"
    assert "using tau-bench" in console.export_text()


def test_build_wizard_reprompts_on_invalid_name() -> None:
    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    # A name normalization can't rescue ("tau/bench" has a path separator) must re-prompt with
    # the friendly validation message, not escape as a ValueError traceback.
    reader = _scripted_reader(["tau/bench", "tau-bench", "", "/tmp/t.jsonl", "", "", "", "", ""])
    params = run_build_wizard(
        console,
        BuildParams(name="default"),
        reader=reader,
        verify=_ok_verify,
        verify_embed=_ok_verify,
    )
    assert params.name == "tau-bench"
    assert "invalid world model name" in console.export_text()


def test_build_wizard_normalizes_flag_name_in_default() -> None:
    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    # A whitespace-y --name flag becomes the normalized bracketed default, acceptable via Enter
    # on the first prompt (no validation error, no re-prompt).
    reader = _scripted_reader(["", "", "/tmp/t.jsonl", "", "", "", "", ""])
    params = run_build_wizard(
        console,
        BuildParams(name="tau bench"),
        reader=reader,
        verify=_ok_verify,
        verify_embed=_ok_verify,
    )
    assert params.name == "tau-bench"
    assert "invalid world model name" not in console.export_text()


def test_build_wizard_drops_invalid_flag_name_from_default() -> None:
    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    # An unrescuable --name flag must not become the bracketed Enter-default (it could never be
    # accepted); blank input then means "no name yet" and re-prompts until a valid one arrives.
    reader = _scripted_reader(["", "tau-bench", "", "/tmp/t.jsonl", "", "", "", "", ""])
    params = run_build_wizard(
        console,
        BuildParams(name="tau/bench"),
        reader=reader,
        verify=_ok_verify,
        verify_embed=_ok_verify,
    )
    assert params.name == "tau-bench"
    assert "[tau/bench]" not in console.export_text()


def test_build_wizard_aborts_cleanly_on_eof() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)

    def eof_reader(_prompt: str) -> str:
        raise EOFError

    # Exhausted piped stdin (or Ctrl-D) must abort the wizard cleanly, not leak an EOFError
    # traceback. typer.Abort is handled by click's runner ("Aborted.", exit 1).
    with pytest.raises(typer.Abort):
        run_build_wizard(console, BuildParams(name="default"), reader=eof_reader, verify=_ok_verify)


def test_build_wizard_reprompts_on_blank_trace_source() -> None:
    # A blank traces path must re-ask, not crash. After a name and the trace-source select, blank
    # the file once, then give a real path; remaining prompts (provider/model/region/budget/
    # embedder) take defaults.
    console = Console(force_terminal=False, no_color=True, width=100)
    reader = _scripted_reader(["mymodel", "", "", "/tmp/t.jsonl", "", "", "", "", ""])
    params = run_build_wizard(
        console,
        BuildParams(name="default"),
        reader=reader,
        verify=_ok_verify,
        verify_embed=_ok_verify,
    )
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
        # name, trace source, file, provider, AWS_REGION, AWS_ACCESS_KEY_ID,
        # AWS_SECRET_ACCESS_KEY (skip), model, region, judge, fidelity, embedder
        ["m", "", "/tmp/t.jsonl", "bedrock", "us-east-1", "test-key-id", "", "1", "", "", "", "1"]
    )
    params = run_build_wizard(
        console,
        BuildParams(name="default"),
        reader=reader,
        verify=_ok_verify,
        verify_embed=_ok_verify,
    )
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
        ["m", "", "/tmp/t.jsonl", "bedrock", "us-east-1", "key-id", "secret", "1", "", "", "", "1"]
    )
    params = run_build_wizard(
        console,
        BuildParams(name="default"),
        reader=reader,
        verify=_ok_verify,
        verify_embed=_ok_verify,
    )
    assert params.provider == "bedrock"
    assert os.environ["AWS_ACCESS_KEY_ID"] == "key-id"  # session env applied anyway
    assert "not saved" in console.export_text()


def test_build_wizard_verifies_provider_and_retries_on_failure() -> None:
    # The live ping runs right after the model id (and region) is chosen; a failure loops back
    # to the provider picker instead of surfacing at the end of the wizard.
    calls: list[str] = []

    def flaky(cfg: ProviderConfig) -> VerifyResult:
        calls.append(cfg.kind.value)
        ok = len(calls) > 1
        return VerifyResult(ok=ok, kind=cfg.kind, model=cfg.model, detail="" if ok else "bad key")

    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    reader = _scripted_reader(["m", "", "/tmp/t.jsonl", "openai", "", "anthropic", "", "", "", ""])
    params = run_build_wizard(
        console, BuildParams(name="default"), reader=reader, verify=flaky, verify_embed=_ok_verify
    )
    assert params.provider == "anthropic"
    # openai fails, anthropic serve verifies, then the judge default (haiku) verifies too.
    assert calls == ["openai", "anthropic", "anthropic"]
    assert "bad key" in console.export_text()


def test_build_wizard_verifies_embedder_with_embed_config() -> None:
    # A provider-backed embedder is pinged right after its model pick, with the embeddings model
    # and phi dimension stamped on the config (mirroring HarnessConfig.embed_provider_config).
    seen: dict[str, ProviderConfig] = {}

    def embed_check(cfg: ProviderConfig) -> VerifyResult:
        seen["cfg"] = cfg
        return VerifyResult(ok=True, kind=cfg.kind, model=cfg.embed_model or cfg.model)

    console = Console(force_terminal=False, no_color=True, width=100)
    reader = _scripted_reader(
        ["m", "", "/tmp/t.jsonl", "openai", "", "", "", "openai", "text-embedding-3-large"]
    )
    run_build_wizard(
        console,
        BuildParams(name="default"),
        reader=reader,
        verify=_ok_verify,
        verify_embed=embed_check,
    )
    assert seen["cfg"].kind is ProviderKind.OPENAI
    assert seen["cfg"].embed_model == "text-embedding-3-large"
    assert seen["cfg"].embed_dim == 512


def test_build_wizard_defaults_to_first_provider_with_creds(monkeypatch) -> None:  # noqa: ANN001
    # No --provider flag: the suggested default is the first provider (in openai, anthropic,
    # bedrock, azure order) whose creds are present — here anthropic, once openai's key is gone.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    console = Console(force_terminal=False, no_color=True, width=100)
    reader = _scripted_reader(["m", "", "/tmp/t.jsonl", "", "", "", "", ""])
    params = run_build_wizard(
        console,
        BuildParams(name="default"),
        reader=reader,
        verify=_ok_verify,
        verify_embed=_ok_verify,
    )
    assert params.provider == "anthropic"
    assert params.model == "claude-opus-4-8"  # model default follows the provider


def test_build_wizard_annotates_providers_that_have_keys(monkeypatch) -> None:  # noqa: ANN001
    # Providers whose creds are present are labeled in the picker; cleared ones are not.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    reader = _scripted_reader(["m", "", "/tmp/t.jsonl", "openai", "", "", "", ""])
    run_build_wizard(
        console,
        BuildParams(name="default"),
        reader=reader,
        verify=_ok_verify,
        verify_embed=_ok_verify,
    )
    out = console.export_text()
    assert "openai  (OPENAI_API_KEY set)" in out  # names the variable: traceable to zshrc/.env
    assert "bedrock  (creds set)" in out  # multi-var providers get the generic label
    assert "anthropic" in out and "ANTHROPIC_API_KEY set" not in out


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


def test_play_repl_shows_sampled_action_suggestions() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    with console.capture() as cap:
        run_play_repl(
            console,
            _world_model(),
            "airline",
            task=None,
            reader=_scripted_reader([":quit"]),
            suggestions=['get_user {"id": "u1"}', "list_flights"],
        )
    out = cap.get()
    assert "Real actions from this model's traces" in out
    assert 'get_user {"id": "u1"}' in out


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


def test_build_wizard_high_fidelity_keeps_hashing_default() -> None:
    # Lexical hashing is the measured-best phi at EVERY tier (semantic lost on all benchmarks —
    # PR #72's matrix + the tier ladder's tau decline), so high/max no longer nudge toward
    # provider embeddings: a blank at the embedder prompt stays hashing, no model picker follows.
    console = Console(force_terminal=False, no_color=True, width=100, record=True)
    reader = _scripted_reader(["m", "", "/tmp/t.jsonl", "bedrock", "", "", "", "high", ""])
    params = run_build_wizard(
        console,
        BuildParams(name="default"),
        reader=reader,
        verify=_ok_verify,
        verify_embed=_ok_verify,
    )
    assert params.fidelity == "high"
    assert params.embed_provider == "hashing"
    assert params.embed_model is None
    assert "semantic embeddings recommended" not in console.export_text()


def test_build_wizard_low_fidelity_keeps_hashing_default() -> None:
    console = Console(force_terminal=False, no_color=True, width=100)
    reader = _scripted_reader(["m", "", "/tmp/t.jsonl", "bedrock", "", "", "", "low", ""])
    params = run_build_wizard(
        console,
        BuildParams(name="default"),
        reader=reader,
        verify=_ok_verify,
        verify_embed=_ok_verify,
    )
    assert params.fidelity == "low"
    assert params.embed_provider == "hashing"


def test_build_wizard_reprompts_on_unicode_digit_fidelity() -> None:
    # Same unicode-digit footgun in the fidelity picker: must re-ask, not crash. '²' is a bad
    # pick that must be rejected and re-asked; '1' then selects low.
    console = Console(force_terminal=False, no_color=True, width=100)
    reader = _scripted_reader(["", "", "", "", "²", "1", ""])
    params = run_build_wizard(
        console,
        BuildParams(name="m", file="/tmp/t.jsonl"),
        reader=reader,
        verify=_ok_verify,
        verify_embed=_ok_verify,
    )
    assert params.fidelity == "low"
