"""CLI surface tests for explicit manual judge configuration."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from click import unstyle
from rich.console import Console
from rich.text import Text
from typer.testing import CliRunner

from wmo.cli import judge_config as judge_config_module
from wmo.cli.app import app
from wmo.cli.judge_config import _label_value
from wmo.common.core.artifacts import FailureCode, SourceIdentity, StructuredFailure
from wmo.common.judging import Rubric, default_task_success_axis
from wmo.common.models import (
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelSnapshot,
    write_model_catalog,
)
from wmo.common.traces import Trace, TraceOutcome, TraceSource, TraceSpan
from wmo.optimize.router.judging.contracts import (
    JudgeCalibrationBudget,
    JudgeTracePreview,
    ManualJudgeLabel,
    ManualJudgeSetupArtifact,
)
from wmo.optimize.router.judging.service import (
    ManualJudgeCalibrationPlan,
    ManualJudgeSetupPlan,
    default_judge_dimensions,
)
from wmo.optimize.router.judging.service_test import _built_store, _catalog, _setup
from wmo.runtime.models.providers.errors import ProviderError


class _PairwiseAnswer:
    """Return candidate B while retaining the exact human-facing prompt."""

    prompt = ""

    @classmethod
    def ask(cls, prompt: str, *, choices: list[str]) -> str:
        """Record one prompt and require plain A, B, and tie choices."""
        cls.prompt = prompt
        assert choices == ["A", "B", "tie"]
        return "B"


class _ScalarAnswer:
    """Return the default-axis success score while retaining the prompt."""

    prompt = ""

    @classmethod
    def ask(cls, prompt: str) -> int:
        """Record one prompt and return a valid scalar score."""
        cls.prompt = prompt
        return 1


@pytest.mark.parametrize(
    "arguments",
    [
        ["config", "judge", "setup", "support"],
        ["config", "judge", "calibrate", "support"],
    ],
)
def test_malformed_release_revision_fails_before_judge_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    """Reject malformed producer identity before judge project or artifact writes.

    Args:
        tmp_path: Pytest-owned state directory.
        monkeypatch: Scoped invalid release revision override.
        arguments: Exact nested judge command invocation.
    """
    root = tmp_path / ".wmo"
    monkeypatch.setenv("WMO_RELEASE_REVISION", "HEAD")

    result = CliRunner().invoke(app, [*arguments, "--root", str(root)])

    assert result.exit_code == 2
    assert "full lowercase 40-hex" in result.output
    assert not root.exists()


def test_judge_commands_render_as_nested_config_commands() -> None:
    """Both manual judge stages are discoverable without expanding the root surface."""
    runner = CliRunner()

    setup = runner.invoke(app, ["config", "judge", "setup", "--help"])
    calibrate = runner.invoke(app, ["config", "judge", "calibrate", "--help"])

    assert setup.exit_code == 0, setup.output
    assert calibrate.exit_code == 0, calibrate.output
    setup_output = unstyle(setup.output)
    calibrate_output = unstyle(calibrate.output)
    assert "--approve" in setup_output
    assert "zero-to-five" not in setup_output
    assert "rubric axes" in setup_output
    assert "--yes" in calibrate_output
    assert "--approve" in calibrate_output
    assert "--debug" in calibrate_output
    assert "--page" in calibrate_output
    assert "Advanced" in calibrate_output
    assert "--input-usd-per-million" in calibrate_output
    assert "--output-usd-per-million" in calibrate_output
    assert "--maximum-cost-usd" in calibrate_output
    assert "command-budget" in calibrate_output


def _write_catalog(root: Path, catalog: ModelCatalog) -> None:
    """Persist one secret-free catalog at the project root."""
    write_model_catalog(root / "models.toml", catalog)


def _priced_catalog() -> ModelCatalog:
    """Return the fixture catalog with explicit persisted judge prices."""
    catalog = _catalog()
    record = catalog.models["judge-main"]
    return catalog.model_copy(
        update={
            "models": {
                **catalog.models,
                "judge-main": ModelRecord(
                    connection=record.connection,
                    model=record.model,
                    capabilities=ModelCapabilities(
                        input_cost_per_million_tokens_usd=1.0,
                        output_cost_per_million_tokens_usd=2.0,
                    ),
                ),
            }
        }
    )


def test_calibrate_prints_catalog_pricing_breakdown_before_labels(tmp_path: Path) -> None:
    """Known catalog prices produce a spend preflight before consent or labels."""
    store = _built_store(tmp_path)
    _setup(store)
    _write_catalog(store.paths.root, _priced_catalog())

    result = CliRunner().invoke(
        app,
        [
            "config",
            "judge",
            "calibrate",
            "support",
            "--root",
            str(store.paths.root),
            "--sample-size",
            "3",
            "--non-interactive",
        ],
    )

    output = unstyle(result.output)
    assert result.exit_code == 2
    assert "Judge name: judge-main" in output
    assert "Exact model: openai/judge-model" in output
    assert "Pricing: configured" in output
    assert "Judge calls authorized: 3" in output
    assert "Tokens: up to 32768 input and 4096 output per attempt" in output
    assert "Maximum estimated cost:" in output
    assert "Hard spend ceiling: $10.0000" in output
    assert "missing labels" not in output


def test_calibrate_fails_closed_when_catalog_pricing_is_missing(tmp_path: Path) -> None:
    """Missing catalog and known-model prices fail before labels or credentials."""
    store = _built_store(tmp_path)
    _setup(store)
    _write_catalog(store.paths.root, _catalog())

    result = CliRunner().invoke(
        app,
        [
            "config",
            "judge",
            "calibrate",
            "support",
            "--root",
            str(store.paths.root),
            "--non-interactive",
        ],
    )

    output = unstyle(result.output)
    assert result.exit_code == 2
    assert "no trustworthy input/output" in output
    assert "missing labels" not in output


def test_calibrate_uses_shared_command_budget_when_flag_is_omitted(tmp_path: Path) -> None:
    """The shared command-budget setting becomes the calibration ceiling."""
    store = _built_store(tmp_path)
    _setup(store)
    root = store.paths.root
    _write_catalog(root, _priced_catalog())
    (root / "settings.toml").write_text(
        "[commands]\nmaximum_cost_usd = 0.000001\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "config",
            "judge",
            "calibrate",
            "support",
            "--root",
            str(root),
            "--sample-size",
            "3",
            "--non-interactive",
        ],
    )

    output = unstyle(result.output)
    assert result.exit_code == 2
    assert "exceeds --maximum-cost-usd" in output
    assert "missing labels" not in output


def test_calibrate_renders_provider_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A typed provider rejection exits with saved-label progress and no traceback.

    Args:
        monkeypatch: Scoped replacements for the calibration command's project work.
        tmp_path: Isolated local project root passed through the retry command.
    """
    secret = "sk-secret-live-key-1234567890"
    labels = (ManualJudgeLabel(trace_id="trace-1", dimension_id="task-success", score=1),)
    budget = JudgeCalibrationBudget(
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        maximum_input_tokens_per_call=4096,
        maximum_attempts_per_call=3,
        call_count=1,
        estimated_cost_usd=0.1,
        maximum_cost_usd=5.0,
    )
    error = ProviderError(
        f"Unsupported parameter: 'temperature' is not supported. Authorization: Bearer {secret}",
        provider="openai",
        endpoint_class="responses",
        status_code=400,
        error_code="unsupported_parameter",
        error_type="invalid_request_error",
        rejected_parameter="temperature",
        request_id="req_safe_1",
    )

    monkeypatch.setattr(judge_config_module, "installed_release_revision", lambda: "a" * 40)
    monkeypatch.setattr(
        judge_config_module,
        "ProjectStore",
        lambda root, project: SimpleNamespace(model_catalog_path=tmp_path / "models.toml"),
    )
    monkeypatch.setattr(
        judge_config_module,
        "prepare_manual_judge_calibration",
        lambda store, sample_size: SimpleNamespace(setup=object(), previews=()),
    )
    monkeypatch.setattr(judge_config_module, "_load_setup_rubric", lambda store, setup: object())
    monkeypatch.setattr(judge_config_module, "render_rubric_table", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        judge_config_module, "_render_spend_preflight", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        judge_config_module, "_render_calibration_review", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(judge_config_module, "calibration_sample", lambda plan: ())
    monkeypatch.setattr(
        judge_config_module, "calibration_sample_digest", lambda setup, sample: "b" * 64
    )
    monkeypatch.setattr(judge_config_module, "read_label_draft", lambda *args: labels)
    monkeypatch.setattr(
        judge_config_module, "manual_judge_calibration_is_complete", lambda store: False
    )
    monkeypatch.setattr(
        judge_config_module,
        "_collect_labels",
        lambda *args, **kwargs: labels,
    )
    monkeypatch.setattr(
        judge_config_module,
        "estimate_manual_judge_budget",
        lambda *args, **kwargs: budget,
    )
    monkeypatch.setattr(judge_config_module, "require_spend_consent", lambda *args, **kwargs: True)
    monkeypatch.setattr(judge_config_module, "load_model_catalog", lambda path: object())
    monkeypatch.setattr(judge_config_module, "RuntimeModelCatalog", lambda catalog: object())

    def fail_calibrate(*args: object, **kwargs: object) -> None:
        """Raise the typed provider rejection after labels are already durable."""
        raise error

    monkeypatch.setattr(judge_config_module, "calibrate_manual_judge", fail_calibrate)

    root = tmp_path / ".wmo"
    result = CliRunner().invoke(
        app,
        [
            "config",
            "judge",
            "calibrate",
            "support",
            "--root",
            str(root),
            "--input-usd-per-million",
            "1.0",
            "--output-usd-per-million",
            "2.0",
            "--yes",
        ],
    )
    printed = " ".join(unstyle(result.output).split())

    assert result.exit_code == 1
    assert "Provider call failed" in printed
    assert "openai responses HTTP 400" in printed
    assert "unsupported_parameter" in printed
    assert "rejected parameter: temperature" in printed
    assert "1 human labels saved" in printed
    assert "the failed provider attempt was not recorded" in printed
    assert f"wmo config judge calibrate support --root {root}" in printed
    assert "--yes" in printed
    assert "Traceback" not in printed
    assert secret not in printed
    assert "Authorization" not in printed
    assert isinstance(result.exception, SystemExit)


def test_scalar_label_values_follow_the_axis_range() -> None:
    """CLI label parsing accepts 0-1 default scores and rejects out-of-range values."""
    axis = default_task_success_axis()

    assert _label_value("trace-1:task-success=1", pairwise=False, axis=axis) == 1
    with pytest.raises(ValueError, match="from 0 through 1"):
        _label_value("trace-1:task-success=4", pairwise=False, axis=axis)


def test_setup_output_is_plain_language_and_hides_execution_internals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default setup shows identity, full anchors, and tasks without prompt machinery."""
    buffer = io.StringIO()
    monkeypatch.setattr(
        judge_config_module,
        "_console",
        Console(file=buffer, width=42, color_system=None),
    )
    plan = cast(
        ManualJudgeSetupPlan,
        SimpleNamespace(
            judge_alias="review-judge",
            judge_model=_model(),
            prompt_template=SimpleNamespace(response_shape="scalar"),
            dimensions=default_judge_dimensions(),
            previews=(
                SimpleNamespace(task="Summarize the customer issue.", outcome="success"),
                SimpleNamespace(task="Find the failed command.", outcome="failure"),
            ),
        ),
    )

    judge_config_module._render_setup(plan)

    output = buffer.getvalue()
    compact = " ".join(output.split())
    assert "Judge name: review-judge" in output
    assert "Exact model: openai/judge-model" in output
    assert "Integer scoring from 0 to 1" in compact
    assert "Task success" in output
    assert "Range: 0-1" in output
    assert "did not complete the requested task" in compact
    assert "Summarize the customer issue." in output
    assert "Find the failed command." in output
    assert "Prompt:" not in output
    assert "Variable mapping" not in output
    assert "response schema" not in output.lower()
    assert "Score projection" not in output
    assert "0000000000000000" not in output


def test_spend_preflight_preserves_a_small_positive_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A small admitted estimate and ceiling never display as zero."""
    buffer = io.StringIO()
    monkeypatch.setattr(
        judge_config_module,
        "_console",
        Console(file=buffer, width=80, color_system=None),
    )
    plan = cast(
        ManualJudgeCalibrationPlan,
        SimpleNamespace(setup=SimpleNamespace(judge_alias="judge", judge_model=_model())),
    )
    budget = JudgeCalibrationBudget(
        input_usd_per_million_tokens=0,
        output_usd_per_million_tokens=0,
        maximum_input_tokens_per_call=1,
        maximum_attempts_per_call=1,
        call_count=1,
        estimated_cost_usd=0.000049,
        maximum_cost_usd=0.000049,
    )

    judge_config_module._render_spend_preflight(plan, budget)

    output = buffer.getvalue()
    assert "Maximum estimated cost: $0.000049" in output
    assert "Hard spend ceiling: $0.000049" in output
    assert "$0.0000\n" not in output


def test_pairwise_calibration_renders_roles_and_truthful_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A/B review separates task, assistant, tools, result, and terminal failure."""
    buffer = io.StringIO()
    monkeypatch.setattr(
        judge_config_module,
        "_console",
        Console(file=buffer, width=36, color_system=None),
    )
    first = _trace("trace-a", completion="A" * 80, failed=True)
    second = _trace("trace-b", completion="Candidate B finished.", failed=False)
    plan = cast(
        ManualJudgeCalibrationPlan,
        SimpleNamespace(
            setup=SimpleNamespace(prompt_template=SimpleNamespace(response_shape="pairwise")),
            traces=(first,),
            reference_traces=(second,),
        ),
    )

    judge_config_module._render_calibration_review(
        plan,
        cast(Rubric, SimpleNamespace(dimensions=default_judge_dimensions())),
        character_limit=40,
        page=False,
    )

    output = buffer.getvalue()
    assert "PAIRWISE A/B CALIBRATION" in output
    assert "Pair 1, candidate A" in output
    assert "Pair 1, candidate B" in output
    assert "User / task:" in output
    assert "User message:" in output
    assert "Please resolve customer issue trace-a." in " ".join(output.split())
    assert "Assistant / model:" in output
    assert "Assistant output:" in output
    assert "Tool call:" in output
    assert "Tool arguments:" in output
    assert "Tool result:" in output
    assert "Tool output:" in output
    assert "Final outcome:" in output
    assert "Final failure:" in output
    assert "truncated 40 characters" in output
    assert "use --page for the full transcript" in " ".join(output.split())


def test_pairwise_prompt_keeps_anchors_adjacent_and_uses_plain_a_b_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interactive pairwise scoring says A and B while persisting the typed winner."""
    buffer = io.StringIO()
    monkeypatch.setattr(
        judge_config_module,
        "_console",
        Console(file=buffer, width=80, color_system=None),
    )
    monkeypatch.setattr(judge_config_module, "Prompt", _PairwiseAnswer)
    setup = cast(
        ManualJudgeSetupArtifact,
        SimpleNamespace(prompt_template=SimpleNamespace(response_shape="pairwise")),
    )
    preview = cast(
        JudgeTracePreview,
        SimpleNamespace(trace_id="trace-a", reference_trace_id="trace-b"),
    )
    rubric = cast(Rubric, SimpleNamespace(dimensions=default_judge_dimensions()))
    persisted: list[tuple[ManualJudgeLabel, ...]] = []

    def persist(labels: tuple[ManualJudgeLabel, ...]) -> None:
        """Retain incremental labels exactly as the command would save them."""
        persisted.append(labels)

    labels = judge_config_module._collect_labels(
        setup,
        rubric,
        (),
        (preview,),
        (),
        persist,
        non_interactive=False,
    )

    assert labels[0].winner == "winner_b"
    assert persisted[-1] == labels
    assert "choose candidate A, candidate B, or tie" in _PairwiseAnswer.prompt
    output = buffer.getvalue()
    assert "Score prompt: Task success" in output
    assert "0: The agent did not complete the requested task." in output
    assert "1: The agent successfully completed the requested task." in output


@pytest.mark.parametrize(
    ("shape", "name"),
    [
        ("pairwise", "[/]"),
        ("scalar", "[link=https://invalid.example]linked name[/link]"),
    ],
)
def test_score_prompt_escapes_user_authored_rich_markup(
    monkeypatch: pytest.MonkeyPatch,
    shape: str,
    name: str,
) -> None:
    """Malformed and link-like dimension names remain literal scoring text.

    Args:
        monkeypatch: Scoped prompt replacement.
        shape: Prompt response shape under test.
        name: Valid user-authored dimension name containing Rich markup syntax.
    """
    dimension = default_judge_dimensions()[0].model_copy(update={"name": name})
    setup = cast(
        ManualJudgeSetupArtifact,
        SimpleNamespace(prompt_template=SimpleNamespace(response_shape=shape)),
    )
    preview = cast(
        JudgeTracePreview,
        SimpleNamespace(
            trace_id="trace-a",
            reference_trace_id="trace-b" if shape == "pairwise" else None,
        ),
    )
    rubric = cast(Rubric, SimpleNamespace(dimensions=(dimension,)))
    persisted: list[tuple[ManualJudgeLabel, ...]] = []

    def persist(labels: tuple[ManualJudgeLabel, ...]) -> None:
        """Retain the escaped-prompt test's incremental label."""
        persisted.append(labels)

    if shape == "pairwise":
        monkeypatch.setattr(judge_config_module, "Prompt", _PairwiseAnswer)
        answer_type = _PairwiseAnswer
    else:
        monkeypatch.setattr(judge_config_module, "IntPrompt", _ScalarAnswer)
        answer_type = _ScalarAnswer

    labels = judge_config_module._collect_labels(
        setup,
        rubric,
        (),
        (preview,),
        (),
        persist,
        non_interactive=False,
    )

    assert persisted[-1] == labels
    assert name in Text.from_markup(answer_type.prompt).plain


def _model() -> ModelSnapshot:
    """Return one exact secret-free judge identity."""
    return ModelSnapshot(
        provider="openai",
        model_id="judge-model",
        revision=None,
        capabilities_sha256="0" * 64,
        connection_sha256="1" * 64,
    )


def _trace(trace_id: str, *, completion: str, failed: bool) -> Trace:
    """Return one transcript fixture with assistant and paired tool evidence.

    Args:
        trace_id: Stable fixture trace identity.
        completion: Captured assistant response.
        failed: Whether terminal evidence records a failure.

    Returns:
        Complete normalized trace for CLI rendering.
    """
    started = datetime(2026, 8, 16, tzinfo=UTC)
    spans = (
        TraceSpan(
            span_id=f"{trace_id}-assistant",
            name="agent.model_call",
            started_at=started,
            ended_at=started + timedelta(seconds=1),
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.input.messages": (
                    f'[{{"role":"user","content":"Please resolve customer issue {trace_id}."}}]'
                ),
                "gen_ai.output.messages": [
                    {"role": "assistant", "content": [{"type": "text", "text": completion}]}
                ],
            },
            model=_model(),
        ),
        TraceSpan(
            span_id=f"{trace_id}-tool-call",
            name="agent.model_call",
            started_at=started + timedelta(seconds=2),
            ended_at=started + timedelta(seconds=3),
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.tool.name": "search",
                "gen_ai.tool.call.arguments": '{"query":"customer issue"}',
            },
            model=_model(),
        ),
        TraceSpan(
            span_id=f"{trace_id}-tool-result",
            name="agent.tool_call",
            started_at=started + timedelta(seconds=4),
            ended_at=started + timedelta(seconds=5),
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "search",
                "gen_ai.tool.output": "Found the relevant account record.",
            },
        ),
    )
    failure = StructuredFailure(code=FailureCode.INTERNAL, message="Customer request failed")
    return Trace(
        trace_id=trace_id,
        task="Resolve the customer's support request.",
        initial_context={"account_tier": "business"},
        spans=spans,
        outcome=(
            TraceOutcome(status="failure", failure=failure)
            if failed
            else TraceOutcome(status="success", outcome_name="resolved")
        ),
        source=TraceSource(
            identity=SourceIdentity(kind="manual", source_id=trace_id, sha256="2" * 64),
            semantic_convention_version="test-v1",
        ),
    )
