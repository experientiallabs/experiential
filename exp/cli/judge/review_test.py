"""Tests for one-trace configured-judge proposal review."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast

import pytest
from rich.console import Console
from rich.text import Text

from exp.cli.judge import review as review_module
from exp.common.core.artifacts import FailureCode, SourceIdentity, StructuredFailure
from exp.common.judging import Rubric, RubricDimension
from exp.common.judging.judgment import Judgment
from exp.common.models import BillingSource, ModelSnapshot
from exp.common.traces import Trace, TraceOutcome, TraceSource, TraceSpan
from exp.optimize.router.judging.contracts import (
    JudgeTracePreview,
    ManualJudgeSetupArtifact,
)
from exp.optimize.router.judging.review import ManualJudgeTraceProposal
from exp.optimize.router.judging.service import default_judge_dimensions


class _ConfirmAnswers:
    """Return queued accept-or-correct choices and retain pre-prompt output."""

    answers: list[bool] = []
    output_before_prompt: list[str] = []
    prompts: list[str] = []

    @classmethod
    def ask(
        cls,
        prompt: str,
        *,
        default: bool,
        console: Console,
    ) -> bool:
        """Return the next queued answer after recording visible proposal context.

        Args:
            prompt: Rich-compatible accept-or-correct question.
            default: Expected default acceptance choice.
            console: Review console whose prior output must contain every proposal.

        Returns:
            Next queued explicit human choice.
        """
        assert default is True
        file = console.file
        assert isinstance(file, io.StringIO)
        cls.output_before_prompt.append(file.getvalue())
        cls.prompts.append(prompt)
        return cls.answers.pop(0)


class _CorrectedScore:
    """Supply one corrected rubric score."""

    @classmethod
    def ask(cls, _prompt: str, *, console: Console) -> int:
        """Return score zero through the expected review console.

        Args:
            _prompt: Corrected-score prompt, unused by this deterministic fixture.
            console: Review console supplied to the Rich prompt.

        Returns:
            Corrected score zero.
        """
        assert isinstance(console, Console)
        return 0


class _CorrectedJudgment:
    """Supply one human-authored corrected judgment."""

    @classmethod
    def ask(cls, _prompt: str, *, console: Console) -> str:
        """Return explicit correction text through the expected review console.

        Args:
            _prompt: Corrected-judgment prompt, unused by this fixture.
            console: Review console supplied to the Rich prompt.

        Returns:
            Deterministic human-authored correction text.
        """
        assert isinstance(console, Console)
        return "The tool result did not establish that the customer request was resolved."


class _PagingConsole(Console):
    """Record one pager expansion while retaining rendered content in memory."""

    pager_count: int

    def __init__(self, buffer: io.StringIO) -> None:
        """Initialize a deterministic non-terminal console around ``buffer``.

        Args:
            buffer: In-memory destination retaining pager output.
        """
        super().__init__(file=buffer, width=88, color_system=None)
        self.pager_count = 0

    @contextmanager
    def pager(self, styles: bool = False, links: bool = False) -> Iterator[None]:
        """Record pager use without starting an external terminal process.

        Args:
            styles: Whether Rich styles are retained in pager output.
            links: Whether terminal hyperlinks are retained.

        Yields:
            Control to the current-trace renderer.
        """
        assert styles is True
        assert links is False
        self.pager_count += 1
        yield


class _RubricView:
    """Expose the ordered-axis methods used by the review adapter."""

    def __init__(self, dimensions: tuple[RubricDimension, ...]) -> None:
        """Retain deterministic axes under one fixture revision.

        Args:
            dimensions: Ordered rubric axes used by the proposal.
        """
        self.rubric_id = "rubric-revision-1"
        self.dimensions = dimensions

    def axis(self, dimension_id: str) -> RubricDimension:
        """Return the fixture axis with ``dimension_id``.

        Args:
            dimension_id: Stable axis identity.

        Returns:
            Matching fixture axis.

        Raises:
            ValueError: The fixture has no matching axis.
        """
        for dimension in self.dimensions:
            if dimension.dimension_id == dimension_id:
                return dimension
        raise ValueError(f"fixture rubric has no axis {dimension_id}")


def test_one_trace_view_separates_conversation_and_truthfully_truncates() -> None:
    """The current A/B trace alone shows role, tool, final, rubric, and cited evidence fields."""
    buffer = io.StringIO()
    console = Console(file=buffer, width=48, color_system=None)
    first = _trace("trace-a", completion="A" * 80, failed=True)
    second = _trace("trace-b", completion="Candidate B finished.", failed=False)
    proposal = _proposal(first, reference=second, total=5, multi_axis=True)

    review_module.render_trace_proposal(
        proposal,
        character_limit=40,
        page=False,
        non_interactive=False,
        console=console,
    )

    output = buffer.getvalue()
    flattened = " ".join(output.split())
    assert "Trace 1 of 5" in output
    assert "Trace 2 of 5" not in output
    assert "Candidate A" in output
    assert "Candidate B" in output
    assert "Original user request:" in output
    assert "User message:" in output
    assert "Assistant message:" in output
    assert "Final response:" in output
    assert "Tool call:" in output
    assert "Tool arguments:" in output
    assert "Tool result:" in output
    assert "Final outcome:" in output
    assert "Final failure:" in output
    assert "Axis 1 of 2: Task success" in output
    assert "Description: The agent successfully completed the task requested" in flattened
    assert "Numeric range: 0 to 1" in output
    assert "0: The agent did not complete the requested task." in flattened
    assert "1: The agent successfully completed the requested task." in flattened
    assert "Proposed score: 1" in output
    assert "Proposed judgment:" in output
    assert "Cited trace evidence:" in output
    assert "trace-a-assistant" in output
    assert "Cited reference trace evidence:" in output
    assert "trace-b-assistant" in output
    assert "Evidence content:" in output
    assert "truncated 40 characters" in output
    assert "use --page for the full transcript" in flattened


@pytest.mark.parametrize(
    ("page", "non_interactive", "expect_viewer"),
    [
        (False, False, True),
        (True, False, False),
        (False, True, False),
    ],
)
def test_full_screen_viewer_runs_only_for_interactive_unpaged_reviews(
    monkeypatch: pytest.MonkeyPatch,
    page: bool,
    non_interactive: bool,
    expect_viewer: bool,
) -> None:
    """The viewer never intercepts paged or fully flag-driven reviews on a TTY.

    Args:
        monkeypatch: Scoped TTY detection and viewer replacement.
        page: Whether the explicit pager mode is selected.
        non_interactive: Whether every decision comes from explicit flags.
        expect_viewer: Whether the full-screen viewer must own the display.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, color_system=None, force_terminal=True)
    monkeypatch.setattr(
        review_module,
        "_interactive_viewer_available",
        lambda _console: True,
    )
    viewed: list[str] = []
    monkeypatch.setattr(
        review_module,
        "view_trace_proposal",
        lambda proposal, *, console: viewed.append(proposal.trace.trace_id),
    )
    proposal = _proposal(_trace("trace-a", completion="Done.", failed=False))

    review_module.render_trace_proposal(
        proposal,
        character_limit=1_200,
        page=page,
        non_interactive=non_interactive,
        console=console,
    )

    assert viewed == (["trace-a"] if expect_viewer else [])
    if not expect_viewer and not page:
        assert "Original user request:" in buffer.getvalue()


def test_acceptance_prompts_only_after_every_axis_proposal_is_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All axis descriptions, anchors, scores, judgments, and citations precede human input.

    Args:
        monkeypatch: Scoped confirmation prompt replacement.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, color_system=None)
    proposal = _proposal(_trace("trace-a", completion="Done.", failed=False), multi_axis=True)
    _ConfirmAnswers.answers = [True, True]
    _ConfirmAnswers.output_before_prompt = []
    _ConfirmAnswers.prompts = []
    monkeypatch.setattr(review_module, "Confirm", _ConfirmAnswers)
    reviewer = review_module.build_manual_judge_reviewer(
        _setup(),
        proposal.rubric,
        _previews(proposal),
        drafted_labels=(),
        supplied_labels=(),
        supplied_judgments=(),
        non_interactive=False,
        character_limit=1_200,
        page=False,
        console=console,
    )

    decisions = reviewer(proposal)

    assert tuple(item.accepted for item in decisions) == (True, True)
    assert all(item.correction is None for item in decisions)
    assert len(_ConfirmAnswers.output_before_prompt) == 2
    assert "Axis 2 of 2: Policy compliance" in _ConfirmAnswers.output_before_prompt[0]
    assert "Proposed judgment: The response followed the policy." in " ".join(
        _ConfirmAnswers.output_before_prompt[0].split()
    )


def test_axis_markup_is_rendered_and_prompted_as_literal_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User-authored rubric markup cannot alter proposal output or prompt structure.

    Args:
        monkeypatch: Scoped confirmation prompt replacement.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, color_system=None)
    base = _proposal(_trace("trace-a", completion="Done.", failed=False))
    axis_id = "task-success"
    axis_name = "[link=https://invalid.example]literal axis[/link] [/]"
    dimension = base.rubric.dimensions[0].model_copy(
        update={"dimension_id": axis_id, "name": axis_name}
    )
    proposed = base.judgment.dimensions[0]
    proposal = ManualJudgeTraceProposal(
        position=base.position,
        total=base.total,
        trace=base.trace,
        reference_trace=base.reference_trace,
        rubric=cast(Rubric, _RubricView((dimension,))),
        judgment=cast(
            Judgment,
            SimpleNamespace(
                dimensions=(
                    SimpleNamespace(
                        dimension_id=axis_id,
                        raw_score=proposed.raw_score,
                        min_score=proposed.min_score,
                        max_score=proposed.max_score,
                        rationale=proposed.rationale,
                    ),
                )
            ),
        ),
    )
    _ConfirmAnswers.answers = [True]
    _ConfirmAnswers.output_before_prompt = []
    _ConfirmAnswers.prompts = []
    monkeypatch.setattr(review_module, "Confirm", _ConfirmAnswers)
    reviewer = review_module.build_manual_judge_reviewer(
        _setup(),
        proposal.rubric,
        _previews(proposal),
        drafted_labels=(),
        supplied_labels=(),
        supplied_judgments=(),
        non_interactive=False,
        character_limit=1_200,
        page=False,
        console=console,
    )

    decisions = reviewer(proposal)

    assert decisions[0].accepted is True
    assert axis_name in buffer.getvalue()
    assert axis_id in buffer.getvalue()
    assert axis_id in Text.from_markup(_ConfirmAnswers.prompts[0]).plain


def test_interactive_correction_captures_score_and_human_judgment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejecting a proposal records both correction fields without changing judge authorship.

    Args:
        monkeypatch: Scoped score, judgment, and confirmation prompt replacements.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, color_system=None)
    proposal = _proposal(_trace("trace-a", completion="Done.", failed=False))
    _ConfirmAnswers.answers = [False]
    _ConfirmAnswers.output_before_prompt = []
    _ConfirmAnswers.prompts = []
    monkeypatch.setattr(review_module, "Confirm", _ConfirmAnswers)
    monkeypatch.setattr(review_module, "IntPrompt", _CorrectedScore)
    monkeypatch.setattr(review_module, "Prompt", _CorrectedJudgment)
    reviewer = review_module.build_manual_judge_reviewer(
        _setup(),
        proposal.rubric,
        _previews(proposal),
        drafted_labels=(),
        supplied_labels=(),
        supplied_judgments=(),
        non_interactive=False,
        character_limit=1_200,
        page=False,
        console=console,
    )

    decision = reviewer(proposal)[0]

    assert decision.accepted is False
    assert decision.correction is not None
    assert decision.correction.corrected_score == 0
    assert decision.correction.corrected_judgment == (
        "The tool result did not establish that the customer request was resolved."
    )


def test_noninteractive_multi_axis_inputs_accept_and_correct_independently() -> None:
    """Explicit decisions can accept one proposal and correct another with authored text."""
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, color_system=None)
    proposal = _proposal(_trace("trace-a", completion="Done.", failed=False), multi_axis=True)
    reviewer = review_module.build_manual_judge_reviewer(
        _setup(),
        proposal.rubric,
        _previews(proposal),
        drafted_labels=(),
        supplied_labels=("trace-a:task-success=1", "trace-a:policy-compliance=0"),
        supplied_judgments=(
            "trace-a:policy-compliance=The response violated the documented escalation rule.",
        ),
        non_interactive=True,
        character_limit=1_200,
        page=False,
        console=console,
    )

    accepted, corrected = reviewer(proposal)

    assert accepted.dimension_id == "task-success"
    assert accepted.accepted is True
    assert accepted.correction is None
    assert corrected.dimension_id == "policy-compliance"
    assert corrected.accepted is False
    assert corrected.correction is not None
    assert corrected.correction.corrected_score == 0
    assert corrected.correction.corrected_judgment == (
        "The response violated the documented escalation rule."
    )


def test_noninteractive_missing_labels_list_ready_to_paste_expressions() -> None:
    """The non-interactive error lists one paste-ready --label expression per missing key."""
    proposal = _proposal(_trace("trace-a", completion="Done.", failed=False), multi_axis=True)

    with pytest.raises(ValueError) as excinfo:
        review_module.build_manual_judge_reviewer(
            _setup(),
            proposal.rubric,
            _previews(proposal),
            drafted_labels=(),
            supplied_labels=(),
            supplied_judgments=(),
            non_interactive=True,
            character_limit=1_200,
            page=False,
            console=Console(file=io.StringIO(), width=100, color_system=None),
        )

    message = str(excinfo.value)
    assert message.startswith("missing labels: supply ")
    assert "--label trace-a:task-success=SCORE" in message
    assert "--label trace-a:policy-compliance=SCORE" in message


def test_noninteractive_missing_pairwise_labels_list_typed_winner_values() -> None:
    """The pairwise non-interactive error shows the typed winner values to paste."""
    trace = _trace("trace-a", completion="Candidate A finished.", failed=False)
    reference = _trace("trace-b", completion="Candidate B finished.", failed=False)
    proposal = _proposal(trace, reference=reference)

    with pytest.raises(ValueError) as excinfo:
        review_module.build_manual_judge_reviewer(
            _setup(response_shape="pairwise"),
            proposal.rubric,
            _previews(proposal),
            drafted_labels=(),
            supplied_labels=(),
            supplied_judgments=(),
            non_interactive=True,
            character_limit=1_200,
            page=False,
            console=Console(file=io.StringIO(), width=100, color_system=None),
        )

    message = str(excinfo.value)
    assert "--label trace-a:trace-b:task-success=WINNER" in message
    assert "(WINNER is winner_a, winner_b, or tie)" in message


def test_noninteractive_scalar_score_respects_the_saved_axis_range() -> None:
    """Explicit scalar corrections cannot leave the finalized inclusive range."""
    proposal = _proposal(_trace("trace-a", completion="Done.", failed=False))

    with pytest.raises(ValueError, match="from 0 through 1"):
        review_module.build_manual_judge_reviewer(
            _setup(),
            proposal.rubric,
            _previews(proposal),
            drafted_labels=(),
            supplied_labels=("trace-a:task-success=2",),
            supplied_judgments=("trace-a:task-success=Outside the saved range.",),
            non_interactive=True,
            character_limit=1_200,
            page=False,
            console=Console(file=io.StringIO(), color_system=None),
        )


def test_explicit_pairwise_json_target_preserves_trace_id_delimiters() -> None:
    """JSON targets address pairwise trace IDs containing colons and equals signs."""
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, color_system=None)
    trace = _trace("trace:a=1", completion="Candidate A finished.", failed=False)
    reference = _trace("trace:b=2", completion="Candidate B finished.", failed=False)
    proposal = _proposal(trace, reference=reference)
    target = json.dumps([trace.trace_id, reference.trace_id, "task-success"])
    reviewer = review_module.build_manual_judge_reviewer(
        _setup(response_shape="pairwise"),
        proposal.rubric,
        _previews(proposal),
        drafted_labels=(),
        supplied_labels=(target + "=winner_a",),
        supplied_judgments=(),
        non_interactive=True,
        character_limit=1_200,
        page=False,
        console=console,
    )

    decisions = reviewer(proposal)

    assert decisions[0].accepted is True
    assert decisions[0].correction is None


def test_page_expands_only_the_current_full_transcript() -> None:
    """Paging is explicit, current-trace scoped, and removes every truncation marker."""
    buffer = io.StringIO()
    console = _PagingConsole(buffer)
    completion = "full response " * 80
    proposal = _proposal(_trace("trace-a", completion=completion, failed=False), total=5)

    review_module.render_trace_proposal(
        proposal,
        character_limit=40,
        page=True,
        non_interactive=False,
        console=console,
    )

    output = buffer.getvalue()
    assert console.pager_count == 1
    assert "Trace 1 of 5" in output
    assert "Trace 2 of 5" not in output
    assert " ".join(completion.split()) in " ".join(output.split())
    assert "truncated" not in output


def _setup(*, response_shape: str = "scalar") -> ManualJudgeSetupArtifact:
    """Return the scalar or pairwise setup surface needed by the reviewer.

    Args:
        response_shape: Finalized configured-judge response shape.

    Returns:
        Minimal setup surface used by the CLI review adapter.
    """
    return cast(
        ManualJudgeSetupArtifact,
        SimpleNamespace(
            prompt_template=SimpleNamespace(
                response_shape=response_shape,
                score_projection=SimpleNamespace(
                    pairwise_scores={"winner_a": 1, "winner_b": 0, "tie": 1}
                ),
            )
        ),
    )


def _previews(proposal: ManualJudgeTraceProposal) -> tuple[JudgeTracePreview, ...]:
    """Return the frozen preview identity matching ``proposal``.

    Args:
        proposal: Current configured-judge trace proposal.

    Returns:
        Single matching preview identity.
    """
    return (
        cast(
            JudgeTracePreview,
            SimpleNamespace(
                trace_id=proposal.trace.trace_id,
                reference_trace_id=(
                    proposal.reference_trace.trace_id
                    if proposal.reference_trace is not None
                    else None
                ),
            ),
        ),
    )


def _proposal(
    trace: Trace,
    *,
    reference: Trace | None = None,
    total: int = 1,
    multi_axis: bool = False,
) -> ManualJudgeTraceProposal:
    """Return one configured-judge proposal over exact trace span citations.

    Args:
        trace: Target conversation trace.
        reference: Optional same-task candidate B trace.
        total: Total trace count shown in progress.
        multi_axis: Whether to add a second policy rubric axis.

    Returns:
        Deterministic proposal fixture with concrete trace citations.
    """
    task_success = default_judge_dimensions()[0]
    policy = task_success.model_copy(
        update={
            "dimension_id": "policy-compliance",
            "name": "Policy compliance",
            "description": "Whether the response followed the documented escalation policy.",
        }
    )
    dimensions: tuple[RubricDimension, ...] = (
        (task_success, policy) if multi_axis else (task_success,)
    )
    judgments = [
        SimpleNamespace(
            dimension_id="task-success",
            raw_score=1,
            min_score=task_success.min_score,
            max_score=task_success.max_score,
            rationale="The response completed the requested account lookup.",
        )
    ]
    if multi_axis:
        judgments.append(
            SimpleNamespace(
                dimension_id="policy-compliance",
                raw_score=1,
                min_score=policy.min_score,
                max_score=policy.max_score,
                rationale="The response followed the policy.",
            )
        )
    rubric = cast(Rubric, _RubricView(dimensions))
    judgment = cast(Judgment, SimpleNamespace(dimensions=tuple(judgments)))
    pairwise_citations = (
        tuple(
            (
                item.dimension_id,
                (f"{trace.trace_id}-assistant",)
                if item.dimension_id == "task-success"
                else (f"{trace.trace_id}-tool-result",),
                (f"{reference.trace_id}-assistant",)
                if item.dimension_id == "task-success"
                else (f"{reference.trace_id}-tool-result",),
            )
            for item in judgments
        )
        if reference is not None
        else ()
    )
    return ManualJudgeTraceProposal(
        position=1,
        total=total,
        trace=trace,
        reference_trace=reference,
        rubric=rubric,
        judgment=judgment,
        pairwise_citations=pairwise_citations,
    )


def _model() -> ModelSnapshot:
    """Return one exact secret-free assistant identity."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="openai",
        model_id="assistant-model",
        revision=None,
        capabilities_sha256="0" * 64,
        connection_sha256="1" * 64,
    )


def _trace(trace_id: str, *, completion: str, failed: bool) -> Trace:
    """Return a conversation with user, assistant, tool, result, and final response events.

    Args:
        trace_id: Stable trace and span identity prefix.
        completion: First captured assistant message.
        failed: Whether the terminal outcome records a failure.

    Returns:
        Complete normalized conversation fixture.
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
                    f"[{json_message('user', f'Please resolve customer issue {trace_id}.')}]"
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
        TraceSpan(
            span_id=f"{trace_id}-final",
            name="agent.model_call",
            started_at=started + timedelta(seconds=6),
            ended_at=started + timedelta(seconds=7),
            attributes={"gen_ai.response.text": "The customer request is resolved."},
            model=_model(),
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


def json_message(role: str, content: str) -> str:
    """Return one compact JSON object without importing a fixture serializer.

    Args:
        role: Normalized conversation role.
        content: Message content requiring JSON string escaping.

    Returns:
        Compact JSON object text.
    """
    escaped = content.replace("\\", "\\\\").replace('"', '\\"')
    return f'{{"role":"{role}","content":"{escaped}"}}'
