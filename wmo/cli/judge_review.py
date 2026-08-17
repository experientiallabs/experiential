"""One-trace-at-a-time terminal review for configured judge proposals."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm, IntPrompt, Prompt

from wmo.cli.judge_transcript import render_field, render_trace, span_evidence_text
from wmo.common.judging import Rubric, RubricDimension
from wmo.common.traces import TraceSpan
from wmo.optimize.router.judging.contracts import (
    HumanJudgeCorrection,
    JudgeTracePreview,
    ManualJudgeAxisDecision,
    ManualJudgeLabel,
    ManualJudgeSetupArtifact,
)
from wmo.optimize.router.judging.review import (
    ManualJudgeReviewer,
    ManualJudgeTraceProposal,
    manual_label_score,
)

_ReviewKey = tuple[str, str | None, str]


@dataclass(frozen=True)
class _ExplicitReviewInputs:
    """Validated score and judgment corrections keyed by frozen trace and axis."""

    labels: dict[_ReviewKey, ManualJudgeLabel]
    judgments: dict[_ReviewKey, str]


def build_manual_judge_reviewer(
    setup: ManualJudgeSetupArtifact,
    rubric: Rubric,
    previews: Sequence[JudgeTracePreview],
    *,
    drafted_labels: Sequence[ManualJudgeLabel],
    supplied_labels: Sequence[str],
    supplied_judgments: Sequence[str],
    non_interactive: bool,
    character_limit: int,
    page: bool,
    console: Console,
) -> ManualJudgeReviewer:
    """Build a reviewer that displays each proposal before taking a human decision.

    Args:
        setup: Finalized judge setup and score projection.
        rubric: Exact finalized rubric revision.
        previews: Frozen sample identities in review order.
        drafted_labels: Human labels already saved for this frozen sample.
        supplied_labels: Explicit CLI score or pairwise winner expressions.
        supplied_judgments: Explicit CLI corrected-judgment expressions.
        non_interactive: Whether every decision must be provided by flags.
        character_limit: Maximum characters shown for each transcript field.
        page: Whether each full transcript is displayed through a terminal pager.
        console: Rich console used for all review output and prompts.

    Returns:
        Callback invoked only after the configured judge response is immutable.

    Raises:
        ValueError: Explicit inputs are malformed, conflicting, or incomplete.
    """
    expected = tuple(
        (preview.trace_id, preview.reference_trace_id, dimension.dimension_id)
        for preview in previews
        for dimension in rubric.dimensions
    )
    explicit = _parse_explicit_inputs(
        setup,
        rubric,
        expected,
        drafted_labels=drafted_labels,
        supplied_labels=supplied_labels,
        supplied_judgments=supplied_judgments,
    )
    missing = tuple(key for key in expected if key not in explicit.labels)
    if non_interactive and missing:
        raise ValueError("missing labels: " + ", ".join(_display_key(key) for key in missing))

    def review(proposal: ManualJudgeTraceProposal) -> tuple[ManualJudgeAxisDecision, ...]:
        """Render one trace and return explicit decisions for all proposed axes.

        Args:
            proposal: Persisted configured-judge proposal for the current trace.

        Returns:
            One explicit human decision per proposed rubric axis.
        """
        render_trace_proposal(
            proposal,
            character_limit=character_limit,
            page=page,
            console=console,
        )
        reference_id = (
            proposal.reference_trace.trace_id if proposal.reference_trace is not None else None
        )
        decisions: list[ManualJudgeAxisDecision] = []
        for proposed in proposal.judgment.dimensions:
            axis = rubric.axis(proposed.dimension_id)
            key = (proposal.trace.trace_id, reference_id, proposed.dimension_id)
            label = explicit.labels.get(key)
            corrected_judgment = explicit.judgments.get(key)
            if label is not None:
                score = manual_label_score(setup, label)
                decisions.append(
                    _explicit_decision(
                        key,
                        axis=axis,
                        proposed_score=proposed.raw_score,
                        proposed_judgment=proposed.feedback,
                        score=score,
                        corrected_judgment=corrected_judgment,
                        non_interactive=non_interactive,
                        console=console,
                    )
                )
                continue
            accepted = Confirm.ask(
                f"Accept the proposed score and judgment for {escape(proposed.dimension_id)}?",
                default=True,
                console=console,
            )
            if accepted:
                decisions.append(
                    ManualJudgeAxisDecision(
                        dimension_id=proposed.dimension_id,
                        accepted=True,
                    )
                )
                continue
            score = _prompt_score(axis, console)
            judgment = _prompt_judgment(proposed.dimension_id, console)
            decisions.append(
                ManualJudgeAxisDecision(
                    dimension_id=proposed.dimension_id,
                    accepted=False,
                    correction=HumanJudgeCorrection(
                        corrected_score=score,
                        corrected_judgment=judgment,
                    ),
                )
            )
        return tuple(decisions)

    return review


def render_trace_proposal(
    proposal: ManualJudgeTraceProposal,
    *,
    character_limit: int,
    page: bool,
    console: Console,
) -> None:
    """Display one conversation and every configured-judge axis proposal.

    Args:
        proposal: Immutable configured-judge result and its source trace.
        character_limit: Maximum characters per field outside full paging.
        page: Whether to page the complete transcript.
        console: Rich console used for review output.
    """
    limit = None if page else character_limit

    def render_conversation() -> None:
        """Write only the current trace conversation to the active output buffer."""
        console.print(
            f"\n[bold]Trace {proposal.position} of {proposal.total}[/bold]",
        )
        if proposal.reference_trace is None:
            render_trace(console, proposal.trace, character_limit=limit)
            return
        console.print("\n[bold]Candidate A[/bold]")
        render_trace(console, proposal.trace, character_limit=limit)
        console.print("\n[bold]Candidate B[/bold]")
        render_trace(console, proposal.reference_trace, character_limit=limit)

    if page:
        with console.pager(styles=True):
            render_conversation()
    else:
        render_conversation()
    _render_axis_proposals(proposal, character_limit=limit, console=console)


def _render_axis_proposals(
    proposal: ManualJudgeTraceProposal,
    *,
    character_limit: int | None,
    console: Console,
) -> None:
    """Show complete rubric context and judge evidence before any human prompt.

    Args:
        proposal: Current trace and immutable configured-judge result.
        character_limit: Per-field limit, or ``None`` for complete evidence.
        console: Rich console receiving review output.
    """
    dimensions = {item.dimension_id: item for item in proposal.rubric.dimensions}
    reference_citations = {
        dimension_id: reference for dimension_id, _target, reference in proposal.pairwise_citations
    }
    console.print(
        f"\nConfigured judge proposals (rubric revision {proposal.rubric.rubric_id})",
        style="bold",
        markup=False,
    )
    for index, proposed in enumerate(proposal.judgment.dimensions, start=1):
        dimension = dimensions[proposed.dimension_id]
        console.print(
            f"\nAxis {index} of {len(dimensions)}: {dimension.name}",
            style="bold",
            markup=False,
        )
        console.print(f"Axis ID: {dimension.dimension_id}", markup=False)
        console.print(f"Description: {dimension.description}", markup=False)
        console.print(
            f"Numeric range: {dimension.min_score} to {dimension.max_score}",
            markup=False,
        )
        console.print("Score anchors:", style="bold")
        for anchor in dimension.anchors:
            console.print(f"  {anchor.score}: {anchor.description}", markup=False)
        console.print(f"Proposed score: {proposed.raw_score}", markup=False)
        render_field(
            console,
            "Proposed judgment",
            proposed.feedback,
            character_limit=character_limit,
        )
        console.print("Cited trace evidence:", style="bold")
        for span_id in proposed.evidence_span_ids:
            source, span = _find_evidence_span(proposal, span_id, reference=False)
            console.print(f"  {span_id} ({source}; {span.name})", markup=False)
            render_field(
                console,
                "  Evidence content",
                span_evidence_text(span),
                character_limit=character_limit,
            )
        cited_reference = reference_citations.get(proposed.dimension_id, ())
        if cited_reference:
            console.print("Cited reference trace evidence:", style="bold")
            for span_id in cited_reference:
                source, span = _find_evidence_span(proposal, span_id, reference=True)
                console.print(f"  {span_id} ({source}; {span.name})", markup=False)
                render_field(
                    console,
                    "  Evidence content",
                    span_evidence_text(span),
                    character_limit=character_limit,
                )


def _parse_explicit_inputs(
    setup: ManualJudgeSetupArtifact,
    rubric: Rubric,
    expected: Sequence[_ReviewKey],
    *,
    drafted_labels: Sequence[ManualJudgeLabel],
    supplied_labels: Sequence[str],
    supplied_judgments: Sequence[str],
) -> _ExplicitReviewInputs:
    """Validate persisted and flag-provided decisions against the frozen sample.

    Args:
        setup: Finalized judge response and score-projection contract.
        rubric: Exact ordered axes and inclusive score ranges.
        expected: Frozen trace, reference, and axis keys.
        drafted_labels: Previously persisted human score inputs.
        supplied_labels: Score expressions supplied to this invocation.
        supplied_judgments: Corrected judgment expressions supplied to this invocation.

    Returns:
        Validated score and judgment inputs keyed by frozen review identity.

    Raises:
        ValueError: Inputs are malformed, duplicated, conflicting, or unexpected.
    """
    pairwise = setup.prompt_template.response_shape == "pairwise"
    labels = {_label_key_from_model(item): item for item in drafted_labels}
    for key, label in labels.items():
        if label.score is not None and not rubric.axis(key[2]).contains_score(label.score):
            axis = rubric.axis(key[2])
            raise ValueError(
                f"saved label for {key[2]} must be an integer from "
                f"{axis.min_score} through {axis.max_score}"
            )
    drafted_keys = set(labels)
    supplied_keys: set[_ReviewKey] = set()
    for value in supplied_labels:
        key, body = _expression_parts(
            value,
            expected=expected,
            pairwise=pairwise,
            kind="labels",
        )
        if key in supplied_keys:
            raise ValueError("duplicate label for " + _display_key(key))
        supplied_keys.add(key)
        axis = None if pairwise else rubric.axis(key[2])
        parsed = _label(
            key,
            _label_value(body, pairwise=pairwise, axis=axis),
            pairwise=pairwise,
        )
        existing = labels.get(key)
        if existing is not None and existing != parsed:
            raise ValueError(
                "supplied label conflicts with the saved label for " + _display_key(key)
            )
        if key in labels and key not in drafted_keys:
            raise ValueError("duplicate label for " + _display_key(key))
        labels[key] = parsed
    judgments: dict[_ReviewKey, str] = {}
    for value in supplied_judgments:
        key, body = _expression_parts(
            value,
            expected=expected,
            pairwise=pairwise,
            kind="judgments",
        )
        text = body.strip()
        if not text:
            raise ValueError("corrected judgments must contain nonempty text")
        if key in judgments:
            raise ValueError("duplicate corrected judgment for " + _display_key(key))
        judgments[key] = text
    expected_set = set(expected)
    unexpected = sorted(set(labels).union(judgments).difference(expected_set))
    if unexpected:
        raise ValueError("unexpected review inputs: " + ", ".join(map(_display_key, unexpected)))
    without_score = sorted(set(judgments).difference(labels))
    if without_score:
        raise ValueError(
            "corrected judgments also require --label for "
            + ", ".join(map(_display_key, without_score))
        )
    return _ExplicitReviewInputs(labels=labels, judgments=judgments)


def _explicit_decision(
    key: _ReviewKey,
    *,
    axis: RubricDimension,
    proposed_score: int,
    proposed_judgment: str,
    score: int,
    corrected_judgment: str | None,
    non_interactive: bool,
    console: Console,
) -> ManualJudgeAxisDecision:
    """Convert one explicit score into acceptance or an authored correction.

    Args:
        key: Frozen trace, reference, and axis identity.
        axis: Finalized axis that bounds the accepted score.
        proposed_score: Configured judge's immutable score.
        proposed_judgment: Configured judge's immutable judgment text.
        score: Explicit human-authorized score.
        corrected_judgment: Optional explicit human correction text.
        non_interactive: Whether missing correction text must fail without prompting.
        console: Rich console used for any required correction prompt.

    Returns:
        Explicit acceptance or complete human correction.

    Raises:
        ValueError: The score is outside the axis or a correction lacks authored judgment text.
    """
    dimension_id = key[2]
    if not axis.contains_score(score):
        raise ValueError(
            f"judge labels for {dimension_id} must be integers from "
            f"{axis.min_score} through {axis.max_score}"
        )
    if score == proposed_score and (
        corrected_judgment is None or corrected_judgment == proposed_judgment
    ):
        return ManualJudgeAxisDecision(dimension_id=dimension_id, accepted=True)
    judgment = corrected_judgment
    if judgment is None:
        if non_interactive:
            raise ValueError(
                f"a corrected score requires --judgment {_display_key(key)}=CORRECTED_JUDGMENT"
            )
        console.print(
            f"Saved human score {score} differs from the configured judge proposal "
            f"{proposed_score} for {dimension_id}.",
            markup=False,
        )
        judgment = _prompt_judgment(dimension_id, console)
    return ManualJudgeAxisDecision(
        dimension_id=dimension_id,
        accepted=False,
        correction=HumanJudgeCorrection(
            corrected_score=score,
            corrected_judgment=judgment,
        ),
    )


def _prompt_score(axis: RubricDimension, console: Console) -> int:
    """Prompt until the reviewer supplies an integer in the rubric range.

    Args:
        axis: Rubric axis receiving a corrected score.
        console: Interactive Rich console.

    Returns:
        Human-authored integer inside the axis range.
    """
    while True:
        score = IntPrompt.ask(
            f"Corrected score for {escape(axis.dimension_id)} "
            f"({axis.min_score} through {axis.max_score})",
            console=console,
        )
        if axis.contains_score(score):
            return score
        console.print(
            f"Corrected scores must be integers from {axis.min_score} through {axis.max_score}."
        )


def _prompt_judgment(dimension_id: str, console: Console) -> str:
    """Prompt until the reviewer supplies a nonempty authored judgment.

    Args:
        dimension_id: Rubric axis receiving corrected judgment text.
        console: Interactive Rich console.

    Returns:
        Nonempty stripped human-authored judgment text.
    """
    while True:
        judgment = Prompt.ask(
            f"Corrected judgment for {escape(dimension_id)}",
            console=console,
        ).strip()
        if judgment:
            return judgment
        console.print("Corrected judgments must contain nonempty text.")


def _find_evidence_span(
    proposal: ManualJudgeTraceProposal,
    span_id: str,
    *,
    reference: bool,
) -> tuple[str, TraceSpan]:
    """Resolve one cited span to its displayed candidate without guessing.

    Args:
        proposal: Current trace proposal containing displayed candidates.
        span_id: Exact judge-cited normalized span identity.
        reference: Whether the citation belongs to the pairwise reference trace.

    Returns:
        Human-readable candidate label and exact cited span.

    Raises:
        ValueError: The cited span is absent from its displayed trace.
    """
    trace = proposal.reference_trace if reference else proposal.trace
    source = (
        "candidate B"
        if reference
        else ("candidate A" if proposal.reference_trace is not None else "trace")
    )
    if trace is not None:
        for span in trace.spans:
            if span.span_id == span_id:
                return source, span
    raise ValueError(f"judge cited {source} evidence that is not displayed: {span_id}")


def _label(
    key: _ReviewKey,
    value: int | str,
    *,
    pairwise: bool,
) -> ManualJudgeLabel:
    """Build one validated explicit label from its key and typed value.

    Args:
        key: Frozen trace, optional reference, and rubric axis identity.
        value: Scalar score or typed pairwise preference.
        pairwise: Whether the finalized response shape is pairwise.

    Returns:
        Validated explicit human label.
    """
    trace_id, reference_id, dimension_id = key
    return ManualJudgeLabel.model_validate(
        {
            "trace_id": trace_id,
            "reference_trace_id": reference_id,
            "dimension_id": dimension_id,
            **({"winner": value} if pairwise else {"score": value}),
        }
    )


def _expression_parts(
    value: str,
    *,
    expected: Sequence[_ReviewKey],
    pairwise: bool,
    kind: str,
) -> tuple[_ReviewKey, str]:
    """Parse one score or judgment expression without splitting valid trace IDs.

    Args:
        value: CLI expression containing a target and value.
        expected: Frozen trace, optional reference, and dimension identities.
        pairwise: Whether the target must include a reference trace.
        kind: Human-readable expression class for validation errors.

    Returns:
        Exact frozen review key and the unmodified expression body.

    Raises:
        ValueError: The target is malformed or ambiguous.
    """
    expected_parts = 3 if pairwise else 2
    stripped = value.lstrip()
    if stripped.startswith("["):
        try:
            target, end = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{kind} contain a malformed JSON target") from exc
        remainder = stripped[end:]
        if (
            not isinstance(target, list)
            or len(target) != expected_parts
            or any(not isinstance(part, str) or not part for part in target)
            or not remainder.startswith("=")
        ):
            raise ValueError(f"{kind} JSON targets must contain {expected_parts} nonempty strings")
        parts = cast(list[str], target)
        key: _ReviewKey = (parts[0], parts[1], parts[2]) if pairwise else (parts[0], None, parts[1])
        return key, remainder[1:]
    matches = tuple(
        (key, value[len(prefix) :])
        for key in expected
        if value.startswith(prefix := _display_key(key) + "=")
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        json_target = (
            '["TRACE_ID", "REFERENCE_TRACE_ID", "DIMENSION_ID"]=VALUE'
            if pairwise
            else '["TRACE_ID", "DIMENSION_ID"]=VALUE'
        )
        raise ValueError(
            f"{kind} target is ambiguous; use a JSON array target such as {json_target}"
        )
    syntax = (
        "TRACE_ID:REFERENCE_TRACE_ID:DIMENSION_ID=VALUE"
        if pairwise
        else "TRACE_ID:DIMENSION_ID=VALUE"
    )
    raise ValueError(
        f"{kind} must target a frozen review key using {syntax}; "
        "use a JSON array target when identifiers contain ambiguous delimiters"
    )


def _label_value(
    raw: str,
    *,
    pairwise: bool,
    axis: RubricDimension | None,
) -> int | str:
    """Parse one axis-range score or typed pairwise winner.

    Args:
        raw: Unmodified label body after the target separator.
        pairwise: Whether the value must be a typed pairwise preference.
        axis: Axis that bounds scalar scores, or ``None`` for pairwise input.

    Returns:
        Integer score or typed pairwise winner.

    Raises:
        ValueError: The value is malformed or outside the supported contract.
    """
    if pairwise:
        if raw not in {"winner_a", "winner_b", "tie"}:
            raise ValueError("pairwise judge labels must use winner_a, winner_b, or tie")
        return raw
    try:
        score = int(raw)
    except ValueError as exc:
        raise ValueError("judge labels must use an integer score") from exc
    if axis is None or not axis.contains_score(score):
        if axis is None:
            raise ValueError("scalar judge labels require a finalized rubric axis")
        raise ValueError(
            f"judge labels for {axis.dimension_id} must be integers from "
            f"{axis.min_score} through {axis.max_score}"
        )
    return score


def _label_key_from_model(label: ManualJudgeLabel) -> _ReviewKey:
    """Return the stable review key carried by one persisted human label.

    Args:
        label: Persisted explicit human label.

    Returns:
        Trace, optional reference trace, and rubric axis identity.
    """
    return label.trace_id, label.reference_trace_id, label.dimension_id


def _display_key(key: _ReviewKey) -> str:
    """Render one stable review key in the matching CLI expression form.

    Args:
        key: Trace, optional reference trace, and rubric axis identity.

    Returns:
        Colon-separated CLI target text.
    """
    return ":".join(part for part in key if part is not None)
