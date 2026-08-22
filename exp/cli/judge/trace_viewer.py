"""Full-screen scrollable terminal viewer for one calibration trace proposal.

The viewer renders the complete transcript and every configured-judge proposal as styled
blocks, then lets the reviewer move through them with familiar pager keys: arrow keys and
``j``/``k`` scroll by line, space and ``b`` page, ``d``/``u`` half-page, ``g``/``G`` jump to
the top or bottom, and ``n``/``p`` skip between conversation steps. ``q`` closes the viewer
and returns to the decision prompts. Rendering is pure so the block builder is testable
without a terminal; only the event loop touches raw terminal input.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from pydantic import JsonValue
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from exp.cli.judge.transcript import (
    assistant_completion,
    jsonish_text,
    model_display_name,
    user_input,
)
from exp.common.traces import Trace, TraceSpan
from exp.optimize.router.judging.review import ManualJudgeTraceProposal

_HELP = "j/k scroll  space/b page  d/u half  n/p step  g/G top/end  q continue"


@dataclass(frozen=True)
class ViewerBlock:
    """One renderable transcript unit with an optional navigation anchor.

    Args:
        anchor: Jump-target label shown during step navigation, or ``None``.
        renderable: Styled Rich renderable for this unit.
    """

    anchor: str | None
    renderable: RenderableType


def proposal_blocks(proposal: ManualJudgeTraceProposal) -> tuple[ViewerBlock, ...]:
    """Build every styled transcript and judge-proposal block for one trace.

    Args:
        proposal: Immutable configured-judge result and its source trace.

    Returns:
        Ordered blocks covering the conversation, outcome, and every axis proposal.
    """
    blocks: list[ViewerBlock] = []
    if proposal.reference_trace is None:
        blocks.extend(_trace_blocks(proposal.trace, candidate=None))
    else:
        blocks.append(ViewerBlock("Candidate A", Rule("[bold]Candidate A[/bold]", style="cyan")))
        blocks.extend(_trace_blocks(proposal.trace, candidate="A"))
        blocks.append(ViewerBlock("Candidate B", Rule("[bold]Candidate B[/bold]", style="magenta")))
        blocks.extend(_trace_blocks(proposal.reference_trace, candidate="B"))
    blocks.append(ViewerBlock("Judge proposals", Rule("[bold]Configured judge proposals[/bold]")))
    dimensions = {item.dimension_id: item for item in proposal.rubric.dimensions}
    for index, proposed in enumerate(proposal.judgment.dimensions, start=1):
        dimension = dimensions[proposed.dimension_id]
        body = Text()
        body.append("Score ", style="bold")
        body.append(
            f" {proposed.raw_score} ",
            style="bold black on green"
            if proposed.raw_score == dimension.max_score
            else "bold white on red",
        )
        body.append(f"  range {dimension.min_score} to {dimension.max_score}\n", style="dim")
        body.append(dimension.description + "\n", style="italic")
        for anchor in dimension.anchors:
            body.append(f"  {anchor.score}: ", style="bold")
            body.append(anchor.description + "\n", style="dim")
        body.append("\n")
        body.append(proposed.rationale or "", style="default")
        blocks.append(
            ViewerBlock(
                f"Axis {index}",
                Panel(
                    body,
                    title=(
                        f"[bold]Axis {index} of {len(dimensions)}: {dimension.name}[/bold] "
                        f"[dim]({dimension.dimension_id})[/dim]"
                    ),
                    title_align="left",
                    border_style="blue",
                ),
            )
        )
    return tuple(blocks)


def _trace_blocks(trace: Trace, *, candidate: str | None) -> tuple[ViewerBlock, ...]:
    """Build styled conversation blocks for one normalized trace.

    Args:
        trace: Verified immutable normalized production trace.
        candidate: Pairwise candidate letter, or ``None`` for a single trace.

    Returns:
        Ordered task, span, and outcome blocks.
    """
    prefix = f"{candidate}:" if candidate is not None else ""
    blocks: list[ViewerBlock] = [
        ViewerBlock(
            f"{prefix}Task",
            Panel(
                Text(trace.task, style="bold"),
                title="[bold cyan]Original user request[/bold cyan]",
                title_align="left",
                border_style="cyan",
            ),
        )
    ]
    if trace.initial_context:
        blocks.append(
            ViewerBlock(
                None,
                Panel(
                    Text(jsonish_text(trace.initial_context), style="dim"),
                    title="[dim]Initial context[/dim]",
                    title_align="left",
                    border_style="bright_black",
                ),
            )
        )
    completion_positions = tuple(
        index for index, span in enumerate(trace.spans) if assistant_completion(span.attributes)
    )
    final_completion = completion_positions[-1] if completion_positions else None
    step = 0
    for index, span in enumerate(trace.spans):
        span_blocks = _span_blocks(span, final_response=index == final_completion)
        for block in span_blocks:
            if block.anchor is not None:
                step += 1
                blocks.append(ViewerBlock(f"{prefix}Step {step}", block.renderable))
            else:
                blocks.append(block)
    blocks.append(ViewerBlock(f"{prefix}Outcome", _outcome_block(trace)))
    return tuple(blocks)


def _span_blocks(span: TraceSpan, *, final_response: bool) -> tuple[ViewerBlock, ...]:
    """Build styled blocks for one normalized chronological span.

    Args:
        span: One normalized trace span.
        final_response: Whether this span holds the last captured assistant response.

    Returns:
        Ordered role-styled blocks; the first carries the step anchor.
    """
    attributes = span.attributes
    operation = attributes.get("gen_ai.operation.name")
    tool_name = attributes.get("gen_ai.tool.name")
    arguments = attributes.get("gen_ai.tool.call.arguments")
    result = attributes.get("gen_ai.tool.message")
    if result is None:
        result = attributes.get("gen_ai.tool.output")
    completion = assistant_completion(attributes)
    user_message = user_input(attributes)
    subtitle = f"[dim]{span.span_id} · {span.name}[/dim]"
    blocks: list[ViewerBlock] = []
    if user_message is not None:
        blocks.append(
            ViewerBlock(
                "step",
                Panel(
                    Text(user_message),
                    title="[bold cyan]User[/bold cyan]",
                    subtitle=subtitle,
                    title_align="left",
                    subtitle_align="right",
                    border_style="cyan",
                ),
            )
        )
    if completion:
        model = f" · {model_display_name(span.model)}" if span.model is not None else ""
        title = "Final response" if final_response else "Assistant"
        blocks.append(
            ViewerBlock(
                "step",
                Panel(
                    Text(completion),
                    title=f"[bold green]{title}[/bold green][dim]{model}[/dim]",
                    subtitle=subtitle,
                    title_align="left",
                    subtitle_align="right",
                    border_style="green" if not final_response else "bright_green",
                ),
            )
        )
    if operation != "execute_tool" and isinstance(tool_name, str):
        model = f" · {model_display_name(span.model)}" if span.model is not None else ""
        blocks.append(
            ViewerBlock(
                "step",
                Panel(
                    _tool_call_text(tool_name, arguments),
                    title=f"[bold yellow]Tool call · {tool_name}[/bold yellow][dim]{model}[/dim]",
                    subtitle=subtitle,
                    title_align="left",
                    subtitle_align="right",
                    border_style="yellow",
                ),
            )
        )
    if operation == "execute_tool":
        result_name = tool_name if isinstance(tool_name, str) else span.name
        body = (
            Text(jsonish_text(result), style="grey74")
            if result is not None
            else Text("(no captured output)", style="dim italic")
        )
        blocks.append(
            ViewerBlock(
                "step",
                Panel(
                    body,
                    title=f"[bold bright_black]Output · {result_name}[/bold bright_black]",
                    subtitle=subtitle,
                    title_align="left",
                    subtitle_align="right",
                    border_style="bright_black",
                ),
            )
        )
    if span.failure is not None:
        blocks.append(
            ViewerBlock(
                "step" if not blocks else None,
                Panel(
                    Text(f"{span.failure.code.value}: {span.failure.message}", style="red"),
                    title="[bold red]Span failure[/bold red]",
                    subtitle=subtitle,
                    title_align="left",
                    subtitle_align="right",
                    border_style="red",
                ),
            )
        )
    return tuple(blocks)


def _tool_call_text(tool_name: str, arguments: JsonValue | None) -> Text:
    """Render tool arguments, showing plain commands as a shell line.

    Args:
        tool_name: Captured tool name.
        arguments: Captured native or JSON-encoded tool arguments.

    Returns:
        Shell-style text for single-command tools, otherwise indented JSON.
    """
    decoded = arguments
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError:
            decoded = arguments
    if isinstance(decoded, dict) and set(decoded) == {"command"}:
        command = decoded["command"]
        if isinstance(command, str):
            text = Text()
            text.append("$ ", style="bold bright_green")
            text.append(command, style="bold")
            return text
    if arguments is None:
        return Text(tool_name, style="bold")
    return Text(jsonish_text(arguments if isinstance(arguments, str) else decoded))


def _outcome_block(trace: Trace) -> RenderableType:
    """Build the final-outcome block for one trace.

    Args:
        trace: Verified normalized production trace.

    Returns:
        Styled outcome panel, red when a failure was recorded.
    """
    outcome = trace.outcome
    if outcome is None:
        return Panel(
            Text("Not recorded", style="dim italic"),
            title="[bold]Final outcome[/bold]",
            title_align="left",
            border_style="bright_black",
        )
    body = Text(outcome.status)
    if outcome.outcome_name is not None:
        body.append(f" ({outcome.outcome_name})")
    failed = outcome.failure is not None
    if outcome.failure is not None:
        body.append(
            f"\n{outcome.failure.code.value}: {outcome.failure.message} "
            f"(retryable={str(outcome.failure.retryable).lower()})",
            style="red",
        )
    return Panel(
        body,
        title="[bold]Final outcome[/bold]",
        title_align="left",
        border_style="red" if failed else "magenta",
    )


def render_block_lines(
    blocks: tuple[ViewerBlock, ...],
    *,
    width: int,
) -> tuple[tuple[str, ...], tuple[tuple[int, str], ...]]:
    """Render blocks to ANSI lines and collect anchor start offsets.

    Args:
        blocks: Ordered viewer blocks.
        width: Exact terminal column count used for wrapping.

    Returns:
        All rendered lines and ``(line_offset, anchor)`` pairs for navigation.
    """
    console = Console(
        width=width,
        force_terminal=True,
        color_system="truecolor",
        file=None,
        highlight=False,
    )
    lines: list[str] = []
    anchors: list[tuple[int, str]] = []
    for block in blocks:
        if block.anchor is not None:
            anchors.append((len(lines), block.anchor))
        with console.capture() as capture:
            console.print(block.renderable)
        rendered = capture.get().split("\n")
        if rendered and rendered[-1] == "":
            rendered.pop()
        lines.extend(rendered)
    return tuple(lines), tuple(anchors)


KeyReader = Callable[[], str]


def view_trace_proposal(
    proposal: ManualJudgeTraceProposal,
    *,
    console: Console,
    key_reader: KeyReader | None = None,
) -> None:
    """Show one trace proposal in a full-screen scrollable viewer until ``q``.

    Args:
        proposal: Immutable configured-judge result and its source trace.
        console: Interactive terminal console owning the display.
        key_reader: Optional key supplier; defaults to raw stdin reads.
    """
    blocks = proposal_blocks(proposal)
    reader = key_reader if key_reader is not None else _read_key
    title = f"Trace {proposal.position} of {proposal.total}"
    with _alternate_screen(console):
        _event_loop(console, blocks, title=title, reader=reader)


@contextmanager
def _alternate_screen(console: Console) -> Iterator[None]:
    """Enter the alternate screen with a hidden cursor and always restore it.

    Args:
        console: Terminal console owning the display.

    Yields:
        Control while the alternate screen is active.
    """
    file = console.file
    file.write("\x1b[?1049h\x1b[?25l")
    file.flush()
    try:
        yield
    finally:
        file.write("\x1b[?25h\x1b[?1049l")
        file.flush()


def _event_loop(
    console: Console,
    blocks: tuple[ViewerBlock, ...],
    *,
    title: str,
    reader: KeyReader,
) -> None:
    """Draw the viewport and apply navigation keys until the reviewer continues.

    Args:
        console: Terminal console owning the display.
        blocks: Ordered rendered viewer blocks.
        title: Header title identifying the current trace.
        reader: Blocking single-key supplier.
    """
    top = 0
    width = height = -1
    lines: tuple[str, ...] = ()
    anchors: tuple[tuple[int, str], ...] = ()
    while True:
        size = console.size
        if (size.width, size.height) != (width, height):
            width, height = size.width, size.height
            lines, anchors = render_block_lines(blocks, width=width)
        view_height = max(1, height - 2)
        maximum_top = max(0, len(lines) - view_height)
        top = max(0, min(top, maximum_top))
        _draw(
            console,
            lines,
            anchors,
            top=top,
            width=width,
            height=height,
            view_height=view_height,
            title=title,
        )
        key = reader()
        if key in ("q", "Q"):
            return
        if key == "\x03":
            raise KeyboardInterrupt
        top = _apply_key(
            key,
            top=top,
            view_height=view_height,
            maximum_top=maximum_top,
            anchors=anchors,
        )


def _apply_key(
    key: str,
    *,
    top: int,
    view_height: int,
    maximum_top: int,
    anchors: tuple[tuple[int, str], ...],
) -> int:
    """Return the next scroll offset for one navigation key.

    Args:
        key: Normalized key name.
        top: Current first visible line offset.
        view_height: Visible content rows.
        maximum_top: Largest valid scroll offset.
        anchors: Ordered ``(line_offset, anchor)`` navigation targets.

    Returns:
        Clamped next scroll offset.
    """
    half = max(1, view_height // 2)
    if key in ("j", "down"):
        top += 1
    elif key in ("k", "up"):
        top -= 1
    elif key in (" ", "f", "pgdn"):
        top += view_height
    elif key in ("b", "pgup"):
        top -= view_height
    elif key == "d":
        top += half
    elif key == "u":
        top -= half
    elif key in ("g", "home"):
        top = 0
    elif key in ("G", "end"):
        top = maximum_top
    elif key == "n":
        following = tuple(offset for offset, _ in anchors if offset > top)
        top = following[0] if following else maximum_top
    elif key == "p":
        preceding = tuple(offset for offset, _ in anchors if offset < top)
        top = preceding[-1] if preceding else 0
    return max(0, min(top, maximum_top))


def _draw(
    console: Console,
    lines: tuple[str, ...],
    anchors: tuple[tuple[int, str], ...],
    *,
    top: int,
    width: int,
    height: int,
    view_height: int,
    title: str,
) -> None:
    """Paint the header, visible content window, and key-help footer.

    Args:
        console: Terminal console owning the display.
        lines: All rendered ANSI content lines.
        anchors: Ordered navigation targets used for the position label.
        top: First visible line offset.
        width: Terminal column count.
        height: Terminal row count.
        view_height: Visible content rows.
        title: Header title identifying the current trace.
    """
    bottom = min(len(lines), top + view_height)
    if len(lines) <= view_height:
        percent = "all"
    elif bottom >= len(lines):
        percent = "end"
    else:
        percent = f"{(bottom * 100) // len(lines)}%"
    current = [anchor for offset, anchor in anchors if offset <= top]
    position = current[-1] if current else (anchors[0][1] if anchors else "")
    header = Text(f" {title} ", style="bold black on cyan")
    header.append(f" {position} ", style="bold white on grey23")
    header.append(f" {percent} ", style="black on cyan")
    padding = width - header.cell_len
    if padding > 0:
        header.append(" " * padding, style="on grey15")
    footer = Text(f" {_HELP} ", style="black on grey74")
    footer_padding = width - footer.cell_len
    if footer_padding > 0:
        footer.append(" " * footer_padding, style="on grey74")
    file = console.file
    buffer: list[str] = ["\x1b[H"]
    with console.capture() as capture:
        console.print(header, end="")
    buffer.append(capture.get() + "\x1b[K\r\n")
    for row in range(view_height):
        index = top + row
        content = lines[index] if index < len(lines) else ""
        buffer.append(content + "\x1b[K\r\n")
    with console.capture() as capture:
        console.print(footer, end="")
    buffer.append(capture.get() + "\x1b[K")
    file.write("".join(buffer))
    file.flush()


def _read_key() -> str:
    """Read one normalized key from raw stdin, decoding arrow and page sequences.

    Returns:
        Single character, or a normalized name such as ``up`` or ``pgdn``.
    """
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    saved = termios.tcgetattr(descriptor)
    try:
        tty.setraw(descriptor)
        first = sys.stdin.read(1)
        if first != "\x1b":
            return first
        second = sys.stdin.read(1)
        if second != "[":
            return first
        third = sys.stdin.read(1)
        named = {"A": "up", "B": "down", "H": "home", "F": "end"}
        if third in named:
            return named[third]
        if third in ("5", "6", "1", "4"):
            fourth = sys.stdin.read(1)
            if fourth == "~":
                return {"5": "pgup", "6": "pgdn", "1": "home", "4": "end"}[third]
        return first
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)
