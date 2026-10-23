"""Positional memory log, its ordered compression tree, and a budgeted overview.

The log is append-only. Line ``i`` of a lineage has an identity derived from the lineage and
``i``, so position is identity and a conflicting append fails instead of forking history.
Lines are drafted into the memory version that commits them and materialized only after that
commit, so an interrupted publication never occupies a position.

Summaries cover aligned power-of-two blocks ``[lo, hi)``. They are a derived, model-written
cache: a block is built only after both halves exist, smallest blocks first, from the raw
lines when small and from the two child summaries otherwise. The raw log stays authoritative.

The overview tiles the first ``T`` lines within a line budget. Recent lines stay verbatim and
older blocks are shown compressed. Readers never call a model: a block whose summary is not
built yet, or that contains a withheld line, is opened into its halves instead.

Forgetting never deletes. Each block has a generation, pinned by the memory version a reader
uses; forgetting a block (and every ancestor containing it) raises its generation, and the
rebuilt summary takes a new identity. Until it is built, readers at that version open the
block into its children; readers pinned to older versions keep the older generation. A rebuild
leaves out lines that are retired at build time and records them as ``omitted``, so a reader
that still admits such a line opens the block instead of losing it.
"""

import json
import re
from collections.abc import Callable, Container, Mapping, Sequence
from datetime import datetime
from typing import Literal, cast

from pydantic import Field, field_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    canonical_json_bytes,
    sha256_json,
    stable_id,
)
from exp.common.models.client import RequestCapacityExceeded
from exp.common.project import ArtifactStore, artifact_input
from exp.runtime.models.structured_completion import CompletionNotDispatched, ReturnedOutputRejected

LINE_BYTES = 280
TARGET_BYTES = 220
RAW_MAX = 16
MAX_ATTEMPTS = 3
LIFETIME_ATTEMPTS = 2 * MAX_ATTEMPTS
LineKind = Literal["observed", "hypothesis"]
Block = tuple[int, int]
Generations = Mapping[Block, int]
AUTHORITY: dict[str, str] = {
    "observed": "Deterministic record of a consumed trajectory's native outcomes; it "
    "describes that instance, not the current one",
    "hypothesis": "Unresolved model-written hypothesis from an earlier trajectory",
    "summary": "Model-written lossy compression of earlier memory lines; it may blend "
    "observations and unresolved hypotheses and never describes the current instance",
}
_COMPRESS = """Compress the supplied simulator memory into one line of about {target} bytes; the
hard limit is {limit} bytes and longer lines are rejected.
Keep what helps predict this environment's tool behavior later; drop what does not.
Invent nothing. Keep native observations distinguishable from unresolved hypotheses: write
"[unresolved]" before any hypothesis you keep. Entity values describe past instances only.
Return the required structured summary.
"""
COMPRESSION_PROMPT = _COMPRESS


class LineDraft(ContractModel):
    """Content of one log line before its position is materialized.

    Attributes:
        kind: Native observation record or unresolved hypothesis.
        text: One line of at most ``LINE_BYTES`` UTF-8 bytes.
        origin: Full record identity (task lineage or research record) behind the line.
        derived_from: Consumed trajectory artifact the line was produced from.
        depends_on: Other consumed trajectories supplied to the work that produced it.
    """

    kind: LineKind
    text: str
    origin: str = Field(min_length=1)
    derived_from: ArtifactInput
    depends_on: tuple[ArtifactInput, ...] = ()

    @field_validator("text")
    @classmethod
    def one_line(cls, value: str) -> str:
        """Reject multi-line or oversized text rather than trimming it."""
        return check_line(value)


class MemoryLine(ArtifactEnvelope):
    """One immutable log line at a fixed position.

    Attributes:
        lineage: Memory lineage owning the log.
        index: Zero-based position; identity derives from lineage and index.
        kind: Native observation record or unresolved hypothesis.
        text: One line of at most ``LINE_BYTES`` UTF-8 bytes.
        origin: Full record identity (task lineage or research record) behind the line.
        derived_from: Consumed trajectory artifact the line was produced from.
        depends_on: Other consumed trajectories whose memory was supplied to the work that
            produced this line. Eligibility requires all of them, closing indirect provenance.
    """

    lineage: str = Field(min_length=1)
    index: int = Field(ge=0, strict=True)
    kind: LineKind
    text: str
    origin: str = Field(min_length=1)
    derived_from: ArtifactInput
    depends_on: tuple[ArtifactInput, ...] = ()

    @field_validator("text")
    @classmethod
    def one_line(cls, value: str) -> str:
        """Reject multi-line or oversized text rather than trimming it."""
        return check_line(value)


class MemorySummary(ArtifactEnvelope):
    """Accepted compression of one aligned block.

    Attributes:
        lineage: Memory lineage owning the tree.
        lo: First covered line position.
        hi: One past the last covered line position.
        text: One line of at most ``LINE_BYTES`` UTF-8 bytes.
        request_sha256: Exact compression request that produced the text.
        children: Artifacts compressed into this summary: the block's lines when small,
            otherwise its two child summaries. Source ancestry resolves through them.
        generation: Rebuild count of this block; 0 is the original positional summary.
        omitted: Line positions in the block left out because they were retired when it
            was built. Empty when every line is covered.
    """

    lineage: str = Field(min_length=1)
    lo: int = Field(ge=0, strict=True)
    hi: int = Field(ge=2, strict=True)
    text: str
    request_sha256: Sha256
    children: tuple[ArtifactInput, ...] = Field(min_length=1)
    generation: int = Field(default=0, ge=0, strict=True, exclude_if=lambda value: value == 0)
    omitted: tuple[int, ...] = Field(default=(), exclude_if=lambda value: not value)

    @field_validator("text")
    @classmethod
    def one_line(cls, value: str) -> str:
        """Reject multi-line or oversized summaries rather than trimming them."""
        return check_line(value)


def check_line(text: str) -> str:
    """Require one nonblank line of at most ``LINE_BYTES`` UTF-8 bytes."""
    if not text.strip() or "\n" in text or "\r" in text:
        raise ValueError("a memory line is one nonblank line")
    if len(text.encode()) > LINE_BYTES:
        raise ValueError(f"a memory line is at most {LINE_BYTES} bytes")
    return text


def fit_line(text: str) -> str:
    """Deterministically render code-derived text as one bounded line.

    Only for text the system derives itself (its full source stays recallable); model
    outputs that must be one line are validated with ``check_line`` instead.
    """
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat.encode()) <= LINE_BYTES:
        return flat
    cut = flat.encode()[: LINE_BYTES - 3].decode(errors="ignore").rstrip()
    return cut + "..."


def line_id(lineage: str, index: int) -> str:
    """Only identity line ``index`` of ``lineage`` may occupy."""
    return stable_id("memory-line", {"lineage": lineage, "index": index})


def summary_id(lineage: str, lo: int, hi: int, generation: int = 0) -> str:
    """Only identity generation ``generation`` of block ``[lo, hi)`` may occupy."""
    content: JsonObject = {"lineage": lineage, "lo": lo, "hi": hi}
    if generation:
        content["generation"] = generation
    return stable_id("memory-summary", content)


def containing_blocks(total: int, positions: Sequence[int]) -> list[Block]:
    """Every complete aligned block of size two or more within ``total`` holding a position.

    Forgetting a line's summaries forgets exactly these: the smallest block holding it and
    all of its complete ancestors.
    """
    found: set[Block] = set()
    for position in positions:
        size = 2
        while size <= total:
            lo = position - position % size
            if lo + size <= total:
                found.add((lo, lo + size))
            size *= 2
    return sorted(found, key=lambda block: (block[1] - block[0], block[0]))


def append_lines(
    store: ArtifactStore,
    lineage: str,
    start: int,
    drafts: Sequence[LineDraft],
    *,
    created_at: datetime,
    code_revision: str,
) -> int:
    """Write lines at ``start`` onward; an identical replay succeeds, a different one fails.

    Returns:
        The log length after the appended lines.
    """
    for offset, draft in enumerate(drafts):
        line = MemoryLine(
            schema_version=1,
            created_at=created_at,
            code_revision=code_revision,
            lineage=lineage,
            index=start + offset,
            kind=draft.kind,
            text=draft.text,
            origin=draft.origin,
            derived_from=draft.derived_from,
            depends_on=tuple(sorted(set(draft.depends_on), key=lambda ref: ref.artifact_id)),
        )
        store.write_or_replay(
            artifact_id=line_id(lineage, start + offset),
            artifact_type="memory-line",
            envelope=line,
            envelope_path="line.json",
            envelope_type=MemoryLine,
            files={"line.json": canonical_json_bytes(line)},
        )
    return start + len(drafts)


def read_line(store: ArtifactStore, lineage: str, index: int) -> MemoryLine:
    """Load one log line, verifying its recorded position."""
    line = MemoryLine.model_validate_json(store.read_bytes(line_id(lineage, index), "line.json"))
    if line.lineage != lineage or line.index != index:
        raise ValueError("memory line differs from its log position")
    return line


def read_summary(
    store: ArtifactStore, lineage: str, lo: int, hi: int, generation: int = 0
) -> MemorySummary | None:
    """Load one generation of a block summary, or None when it is not built yet."""
    identity = summary_id(lineage, lo, hi, generation)
    if not store.exists(identity):
        return None
    summary = MemorySummary.model_validate_json(store.read_bytes(identity, "summary.json"))
    if (summary.lineage, summary.lo, summary.hi, summary.generation) != (
        lineage,
        lo,
        hi,
        generation,
    ):
        raise ValueError("memory summary differs from its block position")
    return summary


def cover(total: int, budget: int) -> list[tuple[int, int]]:
    """Tile ``[0, total)`` with aligned power-of-two blocks, at most ``budget`` of them.

    Everything stays verbatim when it fits. Otherwise a block is kept whole when its size is
    small relative to its age (distance from its start to ``total``), with the coarseness
    chosen as the finest that fits the budget; leftover budget refines the newest blocks.
    """
    if total <= 0 or budget <= 0:
        return []
    if total <= budget:
        return [(i, i + 1) for i in range(total)]
    root = 1 << (total - 1).bit_length()

    def tile(alpha: float) -> list[tuple[int, int]]:
        """Tiling for one coarseness factor."""
        out: list[tuple[int, int]] = []
        stack = [(0, root)]
        while stack:
            lo, hi = stack.pop()
            if lo >= total:
                continue
            size = hi - lo
            if size > 1 and (hi > total or size > alpha * (total - lo)):
                mid = (lo + hi) // 2
                stack += [(mid, hi), (lo, mid)]
            else:
                out.append((lo, hi))
        return sorted(out)

    low, high = 0.0, float(root)
    for _ in range(60):
        middle = (low + high) / 2
        if len(tile(middle)) > budget:
            low = middle
        else:
            high = middle
    blocks = tile(high)
    while len(blocks) < budget:
        split = [i for i, (lo, hi) in enumerate(blocks) if hi - lo > 1]
        if not split:
            break
        i = split[-1]
        lo, hi = blocks[i]
        blocks[i : i + 1] = [(lo, (lo + hi) // 2), ((lo + hi) // 2, hi)]
    return blocks


def pending(
    store: ArtifactStore, lineage: str, total: int, generations: Generations | None = None
) -> list[tuple[int, int]]:
    """Unbuilt blocks at their current generation, smallest first (children first)."""
    gens = generations or {}
    todo, size = [], 2
    while size <= total:
        todo += [
            (k * size, (k + 1) * size)
            for k in range(total // size)
            if not store.exists(
                summary_id(
                    lineage, k * size, (k + 1) * size, gens.get((k * size, (k + 1) * size), 0)
                )
            )
        ]
        size *= 2
    return todo


def compression_request(
    store: ArtifactStore,
    lineage: str,
    lo: int,
    hi: int,
    generations: Generations | None = None,
    retired: frozenset[int] = frozenset(),
) -> JsonObject:
    """Build the compression input from raw lines (small blocks) or child summaries.

    Retired lines are left out of a small block's input; larger blocks compress their
    children at the current generation, which already left them out.
    """
    gens = generations or {}
    if hi - lo <= RAW_MAX:
        rows = [read_line(store, lineage, i) for i in range(lo, hi) if i not in retired]
        content: JsonObject = {
            "block": f"#{lo}-{hi - 1}",
            "lines": [{"id": f"#{r.index}", "kind": r.kind, "text": r.text} for r in rows],
        }
    else:
        mid = (lo + hi) // 2
        halves = [
            read_summary(store, lineage, lo, mid, gens.get((lo, mid), 0)),
            read_summary(store, lineage, mid, hi, gens.get((mid, hi), 0)),
        ]
        if any(half is None for half in halves):
            raise ValueError("a block compresses only after both halves are built")
        content = {
            "block": f"#{lo}-{hi - 1}",
            "halves": [
                {"id": f"#{h.lo}-{h.hi - 1}", "text": h.text} for h in halves if h is not None
            ],
        }
    return {
        "messages": [
            {"role": "system", "content": _COMPRESS.format(target=TARGET_BYTES, limit=LINE_BYTES)},
            {"role": "user", "content": json.dumps(content, ensure_ascii=False)},
        ],
        "response_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string", "minLength": 1}},
            "required": ["summary"],
            "additionalProperties": False,
        },
    }


def build_pending(
    store: ArtifactStore,
    lineage: str,
    total: int,
    complete: Callable[[JsonObject, str], JsonObject],
    *,
    limit: int,
    created_at: datetime,
    code_revision: str,
    generations: Generations | None = None,
    retired: frozenset[int] = frozenset(),
) -> tuple[list[str], int]:
    """Spend at most ``limit`` completion calls compressing pending blocks.

    Each block is built at its current generation (``generations``, from the reader's pinned
    memory version); a forgotten block therefore gets a new identity and fresh attempts,
    while older generations stay readable. Lines in ``retired`` are left out of the rebuilt
    content and recorded as the summary's ``omitted`` positions.

    Blocks with fewer earlier rejections go first, and a block is attempted only when both
    halves exist, so a refused block delays its ancestors but never independent blocks.
    Each attempt saves its request before dispatch and its raw proposal after; a restart
    reuses a returned proposal without a call. Only ``ReturnedOutputRejected`` (a returned
    response the completion owner refused) or an out-of-bounds summary closes an attempt as
    rejected; the next attempt carries the refusal and its measured violation. A refusal
    before dispatch (``CompletionNotDispatched``, ``RequestCapacityExceeded``) closes the
    attempt as not dispatched, without feedback and without counting as a rejection, and the
    block waits for the next call. An intent with neither a proposal nor a closing record is
    an ambiguous purchase: that block is skipped (see ``ambiguous_blocks``) and every
    independent block still builds. A block gets at most ``MAX_ATTEMPTS`` attempts per call
    and ``LIFETIME_ATTEMPTS`` rejections overall; nothing is trimmed.

    Returns:
        Built block identities, and how many blocks remain pending.
    """
    gens = generations or {}
    built: list[str] = []
    calls = 0

    def gen(block: Block) -> int:
        """Current generation of one block."""
        return gens.get(block, 0)

    todo = sorted(
        pending(store, lineage, total, gens),
        key=lambda block: (
            _rejections(store, summary_id(lineage, block[0], block[1], gen(block))),
            block[1] - block[0],
        ),
    )
    for lo, hi in todo:
        identity = summary_id(lineage, lo, hi, gen((lo, hi)))
        mid = (lo + hi) // 2
        halves = ((lo, mid), (mid, hi))
        built_halves = [read_summary(store, lineage, a, b, gen((a, b))) for a, b in halves]
        if hi - lo > 2 and any(half is None for half in built_halves):
            continue
        body = compression_request(store, lineage, lo, hi, gens, retired)
        if hi - lo <= RAW_MAX:
            kept = [i for i in range(lo, hi) if i not in retired]
            omitted = tuple(i for i in range(lo, hi) if i in retired)
            if not kept:
                # Nothing left to compress: record the retirement without a model call, so
                # ancestors can still be built.
                _write_summary(
                    store,
                    MemorySummary(
                        schema_version=1,
                        created_at=created_at,
                        code_revision=code_revision,
                        lineage=lineage,
                        lo=lo,
                        hi=hi,
                        text=f"[retired] lines #{lo}-{hi - 1} were superseded",
                        request_sha256=sha256_json(body),
                        children=tuple(_ref(store, line_id(lineage, i)) for i in range(lo, hi)),
                        generation=gen((lo, hi)),
                        omitted=omitted,
                    ),
                )
                built.append(identity)
                continue
            children = tuple(_ref(store, line_id(lineage, i)) for i in kept)
        else:
            children = tuple(_ref(store, summary_id(lineage, a, b, gen((a, b)))) for a, b in halves)
            omitted = tuple(i for half in built_halves if half is not None for i in half.omitted)
        attempt, feedback = 1, cast("JsonObject | None", None)
        while _closed(store, f"{identity}-attempt-{attempt}"):
            if store.exists(f"{identity}-attempt-{attempt}-rejected"):
                feedback = json.loads(
                    store.read_bytes(f"{identity}-attempt-{attempt}-rejected", "record.json")
                )
            attempt += 1
        for _ in range(MAX_ATTEMPTS):
            if _rejections(store, identity) >= LIFETIME_ATTEMPTS:
                break
            base = f"{identity}-attempt-{attempt}"
            request = body if feedback is None else _with_feedback(body, feedback)
            if store.exists(base + "-proposal"):
                # A saved answer belongs to the request that was saved and sent, which may
                # differ from today's rendering; certify that exact request.
                intent = json.loads(store.read_bytes(base + "-intent", "record.json"))
                request = intent["request"]
                proposal = json.loads(store.read_bytes(base + "-proposal", "record.json"))
            elif store.exists(base + "-intent"):
                break  # ambiguous purchase: leave this block, keep building the others
            else:
                if calls >= limit:
                    return built, len(pending(store, lineage, total, gens))
                _save(store, base + "-intent", {"request": request}, created_at, code_revision)
                calls += 1
                try:
                    proposal = complete(request, base)
                except ReturnedOutputRejected as error:
                    feedback = cast(
                        JsonObject, {"reason": str(error)[:2000], "text": None, "bytes": None}
                    )
                    _save(store, base + "-rejected", feedback, created_at, code_revision)
                    attempt += 1
                    continue
                except (CompletionNotDispatched, RequestCapacityExceeded) as error:
                    _save(
                        store,
                        base + "-not-dispatched",
                        {"reason": str(error)[:2000]},
                        created_at,
                        code_revision,
                    )
                    break  # nothing was bought; this block waits for a later call
                _save(store, base + "-proposal", proposal, created_at, code_revision)
            text = proposal.get("summary")
            try:
                summary = MemorySummary(
                    schema_version=1,
                    created_at=created_at,
                    code_revision=code_revision,
                    lineage=lineage,
                    lo=lo,
                    hi=hi,
                    text=text if isinstance(text, str) else "",
                    request_sha256=sha256_json(request),
                    children=children,
                    generation=gen((lo, hi)),
                    omitted=omitted,
                )
            except ValueError as error:
                feedback = cast(
                    JsonObject,
                    {
                        "reason": str(error),
                        "text": text if isinstance(text, str) else None,
                        "bytes": len(text.encode()) if isinstance(text, str) else None,
                    },
                )
                _save(store, base + "-rejected", feedback, created_at, code_revision)
                attempt += 1
                continue
            _write_summary(store, summary)
            built.append(identity)
            break
    return built, len(pending(store, lineage, total, gens))


def refused_blocks(
    store: ArtifactStore, lineage: str, total: int, generations: Generations | None = None
) -> list[str]:
    """Pending blocks that exhausted ``LIFETIME_ATTEMPTS`` and need explicit attention."""
    gens = generations or {}
    return [
        f"#{lo}-{hi - 1}"
        for lo, hi in pending(store, lineage, total, gens)
        if _rejections(store, summary_id(lineage, lo, hi, gens.get((lo, hi), 0)))
        >= LIFETIME_ATTEMPTS
    ]


def overview(
    store: ArtifactStore,
    lineage: str,
    total: int,
    budget: int,
    *,
    admits: Callable[[ArtifactInput], bool] | None = None,
    admits_line: Callable[[MemoryLine], bool] | None = None,
    generations: Generations | None = None,
    pinned: Container[tuple[int, int, int]] | None = None,
) -> list[JsonObject]:
    """Render at most ``budget`` rows of the first ``total`` lines, oldest first.

    Summaries are read at the generations pinned by the reader's memory version. When
    ``pinned`` is given, only those ``(lo, hi, generation)`` summaries may be shown, so a
    summary compressed after the version was pinned never changes what that version shows.
    A summary that omitted a line is shown only if this reader excludes that line too.

    A line appears only if it is admitted: by default its source and every dependency pass
    ``admits``, and ``admits_line`` may additionally exclude it (for example, retired
    hypotheses). A summary appears only if every leaf line beneath it is admitted;
    otherwise, or when it is not built yet, the block opens into its halves. Rows are
    collected newest first; if opened blocks would exceed the budget, the older remainder
    is replaced by one explicit ``omitted`` row stating the line range, never its content.
    The marker counts toward the budget: the oldest collected row gives way to it.
    """
    source_ok = admits or (lambda source: True)

    def line_ok(line: MemoryLine) -> bool:
        """Source, dependencies and caller policy all admit the line."""
        sources = (line.derived_from, *line.depends_on)
        return all(source_ok(s) for s in sources) and (admits_line is None or admits_line(line))

    rows: list[JsonObject] = []
    spans: list[tuple[int, int]] = []
    stack = list(cover(total, budget))
    while stack:
        if len(rows) >= budget:
            if not any(lo < total for lo, _ in stack):
                break
            stack.append(spans.pop())  # the oldest shown row makes room for the marker
            rows.pop()
            omitted = sum(min(hi, total) - lo for lo, hi in stack if lo < total)
            end = max(min(hi, total) for lo, hi in stack if lo < total)
            rows.append(
                {
                    "id": f"#0-{end - 1}",
                    "kind": "omitted",
                    "count": omitted,
                    "authority": "Older memory not shown because its compression is "
                    "incomplete or excluded; exact records remain retrievable",
                }
            )
            break
        lo, hi = stack.pop()
        if lo >= total:
            continue
        if hi - lo == 1:
            line = read_line(store, lineage, lo)
            if not line_ok(line):
                continue
            spans.append((lo, hi))
            rows.append(
                {
                    "id": f"#{lo}",
                    "kind": line.kind,
                    "text": line.text,
                    "origin": line.origin,
                    "authority": AUTHORITY[line.kind],
                }
            )
            continue
        generation = (generations or {}).get((lo, hi), 0)
        summary = (
            read_summary(store, lineage, lo, hi, generation)
            if pinned is None or (lo, hi, generation) in pinned
            else None
        )
        if (
            summary is None
            or not all(line_ok(x) for x in covered_lines(store, summary))
            or any(line_ok(read_line(store, lineage, i)) for i in summary.omitted)
        ):
            mid = (lo + hi) // 2
            stack += [(lo, mid), (mid, hi)]
            continue
        spans.append((lo, hi))
        rows.append(
            {
                "id": f"#{lo}-{hi - 1}",
                "kind": "summary",
                "text": summary.text,
                "authority": AUTHORITY["summary"],
            }
        )
    return list(reversed(rows))


def overview_sources(
    store: ArtifactStore,
    lineage: str,
    rows: Sequence[JsonObject],
    generations: Generations | None = None,
) -> tuple[ArtifactInput, ...]:
    """Every source trajectory, including dependencies, behind the rendered overview rows."""
    refs: set[ArtifactInput] = set()
    for row in rows:
        lo, _, hi = str(row["id"]).lstrip("#").partition("-")
        if row.get("kind") == "omitted":
            continue
        if hi:
            block = (int(lo), int(hi) + 1)
            summary = read_summary(store, lineage, *block, (generations or {}).get(block, 0))
            if summary is None:
                raise ValueError(f"overview summary {row['id']} is not built")
            refs |= summary_sources(store, summary)
        else:
            line = read_line(store, lineage, int(lo))
            refs |= {line.derived_from, *line.depends_on}
    return tuple(sorted(refs, key=lambda ref: ref.artifact_id))


def summary_lines(store: ArtifactStore, summary: MemorySummary) -> tuple[MemoryLine, ...]:
    """Resolve every leaf line a summary was built from, through its recorded children."""
    lines: list[MemoryLine] = []
    for child in summary.children:
        if artifact_input(store.read(child.artifact_id).manifest) != child:
            raise ValueError("summary child artifact changed")
        kind = store.read(child.artifact_id).manifest.artifact_type
        if kind == "memory-line":
            lines.append(
                MemoryLine.model_validate_json(store.read_bytes(child.artifact_id, "line.json"))
            )
        elif kind == "memory-summary":
            nested = MemorySummary.model_validate_json(
                store.read_bytes(child.artifact_id, "summary.json")
            )
            lines.extend(summary_lines(store, nested))
        else:
            raise ValueError(f"summary child {child.artifact_id} is not a memory artifact")
    return tuple(lines)


def covered_lines(store: ArtifactStore, summary: MemorySummary) -> tuple[MemoryLine, ...]:
    """Leaf lines whose content the summary text was built from (omitted lines excluded)."""
    omitted = set(summary.omitted)
    return tuple(line for line in summary_lines(store, summary) if line.index not in omitted)


def _write_summary(store: ArtifactStore, summary: MemorySummary) -> None:
    """Publish one accepted summary at its generation's identity; replay must be exact."""
    store.write_or_replay(
        artifact_id=summary_id(summary.lineage, summary.lo, summary.hi, summary.generation),
        artifact_type="memory-summary",
        envelope=summary,
        envelope_path="summary.json",
        envelope_type=MemorySummary,
        files={"summary.json": canonical_json_bytes(summary)},
    )


def _rejections(store: ArtifactStore, identity: str) -> int:
    """Number of rejected attempts recorded for one block; not-dispatched ones do not count."""
    count, attempt = 0, 1
    while _closed(store, f"{identity}-attempt-{attempt}"):
        count += store.exists(f"{identity}-attempt-{attempt}-rejected")
        attempt += 1
    return count


def _closed(store: ArtifactStore, base: str) -> bool:
    """Whether one attempt ended in a rejection or a refusal before dispatch."""
    return store.exists(base + "-rejected") or store.exists(base + "-not-dispatched")


def ambiguous_blocks(
    store: ArtifactStore, lineage: str, total: int, generations: Generations | None = None
) -> list[str]:
    """Pending blocks whose latest attempt has an intent but no proposal or closing record.

    Such a request may or may not have been bought; it is never retried automatically.
    """
    gens = generations or {}
    found = []
    for lo, hi in pending(store, lineage, total, gens):
        identity = summary_id(lineage, lo, hi, gens.get((lo, hi), 0))
        attempt = 1
        while _closed(store, f"{identity}-attempt-{attempt}"):
            attempt += 1
        base = f"{identity}-attempt-{attempt}"
        if store.exists(base + "-intent") and not store.exists(base + "-proposal"):
            found.append(f"#{lo}-{hi - 1}")
    return found


def summary_sources(store: ArtifactStore, summary: MemorySummary) -> frozenset[ArtifactInput]:
    """Every source and dependency beneath a summary, resolved through its children."""
    sources: set[ArtifactInput] = set()
    for line in covered_lines(store, summary):
        sources.add(line.derived_from)
        sources.update(line.depends_on)
    return frozenset(sources)


def _ref(store: ArtifactStore, identity: str) -> ArtifactInput:
    """Exact manifest reference to a persisted memory artifact."""
    return artifact_input(store.read(identity).manifest)


def _with_feedback(body: JsonObject, feedback: JsonObject) -> JsonObject:
    """Retry request: the rejected output, then its measured violation. Nothing is trimmed."""
    messages = body["messages"]
    if not isinstance(messages, list):
        raise ValueError("compression request messages must be a list")
    measured = (
        f"{feedback['bytes']} bytes; the limit is {LINE_BYTES} bytes on one line"
        if feedback.get("bytes") is not None
        else f"refused ({feedback.get('reason')}); the limit is {LINE_BYTES} bytes on one line"
    )
    return {
        **body,
        "messages": [
            *messages,
            {"role": "assistant", "content": str(feedback.get("text") or "")},
            {
                "role": "user",
                "content": f"Rejected: that summary was {measured}. Rewrite it as one line of at "
                f"most {TARGET_BYTES} bytes, keeping only what has lasting effect. "
                "Do not invent anything.",
            },
        ],
    }


def _save(
    store: ArtifactStore, identity: str, payload: JsonObject, created_at: datetime, revision: str
) -> None:
    """Publish one immutable compression receipt."""
    store.write_json(
        artifact_id=identity,
        artifact_type="memory-compression",
        envelope=ArtifactEnvelope(schema_version=1, created_at=created_at, code_revision=revision),
        files={"record.json": payload},
    )
