"""Cross-session knowledge base: the world model's persistent, human-editable memory.

A `KnowledgeBase` is a directory of plain markdown files under the model artifact
(`models/<name>/knowledge/`) holding the environment's *canonical* facts — entities, business
rules, response schemas, and the state-dependent gates (auth, availability, preconditions) an LLM
env otherwise guesses wrong. It is:

- **seeded at build time** from TRAIN traces only (`seed_knowledge`, an LLM extraction pass),
- **read at serve time** — rendered whole (size-budgeted) into the env prompt's KNOWLEDGE BASE
  section (`wmh.core.render.build_env_prompt`),
- **written at serve time** — the env's `kb_note` contract field appends to `learned.md` and the
  grounder caches web results in `grounded.md`; the seeded files are never auto-modified,
- **edited by humans** — it's just markdown; open the folder in any editor.

The session scratchpad (`EnvState.scratchpad`) stays session-local; this store is what persists
ACROSS sessions. Models without a `knowledge/` directory load and serve exactly as before.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from pydantic import BaseModel, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.core.render import render_demo
from wmh.core.types import Trace
from wmh.providers.base import Message, Provider

# Build-time seeded files (never auto-modified after seeding; humans edit freely).
SEEDED_FILES = ("rules.md", "entities.md", "schemas.md")
# Serve-time append targets: env kb_notes and grounder cache. Auto-appended, size-capped.
LEARNED_FILE = "learned.md"
GROUNDED_FILE = "grounded.md"

# Rendering budget (chars) for the KNOWLEDGE BASE prompt section. The KB is curated facts, not a
# record dump — tau-bench's full 3.3MB DB per step is exactly the failure mode this cap prevents.
DEFAULT_RENDER_BUDGET = 24_000
# Per-file cap (chars) for the auto-appended files, so serve-time writes can't grow unboundedly.
DEFAULT_APPEND_CAP = 50_000

_TRUNCATION_MARKER = "\n[KNOWLEDGE BASE TRUNCATED: over render budget — curate the files]\n"


class KnowledgeBase:
    """A directory of markdown files acting as the env's cross-session memory.

    Missing directory == empty knowledge base; nothing is created until the first write, so
    models built before this feature (no `knowledge/` dir) are served unchanged.
    """

    def __init__(self, directory: str | Path, *, append_cap: int = DEFAULT_APPEND_CAP) -> None:
        self.directory = Path(directory)
        self._append_cap = append_cap
        # Serializes read-check-append against itself: FastAPI sync handlers run in a thread
        # pool, so two sessions stepping the SAME served model can race append_learned /
        # append_grounded and double-append (or blow past the cap). In-process only — one
        # server process owns an artifact's knowledge/ dir; cross-process co-serving of one
        # artifact is out of scope.
        self._write_lock = threading.Lock()

    @property
    def is_empty(self) -> bool:
        """True when no markdown file has any content."""
        return not any(content.strip() for content in self.files().values())

    def files(self) -> dict[str, str]:
        """Return {file name: content} for every markdown file, in sorted-name order."""
        if not self.directory.is_dir():
            return {}
        with self._write_lock:  # a torn read of a file mid-write must not reach the prompt
            return {
                path.name: path.read_text(encoding="utf-8")
                for path in sorted(self.directory.glob("*.md"))
            }

    def write_file(self, name: str, content: str) -> None:
        """Create/replace one markdown file (seeding, HTTP edit surface)."""
        _validate_file_name(name)
        with self._write_lock:  # an HTTP edit racing a stepping session's append
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / name).write_text(content, encoding="utf-8")

    def render(self, budget: int = DEFAULT_RENDER_BUDGET) -> str:
        """Render the whole KB for the env prompt: `## <file>` sections, sorted, budget-capped.

        Deterministic: what a human sees in the files is exactly what the env reads. Over-budget
        content is cut with a loud marker rather than silently — an oversized KB is a curation
        problem the user should see, not hidden lossage.
        """
        sections = [
            f"## {name}\n{content.strip()}"
            for name, content in self.files().items()
            if content.strip()
        ]
        text = "\n\n".join(sections)
        if len(text) <= budget:
            return text
        keep = max(budget - len(_TRUNCATION_MARKER), 0)
        return (text[:keep] + _TRUNCATION_MARKER)[:budget]

    def append_learned(self, fact: str, *, provenance: str) -> bool:
        """Append one cross-session fact (env `kb_note`) to `learned.md` with provenance.

        Returns False (and writes nothing) when the exact fact is already recorded or the file is
        at its cap. Only `learned.md` is ever auto-written — seeded files and human edits are
        never clobbered.
        """
        fact = fact.strip()
        if not fact:
            return False
        with self._write_lock:
            existing = self._read(LEARNED_FILE)
            # Exact-fact dedupe: a raw substring test treated any new fact that PREFIXES an
            # existing entry as already recorded, silently dropping distinct general facts.
            recorded = {
                line[2:].split("  <!--", 1)[0].strip()
                for line in existing.splitlines()
                if line.startswith("- ")
            }
            if fact in recorded:
                return False
            if len(existing) > self._append_cap:
                return False
            entry = f"- {fact}  <!-- {provenance} -->\n"
            self._append(LEARNED_FILE, entry)
        return True

    def append_grounded(self, query: str, results_text: str) -> None:
        """Cache one grounder result under `grounded.md` so a query is searched at most once."""
        entry = f"### query: {query.strip()}\n{results_text.strip()}\n\n"
        with self._write_lock:
            # Same query grounded by two racing sessions: keep the first cache entry.
            if f"### query: {query.strip()}\n" in self._read(GROUNDED_FILE):
                return
            self._append(GROUNDED_FILE, entry)

    def lookup_grounded(self, query: str) -> str | None:
        """Return the cached results for `query`, or None on a cache miss."""
        content = self._read(GROUNDED_FILE)
        header = f"### query: {query.strip()}\n"
        start = content.find(header)
        if start == -1:
            return None
        body_start = start + len(header)
        end = content.find("### query: ", body_start)
        return content[body_start : end if end != -1 else len(content)].strip()

    def _read(self, name: str) -> str:
        path = self.directory / name
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def _append(self, name: str, text: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / name).open("a", encoding="utf-8") as fh:
            fh.write(text)


def _validate_file_name(name: str) -> None:
    if name != Path(name).name or not name.endswith(".md"):
        raise ValueError(
            f"knowledge file name must be a bare '*.md' name (got {name!r}); "
            "the knowledge base is a flat folder of markdown files"
        )


class _Extraction(BaseModel):
    """The seeding LLM's per-chunk output: full updated content of each seeded file."""

    rules: str = ""
    entities: str = ""
    schemas: str = ""


_SEED_SYSTEM_PROMPT = """You are building the canonical KNOWLEDGE BASE for a simulated \
environment reconstructed from real agent traces. Extract only DURABLE, CROSS-SESSION facts the \
environment must stay consistent about:

- rules: business rules and STATE-DEPENDENT GATES the environment ITSELF enforces — auth \
requirements, availability checks, preconditions, completion/cancellation rules. These decide \
success vs. error. Distinguish carefully: if traces show a tool executing successfully despite \
a policy the agent was told to follow, that policy is NOT an environment gate — record it as \
"agent policy (NOT enforced: tools execute mechanically)". Also record SYSTEM LIMITS evidenced \
in traces with their observed values: command timeouts, rate limits, output truncation, caps.
- entities: canonical entities that exist (ids, names, relations) — stated generally, never \
per-session conversation details.
- schemas: tool/API response shapes, field names, and exact error formats/messages.

Do NOT copy session-specific events ("the agent booked X") — only what is true of the \
environment itself. Prefer terse markdown bullet lists. You are shown the CURRENT knowledge base \
and a NEW trace excerpt; return the UPDATED full content of all three files (carry existing \
facts forward, merge and dedupe, drop nothing that is still true).

Respond with ONLY a JSON object: {"rules": "<markdown>", "entities": "<markdown>", \
"schemas": "<markdown>"}"""

# Chars of rendered trace text per extraction call. Together with `max_calls` this bounds the
# build-time seeding cost regardless of corpus size.
_SEED_CHUNK_CHARS = 40_000


def seed_knowledge(
    kb: KnowledgeBase,
    train_traces: list[Trace],
    provider: Provider,
    *,
    max_calls: int = 8,
    reporter_note: list[str] | None = None,
) -> None:
    """Seed `kb` from TRAIN traces via chunked knowledge-accumulation extraction.

    Renders the corpus steps into text chunks and folds each chunk into the running KB (the same
    accumulate-and-carry-forward pattern GEPA's reflection uses), writing `SEEDED_FILES` after
    every successful call so a partial run still leaves a usable KB. `max_calls` is the hard cost
    bound: chunks beyond it are skipped (coverage saturates quickly — rules/schemas repeat across
    traces). Eval-integrity note: callers must pass TRAIN traces only (mirrors
    `wmh.retrieval.leakfree`), so a KB used during eval can never contain a held-out answer.
    """
    chunks = _corpus_chunks(train_traces)
    skipped = max(len(chunks) - max_calls, 0)
    current = _Extraction(
        rules=kb.files().get("rules.md", ""),
        entities=kb.files().get("entities.md", ""),
        schemas=kb.files().get("schemas.md", ""),
    )
    for chunk in chunks[:max_calls]:
        user = (
            f"CURRENT KNOWLEDGE BASE:\n"
            f"rules.md:\n{current.rules or '(empty)'}\n\n"
            f"entities.md:\n{current.entities or '(empty)'}\n\n"
            f"schemas.md:\n{current.schemas or '(empty)'}\n\n"
            f"NEW TRACE EXCERPT:\n{chunk}"
        )
        completion = provider.complete(_SEED_SYSTEM_PROMPT, [Message(role="user", content=user)])
        extraction = _parse_extraction(completion.text)
        if extraction is None:
            continue  # off-contract reply: keep accumulating from the current state
        current = extraction
        kb.write_file("rules.md", current.rules)
        kb.write_file("entities.md", current.entities)
        kb.write_file("schemas.md", current.schemas)
    if skipped and reporter_note is not None:
        reporter_note.append(
            f"knowledge seeding: {skipped} corpus chunk(s) beyond the {max_calls}-call budget "
            "were not read"
        )


def seeded_knowledge_text(
    train_traces: list[Trace], provider: Provider, *, max_calls: int = 8
) -> str | None:
    """Seed an ephemeral KB from TRAIN traces and return its rendered text (None when empty).

    For eval/research contexts that need train-derived knowledge in the prompt without touching
    any model artifact: the KB lives in a temp dir for the duration of seeding only, so nothing a
    serve session ever wrote (learned/grounded facts) can leak into a scored run.
    """
    with tempfile.TemporaryDirectory(prefix="wmh-kb-") as tmp:
        kb = KnowledgeBase(Path(tmp))
        seed_knowledge(kb, train_traces, provider, max_calls=max_calls)
        return kb.render() or None


def _parse_extraction(text: str) -> _Extraction | None:
    raw = extract_json_object(text)
    if raw is None:
        return None
    try:
        return _Extraction.model_validate_json(raw)
    except ValidationError:
        return None


def _corpus_chunks(traces: list[Trace]) -> list[str]:
    """Render every step once (canonical `render_demo`) and pack into char-bounded chunks."""
    chunks: list[str] = []
    buffer: list[str] = []
    size = 0
    for trace in traces:
        for step in trace.steps:
            text = render_demo(step)
            if size + len(text) > _SEED_CHUNK_CHARS and buffer:
                chunks.append("\n\n".join(buffer))
                buffer, size = [], 0
            buffer.append(text)
            size += len(text)
    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks
