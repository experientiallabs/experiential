"""Ingest: build a harness from an existing agent repo by **body mapping**.

An agent you already have — a repo of prompts, loop code, tool definitions, and configs — is the
agent's *body*. Body mapping walks that body and turns it into a `HarnessDoc`: every relevant
textual file becomes a pathful CODE surface (directory hierarchy preserved in `Surface.path`), and
an LLM writes a **harnessdoc** per file (`Surface.doc`) describing what the file is and how it
serves the agent. Inclusion is deliberately zealous: anything textual that survives the exclusion
rules is mapped — when in doubt, a file goes in with a harnessdoc rather than being dropped.

The built document also carries:
- `prompt:core` — an LLM-written overview of the agent (what it is, how its body is organized),
- `BODYMAP.md` (a pathful CODE surface) — the index: one line per mapped file, plus what was
  skipped and why,
- the default tool policy and loop params, so the doc validates and renders like any other.

An ingested harness is a *representation and editing substrate*: it does not execute the repo's
own agent loop (no runtime knows how to run an arbitrary repo). It is the v0 a platform agent is
born with, the thing the harness editor edits, and the context `wmh harness create` mutates.

Callers own provider metering (wrap with `MeteredProvider`) and checkout acquisition: the CLI
clones or walks a directory; the platform fetches a tarball. Everything here is pure library code.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field

from wmh.core.parsing import extract_json_object
from wmh.harness.doc import (
    DEFAULT_TEMPERATURE,
    MAX_TURNS_ID,
    TEMPERATURE_ID,
    TOOL_POLICY_ID,
    HarnessDoc,
    Surface,
    SurfaceKind,
)
from wmh.harness.runtime import DEFAULT_MAX_TURNS
from wmh.harness.tools import DEFAULT_TOOLS
from wmh.providers.base import Message, Provider

DEFAULT_MAX_FILE_BYTES = 65_536
DEFAULT_MAX_TOTAL_BYTES = 2_500_000

BODYMAP_PATH = "BODYMAP.md"
_BODYMAP_SLUG = "bodymap-md"

# Paths the harness store's render owns (SYSTEM.md, config.toml, ...): a repo
# file at one of these paths keeps its content but moves aside (hash suffix),
# so the rendered bundle never carries two files with one path.
RESERVED_RENDER_PATHS = frozenset({"SYSTEM.md", "config.toml", "runtime.py", "doc.json"})

# Directories that never contain agent body: VCS internals, dependency trees, caches, build
# output. Pruned by basename at any depth.
EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
        ".idea",
        ".vscode",
        "dist",
        "build",
        ".next",
        ".turbo",
        "target",
        ".cache",
        ".terraform",
        ".wmh",  # a repo that used wmh locally carries artifacts, not body
    }
)

# Generated files with no mapping value (lockfiles, OS litter).
EXCLUDED_FILES = frozenset(
    {
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "bun.lockb",
        "uv.lock",
        "poetry.lock",
        "Pipfile.lock",
        "Cargo.lock",
        "composer.lock",
        "Gemfile.lock",
        "go.sum",
        ".DS_Store",
    }
)

# Files that plausibly hold credentials are never ingested, zealousness notwithstanding: a
# harness doc travels (registry push, platform rows, editor payloads) and must not carry secrets.
SECRET_FILE_PATTERNS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "*.keystore",
)

_OVERVIEW_CLIP_CHARS = 6_000
_FILE_PROMPT_CLIP_CHARS = 12_000
_TREE_CLIP_LINES = 400

INGEST_SYSTEM = """You are mapping the body of an existing software agent: its repo of prompts, \
loop code, tool definitions, and configuration. For the ONE file you are shown, write its \
harnessdoc. Reply with ONLY a JSON object, no prose:
{"role": "<one line: what this file is, <= 100 chars>",
 "doc": "<2-5 sentences: what the file contains, the key symbols/sections, and how it serves \
the agent>"}
Ground everything in the file content shown; never invent symbols or behavior."""

OVERVIEW_SYSTEM = """You are mapping the body of an existing software agent: its repo of \
prompts, loop code, tool definitions, and configuration. From the file tree and excerpts shown, \
write a compact overview of this agent for someone about to work on it. Reply with ONLY a JSON \
object, no prose:
{"overview": "<1-3 paragraphs: what the agent is and does, how the repo is organized, where the \
prompts / loop / tools live>"}
Ground everything in what is shown; never invent files or behavior."""


class _GitignoreMatcher(BaseModel):
    """One compiled .gitignore rule: a regex over the relative posix path, plus its kind.

    We deliberately implement only the conservative subset git uses that cannot silently drop
    body files: no negations (a whitelist .gitignore disables gitignore handling entirely, see
    `_gitignore_patterns`), no per-parent-directory .gitignore files, no character classes. A
    root-anchored pattern (one containing a non-trailing '/') matches the full relative path; a
    bare pattern matches any path segment at any depth; a trailing '/' marks a directory prefix.
    """

    model_config = {"arbitrary_types_allowed": True}

    regex: re.Pattern[str]

    def matches(self, rel: str) -> bool:
        return self.regex.match(rel) is not None


class RepoFile(BaseModel):
    """One textual file of the agent's body, ready to map."""

    path: str  # relative posix path inside the repo
    content: str


class SkippedFile(BaseModel):
    """One file body mapping left out, and exactly why (skips are reported, never silent)."""

    path: str
    reason: str


class CollectedRepo(BaseModel):
    """The walk result: what will be mapped and what was skipped."""

    files: list[RepoFile]
    skipped: list[SkippedFile]


class FileDoc(BaseModel):
    """The harnessdoc body mapping wrote for one file."""

    path: str
    role: str  # one line
    doc: str  # 2-5 sentences


class BodyMap(BaseModel):
    """The body-mapping output: the agent overview plus one harnessdoc per mapped file."""

    overview: str
    file_docs: list[FileDoc]
    unmapped: list[str] = Field(default_factory=list)  # files whose model reply was unusable

    def doc_for(self, path: str) -> FileDoc | None:
        for file_doc in self.file_docs:
            if file_doc.path == path:
                return file_doc
        return None


def collect_repo_files(
    root: Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    extra_excludes: tuple[str, ...] = (),
) -> CollectedRepo:
    """Walk a checkout and collect every textual file worth mapping, in deterministic path order.

    Exclusions: the fixed dir/file lists, secret-shaped files, extra caller globs, patterns from
    the repo's root `.gitignore` (conservative gitignore subset, negations disable it entirely),
    binaries (NUL sniff or undecodable UTF-8), files over `max_file_bytes`, and everything past
    `max_total_bytes` cumulative content (paths sort first, so the cut is deterministic). Every
    exclusion is recorded with its reason, including symlinks and pruned directories from the walk.
    """
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    ignore_matchers = _gitignore_patterns(root)
    files: list[RepoFile] = []
    walked, skipped = _walk_sorted(root)
    total = 0
    for path in walked:
        rel = path.relative_to(root).as_posix()
        reason = _exclusion_reason(rel, path.name, ignore_matchers, extra_excludes)
        if reason is not None:
            skipped.append(SkippedFile(path=rel, reason=reason))
            continue
        size = path.stat().st_size
        if size > max_file_bytes:
            skipped.append(
                SkippedFile(path=rel, reason=f"over per-file cap ({size} > {max_file_bytes} bytes)")
            )
            continue
        raw = path.read_bytes()
        if b"\x00" in raw[:8192]:
            skipped.append(SkippedFile(path=rel, reason="binary (NUL bytes)"))
            continue
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped.append(SkippedFile(path=rel, reason="binary (not UTF-8)"))
            continue
        if total + len(raw) > max_total_bytes:
            skipped.append(
                SkippedFile(path=rel, reason=f"over total budget ({max_total_bytes} bytes)")
            )
            continue
        total += len(raw)
        files.append(RepoFile(path=rel, content=content))
    skipped.sort(key=lambda s: s.path)
    return CollectedRepo(files=files, skipped=skipped)


def body_map(
    collected: CollectedRepo,
    provider: Provider,
    *,
    name: str,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> BodyMap:
    """Write the harnessdocs: one LLM call per file plus one overview call.

    A file whose reply is unusable (no JSON, wrong shape) still ships — with a placeholder
    harnessdoc and its path recorded in `unmapped` — so one flaky reply costs one annotation,
    not the ingest. Deterministic prompts; sampling is the only nondeterminism.
    """
    file_docs: list[FileDoc] = []
    unmapped: list[str] = []
    total = len(collected.files)
    for index, repo_file in enumerate(collected.files, start=1):
        if on_progress is not None:
            on_progress(index, total, repo_file.path)
        parsed = _complete_json(provider, INGEST_SYSTEM, _file_prompt(repo_file), _RawFileDoc)
        if parsed is None:
            unmapped.append(repo_file.path)
            file_docs.append(
                FileDoc(
                    path=repo_file.path,
                    role="(unmapped: unusable model reply)",
                    doc="(no harnessdoc: the mapping model returned an unusable reply)",
                )
            )
            continue
        # The mapping model's output travels into prompt:core, BODYMAP.md, and (transitively) the
        # mutate proposal prompt, so a hostile repo could poison those with fence injection or
        # unbounded length. Sanitize at this boundary before the text is ever embedded.
        file_docs.append(
            FileDoc(
                path=repo_file.path,
                role=_clean_role(parsed.role),
                doc=_clean_doc(parsed.doc),
            )
        )
    overview = _complete_json(
        provider, OVERVIEW_SYSTEM, _overview_prompt(name, collected, file_docs), _RawOverview
    )
    overview_text = (
        _clean_overview(overview.overview)
        if overview is not None
        else f"{name}: an agent ingested from an existing repo ({total} files mapped)."
    )
    return BodyMap(overview=overview_text, file_docs=file_docs, unmapped=unmapped)


def build_ingest_doc(
    name: str, collected: CollectedRepo, mapping: BodyMap, *, source: str
) -> HarnessDoc:
    """Assemble the validated `HarnessDoc`: prompt:core, BODYMAP.md, and one CODE surface per file.

    Each file keeps its directory hierarchy in `Surface.path` (sanitized where the safe-path rule
    requires, with the original path noted in the harnessdoc) and carries its harnessdoc in
    `Surface.doc`. Defaults (tool policy, loop params) are explicit so the rendered `config.toml`
    shows them.
    """
    taken_slugs = {_BODYMAP_SLUG}
    taken_paths = {BODYMAP_PATH, *RESERVED_RENDER_PATHS}
    surfaces = [
        Surface(
            id="prompt:core",
            kind=SurfaceKind.PROMPT,
            content=(
                f"{mapping.overview.strip()}\n\n"
                f"This agent was ingested from {source}. Its body (the repo's files) is carried "
                f"as code surfaces with their original paths; the body map index in "
                f"{BODYMAP_PATH} lists every file with its role."
            ),
        ),
        Surface(
            id=f"code:{_BODYMAP_SLUG}",
            kind=SurfaceKind.CODE,
            path=BODYMAP_PATH,
            content=_render_bodymap(name, collected, mapping, source=source),
            doc="The body map index: every mapped file with its one-line role, plus skips.",
        ),
        Surface(id=TOOL_POLICY_ID, kind=SurfaceKind.TOOL_POLICY, content="\n".join(DEFAULT_TOOLS)),
        Surface(id=MAX_TURNS_ID, kind=SurfaceKind.PARAM, content=str(DEFAULT_MAX_TURNS)),
        Surface(id=TEMPERATURE_ID, kind=SurfaceKind.PARAM, content=str(DEFAULT_TEMPERATURE)),
    ]
    for repo_file in collected.files:
        slug = slug_for_path(repo_file.path, taken_slugs)
        taken_slugs.add(slug)
        safe_path, changed = sanitize_path(repo_file.path)
        safe_path = _disambiguate_path(safe_path, repo_file.path, taken_paths)
        taken_paths.add(safe_path)
        file_doc = mapping.doc_for(repo_file.path)
        doc_text = file_doc.doc if file_doc is not None else "(no harnessdoc)"
        if changed or safe_path != repo_file.path:
            doc_text = f"Original path: {repo_file.path}. {doc_text}"
        surfaces.append(
            Surface(
                id=f"code:{slug}",
                kind=SurfaceKind.CODE,
                path=safe_path,
                content=repo_file.content,
                doc=doc_text,
            )
        )
    return HarnessDoc(name=name, surfaces=surfaces)


def slug_for_path(path: str, taken: set[str]) -> str:
    """A valid surface-id slug for a repo path; collisions get a short content-free hash suffix.

    The mapping is lossy by design (ids are flat kebab); `Surface.path` keeps the real location.
    """
    slug = _kebab(path)
    if slug not in taken:
        return slug
    return f"{slug}-{_path_digest(path)}"


def sanitize_path(path: str) -> tuple[str, bool]:
    """Rewrite a repo path so it satisfies the surface safe-path rule; report whether it changed.

    Characters outside `[A-Za-z0-9._-]` become `_` per segment (e.g. `app/[id]/page.tsx` ->
    `app/_id_/page.tsx`); `.` and `..` segments become `_`. The original path is preserved in the
    harnessdoc by the caller.
    """
    segments = []
    changed = False
    for segment in path.split("/"):
        clean = "".join(
            c if (c.isascii() and (c.isalnum() or c in "._-")) else "_" for c in segment
        )
        if not clean or clean in {".", ".."}:
            clean = "_"
        changed = changed or clean != segment
        segments.append(clean)
    return "/".join(segments), changed


def _disambiguate_path(safe_path: str, original: str, taken: set[str]) -> str:
    if safe_path not in taken:
        return safe_path
    directory, sep, basename = safe_path.rpartition("/")
    stem, dot, ext = basename.rpartition(".")
    digest = _path_digest(original)
    # `stem` guards dotfiles: '.gitignore'.rpartition('.') == ('', '.', 'gitignore'), so an empty
    # stem means there is no real extension to preserve and we suffix the whole basename instead.
    if dot and stem:
        moved = f"{stem}-{digest}.{ext}"
    else:
        moved = f"{basename}-{digest}"
    return f"{directory}{sep}{moved}"


def _kebab(text: str) -> str:
    out: list[str] = []
    for c in text.lower():
        if c.isascii() and c.isalnum():
            out.append(c)
        elif not out or out[-1] != "-":
            out.append("-")
    slug = "".join(out).strip("-")
    return slug or "file"


def _path_digest(path: str) -> str:
    return hashlib.blake2b(path.encode("utf-8"), digest_size=3).hexdigest()


_ROLE_MAX_CHARS = 120
_DOC_MAX_CHARS = 600
_OVERVIEW_MAX_CHARS = 4_000


def _clean_role(text: str) -> str:
    """One-line role: drop backticks, collapse all whitespace, cap at `_ROLE_MAX_CHARS`."""
    collapsed = " ".join(text.replace("`", "").split())
    return collapsed[:_ROLE_MAX_CHARS]


def _clean_doc(text: str) -> str:
    """Harnessdoc body: strip backticks (defence against fence injection), strip, cap length."""
    return text.replace("`", "").strip()[:_DOC_MAX_CHARS]


def _clean_overview(text: str) -> str:
    """Agent overview: strip backticks, cap length; newlines are kept (paragraph structure)."""
    return text.replace("`", "").strip()[:_OVERVIEW_MAX_CHARS]


def _walk_sorted(root: Path) -> tuple[list[Path], list[SkippedFile]]:
    """Every regular file under `root`, sorted by relative posix path, plus the walk's own skips.

    Symlinks are not followed (they can escape the checkout) and pruned excluded dirs are not
    descended, but neither is silently dropped: each symlink and each pruned dir gets a
    `SkippedFile` so the reported skips match the 'skips are reported, never silent' contract. A
    pruned directory yields exactly one skip for the dir, not one per file it would have contained.
    """
    results: list[Path] = []
    skips: list[SkippedFile] = []

    def _walk(directory: Path) -> None:
        for entry in sorted(directory.iterdir(), key=lambda p: p.name):
            rel = entry.relative_to(root).as_posix()
            if entry.is_symlink():
                skips.append(SkippedFile(path=rel, reason="symlink (not followed)"))
                continue
            if entry.is_dir():
                if entry.name in EXCLUDED_DIRS:
                    skips.append(
                        SkippedFile(
                            path=f"{rel}/", reason="excluded directory (contents not walked)"
                        )
                    )
                else:
                    _walk(entry)
            elif entry.is_file():
                results.append(entry)

    _walk(root)
    return sorted(results, key=lambda p: p.relative_to(root).as_posix()), skips


def _exclusion_reason(
    rel: str,
    basename: str,
    ignore_matchers: list[_GitignoreMatcher],
    extra_excludes: tuple[str, ...],
) -> str | None:
    if basename in EXCLUDED_FILES:
        return "generated file (lockfile or OS litter)"
    for pattern in SECRET_FILE_PATTERNS:
        if fnmatch.fnmatch(basename, pattern):
            return "may contain secrets"
    for pattern in extra_excludes:
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(basename, pattern):
            return f"excluded by --exclude {pattern!r}"
    for matcher in ignore_matchers:
        if matcher.matches(rel):
            return "matched .gitignore"
    return None


def _gitignore_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile one conservative gitignore pattern into an anchored full-relative-path regex.

    A pattern with a non-trailing '/' is root-anchored: it matches the whole relative path from the
    repo root. A bare pattern (no '/') matches that segment as a basename at any depth. A trailing
    '/' marks a directory: it and everything under it match. '**' becomes '.*', a single '*'
    becomes '[^/]*' (no directory crossing), and all other characters are escaped literally.
    """
    is_dir = pattern.endswith("/")
    body = pattern.rstrip("/")
    anchored = "/" in body
    # Translate glob tokens to regex on the literal-escaped text so '.' and friends stay literal.
    escaped = re.escape(body).replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
    if anchored:
        prefix = ""
        core = escaped
    else:
        # Match the pattern as a full path segment at any depth: at the start or after a '/'.
        prefix = r"(?:.*/)?"
        core = escaped
    if is_dir:
        # A directory pattern matches the directory itself and everything beneath it.
        suffix = r"/.*"
    else:
        # A file/either pattern matches the path itself or, if it names a dir, its contents.
        suffix = r"(?:/.*)?"
    return re.compile(f"^{prefix}{core}{suffix}$")


def _gitignore_patterns(root: Path) -> list[_GitignoreMatcher]:
    """Root `.gitignore` as conservative compiled matchers, or [] if it uses negation/whitelist.

    Whitelist-style .gitignore files (any '!' line, e.g. '*' then '!src/') re-include paths a
    plain pattern list would exclude; approximating them by dropping the '!' lines would collapse
    ingest to zero. Since body files must never be silently dropped, the presence of ANY negation
    disables gitignore handling entirely rather than over-excluding.
    """
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        return []
    lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
    matchers: list[_GitignoreMatcher] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("!"):
            return []  # negation/whitelist: do not apply gitignore at all
        stripped = stripped.lstrip("/")
        if not stripped:
            continue
        matchers.append(_GitignoreMatcher(regex=_gitignore_to_regex(stripped)))
    return matchers


class _RawFileDoc(BaseModel):
    role: str
    doc: str


class _RawOverview(BaseModel):
    overview: str


_ReplyT = TypeVar("_ReplyT", bound=BaseModel)


def _complete_json(
    provider: Provider, system: str, user: str, model: type[_ReplyT]
) -> _ReplyT | None:
    completion = provider.complete(
        system, [Message(role="user", content=user)], temperature=0.2, max_tokens=1024
    )
    raw = extract_json_object(completion.text)
    if raw is None:
        return None
    try:
        return model.model_validate_json(raw)
    except ValueError:
        return None


def _file_prompt(repo_file: RepoFile) -> str:
    content = repo_file.content
    clipped = ""
    if len(content) > _FILE_PROMPT_CLIP_CHARS:
        content = content[:_FILE_PROMPT_CLIP_CHARS]
        clipped = f"\n... [clipped at {_FILE_PROMPT_CLIP_CHARS} chars]"
    return f"## File: {repo_file.path}\n\n```\n{content}{clipped}\n```"


def _overview_prompt(name: str, collected: CollectedRepo, file_docs: list[FileDoc]) -> str:
    tree_lines = [f.path for f in collected.files][:_TREE_CLIP_LINES]
    readme = next(
        (f for f in collected.files if f.path.lower() in {"readme.md", "readme.rst", "readme"}),
        None,
    )
    readme_block = (
        f"\n\n## README excerpt\n\n{readme.content[:_OVERVIEW_CLIP_CHARS]}" if readme else ""
    )
    roles = "\n".join(f"- {d.path}: {d.role}" for d in file_docs[:_TREE_CLIP_LINES])
    return (
        f"## Agent: {name}\n\n## File tree ({len(collected.files)} files)\n\n"
        + "\n".join(tree_lines)
        + readme_block
        + f"\n\n## Per-file roles\n\n{roles}"
    )


def _render_bodymap(name: str, collected: CollectedRepo, mapping: BodyMap, *, source: str) -> str:
    lines = [
        f"# Body map: {name}",
        "",
        f"Source: {source}. {len(collected.files)} files mapped, {len(collected.skipped)} skipped.",
        "",
        "## Files",
        "",
    ]
    for file_doc in mapping.file_docs:
        lines.append(f"- `{file_doc.path}` - {file_doc.role}")
    if collected.skipped:
        lines += ["", "## Skipped", ""]
        for skip in collected.skipped:
            lines.append(f"- `{skip.path}` - {skip.reason}")
    if mapping.unmapped:
        lines += ["", "## Unmapped (kept, but the mapping reply was unusable)", ""]
        for path in mapping.unmapped:
            lines.append(f"- `{path}`")
    return "\n".join(lines) + "\n"
