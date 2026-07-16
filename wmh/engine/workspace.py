"""Repo-tree grounding: answer ls/find/grep over pinned repos without executing anything.

The furthest point of test-time grounding that needs NOTHING provisioned: the repo tree at the
instance's pinned base commit comes from one keyless GitHub trees-API call, file contents come
from the same raw fetches `SourceResolver` already makes, and grep runs LOCALLY in pure Python
over that fetched text — no shell, no subprocess, no sandbox. Exploratory `grep`/`find`/`ls`
were a measured fabrication class (the model inventing plausible hits); the tree answers them
deterministically.

Safety line: reads only (public GETs), pure in-process text processing over what those reads
returned, and honest degradation — a query whose inputs can't be fully known (over-cap greps,
piped output) yields a clearly-labeled partial answer or nothing, never a guess dressed as
ground truth.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass

from wmh.core.types import Action
from wmh.engine.grounding import FetchFn, FileRead, SourceResolver, http_get

# Files fetched per grep before degrading to the filename-list partial. Keeps a dir-wide grep
# over a big repo from turning into hundreds of raw fetches.
_GREP_FETCH_CAP = 30
# Grep output lines kept (grep on a broad pattern can match thousands of lines).
_MAX_MATCH_LINES = 120

_LS = re.compile(r"(?:^|&&\s*)ls\s+(?:-[A-Za-z]+\s+)*((?!-)[\w./-]+)\s*$")
_FIND = re.compile(
    r"(?:^|&&\s*)find\s+((?!-)[\w./-]+)(?:\s+-type\s+[fd])?(?:\s+-name\s+'([^']+)')?\s*$"
)
_GREP = re.compile(r"(?:^|&&\s*)grep\s+-([A-Za-z]+)\s+['\"]([^'\"]+)['\"]\s+([\w./-]+)/?\s*$")
_WRITE_OR_PIPE = re.compile(r"[|>]")
# Prior-action write markers: session content under a path is unknowable after any of these.
_SESSION_WRITE = re.compile(
    r"([>]|\bsed\s+-i\b|\btee\b|\bpatch\b|git\s+apply|\brm\b|\bmv\b|\bcp\b)"
)


@dataclass(frozen=True)
class TreeQuery:
    """One tree-answerable action: a listing or a content grep under a repo path."""

    kind: str  # "list" | "grep"
    target: str  # repo-relative path prefix ("" = repo root)
    pattern: str | None  # -name glob (list) or the grep pattern
    recursive: bool = False
    line_numbers: bool = False
    names_only: bool = False
    ignore_case: bool = False


def _repo_relative(path: str) -> str:
    """Map an absolute-or-relative shell path to a repo-relative prefix ('' = repo root).

    `.`/`./` mean the repo root (the working dir in these corpora), not a literal dot path —
    unnormalized they'd match nothing in the tree and assert '(no matches)' as ground truth.
    """
    rel = path.removeprefix("/testbed/").removeprefix("/testbed").lstrip("/").rstrip("/")
    if rel in (".", "./"):
        return ""
    return rel.removeprefix("./")


def extract_tree_query(action: Action) -> TreeQuery | None:
    """Parse a pure `ls`/`find`/`grep` over a repo path, or None.

    Conservative like the other extractors: any pipe or redirect disqualifies (the observation
    would be transformed output, not the raw listing/matches).
    """
    if action.name != "bash":
        return None
    command = action.arguments.get("command")
    if not isinstance(command, str) or _WRITE_OR_PIPE.search(command):
        return None
    command = command.strip()
    if m := _GREP.search(command):
        flags, pattern, target = m.group(1), m.group(2), m.group(3)
        if any(f not in "rnliE" for f in flags):
            return None  # an unmodeled grep flag: refuse rather than mis-answer
        return TreeQuery(
            kind="grep",
            target=_repo_relative(target),
            pattern=pattern,
            recursive="r" in flags,
            line_numbers="n" in flags,
            names_only="l" in flags,
            ignore_case="i" in flags,
        )
    if m := _FIND.search(command):
        return TreeQuery(
            kind="list", target=_repo_relative(m.group(1)), pattern=m.group(2), recursive=True
        )
    if m := _LS.search(command):
        return TreeQuery(kind="list", target=_repo_relative(m.group(1)), pattern=None)
    return None


class RepoTreeResolver:
    """The pinned repo tree, one keyless trees-API call per instance (memoized)."""

    def __init__(
        self,
        pins: dict[str, dict[str, str]],
        *,
        fetch: FetchFn = http_get,
        grep_fetch_cap: int = _GREP_FETCH_CAP,
    ) -> None:
        self._pins = pins
        self._fetch = fetch
        self._grep_fetch_cap = grep_fetch_cap
        self._trees: dict[str, list[str] | None] = {}

    def paths(self, instance_id: str) -> list[str] | None:
        pin = self._pins.get(instance_id)
        if pin is None:
            return None
        key = f"{pin['repo']}@{pin['base_commit']}"
        if key not in self._trees:
            url = (
                f"https://api.github.com/repos/{pin['repo']}/git/trees/"
                f"{pin['base_commit']}?recursive=1"
            )
            try:
                payload = json.loads(self._fetch(url, {"User-Agent": "wmh-grounder"}))
                self._trees[key] = [
                    str(entry["path"]) for entry in payload.get("tree", []) if "path" in entry
                ]
            except Exception:  # noqa: BLE001 - unreachable API degrades to "no grounding"
                self._trees[key] = None
        return self._trees[key]

    def answer(
        self, instance_id: str, query: TreeQuery, *, source: SourceResolver | None = None
    ) -> str | None:
        """Answer a TreeQuery from the pinned tree; grep additionally needs `source` for content."""
        paths = self.paths(instance_id)
        if paths is None:
            return None
        prefix = f"{query.target}/" if query.target else ""
        under = [p for p in paths if p.startswith(prefix) or p == query.target]
        if query.kind == "list":
            if not query.recursive:  # plain ls: direct children only
                depth = prefix.count("/")
                children = [p for p in under if p.count("/") == depth and p != query.target]
                if not children and query.target in paths:
                    return query.target  # `ls <file>` on an existing file prints the path
                under = children
            if query.pattern:
                under = [p for p in under if fnmatch.fnmatch(p.rsplit("/", 1)[-1], query.pattern)]
            return "\n".join(under) if under else "(no matches)"
        # grep: fetch candidate text files under the target and match locally.
        exts = (".py", ".txt", ".rst", ".md", ".cfg", ".toml")
        candidates = [p for p in under if p.endswith(exts)]
        skipped = len([p for p in under if not p.endswith(exts)])
        if source is None:
            return None
        if len(candidates) > self._grep_fetch_cap:
            listing = "\n".join(candidates[: self._grep_fetch_cap * 2])
            return (
                f"(too many files to search exhaustively — {len(candidates)} candidates; "
                f"the files under {query.target or 'the repo root'} are:)\n{listing}"
            )
        try:
            rx = re.compile(query.pattern or "", re.IGNORECASE if query.ignore_case else 0)
        except re.error:
            return None  # unsupported pattern syntax: refuse rather than mis-answer
        lines_out: list[str] = []
        for path in candidates:
            content = source.resolve(instance_id, FileRead(path=path, start=None, end=None))
            if content is None:
                continue
            for lineno, line in enumerate(content.splitlines(), start=1):
                if rx.search(line):
                    if query.names_only:
                        lines_out.append(path)
                        break
                    prefix_txt = f"{path}:{lineno}:" if query.line_numbers else f"{path}:"
                    lines_out.append(f"{prefix_txt}{line}")
                    if len(lines_out) >= _MAX_MATCH_LINES:
                        break
            if len(lines_out) >= _MAX_MATCH_LINES:
                break
        # Honest partials: a filtered file set or a hit-line cap must be visible — a silent
        # subset presented as THE matches is a fabricated negative for everything outside it.
        notes: list[str] = []
        if skipped:
            searched = ", ".join(exts)
            notes.append(
                f"(searched {len(candidates)} {searched} files; {skipped} other files"
                " under this path were not searched)"
            )
        if len(lines_out) >= _MAX_MATCH_LINES:
            notes.append(f"[truncated at {_MAX_MATCH_LINES} matching lines]")
        body = "\n".join(lines_out) if lines_out else "(no matches in the searched files)"
        return "\n".join([body, *notes]) if notes else body


def tree_grounded_knowledge(
    knowledge: str | None,
    action: Action,
    instance_id: str | None,
    prior_actions: list[Action],
    tree: RepoTreeResolver | None,
    source: SourceResolver | None,
) -> str | None:
    """Append the tree's answer for a listing/grep action to the knowledge text.

    Session writes can't be replayed, so answers over touched paths are ANNOTATED (base-commit
    truth + a pointer at the in-context edits) rather than asserted — same discipline as
    source2.
    """
    if tree is None or instance_id is None:
        return knowledge
    query = extract_tree_query(action)
    if query is None:
        return knowledge
    answer = tree.answer(instance_id, query, source=source)
    if answer is None:
        return knowledge
    touched = False
    for prior in prior_actions:
        cmd = prior.arguments.get("command") if prior.name else prior.content
        if isinstance(cmd, str) and _SESSION_WRITE.search(cmd) and (query.target or "/") in cmd:
            touched = True
            break
    label = f"## repo tree answer: {query.kind} under {query.target or '(root)'} (base commit)"
    if touched:
        label += (
            " — the session has since modified content under this path;"
            " combine with the history's edits"
        )
    block = f"{label}\n{answer}"
    return f"{knowledge}\n\n{block}" if knowledge else block


# --- textops: pure text computation over content the session itself wrote -----------------------

_HEREDOC = re.compile(r"cat\s*>\s*(\S+)\s*<<\s*'?(\w+)'?\n(.*?)\n\2", re.S)
_ECHO_WRITE = re.compile(r"echo\s+(['\"])(.*?)\1\s*(>>?)\s*(\S+)")
_PRINTF_WRITE = re.compile(r"printf\s+(['\"])(.*?)\1\s*(>>?)\s*(\S+)")
_WC = re.compile(r"^wc\s+-([lwc])\s+(<\s*)?(\S+)$")
_SORT = re.compile(r"^sort((?:\s+-[unr]|\s+-k\s*\d+)*)\s+(\S+)$")
_UNIQ = re.compile(r"^uniq\s+(\S+)$")
_CAT = re.compile(r"^cat\s+(\S+)$")
_CUT = re.compile(r"^cut\s+-d\s*'?([^'\s])'?\s+-f\s*([\d,]+)\s+(\S+)$")
_HEAD_TAIL = re.compile(r"^(head|tail)\s+-n\s*(\d+)\s+(\S+)$")
# Segment separators for compound commands; heredoc bodies are cut out before splitting so
# their newlines don't count.
_SEGMENT_SPLIT = re.compile(r"&&|\|\||;|\n")


def _interpret_escapes(text: str, quote: str) -> str | None:
    """Bytes a shell write puts in the file, or None when we can't know them exactly."""
    if "$" in text or "`" in text:
        return None  # expansion: the written bytes depend on shell state we don't model
    if quote == "'":
        return text
    return text.replace("\\n", "\n").replace("\\t", "\t")


class SessionFiles:
    """File contents reconstructable EXACTLY from shell history (heredoc/echo/printf writes).

    Only writes whose full bytes appear in the command qualify; anything else (sed -i, tee,
    program output, `$VAR` expansion) taints the path so textops refuse rather than compute on
    a guess.
    """

    def __init__(self) -> None:
        self._files: dict[str, str] = {}
        self._tainted: set[str] = set()

    @classmethod
    def from_actions(cls, actions: list[Action]) -> SessionFiles:
        files = cls()
        for action in actions:
            cmd = action.arguments.get("command") if action.name == "bash" else None
            if isinstance(cmd, str):
                files.apply_command(cmd)
        return files

    def apply_command(self, cmd: str) -> None:
        """Register the command's knowable writes, tainting paths it changes unknowably."""
        known_spans: list[tuple[int, int]] = []
        # Contents are EXACT file bytes: heredocs and echo append the trailing newline the real
        # shell writes; printf writes precisely its escapes (no implicit newline) — `wc -c`/`-l`
        # answers are wrong otherwise (printf 'abc' really is 3 bytes, 0 newlines).
        for m in _HEREDOC.finditer(cmd):
            self._files[m.group(1)] = m.group(3) + "\n"
            self._tainted.discard(m.group(1))
            known_spans.append(m.span())
        for m in _ECHO_WRITE.finditer(cmd):
            known = _interpret_escapes(m.group(2), m.group(1))
            self._write(m.group(4), m.group(3), known + "\n" if known is not None else None)
            known_spans.append(m.span())
        for m in _PRINTF_WRITE.finditer(cmd):
            text = m.group(2)
            known = None if "%" in text else _interpret_escapes(text, m.group(1))
            self._write(m.group(4), m.group(3), known)
            known_spans.append(m.span())
        # Taint scan runs on the command WITHOUT its known-write spans, so a heredoc followed by
        # `sed -i` in the same command still taints.
        residue = cmd
        for lo, hi in sorted(known_spans, reverse=True):
            residue = residue[:lo] + residue[hi:]
        if _SESSION_WRITE.search(residue):
            # Taint every KNOWN path the residue mentions — an extension heuristic missed
            # extensionless files (Makefile, `results`), serving pre-edit content as exact.
            for path in list(self._files):
                if path in residue:
                    self._tainted.add(path)

    def _write(self, path: str, op: str, text: str | None) -> None:
        if text is None:  # unknowable bytes: the path's content is no longer exact
            self._tainted.add(path)
            return
        if op == ">":
            self._files[path] = text
            self._tainted.discard(path)
        elif path in self._files and path not in self._tainted:
            self._files[path] = self._files[path] + text
        else:  # append to a file we never saw created: contents unknown
            self._tainted.add(path)

    def content(self, path: str) -> str | None:
        if path in self._tainted:
            return None
        return self._files.get(path)


def _sort_lines(lines: list[str], flags: str) -> list[str] | None:
    numeric = "-n" in flags
    key_match = re.search(r"-k\s*(\d+)", flags)
    field = int(key_match.group(1)) if key_match else None

    def key_of(line: str) -> tuple[float, str] | None:
        text = line
        if field is not None:
            parts = line.split()
            if field > len(parts):
                return None
            text = parts[field - 1]
        if numeric:
            m = re.match(r"\s*(-?\d+(?:\.\d+)?)", text)
            return (float(m.group(1)) if m else 0.0, line)
        return (0.0, text)

    keyed: list[tuple[tuple[float, str], str]] = []
    for line in lines:
        key = key_of(line)
        if key is None:
            return None  # a line lacks the sort field: GNU semantics get subtle, refuse
        keyed.append((key, line))
    out = [line for _, line in sorted(keyed, key=lambda pair: pair[0])]
    if "-r" in flags:
        out.reverse()
    if "-u" in flags:
        seen: set[str] = set()
        deduped: list[str] = []
        for line in out:
            if line not in seen:
                seen.add(line)
                deduped.append(line)
        out = deduped
    return out


def _answer_segment(segment: str, files: SessionFiles) -> str | None:
    """Answer one pure textop segment from known content, or refuse."""
    if m := _WC.match(segment):
        flag, redirected, path = m.group(1), m.group(2), m.group(3)
        content = files.content(path)
        if content is None:
            return None
        if flag == "l":
            count = content.count("\n")  # POSIX wc -l counts newlines, not display lines
        elif flag == "w":
            count = len(content.split())
        else:
            count = len(content.encode())  # contents are exact bytes, newlines included
        return str(count) if redirected else f"{count} {path}"
    if m := _SORT.match(segment):
        content = files.content(m.group(2))
        if content is None:
            return None
        lines = _sort_lines(content.splitlines(), m.group(1))
        return "\n".join(lines) if lines is not None else None
    if m := _UNIQ.match(segment):
        content = files.content(m.group(1))
        if content is None:
            return None
        out: list[str] = []
        for line in content.splitlines():
            if not out or out[-1] != line:
                out.append(line)
        return "\n".join(out)
    if m := _CAT.match(segment):
        content = files.content(m.group(1))
        if content is None:
            return None
        # Observations show the text, not the file's final newline byte.
        return content[:-1] if content.endswith("\n") else content
    if m := _CUT.match(segment):
        delim, fields_spec, path = m.group(1), m.group(2), m.group(3)
        content = files.content(path)
        if content is None:
            return None
        wanted = [int(f) for f in fields_spec.split(",")]
        out_lines: list[str] = []
        for line in content.splitlines():
            if delim not in line:
                out_lines.append(line)  # cut passes delimiter-free lines through whole
                continue
            parts = line.split(delim)
            out_lines.append(delim.join(parts[f - 1] for f in wanted if f <= len(parts)))
        return "\n".join(out_lines)
    if m := _HEAD_TAIL.match(segment):
        verb, count, path = m.group(1), int(m.group(2)), m.group(3)
        content = files.content(path)
        if content is None:
            return None
        lines = content.splitlines()
        picked = lines[:count] if verb == "head" else lines[-count:]
        return "\n".join(picked)
    return None


def textop_answer(action: Action, files: SessionFiles) -> str | None:
    """Compute the action's final textop segment over session-known content, or refuse.

    The measured corpus shape is compound: the file is written and queried in ONE command
    (`cat > f <<EOF ... EOF && sort -k2 -n f`). Earlier segments' knowable writes are applied
    (on a copy-in — the caller's `files` picks them up via `apply_command` if it wants them),
    then the LAST segment is answered iff it is a pure read-only op with no redirect/pipe.
    """
    if action.name != "bash":
        return None
    command = action.arguments.get("command")
    if not isinstance(command, str):
        return None
    files.apply_command(command)
    stripped = _HEREDOC.sub("", command)
    segments = [seg.strip() for seg in _SEGMENT_SPLIT.split(stripped) if seg.strip()]
    if not segments:
        return None
    final = segments[-1]
    if _WRITE_OR_PIPE.search(final):
        return None  # the observation is a redirect/pipe product, not the op's stdout
    return _answer_segment(final, files)


def textop_grounded_knowledge(
    knowledge: str | None, action: Action, prior_actions: list[Action]
) -> str | None:
    """Append the deterministic result of a pure text op over session-written content."""
    files = SessionFiles.from_actions(prior_actions)
    answer = textop_answer(action, files)
    if answer is None:
        return knowledge
    block = (
        f"## text-op result (computed deterministically from content this session wrote)\n{answer}"
    )
    return f"{knowledge}\n\n{block}" if knowledge else block
