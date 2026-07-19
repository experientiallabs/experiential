"""Web grounding: bounded search for entities the world model cannot ground internally.

When the env encounters a real-world entity outside its traces and knowledge base (an API's error
format, a package name, a flight code), it may emit a `ground_query` (see
`wmh.core.render.output_contract`) instead of hallucinating. A `Grounder` serves that query; the
engine caches results into the knowledge base (`grounded.md`) so an entity is searched at most
once per model, and re-completes the step with the results in context.

The default is `NullGrounder` — no network, tests and evals stay hermetic. Real backends:

- `BraveGrounder` (`BRAVE_SEARCH_API_KEY`; free tier at https://api-dashboard.search.brave.com/),
  a plain keyed JSON API with no scraping fragility — for free-text entity queries.
- `FetchGrounder` (keyless): when the agent's action is itself a read-only `curl` GET of a public
  URL, fetch that URL live and let the model shape the real body into the observation. Found
  empirically: 42% of the terminal-tasks test slice is curl-with-URL, scoring ~0.10 below every
  other step kind — the values (API payloads, search rankings) are unknowable without the network.
  Live fetches are inherently non-hermetic (the web has moved since capture); use only in serve or
  in explicitly-labeled eval modes.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from wmh.core.types import Action

GROUNDER_KINDS = ("none", "brave", "fetch")
_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_TIMEOUT_SECONDS = 15.0


class GroundingResult(BaseModel):
    """One search hit: enough to ground an entity, small enough to cache in the KB."""

    title: str = ""
    url: str = ""
    snippet: str = ""


class Grounder(Protocol):
    """Anything that can answer a grounding query with search results."""

    def ground(self, query: str) -> list[GroundingResult]:
        """Return search results for `query` (empty when grounding is unavailable)."""
        ...


class NullGrounder:
    """The default: grounding disabled, never touches the network."""

    def ground(self, query: str) -> list[GroundingResult]:
        return []


# Injectable HTTP GET (url, headers) -> response body; lets tests exercise BraveGrounder offline.
FetchFn = Callable[[str, dict[str, str]], str]


def _assert_public_http_url(url: str) -> None:
    """Reject URLs that could reach anything but a public web host (SSRF guard).

    Fetch targets can derive from MODEL OUTPUT (FetchGrounder GETs the action's own curl URL;
    `ground_query` is model-written), so before any request: scheme must be http(s) — no
    file://, ftp:// or scheme-less tricks — and the host must not resolve into private,
    loopback, or link-local space (cloud metadata endpoints live at 169.254.169.254).
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"grounding fetch requires http(s), got {parsed.scheme!r}: {url}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"grounding fetch URL has no host: {url}")
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise ValueError(f"grounding fetch host does not resolve: {host}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise ValueError(
                f"grounding fetch host {host} resolves to non-public address {address}"
            )


class _PublicOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """Re-validate every 3xx hop: a public host 302-ing to 169.254.169.254 is the classic pivot.

    (The remaining residue is DNS rebinding — resolve-at-check vs resolve-at-connect — which
    urllib cannot pin without a custom transport; grounding fetches stay read-only GETs, so the
    blast radius is exfil of a fetched body into the KB, not a write.)
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201, ANN202
        _assert_public_http_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_PublicOnlyRedirects)


def http_get(url: str, headers: dict[str, str]) -> str:
    """Fetch a public http(s) URL as text: the default, SSRF-guarded `FetchFn`."""
    _assert_public_http_url(url)
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 — guarded above
    with _OPENER.open(request, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310
        body: str = response.read().decode("utf-8")
        return body


class _BraveWeb(BaseModel):
    results: list[GroundingResult] = Field(default_factory=list)


class _BraveResponse(BaseModel):
    web: _BraveWeb = Field(default_factory=_BraveWeb)


class BraveGrounder:
    """Brave Search API backend (`X-Subscription-Token` keyed GET, JSON response)."""

    def __init__(self, api_key: str, *, count: int = 5, fetch: FetchFn = http_get) -> None:
        self._api_key = api_key
        self._count = count
        self._fetch = fetch

    def ground(self, query: str) -> list[GroundingResult]:
        params = urllib.parse.urlencode({"q": query, "count": str(self._count)})
        headers = {"Accept": "application/json", "X-Subscription-Token": self._api_key}
        body = self._fetch(f"{_BRAVE_ENDPOINT}?{params}", headers)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Brave search returned non-JSON for {query!r}: {body[:200]}") from exc
        try:
            # Brave's result objects use `description` for the snippet; map it before validation.
            raw_results = payload.get("web", {}).get("results", [])
            for item in raw_results:
                if isinstance(item, dict) and "description" in item and "snippet" not in item:
                    item["snippet"] = item.pop("description")
            return _BraveResponse.model_validate(payload).web.results
        except (ValidationError, AttributeError) as exc:
            raise ValueError(
                f"Brave search response for {query!r} did not match the expected shape: {exc}"
            ) from exc


class FetchGrounder:
    """Keyless grounder for URL-shaped queries: GET the URL, return its (capped) body.

    Non-URL queries yield no results — compose with a search backend if both are wanted. Fetches
    are memoized per instance (an eval or session asks about the same endpoint repeatedly) and
    failures return no results rather than raising: an unreachable URL should degrade to the
    ungrounded prediction, never break the step.
    """

    def __init__(self, *, max_chars: int = 8_000, fetch: FetchFn = http_get) -> None:
        self._max_chars = max_chars
        self._fetch = fetch
        self._memo: dict[str, list[GroundingResult]] = {}

    def ground(self, query: str) -> list[GroundingResult]:
        url = query.strip()
        if not url.startswith(("http://", "https://")):
            return []
        if url in self._memo:
            return self._memo[url]
        try:
            body = self._fetch(url, {"Accept": "*/*", "User-Agent": "wmh-grounder"})
        except Exception:  # noqa: BLE001 - any transport failure degrades to "no grounding"
            results: list[GroundingResult] = []
        else:
            if len(body) > self._max_chars:
                body = body[: self._max_chars] + "\n[truncated]"
            results = [GroundingResult(title=url, url=url, snippet=body)]
        self._memo[url] = results
        return results


# curl flags that mean the request mutates state (or uploads); those commands are never fetched.
_CURL_MUTATING_FLAGS = re.compile(
    r"(^|\s)(-X\s*(?!GET\b)\w+|--request\s+(?!GET\b)\w+|-d\b|--data\b|--data-\w+|-F\b|--form\b"
    r"|-T\b|--upload-file\b)"
)
_URL_IN_COMMAND = re.compile(r"https?://[^\s\"'|;>]+")


def extract_get_url(action: Action) -> str | None:
    """Return the URL a read-only `curl` GET in `action` targets, or None.

    Conservative by design: only bash tool calls whose command invokes `curl` without any
    mutating/upload flag qualify — a fetch must be side-effect-free to be safe to execute for
    grounding. The first URL in the command is taken (pipes/filters after it are the model's job
    to apply to the fetched body).
    """
    if action.name != "bash":
        return None
    command = action.arguments.get("command")
    if not isinstance(command, str) or "curl" not in command:
        return None
    if _CURL_MUTATING_FLAGS.search(command):
        return None
    match = _URL_IN_COMMAND.search(command)
    return match.group(0) if match else None


def prefetched_knowledge(
    knowledge: str | None, action: Action, grounder: Grounder | None
) -> str | None:
    """Append a live fetch of `action`'s read-only GET URL to the knowledge text, when possible.

    The stateless prefetch used by replay experiments; the serving engine has its own budgeted,
    KB-cached variant (`WorldModel._predict`). Returns `knowledge` unchanged when there is no
    grounder, no fetchable URL, or the fetch yields nothing.
    """
    if grounder is None:
        return knowledge
    url = extract_get_url(action)
    if url is None:
        return knowledge
    results = grounder.ground(url)
    if not results:
        return knowledge
    block = f"## live fetch: {url}\n{render_grounding(results)}"
    return f"{knowledge}\n\n{block}" if knowledge else block


# --- source grounding: pinned repo files as ground truth for read-verb actions -------------------

# Read-verb patterns whose observation is (a slice of) one file's true content. Any write/pipe
# transformation disqualifies — the observation would no longer be the raw file bytes.
_SED_RANGE = re.compile(r"\bsed\s+-n\s+'(\d+),(\d+)p'\s+(\S+)\s*$")
_CAT_FILE = re.compile(r"\bcat\s+(\S+)\s*$")
_HEAD_FILE = re.compile(r"\bhead\s+-(?:n\s*)?(\d+)\s+(\S+)\s*$")
_WRITE_MARKERS = re.compile(r"(>|>>|\bsed\s+-i\b|\btee\b|\bmv\b|\bcp\b|\bpatch\b|git\s+apply)")


@dataclass(frozen=True)
class FileRead:
    """One read-verb action: the repo-relative path and the requested 1-based line range."""

    path: str
    start: int | None
    end: int | None


def extract_file_read(action: Action) -> FileRead | None:
    """Return the file slice a pure read action requests, or None.

    Conservative by design (mirrors `extract_get_url`): only bash `sed -n 'A,Bp'`/`cat`/`head -N`
    with the file as the FINAL token qualify — pipes, redirects, in-place edits, or interpreters
    mean the observation is not the raw file content. Paths are made repo-relative by stripping
    the conventional sandbox prefix.
    """
    if action.name != "bash":
        return None
    command = action.arguments.get("command")
    if not isinstance(command, str) or _WRITE_MARKERS.search(command) or "|" in command:
        return None
    command = command.strip()
    if m := _SED_RANGE.search(command):
        return FileRead(_repo_relative(m.group(3)), int(m.group(1)), int(m.group(2)))
    if m := _HEAD_FILE.search(command):
        return FileRead(_repo_relative(m.group(2)), 1, int(m.group(1)))
    if m := _CAT_FILE.search(command):
        return FileRead(_repo_relative(m.group(1)), None, None)
    return None


def _repo_relative(path: str) -> str:
    return path.removeprefix("/testbed/").lstrip("/")


class SourceResolver:
    """Ground read-verb actions in the REAL file content at a pinned commit.

    `pins` maps a trace's instance id -> {repo, base_commit} (for swe-bench this is the
    committed `examples/swe-bench/instance_commits.json`, built once from the public dataset —
    any corpus whose traces can be pinned the same way gets the same machinery). Files are
    fetched keylessly from raw.githubusercontent.com, memoized per file, and sliced to the
    requested line range. Callers own the STALENESS GATE: never resolve a path the session has
    already touched — the pinned content is wrong the moment the agent edits the file (verified
    live: first-touch reads match the pin exactly; post-edit re-reads do not).
    """

    def __init__(self, pins: dict[str, dict[str, str]], *, fetch: FetchFn = http_get) -> None:
        self._pins = pins
        self._fetch = fetch
        self._memo: dict[str, list[str] | None] = {}

    @classmethod
    def from_file(cls, path: str | Path) -> SourceResolver:
        pins = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(pins)

    @property
    def pins(self) -> dict[str, dict[str, str]]:
        """The instance pins (shared with RepoTreeResolver so both ground the same commit)."""
        return self._pins

    def resolve(self, instance_id: str, read: FileRead) -> str | None:
        pin = self._pins.get(instance_id)
        if pin is None:
            return None
        url = f"https://raw.githubusercontent.com/{pin['repo']}/{pin['base_commit']}/{read.path}"
        if url not in self._memo:
            try:
                self._memo[url] = self._fetch(url, {"User-Agent": "wmh-grounder"}).splitlines()
            except Exception:  # noqa: BLE001 - unreachable file degrades to "no grounding"
                self._memo[url] = None
        lines = self._memo[url]
        if lines is None:
            return None
        if read.start is not None:
            lines = lines[read.start - 1 : read.end]
        return "\n".join(lines)


def source_grounded_knowledge(
    knowledge: str | None,
    action: Action,
    instance_id: str | None,
    prior_actions: list[Action],
    resolver: SourceResolver | None,
    *,
    annotate_stale: bool = False,
) -> str | None:
    """Append the pinned source slice for a read action to the knowledge text.

    First-touch reads get the pin as ground truth. Previously-touched paths are refused by
    default (the agent may have edited the file — a stale pin served as truth is wrong).
    `annotate_stale=True` instead serves the BASE version explicitly labeled as pre-edit — the
    model combines it with the session's in-context edits itself. Zero interpretation risk
    (nothing replays the edits mechanically), and it unlocks the ~20%-of-steps population of
    reads on edited files that the strict gate refuses.
    """
    if resolver is None or instance_id is None:
        return knowledge
    read = extract_file_read(action)
    if read is None:
        return knowledge
    basename = read.path.rsplit("/", 1)[-1]
    touched = False
    for prior in prior_actions:
        prior_cmd = prior.arguments.get("command") if prior.name else prior.content
        if isinstance(prior_cmd, str) and basename in prior_cmd:
            touched = True
            break
    if touched and not annotate_stale:
        return knowledge
    content = resolver.resolve(instance_id, read)
    if content is None:
        return knowledge
    span = f" lines {read.start}-{read.end}" if read.start is not None else ""
    if touched:
        header = (
            f"## source file: {read.path}{span} (BASE-COMMIT version — this session has since"
            " touched/edited this file; the current content is this base with the history's"
            " edits applied)"
        )
    else:
        header = f"## source file: {read.path}{span} (ground truth at the base commit)"
    block = f"{header}\n{content}"
    return f"{knowledge}\n\n{block}" if knowledge else block


# --- registry polling: package facts from PyPI/npm JSON (keyless, read-only) --------------------

# The args slice after `pip show|install`, up to the next shell operator. Flags are stripped
# token-by-token below (the corpus shape is flag soup: `pip install --break-system-packages X
# 2>&1 | tail -3`, `python3 -m pip install ...`).
_PIP_CMD = re.compile(r"\bpip3?\s+(?:show|install)\s+([^|;&\n]*)")
_NPM_QUERY = re.compile(r"\bnpm\s+(?:view|info)\s+([A-Za-z0-9@/_.-]+)")
# Flags whose argument replaces the registry as the install source: the action isn't asking
# the registry about a named package, so refuse.
_PIP_NON_REGISTRY_FLAGS = frozenset(
    {"-r", "--requirement", "-e", "--editable", "-c", "--constraint"}
)


def extract_package_query(action: Action) -> tuple[str, str] | None:
    """Return (registry, package) for a pip/npm package action, or None.

    Version specifiers are stripped (the registry answer covers them); invocations that don't
    name a registry package (`pip freeze`, `pip install -r ...`, `pip install -e .`) are refused.
    """
    if action.name != "bash":
        return None
    command = action.arguments.get("command")
    if not isinstance(command, str):
        return None
    if m := _PIP_CMD.search(command):
        for token in m.group(1).split():
            if token in _PIP_NON_REGISTRY_FLAGS:
                return None
            if token.startswith("-") or token.startswith("2>"):
                continue
            name = re.split(r"[=<>!\[]", token.strip("'\""))[0]
            if not name or name.startswith((".", "/")):
                return None  # a path, not a registry package
            return ("pypi", name)
        return None
    if m := _NPM_QUERY.search(command):
        return ("npm", m.group(1))
    return None


# (registry, package) -> rendered block, or None for a failed lookup. Failures are cached
# too: an unreachable registry must cost ONE timeout per package, not one per step.
_REGISTRY_MEMO: dict[tuple[str, str], str | None] = {}


def registry_grounded_knowledge(
    knowledge: str | None, action: Action, *, fetch: FetchFn = http_get
) -> str | None:
    """Append the real registry record for a package action (harmless keyless poll).

    PyPI's JSON API answers `pip show/install` with the actual current name/version/summary/
    dependencies — the values the model otherwise invents. Unreachable registries degrade to
    no grounding.
    """
    query = extract_package_query(action)
    if query is None:
        return knowledge
    registry, package = query
    memoize = fetch is http_get  # injected fetch fns (tests) must not share process state
    if memoize and query in _REGISTRY_MEMO:
        block = _REGISTRY_MEMO[query]
        if block is None:
            return knowledge
        return f"{knowledge}\n\n{block}" if knowledge else block
    if registry == "pypi":
        url = f"https://pypi.org/pypi/{package}/json"
    else:
        url = f"https://registry.npmjs.org/{package}/latest"
    try:
        payload = json.loads(fetch(url, {"Accept": "application/json"}))
    except Exception:  # noqa: BLE001 - registry unreachable: degrade to no grounding
        if memoize:
            _REGISTRY_MEMO[query] = None
        return knowledge
    info = payload.get("info", payload)
    requires = info.get("requires_dist") or info.get("dependencies") or []
    if isinstance(requires, dict):
        requires = [f"{k}{v}" for k, v in requires.items()]
    block = (
        f"## registry record: {package} ({registry}, live)\n"
        f"name: {info.get('name', package)}\n"
        f"version: {info.get('version', '?')}\n"
        f"summary: {info.get('summary', '')}\n"
        f"requires: {', '.join(str(r) for r in requires[:12])}\n"
        f"requires_python: {info.get('requires_python', '')}"
    )
    if memoize:
        _REGISTRY_MEMO[query] = block
    return f"{knowledge}\n\n{block}" if knowledge else block


def get_grounder(kind: str) -> Grounder:
    """Construct the configured grounder (`HarnessConfig.grounder`): none | brave | fetch."""
    if kind == "none":
        return NullGrounder()
    if kind == "fetch":
        return FetchGrounder()
    if kind == "brave":
        api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
        if not api_key:
            raise ValueError(
                "grounder 'brave' needs BRAVE_SEARCH_API_KEY set; get a free key at "
                "https://api-dashboard.search.brave.com/ or set grounder = 'none'"
            )
        return BraveGrounder(api_key)
    raise ValueError(f"unknown grounder {kind!r}; choose one of {', '.join(GROUNDER_KINDS)}")


def render_grounding(results: list[GroundingResult]) -> str:
    """Render results as compact markdown for the KB cache and the re-completion prompt."""
    if not results:
        return "(no results)"
    return "\n".join(f"- {r.title} ({r.url}): {r.snippet}".strip() for r in results)
