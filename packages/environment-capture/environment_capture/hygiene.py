"""Corpus hygiene: detect trajectories that escaped the task workspace onto the host.

`LocalBashEnv` executes real commands on the host, and an exploring agent that doesn't
immediately find its data can wander out of the workspace — capturing real host content
(home-directory listings, dotfile names, interpreter paths) into a corpus that gets committed
and redistributed. That is a privacy leak AND wrong environment dynamics: the world model should
learn the benchmark's workspace, not this machine.

Two detectors, used at capture emit time and at conversion (and by integrators to audit a
committed corpus): commands that TARGET host locations, and observations that CARRY host
markers. Flagged trajectories are dropped whole (never silently redacted — the discipline is
whole-trajectory drop, matching the terminal-tasks converter's `--exclude-substr`).
"""

from __future__ import annotations

import functools
import getpass
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from environment_capture.trajectory import Trajectory

# Commands that target host locations: absolute host roots, the home directory, walking out of
# the workspace, or sweeping the filesystem root. Workspace-relative paths never match.
#
# Absolute host roots and `$HOME` accept a wider leading boundary — the shell word boundaries
# PLUS quote chars, the `<`/`>` redirection operators, and `=` (assignment) — so quoted
# (`cat "/etc/passwd"`), redirected (`cat </etc/passwd`), and assigned (`H=/etc`) host paths are
# caught. `~` keeps the narrow shell-word boundary (NOT quotes): a simulated filesystem's quoted
# `'~/documents'` API argument is content, not a host escape (see `generic_path_markers`).
_ABS_BOUNDARY = r"(?:^|[\s;&|(`=\"'<>])"
_HOST_BOUNDARY = r"(?:^|[\s;&|(`])"
_CMD_ESCAPE_RE = re.compile(
    _ABS_BOUNDARY
    + r"(/(?:Users|home|root|etc|usr|opt|var|private|tmp)(?:/|\b)|\$HOME\b)"
    + r"|"
    + _HOST_BOUNDARY
    + r"(~(?:[/\s]|$)|cd\s+\.\.|(?:find|ls|tree|du)\s+(?:-[^\s]+\s+)*/(?:\s|$))"
)

# Multi-level relative traversal out of the workspace (`cat ../../../root/.ssh/id_rsa`). A SINGLE
# `../` can legitimately reference a workspace-internal sibling after a `cd` (e.g. `cd data && cat
# ../manual.md`), so only two-or-more consecutive `../` components — which cannot stay inside a
# freshly-rooted workspace — are flagged. `cd ..` is already covered by _CMD_ESCAPE_RE.
_RELATIVE_TRAVERSAL_RE = re.compile(r"(?:\.\.[/\\]){2,}")

# Environment dumps that never target task data: `env` / `env | grep ...` / `printenv [VAR]`.
# `env` used to run a command with a modified environment is legitimate and is NOT matched: the
# negative lookahead skips a following assignment (`env VAR=value cmd`) or option flag (`env -u
# VAR cmd`, `env -i cmd`). Only the bare dumping forms remain.
_ENV_DUMP_RE = re.compile(r"(?:^|[\s;&|(`])(?:printenv\b|env\b(?!\s+(?:-|[A-Za-z_]\S*=)))")

# References to credential-shaped environment variables (`echo $AWS_SECRET_ACCESS_KEY`,
# `printf %s "$HF_TOKEN"`); the surrounding `${...}` braces are stripped by _normalize_command.
# The name must be a COMPOUND uppercase env var (a segment, then a credential word) so a plain
# `$SECRET`/`$KEY` placeholder is not swept up — real provider vars are `AWS_SECRET_ACCESS_KEY`,
# `ANTHROPIC_API_KEY`, `HF_TOKEN`, ...
_SECRET_ENV_REF_RE = re.compile(
    r"\$[A-Z][A-Z0-9]*_[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)[A-Z0-9_]*"
)

# Value-based secret detection for OBSERVATIONS: opaque credential values that carry no path- or
# filename-shaped marker (an SSH key body, an AWS key id, provider tokens, an `env` dump line).
# Substring/path markers are not the security boundary — a leaked value must drop the trajectory
# on its own. The finding never echoes the value itself (the excerpt is redacted).
_SECRET_VALUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pem-private-key", re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")),
    ("aws-access-key-id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("anthropic-api-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{8,}")),
    ("openai-api-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("huggingface-token", re.compile(r"\bhf_[A-Za-z0-9]{16,}")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{16,}")),
    ("github-token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{20,})")),
    # An `env`/`printenv` dump line for a credential-shaped variable whose VALUE is itself
    # secret-shaped: >=16 chars of key/token alphabet. The length+alphabet guard keeps ordinary
    # benchmark output like `PRIMARY_KEY=1001` or `BUILD_TOKEN=github_runner` from being dropped.
    (
        "env-dump-secret",
        re.compile(
            r"(?m)^[ \t]*(?:export[ \t]+)?"
            r"[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)[A-Z0-9_]*"
            r"[ \t]*=[ \t]*['\"]?[A-Za-z0-9+/=_-]{16,}"
        ),
    ),
)

# PATH-shaped host content in observations. Relaxable per benchmark (generic_path_markers=False)
# for environments whose OWN simulated filesystem legitimately uses home-style paths. The
# workspace's own tempdir prefixes (/var/folders, /private/tmp on macOS) are deliberately NOT
# markers: LocalBashEnv workspaces live there, so `pwd`/tracebacks echoing the workspace's own
# absolute path would mass-flag legitimate trajectories.
_PATH_MARKERS = (
    "/Users/",
    "/home/",
    "/root",
    "$HOME",
    "~/",
)

# Credential/toolchain content that is NEVER legitimate benchmark dynamics — checked
# unconditionally, regardless of the path-marker policy.
_SENSITIVE_MARKERS = (
    ".ssh",
    "id_rsa",
    "id_ecdsa",
    "id_ed25519",
    "anaconda3",
    "site-packages",
    "node_modules",
    "Application Support",
)


# Manual success-only caches: a TRANSIENT resolution failure (a bare uid with no passwd entry,
# a momentarily unset USER in a container) must degrade that one call, not get baked into an
# lru_cache and silently disable identity detection for the rest of the process.
_runtime_markers_cache: tuple[str, ...] | None = None
_identity_regexes_cache: tuple[re.Pattern[str], ...] | None = None


def _runtime_markers() -> tuple[str, ...]:
    """Machine-identity markers, learned at runtime (never committed as literals).

    The home PATH always contributes when resolvable. The bare username is NOT a marker on its
    own — common CI usernames (`runner`, `ubuntu`) appear as ordinary words in legitimate
    output — it is matched only in identity-shaped contexts (see `_identity_regexes`). Resolution
    failures degrade to fewer markers for THIS call instead of crashing the import; success is
    cached, failure is retried.
    """
    global _runtime_markers_cache
    if _runtime_markers_cache is None:
        try:
            _runtime_markers_cache = (str(Path.home()),)
        except (KeyError, OSError, RuntimeError):
            return ()
    return _runtime_markers_cache


def _identity_regexes() -> tuple[re.Pattern[str], ...]:
    """Username-in-context patterns: `ls -l` ownership columns and /home-style paths."""
    global _identity_regexes_cache
    if _identity_regexes_cache is None:
        try:
            user = getpass.getuser()
        except (KeyError, OSError):
            return ()
        quoted = re.escape(user)
        _identity_regexes_cache = (
            # ls -l style: permission bits ... links ... owner column.
            re.compile(rf"[dl\-][rwxsStT\-]{{9}}\S*\s+\d+\s+{quoted}\b"),
            # A home path constructed for this account on either platform.
            re.compile(rf"/(?:Users|home)/{quoted}\b"),
        )
    return _identity_regexes_cache


_BRACE_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _normalize_command(command: str) -> str:
    """Rewrite `${VAR}` to `$VAR` so `${HOME}`/`${AWS_SECRET_ACCESS_KEY}` read as bare `$VAR`.

    Only variable-brace syntax is rewritten (brace expansion like `cp f{,.bak}` is left intact).
    Quotes and redirection operators are NOT rewritten — they are handled by the leading-boundary
    class in `_CMD_ESCAPE_RE`, which distinguishes a quoted absolute host path (an escape) from a
    simulated filesystem's quoted `~/` API argument (content). This is screening, not a shell
    parser.
    """
    return _BRACE_VAR_RE.sub(r"$\1", command)


def _match_host_command(command: str) -> str | None:
    """The first host-targeting signal in a command, or None. See :func:`command_targets_host`."""
    normalized = _normalize_command(command)
    for pattern in (
        _CMD_ESCAPE_RE,
        _RELATIVE_TRAVERSAL_RE,
        _ENV_DUMP_RE,
        _SECRET_ENV_REF_RE,
    ):
        match = pattern.search(normalized)
        if match is not None:
            return match.group(0)
    return None


def command_targets_host(command: str) -> bool:
    """Whether a command targets host locations outside the task workspace.

    Covers absolute host roots (including quoted/redirected/assigned forms), `~`/`$HOME`, relative
    `../` traversal, filesystem-root sweeps, environment dumps (`env`/`printenv`), and references to
    credential-shaped environment variables. This is a defense-in-depth guard, not the security
    boundary: capture subprocesses additionally run with a scrubbed environment (see
    ``localexec.LocalBashEnv``) and observations are screened for secret values.
    """
    return _match_host_command(command) is not None


def _secret_value_findings(field: str, text: str) -> list[HygieneFinding]:
    """Findings for opaque credential VALUES (key bodies, tokens, env-dump lines) in text.

    The finding redacts the value: only the detector name is reported, never the secret itself.
    """
    for name, pattern in _SECRET_VALUE_PATTERNS:
        if pattern.search(text) is not None:
            return [HygieneFinding(field=field, marker=name, excerpt="[redacted secret value]")]
    return []


@dataclass(frozen=True)
class HygieneFinding:
    """One host-escape signal in a trajectory: where it was seen and what matched."""

    field: str  # "command" | "output"
    marker: str
    excerpt: str


def _marker_regex(generic_path_markers: bool) -> re.Pattern[str]:
    """One alternation over the active marker set (single pass per text)."""
    markers = _SENSITIVE_MARKERS + _runtime_markers()
    if generic_path_markers:
        markers = _PATH_MARKERS + markers
    return _compile_alternation(markers)


@functools.lru_cache(maxsize=8)
def _compile_alternation(markers: tuple[str, ...]) -> re.Pattern[str]:
    """Compile once per distinct marker set — keyed on the markers themselves, so a runtime
    marker that resolves later (see the success-only caches above) is picked up, not baked in."""
    return re.compile("|".join(re.escape(marker) for marker in markers))


def _check_text(
    field: str, text: str, *, generic_path_markers: bool = True
) -> list[HygieneFinding]:
    findings: list[HygieneFinding] = []
    # Opaque credential VALUES are never legitimate benchmark dynamics; screen every field for
    # them first, regardless of the path-marker policy.
    secret_findings = _secret_value_findings(field, text)
    if secret_findings:
        return secret_findings
    if field == "command":
        marker = _match_host_command(text)
        if marker is not None:
            findings.append(
                HygieneFinding(
                    field=field, marker=marker.strip(" \t;&|(`=\"'<>")[:60], excerpt=text[:120]
                )
            )
        return findings
    # Sensitive + identity markers always run; only the PATH-shaped markers are relaxable (a
    # simulated filesystem's ~/ paths are content, a real .ssh/id_rsa never is).
    match = _marker_regex(generic_path_markers).search(text)
    if match is not None:
        index = match.start()
        findings.append(
            HygieneFinding(
                field=field, marker=match.group(0), excerpt=text[max(0, index - 40) : index + 80]
            )
        )
        return findings
    for pattern in _identity_regexes():
        id_match = pattern.search(text)
        if id_match is not None:
            index = id_match.start()
            findings.append(
                HygieneFinding(
                    field=field,
                    marker=id_match.group(0)[:40],
                    excerpt=text[max(0, index - 40) : index + 80],
                )
            )
            break
    return findings


def host_escape_findings(
    trajectory: Trajectory, *, generic_path_markers: bool = True
) -> list[HygieneFinding]:
    """Every host-escape signal in a trajectory's commands and observations.

    Command-level checks always run. Set ``generic_path_markers=False`` to skip the generic path
    markers (``~/``, ``/home/``, ``/root``, ...) FOR OBSERVATIONS ONLY — for a benchmark whose own
    environment legitimately emits such paths as content (e.g. AppWorld's simulated file system).
    The runtime identity markers (real username + real home) still run unconditionally.
    """
    findings: list[HygieneFinding] = []
    for step in trajectory.steps:
        for value in step.action.arguments.values():
            if isinstance(value, str):
                findings.extend(_check_text("command", value))
        findings.extend(
            _check_text("output", step.output, generic_path_markers=generic_path_markers)
        )
    return findings


def partition_contained(
    trajectories: list[Trajectory], *, generic_path_markers: bool = True
) -> tuple[list[Trajectory], list[Trajectory]]:
    """Split trajectories into (workspace-contained, flagged), preserving order.

    ``generic_path_markers`` is forwarded to :func:`host_escape_findings`.
    """
    clean: list[Trajectory] = []
    flagged: list[Trajectory] = []
    for trajectory in trajectories:
        target = (
            flagged
            if host_escape_findings(trajectory, generic_path_markers=generic_path_markers)
            else clean
        )
        target.append(trajectory)
    return clean, flagged


def scan_spans_jsonl(
    path: Path, *, generic_path_markers: bool = True
) -> dict[str, list[HygieneFinding]]:
    """Audit a committed OTel GenAI corpus; returns flagged trace ids with their findings.

    ``generic_path_markers`` mirrors the capture-time policy so a benchmark's declared
    relaxation (e.g. a simulated filesystem) is auditable with the same semantics; sensitive
    and identity markers always run. Streams the file — corpora reach tens of MB.
    """
    flagged: dict[str, list[HygieneFinding]] = {}
    with path.open(encoding="utf-8") as handle:
        return _scan_lines(handle, flagged, generic_path_markers)


def _scan_lines(
    lines: Iterable[str], flagged: dict[str, list[HygieneFinding]], generic_path_markers: bool
) -> dict[str, list[HygieneFinding]]:
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            span = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"corpus line {lineno} is not valid JSON ({error}); the file is corrupt — "
                "re-emit the corpus before auditing"
            ) from error
        trace_id = str(span.get("traceId", ""))
        for attribute in span.get("attributes", []):
            key = attribute.get("key", "")
            value = attribute.get("value", {}).get("stringValue", "")
            findings: list[HygieneFinding] = []
            if key == "gen_ai.tool.call.arguments":
                try:
                    arguments = json.loads(value)
                except json.JSONDecodeError:
                    arguments = None
                if isinstance(arguments, dict):
                    for argument in arguments.values():
                        if isinstance(argument, str):
                            findings.extend(_check_text("command", argument))
                else:
                    # Scalar/array-shaped tool arguments (some tool schemas emit these): scan
                    # the raw text instead of crashing the audit on .values().
                    findings.extend(_check_text("command", value))
            elif key == "gen_ai.tool.message":
                findings.extend(
                    _check_text("output", value, generic_path_markers=generic_path_markers)
                )
            if findings:
                flagged.setdefault(trace_id, []).extend(findings)
    return flagged
