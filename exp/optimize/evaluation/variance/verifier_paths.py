"""Put a start's verifier pointers in the coordinates the verifier is graded in.

A captured start is graded over its evaluator projection, not its raw state: the workflow's
``initial_state`` and ``final_state`` hold each probe's value keyed by probe name
(``probe_paths``). Authors kept writing verifier pointers in raw state coordinates
(``/final_state/Workspace filesystem/files//workspace/smoothing.py``), often with an unescaped
``/`` inside a key, so every completed branch was unassessable: the pointer named nothing the
grade could see. This module rewrites such a pointer onto the probe that covers it, escaping
keys per RFC 6901, and, while a start is authored, adds a probe for a verifier pointer no probe
covers. A pointer it cannot place is left unchanged for the authoring check to refuse.

A files map (:func:`keyed_by_paths`) holds each file under one flat key such as
``confcheck/__init__.py``; the simulator never nests directories in it. A pointer that walks
into such a map as nested segments (``.../files/confcheck/__init__.py``) names nothing any run
holds, so it is rewritten to the map's own key style (``.../files/confcheck~1__init__.py``), or
left for the authoring check to refuse when the map's style cannot decide it
(:func:`map_style`).
"""

from __future__ import annotations

from collections.abc import Mapping

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.simulation.reconstruction.models import resolve

WORKFLOW_STATES = ("final_state", "initial_state")
"""Workflow roots that hold the evaluator projection."""

WORKFLOW_ROOTS = (*WORKFLOW_STATES, "final_answer", "steps", "stop_reason")
"""Every root of the normalized workflow a verifier pointer may start at."""

VERIFIER_PROBE = "verifier"
"""Name stem of a probe added so the verifier can observe a path no probe covered."""


def token(key: str) -> str:
    """One RFC 6901 reference token for a key."""
    return key.replace("~", "~0").replace("/", "~1")


def split(pointer: str) -> list[str]:
    """A pointer's unescaped keys; raises ValueError for a relative pointer or bad escape."""
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise ValueError("pointer must be absolute")
    keys = []
    for raw in pointer[1:].split("/"):
        if "~" in raw.replace("~0", "").replace("~1", ""):
            raise ValueError("invalid pointer escape")
        keys.append(raw.replace("~1", "/").replace("~0", "~"))
    return keys


def join(keys: list[str]) -> str:
    """The pointer naming these keys."""
    return "".join("/" + token(key) for key in keys)


def _path_key(key: str) -> bool:
    """Whether a key names a file by path or file name (it holds a ``/`` or ``.``)."""
    return "/" in key or "." in key


def keyed_by_paths(value: JsonValue) -> bool:
    """Whether an object is a map keyed by file paths or names, not an object of named fields.

    This is the shape contract a start's domain ``state_schema`` is derived from: an object any
    of whose keys holds a ``/`` or ``.`` is a files map, which may gain and lose entries, each
    under one key. An empty object says neither.
    """
    return isinstance(value, dict) and any(_path_key(key) for key in value)


def _leaf_entries(value: JsonObject) -> bool:
    """Whether every file entry of a map is a leaf (a file's text), so a file key ends a path.

    Only entries keyed by paths are files; a named annotation beside them (a list of other
    entries) does not decide how files are held.
    """
    return all(
        not isinstance(entry, (dict, list)) for key, entry in value.items() if _path_key(key)
    )


def map_style(value: JsonValue, keys: list[str]) -> list[str] | None:
    """The keys a pointer means below ``value`` in the state's own key style; None if undecided.

    Keys that resolve are kept. At a files map (:func:`keyed_by_paths`) the remaining keys are
    one file key: the longest run of them that is an existing key is followed, and when none is
    and every entry is a file's text, all of them join into one new key (``a/b.py``). At an
    empty object more than one remaining key is undecided: the state does not say whether it is
    a files map or an object of named fields. Elsewhere a missing key starts a path the actor
    may create, kept key by key.

    Args:
        value: The initial value the pointer starts at (a probe's projected value).
        keys: The pointer's unescaped keys below it.

    Returns:
        The keys in the state's key style, or None for a pointer the style cannot decide (a
        files map whose entries are objects, or several keys under an empty object).
    """
    result: list[str] = []
    node: JsonValue = value
    index = 0
    while index < len(keys):
        rest = keys[index:]
        if isinstance(node, dict):
            if keyed_by_paths(node):
                end = next(
                    (end for end in range(len(rest), 0, -1) if "/".join(rest[:end]) in node),
                    None,
                )
                if end is not None:
                    key = "/".join(rest[:end])
                    result.append(key)
                    node, index = node[key], index + end
                    continue
                if len(rest) == 1 or _leaf_entries(node):
                    return [*result, "/".join(rest)]
                return None
            if rest[0] in node:
                result.append(rest[0])
                node, index = node[rest[0]], index + 1
                continue
            if not node and len(rest) > 1:
                return None
            return [*result, *rest]
        if (
            isinstance(node, list)
            and rest[0].isascii()
            and rest[0].isdigit()
            and str(int(rest[0])) == rest[0]
            and int(rest[0]) < len(node)
        ):
            result.append(rest[0])
            node, index = node[int(rest[0])], index + 1
            continue
        return [*result, *rest]
    return result


def state_keys(state: JsonValue, pointer: str) -> list[str] | None:
    """The keys a raw-state pointer means, escaping slashes inside keys; None when unplaceable.

    A pointer resolving as written is kept. Otherwise its tokens are rejoined into existing keys
    when exactly one way resolves. Otherwise its longest resolving prefix is kept and the rest
    names one new key under a map keyed by paths (a ``files`` map holding ``/workspace/a.py``),
    or one key per token elsewhere (:func:`map_style`): a path the actor creates is still a
    place the verifier may read. A pointer whose first key is not in the state is tried under
    each top-level domain and placed only when exactly one domain resolves it in full.
    """
    try:
        tokens = pointer[1:].split("/") if pointer.startswith("/") else None
        keys = split(pointer)
    except ValueError:
        return None
    if tokens is None or not isinstance(state, dict):
        return None
    try:
        resolve(state, pointer)
    except ValueError:
        pass
    else:
        return keys
    raw = [part.replace("~1", "/").replace("~0", "~") for part in tokens]
    joined = _rejoined(state, raw)
    if joined is not None:
        return joined
    if raw and raw[0] not in state:
        placed = [
            [domain, *found]
            for domain, value in state.items()
            if (found := _rejoined(value, raw)) is not None
        ]
        if len(placed) == 1:
            return placed[0]
    for count in range(len(raw) - 1, 0, -1):
        prefix = _rejoined(state, raw[:count])
        if prefix is None:
            continue
        holder = resolve(state, join(prefix))
        rest = raw[count:]
        if isinstance(holder, dict) and rest[0] == "":
            return [*prefix, "/".join(rest)]
        if isinstance(holder, dict):
            styled = map_style(holder, rest)
            return None if styled is None else [*prefix, *styled]
        return None
    return None


def _rejoined(value: JsonValue, raw: list[str]) -> list[str] | None:
    """The keys raw tokens mean when exactly one rejoining resolves in ``value``."""
    found: list[list[str]] = []

    def walk(node: JsonValue, index: int, keys: list[str]) -> None:
        if len(found) > 1:
            return
        if index == len(raw):
            found.append(keys)
            return
        if isinstance(node, dict):
            for end in range(index + 1, len(raw) + 1):
                key = "/".join(raw[index:end])
                if key in node:
                    walk(node[key], end, [*keys, key])
        elif isinstance(node, list):
            part = raw[index]
            if (
                part.isascii()
                and part.isdigit()
                and str(int(part)) == part
                and int(part) < len(node)
            ):
                walk(node[int(part)], index + 1, [*keys, part])

    walk(value, 0, [])
    return found[0] if len(found) == 1 else None


def projected(
    pointer: str,
    state: JsonValue,
    probes: dict[str, str],
    *,
    rooted: bool,
    add: bool,
) -> str:
    """Return a verifier pointer in projection coordinates, adding a covering probe if allowed.

    Args:
        pointer: The authored pointer: workflow-rooted when ``rooted`` (``/final_state/...``),
            else projection-relative (a preserved path).
        state: The start's raw initial state.
        probes: Probe name to raw-state pointer; extended in place when ``add``.
        rooted: Whether the pointer starts at a workflow root.
        add: Whether a probe may be added for a path no probe covers.

    Returns:
        The rewritten pointer, or the pointer unchanged when it already names a probe, starts
        at a non-state workflow root, or cannot be placed.
    """
    try:
        keys = split(pointer)
    except ValueError:
        return pointer
    root: list[str] = []
    if rooted:
        if not keys or keys[0] not in WORKFLOW_STATES:
            return pointer
        root, keys = keys[:1], keys[1:]
    if not keys:
        return pointer
    if keys[0] in probes:
        return in_probe_style(pointer, state, probes, rooted=rooted)
    target = state_keys(state, "/" + "/".join(pointer.split("/")[len(root) + 1 :]))
    if target is None:
        return pointer
    best: tuple[str, list[str]] | None = None
    for name, probe in probes.items():
        try:
            covered = split(probe)
        except ValueError:
            continue
        if target[: len(covered)] == covered and (best is None or len(covered) > len(best[1])):
            best = (name, covered)
    if best is None and add:
        holder = _deepest_resolving(state, target)
        if holder is None:
            return pointer
        name = _fresh_name(probes)
        probes[name] = join(holder)
        best = (name, holder)
    if best is None:
        return pointer
    name, covered = best
    return join([*root, name, *target[len(covered) :]])


def in_probe_style(
    pointer: str, state: JsonValue, probes: Mapping[str, str], *, rooted: bool
) -> str:
    """A probe-coordinate pointer rewritten to its probe value's key style (:func:`map_style`).

    Args:
        pointer: ``/final_state/<probe>/...`` when ``rooted``, else ``/<probe>/...``.
        state: The start's raw initial state.
        probes: Probe name to raw-state pointer.
        rooted: Whether the pointer starts at a workflow root.

    Returns:
        The rewritten pointer, or the pointer unchanged when it is not under a probe that
        resolves or the style cannot decide it (the authoring check refuses those).
    """
    try:
        keys = split(pointer)
    except ValueError:
        return pointer
    root, keys = (keys[:1], keys[1:]) if rooted else ([], keys)
    if (rooted and root[0] not in WORKFLOW_STATES) or not keys or keys[0] not in probes:
        return pointer
    try:
        value = resolve(state, probes[keys[0]])
    except ValueError:
        return pointer
    styled = map_style(value, keys[1:])
    return pointer if styled is None else join([*root, keys[0], *styled])


def off_style(verifier: JsonObject, state: JsonValue, probes: Mapping[str, str]) -> list[str]:
    """Verifier pointers not in their probe value's key style, for the authoring check.

    A pointer walking into a files map as nested segments, or several keys deep into an empty
    object, names a shape no simulated run produces: the files map holds each file under one
    flat key.
    """
    rows: list[str] = []
    for pointer, rooted in _pointers(verifier):
        if in_probe_style(pointer, state, probes, rooted=rooted) != pointer:
            rows.append(pointer)
            continue
        try:
            keys = split(pointer)
        except ValueError:
            continue
        keys = keys[1:] if rooted else keys
        if not keys or keys[0] not in probes:
            continue
        try:
            value = resolve(state, probes[keys[0]])
        except ValueError:
            continue
        if map_style(value, keys[1:]) is None:
            rows.append(pointer)
    return rows


def _pointers(verifier: JsonObject) -> list[tuple[str, bool]]:
    """Every verifier pointer and whether it starts at a workflow root."""
    rows: list[tuple[str, bool]] = []

    def listed(key: str) -> list[JsonValue]:
        value = verifier.get(key)
        return value if isinstance(value, list) else []

    rows.extend((p, True) for p in listed("required_paths") if isinstance(p, str))
    rows.extend((p, False) for p in listed("preserved_paths") if isinstance(p, str))
    for key, sides in (("rules", ("path",)), ("relations", ("left", "right"))):
        for row in listed(key):
            if isinstance(row, dict):
                rows.extend(
                    (pointer, True) for side in sides if isinstance(pointer := row.get(side), str)
                )
    return rows


def _deepest_resolving(state: JsonValue, keys: list[str]) -> list[str] | None:
    """The longest prefix of ``keys`` below a top-level domain that resolves in the state."""
    for count in range(len(keys), 1, -1):
        try:
            resolve(state, join(keys[:count]))
        except ValueError:
            continue
        return keys[:count]
    return None


def _fresh_name(probes: Mapping[str, str]) -> str:
    """A probe name not yet taken."""
    index = 0
    while f"{VERIFIER_PROBE}_{index}" in probes:
        index += 1
    return f"{VERIFIER_PROBE}_{index}"


def project_verifier(
    verifier: JsonObject, state: JsonValue, probes: dict[str, str], *, add: bool
) -> tuple[JsonObject, list[str]]:
    """Rewrite every verifier pointer onto the probes; return the verifier and what changed.

    Args:
        verifier: The authored verifier.
        state: The start's raw initial state.
        probes: Probe name to raw-state pointer; extended in place when ``add``.
        add: Whether probes may be added for uncovered paths.

    Returns:
        The verifier with rewritten ``required_paths``, rule paths, relation sides and
        ``preserved_paths``, and one note per rewritten pointer.
    """
    notes: list[str] = []
    result = dict(verifier)

    def move(pointer: JsonValue, *, rooted: bool) -> JsonValue:
        if not isinstance(pointer, str):
            return pointer
        moved = projected(pointer, state, probes, rooted=rooted, add=add)
        if moved != pointer:
            notes.append(f"verifier pointer {pointer!r} is graded as {moved!r}")
        return moved

    for key, rooted in (("required_paths", True), ("preserved_paths", False)):
        rows = result.get(key)
        if isinstance(rows, list):
            result[key] = [move(row, rooted=rooted) for row in rows]
    for key, sides in (("rules", ("path",)), ("relations", ("left", "right"))):
        rows = result.get(key)
        if isinstance(rows, list):
            result[key] = [
                {**row, **{side: move(row.get(side), rooted=True) for side in sides}}
                if isinstance(row, dict)
                else row
                for row in rows
            ]
    return result, notes


def unplaced(verifier: JsonObject, probes: Mapping[str, str]) -> list[str]:
    """Verifier pointers that name nothing the grade observes, for the authoring check."""
    rows: list[str] = []

    def check(pointer: JsonValue, *, rooted: bool) -> None:
        if not isinstance(pointer, str):
            return
        try:
            keys = split(pointer)
        except ValueError:
            rows.append(pointer)
            return
        if rooted:
            if not keys or keys[0] not in WORKFLOW_ROOTS:
                rows.append(pointer)
                return
            if keys[0] not in WORKFLOW_STATES:
                return
            keys = keys[1:]
        if not keys or keys[0] not in probes:
            rows.append(pointer)

    def listed(key: str) -> list[JsonValue]:
        value = verifier.get(key)
        return value if isinstance(value, list) else []

    for pointer in listed("required_paths"):
        check(pointer, rooted=True)
    for pointer in listed("preserved_paths"):
        check(pointer, rooted=False)
    for row in listed("rules"):
        if isinstance(row, dict):
            check(row.get("path"), rooted=True)
    for row in listed("relations"):
        if isinstance(row, dict):
            check(row.get("left"), rooted=True)
            check(row.get("right"), rooted=True)
    return rows
