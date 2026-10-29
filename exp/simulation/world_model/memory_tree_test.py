"""Positional log, ordered compression and budgeted overview without any provider."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject, sha256_json
from exp.common.models.client import RequestCapacityExceeded
from exp.common.project import ArtifactStore, ProjectConfig, ProjectStore, artifact_input
from exp.runtime.models.structured_completion import CompletionNotDispatched, ReturnedOutputRejected
from exp.simulation.world_model.memory_tree import (
    LIFETIME_ATTEMPTS,
    LINE_BYTES,
    LineDraft,
    ambiguous_blocks,
    append_lines,
    build_pending,
    check_line,
    cover,
    fit_line,
    overview,
    pending,
    read_summary,
    refused_blocks,
    summary_id,
    summary_sources,
)

NOW = datetime(2026, 10, 15, tzinfo=UTC)


def source(store: ArtifactStore, name: str) -> ArtifactInput:
    """Persist one consumed-trajectory artifact and return its typed reference."""
    identity = "consumed-trajectory-" + name
    store.write_json(
        artifact_id=identity,
        artifact_type="consumed-trajectory",
        envelope=ArtifactEnvelope(schema_version=1, created_at=NOW, code_revision="fixture"),
        files={"trajectory.json": {"trajectory": {"task_lineage": name}}},
    )
    return artifact_input(store.read(identity).manifest)


def draft(text: str, origin: str, ref: ArtifactInput) -> LineDraft:
    """One observed line drafted from a source trajectory."""
    return LineDraft(kind="observed", text=text, origin=origin, derived_from=ref)


def store_with(tmp_path: Path, count: int) -> ArtifactStore:
    """A project whose lineage ``m`` holds ``count`` lines from alternating sources a/b."""
    project = ProjectStore(tmp_path, "tree")
    project.initialize(ProjectConfig(project_id="tree"))
    refs = [source(project.artifacts, "a"), source(project.artifacts, "b")]
    append_lines(
        project.artifacts,
        "m",
        0,
        [draft(f"line {i}", f"origin-{i}", refs[i % 2]) for i in range(count)],
        created_at=NOW,
        code_revision="fixture",
    )
    return project.artifacts


@pytest.mark.parametrize("budget", [1, 4, 16])
def test_cover_tiles_exactly_with_recent_detail(budget: int) -> None:
    """Aligned power-of-two blocks partition [0, T), coarse in the past, fine at present."""
    for total in range(1, 200):
        blocks = cover(total, budget)
        assert [lo for lo, _ in blocks] == sorted(lo for lo, _ in blocks)
        assert blocks[0][0] == 0 and blocks[-1][1] == total
        assert all(a[1] == b[0] for a, b in zip(blocks, blocks[1:], strict=False))
        for lo, hi in blocks:
            size = hi - lo
            assert size & (size - 1) == 0 and lo % size == 0
        assert len(blocks) <= max(budget, bin(total).count("1"))
        sizes = [hi - lo for lo, hi in blocks]
        assert sizes == sorted(sizes, reverse=True)
        if total <= budget:
            assert sizes == [1] * total


def test_lines_are_positional_bounded_and_never_trimmed(tmp_path: Path) -> None:
    """Same content replays; different content at a used position fails; models are not cut."""
    store = store_with(tmp_path, 2)
    append_lines(
        store,
        "m",
        1,
        [draft("line 1", "origin-1", ref_b(store))],
        created_at=NOW,
        code_revision="fixture",
    )
    with pytest.raises(ValueError):
        append_lines(
            store,
            "m",
            1,
            [draft("different", "origin-1", ref_b(store))],
            created_at=NOW,
            code_revision="fixture",
        )
    with pytest.raises(ValueError):
        check_line("two\nlines")
    with pytest.raises(ValueError):
        check_line("x" * (LINE_BYTES + 1))
    assert len(fit_line("é" * 400).encode()) <= LINE_BYTES


def test_compression_runs_in_order_reuses_proposals_and_rejects_overlong(tmp_path: Path) -> None:
    """Children before parents; a saved proposal is reused; bad output is rejected, not cut."""
    store = store_with(tmp_path, 5)
    assert pending(store, "m", 5) == [(0, 2), (2, 4), (0, 4)]
    calls: list[str] = []

    def complete(body: JsonObject, logical_id: str) -> JsonObject:
        """Return an overlong first answer for #0-1, then valid summaries."""
        calls.append(logical_id)
        if logical_id == summary_id("m", 0, 2) + "-attempt-1":
            return {"summary": "x" * (LINE_BYTES + 1)}
        return {"summary": f"summary of {len(calls)}"}

    built, remaining = build_pending(
        store, "m", 5, complete, limit=10, created_at=NOW, code_revision="fixture"
    )
    assert len(built) == 3 and remaining == 0
    first = read_summary(store, "m", 0, 2)
    assert first is not None and first.text == "summary of 2"
    assert store.exists(summary_id("m", 0, 2) + "-attempt-1-rejected")
    assert build_pending(
        store, "m", 5, complete, limit=10, created_at=NOW, code_revision="fixture"
    ) == ([], 0)
    assert len(calls) == 4


def test_ambiguous_compression_stops_instead_of_repurchasing(tmp_path: Path) -> None:
    """A saved request without a saved answer is never silently bought again."""
    store = store_with(tmp_path, 2)
    store.write_json(
        artifact_id=summary_id("m", 0, 2) + "-attempt-1-intent",
        artifact_type="memory-compression",
        envelope=ArtifactEnvelope(schema_version=1, created_at=NOW, code_revision="fixture"),
        files={"record.json": {"request": {}}},
    )

    def never(body: JsonObject, logical_id: str) -> JsonObject:
        """Fail if dispatch is attempted."""
        raise AssertionError(logical_id)

    assert build_pending(
        store, "m", 2, never, limit=1, created_at=NOW, code_revision="fixture"
    ) == ([], 1)
    assert ambiguous_blocks(store, "m", 2) == ["#0-1"]


def test_an_ambiguous_block_never_stalls_independent_blocks(tmp_path: Path) -> None:
    """Only the ambiguous block and its ancestors wait; every other block still builds."""
    store = store_with(tmp_path, 4)
    store.write_json(
        artifact_id=summary_id("m", 0, 2) + "-attempt-1-intent",
        artifact_type="memory-compression",
        envelope=ArtifactEnvelope(schema_version=1, created_at=NOW, code_revision="fixture"),
        files={"record.json": {"request": {}}},
    )
    seen: list[str] = []

    def answer(body: JsonObject, logical_id: str) -> JsonObject:
        """Record each dispatch."""
        seen.append(logical_id)
        return {"summary": "compressed"}

    built, remaining = build_pending(
        store, "m", 4, answer, limit=5, created_at=NOW, code_revision="fixture"
    )
    assert built == [summary_id("m", 2, 4)] and remaining == 2
    assert all(summary_id("m", 0, 2) not in call for call in seen)


@pytest.mark.parametrize(
    "refusal",
    [
        CompletionNotDispatched("completion deadline reached before dispatch"),
        RequestCapacityExceeded("request exceeds the pinned output reservation"),
    ],
)
def test_a_refusal_before_dispatch_closes_the_attempt_without_a_rejection(
    tmp_path: Path, refusal: Exception
) -> None:
    """Nothing was bought, so the block simply waits and builds on a later call."""
    store = store_with(tmp_path, 2)
    calls: list[JsonObject] = []

    def refuse_once(body: JsonObject, logical_id: str) -> JsonObject:
        """Refuse before dispatch first, then answer."""
        calls.append(body)
        if len(calls) == 1:
            raise refusal
        return {"summary": "fits"}

    first = build_pending(store, "m", 2, refuse_once, limit=3, created_at=NOW, code_revision="f")
    assert first == ([], 1) and ambiguous_blocks(store, "m", 2) == []
    second = build_pending(store, "m", 2, refuse_once, limit=3, created_at=NOW, code_revision="f")
    assert second == ([summary_id("m", 0, 2)], 0)
    assert "Rejected" not in json.dumps(calls[1])
    assert refused_blocks(store, "m", 2) == []


def test_overview_opens_unbuilt_and_withheld_blocks_without_a_model(tmp_path: Path) -> None:
    """Readers never compress; withheld lines vanish with every summary covering them."""
    store = store_with(tmp_path, 8)
    rows = overview(store, "m", 8, 3)
    assert [row["id"] for row in rows] == ["#0-5", "#6", "#7"]
    assert rows[0]["kind"] == "omitted" and "text" not in rows[0]
    build_pending(
        store,
        "m",
        8,
        lambda body, logical_id: {"summary": "compressed " + logical_id[-12:]},
        limit=10,
        created_at=NOW,
        code_revision="fixture",
    )
    compact = overview(store, "m", 8, 2)
    assert len(compact) <= 3 and compact[0]["kind"] == "summary"
    reserved = ref_b(store)
    hidden = overview(store, "m", 8, 8, admits=lambda ref: ref != reserved)
    assert [row["id"] for row in hidden] == ["#0", "#2", "#4", "#6"]
    assert all(row["kind"] != "summary" for row in hidden)


def test_summaries_carry_resolvable_ancestry_and_inherit_exclusion(tmp_path: Path) -> None:
    """A summary's children resolve to every source; excluding one hides all its ancestors."""
    store = store_with(tmp_path, 32)
    build_pending(
        store,
        "m",
        32,
        lambda body, logical_id: {"summary": "s " + logical_id[-10:]},
        limit=100,
        created_at=NOW,
        code_revision="fixture",
    )
    root = read_summary(store, "m", 0, 32)
    assert root is not None and len(root.children) == 2
    assert summary_sources(store, root) == {ref_a(store), ref_b(store)}
    pair = read_summary(store, "m", 0, 2)
    assert pair is not None and len(pair.children) == 2
    full = overview(store, "m", 32, 4)
    assert any(row["kind"] == "summary" for row in full)
    reserved = ref_a(store)
    visible = overview(store, "m", 32, 4, admits=lambda ref: ref != reserved)
    assert all(row["kind"] != "summary" for row in visible)
    assert [row["id"] for row in visible[1:]] == ["#27", "#29", "#31"]
    assert visible[0]["kind"] == "omitted"
    full_view = overview(store, "m", 32, 32, admits=lambda ref: ref != reserved)
    assert {row["id"] for row in full_view} == {f"#{i}" for i in range(1, 32, 2)}


def ref_a(store: ArtifactStore) -> ArtifactInput:
    """Reference of source a."""
    return artifact_input(store.read("consumed-trajectory-a").manifest)


def ref_b(store: ArtifactStore) -> ArtifactInput:
    """Reference of source b."""
    return artifact_input(store.read("consumed-trajectory-b").manifest)


def test_rejected_compression_retries_with_measured_feedback_and_never_stalls(
    tmp_path: Path,
) -> None:
    """A retry states the measured overshoot; later passes may try again; nothing is cut."""
    store = store_with(tmp_path, 2)
    requests: list[JsonObject] = []

    def overlong(body: JsonObject, logical_id: str) -> JsonObject:
        """Always overshoot the line limit by two bytes."""
        requests.append(body)
        return {"summary": "y" * (LINE_BYTES + 2)}

    built, remaining = build_pending(
        store, "m", 2, overlong, limit=3, created_at=NOW, code_revision="fixture"
    )
    assert built == [] and remaining == 1 and len(requests) == 3
    assert f"hard limit is {LINE_BYTES} bytes" in json.dumps(requests[0]["messages"])
    retry = json.dumps(requests[1])
    assert f"{LINE_BYTES + 2} bytes" in retry and "y" * 20 in retry
    accepted = "fits now"
    built, remaining = build_pending(
        store,
        "m",
        2,
        lambda body, logical_id: {"summary": accepted},
        limit=1,
        created_at=NOW,
        code_revision="fixture",
    )
    assert len(built) == 1 and remaining == 0
    summary = read_summary(store, "m", 0, 2)
    assert summary is not None and summary.text == accepted


def test_invalid_returned_compression_is_a_rejection_not_an_ambiguous_purchase(
    tmp_path: Path,
) -> None:
    """A completion owner refusing returned output closes the attempt; the retry explains why."""
    store = store_with(tmp_path, 2)
    requests: list[JsonObject] = []

    def refuse_then_fit(body: JsonObject, logical_id: str) -> JsonObject:
        """First response fails the owner's validation; the retry fits."""
        requests.append(body)
        if len(requests) == 1:
            raise ReturnedOutputRejected("returned summary failed structured validation")
        return {"summary": "fits"}

    built, remaining = build_pending(
        store, "m", 2, refuse_then_fit, limit=2, created_at=NOW, code_revision="fixture"
    )
    assert len(built) == 1 and remaining == 0
    assert store.exists(summary_id("m", 0, 2) + "-attempt-1-rejected")
    assert "failed structured validation" in json.dumps(requests[1])

    def transport(body: JsonObject, logical_id: str) -> JsonObject:
        """A failure without a returned response stays ambiguous."""
        raise TimeoutError("no response")

    more = store_with(tmp_path / "more", 2)
    with pytest.raises(TimeoutError):
        build_pending(more, "m", 2, transport, limit=1, created_at=NOW, code_revision="fixture")
    assert build_pending(
        more, "m", 2, transport, limit=1, created_at=NOW, code_revision="fixture"
    ) == ([], 1)
    assert ambiguous_blocks(more, "m", 2) == ["#0-1"]


def test_overview_overflow_is_bounded_and_explicit(tmp_path: Path) -> None:
    """Missing summaries never expand the prompt into the whole log."""
    store = store_with(tmp_path, 1024)
    rows = overview(store, "m", 1024, 32)
    assert len(rows) == 32
    marker = rows[0]
    assert marker["kind"] == "omitted" and marker["count"] == 1024 - 31
    assert marker["id"] == f"#0-{1024 - 32}"
    assert [row["id"] for row in rows[1:]] == [f"#{i}" for i in range(1024 - 31, 1024)]
    for budget in (1, 2, 3):
        assert len(overview(store, "m", 1024, budget)) == budget


def test_compression_allowance_counts_calls_not_blocks(tmp_path: Path) -> None:
    """A one-call allowance makes exactly one completion call."""
    store = store_with(tmp_path, 2)
    calls: list[str] = []

    def overlong(body: JsonObject, logical_id: str) -> JsonObject:
        """Always overshoot."""
        calls.append(logical_id)
        return {"summary": "z" * (LINE_BYTES + 1)}

    build_pending(store, "m", 2, overlong, limit=1, created_at=NOW, code_revision="fixture")
    assert len(calls) == 1


def test_a_refused_block_does_not_starve_independent_blocks(tmp_path: Path) -> None:
    """Independent blocks progress; parents of a refused block wait for it."""
    store = store_with(tmp_path, 4)
    calls: list[str] = []
    stubborn = summary_id("m", 0, 2)

    def complete(body: JsonObject, logical_id: str) -> JsonObject:
        """Block #0-1 is always too long; everything else fits."""
        calls.append(logical_id)
        if logical_id.startswith(stubborn):
            return {"summary": "z" * (LINE_BYTES + 1)}
        return {"summary": "ok"}

    for _ in range(2):
        build_pending(store, "m", 4, complete, limit=6, created_at=NOW, code_revision="fixture")
    assert read_summary(store, "m", 2, 4) is not None
    assert read_summary(store, "m", 0, 4) is None
    assert sum(c.startswith(stubborn) for c in calls) == LIFETIME_ATTEMPTS
    build_pending(store, "m", 4, complete, limit=6, created_at=NOW, code_revision="fixture")
    assert sum(c.startswith(stubborn) for c in calls) == LIFETIME_ATTEMPTS
    assert refused_blocks(store, "m", 4) == ["#0-1"]


def test_only_a_returned_output_rejection_is_retried(tmp_path: Path) -> None:
    """Other ValueErrors (for example accounting) leave the purchase ambiguous."""
    store = store_with(tmp_path, 2)

    def accounting(body: JsonObject, logical_id: str) -> JsonObject:
        """An owner error that does not prove a response returned."""
        raise ValueError("dispatch accounting unresolved")

    with pytest.raises(ValueError, match="accounting"):
        build_pending(store, "m", 2, accounting, limit=3, created_at=NOW, code_revision="fixture")
    assert build_pending(
        store, "m", 2, accounting, limit=3, created_at=NOW, code_revision="fixture"
    ) == ([], 1)
    assert ambiguous_blocks(store, "m", 2) == ["#0-1"]
    other = store_with(tmp_path / "other", 2)
    seen: list[str] = []

    def rejected_then_ok(body: JsonObject, logical_id: str) -> JsonObject:
        """The owner refuses returned output once."""
        seen.append(logical_id)
        if len(seen) == 1:
            raise ReturnedOutputRejected("returned arguments failed the response schema")
        return {"summary": "ok"}

    built, _ = build_pending(
        other, "m", 2, rejected_then_ok, limit=3, created_at=NOW, code_revision="fixture"
    )
    assert len(built) == 1 and len(seen) == 2


def test_a_summary_covering_an_inadmissible_line_is_opened(tmp_path: Path) -> None:
    """Line-level policy (for example retirement) applies to every summary above the line."""
    store = store_with(tmp_path, 4)
    build_pending(
        store,
        "m",
        4,
        lambda body, logical_id: {"summary": "compressed"},
        limit=10,
        created_at=NOW,
        code_revision="fixture",
    )
    assert [r["id"] for r in overview(store, "m", 4, 1)] == ["#0-3"]
    rows = overview(store, "m", 4, 2, admits_line=lambda line: line.origin != "origin-1")
    assert [r["id"] for r in rows] == ["#0", "#2-3"]


def test_replayed_proposal_certifies_the_request_actually_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saved answer is bound to its saved request, even after the prompt changes."""
    from exp.simulation.world_model import memory_tree

    store = store_with(tmp_path, 2)
    identity = summary_id("m", 0, 2)
    sent = memory_tree.compression_request(store, "m", 0, 2)
    memory_tree._save(store, identity + "-attempt-1-intent", {"request": sent}, NOW, "fixture")
    memory_tree._save(store, identity + "-attempt-1-proposal", {"summary": "ok"}, NOW, "fixture")
    monkeypatch.setattr(memory_tree, "TARGET_BYTES", 120)
    assert memory_tree.compression_request(store, "m", 0, 2) != sent

    def never(body: JsonObject, logical_id: str) -> JsonObject:
        """Replay must not call the provider."""
        raise AssertionError(logical_id)

    built, _ = build_pending(store, "m", 2, never, limit=3, created_at=NOW, code_revision="fixture")
    summary = read_summary(store, "m", 0, 2)
    assert built and summary is not None
    assert summary.request_sha256 == sha256_json(sent)
