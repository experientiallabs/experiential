"""CLI tests for `wmo runs`, driven via CliRunner over a mocked platform."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner, Result

from wmo.cli import runs_app as runs_module
from wmo.cli.app import app
from wmo.core.types import JsonObject
from wmo.platform.client import PlatformClient
from wmo.runs.reader import EventPage, EventRow, RunDetail, RunsReader
from wmo.runs.schema import RUN_SEQ_BAND

runner = CliRunner()

ORG = "org-1"
RUN = "jt/grid-c2/identity"
CREATED = "2026-07-27T09:00:00+00:00"
LEDGER_TS = "2026-07-27T10:00:00+00:00"
MERGED_TS = "2026-07-27T11:30:00+00:00"

_RUN_ROW: JsonObject = {
    "id": "run-1",
    "org_id": ORG,
    "external_id": RUN,
    "kind": "grid_arm",
    "status": "running",
    "benchmark": "tau-bench",
    "arm": "identity",
    "progress": {"done": 220, "total": 440, "scored": 190},
    "candidate_usd": 6.5,
    "compressor_usd": 0.25,
    "wm_usd": 18.0,
    "started_at": CREATED,
    "heartbeat_at": LEDGER_TS,
    "last_seq": 100002,
    "created_at": CREATED,
    "updated_at": LEDGER_TS,
}


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give rich a wide terminal, so a table assertion is about content and not about ellipses.

    `CliRunner` presents an 80-column non-terminal, which truncates a run name like
    `jt/grid-c2/identity` mid-cell. Rich reads `COLUMNS` per render, so this is enough.
    """
    monkeypatch.setenv("COLUMNS", "200")


def _invoke(*args: str) -> Result:
    return runner.invoke(app, ["runs", *args])


def _reader(handler: Callable[[httpx.Request], httpx.Response]) -> RunsReader:
    return RunsReader(
        PlatformClient("https://api.test", "xpl_secret", transport=httpx.MockTransport(handler)),
        ORG,
    )


def _connect(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    """Make the CLI's reader talk to a mocked platform instead of a real credential."""
    monkeypatch.setattr(
        runs_module.RunsReader, "open", classmethod(lambda cls, **_: _reader(handler))
    )


# -- artifacts -----------------------------------------------------------------------------------


def _write_grid(root: Path) -> Path:
    """A minimal but real grid directory: cohort, ledger, one finished chunk, one in-flight one.

    Small on purpose (2 cells per chunk instead of 110). What it has to exercise is the SHAPE the
    mapping walks: a ledger'd chunk whose file exists, a partial sidecar with no ledger line yet,
    and a merge meta that makes the arm terminal.
    """
    grid = root / ".wmo" / "jt" / "grid-t1"
    arm = grid / "identity"
    arm.mkdir(parents=True)
    (grid / "cohort.json").write_text(
        json.dumps(
            {
                "tip_sha": "abc123def456",
                "max_steps": 20,
                "episodes": 1,
                "scenarios": 2,
                "chunk_size": 1,
                "history_chars": 2000,
                "model_dir": str(root / ".wmo" / "models" / "tau-bench"),
                "pool_file": str(grid / "pool.toml"),
                "traces_file": str(root / "traces.otel.jsonl"),
                "created": CREATED,
            }
        ),
        encoding="utf-8",
    )
    (grid / "ledger.jsonl").write_text(
        "\n".join(
            json.dumps(line)
            for line in (
                {
                    "event": "chunk",
                    "arm": "identity",
                    "chunk": 0,
                    "cells": 2,
                    "scored": 1,
                    "candidate_usd": 0.5,
                    "compressor_usd": 0.0,
                    "wm_usd": 2.0,
                    "wall_s": 12.0,
                    "ts": LEDGER_TS,
                    "cumulative_usd": 2.5,
                    "tip_sha": "abc123def456",
                    "max_steps": 20,
                    "episodes": 1,
                    "note": "",
                },
                {
                    "event": "merge",
                    "arm": "identity",
                    "chunk": None,
                    "cells": 2,
                    "scored": 1,
                    "candidate_usd": 0.0,
                    "compressor_usd": 0.0,
                    "wm_usd": 0.0,
                    "wall_s": 0.0,
                    "ts": MERGED_TS,
                    "cumulative_usd": 2.5,
                    "tip_sha": "abc123def456",
                    "max_steps": 20,
                    "episodes": 1,
                    "note": "1 chunk(s)",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (arm / "chunk-0.json").write_text(
        json.dumps(
            {
                "pool": [{"name": "haiku", "provider": "bedrock", "model": "haiku-4-5"}],
                "outcomes": [
                    _outcome("s1", reward=0.75),
                    _outcome("s2", reward=None, error="ThrottlingException"),
                ],
            }
        ),
        encoding="utf-8",
    )
    # An in-flight chunk: a sidecar with a header and one row, and no ledger line of its own.
    (arm / "chunk-1.json.partial.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "version": 1,
                        "identity": {
                            "pool": "poolhash",
                            "scenarios": ["s3"],
                            "episodes": 1,
                            "max_steps": 20,
                            "history_chars": 2000,
                            "compression": "none",
                        },
                    }
                ),
                json.dumps(_outcome("s3", reward=0.5)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (arm / "matrix.meta.json").write_text(
        json.dumps({"arm": "identity", "cells": 2, "scored": 1, "merged_at": MERGED_TS}),
        encoding="utf-8",
    )
    return grid


def _outcome(scenario: str, *, reward: float | None, error: str | None = None) -> JsonObject:
    return {
        "scenario_id": scenario,
        "task": "book a flight",
        "model": "haiku",
        "episode": 0,
        "reward": reward,
        "success": reward is not None,
        "critique": "fine",
        "steps": 3,
        "stop_reason": "done" if reward is not None else "",
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        "cost_usd": 0.01,
        "call_seconds": [1.5],
        "replies": ["{}"],
        "error": error,
        "remeasured": False,
        "tokens_in_raw": 0,
        "tokens_in_compressed": 0,
        "compressor_id": "",
        "compressor_version": "",
        "aggressiveness": 0.0,
        "compressor_latency_s": 0.0,
        "compressor_cost_usd": 0.0,
    }


def _write_manifest(root: Path) -> Path:
    """A minimal `wmo optimize model` manifest, where the pipeline path reads it from."""
    run_dir = root / ".wmo" / "models" / "tau-toy" / "optimize"
    run_dir.mkdir(parents=True)
    manifest = run_dir / "optimize-run.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "world_model": "tau-toy",
                "lifetime_spend_usd": 3.25,
                "stages": [
                    {
                        "stage": "sweep",
                        "fingerprint": {"pool": "poolhash"},
                        "artifact_path": "matrix.json",
                        "artifact_identity": "deadbeef",
                        "completed_at": "2026-07-27T08:00:00+00:00",
                        "spend_usd": 1.0,
                        "compressor_spend_usd": 0.0,
                        "world_model_spend_usd": 2.25,
                    },
                    {
                        "stage": "report",
                        "fingerprint": {"policy": "policyhash"},
                        "artifact_path": "report.json",
                        "artifact_identity": "cafe",
                        "completed_at": "2026-07-27T08:10:00+00:00",
                        "spend_usd": 0.0,
                        "compressor_spend_usd": 0.0,
                        "world_model_spend_usd": 0.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


# -- backfill ------------------------------------------------------------------------------------


def test_backfill_dry_run_walks_a_grid_arm_into_banded_events(tmp_path: Path) -> None:
    """The grid mapping: run.meta, a chunk's cells in its own band, ledger lines, a merge status."""
    grid = _write_grid(tmp_path)
    out = tmp_path / "events.jsonl"

    result = _invoke("backfill", str(grid), "--arm", "identity", "--dry-run", "--out", str(out))

    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [event["type"] for event in events] == [
        "run.meta",
        "cell.batch",
        "ledger.line",
        "ledger.line",
        "cell.batch",
        "heartbeat",
        "run.status",
    ]
    assert {event["external_id"] for event in events} == {"jt/grid-t1/identity"}
    # Chunk 0's cells sit in band 1; everything else, the in-flight sidecar's cell included, is the
    # run-level walk in band 0.
    by_type = {event["type"]: event for event in events}
    cells = [event for event in events if event["type"] == "cell.batch"]
    assert RUN_SEQ_BAND < cells[0]["seq"] <= 2 * RUN_SEQ_BAND
    assert cells[1]["seq"] <= RUN_SEQ_BAND
    # Every timestamp comes off the artifacts, never the clock.
    assert by_type["run.meta"]["ts"] == CREATED
    assert by_type["run.status"]["ts"] == MERGED_TS
    assert by_type["run.status"]["payload"] == {"status": "completed", "finished_at": MERGED_TS}
    assert by_type["heartbeat"]["payload"]["progress"] == {"done": 2, "total": None, "scored": 1}
    assert by_type["heartbeat"]["payload"]["spend"]["wm_usd"] == 2.0


def test_backfill_is_deterministic_across_runs(tmp_path: Path) -> None:
    """Same artifacts, same events, byte for byte: what makes a replay a free no-op."""
    grid = _write_grid(tmp_path)
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"

    for out in (first, second):
        result = _invoke("backfill", str(grid), "--arm", "identity", "--dry-run", "--out", str(out))
        assert result.exit_code == 0, result.output

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_backfill_finds_every_arm_from_the_ledger_and_the_directories(tmp_path: Path) -> None:
    """Without --arm, every arm the grid mentions is replayed as its own run."""
    grid = _write_grid(tmp_path)
    (grid / "ledger.jsonl").open("a", encoding="utf-8").write(
        json.dumps(
            {
                "event": "stop",
                "arm": "truncate",
                "cells": 0,
                "scored": 0,
                "candidate_usd": 0.0,
                "compressor_usd": 0.0,
                "wm_usd": 0.0,
                "wall_s": 0.0,
                "ts": LEDGER_TS,
                "cumulative_usd": 0.0,
                "tip_sha": "abc123def456",
                "max_steps": 20,
                "episodes": 1,
                "note": "cap",
            }
        )
        + "\n"
    )
    out = tmp_path / "events.jsonl"

    result = _invoke("backfill", str(grid), "--dry-run", "--out", str(out))

    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {event["external_id"] for event in events} == {
        "jt/grid-t1/identity",
        "jt/grid-t1/truncate",
    }


def test_backfill_names_a_pipeline_run_from_the_manifest(tmp_path: Path) -> None:
    """A manifest's own `world_model` names the run, not the directory someone copied it to."""
    manifest = _write_manifest(tmp_path)
    out = tmp_path / "events.jsonl"

    result = _invoke("backfill", str(manifest), "--dry-run", "--out", str(out))

    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {event["external_id"] for event in events} == {"tau-toy/optimize"}
    assert [event["type"] for event in events] == [
        "run.meta",
        "stage.upsert",
        "stage.upsert",
        "heartbeat",
        "run.status",
    ]
    assert all(event["seq"] <= RUN_SEQ_BAND for event in events), "a pipeline uses band 0 alone"


def test_backfill_accepts_a_model_directory(tmp_path: Path) -> None:
    """Pointing at the world model, not its manifest, is the same run."""
    _write_manifest(tmp_path)
    out = tmp_path / "events.jsonl"

    result = _invoke(
        "backfill", str(tmp_path / ".wmo" / "models" / "tau-toy"), "--dry-run", "--out", str(out)
    )

    assert result.exit_code == 0, result.output
    assert "tau-toy/optimize" in out.read_text(encoding="utf-8")


def test_backfill_refuses_a_path_that_is_neither(tmp_path: Path) -> None:
    """A mistyped path says what was looked for in both shapes."""
    result = _invoke("backfill", str(tmp_path), "--dry-run")

    assert result.exit_code != 0
    assert "cohort.json" in result.output
    assert "optimize-run.json" in result.output


def test_backfill_pushes_and_reports_what_was_already_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A push says how many events were new, so a replay reads as the no-op it is.

    Also covers the single-client path: the guard's read and the push travel over ONE connection
    pool, which is why one handler serves both here.
    """
    grid = _write_grid(tmp_path)
    pushed: list[JsonObject] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = json.loads(request.read())
            events = body["events"]
            pushed.extend(dict(event) for event in events)
            # Two of the seven were already held: the replay case.
            return httpx.Response(
                200, json={"accepted": len(events) - 2, "last_seq": 100002, "control": []}
            )
        # The backfill guard's probe: a run the platform has never heard of.
        return httpx.Response(404, json={"error": "Run not found"})

    _connect(monkeypatch, handler)

    result = _invoke("backfill", str(grid), "--arm", "identity")

    assert result.exit_code == 0, result.output
    assert len(pushed) == 7
    assert "already recorded" in result.output
    # No event carries the run in its body: on the wire the run is named in the URL.
    assert all("external_id" not in event for event in pushed)


def test_backfill_refuses_a_run_that_already_reported_itself(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A live-emitted run must not be merged with a replay, or its spend curve doubles."""
    grid = _write_grid(tmp_path)
    posts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            posts.append(request)
            raise AssertionError("a refused backfill must not push")
        return httpx.Response(200, json={"run": _RUN_ROW, "event_count": 42})

    _connect(monkeypatch, handler)

    result = _invoke("backfill", str(grid), "--arm", "identity")

    assert result.exit_code != 0
    assert "--force" in result.output
    assert posts == [], "the refusal happens before anything is sent"


# -- reads ---------------------------------------------------------------------------------------


def test_list_shows_progress_and_spend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The list is the panel's table: what is running, how far in, what it cost."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/api/orgs/{ORG}/runs"
        return httpx.Response(200, json={"runs": [_RUN_ROW], "next_cursor": None})

    _connect(monkeypatch, handler)

    result = _invoke("list")

    assert result.exit_code == 0, result.output
    assert RUN in result.output
    assert "220/440" in result.output
    # candidate + world model, and NOT the compressor subset on top of it.
    assert "$24.50" in result.output


def test_list_json_is_machine_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    """An agent reads this as often as a person does."""
    _connect(
        monkeypatch,
        lambda request: httpx.Response(200, json={"runs": [_RUN_ROW], "next_cursor": None}),
    )

    result = _invoke("list", "--json")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["runs"][0]["external_id"] == RUN


def test_show_renders_stages_and_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    """One run's detail: stage table, per-candidate cells, and pending commands."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "run": _RUN_ROW,
                "org": {"id": ORG, "name": "Acme", "slug": "acme"},
                "event_count": 64,
                "stages": [
                    {
                        "run_id": "run-1",
                        "stage": "sweep",
                        "status": "completed",
                        "candidate_usd": 6.5,
                        "wm_usd": 18.0,
                        "completed_at": LEDGER_TS,
                        "updated_at": LEDGER_TS,
                    }
                ],
                "cell_stats": [
                    {
                        "model": "haiku",
                        "cell_count": 40,
                        "scored_count": 38,
                        "error_count": 2,
                        "unpriced_count": 0,
                        "cost_usd_total": 1.25,
                        "reward_mean": 0.62,
                    }
                ],
                "pending_control": [
                    {
                        "id": "c1",
                        "run_id": "run-1",
                        "command": "stop",
                        "args": {},
                        "status": "pending",
                        "requested_by": "user-1",
                        "created_at": LEDGER_TS,
                    }
                ],
            },
        )

    _connect(monkeypatch, handler)

    result = _invoke("show", RUN)

    assert result.exit_code == 0, result.output
    assert "sweep" in result.output
    assert "haiku" in result.output
    assert "0.620" in result.output
    assert "stop" in result.output


def test_stop_queues_a_pull_based_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stopping asks; it does not kill, and it does not change the run's status itself."""
    seen: list[JsonObject] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.read()))
        return httpx.Response(
            200,
            json={
                "control": {
                    "id": "c7",
                    "run_id": "run-1",
                    "command": "stop",
                    "args": {},
                    "status": "pending",
                    "requested_by": "user-1",
                    "created_at": LEDGER_TS,
                }
            },
        )

    _connect(monkeypatch, handler)

    result = _invoke("stop", RUN)

    assert result.exit_code == 0, result.output
    assert seen == [{"command": "stop", "args": {}}]
    assert "queued" in result.output
    assert "next reports in" in result.output


def test_retry_passes_the_chunk_as_an_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--chunk` narrows the request the runner will answer (or reasonably refuse)."""
    seen: list[JsonObject] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.read()))
        return httpx.Response(
            200,
            json={
                "control": {
                    "id": "c8",
                    "run_id": "run-1",
                    "command": "retry_unscored",
                    "args": {"chunk": 3},
                    "status": "pending",
                    "requested_by": "user-1",
                    "created_at": LEDGER_TS,
                }
            },
        )

    _connect(monkeypatch, handler)

    result = _invoke("retry", RUN, "--chunk", "3")

    assert result.exit_code == 0, result.output
    assert seen == [{"command": "retry_unscored", "args": {"chunk": 3}}]


def test_reads_without_a_login_say_what_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not connected is a state with a next step, never a traceback."""
    monkeypatch.setattr(runs_module.RunsReader, "open", classmethod(lambda cls, **_: None))

    result = _invoke("list")

    assert result.exit_code != 0
    assert "wmo login" in result.output


class _FakeReader:
    """A reader whose backlog and stream are fixed, for driving the tail without a platform."""

    def __init__(
        self,
        backlog: list[JsonObject],
        streamed: list[JsonObject],
        *,
        filtered: list[JsonObject] | None = None,
        statuses: list[str] | None = None,
    ) -> None:
        self._backlog = backlog
        self._streamed = streamed
        self._filtered = filtered or []
        self._statuses = statuses or ["completed"]
        self.resumed_from: int | None = None
        self.filters: list[str | None] = []
        self.cursors: list[int] = []

    def __enter__(self) -> _FakeReader:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def list_events(
        self,
        external_id: str,
        *,
        after_pos: int = 0,
        limit: int = 500,
        tail: bool = False,
        event_type: str | None = None,
    ) -> EventPage:
        self.filters.append(event_type)
        self.cursors.append(after_pos)
        source = self._filtered if event_type is not None and after_pos else self._backlog
        rows = tuple(EventRow.model_validate(row) for row in source)
        return EventPage(events=rows, last_pos=rows[-1].pos if rows else after_pos)

    def get_run(self, external_id: str) -> RunDetail:
        """Answer `running` until the statuses run out, so a poll loop takes a lap then ends."""
        status = self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]
        return RunDetail.model_validate({"run": {**_RUN_ROW, "status": status}, "event_count": 3})

    def tail(self, external_id: str, *, after_pos: int = 0) -> Iterator[EventRow]:
        self.resumed_from = after_pos
        for row in self._streamed:
            yield EventRow.model_validate(row)


def test_tail_prints_the_backlog_then_follows_from_where_it_ended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tail paints history first and then follows, resuming at the last position it printed."""
    reader = _FakeReader(
        backlog=[
            {
                "pos": 8,
                "seq": 1,
                "type": "ledger.line",
                "payload": {"event": "chunk", "chunk": 0, "cells": 2, "scored": 1},
                "ts": LEDGER_TS,
            }
        ],
        streamed=[
            {
                "pos": 9,
                "seq": 100001,
                "type": "cell.batch",
                "payload": {"cells": [{}, {}]},
                "ts": LEDGER_TS,
            }
        ],
    )
    monkeypatch.setattr(runs_module.RunsReader, "open", classmethod(lambda cls, **_: reader))

    result = _invoke("tail", RUN)

    assert result.exit_code == 0, result.output
    assert "ledger.line" in result.output
    assert "2 cell(s)" in result.output
    # Resumed by `pos`, from the last event the backlog printed, so nothing is repeated or skipped.
    assert reader.resumed_from == 8


def test_backfill_name_overrides_a_moved_grids_prefix(tmp_path: Path) -> None:
    """Artifacts copied out of `.wmo` no longer say which run they are, so `--name` says it.

    The arm still appends: one grid directory holds several arms and each is its own run, so a
    `--name` that replaced the whole run name would collapse them into one.
    """
    grid = _write_grid(tmp_path)
    moved = tmp_path / "restored"
    moved.mkdir()
    for child in grid.iterdir():
        target = moved / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)
    out = tmp_path / "events.jsonl"

    result = _invoke(
        "backfill",
        str(moved),
        "--arm",
        "identity",
        "--name",
        "jt/grid-c2",
        "--dry-run",
        "--out",
        str(out),
    )

    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {event["external_id"] for event in events} == {"jt/grid-c2/identity"}


def test_backfill_name_is_the_whole_run_name_for_a_manifest(tmp_path: Path) -> None:
    """For a manifest there is one run, so `--name` is that run's name verbatim."""
    manifest = _write_manifest(tmp_path)
    out = tmp_path / "events.jsonl"

    result = _invoke(
        "backfill", str(manifest), "--name", "tau-jt-toy/optimize", "--dry-run", "--out", str(out)
    )

    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {event["external_id"] for event in events} == {"tau-jt-toy/optimize"}


def test_backfill_without_name_still_derives_from_the_path(tmp_path: Path) -> None:
    """Default behavior is unchanged: artifacts in place need no naming."""
    grid = _write_grid(tmp_path)
    out = tmp_path / "events.jsonl"

    result = _invoke("backfill", str(grid), "--arm", "identity", "--dry-run", "--out", str(out))

    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {event["external_id"] for event in events} == {"jt/grid-t1/identity"}


def test_tail_of_one_event_type_polls_the_filtered_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--type` pages the server's filtered read instead of draining the whole stream.

    The server filters in the database, so a ledger-line tail reads the handful of events it wants;
    filtering an SSE stream client-side would pull every cell batch of a grid over the wire to throw
    it away. The poll also has to END, which it does through `is_terminal_status`.
    """
    ledger: JsonObject = {
        "pos": 8,
        "seq": 2,
        "type": "ledger.line",
        "payload": {"event": "chunk", "chunk": 0, "cells": 2, "scored": 1},
        "ts": LEDGER_TS,
    }
    later: JsonObject = {**ledger, "pos": 9, "seq": 3}
    reader = _FakeReader(
        backlog=[ledger], streamed=[], filtered=[later], statuses=["running", "completed"]
    )
    monkeypatch.setattr(runs_module, "_sleep_between_polls", lambda: None)
    monkeypatch.setattr(runs_module.RunsReader, "open", classmethod(lambda cls, **_: reader))

    result = _invoke("tail", RUN, "--type", "ledger.line")

    assert result.exit_code == 0, result.output
    # Never opened the stream, and every read carried the filter.
    assert reader.resumed_from is None
    assert set(reader.filters) == {"ledger.line"}
    # Paged forward by pos rather than re-reading the same window.
    assert reader.cursors == [0, 8, 9]
    assert "chunk=0" in result.output
