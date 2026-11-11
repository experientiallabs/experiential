import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { currentRun, EventLog, readEventLog, TEXT_FLUSH_MS } from "./event-log.js";
import type { RunEvent } from "./protocol.js";

let dir: string;
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "exp-code-log-"));
});
afterEach(() => rm(dir, { recursive: true, force: true }));

describe("EventLog", () => {
  it("writes to the file before listeners see an event and continues seqs across instances", () => {
    const path = join(dir, "events.jsonl");
    const seen: number[] = [];
    const log = new EventLog(path, () => 7);
    log.onWritten((event) => seen.push(readEventLog(path).filter((written) => written.seq === event.seq).length));
    log.emit({ type: "error", message: "a" });
    log.emit({ type: "error", message: "b" });
    expect(seen).toEqual([1, 1]);
    const again = new EventLog(path);
    again.emit({ type: "error", message: "c" });
    expect(readEventLog(path).map((event) => event.seq)).toEqual([1, 2, 3]);
  });

  it("coalesces text and flushes on a different event type or after the interval", async () => {
    const path = join(dir, "events.jsonl");
    const log = new EventLog(path);
    log.emit({ type: "text", delta: "Hel" });
    log.emit({ type: "text", delta: "lo" });
    log.emit({ type: "thinking", delta: "hm" });
    log.emit({ type: "error", message: "x" });
    log.emit({ type: "text", delta: "!" });
    await new Promise((resolve) => setTimeout(resolve, TEXT_FLUSH_MS + 50));
    expect(readEventLog(path).map((event) => [event.type, "delta" in event ? event.delta : ""])).toEqual([
      ["text", "Hello"],
      ["thinking", "hm"],
      ["error", ""],
      ["text", "!"]
    ]);
  });

  it("finds the current run and whether it ended", () => {
    const start = (seq: number): RunEvent => ({ seq, ts: 0, type: "run_start", runId: "r", model: "m", prompt: "p", attachments: [], restored: false });
    const end = (seq: number): RunEvent => ({ seq, ts: 0, type: "run_end", status: "complete", turns: 0, costUsd: 0, durationMs: 0 });
    expect(currentRun([start(1), end(2)])).toMatchObject({ started: true, ended: true });
    expect(currentRun([start(1), end(2), start(3)])).toMatchObject({ started: true, ended: false, events: [start(3)] });
    expect(currentRun([])).toMatchObject({ started: false, ended: false });
  });
});
