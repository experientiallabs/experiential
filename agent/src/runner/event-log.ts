/**
 * The run's append-only event log (events.jsonl).
 *
 * Every event is appended to the file synchronously before any listener (the HTTP push) sees it,
 * so the file is always the complete record. Seqs continue from the highest seq already in the
 * file, so a restarted runner or a later run of the same id never reuses one. Assistant text and
 * thinking deltas are coalesced and flushed at most every 100 ms.
 */

import { appendFileSync, existsSync, readFileSync } from "node:fs";
import { parseRunEvent, type RunEvent, type RunEventBody } from "./protocol.js";

export const TEXT_FLUSH_MS = 100;

/** Every parseable event already in a log file, in file order. */
export function readEventLog(path: string): RunEvent[] {
  if (!existsSync(path)) return [];
  return readFileSync(path, "utf8")
    .split("\n")
    .map(parseRunEvent)
    .filter((event): event is RunEvent => event !== null);
}

/**
 * The events of the latest run in a log: everything from the last run_start on, and whether that
 * run reached run_end.
 */
export function currentRun(events: RunEvent[]): { events: RunEvent[]; started: boolean; ended: boolean } {
  let start = -1;
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (events[index]!.type === "run_start") {
      start = index;
      break;
    }
  }
  if (start < 0) return { events: [], started: false, ended: false };
  const run = events.slice(start);
  return { events: run, started: true, ended: run.some((event) => event.type === "run_end") };
}

export class EventLog {
  private seqValue: number;
  private pendingText: { type: "text" | "thinking"; delta: string } | null = null;
  private flushTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly listeners: ((event: RunEvent) => void)[] = [];

  constructor(
    private readonly path: string,
    private readonly now: () => number = Date.now,
    existing: RunEvent[] = readEventLog(path)
  ) {
    this.seqValue = existing.reduce((max, event) => Math.max(max, event.seq), 0);
  }

  /** The last seq written. */
  get seq(): number {
    return this.seqValue;
  }

  /** Observe events after they are in the file. */
  onWritten(listener: (event: RunEvent) => void): void {
    this.listeners.push(listener);
  }

  /** Emit one event; text and thinking deltas are coalesced, everything else is written now. */
  emit(body: RunEventBody): void {
    if (body.type === "text" || body.type === "thinking") {
      if (this.pendingText !== null && this.pendingText.type !== body.type) this.flushText();
      this.pendingText = this.pendingText === null ? { type: body.type, delta: body.delta } : { type: body.type, delta: this.pendingText.delta + body.delta };
      if (this.flushTimer === null) this.flushTimer = setTimeout(() => this.flushText(), TEXT_FLUSH_MS);
      return;
    }
    this.flushText();
    this.write(body);
  }

  /** Write any coalesced text now. */
  flushText(): void {
    if (this.flushTimer !== null) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
    const pending = this.pendingText;
    this.pendingText = null;
    if (pending !== null && pending.delta.length > 0) this.write(pending);
  }

  private write(body: RunEventBody): void {
    this.seqValue += 1;
    const event = { seq: this.seqValue, ts: this.now(), ...body } as RunEvent;
    appendFileSync(this.path, `${JSON.stringify(event)}\n`);
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch {
        // A listener (the push) must never break the log.
      }
    }
  }
}
