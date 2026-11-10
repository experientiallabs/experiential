/**
 * Batched HTTP push of run events.
 *
 * Events reach the pusher only after they are appended to events.jsonl, so the file log is
 * always complete and authoritative; the push is a low-latency copy. Batches go out every
 * ~150 ms, or sooner when 200 events or ~256 KB are pending. A batch is accepted on any 2xx; the
 * receiver is idempotent by (runId, seq), so a failed batch is resent unchanged with backoff
 * (250 ms doubling to 5 s) while newer events wait for later batches. A 404 or 410 means the run
 * is gone and ends pushing for the rest of the run. Nothing here ever blocks the agent loop.
 */

import type { EventPushBody, RunEvent } from "./protocol.js";

export const PUSH_INTERVAL_MS = 150;
export const PUSH_MAX_EVENTS = 200;
export const PUSH_MAX_BYTES = 256 * 1024;
export const PUSH_BACKOFF_START_MS = 250;
export const PUSH_BACKOFF_MAX_MS = 5_000;
export const PUSH_FINAL_TIMEOUT_MS = 5_000;
const PUSH_REQUEST_TIMEOUT_MS = 10_000;

export interface EventPusherOptions {
  url: string;
  token: string;
  runId: string;
  fetch?: typeof globalThis.fetch;
  intervalMs?: number;
  maxEvents?: number;
  maxBytes?: number;
  backoffStartMs?: number;
  backoffMaxMs?: number;
}

interface Pending {
  event: RunEvent;
  bytes: number;
}

export class EventPusher {
  private readonly queue: Pending[] = [];
  private queuedBytes = 0;
  /** A batch that failed and must be resent before anything newer. */
  private retry: RunEvent[] | null = null;
  private sending = false;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private backoffMs: number;
  private stopped = false;
  private urgent = false;
  private idleWaiters: (() => void)[] = [];

  constructor(private readonly options: EventPusherOptions) {
    this.backoffMs = options.backoffStartMs ?? PUSH_BACKOFF_START_MS;
  }

  /** Whether the receiver said the run is gone. */
  get isStopped(): boolean {
    return this.stopped;
  }

  /** Queue one event that is already in the file log. Never throws and never waits. */
  enqueue(event: RunEvent): void {
    if (this.stopped) return;
    const bytes = Buffer.byteLength(JSON.stringify(event));
    this.queue.push({ event, bytes });
    this.queuedBytes += bytes;
    if (this.sending || this.retry !== null) return;
    if (this.urgent || this.queue.length >= this.maxEvents() || this.queuedBytes >= this.maxBytes()) this.schedule(0);
    else this.schedule(this.options.intervalMs ?? PUSH_INTERVAL_MS);
  }

  /**
   * Send everything queued now and wait until it is accepted, the run is gone, or the timeout
   * passes. Used after run_end, before the runner exits.
   */
  async flush(timeoutMs = PUSH_FINAL_TIMEOUT_MS): Promise<void> {
    this.urgent = true;
    if (!this.sending && this.retry === null) this.schedule(0);
    const idle = new Promise<void>((resolve) => {
      if (this.isIdle()) resolve();
      else this.idleWaiters.push(resolve);
    });
    let deadline: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<void>((resolve) => {
      deadline = setTimeout(resolve, timeoutMs);
    });
    await Promise.race([idle, timeout]);
    if (deadline !== undefined) clearTimeout(deadline);
  }

  /** Stop timers; queued events stay in the file log. */
  close(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
    this.stopped = true;
    this.notifyIdle();
  }

  private maxEvents(): number {
    return this.options.maxEvents ?? PUSH_MAX_EVENTS;
  }

  private maxBytes(): number {
    return this.options.maxBytes ?? PUSH_MAX_BYTES;
  }

  private isIdle(): boolean {
    return this.stopped || (!this.sending && this.retry === null && this.queue.length === 0);
  }

  private notifyIdle(): void {
    if (!this.isIdle()) return;
    const waiters = this.idleWaiters.splice(0);
    for (const resolve of waiters) resolve();
  }

  private schedule(delayMs: number): void {
    if (this.stopped) return;
    if (this.timer !== null) {
      if (delayMs > 0) return;
      clearTimeout(this.timer);
    }
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.send();
    }, delayMs);
  }

  /** Take the next batch off the queue, bounded by count and bytes (always at least one event). */
  private takeBatch(): RunEvent[] {
    const batch: RunEvent[] = [];
    let bytes = 0;
    while (this.queue.length > 0 && batch.length < this.maxEvents()) {
      const next = this.queue[0]!;
      if (batch.length > 0 && bytes + next.bytes > this.maxBytes()) break;
      this.queue.shift();
      this.queuedBytes -= next.bytes;
      bytes += next.bytes;
      batch.push(next.event);
    }
    return batch;
  }

  private async send(): Promise<void> {
    if (this.sending || this.stopped) return;
    const batch = this.retry ?? this.takeBatch();
    if (batch.length === 0) {
      this.notifyIdle();
      return;
    }
    this.sending = true;
    let failed = false;
    try {
      const body: EventPushBody = { runId: this.options.runId, events: batch };
      const response = await (this.options.fetch ?? globalThis.fetch)(this.options.url, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${this.options.token}` },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(PUSH_REQUEST_TIMEOUT_MS)
      });
      // Release the connection; the body carries nothing the runner needs.
      await response.arrayBuffer().catch(() => undefined);
      if (response.status === 404 || response.status === 410) {
        this.stopped = true;
        this.queue.length = 0;
        this.queuedBytes = 0;
      } else if (response.ok) {
        this.retry = null;
        this.backoffMs = this.options.backoffStartMs ?? PUSH_BACKOFF_START_MS;
      } else {
        failed = true;
      }
    } catch {
      failed = true;
    } finally {
      this.sending = false;
    }
    if (this.stopped) {
      this.retry = null;
      this.notifyIdle();
      return;
    }
    if (failed) {
      this.retry = batch;
      this.schedule(this.backoffMs);
      this.backoffMs = Math.min(this.backoffMs * 2, this.options.backoffMaxMs ?? PUSH_BACKOFF_MAX_MS);
      return;
    }
    if (this.queue.length > 0) {
      const full = this.queue.length >= this.maxEvents() || this.queuedBytes >= this.maxBytes();
      this.schedule(this.urgent || full ? 0 : (this.options.intervalMs ?? PUSH_INTERVAL_MS));
    } else {
      this.notifyIdle();
    }
  }
}
