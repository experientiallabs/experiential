import { describe, expect, it } from "vitest";
import type { EventPushBody, RunEvent } from "./protocol.js";
import { EventPusher } from "./push.js";

function event(seq: number, delta = "x"): RunEvent {
  return { seq, ts: seq, type: "text", delta };
}

type Answer = number | "network";

/** A fetch that answers from a script (default 200) and records each body. */
function receiver(answers: Answer[] = []) {
  const bodies: EventPushBody[] = [];
  const headers: Record<string, string>[] = [];
  const fetch = (async (_url: string, init?: RequestInit) => {
    bodies.push(JSON.parse(String(init?.body)) as EventPushBody);
    headers.push(init?.headers as Record<string, string>);
    const answer = answers.shift() ?? 200;
    if (answer === "network") throw new TypeError("fetch failed");
    return new Response("{}", { status: answer });
  }) as typeof globalThis.fetch;
  return { fetch, bodies, headers };
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

describe("EventPusher", () => {
  it("batches events on the interval, in seq order, with the bearer and run id", async () => {
    const { fetch, bodies, headers } = receiver();
    const pusher = new EventPusher({ url: "http://x/events", token: "t", runId: "r", fetch, intervalMs: 40 });
    for (let seq = 1; seq <= 5; seq += 1) pusher.enqueue(event(seq));
    expect(bodies).toHaveLength(0);
    await sleep(80);
    expect(bodies).toEqual([{ runId: "r", events: [1, 2, 3, 4, 5].map((seq) => event(seq)) }]);
    expect(headers[0]).toMatchObject({ authorization: "Bearer t", "content-type": "application/json" });
  });

  it("sends early when the event or byte threshold is reached", async () => {
    const { fetch, bodies } = receiver();
    const pusher = new EventPusher({ url: "u", token: "t", runId: "r", fetch, intervalMs: 10_000, maxEvents: 3 });
    for (let seq = 1; seq <= 3; seq += 1) pusher.enqueue(event(seq));
    await sleep(20);
    expect(bodies.map((body) => body.events.length)).toEqual([3]);
    const bytes = new EventPusher({ url: "u", token: "t", runId: "r", fetch, intervalMs: 10_000, maxBytes: 200 });
    bytes.enqueue(event(1, "y".repeat(300)));
    await sleep(20);
    expect(bodies).toHaveLength(2);
    pusher.close();
    bytes.close();
  });

  it("retries the same batch with backoff and coalesces newer events into later batches", async () => {
    const { fetch, bodies } = receiver([500, "network", 200, 200]);
    const pusher = new EventPusher({ url: "u", token: "t", runId: "r", fetch, intervalMs: 5, backoffStartMs: 20, backoffMaxMs: 40 });
    pusher.enqueue(event(1));
    pusher.enqueue(event(2));
    await sleep(10);
    pusher.enqueue(event(3));
    pusher.enqueue(event(4));
    await pusher.flush(2_000);
    const seqs = bodies.map((body) => body.events.map((e) => e.seq));
    expect(seqs).toEqual([
      [1, 2],
      [1, 2],
      [1, 2],
      [3, 4]
    ]);
  });

  it("stops for the rest of the run on 404 or 410", async () => {
    for (const status of [404, 410]) {
      const { fetch, bodies } = receiver([status]);
      const pusher = new EventPusher({ url: "u", token: "t", runId: "r", fetch, intervalMs: 5 });
      pusher.enqueue(event(1));
      await sleep(20);
      pusher.enqueue(event(2));
      await pusher.flush(200);
      expect(bodies).toHaveLength(1);
      expect(pusher.isStopped).toBe(true);
    }
  });

  it("bounds the final flush when the receiver is down", async () => {
    const { fetch } = receiver(Array(100).fill(503) as Answer[]);
    const pusher = new EventPusher({ url: "u", token: "t", runId: "r", fetch, backoffStartMs: 10 });
    pusher.enqueue(event(1));
    const started = Date.now();
    await pusher.flush(150);
    expect(Date.now() - started).toBeLessThan(1_000);
    pusher.close();
  });
});
