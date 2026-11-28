import type { SessionEntry } from "@earendil-works/pi-coding-agent";
import { describe, expect, it } from "vitest";
import { summarizeSession, toNano } from "./aggregate.js";

let seq = 0;
const at = (ms: number) => new Date(ms).toISOString();

function user(ms = 1000): SessionEntry {
  return { type: "message", id: `u${++seq}`, parentId: null, timestamp: at(ms), message: { role: "user", content: "hi", timestamp: ms } } as unknown as SessionEntry;
}

interface CallSpec {
  model?: string;
  provider?: string;
  input?: number;
  cacheRead?: number;
  cacheWrite?: number;
  output?: number;
  reasoning?: number;
  billing?: { cost: number; byok?: boolean; byokListUsd?: number | null; listUsd?: number | null };
  providerCost?: number;
  estimate?: number;
  durationMs?: number;
  stopReason?: string;
  ms?: number;
}

function call(spec: CallSpec): SessionEntry {
  const ms = spec.ms ?? 2000;
  const message: Record<string, unknown> = {
    role: "assistant",
    provider: spec.provider ?? "experiential",
    model: spec.model ?? "m1",
    content: [],
    stopReason: spec.stopReason ?? "stop",
    timestamp: ms,
    durationMs: spec.durationMs,
    usage: {
      input: spec.input ?? 0,
      output: spec.output ?? 0,
      cacheRead: spec.cacheRead ?? 0,
      cacheWrite: spec.cacheWrite ?? 0,
      reasoning: spec.reasoning,
      totalTokens: 0,
      cost: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        total: spec.providerCost ?? spec.estimate ?? 0,
        totalFromProvider: spec.providerCost !== undefined
      }
    }
  };
  if (spec.billing !== undefined) {
    message.gatewayBilling = { byok: false, byokListUsd: null, listUsd: null, ...spec.billing };
  }
  return { type: "message", id: `a${++seq}`, parentId: null, timestamp: at(ms), message } as unknown as SessionEntry;
}

const summarize = (entries: SessionEntry[], openedAt?: number) => summarizeSession(entries, { sessionId: "s-1", startedAt: at(0), openedAt });

describe("summarizeSession", () => {
  it("reports an empty session", () => {
    const usage = summarize([]);
    expect(usage.totals.calls).toBe(0);
    expect(usage.prompts).toBe(0);
    expect(usage.models).toEqual([]);
    expect(usage.startedAt).toBe(0);
  });

  it("splits charged, own-key and free money per model and in total", () => {
    const usage = summarize([
      user(),
      call({ model: "big", input: 1000, output: 100, billing: { cost: 0.0533 } }),
      call({ model: "big", input: 500, cacheRead: 2000, output: 50, billing: { cost: 0.0101 } }),
      user(),
      call({ model: "mine", input: 300, output: 30, billing: { cost: 0, byok: true, byokListUsd: 0.0042 } }),
      call({ model: "promo", input: 100, output: 10, billing: { cost: 0.001, listUsd: 0.004 } })
    ]);
    expect(usage.prompts).toBe(2);
    expect(usage.totals).toMatchObject({ calls: 4, input: 1900, cacheRead: 2000, output: 190, chargedNano: 64_400_000, byokListNano: 4_200_000, freeNano: 3_000_000, unpricedCalls: 0 });
    const big = usage.models.find((row) => row.model === "big")!;
    expect(big).toMatchObject({ label: "big", calls: 2, chargedNano: 63_400_000 });
    expect(usage.models.map((row) => row.model)).toEqual(["big", "mine", "promo"]);
  });

  it("sums money as integer nano-dollars, without float drift", () => {
    const entries = Array.from({ length: 10 }, () => call({ billing: { cost: 0.1 } }));
    entries.push(call({ billing: { cost: 0.2 } }), call({ billing: { cost: 0.000000001 } }));
    // 0.1 * 10 + 0.2 in floats is 1.2000000000000002; nano-dollars stay exact.
    expect(summarize(entries).totals.chargedNano).toBe(1_200_000_001);
    expect(toNano(0.1) + toNano(0.2)).toBe(toNano(0.3));
  });

  it("never prices a call the gateway did not price, but keeps a provider-reported cost", () => {
    const usage = summarize([
      call({ model: "m1", input: 10, estimate: 0.5 }),
      call({ provider: "anthropic", model: "claude", input: 10, providerCost: 0.25 }),
      call({ model: "m1", input: 10, billing: { cost: 0.01 } })
    ]);
    expect(usage.totals.unpricedCalls).toBe(1);
    expect(usage.totals.chargedNano).toBe(260_000_000);
    expect(usage.models.find((row) => row.provider === "anthropic")?.label).toBe("anthropic/claude");
    expect(usage.models.find((row) => row.model === "m1")).toMatchObject({ calls: 2, unpricedCalls: 1, chargedNano: 10_000_000 });
  });

  it("counts cache reads and writes as prompt parts and reasoning as part of output", () => {
    const usage = summarize([
      call({ input: 100, cacheRead: 900, cacheWrite: 50, output: 80, reasoning: 30 }),
      call({ input: 100, output: 20 }),
      call({ input: 100, output: 20, reasoning: 5 })
    ]);
    expect(usage.totals).toMatchObject({ input: 300, cacheRead: 900, cacheWrite: 50, output: 120, reasoning: 35 });
    expect(summarize([call({ output: 5 })]).totals.reasoning).toBeNull();
  });

  it("counts failed calls, compaction, cache warming, model and tool time, and output speed", () => {
    const tool = { type: "message", id: "t1", parentId: null, timestamp: at(1), message: { role: "toolResult", durationMs: 1500 } } as unknown as SessionEntry;
    const compaction = { type: "compaction", id: "c1", parentId: null, timestamp: at(1), summary: "", firstKeptEntryId: "x", tokensBefore: 0, usage: { input: 400, output: 40, cacheRead: 0, cacheWrite: 0, totalTokens: 440, cost: { total: 0.003 } } } as unknown as SessionEntry;
    const warm = { type: "usage", id: "w1", parentId: null, timestamp: at(1), kind: "cache_warm", provider: "experiential", model: "m1", usage: { input: 0, output: 1, cacheRead: 5000, cacheWrite: 0, totalTokens: 5001, cost: { total: 0 } } } as unknown as SessionEntry;
    const usage = summarize([call({ output: 100, durationMs: 2000, billing: { cost: 0.01 } }), call({ output: 0, stopReason: "error" }), tool, compaction, warm]);
    expect(usage.totals.calls).toBe(4);
    expect(usage.totals.failedCalls).toBe(1);
    expect(usage.modelTimeMs).toBe(2000);
    expect(usage.toolTimeMs).toBe(1500);
    expect(usage.models.map((row) => row.label).sort()).toEqual(["(compaction)", "m1", "m1 (cache warm)"]);
    const m1 = usage.models.find((row) => row.label === "m1")!;
    expect(m1).toMatchObject({ timedMs: 2000, timedOutput: 100, failedCalls: 1 });
  });

  it("marks the calls a resumed session made before this process opened it", () => {
    const usage = summarize([user(1000), call({ ms: 1500, billing: { cost: 0.01 } }), user(9000), call({ ms: 9500, billing: { cost: 0.02 } })], 5000);
    expect(usage.earlierCalls).toBe(1);
    expect(usage.totals.chargedNano).toBe(30_000_000);
    expect(usage.openedAt).toBe(5000);
  });
});
