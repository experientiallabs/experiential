import type { SessionEntry } from "@earendil-works/pi-coding-agent";
import { afterEach, describe, expect, it } from "vitest";
import { type FakePlatform, startFakePlatform } from "../testing/fake-platform.js";
import { buildUsageReport, registerUsageCommand, reportText, USAGE_ENTRY_TYPE } from "./command.js";

let platform: FakePlatform | undefined;
afterEach(async () => {
  await platform?.close();
  platform = undefined;
});

const NOW = Date.parse("2026-11-27T15:00:30Z");

function assistant(model: string, cost: number, ms: number): SessionEntry {
  return {
    type: "message",
    id: `a-${model}-${ms}`,
    parentId: null,
    timestamp: new Date(ms).toISOString(),
    message: {
      role: "assistant",
      provider: "experiential",
      model,
      content: [],
      stopReason: "stop",
      timestamp: ms,
      durationMs: 500,
      usage: { input: 100, output: 50, cacheRead: 0, cacheWrite: 0, totalTokens: 150, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: cost } },
      gatewayBilling: { cost, byok: false, byokListUsd: null, listUsd: null }
    }
  } as unknown as SessionEntry;
}

function fakeContext(entries: SessionEntry[], mode: "tui" | "rpc" = "tui", idle = true) {
  const notices: string[] = [];
  return {
    notices,
    ctx: {
      mode,
      hasUI: true,
      ui: { notify: (text: string) => notices.push(text) },
      isIdle: () => idle,
      model: { id: "m2" },
      getContextUsage: () => ({ tokens: 1500, contextWindow: 100_000, percent: 1.5 }),
      sessionManager: {
        getEntries: () => entries,
        getSessionId: () => "abcdef12-3456",
        getSessionName: () => undefined,
        getHeader: () => ({ type: "session", id: "abcdef12-3456", timestamp: "2026-11-27T15:00:00Z", cwd: "/w" })
      }
    }
  };
}

function fakePi() {
  const commands = new Map<string, { description?: string; handler: (args: string, ctx: unknown) => Promise<void> }>();
  const renderers = new Map<string, (entry: unknown, options: unknown, theme: unknown) => unknown>();
  const appended: { customType: string; data: unknown }[] = [];
  const handlers = new Map<string, ((event: unknown, ctx: unknown) => unknown)[]>();
  const api = {
    registerCommand: (name: string, options: { description?: string; handler: (args: string, ctx: unknown) => Promise<void> }) => commands.set(name, options),
    registerEntryRenderer: (type: string, renderer: (entry: unknown, options: unknown, theme: unknown) => unknown) => renderers.set(type, renderer),
    appendEntry: (customType: string, data: unknown) => appended.push({ customType, data }),
    on: (event: string, handler: (event: unknown, ctx: unknown) => unknown) => handlers.set(event, [...(handlers.get(event) ?? []), handler])
  };
  return { api, commands, renderers, appended, handlers };
}

const theme = { bold: (text: string) => text, fg: (_color: string, text: string) => text };

describe("/usage", () => {
  it("builds the report from the session, the context and the organization", async () => {
    platform = await startFakePlatform({
      "GET /api/v1/credits": { body: { data: { total_credits: 10, total_usage: 2.5 } } },
      "GET /api/gateway/usage/daily": { body: { rows: [{ day: "2026-11-27", requests: 3, paid_nano_usd: 80_000_000 }] } }
    });
    const { ctx } = fakeContext([assistant("m1", 0.0533, Date.parse("2026-11-27T15:00:05Z")), assistant("m2", 0.0101, Date.parse("2026-11-27T15:00:20Z"))]);
    const lines = await buildUsageReport(ctx as never, { gateway: { kind: "hosted", baseUrl: platform.baseUrl, apiKey: "xpl_k" }, now: () => NOW, dashboardUrl: "https://p.example/spend" });
    const out = reportText(lines);
    expect(out).toContain("Usage · session abcdef12, started 2026-11-27 15:00Z");
    expect(out).toContain("30.0s since start · model 1.0s");
    expect(out).toContain("Cost          $0.0634 charged");
    expect(out).toContain("Context       1,500 / 100,000 tokens (2%) on m2");
    expect(out).toMatch(/m1\s+1\s+100\s+0\s+50\s+\$0\.0533\s+100\.0/);
    expect(out).toContain("Organization  Acme · 750.00 credits left ($7.50) · today (UTC) $0.0800 charged over 3 requests");
    expect(out).toContain("History and per-request detail: https://p.example/spend");
  });

  it("still answers when the organization read fails, and while a response streams", async () => {
    platform = await startFakePlatform({ "GET /api/whoami": { status: 500 }, "GET /api/v1/credits": { status: 500 } });
    const { ctx } = fakeContext([], "tui", false);
    const out = reportText(await buildUsageReport(ctx as never, { gateway: { kind: "hosted", baseUrl: platform.baseUrl, apiKey: "xpl_k" }, now: () => NOW }));
    expect(out).toContain("No model calls yet");
    expect(out).toContain("A response is still streaming");
    expect(out).toMatch(/Organization: The Experiential API had a problem \(500\)/);
  });

  it("registers /usage and /cost, draws the report in the terminal UI only for this process", async () => {
    const pi = fakePi();
    registerUsageCommand(pi.api as never, { gateway: null, now: () => NOW });
    expect([...pi.commands.keys()]).toEqual(["usage", "cost"]);
    for (const handler of pi.handlers.get("session_start") ?? []) handler({ reason: "startup" }, {});
    const { ctx } = fakeContext([assistant("m1", 0.01, NOW - 1000)]);
    await pi.commands.get("cost")!.handler("", ctx);
    expect(pi.appended).toHaveLength(1);
    const entry = { type: "custom", customType: USAGE_ENTRY_TYPE, data: pi.appended[0]!.data };
    const renderer = pi.renderers.get(USAGE_ENTRY_TYPE)!;
    expect(renderer(entry, {}, theme)).toBeDefined();
    // A report another process wrote (an older run of a resumed session) is not drawn.
    expect(renderer({ ...entry, data: { ...(entry.data as object), instance: "other" } }, {}, theme)).toBeUndefined();
  });

  it("sends the report as a notice outside the terminal UI", async () => {
    const pi = fakePi();
    registerUsageCommand(pi.api as never, { gateway: { kind: "local", baseUrl: "http://127.0.0.1:8000/v1", apiKey: "k" }, now: () => NOW });
    const { ctx, notices } = fakeContext([], "rpc");
    await pi.commands.get("usage")!.handler("", ctx);
    expect(pi.appended).toEqual([]);
    expect(notices[0]).toContain("No model calls yet");
    expect(notices[0]).toContain("Organization: not shown for a local exp gateway");
    expect(notices[0]).not.toContain("History and per-request detail");
  });
});
