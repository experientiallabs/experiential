/**
 * The automatic retry of a failed reply (pi-patches.ts, core/agent-session.js and the providers'
 * error handling in pi-ai): a reply that failed after it started streaming is kept and not sent
 * again, a failure before any content is retried after the wait the gateway asked for, and a wait
 * longer than the retry cap is reported instead of retried.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type AgentSession, createAgentSession, ModelRuntime, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { GATEWAY_PROVIDER_ID, type GatewayConfig } from "../gateway/config.js";
import { parseListedModel } from "../gateway/models.js";
import { gatewayProviderConfig, gatewayProviderModels } from "../gateway/provider.js";
import { type FakeGateway, type FakeTurn, startFakeGateway } from "../testing/fake-gateway.js";
import { summarizeSession } from "../usage/aggregate.js";

const MODEL = { id: "fake-model", context_window_tokens: 200_000, maximum_output_tokens: 8_000 };

interface AssistantRecord {
  role: string;
  stopReason?: string;
  errorMessage?: string;
  content: { type: string; text?: string }[];
}

describe("automatic retry of a failed reply", () => {
  let gateway: FakeGateway;
  let agentDir: string;

  beforeAll(async () => {
    gateway = await startFakeGateway([], [MODEL]);
    agentDir = mkdtempSync(join(tmpdir(), "exp-code-retry-"));
  });
  afterAll(async () => {
    await gateway.close();
    rmSync(agentDir, { recursive: true, force: true });
  });
  beforeEach(() => {
    gateway.requests.length = 0;
  });

  async function start(turns: FakeTurn[], baseDelayMs = 10): Promise<{ session: AgentSession; retries: { attempt: number; delayMs: number }[] }> {
    gateway.script(turns);
    const config: GatewayConfig = { kind: "hosted", baseUrl: gateway.baseUrl, apiKey: "xpl_test_key" };
    const modelRuntime = await ModelRuntime.create({ authPath: join(agentDir, "auth.json"), modelsPath: join(agentDir, "models.json") });
    modelRuntime.registerProvider(GATEWAY_PROVIDER_ID, gatewayProviderConfig(config, gatewayProviderModels([parseListedModel(MODEL)!], config, "openai-completions")));
    const { session } = await createAgentSession({
      cwd: agentDir,
      agentDir,
      modelRuntime,
      model: modelRuntime.getModel(GATEWAY_PROVIDER_ID, MODEL.id)!,
      sessionManager: SessionManager.inMemory(agentDir),
      settingsManager: SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: true, maxRetries: 3, baseDelayMs } }),
      noTools: "all"
    });
    const retries: { attempt: number; delayMs: number }[] = [];
    session.subscribe((event) => {
      if (event.type === "auto_retry_start") retries.push({ attempt: event.attempt, delayMs: event.delayMs });
    });
    return { session, retries };
  }

  const assistants = (session: AgentSession) => session.messages.filter((message) => message.role === "assistant") as unknown as AssistantRecord[];
  const last = (session: AgentSession) => assistants(session).at(-1)!;

  it("keeps a reply that dropped after it started, does not send the turn again, and shows it to the model on the next prompt", async () => {
    const { session, retries } = await start([{ text: "The parser bug is in the tokenizer loop, where", dropAfterContent: true }, { text: "A second, billed answer." }]);
    await session.prompt("why does the parser fail?");
    expect(gateway.completions()).toHaveLength(1);
    expect(retries).toEqual([]);
    const reply = last(session);
    expect(reply.stopReason).toBe("error");
    expect(reply.content).toEqual([expect.objectContaining({ type: "text", text: "The parser bug is in the tokenizer loop, where" })]);
    expect(reply.errorMessage).toContain("The reply was interrupted after it started");
    expect(reply.errorMessage).toContain('say "continue"');
    // "continue" picks up from what arrived: the partial reply is in the next request.
    gateway.script([{ text: "the index runs one past the end." }]);
    await session.prompt("continue");
    const messages = gateway.completions().at(-1)!.messages as { role: string; content: unknown }[];
    expect(messages.map((message) => message.role)).toEqual(["system", "user", "assistant", "user"]);
    expect(JSON.stringify(messages[2])).toContain("The parser bug is in the tokenizer loop, where");
    // Both calls count in /usage, the failed one as failed.
    const usage = summarizeSession(session.sessionManager.getEntries(), { sessionId: "s" });
    expect(usage.totals.calls).toBe(2);
    expect(usage.totals.failedCalls).toBe(1);
  });

  it("does not resend a reply that dropped after it started streaming a tool call", async () => {
    const { session } = await start([{ calls: [{ name: "write", arguments: { path: "a.txt", content: "x" } }], dropAfterContent: true }]);
    await session.prompt("write a.txt");
    expect(gateway.completions()).toHaveLength(1);
    expect(last(session).errorMessage).toContain("interrupted after it started");
  });

  it("retries a failure before any content, and counts every attempt in /usage", async () => {
    const { session, retries } = await start([{ status: 503, error: "upstream overloaded" }, { text: "Fine now." }]);
    await session.prompt("hello");
    expect(gateway.completions()).toHaveLength(2);
    expect(retries).toEqual([{ attempt: 1, delayMs: 10 }]);
    expect(last(session).content).toEqual([expect.objectContaining({ text: "Fine now." })]);
    const usage = summarizeSession(session.sessionManager.getEntries(), { sessionId: "s" });
    expect(usage.totals.calls).toBe(2);
    expect(usage.totals.failedCalls).toBe(1);
  });

  it("waits as long as Retry-After asks when that is within the cap", async () => {
    const { session, retries } = await start([{ status: 429, error: "rate limited", headers: { "retry-after": "1" } }, { text: "Fine now." }], 30_000);
    const started = Date.now();
    await session.prompt("hello");
    expect(retries).toEqual([{ attempt: 1, delayMs: 1000 }]);
    expect(Date.now() - started).toBeLessThan(15_000);
    expect(last(session).content).toEqual([expect.objectContaining({ text: "Fine now." })]);
  });

  it("does not retry when Retry-After is longer than the cap, and says when to try again", async () => {
    const { session, retries } = await start([
      { status: 429, error: "rate limited", headers: { "retry-after": "3600" } },
      { text: "should not be sent" }
    ]);
    const started = Date.now();
    await session.prompt("hello");
    expect(Date.now() - started).toBeLessThan(5_000);
    expect(retries).toEqual([]);
    expect(gateway.completions()).toHaveLength(1);
    expect(last(session).errorMessage).toMatch(/The gateway asked to wait 1 hour before trying again \(until \d{1,2}:\d{2}/);
    expect(last(session).errorMessage).toContain("exp-code did not retry");
  });

  it("reads the gateway's exhausted rate-limit window when there is no Retry-After", async () => {
    const { session, retries } = await start([
      { status: 429, error: "rate limited", headers: { "x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "2m30s", "x-ratelimit-remaining-tokens": "500", "x-ratelimit-reset-tokens": "1h" } },
      { text: "should not be sent" }
    ]);
    await session.prompt("hello");
    expect(retries).toEqual([]);
    expect(last(session).errorMessage).toContain("wait 3 minutes");
  });
});

describe("the wait a failed response asks for", () => {
  const providerRetry = new URL("../../node_modules/@earendil-works/pi-ai/dist/utils/provider-retry.js", import.meta.url).href;
  const delay = async (headers: Record<string, string>): Promise<number | undefined> => {
    const { serverRetryDelayMs } = (await import(providerRetry)) as { serverRetryDelayMs: (error: unknown) => number | undefined };
    return serverRetryDelayMs(Object.assign(new Error("429"), { status: 429, headers: new Headers(headers) }));
  };

  it("reads retry-after-ms, retry-after in seconds or as a date, and exhausted rate-limit windows", async () => {
    expect(await delay({ "retry-after-ms": "250" })).toBe(250);
    expect(await delay({ "retry-after": "3600" })).toBe(3_600_000);
    expect(await delay({ "retry-after": " 7 " })).toBe(7000);
    const date = await delay({ "retry-after": new Date(Date.now() + 120_000).toUTCString() });
    expect(date).toBeGreaterThan(100_000);
    expect(date).toBeLessThanOrEqual(120_000);
    expect(await delay({ "retry-after": new Date(Date.now() - 60_000).toUTCString() })).toBe(0);
    expect(await delay({ "x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "6m0s" })).toBe(360_000);
    expect(await delay({ "x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "1.5s" })).toBe(1500);
    expect(await delay({ "x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "20ms" })).toBe(20);
    // A window with room left, or no header at all, asks for nothing.
    expect(await delay({ "x-ratelimit-remaining-requests": "3", "x-ratelimit-reset-requests": "1h" })).toBeUndefined();
    expect(await delay({})).toBeUndefined();
    const { serverRetryDelayMs } = (await import(providerRetry)) as { serverRetryDelayMs: (error: unknown) => number | undefined };
    expect(serverRetryDelayMs(new Error("fetch failed"))).toBeUndefined();
  });

  it("is stored by every wire exp-code speaks", async () => {
    const { readFileSync } = await import("node:fs");
    const { fileURLToPath } = await import("node:url");
    const { patchSource } = await import("./pi-patches.js");
    for (const file of ["openai-completions", "openai-responses", "anthropic-messages"]) {
      const path = fileURLToPath(new URL(`../../node_modules/@earendil-works/pi-ai/dist/api/${file}.js`, import.meta.url));
      const source = patchSource(path, readFileSync(path, "utf8"));
      expect(source).toContain("output.retryAfterMs = expRetryAfterMs;");
      expect(source).toContain("serverRetryDelayMs } from \"../utils/provider-retry.js\"");
    }
  });
});

describe("the retry policy", () => {
  const settings = { enabled: true, maxRetries: 3, baseDelayMs: 2000, maxAgentDelayMs: 60_000 };
  const policy = async () =>
    ((await import(new URL("../../node_modules/@earendil-works/pi-coding-agent/dist/core/agent-session.js", import.meta.url).href)) as unknown as { applyRetryPolicy: (message: object, settings: object, now?: number) => void }).applyRetryPolicy;
  const failed = (content: object[], extra: object = {}) => ({ role: "assistant", stopReason: "error", errorMessage: "terminated", content, ...extra }) as Record<string, unknown>;

  it("keeps any reply that streamed text, reasoning or a tool call", async () => {
    const apply = await policy();
    for (const content of [[{ type: "text", text: "Half" }], [{ type: "thinking", thinking: "Hmm" }], [{ type: "toolCall", id: "c", name: "bash", arguments: {} }]]) {
      const message = failed(content);
      apply(message, settings);
      expect(message.expRetry).toEqual({ allowed: false, reason: "interrupted" });
      expect(message.errorMessage).toMatch(/^terminated\. The reply was interrupted after it started\./);
    }
    // Whitespace is not content.
    const blank = failed([{ type: "text", text: " \n" }]);
    apply(blank, settings);
    expect(blank.expRetry).toBeUndefined();
  });

  it("waits what the gateway asked within the cap, and refuses a longer wait", async () => {
    const apply = await policy();
    const short = failed([], { retryAfterMs: 1500.2 });
    apply(short, settings);
    expect(short.expRetry).toEqual({ allowed: true, delayMs: 1501 });
    const long = failed([], { retryAfterMs: 9_000_000, errorMessage: "429 Too many requests" });
    apply(long, settings, new Date(2026, 0, 1, 10, 0).getTime());
    expect(long.expRetry).toEqual({ allowed: false, reason: "wait", waitMs: 9_000_000 });
    expect(long.errorMessage).toMatch(/^429 Too many requests\. The gateway asked to wait 2 hours 30 minutes before trying again \(until 12:30/);
    // A raised cap waits longer; retries turned off change nothing.
    const raised = failed([], { retryAfterMs: 90_000 });
    apply(raised, { ...settings, maxAgentDelayMs: 120_000 });
    expect(raised.expRetry).toEqual({ allowed: true, delayMs: 90_000 });
    const off = failed([{ type: "text", text: "Half" }]);
    apply(off, { ...settings, enabled: false });
    expect(off.expRetry).toBeUndefined();
  });
});

describe("summary-call retries", () => {
  it("wait what the gateway asked, and stop at a wait over the cap", async () => {
    const { retryAssistantCall } = (await import(new URL("../../node_modules/@earendil-works/pi-ai/dist/utils/retry.js", import.meta.url).href)) as {
      retryAssistantCall: (produce: () => Promise<object>, policy: object, signal?: AbortSignal, callbacks?: object) => Promise<{ stopReason: string }>;
    };
    const policy = { enabled: true, maxRetries: 3, baseDelayMs: 30_000, maxAgentDelayMs: 60_000 };
    const failure = (retryAfterMs: number) => ({ role: "assistant", stopReason: "error", errorMessage: "429 rate limited", content: [], retryAfterMs });
    const delays: number[] = [];
    let calls = 0;
    const ok = await retryAssistantCall(async () => (++calls === 1 ? failure(20) : { stopReason: "stop", content: [] }), policy, undefined, {
      onRetryScheduled: (_attempt: number, _max: number, delayMs: number) => delays.push(delayMs)
    });
    expect(ok.stopReason).toBe("stop");
    expect(delays).toEqual([20]);
    calls = 0;
    const refused = await retryAssistantCall(async () => (++calls, failure(3_600_000)), policy);
    expect(refused.stopReason).toBe("error");
    expect(calls).toBe(1);
  });
});
