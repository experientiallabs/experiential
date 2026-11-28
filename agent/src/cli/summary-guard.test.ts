/**
 * The summary guard pi-patches.ts adds to Pi's compaction (`completeSummarization`) and to
 * automatic compaction in the agent session: a truncated, empty or implausibly short summary is
 * retried once with more output room and the lowest reasoning, a second failure keeps the
 * conversation and says so, and a failed automatic compaction is not retried on every call.
 */

import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { type AgentSession, createAgentSession, ModelRuntime, SessionManager, SettingsManager, serializeConversation } from "@earendil-works/pi-coding-agent";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { GATEWAY_PROVIDER_ID, type GatewayConfig } from "../gateway/config.js";
import { parseListedModel } from "../gateway/models.js";
import { gatewayProviderConfig, gatewayProviderModels } from "../gateway/provider.js";
import { type FakeGateway, startFakeGateway } from "../testing/fake-gateway.js";
import { summarizeSession } from "../usage/aggregate.js";
import { patchSource, SHORTENED_TOOL_OUTPUT, TURN_PREFIX_CONTINUES } from "./pi-patches.js";

const compactionFile = fileURLToPath(new URL("../../node_modules/@earendil-works/pi-coding-agent/dist/core/compaction/compaction.js", import.meta.url));

/** A summary long enough to pass for a conversation of any size. */
const SUMMARY = `## Goal\nFix the failing parser test.\n\n## Progress\n${"- Read src/parser.ts and found the off-by-one in the tokenizer loop.\n".repeat(8)}`;

/** A reasoning model with a small window, so a test can cross the compaction threshold. */
const THINKER = {
  id: "thinker",
  context_window_tokens: 200_000,
  maximum_output_tokens: 64_000,
  supports_reasoning: true,
  supported_reasoning_efforts: ["none", "low", "medium", "high"]
};
/** The prompt size the gateway reports to push the session over the threshold (above 119,616). */
const OVER_THRESHOLD = 150_000;

describe("the patched completeSummarization", () => {
  it("is what Pi loads", () => {
    const source = patchSource(compactionFile, readFileSync(compactionFile, "utf8"));
    expect(source).toContain("export function getSummaryProblem(response, inputChars)");
    expect(source).toContain("export function summaryThinkingLevels(model, requested)");
    expect(source.match(/export async function completeSummarization\(/g)).toHaveLength(1);
  });

  it("picks low reasoning first and the lowest level for the retry", async () => {
    const { summaryThinkingLevels } = await import(pathToFileURL(compactionFile).href);
    const map = (efforts: string[]) =>
      Object.fromEntries(["off", "minimal", "low", "medium", "high", "xhigh", "max"].map((level) => [level, efforts.includes(level === "off" ? "none" : level) ? level : null]));
    const model = (efforts: string[]) => ({ reasoning: true, thinkingLevelMap: map(efforts) });
    expect(summaryThinkingLevels(model(["none", "low", "medium"]), "medium")).toEqual({ first: "low", retry: "off" });
    // A session with thinking off keeps it off.
    expect(summaryThinkingLevels(model(["none", "low", "medium"]), undefined)).toEqual({ first: "off", retry: "off" });
    // A route that always reasons: its lowest level both times.
    expect(summaryThinkingLevels(model(["low", "high"]), "high")).toEqual({ first: "low", retry: "low" });
    expect(summaryThinkingLevels(model(["high"]), "high")).toEqual({ first: "high", retry: "high" });
    expect(summaryThinkingLevels({ reasoning: false }, undefined)).toEqual({ first: "off", retry: "off" });
  });

  it("rejects a truncated, empty or implausibly short summary", async () => {
    const { getSummaryProblem } = await import(pathToFileURL(compactionFile).href);
    const answer = (text: string, stopReason = "stop") => ({ stopReason, content: text === "" ? [] : [{ type: "text", text }] });
    expect(getSummaryProblem(answer("`", "length"), 400_000)).toMatch(/cut off at the output limit/);
    expect(getSummaryProblem(answer(""), 400_000)).toMatch(/empty/);
    expect(getSummaryProblem(answer("   \n"), 400_000)).toMatch(/empty/);
    expect(getSummaryProblem({ stopReason: "stop", content: [{ type: "thinking", thinking: "long thoughts" }] }, 400_000)).toMatch(/empty/);
    expect(getSummaryProblem(answer("`"), 400_000)).toMatch(/implausibly short \(1 characters for 400000/);
    expect(getSummaryProblem(answer(SUMMARY), 400_000)).toBeUndefined();
    // A short conversation may have a short summary.
    expect(getSummaryProblem(answer("User said hi."), 300)).toBeUndefined();
    // Errors and aborts are Pi's to report.
    expect(getSummaryProblem({ ...answer(""), stopReason: "error" }, 400_000)).toBeUndefined();
  });
});

describe("what the summarizer is told", () => {
  it("marks tool output shortened for the summary request as seen in full by the assistant", () => {
    const output = `M11000=7825\n${"x".repeat(3000)}\nM11139=9296`;
    const text = serializeConversation([{ role: "toolResult", toolCallId: "call-1", toolName: "read", content: [{ type: "text", text: output }], isError: false, timestamp: 0 }] as never);
    expect(text).toContain("M11000=7825");
    expect(text).not.toContain("M11139=9296");
    expect(text).toContain(`[... ${output.length - 2000} more characters left out of this summary request; the assistant saw the whole output]`);
    expect(text).not.toMatch(/characters truncated\]/);
  });
});

describe("compaction on a session", () => {
  let gateway: FakeGateway;
  let agentDir: string;

  beforeAll(async () => {
    gateway = await startFakeGateway([], [THINKER]);
    agentDir = mkdtempSync(join(tmpdir(), "exp-code-compaction-"));
  });
  afterAll(async () => {
    await gateway.close();
    rmSync(agentDir, { recursive: true, force: true });
  });
  beforeEach(() => {
    gateway.requests.length = 0;
  });

  async function start(): Promise<{ session: AgentSession; events: { reason: string; result?: unknown; errorMessage?: string }[] }> {
    const config: GatewayConfig = { kind: "hosted", baseUrl: gateway.baseUrl, apiKey: "xpl_test_key" };
    const modelRuntime = await ModelRuntime.create({ authPath: join(agentDir, "auth.json"), modelsPath: join(agentDir, "models.json") });
    modelRuntime.registerProvider(GATEWAY_PROVIDER_ID, gatewayProviderConfig(config, gatewayProviderModels([parseListedModel(THINKER)!], config, "openai-completions")));
    const { session } = await createAgentSession({
      cwd: agentDir,
      agentDir,
      modelRuntime,
      model: modelRuntime.getModel(GATEWAY_PROVIDER_ID, THINKER.id)!,
      sessionManager: SessionManager.inMemory(agentDir),
      // Keep almost nothing verbatim, so every earlier turn is summarized.
      settingsManager: SettingsManager.inMemory({ compaction: { enabled: true, keepRecentTokens: 10 } }),
      noTools: "all"
    });
    const events: { reason: string; result?: unknown; errorMessage?: string }[] = [];
    session.subscribe((event) => {
      if (event.type === "compaction_end") events.push({ reason: event.reason, result: event.result, errorMessage: event.errorMessage });
    });
    return { session, events };
  }

  /** The summary requests the gateway received (their system prompt is Pi's summarizer's). */
  const summaryRequests = () =>
    gateway.completions().filter((body) => JSON.stringify((body.messages as unknown[])[0]).includes("context summarization assistant"));

  /** Two ordinary turns; the second reports a prompt over the compaction threshold. */
  async function fill(session: AgentSession, summaries: Parameters<FakeGateway["script"]>[0]) {
    gateway.script([{ text: "First answer." }, { text: "Second answer.", promptTokens: OVER_THRESHOLD }, ...summaries]);
    await session.prompt("first question");
    await session.prompt("second question");
  }

  const conversation = (session: AgentSession) =>
    session.messages.filter((message) => message.role === "user" || message.role === "assistant").map((message) => (message.role === "user" ? "user" : "assistant"));

  it("retries a truncated summary with more output room and the lowest reasoning, then commits it", async () => {
    const { session, events } = await start();
    await fill(session, [{ text: "`", finishReason: "length" }, { text: SUMMARY }]);
    expect(events).toHaveLength(1);
    expect(events[0]!.errorMessage).toBeUndefined();
    const [first, retry] = summaryRequests();
    // Bounded at 32K and low reasoning; the retry gets the model's whole output limit, thinking off.
    expect(first).toMatchObject({ max_completion_tokens: 32_768, reasoning_effort: "low" });
    expect(retry).toMatchObject({ max_completion_tokens: 64_000, reasoning_effort: "none" });
    const compaction = session.sessionManager.getEntries().find((entry) => entry.type === "compaction");
    expect(compaction?.type === "compaction" && compaction.summary).toContain("Fix the failing parser test.");
  });

  it("retries an empty summary", async () => {
    const { session, events } = await start();
    await fill(session, [{ text: "" }, { text: SUMMARY }]);
    expect(summaryRequests()).toHaveLength(2);
    expect(events[0]!.result).toBeDefined();
  });

  it("keeps the conversation when the retry fails too, says why, and does not try again before every call", async () => {
    const { session, events } = await start();
    const before = session.sessionManager.getEntries().length;
    await fill(session, [{ text: "`", finishReason: "length" }, { text: "`", finishReason: "length" }]);
    expect(summaryRequests()).toHaveLength(2);
    expect(session.sessionManager.getEntries().some((entry) => entry.type === "compaction")).toBe(false);
    expect(conversation(session)).toEqual(["user", "assistant", "user", "assistant"]);
    expect(session.sessionManager.getEntries().length).toBeGreaterThan(before);
    expect(events).toHaveLength(1);
    // The cut falls inside the last turn here, so Pi summarizes that turn's prefix.
    expect(events[0]!.errorMessage).toBe(
      "Auto-compaction failed: Turn prefix summarization failed: the summary was cut off at the output limit, and a retry with up to 64000 output tokens and reasoning off failed too (the summary was cut off at the output limit). The conversation was kept unchanged. Automatic compaction is paused for thinker in this session. Run /compact to try again, /model to switch to another model, or /new to start a fresh session."
    );

    // Still over the threshold: the next turns are answered without another summary attempt.
    gateway.script([{ text: "Third answer.", promptTokens: OVER_THRESHOLD }, { text: "Fourth answer.", promptTokens: OVER_THRESHOLD }]);
    await session.prompt("third question");
    await session.prompt("fourth question");
    expect(summaryRequests()).toHaveLength(2);
    expect(events).toHaveLength(1);
    expect(conversation(session)).toHaveLength(8);

    // /compact still works (a split turn asks for two summaries), and a successful compaction
    // resumes automatic compaction.
    gateway.script([{ text: SUMMARY }, { text: SUMMARY }]);
    const result = await session.compact();
    expect(result.summary).toContain("Fix the failing parser test.");
    // Pi ignores usage from a reply timestamped at or before the latest compaction, and a reply is
    // timestamped when its request starts, not when the gateway answers: a delay on the gateway's
    // side did not help, and a prompt sent in the compaction's millisecond (about one run in four
    // on a busy machine) was taken for a pre-compaction reply and never compacted. The next prompt
    // therefore starts after that millisecond.
    const compactedAt = Math.max(
      ...session.sessionManager
        .getEntries()
        .filter((entry) => entry.type === "compaction")
        .map((entry) => new Date(entry.timestamp).getTime())
    );
    while (Date.now() <= compactedAt) await new Promise((resolve) => setTimeout(resolve, 1));
    gateway.script([{ text: "Fifth answer.", promptTokens: OVER_THRESHOLD }, { text: SUMMARY }, { text: SUMMARY }]);
    await session.prompt("fifth question");
    expect(events.at(-1)).toMatchObject({ reason: "threshold", errorMessage: undefined });
    expect(events.at(-1)!.result).toBeDefined();
  });

  it("tells a split turn's prefix summary that the kept messages continue the turn", async () => {
    const { session, events } = await start();
    await fill(session, [{ text: SUMMARY }, { text: SUMMARY }]);
    expect(events[0]!.errorMessage).toBeUndefined();
    const requests = summaryRequests();
    // The cut falls inside the first turn here, so its prefix is the one summary.
    expect(requests).toHaveLength(1);
    const text = JSON.stringify(requests[0]!.messages);
    expect(text).toContain("ongoing conversation");
    expect(text).toContain(JSON.stringify(TURN_PREFIX_CONTINUES).slice(1, -1));
    expect(text).toContain(JSON.stringify(SHORTENED_TOOL_OUTPUT).slice(1, -1));
  });

  it("records the cost the gateway reported for each summary attempt", async () => {
    const { session } = await start();
    await fill(session, [
      { text: "`", finishReason: "length", cost: 0.0011 },
      { text: SUMMARY, cost: 0.0023 }
    ]);
    const compaction = session.sessionManager.getEntries().find((entry) => entry.type === "compaction");
    expect(compaction?.type).toBe("compaction");
    const cost = compaction?.type === "compaction" ? compaction.usage?.cost : undefined;
    // Both attempts were billed, so the entry carries their sum as a provider-reported price.
    expect(cost?.totalFromProvider).toBe(true);
    expect(cost?.total).toBeCloseTo(0.0034, 10);
    const usage = summarizeSession(session.sessionManager.getEntries(), { sessionId: "s" });
    const summary = usage.models.find((model) => model.model === "(compaction)");
    expect(summary).toMatchObject({ calls: 1, unpricedCalls: 0, chargedNano: 3_400_000 });
  });

  it("reports a failed /compact with what to do next", async () => {
    const { session } = await start();
    gateway.script([{ text: "First answer." }, { text: "Second answer." }, { text: "" }, { text: "" }, { text: "" }, { text: "" }]);
    await session.prompt("first question");
    await session.prompt("second question");
    await expect(session.compact()).rejects.toThrow(/empty/);
    expect(session.sessionManager.getEntries().some((entry) => entry.type === "compaction")).toBe(false);
    expect(conversation(session)).toHaveLength(4);
  });
});
