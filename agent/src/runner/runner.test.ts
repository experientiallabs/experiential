import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { type FakeGateway, type FakeTurn, startFakeGateway } from "../testing/fake-gateway.js";
import { readEventLog } from "./event-log.js";
import type { EventPushBody, RunEvent, RunSpec, RunStatus } from "./protocol.js";
import { INTERRUPTED_TOOL_TEXT, RESUME_NOTE } from "./recovery.js";
import { runOnce, STOPPED_NOTE } from "./runner.js";
import { freshState, saveState } from "./state.js";

let root: string;
let sessionDir: string;
let workDir: string;
let gateway: FakeGateway;
const runId = "11111111-2222-3333-4444-555555555555";

function runDir(): string {
  return join(sessionDir, "runs", runId);
}

function writeSpec(overrides: Partial<RunSpec> = {}): void {
  const spec: RunSpec = {
    runId,
    sessionId: "s",
    model: "fake-model",
    prompt: "do the thing",
    attachments: [],
    vision: false,
    reasoningEffort: null,
    webSearch: false,
    gatewayBaseUrl: gateway.baseUrl,
    maxTurns: 10,
    deadline: Date.now() + 60_000,
    callbackUrl: `${gateway.origin}/callback`,
    restored: false,
    ...overrides
  };
  mkdirSync(runDir(), { recursive: true });
  writeFileSync(join(runDir(), "run.json"), JSON.stringify(spec));
}

function events(): RunEvent[] {
  return readEventLog(join(runDir(), "events.jsonl"));
}

function run(turns: FakeTurn[], extra: { callbackToken?: string } = {}) {
  gateway.script(turns);
  return runOnce({ sessionDir, workDir, runId, apiKey: "xpl_run", callbackToken: extra.callbackToken ?? "cb" });
}

async function waitFor(predicate: (log: RunEvent[]) => boolean): Promise<void> {
  for (let waited = 0; waited < 15_000; waited += 50) {
    if (predicate(existsSync(join(runDir(), "events.jsonl")) ? events() : [])) return;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error("timed out waiting for the run");
}

function transcriptRoles(): string[] {
  return readFileSync(join(sessionDir, "session.jsonl"), "utf8")
    .split("\n")
    .filter((line) => line.trim() !== "")
    .map((line) => JSON.parse(line) as { type: string; message?: { role: string } })
    .filter((entry) => entry.type === "message")
    .map((entry) => entry.message!.role);
}

function requestRoles(body: Record<string, unknown>): string[] {
  return (body.messages as { role: string }[]).map((message) => message.role);
}

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "exp-code-runner-"));
  sessionDir = join(root, "session");
  workDir = join(root, "work");
  mkdirSync(workDir, { recursive: true });
  gateway = await startFakeGateway();
});

afterEach(async () => {
  await gateway.close();
  await rm(root, { recursive: true, force: true });
});

describe("runner", () => {
  it("marks a turn served on the user's own provider key, with that provider's list price", async () => {
    writeSpec();
    expect(await run([{ text: "Done.", cost: 0, byokListUsd: 0.0004 }])).toBe("complete");
    const turn = events().find((event) => event.type === "turn");
    expect(turn).toMatchObject({ costUsd: 0, byok: true, byokListUsd: 0.0004 });
  });

  it("marks a free turn with its list price", async () => {
    writeSpec();
    expect(await run([{ text: "Done.", cost: 0, listUsd: 0.003 }])).toBe("complete");
    const turn = events().find((event) => event.type === "turn");
    expect(turn).toMatchObject({ costUsd: 0, listUsd: 0.003 });
    expect(turn).not.toHaveProperty("byok");
  });

  it("logs a full run with ordered seqs, todo, files, usage, and the Pi session as transcript", async () => {
    writeSpec();
    const status = await run([
      { calls: [{ name: "todo_write", arguments: { todos: [{ content: "Write file", status: "in_progress" }, { content: "Report", status: "pending" }] } }] },
      { calls: [{ name: "write", arguments: { path: "out/report.md", content: "# Hi" } }] },
      { text: "Saved out/report.md.", cost: 0.25 }
    ]);
    expect(status).toBe("complete");
    const log = events();
    expect(log.map((event) => event.seq)).toEqual(log.map((_, index) => index + 1));
    expect(log[0]?.type).toBe("run_start");
    expect(log.at(-1)).toMatchObject({ type: "run_end", status: "complete", turns: 3 });
    expect(log.find((event) => event.type === "todo")).toMatchObject({ items: [{ content: "Write file", status: "in_progress" }, { content: "Report", status: "pending" }] });
    const turns = log.filter((event) => event.type === "turn");
    expect(turns).toHaveLength(3);
    // The gateway's settled cost is reported per turn and summed at the end.
    expect(turns.at(-1)).toMatchObject({ costUsd: 0.25, inputTokens: 50, outputTokens: 5 });
    expect((log.at(-1) as { costUsd: number }).costUsd).toBeCloseTo(0.252);
    expect(log.filter((event) => event.type === "files").at(-1)).toMatchObject({ files: [{ path: "out/report.md", size: 4 }] });
    expect(log.filter((event) => event.type === "text").map((event) => (event as { delta: string }).delta).join("")).toBe("Saved out/report.md.");
    expect(transcriptRoles()).toEqual(["system", "user", "assistant", "toolResult", "assistant", "toolResult", "assistant"]);
    const final = JSON.parse(readFileSync(join(runDir(), "status.json"), "utf8")) as RunStatus;
    expect(final).toMatchObject({ state: "complete", turns: 3, lastSeq: log.length });
    // The completion callback carries the bearer.
    const callback = gateway.requests.find((request) => request.path === "/callback");
    expect(callback?.headers.authorization).toBe("Bearer cb");
    expect(callback?.body).toEqual({ runId, status: "complete" });
  });

  it("applies a steering message mid-run and stops on request", async () => {
    writeSpec();
    const inbox = join(runDir(), "inbox.jsonl");
    const done = run([
      { calls: [{ name: "bash", arguments: { command: "sleep 1; echo first" } }] },
      { calls: [{ name: "bash", arguments: { command: "sleep 2; echo second" } }] },
      { text: "never reached", delayMs: 5_000 }
    ]);
    await waitFor((log) => log.some((event) => event.type === "tool_start"));
    appendFileSync(inbox, `${JSON.stringify({ type: "steer", text: "Also mention the date.", requestId: "r1" })}\n`);
    await waitFor((log) => log.filter((event) => event.type === "tool_start").length >= 2);
    appendFileSync(inbox, `${JSON.stringify({ type: "stop", requestId: "r2" })}\n`);
    expect(await done).toBe("aborted");
    const log = events();
    expect(log.filter((event) => event.type === "user_message")).toEqual([expect.objectContaining({ text: "Also mention the date." })]);
    expect(JSON.stringify(gateway.completions()[1])).toContain("Also mention the date.");
    expect(log.at(-1)).toMatchObject({ type: "run_end", status: "aborted" });
  });

  it("ignores inbox commands whose requestId was already applied", async () => {
    writeSpec();
    const inbox = join(runDir(), "inbox.jsonl");
    const done = run([
      { calls: [{ name: "bash", arguments: { command: "sleep 1" } }] },
      { calls: [{ name: "bash", arguments: { command: "sleep 1" } }] },
      { text: "ok" }
    ]);
    await waitFor((log) => log.some((event) => event.type === "tool_start"));
    const steer = `${JSON.stringify({ type: "steer", text: "Use metric units.", requestId: "same" })}\n`;
    appendFileSync(inbox, steer + steer);
    await waitFor((log) => log.some((event) => event.type === "user_message"));
    appendFileSync(inbox, steer);
    expect(await done).toBe("complete");
    expect(events().filter((event) => event.type === "user_message")).toHaveLength(1);
    const lastRequest = JSON.stringify(gateway.completions().at(-1));
    expect(lastRequest.split("Use metric units.").length - 1).toBe(1);
  });

  it("ends a run on a question and carries it as an event", async () => {
    writeSpec();
    await run([{ calls: [{ name: "ask_user", arguments: { question: "Which quarter?", options: ["Q1", "Q2"] } }] }]);
    const log = events();
    expect(log.find((event) => event.type === "question")).toMatchObject({ question: "Which quarter?", options: ["Q1", "Q2"] });
    expect(log.at(-1)).toMatchObject({ type: "run_end", status: "complete", turns: 1 });
    expect(gateway.completions()).toHaveLength(1);
  });

  it("reports a gateway refusal with a code", async () => {
    writeSpec();
    const status = await run([{ status: 402, error: "Insufficient credits" }]);
    expect(status).toBe("error");
    expect(events().find((event) => event.type === "error")).toMatchObject({ code: "insufficient_credits" });
    expect(events().at(-1)).toMatchObject({ type: "run_end", status: "error" });
  });

  it("turns a refused model into plain words with a code", async () => {
    writeSpec();
    const status = await run([{ status: 403, error: "The requested model alias is not granted to this identity. GET /v1/models lists the model aliases available to this key." }]);
    expect(status).toBe("error");
    const error = events().find((event) => event.type === "error");
    expect(error).toMatchObject({ code: "model_unavailable" });
    expect(error && "message" in error ? error.message : "").toMatch(/^This model is not available to your account/);
  });

  it("tells the next run that a stopped request was cancelled", async () => {
    writeSpec({ prompt: "Run sleep 60" });
    const stopped = run([{ delayMs: 5_000, text: "never" }]);
    await waitFor((log) => log.some((event) => event.type === "run_start"));
    await new Promise((resolve) => setTimeout(resolve, 300));
    appendFileSync(join(runDir(), "inbox.jsonl"), `${JSON.stringify({ type: "stop" })}\n`);
    expect(await stopped).toBe("aborted");
    writeSpec({ prompt: "Just say ready" });
    expect(await run([{ text: "ready" }])).toBe("complete");
    const body = gateway.completions().at(-1)!;
    const messages = body.messages as { role: string; content: unknown }[];
    const users = messages.filter((message) => message.role === "user").map((message) => JSON.stringify(message.content));
    expect(users[0]).toContain("Run sleep 60");
    expect(users.at(-2)).toContain(STOPPED_NOTE.slice(1, 30));
    expect(users.at(-1)).toContain("Just say ready");
  });

  it("continues the Pi session on the next run, with seqs never reused", async () => {
    writeSpec();
    await run([{ text: "One." }]);
    writeSpec({ prompt: "and again" });
    await run([{ text: "Two." }]);
    expect(requestRoles(gateway.completions().at(-1)!)).toEqual(["system", "user", "assistant", "user"]);
    const seqs = events().map((event) => event.seq);
    expect(new Set(seqs).size).toBe(seqs.length);
    expect(seqs).toEqual(seqs.map((_, index) => index + 1));
  });

  it("gives the model the cowork prompt section and the document skills", async () => {
    writeSpec();
    await run([{ text: "ok" }]);
    const system = JSON.stringify((gateway.completions()[0]?.messages as { content: unknown }[])[0]);
    expect(system).toContain("<cowork>");
    for (const skill of ["docx", "xlsx", "pptx", "pdf", "charts", "data-analysis"]) expect(system).toContain(`<name>${skill}</name>`);
  });

  it("sends the chosen reasoning effort on every model call", async () => {
    writeSpec({ reasoningEffort: "low" });
    await run([{ text: "ok" }]);
    expect(gateway.completions()[0]?.reasoning_effort).toBe("low");
  });

  it("logs streamed reasoning as thinking events", async () => {
    writeSpec();
    await run([{ reasoning: "Checking divisors of 91.", text: "No." }]);
    const thinking = events()
      .filter((event) => event.type === "thinking")
      .map((event) => (event as { delta: string }).delta)
      .join("");
    expect(thinking).toBe("Checking divisors of 91.");
  });

  it("asks a reasoning model for its reasoning text and keeps it out of the answer", async () => {
    gateway.setModels([{ id: "fake-model", object: "model", created: 0, owned_by: "exp", supports_reasoning: true }]);
    writeSpec();
    await run([
      { reasoning: "I should look at the files first.", calls: [{ name: "bash", arguments: { command: "ls" } }] },
      { reasoning: "Nothing there, so the answer is short.", reasoningField: "reasoning_content", text: "The folder is empty." }
    ]);
    for (const body of gateway.completions()) {
      expect(body.include_reasoning).toBe(true);
      expect(body).not.toHaveProperty("reasoning_effort");
    }
    const sequence = events()
      .filter((event) => event.type === "thinking" || event.type === "text" || event.type === "tool_start")
      .map((event) => event.type);
    // One thinking block before the call, one before the answer, never interleaved with the text.
    expect(sequence.filter((type, i) => type !== sequence[i - 1])).toEqual(["thinking", "tool_start", "thinking", "text"]);
    const joined = (type: string) =>
      events()
        .filter((event) => event.type === type)
        .map((event) => (event as { delta: string }).delta)
        .join("");
    expect(joined("thinking")).toBe("I should look at the files first.Nothing there, so the answer is short.");
    expect(joined("text")).toBe("The folder is empty.");
  });

  it("asks a model the listing does not mark as reasoning for no reasoning text", async () => {
    writeSpec();
    await run([{ text: "ok" }]);
    expect(gateway.completions()[0]).not.toHaveProperty("include_reasoning");
  });

  it("reads a chosen effort as a reasoning model when the listing does not say", async () => {
    gateway.setModels([]);
    writeSpec({ reasoningEffort: "high" });
    await run([{ text: "ok" }]);
    expect(gateway.completions()[0]).toMatchObject({ reasoning_effort: "high", include_reasoning: true });
  });

  it("names the conversation and the run on every model call, never on the model listing", async () => {
    writeSpec({ sessionId: "conv-1" });
    await run([{ calls: [{ name: "todo_write", arguments: { todos: [{ content: "a", status: "pending" }] } }] }, { text: "ok" }]);
    const calls = gateway.requests.filter((request) => request.path === "/v1/chat/completions");
    expect(calls.length).toBe(2);
    for (const call of calls) {
      expect(call.headers["x-explabs-session-id"]).toBe("conv-1");
      expect(JSON.parse(String(call.headers["x-explabs-tags"]))).toEqual({ conversation: "conv-1", turn: runId });
    }
    for (const listing of gateway.requests.filter((request) => request.path === "/v1/models")) {
      expect(listing.headers["x-explabs-tags"]).toBeUndefined();
    }
  });

  it("exposes web_search only when the spec enables it", async () => {
    writeSpec({ webSearch: false });
    await run([{ text: "ok" }]);
    const names = (body: Record<string, unknown>) => ((body.tools as { function: { name: string } }[]) ?? []).map((tool) => tool.function.name);
    expect(names(gateway.completions()[0]!)).not.toContain("web_search");
    expect(names(gateway.completions()[0]!)).toEqual(expect.arrayContaining(["read", "bash", "edit", "write", "todo_write", "ask_user", "web_fetch"]));
    writeSpec({ webSearch: true });
    await run([{ text: "ok" }]);
    expect(names(gateway.completions()[1]!)).toContain("web_search");
  });

  it("pushes events in seq order with the bearer and flushes run_end before exit", async () => {
    const pushed: EventPushBody[] = [];
    gateway.onPost((request) => {
      if (request.path === "/events") pushed.push(request.body as EventPushBody);
      return { status: 200 };
    });
    writeSpec({ eventsUrl: `${gateway.origin}/events` });
    await run([{ text: "Hello." }], { callbackToken: "push-token" });
    const all = pushed.flatMap((batch) => batch.events);
    expect(all).toEqual(events());
    expect(pushed.every((batch) => batch.runId === runId)).toBe(true);
    expect(gateway.requests.find((request) => request.path === "/events")?.headers.authorization).toBe("Bearer push-token");
  });

  it("keeps the file log and stops pushing when the receiver says the run is gone", async () => {
    let calls = 0;
    gateway.onPost((request) => {
      if (request.path === "/events") calls += 1;
      return { status: request.path === "/events" ? 410 : 200 };
    });
    writeSpec({ eventsUrl: `${gateway.origin}/events` });
    await run([{ calls: [{ name: "todo_write", arguments: { todos: [{ content: "a", status: "pending" }] } }] }, { text: "done" }]);
    expect(calls).toBe(1);
    expect(events().at(-1)).toMatchObject({ type: "run_end", status: "complete" });
  });

  it("continues a crashed run: replays safe tool calls, marks unsafe ones interrupted", async () => {
    writeSpec();
    writeFileSync(join(workDir, "notes.txt"), "remember this\n");
    // A first runner that dies after the model asked for two tools: simulate with a run whose
    // log has run_start but no run_end, and a session ending in unanswered tool calls.
    await run([{ calls: [{ name: "read", arguments: { path: "notes.txt" } }, { name: "bash", arguments: { command: "echo hi > side-effect.txt" } }] }, { text: "ok" }]);
    const lines = readFileSync(join(sessionDir, "session.jsonl"), "utf8").split("\n").filter((line) => line.trim() !== "");
    const kept = lines.filter((line) => {
      const entry = JSON.parse(line) as { type: string; message?: { role: string } };
      return !(entry.type === "message" && (entry.message?.role === "toolResult" || (entry.message?.role === "assistant" && line.includes('"stop"'))));
    });
    writeFileSync(join(sessionDir, "session.jsonl"), `${kept.join("\n")}\n`);
    const log = readFileSync(join(runDir(), "events.jsonl"), "utf8").split("\n").filter((line) => line !== "" && !line.includes('"run_end"'));
    writeFileSync(join(runDir(), "events.jsonl"), `${log.join("\n")}\n`);
    saveState(join(runDir(), "runner.json"), { ...freshState(runId), promptDelivered: true });

    expect(await run([{ text: "Recovered." }])).toBe("complete");
    const body = gateway.completions().at(-1)!;
    const messages = body.messages as { role: string; content: unknown; tool_call_id?: string }[];
    const toolResults = messages.filter((message) => message.role === "tool").map((message) => JSON.stringify(message.content));
    expect(toolResults[0]).toContain("remember this");
    expect(toolResults[1]).toContain(INTERRUPTED_TOOL_TEXT.slice(0, 40));
    expect(JSON.stringify(messages.at(-1)?.content)).toContain(RESUME_NOTE.slice(1, 40));
    // One log, one run_start, one run_end; seqs keep climbing.
    const all = events();
    expect(all.filter((event) => event.type === "run_start")).toHaveLength(1);
    expect(all.filter((event) => event.type === "run_end")).toHaveLength(1);
    expect(all.map((event) => event.seq)).toEqual(all.map((_, index) => index + 1));
  });

  it("does not ask the model again when a crash came after the final answer", async () => {
    writeSpec();
    await run([{ text: "All done." }]);
    const log = readFileSync(join(runDir(), "events.jsonl"), "utf8").split("\n").filter((line) => line !== "" && !line.includes('"run_end"'));
    writeFileSync(join(runDir(), "events.jsonl"), `${log.join("\n")}\n`);
    saveState(join(runDir(), "runner.json"), { ...freshState(runId), promptDelivered: true });
    expect(await run([{ text: "should not be asked" }])).toBe("complete");
    expect(gateway.completions()).toHaveLength(1);
    expect(events().filter((event) => event.type === "run_end")).toHaveLength(1);
  });

  it("counts a model call interrupted by a crash as a retry and gives up past the cap", async () => {
    writeSpec();
    writeFileSync(join(runDir(), "events.jsonl"), `${JSON.stringify({ seq: 1, ts: 1, type: "run_start", runId, model: "fake-model", prompt: "p", attachments: [], restored: false })}\n`);
    saveState(join(runDir(), "runner.json"), { ...freshState(runId), modelInFlight: true, interruptedModelCalls: 3 });
    expect(await run([{ text: "never" }])).toBe("error");
    expect(events().find((event) => event.type === "error")).toMatchObject({ code: "interrupted" });
    expect(gateway.completions()).toHaveLength(0);
  });

  it("ends a run that keeps restarting", async () => {
    writeSpec();
    writeFileSync(join(runDir(), "events.jsonl"), `${JSON.stringify({ seq: 1, ts: 1, type: "run_start", runId, model: "fake-model", prompt: "p", attachments: [], restored: false })}\n`);
    saveState(join(runDir(), "runner.json"), { ...freshState(runId), restarts: 5 });
    expect(await run([{ text: "never" }])).toBe("error");
    expect(events().find((event) => event.type === "error")).toMatchObject({ code: "restart_limit" });
    expect(events().at(-1)).toMatchObject({ type: "run_end", status: "error" });
  });
});
