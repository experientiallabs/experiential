/**
 * The headless runner: executes one run of a session independent of any web request.
 *
 * It reads the RunSpec, drives an exp-code agent session built on Pi's SDK, appends RunEvents to
 * events.jsonl (and optionally pushes them over HTTP), applies steering and stop commands from
 * inbox.jsonl, keeps status.json's heartbeat fresh, and reports its end. Pi's session file is the
 * transcript, so compaction and continuity carry across runs. Everything is file based, so a
 * control plane that restarts, a closed browser, or a reconnect elsewhere changes nothing about
 * the run, and a runner that crashes continues the same run when started again.
 */

import { readFileSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import {
  type AgentSession,
  type AgentSessionEvent,
  createAgentSession,
  createBashToolDefinition,
  createEditToolDefinition,
  createLocalBashOperations,
  createReadToolDefinition,
  createWriteToolDefinition,
  DefaultResourceLoader,
  type ExtensionFactory,
  ModelRuntime,
  SessionManager,
  SettingsManager,
  type ToolDefinition
} from "@earendil-works/pi-coding-agent";
import type { AskUserDetails } from "../extension/ask-user.js";
import { createExpCodeExtension } from "../extension/index.js";
import { resolveSkillsDir } from "../extension/skills.js";
import { normalizeTodos } from "../extension/todo.js";
import { createWebFetchTool } from "../extension/web-fetch.js";
import { createWebSearchTool } from "../extension/web-search.js";
import { GATEWAY_PROVIDER_ID, type GatewayConfig, trimBaseUrl } from "../gateway/config.js";
import { DEFAULT_MAX_TOKENS, type ListedModel } from "../gateway/models.js";
import { gatewayProviderConfig, gatewayProviderModels } from "../gateway/provider.js";
import { INTERRUPTED_LIMIT_MESSAGE, modelErrorEvent, RESTART_LIMIT_MESSAGE, RUN_LIMIT_MESSAGE } from "./errors.js";
import { currentRun, EventLog, readEventLog } from "./event-log.js";
import { listWorkFiles } from "./files.js";
import { readInbox } from "./inbox.js";
import type { RunEndStatus, RunSpec, RunStatus } from "./protocol.js";
import { EventPusher } from "./push.js";
import { repairDanglingToolCalls, RESUME_NOTE } from "./recovery.js";
import { type SandboxAccount, sandboxBashOperations, sandboxEditOperations, sandboxWriteOperations } from "./sandbox.js";
import { freshState, loadState, saveState } from "./state.js";

const HEARTBEAT_MS = 5_000;
const INBOX_POLL_MS = 400;
const TOOL_OUTPUT_EVENT_CHARS = 6_000;
const TOOL_UPDATE_MIN_MS = 250;
/** Runner starts on one run after the first before the run is ended. */
export const RESTART_LIMIT = 5;
/** Model calls a crash may interrupt before the run is ended. */
export const INTERRUPTED_MODEL_CALL_LIMIT = 3;

/** What the next run sees after the user stopped this one. */
export const STOPPED_NOTE = "(I stopped the previous request before it finished. Do not continue it unless I ask again.)";

export interface RunnerOptions {
  /** The session directory (protocol SESSION_DIR). */
  sessionDir: string;
  /** The agent's working directory (protocol WORK_DIR). */
  workDir: string;
  runId: string;
  /** The run's gateway key. */
  apiKey: string;
  /** Bearer for the completion callback and the event push. */
  callbackToken?: string;
  /** Run the model's commands as this account (when the runner is root). */
  account?: SandboxAccount;
  /** fetch for the callback, push, and web search (model calls use Pi's own HTTP client). */
  fetch?: typeof globalThis.fetch;
  /** Clock, for tests. */
  now?: () => number;
}

function tail(text: string, limit: number): string {
  return text.length <= limit ? text : `...${text.slice(text.length - limit)}`;
}

function toolOutputText(result: unknown): string {
  const content = (result as { content?: { type: string; text?: string }[] } | undefined)?.content ?? [];
  return content.map((part) => (part.type === "text" ? (part.text ?? "") : `[${part.type}]`)).join("\n");
}

function writeAtomic(path: string, value: unknown): void {
  writeFileSync(`${path}.tmp`, JSON.stringify(value));
  renameSync(`${path}.tmp`, path);
}

/** The runner's own tool set: built-ins on sandbox operations plus the exp-code tools. */
function runnerTools(spec: RunSpec, options: RunnerOptions, gateway: GatewayConfig): Map<string, ToolDefinition> {
  const account = options.account;
  const shell = account ? sandboxBashOperations(account) : createLocalBashOperations();
  const tools: ToolDefinition[] = [
    createReadToolDefinition(options.workDir) as ToolDefinition,
    createWebFetchTool({ shell, cwd: options.workDir }) as ToolDefinition,
    createWebSearchTool({ gateway, model: spec.model, fetch: options.fetch }) as ToolDefinition
  ];
  if (account) {
    tools.push(
      createBashToolDefinition(options.workDir, { operations: shell }) as ToolDefinition,
      createWriteToolDefinition(options.workDir, { operations: sandboxWriteOperations(account) }) as ToolDefinition,
      createEditToolDefinition(options.workDir, { operations: sandboxEditOperations(account) }) as ToolDefinition
    );
  }
  return new Map(tools.map((tool) => [tool.name, tool]));
}

/** Run one RunSpec to its end. Resolves with the end status; never throws for agent failures. */
export async function runOnce(options: RunnerOptions): Promise<RunEndStatus> {
  const now = options.now ?? Date.now;
  const runDir = join(options.sessionDir, "runs", options.runId);
  const spec = JSON.parse(readFileSync(join(runDir, "run.json"), "utf8")) as RunSpec;
  const eventsPath = join(runDir, "events.jsonl");
  const statePath = join(runDir, "runner.json");
  const statusPath = join(runDir, "status.json");
  const existing = readEventLog(eventsPath);
  const previous = currentRun(existing);
  const restarting = previous.started && !previous.ended;
  const stored = loadState(statePath, spec.runId);
  // A new run of the same id starts fresh but never re-applies inbox commands already read.
  const state = restarting
    ? stored
    : { ...freshState(spec.runId), inboxOffset: stored.inboxOffset, appliedRequestIds: stored.appliedRequestIds };
  if (restarting) state.restarts += 1;
  saveState(statePath, state);

  const log = new EventLog(eventsPath, now, existing);
  const pusher =
    spec.eventsUrl && options.callbackToken
      ? new EventPusher({ url: spec.eventsUrl, token: options.callbackToken, runId: spec.runId, fetch: options.fetch })
      : null;
  if (pusher !== null) {
    log.onWritten((event) => pusher.enqueue(event));
    // Events of this run written before a crash may not have been pushed; resending is safe.
    if (restarting) for (const event of previous.events) pusher.enqueue(event);
  }
  const emit = log.emit.bind(log);

  const startedAt = now();
  const status: RunStatus = { runId: spec.runId, state: "running", heartbeatAt: startedAt, startedAt, endedAt: null, turns: 0, costUsd: 0, lastSeq: log.seq };
  if (restarting) {
    for (const event of previous.events) if (event.type === "turn") ((status.turns += 1), (status.costUsd += event.costUsd ?? 0));
  }
  const writeStatus = () => {
    status.heartbeatAt = now();
    status.lastSeq = log.seq;
    writeAtomic(statusPath, status);
  };
  const finish = async (endStatus: RunEndStatus): Promise<RunEndStatus> => {
    emit({ type: "run_end", status: endStatus, turns: status.turns, costUsd: status.costUsd, durationMs: now() - startedAt });
    log.flushText();
    status.state = endStatus;
    status.endedAt = now();
    writeStatus();
    if (pusher !== null) {
      await pusher.flush();
      pusher.close();
    }
    await reportEnd(spec, options, endStatus);
    return endStatus;
  };

  writeStatus();
  if (!restarting) {
    emit({ type: "run_start", runId: spec.runId, model: spec.model, prompt: spec.prompt, attachments: spec.attachments, restored: spec.restored });
  }
  if (state.restarts > RESTART_LIMIT) {
    emit({ type: "error", message: RESTART_LIMIT_MESSAGE, code: "restart_limit" });
    return finish("error");
  }
  if (restarting && state.modelInFlight) {
    state.interruptedModelCalls += 1;
    state.modelInFlight = false;
    saveState(statePath, state);
    if (state.interruptedModelCalls > INTERRUPTED_MODEL_CALL_LIMIT) {
      emit({ type: "error", message: INTERRUPTED_LIMIT_MESSAGE, code: "interrupted" });
      return finish("error");
    }
  }

  const gateway: GatewayConfig = { kind: "hosted", baseUrl: trimBaseUrl(spec.gatewayBaseUrl), apiKey: options.apiKey };
  const tools = runnerTools(spec, options, gateway);
  const sessionManager = SessionManager.open(join(options.sessionDir, "session.jsonl"), options.sessionDir, options.workDir);
  // A restarted run whose agent had already finished its answer has nothing left to ask for.
  let answered = false;
  if (restarting) {
    await repairDanglingToolCalls({ sessionManager, state, tools, replayLimit: RESTART_LIMIT, now });
    state.toolsInFlight = [];
    saveState(statePath, state);
    const last = sessionManager.buildSessionContext().messages.at(-1);
    answered = state.promptDelivered && last?.role === "assistant" && last.stopReason === "stop";
  }

  const agentDir = join(options.sessionDir, "agent");
  const runnerModels: ListedModel[] = [{ id: spec.model, vision: spec.vision, reasoning: false, maxTokens: DEFAULT_MAX_TOKENS }];
  const settingsManager = SettingsManager.inMemory(
    { compaction: { enabled: true } },
    { projectTrusted: false }
  );
  const runnerExtension: ExtensionFactory = (pi) => {
    if (options.account) for (const name of ["bash", "write", "edit"]) pi.registerTool(tools.get(name)!);
    if (spec.reasoningEffort !== null) {
      const effort = spec.reasoningEffort;
      pi.on("before_provider_request", (event) => ({ ...(event.payload as Record<string, unknown>), reasoning_effort: effort }));
    }
  };
  const resourceLoader = new DefaultResourceLoader({
    cwd: options.workDir,
    agentDir,
    settingsManager,
    // The working directory is writable by the model; never load code or templates from it.
    noExtensions: true,
    noPromptTemplates: true,
    noThemes: true,
    additionalSkillPaths: [resolveSkillsDir(agentDir)],
    extensionFactories: [
      {
        name: "exp-code",
        factory: createExpCodeExtension({
          gateway,
          shell: options.account ? sandboxBashOperations(options.account) : undefined,
          searchModel: spec.model,
          skillsRoot: agentDir,
          provider: {
            fetch: options.fetch,
            wire: "openai-completions",
            models: runnerModels
          }
        })
      },
      { name: "exp-code-runner", factory: runnerExtension }
    ]
  });
  await resourceLoader.reload();

  // The model is resolved here, never by Pi's fallback, so a run can only call its gateway.
  const providerModels = gatewayProviderModels(runnerModels, gateway, "openai-completions");
  const modelRuntime = await ModelRuntime.create({ authPath: join(agentDir, "auth.json"), modelsPath: join(agentDir, "models.json") });
  modelRuntime.registerProvider(GATEWAY_PROVIDER_ID, gatewayProviderConfig(gateway, providerModels));
  const model = modelRuntime.getModel(GATEWAY_PROVIDER_ID, spec.model);
  let session: AgentSession;
  try {
    if (model === undefined) throw new Error(`model ${spec.model} is not registered`);
    ({ session } = await createAgentSession({
      cwd: options.workDir,
      agentDir,
      modelRuntime,
      model,
      sessionManager,
      settingsManager,
      resourceLoader,
      thinkingLevel: "off",
      excludeTools: spec.webSearch ? [] : ["web_search"]
    }));
  } catch (error) {
    emit({ type: "error", message: `The agent could not start: ${(error as Error).message}` });
    return finish("error");
  }

  let endStatus = "complete" as RunEndStatus;
  let pendingError: { message: string; code?: string } | undefined;
  let questionAsked = false;
  let running = false;
  const lateMessages: string[] = [];
  const lastToolUpdate = new Map<string, number>();
  let turnStartedAt = now();
  let filesUpdate: Promise<void> = Promise.resolve();
  const refreshFiles = () => {
    filesUpdate = filesUpdate.then(async () => emit({ type: "files", files: await listWorkFiles(options.workDir) }));
  };
  const stop = (reason: RunEndStatus) => {
    if (endStatus === "complete") endStatus = reason;
    void session.abort();
  };

  const onEvent = (event: AgentSessionEvent) => {
    switch (event.type) {
      case "message_update": {
        const update = event.assistantMessageEvent;
        if (update.type === "text_delta") emit({ type: "text", delta: update.delta });
        else if (update.type === "thinking_delta") emit({ type: "thinking", delta: update.delta });
        break;
      }
      case "message_start": {
        if (event.message.role === "assistant") {
          turnStartedAt = now();
          state.modelInFlight = true;
          saveState(statePath, state);
        }
        break;
      }
      case "tool_execution_start": {
        const args = (event.args ?? {}) as Record<string, unknown>;
        state.toolsInFlight.push({ id: event.toolCallId, name: event.toolName, args });
        saveState(statePath, state);
        if (event.toolName === "todo_write") emit({ type: "todo", items: normalizeTodos(args.todos) });
        emit({ type: "tool_start", id: event.toolCallId, name: event.toolName, args });
        break;
      }
      case "tool_execution_update": {
        const at = now();
        if (at - (lastToolUpdate.get(event.toolCallId) ?? 0) < TOOL_UPDATE_MIN_MS) break;
        lastToolUpdate.set(event.toolCallId, at);
        emit({ type: "tool_update", id: event.toolCallId, output: tail(toolOutputText(event.partialResult), TOOL_OUTPUT_EVENT_CHARS) });
        break;
      }
      case "tool_execution_end": {
        state.toolsInFlight = state.toolsInFlight.filter((tool) => tool.id !== event.toolCallId);
        saveState(statePath, state);
        emit({
          type: "tool_end",
          id: event.toolCallId,
          name: event.toolName,
          isError: event.isError,
          output: tail(toolOutputText(event.result), TOOL_OUTPUT_EVENT_CHARS),
          durationMs: event.durationMs ?? 0
        });
        const details = (event.result as { details?: AskUserDetails } | undefined)?.details;
        if (event.toolName === "ask_user" && !event.isError && details !== undefined && details.answer === undefined) {
          questionAsked = true;
          emit({ type: "question", id: event.toolCallId, question: details.question, options: details.options });
        }
        break;
      }
      case "turn_end":
        refreshFiles();
        break;
      case "auto_retry_start":
        pendingError = undefined;
        break;
      case "message_end": {
        const message = event.message;
        if (message.role === "user" && running) {
          if (!state.promptDelivered) {
            state.promptDelivered = true;
            saveState(statePath, state);
          }
          break;
        }
        if (message.role !== "assistant") break;
        const assistant = message as AssistantMessage;
        state.modelInFlight = false;
        saveState(statePath, state);
        status.turns += 1;
        const cost = assistant.usage.cost.totalFromProvider ? assistant.usage.cost.total : null;
        status.costUsd += cost ?? 0;
        emit({
          type: "turn",
          model: assistant.responseModel ?? spec.model,
          inputTokens: assistant.usage.input + assistant.usage.cacheRead,
          cachedInputTokens: assistant.usage.cacheRead,
          outputTokens: assistant.usage.output,
          costUsd: cost,
          durationMs: assistant.durationMs ?? now() - turnStartedAt,
          stopReason: assistant.stopReason
        });
        if (assistant.stopReason === "error") pendingError = modelErrorEvent(assistant.errorMessage);
        if (status.turns >= spec.maxTurns && assistant.stopReason === "toolUse") stop("limit");
        writeStatus();
        break;
      }
      default:
        break;
    }
  };
  const unsubscribe = session.subscribe(onEvent);

  const pollInbox = () => {
    const { commands, changed } = readInbox(join(runDir, "inbox.jsonl"), state);
    if (changed) saveState(statePath, state);
    for (const command of commands) {
      if (command.type === "stop") {
        stop("aborted");
      } else if (command.text.trim() !== "") {
        emit({ type: "user_message", text: command.text });
        if (running && session.isStreaming) {
          void session.steer(command.text).catch(() => lateMessages.push(command.text));
        } else {
          lateMessages.push(command.text);
        }
      }
    }
  };

  const heartbeat = setInterval(() => {
    try {
      writeStatus();
    } catch {
      // The next beat retries.
    }
  }, HEARTBEAT_MS);
  const inbox = setInterval(pollInbox, INBOX_POLL_MS);
  const deadline = setTimeout(() => stop("limit"), Math.max(1_000, spec.deadline - now()));
  const onSignal = () => stop("interrupted");
  process.once("SIGTERM", onSignal);
  process.once("SIGINT", onSignal);

  const prompt = async (text: string) => {
    running = true;
    try {
      await session.prompt(text);
    } finally {
      running = false;
    }
  };
  try {
    pollInbox();
    if (endStatus === "complete" && !answered) {
      const attachmentNote = spec.attachments.length > 0 ? `\n\n(Attached files, in the working directory: ${spec.attachments.join(", ")})` : "";
      await prompt(state.promptDelivered ? RESUME_NOTE : spec.prompt + attachmentNote);
    }
    // Messages that arrived after the agent stopped working become one more prompt.
    pollInbox();
    while (lateMessages.length > 0 && endStatus === "complete" && !questionAsked && pendingError === undefined) {
      await prompt(lateMessages.splice(0, lateMessages.length).join("\n\n"));
      pollInbox();
    }
  } catch (error) {
    if (endStatus === "complete") endStatus = "error";
    emit({ type: "error", message: error instanceof Error ? error.message : String(error) });
  } finally {
    clearInterval(heartbeat);
    clearInterval(inbox);
    clearTimeout(deadline);
    process.removeListener("SIGTERM", onSignal);
    process.removeListener("SIGINT", onSignal);
  }
  await session.waitForIdle().catch(() => undefined);
  unsubscribe();

  if (pendingError !== undefined && endStatus === "complete") {
    endStatus = "error";
    emit({ type: "error", ...pendingError });
  }
  if (endStatus === "aborted") {
    // Without a word, the next message would read as an addition to the stopped request and
    // the agent would pick that work back up. Say it was cancelled.
    sessionManager.appendMessage({ role: "user", content: STOPPED_NOTE, timestamp: now() });
  }
  session.dispose();
  await filesUpdate;
  emit({ type: "files", files: await listWorkFiles(options.workDir) });
  if (endStatus === "limit") emit({ type: "error", message: RUN_LIMIT_MESSAGE, code: "run_limit" });
  return finish(endStatus);
}

async function reportEnd(spec: RunSpec, options: RunnerOptions, status: RunEndStatus): Promise<void> {
  if (spec.callbackUrl === null || !options.callbackToken) return;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const response = await (options.fetch ?? globalThis.fetch)(spec.callbackUrl, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${options.callbackToken}` },
        body: JSON.stringify({ runId: spec.runId, status }),
        signal: AbortSignal.timeout(10_000)
      });
      if (response.ok) return;
    } catch {
      // Retry below.
    }
    await new Promise((resolve) => setTimeout(resolve, 1_000 * (attempt + 1)));
  }
}
