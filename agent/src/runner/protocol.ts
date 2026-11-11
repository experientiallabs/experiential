/**
 * The run protocol between a control plane and the headless runner that executes inside a
 * sandbox. It is also the on-disk format of a run's event log, so every field is a stable
 * contract for whatever reads it.
 *
 * Sandbox layout (all under SESSION_DIR, owned by root and never readable by the agent's own
 * commands, which run as an unprivileged user in WORK_DIR):
 *   session.jsonl               Pi's session file: the transcript, with compaction and branches
 *   runs/<runId>/run.json       RunSpec written by the control plane before the runner starts
 *   runs/<runId>/events.jsonl   RunEvent per line, append-only, seq from 1
 *   runs/<runId>/inbox.jsonl    InboxCommand per line, appended by the control plane
 *   runs/<runId>/status.json    RunStatus, rewritten atomically by the runner
 *   runs/<runId>/runner.json    The runner's own recovery state (restart count, inbox cursor)
 */

export const SESSION_DIR = "/opt/agent/session";
export const WORK_DIR = "/home/user/work";
export const RUNNER_PATH = "/opt/agent/runner.mjs";
/** Bumped whenever the runner bundle or the sandbox layout changes. */
export const RUNNER_PROTOCOL_VERSION = 4;

/** What the control plane asks one run to do. */
export interface RunSpec {
  runId: string;
  sessionId: string;
  /** Catalog slug the agent drives. */
  model: string;
  prompt: string;
  /** Paths (relative to WORK_DIR) the user attached to this message. */
  attachments: string[];
  vision: boolean;
  /** A raw gateway `reasoning_effort`, when the user chose one. */
  reasoningEffort: string | null;
  webSearch: boolean;
  /** OpenAI-compatible gateway base URL (`.../v1`). The key arrives in the env, never here. */
  gatewayBaseUrl: string;
  maxTurns: number;
  /** Epoch ms after which the runner stops itself. */
  deadline: number;
  /** Where the runner reports its end (best effort); bearer in the env. */
  callbackUrl: string | null;
  /**
   * Where the runner pushes batches of events as they are written (best effort, idempotent by
   * seq); bearer in the env. Null or absent means no push. The file log is always written.
   */
  eventsUrl?: string | null;
  /** True when this sandbox was just rebuilt from storage. */
  restored: boolean;
}

export type TodoStatus = "pending" | "in_progress" | "completed";

export interface TodoItem {
  content: string;
  status: TodoStatus;
}

export interface AgentFile {
  /** Relative to WORK_DIR. */
  path: string;
  size: number;
}

export type RunEndStatus = "complete" | "aborted" | "limit" | "error" | "interrupted";

/** One line of events.jsonl and one pushed event. */
export type RunEvent = { seq: number; ts: number } & RunEventBody;

export type RunEventBody =
  | { type: "run_start"; runId: string; model: string; prompt: string; attachments: string[]; restored: boolean }
  /** A message the user sent while the run was working (steering). */
  | { type: "user_message"; text: string }
  /** Coalesced assistant text (the runner flushes at most every 100 ms). */
  // endTs: set by a host that archives a compacted log, on an event that merged several
  // consecutive deltas (the last one's time); the runner never writes it.
  | { type: "text"; delta: string; endTs?: number }
  | { type: "thinking"; delta: string; endTs?: number }
  | { type: "tool_start"; id: string; name: string; args: Record<string, unknown> }
  /** The tail of a running tool's output so far (bash). */
  | { type: "tool_update"; id: string; output: string }
  | { type: "tool_end"; id: string; name: string; isError: boolean; output: string; durationMs: number }
  /** The agent's task list, replaced whole on every todo_write. */
  | { type: "todo"; items: TodoItem[] }
  /** The agent asked the user something and ended its turn. */
  | { type: "question"; id: string; question: string; options: string[] }
  /** One model response finished. costUsd is the gateway's settled cost when it reported one. */
  | {
      type: "turn";
      model: string;
      inputTokens: number;
      cachedInputTokens: number;
      outputTokens: number;
      costUsd: number | null;
      durationMs: number;
      stopReason: string;
    }
  /** The working directory changed (emitted after tool batches and at the end). */
  | { type: "files"; files: AgentFile[] }
  | { type: "error"; message: string; code?: string }
  | { type: "run_end"; status: RunEndStatus; turns: number; costUsd: number; durationMs: number };

export type RunEventType = RunEventBody["type"];

/**
 * Appended by the control plane to inbox.jsonl. A `requestId` makes the command idempotent: a
 * repeated id is ignored, including across runner restarts.
 */
export type InboxCommand = { type: "steer"; text: string; requestId?: string } | { type: "stop"; requestId?: string };

/** status.json: the run's lifecycle, read by the control plane to decide whether a run is alive. */
export interface RunStatus {
  runId: string;
  state: "running" | RunEndStatus;
  /** Epoch ms of the runner's last heartbeat (every 5 s while running). */
  heartbeatAt: number;
  startedAt: number;
  endedAt: number | null;
  turns: number;
  costUsd: number;
  lastSeq: number;
}

/** Body of one event push: events in seq order, each exactly as written to events.jsonl. */
export interface EventPushBody {
  runId: string;
  events: RunEvent[];
}

/** Parse one events.jsonl line, or null for a torn or foreign line. */
export function parseRunEvent(line: string): RunEvent | null {
  const trimmed = line.trim();
  if (trimmed === "") return null;
  try {
    const value = JSON.parse(trimmed) as Partial<RunEvent>;
    if (typeof value.seq !== "number" || typeof value.ts !== "number" || typeof value.type !== "string") return null;
    return value as RunEvent;
  } catch {
    return null;
  }
}
