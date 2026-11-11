/**
 * The runner's recovery state for one run (runs/<runId>/runner.json).
 *
 * It records what a restarted runner needs to continue safely: how many times the run has
 * restarted, how many model calls a crash interrupted, whether the prompt reached the session,
 * which tool calls were running, and how far the inbox has been applied (by byte offset and by
 * requestId). It is rewritten atomically whenever it changes.
 */

import { existsSync, readFileSync, renameSync, writeFileSync } from "node:fs";

export interface InFlightTool {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export interface RunnerState {
  runId: string;
  /** Times a runner process started on this run after the first. */
  restarts: number;
  /** Model calls that were interrupted by a crash (each counts as a retry). */
  interruptedModelCalls: number;
  /** Whether the run's prompt is in the session. */
  promptDelivered: boolean;
  /** A model call was streaming when the state was last written. */
  modelInFlight: boolean;
  /** Tool calls that had started and not finished. */
  toolsInFlight: InFlightTool[];
  /** Times each tool call id has been replayed after a restart. */
  replays: Record<string, number>;
  /** Byte offset of the first inbox line not yet read. */
  inboxOffset: number;
  /** requestIds already applied; later duplicates are ignored. */
  appliedRequestIds: string[];
}

export function freshState(runId: string): RunnerState {
  return {
    runId,
    restarts: 0,
    interruptedModelCalls: 0,
    promptDelivered: false,
    modelInFlight: false,
    toolsInFlight: [],
    replays: {},
    inboxOffset: 0,
    appliedRequestIds: []
  };
}

/** Load the state for a run, or a fresh one when the file is missing, torn, or for another run. */
export function loadState(path: string, runId: string): RunnerState {
  if (!existsSync(path)) return freshState(runId);
  try {
    const parsed = JSON.parse(readFileSync(path, "utf8")) as Partial<RunnerState>;
    if (parsed.runId !== runId) return freshState(runId);
    return { ...freshState(runId), ...parsed };
  } catch {
    return freshState(runId);
  }
}

export function saveState(path: string, state: RunnerState): void {
  writeFileSync(`${path}.tmp`, JSON.stringify(state));
  renameSync(`${path}.tmp`, path);
}
