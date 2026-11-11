/**
 * The headless runner process (bundled as `dist/runner.mjs` from `entry.ts`), executed inside a sandbox, normally as
 * root:
 *
 *   AGENT_GATEWAY_KEY=xpl_... [AGENT_CALLBACK_TOKEN=...] node runner.mjs <runId>
 *     [--session-dir DIR] [--work-dir DIR] [--user NAME]
 *
 * The secrets are read once and deleted from this process's environment. When the runner is
 * root, the model's commands run as the unprivileged account (default `user`) with a minimal
 * environment of their own.
 */

import { applyCliEnvironment } from "../cli/environment.js";
import { SESSION_DIR, WORK_DIR } from "./protocol.js";
import { accountEnvironment, accountIds, type SandboxAccount } from "./sandbox.js";

interface RunnerArgs {
  runId: string;
  sessionDir: string;
  workDir: string;
  user: string;
}

/** Parse the runner's arguments, or return a usage error. */
export function parseRunnerArgs(argv: string[]): RunnerArgs | string {
  const args: RunnerArgs = { runId: "", sessionDir: SESSION_DIR, workDir: WORK_DIR, user: "user" };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]!;
    const value = argv[index + 1];
    if (arg === "--session-dir" && value !== undefined) ((args.sessionDir = value), (index += 1));
    else if (arg === "--work-dir" && value !== undefined) ((args.workDir = value), (index += 1));
    else if (arg === "--user" && value !== undefined) ((args.user = value), (index += 1));
    else if (args.runId === "" && !arg.startsWith("-")) args.runId = arg;
    else return `unexpected argument: ${arg}`;
  }
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(args.runId)) return "usage: runner.mjs <runId> [--session-dir DIR] [--work-dir DIR] [--user NAME]";
  return args;
}

/** Run the runner for this process's arguments and exit with its status. */
export async function main(): Promise<void> {
  const parsed = parseRunnerArgs(process.argv.slice(2));
  if (typeof parsed === "string") {
    console.error(parsed);
    process.exit(2);
  }
  const apiKey = process.env.AGENT_GATEWAY_KEY ?? "";
  const callbackToken = process.env.AGENT_CALLBACK_TOKEN;
  delete process.env.AGENT_GATEWAY_KEY;
  delete process.env.AGENT_CALLBACK_TOKEN;
  // A missing key still runs: the gateway's refusal becomes the run's error event and the log
  // closes with run_end, so nobody waits on a silent exit.
  if (apiKey === "") console.error("AGENT_GATEWAY_KEY is empty; model calls will be refused");
  applyCliEnvironment(process.env);
  process.env.PI_CODING_AGENT_DIR = `${parsed.sessionDir}/agent`;
  let account: SandboxAccount | undefined;
  if (process.getuid?.() === 0) {
    const ids = accountIds(parsed.user);
    if (ids === undefined) {
      console.error(`account ${parsed.user} does not exist; refusing to run the model's commands as root`);
      process.exit(2);
    }
    account = { uid: ids.uid, gid: ids.gid, env: accountEnvironment(parsed.user, ids.home) };
  }
  const { runOnce } = await import("./runner.js");
  const status = await runOnce({ sessionDir: parsed.sessionDir, workDir: parsed.workDir, runId: parsed.runId, apiKey, callbackToken, account });
  process.exit(status === "error" ? 1 : 0);
}

