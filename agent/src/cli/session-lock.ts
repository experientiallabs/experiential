/**
 * One exp-code process per session file.
 *
 * Pi appends to a session file from memory: each process keeps its own copy of the session tree
 * and writes entries whose parent is its own last entry. Two processes on one session (`-c` in two
 * terminals) therefore interleaved their entries, attached replies to the other process's messages,
 * and left `-c` resuming one branch with the other's replies hidden. A process now claims the
 * session file it opens with a lock file (`<agent dir>/session-locks/<digest>.json`) naming its pid
 * and start time, and releases it when it moves to another session or exits. A second process that
 * opens a claimed session is refused (print, JSON and RPC modes) or asked to fork it or start a new
 * one (the terminal UI); switching to a claimed session inside the UI is refused with the same
 * explanation (session-lock extension below).
 *
 * A lock whose pid is gone, or whose pid now belongs to a process started at another time (the pid
 * was reused after a crash), is stale and is replaced. A lock from another machine (a shared
 * configuration directory) is never judged stale, as its process cannot be checked from here.
 */

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { hostname } from "node:os";
import { join, resolve } from "node:path";
import type { ExtensionAPI, ExtensionFactory } from "@earendil-works/pi-coding-agent";

/** Where the session lock of the running exp-code is left for Pi's startup (pi-patches.ts). */
export const SESSION_LOCK = Symbol.for("exp-code.session-lock");

/** What a lock file records about the process holding a session. */
export interface SessionLockHolder {
  pid: number;
  /** When the holding process started, in milliseconds since the epoch (second resolution). */
  startedAt: number;
  host: string;
  sessionFile: string;
  /** When the lock was taken. */
  since: number;
}

/** Start times further apart than this belong to different processes. */
const START_TIME_TOLERANCE_MS = 2000;

/** When a process started (ms since the epoch), or undefined when it cannot be read here. */
export function processStartTime(pid: number): number | undefined {
  try {
    if (process.platform === "linux") {
      // Field 22 of /proc/<pid>/stat (after the parenthesized command name) is the start time in
      // clock ticks since boot; /proc/stat's btime is the boot time in seconds.
      const stat = readFileSync(`/proc/${pid}/stat`, "utf8");
      const fields = stat.slice(stat.lastIndexOf(")") + 2).split(" ");
      const ticks = Number(fields[19]);
      const btime = Number(/^btime (\d+)$/m.exec(readFileSync("/proc/stat", "utf8"))?.[1]);
      if (!Number.isFinite(ticks) || !Number.isFinite(btime)) return undefined;
      // Linux reports in USER_HZ, which is 100 on every supported architecture.
      return Math.floor((btime + ticks / 100) * 1000);
    }
    if (process.platform === "win32") return undefined;
    const started = execFileSync("ps", ["-o", "lstart=", "-p", String(pid)], { encoding: "utf8", env: { ...process.env, LC_ALL: "C" }, stdio: ["ignore", "pipe", "ignore"] }).trim();
    const ms = Date.parse(started);
    return Number.isFinite(ms) ? ms : undefined;
  } catch {
    return undefined;
  }
}

/** Whether a process with this pid exists. */
function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    // EPERM: it exists but belongs to someone else.
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

/** The lock file of a session file. */
export function sessionLockPath(lockDir: string, sessionFile: string): string {
  return join(lockDir, `${createHash("sha256").update(resolve(sessionFile)).digest("hex").slice(0, 32)}.json`);
}

/** Whether a recorded holder is still running: on this machine, alive, and the same process. */
export function holderAlive(holder: SessionLockHolder, startTime: (pid: number) => number | undefined = processStartTime): boolean {
  if (holder.host !== hostname()) return true;
  if (!Number.isInteger(holder.pid) || holder.pid <= 0 || !pidAlive(holder.pid)) return false;
  const started = startTime(holder.pid);
  // A start time that cannot be read is not evidence of reuse.
  return started === undefined || Math.abs(started - holder.startedAt) <= START_TIME_TOLERANCE_MS;
}

/** The session locks of one process. */
export class SessionLocks {
  private readonly held = new Map<string, string>();
  private readonly self: Omit<SessionLockHolder, "sessionFile" | "since">;

  constructor(
    readonly lockDir: string,
    private readonly startTime: (pid: number) => number | undefined = processStartTime
  ) {
    this.self = { pid: process.pid, startedAt: startTime(process.pid) ?? Math.round(Date.now() - process.uptime() * 1000), host: hostname() };
  }

  /**
   * Claim a session file for this process. Returns undefined when it is now ours (or there is no
   * file: an in-memory session), or the live process that holds it.
   */
  claim(sessionFile: string | undefined): SessionLockHolder | undefined {
    if (sessionFile === undefined) return undefined;
    const file = resolve(sessionFile);
    if (this.held.has(file)) return undefined;
    const path = sessionLockPath(this.lockDir, file);
    mkdirSync(this.lockDir, { recursive: true, mode: 0o700 });
    const record: SessionLockHolder = { ...this.self, sessionFile: file, since: Date.now() };
    // Two attempts: the second after removing a stale lock.
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        writeFileSync(path, `${JSON.stringify(record)}\n`, { flag: "wx", mode: 0o600 });
        this.held.set(file, path);
        return undefined;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
      }
      const holder = this.read(path);
      if (holder !== undefined && holder.pid === this.self.pid && holder.host === this.self.host && holder.startedAt === this.self.startedAt) {
        this.held.set(file, path);
        return undefined;
      }
      if (holder !== undefined && holderAlive(holder, this.startTime)) return holder;
      try {
        unlinkSync(path);
      } catch {
        // Another process cleared it first; try again.
      }
    }
    // Lost a race for a stale lock to another process, which now holds it.
    return this.read(path) ?? { ...this.self, pid: -1, sessionFile: file, since: Date.now() };
  }

  /** The live process holding a session file other than this one, if any. */
  holder(sessionFile: string): SessionLockHolder | undefined {
    const file = resolve(sessionFile);
    if (this.held.has(file)) return undefined;
    const holder = this.read(sessionLockPath(this.lockDir, file));
    if (holder === undefined || (holder.pid === this.self.pid && holder.host === this.self.host)) return undefined;
    return holderAlive(holder, this.startTime) ? holder : undefined;
  }

  /** Release every session this process holds except `keep`. */
  releaseAllBut(keep: string | undefined): void {
    const kept = keep === undefined ? undefined : resolve(keep);
    for (const [file, path] of this.held) {
      if (file === kept) continue;
      this.held.delete(file);
      try {
        const holder = this.read(path);
        if (holder?.pid === this.self.pid && holder.host === this.self.host) unlinkSync(path);
      } catch {
        // Already gone.
      }
    }
  }

  private read(path: string): SessionLockHolder | undefined {
    try {
      const parsed = JSON.parse(readFileSync(path, "utf8")) as Partial<SessionLockHolder>;
      if (typeof parsed.pid !== "number" || typeof parsed.startedAt !== "number") return undefined;
      return { pid: parsed.pid, startedAt: parsed.startedAt, host: String(parsed.host ?? ""), sessionFile: String(parsed.sessionFile ?? ""), since: Number(parsed.since ?? 0) };
    } catch {
      // Missing, or a half-written file of a process that died while writing it: not a holder.
      return undefined;
    }
  }
}

/** Why a session cannot be opened here, in one sentence naming the other process. */
export function describeHolder(holder: SessionLockHolder): string {
  const where = holder.host === hostname() ? `process ${holder.pid}` : `process ${holder.pid} on ${holder.host}`;
  const since = holder.since > 0 ? `, since ${new Date(holder.since).toLocaleString()}` : "";
  return `This session is open in another exp-code (${where}${since}). Two exp-code processes on one session would attach replies to each other's messages.`;
}

/** What a non-interactive run says when its session is open elsewhere. */
export function refusalMessage(holder: SessionLockHolder, sessionId: string | undefined): string {
  return `exp-code: ${describeHolder(holder)} Close it first, or continue in a copy of it: ${forkCommand(sessionId)}`;
}

/** The command that continues in a copy of a session. */
function forkCommand(sessionId: string | undefined): string {
  return `exp-code --fork ${sessionId ?? "<session>"} "..."`;
}

/** The session id in a session file name (`<timestamp>_<id>.jsonl`). */
export function sessionIdOf(sessionFile: string): string | undefined {
  return /_([^_/\\]+)\.jsonl$/.exec(sessionFile)?.[1];
}

/**
 * Install the process's session locks for Pi's startup (the session it opens) and release them
 * when the process exits. Returns the locks for the session-lock extension.
 */
export function installSessionLocks(lockDir: string): SessionLocks {
  const locks = new SessionLocks(lockDir);
  (process as unknown as Record<symbol, unknown>)[SESSION_LOCK] = {
    claim: (file: string | undefined) => locks.claim(file),
    refusal: refusalMessage,
    describe: describeHolder
  };
  process.once("exit", () => locks.releaseAllBut(undefined));
  return locks;
}

/**
 * Follow the session through the run: `/new`, `/resume`, `/fork` and RPC session switches claim
 * the new session file and release the previous one, and a switch to a session another process
 * holds is refused with the explanation.
 */
export function createSessionLockExtension(locks: SessionLocks): ExtensionFactory {
  return (pi: ExtensionAPI) => {
    pi.on("session_before_switch", (event, ctx) => {
      if (event.targetSessionFile === undefined) return undefined;
      const holder = locks.holder(event.targetSessionFile);
      if (holder === undefined) return undefined;
      ctx.ui.notify(`${describeHolder(holder)} Close it there first, or continue in a copy of it: ${forkCommand(sessionIdOf(event.targetSessionFile))}`, "error");
      return { cancel: true };
    });
    pi.on("session_start", (_event, ctx) => {
      const file = ctx.sessionManager.getSessionFile();
      // The startup claim happened before the session started (pi-patches.ts); a claim that fails
      // here is a session switched to without session_before_switch, which keeps running unlocked.
      locks.claim(file);
      locks.releaseAllBut(file);
    });
  };
}
