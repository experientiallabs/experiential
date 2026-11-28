/**
 * Sessions and signals end to end, with the built CLI against the local fake gateway: one process
 * per session file (session-lock.ts), a SIGINT from another process restoring the terminal, and
 * RPC mode finishing the running turn when stdin closes.
 */

import { type ChildProcess, execFile, execFileSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { hostname, tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { type FakeGateway, startFakeGateway } from "../testing/fake-gateway.js";
import { findPython, pause, Terminal } from "../testing/pty.js";
import { holderAlive, processStartTime, SessionLocks, sessionLockPath } from "./session-lock.js";

const packageDir = fileURLToPath(new URL("../..", import.meta.url));
// A build of its own, so this file never races the other end-to-end builds.
const outDir = join(packageDir, ".e2e-dist-sessions");
const cli = join(outDir, "cli.js");
const python = findPython();

let root: string;
let gateway: FakeGateway;

beforeAll(async () => {
  execFileSync(process.execPath, [join(packageDir, "node_modules", "typescript", "bin", "tsc"), "-p", join(packageDir, "tsconfig.build.json"), "--outDir", outDir], { stdio: "inherit" });
  root = mkdtempSync(join(tmpdir(), "exp-code-sessions-e2e-"));
  gateway = await startFakeGateway();
}, 120_000);

afterAll(async () => {
  await gateway.close();
  rmSync(root, { recursive: true, force: true });
  rmSync(outDir, { recursive: true, force: true });
});

/** A fresh home and project folder. */
function fresh(): { home: string; project: string; env: NodeJS.ProcessEnv } {
  const home = mkdtempSync(join(root, "home-"));
  const project = mkdtempSync(join(home, "project-"));
  const env = { PATH: process.env.PATH, HOME: home, EXP_CODE_HOME: join(home, ".exp-code"), EXP_GATEWAY_KEY: "k", EXP_CODE_BASE_URL: gateway.baseUrl, TERM: "xterm-256color", PI_OFFLINE: "1" };
  return { home, project, env };
}

/** Run the built CLI to the end with stdin closed. */
function run(args: string[], env: NodeJS.ProcessEnv, cwd: string): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const child = execFile(process.execPath, [cli, ...args], { cwd, env, timeout: 60_000 }, (error, stdout, stderr) =>
      resolve({ code: error ? (typeof error.code === "number" ? error.code : 1) : 0, stdout, stderr })
    );
    child.stdin?.end();
  });
}

/** Every session file under a home. */
function sessionFiles(home: string): string[] {
  const dir = join(home, ".exp-code", "agent", "sessions");
  if (!existsSync(dir)) return [];
  return readdirSync(dir, { recursive: true, encoding: "utf8" })
    .filter((name) => name.endsWith(".jsonl"))
    .map((name) => join(dir, name));
}

/** An RPC-mode exp-code that stays up until its stdin closes; `lines` collects its output. */
function startRpc(args: string[], env: NodeJS.ProcessEnv, cwd: string): { child: ChildProcess; lines: () => Record<string, unknown>[]; exited: Promise<number> } {
  const child = spawn(process.execPath, [cli, "--mode", "rpc", ...args], { cwd, env, stdio: ["pipe", "pipe", "pipe"] });
  let out = "";
  child.stdout!.on("data", (chunk: Buffer) => (out += chunk.toString("utf8")));
  const exited = new Promise<number>((resolve) => child.once("exit", (code) => resolve(code ?? -1)));
  const lines = () =>
    out
      .split("\n")
      .filter((line) => line.trim() !== "")
      .map((line) => JSON.parse(line) as Record<string, unknown>);
  return { child, lines, exited };
}

async function until(check: () => boolean, what: string, ms = 30_000): Promise<void> {
  const end = Date.now() + ms;
  while (!check()) {
    if (Date.now() > end) throw new Error(`timed out waiting for ${what}`);
    await pause(50);
  }
}

describe("RPC mode", () => {
  it("finishes the running turn when stdin closes right after the prompt", async () => {
    const { project, env } = fresh();
    gateway.script([{ text: "the late answer", delayMs: 1500 }]);
    const rpc = startRpc(["--no-session"], env, project);
    // A last line without a line feed, then end of input at once.
    rpc.child.stdin!.end('{"id":"p","type":"prompt","message":"hi"}');
    expect(await rpc.exited).toBe(0);
    const types = rpc.lines().map((line) => line.type);
    expect(types).toContain("agent_end");
    expect(JSON.stringify(rpc.lines())).toContain("the late answer");
  }, 60_000);
});

describe("one process per session", () => {
  it("refuses a second process on a session that is open, and allows it once it is closed", async () => {
    const { home, project, env } = fresh();
    gateway.script([{ text: "first answer" }]);
    expect((await run(["-p", "first"], env, project)).code).toBe(0);
    const [file] = sessionFiles(home);
    expect(file).toBeDefined();

    // Another exp-code continues the session and stays open.
    const holder = startRpc(["-c"], env, project);
    holder.child.stdin!.write('{"id":"s","type":"get_state"}\n');
    await until(() => holder.lines().some((line) => line.id === "s"), "the holder's state");
    const lock = sessionLockPath(join(home, ".exp-code", "agent", "session-locks"), file!);
    expect(JSON.parse(readFileSync(lock, "utf8")).pid).toBe(holder.child.pid);

    const before = gateway.completions().length;
    const second = await run(["-p", "-c", "second"], env, project);
    expect(second.code).toBe(1);
    expect(second.stderr).toContain(`This session is open in another exp-code (process ${holder.child.pid}`);
    expect(second.stderr).toMatch(/exp-code --fork \S+/);
    expect(gateway.completions()).toHaveLength(before);

    // Forking it as the message says works while it is open.
    const id = /exp-code --fork (\S+)/.exec(second.stderr)![1]!;
    gateway.script([{ text: "forked answer" }]);
    const forked = await run(["-p", "--fork", id, "in the copy"], env, project);
    expect(forked.stdout).toContain("forked answer");

    holder.child.stdin!.end();
    expect(await holder.exited).toBe(0);
    expect(existsSync(lock)).toBe(false);
    gateway.script([{ text: "third answer" }]);
    const third = await run(["-p", "-c", "third"], env, project);
    expect(third.code).toBe(0);
    expect(third.stdout).toContain("third answer");
  }, 90_000);

  it("clears the lock of a process that is gone, or whose pid now belongs to another process", async () => {
    const { home, project, env } = fresh();
    gateway.script([{ text: "first answer" }]);
    await run(["-p", "first"], env, project);
    const [file] = sessionFiles(home);
    const lockDir = join(home, ".exp-code", "agent", "session-locks");
    const lock = sessionLockPath(lockDir, file!);
    mkdirSync(lockDir, { recursive: true });

    // A crashed process: its pid no longer exists.
    const dead = spawn(process.execPath, ["-e", ""]);
    await new Promise((resolve) => dead.once("exit", resolve));
    writeFileSync(lock, JSON.stringify({ pid: dead.pid, startedAt: Date.now(), host: hostname(), sessionFile: file, since: Date.now() }));
    gateway.script([{ text: "after a crash" }]);
    expect((await run(["-p", "-c", "again"], env, project)).stdout).toContain("after a crash");

    // A reused pid: alive (this test), but started at another time than the lock records.
    writeFileSync(lock, JSON.stringify({ pid: process.pid, startedAt: 1, host: hostname(), sessionFile: file, since: Date.now() }));
    gateway.script([{ text: "after pid reuse" }]);
    expect((await run(["-p", "-c", "again"], env, project)).stdout).toContain("after pid reuse");
    expect(existsSync(lock)).toBe(false);
  }, 90_000);

  it.skipIf(python === undefined)("offers a fork in the terminal UI, which continues in a copy of the session", async () => {
    const { home, project, env } = fresh();
    gateway.script([{ text: "first answer" }]);
    await run(["-p", "first"], env, project);
    const [file] = sessionFiles(home);
    const original = readFileSync(file!, "utf8");
    const holder = startRpc(["-c"], env, project);
    holder.child.stdin!.write('{"id":"s","type":"get_state"}\n');
    await until(() => holder.lines().some((line) => line.id === "s"), "the holder's state");
    const terminal = new Terminal(cli, env, project, ["-c"]);
    try {
      await terminal.waitForText("This session is open in another exp-code");
      await terminal.waitForText("Continue in a copy of this session");
      terminal.send("\r");
      await terminal.waitForText("first answer");
      await pause(1000);
      gateway.script([{ text: "answer in the copy" }]);
      await terminal.type("next\r");
      await terminal.waitForText("answer in the copy");
      await terminal.waitFor(() => sessionFiles(home).length === 2, "the forked session file");
      // The open session was not written to by the second process.
      expect(readFileSync(file!, "utf8")).toBe(original);
    } finally {
      terminal.kill();
      holder.child.stdin!.end();
      await holder.exited;
    }
  }, 90_000);

  it("judges holders by pid and start time", () => {
    const lockDir = mkdtempSync(join(root, "locks-"));
    const mine = new SessionLocks(lockDir);
    expect(mine.claim(join(lockDir, "a.jsonl"))).toBeUndefined();
    // The same process claims again without conflict; another process (another start time) is refused.
    expect(mine.claim(join(lockDir, "a.jsonl"))).toBeUndefined();
    const self = processStartTime(process.pid);
    if (self !== undefined) expect(Math.abs(self - (Date.now() - process.uptime() * 1000))).toBeLessThan(3000);
    expect(holderAlive({ pid: process.pid, startedAt: 5, host: hostname(), sessionFile: "", since: 0 }, () => 10_000)).toBe(false);
    expect(holderAlive({ pid: process.pid, startedAt: 10_000, host: hostname(), sessionFile: "", since: 0 }, () => 10_500)).toBe(true);
    // Another machine's process cannot be checked from here: it holds.
    expect(holderAlive({ pid: 999_999_999, startedAt: 0, host: "elsewhere", sessionFile: "", since: 0 })).toBe(true);
    mine.releaseAllBut(undefined);
    expect(readdirSync(lockDir).filter((name) => name.endsWith(".json"))).toEqual([]);
    expect(statSync(lockDir).isDirectory()).toBe(true);
  });
});

describe.skipIf(python === undefined)("signals in the terminal UI", () => {
  it("restores the terminal on a SIGINT from another process, and still suspends with Ctrl+Z and quits with Ctrl+C", async () => {
    const { project, env } = fresh();
    const terminal = new Terminal(cli, env, project);
    try {
      await terminal.waitForText("fake-model");
      await pause(1500);

      // Ctrl+Z suspends: the UI stops and gives the terminal back. (The kernel discards the stop
      // signal itself here, as a pseudo-terminal's only process group has no shell to stop it for,
      // so the process waits for SIGCONT running.) A SIGINT while suspended is ignored.
      const pid = terminal.cliPid();
      const beforeSuspend = terminal.output.length;
      terminal.send("\x1a");
      await terminal.waitFor(() => terminal.output.slice(beforeSuspend).includes("\x1b[?2004l"), "the suspended UI");
      process.kill(pid, "SIGINT");
      await pause(1000);
      expect(() => process.kill(pid, 0)).not.toThrow();
      const resumed = terminal.screen().length;
      process.kill(pid, "SIGCONT");
      await pause(500);
      await terminal.type("/session\r");
      await terminal.waitForText("Session Info", resumed);

      // kill -INT from outside: the terminal modes are reset, and it exits 130.
      const seen = terminal.output.length;
      process.kill(pid, "SIGINT");
      const code = await terminal.exited;
      const tail = terminal.output.slice(seen);
      expect(tail).toContain("\x1b[?2004l");
      expect(tail).toContain("\x1b[?25h");
      expect(code).toBe(130);
    } finally {
      terminal.kill();
    }

    // Ctrl+C inside the UI is a key, not a signal: twice on an empty prompt quits.
    const second = new Terminal(cli, env, project);
    try {
      await second.waitForText("fake-model");
      await pause(1500);
      second.send("\x03");
      await pause(300);
      second.send("\x03");
      expect(await second.exited).toBe(0);
    } finally {
      second.kill();
    }
  }, 120_000);
});
