/**
 * Where the model's commands and file writes happen when the runner is root.
 *
 * Every shell command runs as an unprivileged account with a fixed, minimal environment, so the
 * runner's secrets never reach the model's processes (another user's /proc/<pid>/environ is
 * unreadable, and the secrets are already deleted from the runner's own environment). Files the
 * model reads and writes run as that account too. The operations plug into Pi's built-in tools.
 */

import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import {
  detectSupportedImageMimeTypeFromFile,
  type BashOperations,
  type EditOperations,
  type ReadOperations,
  type WriteOperations
} from "@earendil-works/pi-coding-agent";

export interface SandboxAccount {
  uid: number;
  gid: number;
  /** The complete environment for the account's commands. */
  env: NodeJS.ProcessEnv;
}

/** Variables Pi adds to describe the session; the only ones passed through from Pi. */
const SESSION_VARIABLES = ["PI_SESSION_ID", "PI_SESSION_FILE", "PI_PROVIDER", "PI_MODEL", "PI_REASONING_LEVEL"];

/** uid and gid of a local account, from /etc/passwd. */
export function accountIds(name: string, passwd = "/etc/passwd"): { uid: number; gid: number; home: string } | undefined {
  try {
    for (const line of readFileSync(passwd, "utf8").split("\n")) {
      const [user, , uid, gid, , home] = line.split(":");
      if (user === name && uid !== undefined && gid !== undefined) return { uid: Number(uid), gid: Number(gid), home: home ?? `/home/${name}` };
    }
  } catch {
    // Not a Linux sandbox.
  }
  return undefined;
}

/** The scrubbed environment for an account's commands. */
export function accountEnvironment(name: string, home: string): NodeJS.ProcessEnv {
  return {
    HOME: home,
    USER: name,
    LOGNAME: name,
    PATH: "/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin",
    LANG: "C.UTF-8",
    TERM: "dumb",
    PYTHONUNBUFFERED: "1",
    MPLBACKEND: "Agg",
    PIP_DISABLE_PIP_VERSION_CHECK: "1"
  };
}

/** Run bash commands as the account, killing the whole process group on abort or timeout. */
export function sandboxBashOperations(account: SandboxAccount): BashOperations {
  return {
    exec: (command, cwd, { onData, signal, timeout, env }) =>
      new Promise((resolve, reject) => {
        if (signal?.aborted) {
          reject(new Error("aborted"));
          return;
        }
        const childEnv: NodeJS.ProcessEnv = { ...account.env };
        for (const name of SESSION_VARIABLES) if (env?.[name] !== undefined) childEnv[name] = env[name];
        const child = spawn("bash", ["-c", command], {
          cwd,
          env: childEnv,
          uid: account.uid,
          gid: account.gid,
          detached: true,
          stdio: ["ignore", "pipe", "pipe"]
        });
        let timedOut = false;
        const killTree = () => {
          try {
            if (child.pid !== undefined) process.kill(-child.pid, "SIGKILL");
          } catch {
            child.kill("SIGKILL");
          }
        };
        const timer = timeout !== undefined && timeout > 0 ? setTimeout(() => ((timedOut = true), killTree()), timeout * 1000) : undefined;
        signal?.addEventListener("abort", killTree, { once: true });
        child.stdout.on("data", onData);
        child.stderr.on("data", onData);
        child.on("error", (error) => onData(Buffer.from(`\n${error.message}\n`)));
        child.on("close", (code, killSignal) => {
          if (timer !== undefined) clearTimeout(timer);
          signal?.removeEventListener("abort", killTree);
          if (signal?.aborted) reject(new Error("aborted"));
          else if (timedOut) reject(new Error(`Command timed out after ${timeout} seconds`));
          else resolve({ exitCode: code ?? (killSignal ? 128 + 9 : 1) });
        });
      })
  };
}

/** One short command run as the account: its exit code, stdout bytes and stderr text. */
export function runAsAccount(account: SandboxAccount, argv: string[], input?: string | Uint8Array): Promise<{ code: number; stdout: Buffer; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(argv[0]!, argv.slice(1), {
      cwd: "/",
      env: account.env,
      uid: account.uid,
      gid: account.gid,
      stdio: [input === undefined ? "ignore" : "pipe", "pipe", "pipe"]
    });
    const out: Buffer[] = [];
    const err: Buffer[] = [];
    child.stdout!.on("data", (chunk: Buffer) => out.push(chunk));
    child.stderr!.on("data", (chunk: Buffer) => err.push(chunk));
    child.on("error", reject);
    child.on("close", (code) => resolve({ code: code ?? 1, stdout: Buffer.concat(out), stderr: Buffer.concat(err).toString("utf8").trim() }));
    if (input !== undefined) {
      // A command that fails early (no permission to create the file) closes its input before
      // reading it; the exit code reports that, so the broken pipe is not a second error.
      child.stdin!.on("error", () => undefined);
      child.stdin!.end(input);
    }
  });
}

/** Fail with the command's own message (e.g. "Permission denied") when it did not succeed. */
async function asAccount(account: SandboxAccount, argv: string[], what: string, input?: string | Uint8Array): Promise<Buffer> {
  const result = await runAsAccount(account, argv, input);
  if (result.code !== 0) throw new Error(result.stderr || `${what} failed`);
  return result.stdout;
}

/**
 * Every file operation the model's tools perform runs AS the account, so the kernel decides what
 * the model may read or change. The runner itself is root and holds the gateway key in the private
 * session directory: a tool that read or wrote with the runner's own privileges could leak the key
 * or rewrite the runner. A permission check before a root-side read would race a symlink swap.
 */
const readAs = (account: SandboxAccount, path: string) => asAccount(account, ["cat", "--", path], `reading ${path}`);
const writeAs = (account: SandboxAccount, path: string, content: string) =>
  asAccount(account, ["sh", "-c", 'cat > "$1"', "sh", path], `writing ${path}`, content).then(() => undefined);
const mkdirAs = (account: SandboxAccount, dir: string) => asAccount(account, ["mkdir", "-p", "--", dir], `creating ${dir}`).then(() => undefined);

/** Reads through the account. Image detection sniffs the header the account can read. */
export function sandboxReadOperations(account: SandboxAccount, scratchDir: string): ReadOperations {
  return {
    readFile: (path) => readAs(account, path),
    access: (path) => asAccount(account, ["test", "-r", path], `${path} is not readable`).then(() => undefined),
    detectImageMimeType: async (path) => {
      const header = await asAccount(account, ["head", "-c", "4100", "--", path], `reading ${path}`);
      // Pi's detector reads a file; hand it the header in a private copy of our own.
      const dir = await mkdtemp(join(scratchDir, "sniff-"));
      try {
        await writeFile(join(dir, "header"), header, { mode: 0o600 });
        return await detectSupportedImageMimeTypeFromFile(join(dir, "header"));
      } finally {
        await rm(dir, { recursive: true, force: true });
      }
    }
  };
}

/** File writes made by the account, so it owns what it creates and cannot write what it may not. */
export function sandboxWriteOperations(account: SandboxAccount): WriteOperations {
  return {
    mkdir: (dir) => mkdirAs(account, dir),
    writeFile: async (path, content) => {
      await mkdirAs(account, dirname(path));
      await writeAs(account, path, content);
    }
  };
}

/** Edits made by the account. */
export function sandboxEditOperations(account: SandboxAccount): EditOperations {
  return {
    readFile: (path) => readAs(account, path),
    access: (path) => asAccount(account, ["sh", "-c", 'test -r "$1" && test -w "$1"', "sh", path], `${path} is not readable and writable`).then(() => undefined),
    writeFile: (path, content) => writeAs(account, path, content)
  };
}
