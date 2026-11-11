/**
 * Where the model's commands and file writes happen when the runner is root.
 *
 * Every shell command runs as an unprivileged account with a fixed, minimal environment, so the
 * runner's secrets never reach the model's processes (another user's /proc/<pid>/environ is
 * unreadable, and the secrets are already deleted from the runner's own environment). Files the
 * model writes are handed to that account. The operations plug into Pi's built-in tools.
 */

import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { access, chown, constants, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { BashOperations, EditOperations, WriteOperations } from "@earendil-works/pi-coding-agent";

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

/** Create a directory chain, handing every directory it creates to the account. */
async function mkdirOwned(directory: string, account: SandboxAccount): Promise<void> {
  const created = await mkdir(directory, { recursive: true });
  if (created === undefined) return;
  const chain: string[] = [];
  for (let current = directory; current.length >= created.length; current = dirname(current)) {
    chain.unshift(current);
    if (current === created) break;
  }
  for (const entry of chain) await chown(entry, account.uid, account.gid);
}

/** File writes that leave the account owning what the model created. */
export function sandboxWriteOperations(account: SandboxAccount): WriteOperations {
  return {
    mkdir: (dir) => mkdirOwned(dir, account),
    writeFile: async (path, content) => {
      await mkdirOwned(dirname(path), account);
      await writeFile(path, content);
      await chown(path, account.uid, account.gid);
    }
  };
}

/** Edits that keep the edited file owned by the account. */
export function sandboxEditOperations(account: SandboxAccount): EditOperations {
  return {
    readFile: (path) => readFile(path),
    access: (path) => access(path, constants.R_OK | constants.W_OK),
    writeFile: async (path, content) => {
      const existed = await stat(path).then(
        () => true,
        () => false
      );
      await writeFile(path, content);
      if (!existed) await chown(path, account.uid, account.gid);
    }
  };
}
