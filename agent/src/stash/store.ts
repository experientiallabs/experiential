/**
 * The prompt stash on disk: one file per exp-code process and working directory, so a stash
 * survives exit and crashes, and two exp-code instances in one folder never write the same file.
 *
 * Layout, under `<agent dir>/prompt-stash/` (directory 0700, files 0600, since prompts are private):
 *
 *     <folder key>.<host key>.<pid>.<token>.json             the live stack of one process
 *     <folder key>.<host key>.<pid>.<token>.claim-<n>.json    a file this process is adopting
 *
 * The folder key is a digest of the working directory; the file repeats the directory and is
 * ignored on a mismatch. Writes go to a temporary file that is synced and renamed over the target,
 * so a reader sees the old stack or the new one, never half of either.
 *
 * A file whose process has exited (same host, pid no longer running) is an orphan. The next
 * exp-code started in that folder adopts it: it renames the file to one of its own claim names
 * (atomic, so of two instances starting together exactly one wins it), merges the drafts into its
 * stack, writes its own file, and only then deletes the claim. A crash between those steps leaves
 * the drafts in two files, and adoption keeps one copy per draft id. A file that cannot be read or
 * parsed is renamed aside with a `.corrupt` suffix and never read again. Files from other hosts
 * (a shared home directory) are never adopted, because their pids mean nothing here.
 */

import { createHash, randomBytes } from "node:crypto";
import { closeSync, fsyncSync, mkdirSync, openSync, readdirSync, readFileSync, renameSync, unlinkSync, writeSync } from "node:fs";
import { hostname } from "node:os";
import { join, resolve } from "node:path";
import type { Cursor, StashEntry } from "./machine.js";

const FILE_VERSION = 1;
const NAME_PATTERN = /^([0-9a-f]{16})\.([0-9a-f]{8})\.(\d+)\.([0-9a-z]+)(?:\.claim-[0-9a-z]+)?\.json$/;
const TEMP_PATTERN = /^([0-9a-f]{16})\.([0-9a-f]{8})\.(\d+)\.([0-9a-z]+)\..*\.tmp$/;

export interface StashStoreOptions {
  /** The directory holding stash files (`<agent dir>/prompt-stash`). */
  dir: string;
  /** The working directory this stack belongs to. */
  cwd: string;
  /** This process's id (tests pass another). */
  pid?: number;
  /** This machine's name (tests pass another). */
  host?: string;
  /** Unique per store instance; defaults to random. */
  token?: string;
  /** Whether a process is running; defaults to a signal-0 probe. */
  isAlive?: (pid: number) => boolean;
}

/** The result of adopting orphaned stash files. */
export interface Adoption {
  entries: StashEntry[];
  /** How many orphaned files were adopted. */
  files: number;
  /** How many unreadable files were set aside. */
  corrupt: number;
}

/** The on-disk stack of one process for one working directory. */
export class StashStore {
  readonly dir: string;
  readonly cwd: string;
  readonly path: string;
  private readonly folderKey: string;
  private readonly hostKey: string;
  private readonly pid: number;
  private readonly token: string;
  private readonly isAlive: (pid: number) => boolean;
  private claims = 0;

  constructor(options: StashStoreOptions) {
    this.dir = options.dir;
    this.cwd = resolve(options.cwd);
    this.folderKey = digest(this.cwd, 16);
    this.hostKey = digest(options.host ?? hostname(), 8);
    this.pid = options.pid ?? process.pid;
    this.token = options.token ?? randomBytes(6).toString("hex");
    this.isAlive = options.isAlive ?? processIsAlive;
    this.path = join(this.dir, `${this.prefix()}.json`);
  }

  /** Write the stack, replacing the previous one atomically; an empty stack removes the file. */
  save(entries: readonly StashEntry[]): void {
    if (entries.length === 0) {
      removeIfPresent(this.path);
      return;
    }
    mkdirSync(this.dir, { recursive: true, mode: 0o700 });
    const body = `${JSON.stringify({ version: FILE_VERSION, cwd: this.cwd, pid: this.pid, entries: entries.map(persisted) })}\n`;
    const temp = join(this.dir, `${this.prefix()}.${randomBytes(4).toString("hex")}.tmp`);
    const fd = openSync(temp, "w", 0o600);
    try {
      writeSync(fd, body);
      fsyncSync(fd);
    } finally {
      closeSync(fd);
    }
    try {
      renameSync(temp, this.path);
    } catch (error) {
      removeIfPresent(temp);
      throw error;
    }
  }

  /** Read this store's own file (used after a reload in the same process). */
  load(): StashEntry[] {
    try {
      return parseStashFile(readFileSync(this.path, "utf8"), this.cwd) ?? [];
    } catch {
      return [];
    }
  }

  /**
   * Adopt the stash files of exited processes in this folder: claim each, merge its drafts with
   * `current` through `merge`, save the result as this store's file, then delete the claims.
   */
  adoptOrphans(current: readonly StashEntry[], merge: (current: readonly StashEntry[], adopted: readonly StashEntry[]) => StashEntry[]): Adoption {
    let names: string[];
    try {
      names = readdirSync(this.dir);
    } catch {
      return { entries: [...current], files: 0, corrupt: 0 };
    }
    const adopted: StashEntry[] = [];
    const claimed: string[] = [];
    let corrupt = 0;
    for (const name of names) {
      const temp = TEMP_PATTERN.exec(name);
      if (temp !== null) {
        // A temporary file of an exited process is an unfinished write; its target is intact.
        if (this.isOrphan(temp)) removeIfPresent(join(this.dir, name));
        continue;
      }
      const match = NAME_PATTERN.exec(name);
      if (match === null || !this.isOrphan(match)) continue;
      const claim = join(this.dir, `${this.prefix()}.claim-${(this.claims++).toString(36)}.json`);
      try {
        renameSync(join(this.dir, name), claim);
      } catch {
        continue; // Another instance claimed it first, or it is gone.
      }
      let entries: StashEntry[] | undefined;
      try {
        entries = parseStashFile(readFileSync(claim, "utf8"), this.cwd);
      } catch {
        entries = undefined;
      }
      if (entries === undefined) {
        corrupt++;
        try {
          renameSync(claim, join(this.dir, `${name}.corrupt`));
        } catch {
          // Leave the claim in place; it is retried (and set aside) on a later start.
        }
        continue;
      }
      adopted.push(...entries);
      claimed.push(claim);
    }
    const entries = merge(current, adopted);
    if (claimed.length > 0) {
      this.save(entries);
      for (const claim of claimed) removeIfPresent(claim);
    }
    return { entries, files: claimed.length, corrupt };
  }

  private prefix(): string {
    return `${this.folderKey}.${this.hostKey}.${this.pid}.${this.token}`;
  }

  private isOrphan(match: RegExpExecArray): boolean {
    const [, folder, host, pidText] = match;
    if (folder !== this.folderKey || host !== this.hostKey) return false;
    const pid = Number(pidText);
    // A file of this very process belongs to a store that is still running here.
    if (pid === this.pid) return false;
    return !this.isAlive(pid);
  }
}

/** The fields of an entry that go to disk (the undo history stays in memory). */
function persisted(entry: StashEntry): StashEntry {
  const saved: StashEntry = { id: entry.id, text: entry.text, stashedAt: entry.stashedAt };
  if (entry.cursor !== undefined) saved.cursor = entry.cursor;
  if (entry.collapsed !== undefined) saved.collapsed = entry.collapsed;
  return saved;
}

/** Parse a stash file, or `undefined` when it is not a valid one for `cwd`. */
export function parseStashFile(text: string, cwd: string): StashEntry[] | undefined {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return undefined;
  }
  if (!isRecord(value) || value.version !== FILE_VERSION || value.cwd !== cwd || !Array.isArray(value.entries)) return undefined;
  const entries: StashEntry[] = [];
  for (const item of value.entries) {
    const entry = parseEntry(item);
    if (entry === undefined) return undefined;
    entries.push(entry);
  }
  return entries;
}

function parseEntry(value: unknown): StashEntry | undefined {
  if (!isRecord(value)) return undefined;
  const { id, text, stashedAt, cursor, collapsed } = value;
  if (typeof id !== "string" || id.length === 0 || typeof text !== "string" || typeof stashedAt !== "number" || !Number.isFinite(stashedAt)) return undefined;
  const entry: StashEntry = { id, text, stashedAt };
  if (cursor !== undefined) {
    const parsed = parseCursor(cursor);
    if (parsed === undefined) return undefined;
    entry.cursor = parsed;
  }
  if (collapsed !== undefined) {
    if (!isRecord(collapsed) || typeof collapsed.text !== "string" || !Array.isArray(collapsed.pastes)) return undefined;
    const pastes: [number, string][] = [];
    for (const paste of collapsed.pastes) {
      if (!Array.isArray(paste) || paste.length !== 2 || !Number.isInteger(paste[0]) || typeof paste[1] !== "string") return undefined;
      pastes.push([paste[0] as number, paste[1] as string]);
    }
    entry.collapsed = { text: collapsed.text, pastes };
  }
  return entry;
}

function parseCursor(value: unknown): Cursor | undefined {
  if (!isRecord(value) || !Number.isInteger(value.line) || !Number.isInteger(value.col)) return undefined;
  const line = value.line as number;
  const col = value.col as number;
  return line >= 0 && col >= 0 ? { line, col } : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function digest(text: string, length: number): string {
  return createHash("sha256").update(text).digest("hex").slice(0, length);
}

/** Whether a process is running: signal 0 checks without sending anything. */
export function processIsAlive(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    // EPERM: it exists but belongs to someone else.
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

function removeIfPresent(path: string): void {
  try {
    unlinkSync(path);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
}
