/**
 * The key `exp-code login` saved, in `<exp-code home>/agent/credentials.json`.
 *
 * The directory is created owner-only (0700) and the file is written owner-only (0600) through a
 * temporary file and a rename, so a crash never leaves a half-written key and no other local user
 * can read it. Environment variables always win over the saved key (see gateway/config.ts).
 */

import { randomBytes } from "node:crypto";
import { chmod, mkdir, open, readFile, rename, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { expCodeHome } from "../cli/environment.js";

/** What a login saves. */
export interface SavedCredentials {
  /** The hosted gateway key (`xpl_...`). */
  hostedKey: string;
  /** The gateway root the key was verified against, when it was not the hosted default. */
  baseUrl?: string;
  /** ISO timestamp of the login. */
  savedAt: string;
  /** The key's name on the platform. */
  keyName: string;
}

/** Where the saved login lives. */
export function credentialsPath(env: NodeJS.ProcessEnv): string {
  return join(expCodeHome(env), "agent", "credentials.json");
}

/** Validate a parsed credentials document; anything malformed reads as no saved login. */
export function parseCredentials(value: unknown): SavedCredentials | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const { hostedKey, baseUrl, savedAt, keyName } = record;
  if (typeof hostedKey !== "string" || !hostedKey.startsWith("xpl_")) return null;
  if (baseUrl !== undefined && typeof baseUrl !== "string") return null;
  return {
    hostedKey,
    ...(typeof baseUrl === "string" && baseUrl.trim() !== "" ? { baseUrl } : {}),
    savedAt: typeof savedAt === "string" ? savedAt : "",
    keyName: typeof keyName === "string" ? keyName : ""
  };
}

/** Read the saved login, or `null` when there is none (or it cannot be read). */
export async function readCredentials(env: NodeJS.ProcessEnv): Promise<SavedCredentials | null> {
  let text: string;
  try {
    text = await readFile(credentialsPath(env), "utf8");
  } catch {
    return null;
  }
  try {
    return parseCredentials(JSON.parse(text));
  } catch {
    return null;
  }
}

/** Save a login atomically with owner-only permissions. Returns the file path. */
export async function saveCredentials(env: NodeJS.ProcessEnv, credentials: SavedCredentials): Promise<string> {
  const path = credentialsPath(env);
  const dir = dirname(path);
  await mkdir(dir, { recursive: true, mode: 0o700 });
  // mkdir leaves an existing directory's mode alone; the key directory is always owner-only.
  await chmod(dir, 0o700);
  const temporary = join(dir, `.credentials.${process.pid}.${randomBytes(6).toString("hex")}.tmp`);
  const handle = await open(temporary, "wx", 0o600);
  try {
    await handle.writeFile(`${JSON.stringify(credentials, null, 2)}\n`, "utf8");
    await handle.sync();
  } catch (error) {
    await handle.close();
    await rm(temporary, { force: true });
    throw error;
  }
  await handle.close();
  try {
    await rename(temporary, path);
  } catch (error) {
    await rm(temporary, { force: true });
    throw error;
  }
  await chmod(path, 0o600);
  return path;
}

/** Delete the saved login. Returns whether there was one. */
export async function deleteCredentials(env: NodeJS.ProcessEnv): Promise<boolean> {
  const path = credentialsPath(env);
  try {
    await rm(path);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

/** Show a key without revealing it: `xpl_` and its next seven characters. */
export function maskKey(key: string): string {
  const visible = key.startsWith("xpl_") ? key.slice(0, 11) : key.slice(0, 7);
  return `${visible}…`;
}
