/** The working directory listing reported in `files` events. */

import { readdir, stat } from "node:fs/promises";
import { join, relative } from "node:path";
import type { AgentFile } from "./protocol.js";

export const MAX_LISTED_FILES = 500;
const MAX_DEPTH = 8;
const SKIPPED = new Set(["node_modules", "__pycache__"]);

/** Files under the working directory, skipping dot folders, node_modules, and caches. */
export async function listWorkFiles(workDir: string): Promise<AgentFile[]> {
  const files: AgentFile[] = [];
  async function walk(directory: string, depth: number): Promise<void> {
    if (depth > MAX_DEPTH || files.length >= MAX_LISTED_FILES) return;
    let entries;
    try {
      entries = await readdir(directory, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (entry.name.startsWith(".") || SKIPPED.has(entry.name)) continue;
      const path = join(directory, entry.name);
      if (entry.isDirectory()) await walk(path, depth + 1);
      else if (entry.isFile()) {
        try {
          const info = await stat(path);
          files.push({ path: relative(workDir, path), size: info.size });
        } catch {
          continue;
        }
        if (files.length >= MAX_LISTED_FILES) return;
      }
    }
  }
  await walk(workDir, 0);
  return files.sort((a, b) => a.path.localeCompare(b.path));
}
