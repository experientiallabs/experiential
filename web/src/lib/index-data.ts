/**
 * Loader over the generated local db (src/data/index.json). Static import so the gallery and
 * model pages render with no backend; regenerate with `npm run index`.
 */

import rawIndex from "@/data/index.json" assert { type: "json" };
import type { IndexEntry, SiteIndex } from "./types";

const index = rawIndex as SiteIndex;

export function allModels(): IndexEntry[] {
  return index.models;
}

export function findModel(name: string): IndexEntry | undefined {
  return index.models.find((m) => m.card.name === name);
}

/** The exact serve command that exposes every indexed model — shown when the API is down. */
export function serveCommand(): string {
  const roots = [...new Set(index.models.map((m) => m.serve_root))];
  return `uv run wmh serve ${roots.map((r) => `--root ${r}`).join(" ")}`;
}
