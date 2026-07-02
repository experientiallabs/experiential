#!/usr/bin/env node
/**
 * Generate src/data/index.json — the site's local "db" of world models.
 *
 * Walks every model dir under examples/<task>/models/ and .wmh/models/ in the repo root,
 * reading card.json (the record the gallery renders) and metrics.json (build accuracy).
 * Models without a card are listed with a minimal synthesized card so the gallery never
 * hides a built model. Run whenever cards change: `npm run index`.
 */

import { readdirSync, readFileSync, writeFileSync, existsSync, statSync } from "node:fs";
import { join, dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");
const outPath = join(here, "..", "src", "data", "index.json");

function modelRoots() {
  // .wmh (the writable build root) comes FIRST so serveCommand() lists it first — that is where
  // `wmh serve` writes server-side builds and uploads; putting a committed examples/ dir first
  // would send build artifacts into the git tree.
  const roots = [];
  const local = join(repoRoot, ".wmh", "models");
  if (existsSync(local)) roots.push({ serveRoot: ".wmh", modelsDir: local });
  const examples = join(repoRoot, "examples");
  if (existsSync(examples)) {
    for (const task of readdirSync(examples).sort()) {
      const dir = join(examples, task, "models");
      if (existsSync(dir)) roots.push({ serveRoot: join("examples", task), modelsDir: dir });
    }
  }
  return roots;
}

function readJson(path) {
  try {
    return JSON.parse(readFileSync(path, "utf-8"));
  } catch {
    return null;
  }
}

function countLines(path) {
  if (!existsSync(path)) return 0;
  const text = readFileSync(path, "utf-8");
  return text.split("\n").filter(Boolean).length;
}

function clip(text, max) {
  const flat = String(text).replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat;
}

/**
 * The card's "screenshot": a couple of real (action -> observation) steps from the model's own
 * replay index, rendered as a mini terminal in the gallery tile. Standardized across every
 * model — no per-model artwork.
 */
function samplePreview(dir) {
  const stepsPath = join(dir, "index", "steps.jsonl");
  if (!existsSync(stepsPath)) return [];
  const lines = readFileSync(stepsPath, "utf-8").split("\n").filter(Boolean);
  const preview = [];
  for (const line of lines) {
    let step;
    try {
      step = JSON.parse(line);
    } catch {
      continue;
    }
    const action =
      step.action?.kind === "tool_call"
        ? `${step.action.name} ${Object.keys(step.action.arguments ?? {}).length ? JSON.stringify(step.action.arguments) : ""}`
        : step.action?.content;
    const observation = step.observation?.content;
    if (!action || !observation) continue;
    preview.push({ action: clip(action, 76), observation: clip(observation, 110) });
    if (preview.length === 2) break;
  }
  return preview;
}

const entries = [];
const seen = new Set();
for (const { serveRoot, modelsDir } of modelRoots()) {
  for (const name of readdirSync(modelsDir).sort()) {
    const dir = join(modelsDir, name);
    if (!statSync(dir).isDirectory() || !existsSync(join(dir, "config.toml"))) continue;
    // A name can exist under two roots; the server refuses to serve that ambiguity, so the
    // gallery must not list it twice (duplicate React keys / static params). First root wins.
    if (seen.has(name)) {
      console.warn(`skipping duplicate model name '${name}' under ${serveRoot} (already indexed)`);
      continue;
    }
    seen.add(name);
    let card = readJson(join(dir, "card.json"));
    if (!card) {
      // Cardless model: synthesize the minimum the gallery needs, honestly labeled.
      card = {
        schema_version: 1,
        name,
        title: name,
        description: "",
        task: null,
        corpus: { traces: null, steps: countLines(join(dir, "index", "steps.jsonl")) },
        provider: "unknown",
        model_id: "unknown",
        tags: [],
      };
    }
    const metrics = readJson(join(dir, "metrics.json"));
    entries.push({
      card,
      dir: relative(repoRoot, dir),
      held_out_accuracy: typeof metrics?.held_out_accuracy === "number" ? metrics.held_out_accuracy : null,
      serve_root: serveRoot,
      preview: samplePreview(dir),
    });
  }
}

entries.sort((a, b) => a.card.name.localeCompare(b.card.name));
const index = { generated_at: new Date().toISOString(), models: entries };
writeFileSync(outPath, JSON.stringify(index, null, 2) + "\n");
console.log(`wrote ${relative(process.cwd(), outPath)} (${entries.length} models)`);
