#!/usr/bin/env node
/**
 * Generate src/data/index.json - the site's local "db" of world models.
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
  // .wmh (the writable build root) comes FIRST so serveCommand() lists it first - that is where
  // `wmh serve` writes server-side builds and uploads; putting a committed corpus dir first
  // would send build artifacts into the git tree.
  const roots = [];
  const local = join(repoRoot, ".wmh", "models");
  if (existsSync(local)) roots.push({ serveRoot: ".wmh", modelsDir: local });
  // Bundled corpora live under packages/environment-capture/<task>/models (WS-B2 layout).
  const captures = join(repoRoot, "packages", "environment-capture");
  if (existsSync(captures)) {
    for (const task of readdirSync(captures).sort()) {
      const dir = join(captures, task, "models");
      if (existsSync(dir)) {
        roots.push({ serveRoot: join("packages", "environment-capture", task), modelsDir: dir });
      }
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

/** Format an action in the wmh-play grammar: `name {json}` / `name` / `say <msg>`. */
function formatAction(action) {
  if (!action) return null;
  if (action.kind === "tool_call") {
    const args = action.arguments ?? {};
    return Object.keys(args).length ? `${action.name} ${JSON.stringify(args)}` : action.name;
  }
  return action.content ? `say ${action.content}` : null;
}

/** A short, readable label for a recorded task (tau tasks are JSON; terminal/swe are plain text). */
function scenarioLabel(task, fallbackIndex) {
  if (!task) return `Scenario ${fallbackIndex + 1}`;
  try {
    const parsed = JSON.parse(task);
    const text = parsed.reason_for_call || parsed.task_instructions || parsed.query || task;
    return clip(text, 68);
  } catch {
    return clip(task, 68);
  }
}

/** Read all valid steps from a model's replay index once. */
function readSteps(dir) {
  const stepsPath = join(dir, "index", "steps.jsonl");
  if (!existsSync(stepsPath)) return [];
  const steps = [];
  for (const line of readFileSync(stepsPath, "utf-8").split("\n")) {
    if (!line) continue;
    try {
      steps.push(JSON.parse(line));
    } catch {
      // skip malformed line
    }
  }
  return steps;
}

/** The card's "screenshot": a couple of real (action -> observation) steps as a mini terminal. */
function samplePreview(steps) {
  const preview = [];
  for (const step of steps) {
    const action = formatAction(step.action);
    const observation = step.observation?.content;
    if (!action || !observation) continue;
    preview.push({ action: clip(action, 76), observation: clip(observation, 110) });
    if (preview.length === 2) break;
  }
  return preview;
}

/** A few distinct example actions in play grammar, for suggestion chips + the default input. */
function suggestions(steps, max = 5) {
  const out = [];
  const seenActions = new Set();
  for (const step of steps) {
    const label = formatAction(step.action);
    if (!label || seenActions.has(label) || label.length > 120) continue;
    seenActions.add(label);
    out.push(label);
    if (out.length === max) break;
  }
  return out;
}

/**
 * Recorded traces grouped by task = replayable scenarios. Bounded so index.json stays small:
 * at most `maxScenarios` tasks, each capped at `maxSteps` steps. Each scenario can be replayed
 * open-loop (seed a session with the task, feed the recorded actions, compare observations).
 */
function scenarios(steps, maxScenarios = 6, maxSteps = 10) {
  const byTask = new Map();
  for (const step of steps) {
    const task = step.task ?? null;
    const key = task ?? "__none__";
    if (!byTask.has(key)) byTask.set(key, { task, steps: [] });
    const group = byTask.get(key);
    if (group.steps.length >= maxSteps) continue;
    const label = formatAction(step.action);
    if (!label || step.observation?.content == null) continue;
    group.steps.push({
      action: step.action,
      action_label: clip(label, 100),
      observation: step.observation.content,
      is_error: Boolean(step.observation.is_error),
    });
  }
  const out = [];
  let i = 0;
  for (const group of byTask.values()) {
    if (group.steps.length === 0) continue;
    out.push({
      id: `s${i}`,
      label: scenarioLabel(group.task, i),
      task: group.task,
      steps: group.steps,
    });
    i += 1;
    if (out.length === maxScenarios) break;
  }
  return out;
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
    const steps = readSteps(dir);
    entries.push({
      card,
      dir: relative(repoRoot, dir),
      held_out_accuracy: typeof metrics?.held_out_accuracy === "number" ? metrics.held_out_accuracy : null,
      serve_root: serveRoot,
      preview: samplePreview(steps),
      suggestions: suggestions(steps),
      scenarios: scenarios(steps),
    });
  }
}

entries.sort((a, b) => a.card.name.localeCompare(b.card.name));
const index = { generated_at: new Date().toISOString(), models: entries };
writeFileSync(outPath, JSON.stringify(index, null, 2) + "\n");
console.log(`wrote ${relative(process.cwd(), outPath)} (${entries.length} models)`);
