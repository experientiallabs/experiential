/**
 * The prompt editor's autocomplete as patched (pi-ux-patches.ts EDITOR_PATCH): Pi's real editor and
 * autocomplete provider in a real TUI, driven by keystrokes through the terminal input path.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CombinedAutocompleteProvider } from "@earendil-works/pi-tui";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { StashHarness } from "../testing/stash-harness.js";

let dir: string;
let harness: StashHarness;

const settle = () => new Promise((resolve) => setTimeout(resolve, 20));

const models = [
  { value: "experiential/experiential-small-model", label: "experiential-small-model", description: "experiential" },
  { value: "experiential/experiential-small-model-2", label: "experiential-small-model-2", description: "experiential" }
];

beforeEach(async () => {
  dir = mkdtempSync(join(tmpdir(), "exp-code-editor-"));
  harness = await StashHarness.start({ agentDir: dir, cwd: dir });
  const commands = [
    { name: "settings", description: "Open settings" },
    { name: "export", description: "Export session" },
    { name: "session", description: "Session info" },
    {
      name: "model",
      description: "Select model",
      getArgumentCompletions: (prefix: string) => models.filter((model) => model.label.startsWith(prefix))
    }
  ];
  harness.editor.setAutocompleteProvider(new CombinedAutocompleteProvider(commands, dir));
});

afterEach(async () => {
  await harness.shutdown();
  rmSync(dir, { recursive: true, force: true });
});

/** Type one key at a time and let each key's suggestions arrive. */
async function typeSlowly(text: string): Promise<void> {
  for (const char of text) {
    harness.key(char);
    await settle();
  }
}

describe("autocomplete in the prompt", () => {
  it("submits a line as typed when the list on screen was computed for earlier text", async () => {
    // `/` opens the list with `settings` first; the rest arrives before any new list does.
    harness.key("/");
    await settle();
    harness.type("export out1.jsonl");
    await harness.submit();
    harness.key("/");
    await settle();
    harness.type("session");
    await harness.submit();
    expect(harness.sent).toEqual(["/export out1.jsonl", "/session"]);
  });

  it("still completes a command from a current list", async () => {
    await typeSlowly("/sess");
    await harness.submit();
    expect(harness.sent).toEqual(["/session"]);
  });

  it("asks again on Tab instead of applying a stale suggestion", async () => {
    harness.key("/");
    await settle();
    harness.type("expo");
    harness.key("\t");
    await settle();
    expect(harness.text()).toBe("/expo");
    harness.key("\t");
    await settle();
    expect(harness.text()).toBe("/export ");
  });

  it("runs an exactly typed argument on the first Enter", async () => {
    await typeSlowly("/model experiential-small-model");
    await harness.submit();
    expect(harness.sent).toEqual(["/model experiential/experiential-small-model"]);
  });

  it("completes a partly typed argument first", async () => {
    await typeSlowly("/model experiential-sm");
    await harness.submit();
    expect(harness.sent).toEqual([]);
    expect(harness.text()).toBe("/model experiential/experiential-small-model");
  });
});
