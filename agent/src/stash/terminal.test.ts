/**
 * The prompt stash end to end: the built exp-code CLI in a real pseudo-terminal (through Python's
 * standard `pty` module, so no native Node addon is needed), driven by raw keystrokes, against the
 * local fake gateway. What reached the gateway proves what the prompt held.
 *
 * The pseudo-terminal starts with XON/XOFF flow control on (IXON), like a fresh terminal, so a
 * stash that works here also proves Ctrl+S reaches exp-code instead of freezing the output.
 */

import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { type FakeGateway, startFakeGateway } from "../testing/fake-gateway.js";
import { findPython, pause, Terminal } from "../testing/pty.js";

const packageDir = fileURLToPath(new URL("../..", import.meta.url));
// A build of its own, beside dist/, so this file never races the CLI tests' build.
const outDir = join(packageDir, ".e2e-dist");
const python = findPython();

let home: string;
let gateway: FakeGateway;

beforeAll(async () => {
  if (python === undefined) return;
  execFileSync(process.execPath, [join(packageDir, "node_modules", "typescript", "bin", "tsc"), "-p", join(packageDir, "tsconfig.build.json"), "--outDir", outDir], { stdio: "inherit" });
  home = mkdtempSync(join(tmpdir(), "exp-code-stash-e2e-"));
  gateway = await startFakeGateway();
}, 120_000);

afterAll(async () => {
  if (python === undefined) return;
  await gateway.close();
  rmSync(home, { recursive: true, force: true });
  rmSync(outDir, { recursive: true, force: true });
});

/** The last user message the gateway received. */
function lastUserMessage(): string | undefined {
  const body = gateway.completions().at(-1);
  const messages = (body?.messages as { role: string; content: unknown }[] | undefined) ?? [];
  const user = messages.filter((message) => message.role === "user").at(-1);
  if (user === undefined) return undefined;
  return typeof user.content === "string" ? user.content : (user.content as { text?: string }[]).map((part) => part.text ?? "").join("");
}

describe.skipIf(python === undefined)("prompt stash in a real terminal", () => {
  it("stashes, sends, restores, double-saves, and survives exit and restart", async () => {
    const project = mkdtempSync(join(home, "project-"));
    const env = { PATH: process.env.PATH, HOME: home, EXP_CODE_HOME: join(home, ".exp-code"), EXP_GATEWAY_KEY: "k", EXP_CODE_BASE_URL: gateway.baseUrl, TERM: "xterm-256color", PI_OFFLINE: "1" };
    const stashDir = join(home, ".exp-code", "agent", "prompt-stash");

    const first = new Terminal(join(outDir, "cli.js"), env, project);
    try {
      await first.waitForText("fake-model");
      expect(first.log).toContain("IXON-AT-START 1");
      first.send("\x00IXON?");
      await first.waitFor(() => first.log.includes("IXON-NOW"), "IXON report");
      // Raw mode turned flow control off while exp-code runs.
      expect(first.log).toContain("IXON-NOW 0");

      await first.type("stashed draft  ");
      first.send("\x13");
      let seen = await first.waitForText("Prompt stashed.");
      await first.type("send this first");
      first.send("\r");
      await first.waitFor(() => lastUserMessage() === "send this first", "the first message");
      seen = await first.waitForText("Stashed prompt restored.", seen);

      // Double save: stash the restored draft and a second one, both kept.
      first.send("\x13");
      await first.type("second draft");
      first.send("\x13");
      seen = await first.waitForText("(2 stashed)", seen);
      first.send("\x13"); // newest back
      seen = await first.waitForText("1 more stashed", seen);
      first.send("\r");
      await first.waitFor(() => lastUserMessage() === "second draft", "the second draft sent");
      seen = await first.waitForText("Stashed prompt restored.", seen);
      // "stashed draft  " is back in the prompt now; stash it and exit with Ctrl+D.
      first.send("\x13");
      await first.waitForText("Prompt stashed.", seen);
      await pause(200);
      first.send("\x04");
      expect(await first.exited).toBe(0);
    } finally {
      first.kill();
    }
    expect(readdirSync(stashDir).filter((name) => name.endsWith(".json"))).toHaveLength(1);

    const second = new Terminal(join(outDir, "cli.js"), env, project);
    try {
      await second.waitForText("Restored the prompt you stashed in an earlier exp-code session.");
      expect(readdirSync(stashDir).filter((name) => name.endsWith(".json"))).toHaveLength(0);
      second.send("\r");
      // Pi trims a message before sending it.
      await second.waitFor(() => lastUserMessage() === "stashed draft", "the restored draft sent");
      await pause(200);
      second.send("\x04");
      expect(await second.exited).toBe(0);
    } finally {
      second.kill();
    }
    expect(existsSync(stashDir) ? readdirSync(stashDir).filter((name) => name.endsWith(".json")) : []).toEqual([]);
  }, 120_000);
});
