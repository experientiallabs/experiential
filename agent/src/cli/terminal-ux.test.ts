/**
 * The terminal UI fixes of pi-ux-patches.ts and early-input.ts end to end: the built CLI in a real
 * pseudo-terminal, driven by raw keystrokes, against the local fake gateway.
 */

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { type FakeGateway, startFakeGateway } from "../testing/fake-gateway.js";
import { findPython, pause, Terminal } from "../testing/pty.js";

const packageDir = fileURLToPath(new URL("../..", import.meta.url));
// A build of its own, so this file never races the other end-to-end builds.
const outDir = join(packageDir, ".e2e-dist-ux");
const python = findPython();

let home: string;
let gateway: FakeGateway;

beforeAll(async () => {
  if (python === undefined) return;
  execFileSync(process.execPath, [join(packageDir, "node_modules", "typescript", "bin", "tsc"), "-p", join(packageDir, "tsconfig.build.json"), "--outDir", outDir], { stdio: "inherit" });
  home = mkdtempSync(join(tmpdir(), "exp-code-ux-e2e-"));
  gateway = await startFakeGateway();
}, 120_000);

afterAll(async () => {
  if (python === undefined) return;
  await gateway.close();
  rmSync(home, { recursive: true, force: true });
  rmSync(outDir, { recursive: true, force: true });
});

/** The user messages of every chat request the gateway received. */
function userMessages(): string[] {
  return gateway.completions().map((body) => {
    const user = ((body.messages as { role: string; content: unknown }[] | undefined) ?? []).filter((message) => message.role === "user").at(-1);
    if (user === undefined) return "";
    return typeof user.content === "string" ? user.content : (user.content as { text?: string }[]).map((part) => part.text ?? "").join("");
  });
}

/** A fresh exp-code home and project folder, optionally with settings. */
function start(settings?: unknown): { terminal: Terminal; project: string } {
  const root = mkdtempSync(join(home, "run-"));
  const project = mkdtempSync(join(root, "project-"));
  if (settings !== undefined) {
    mkdirSync(join(root, ".exp-code", "agent"), { recursive: true });
    writeFileSync(join(root, ".exp-code", "agent", "settings.json"), JSON.stringify(settings));
  }
  const env = { PATH: process.env.PATH, HOME: root, EXP_CODE_HOME: join(root, ".exp-code"), EXP_GATEWAY_KEY: "k", EXP_CODE_BASE_URL: gateway.baseUrl, TERM: "xterm-256color", PI_OFFLINE: "1" };
  return { terminal: new Terminal(join(outDir, "cli.js"), env, project), project };
}

const model = (id: string) => ({ id, object: "model", created: 0, owned_by: "exp" });

/** The screen after `from`, as one string. */
const after = (terminal: Terminal, from: number) => terminal.screen().slice(from);

describe.skipIf(python === undefined)("the terminal UI", () => {
  it("runs commands typed before the screen is ready, key for key", async () => {
    gateway.setModels([model("fake-model")]);
    gateway.script([{ text: "hello back" }]);
    const { terminal } = start();
    try {
      // Typed at once, while exp-code is still loading (the terminal is not in raw mode yet).
      terminal.send("/name early\r");
      await terminal.type("hi\r");
      await terminal.waitFor(() => userMessages().length === 1, "the message");
      await terminal.waitForText("hello back");
      await pause(300);
      // `/name early` ran as a command; only `hi` reached the model.
      expect(userMessages()).toEqual(["hi"]);
      await terminal.type("/session\r");
      await terminal.waitForText("Name: early");
    } finally {
      terminal.kill();
    }
  }, 90_000);

  it("keeps a reply that dropped mid-stream, and waits the Retry-After a failure before any content asks", async () => {
    gateway.setModels([model("fake-model")]);
    gateway.script([
      { text: "The first half of the answer", dropAfterContent: true },
      { status: 429, error: "Rate limit reached.", headers: { "retry-after": "2" } },
      { text: "the rest of it" }
    ]);
    const before = gateway.completions().length;
    const { terminal } = start();
    try {
      await terminal.waitForText("fake-model");
      await terminal.type("hi\r");
      let seen = await terminal.waitForText("The reply was interrupted after it started.");
      expect(terminal.screen()).toContain("The first half of the answer");
      await pause(500);
      expect(gateway.completions()).toHaveLength(before + 1);
      await terminal.type("continue\r");
      seen = await terminal.waitForText("Retrying (1/3) in 2s", seen);
      await terminal.waitForText("the rest of it", seen);
      expect(gateway.completions()).toHaveLength(before + 3);
      expect(JSON.stringify(gateway.completions().at(-1)!.messages)).toContain("The first half of the answer");
    } finally {
      terminal.kill();
    }
  }, 90_000);

  it("never applies a stale suggestion, runs an exact /model on one Enter, and checks /export names", async () => {
    gateway.setModels([model("fake-model"), model("fake-model-b"), model("fake-model-bb")]);
    const { terminal, project } = start();
    try {
      await terminal.waitForText("fake-model");
      await pause(1500);
      // `/` opens the command list (`settings` first); the rest arrives in one burst, before the
      // list can follow it. Enter used to apply `settings` to the end: `out1.jsonsettings`.
      terminal.send("/");
      await pause(400);
      terminal.send("export out1.jsonl\r");
      await terminal.waitFor(() => existsSync(join(project, "out1.jsonl")), "out1.jsonl");
      terminal.send("/");
      await pause(400);
      let seen = terminal.screen().length;
      terminal.send("session\r");
      await terminal.waitForText("Cache Warming", seen);
      expect(after(terminal, seen)).not.toContain("sessiosettings");

      // A model id typed exactly runs on the first Enter.
      seen = terminal.screen().length;
      await terminal.type("/model fake-model-b");
      await pause(400);
      terminal.send("\r");
      await terminal.waitForText("Model: fake-model-b", seen);

      seen = terminal.screen().length;
      await terminal.type("/export notes.txt");
      await pause(300);
      terminal.send("\r");
      await terminal.waitForText("Error: /export writes .html", seen);
      expect(existsSync(join(project, "notes.txt"))).toBe(false);

      seen = terminal.screen().length;
      await terminal.type("/hotkeys");
      await pause(300);
      terminal.send("\r");
      await terminal.waitForText("Stash the prompt", seen);
      expect(after(terminal, seen)).toContain("Ctrl+S");
    } finally {
      terminal.kill();
    }
  }, 90_000);

  it("cycles Ctrl+P through the Experiential models and estimates the context after a compaction", async () => {
    gateway.setModels([model("aaa"), model("experiential-large-model"), model("zzz"), model("experiential-small-model"), model("experiential-medium-model")]);
    const summary = { text: "## Goal\nThe user said hi and then more, and the assistant answered both. Nothing else happened in this conversation." };
    gateway.script([{ text: "ok there, a longer answer." }, { text: "fine" }, summary, summary, summary, summary]);
    const { terminal } = start({ compaction: { keepRecentTokens: 1 } });
    try {
      await terminal.waitForText("experiential-large-model");
      await pause(1500);
      const switched: string[] = [];
      for (let i = 0; i < 3; i++) {
        const seen = terminal.screen().length;
        terminal.send("\x10");
        await terminal.waitForText("Switched to ", seen);
        await pause(300);
        switched.push(/Switched to ([\w-]+)/.exec(after(terminal, seen))![1]!);
      }
      expect(switched).toEqual(["experiential-small-model", "experiential-medium-model", "experiential-large-model"]);

      await terminal.type("hi\r");
      await terminal.waitFor(() => userMessages().at(-1) === "hi", "hi");
      await terminal.waitForText("ok there");
      await pause(300);
      await terminal.type("more\r");
      await terminal.waitForText("fine");
      await pause(300);
      const seen = terminal.screen().length;
      await terminal.type("/compact");
      await pause(300);
      terminal.send("\r");
      await terminal.waitForText("ompacted from", seen);
      await pause(500);
      // Pi showed "?" for the context until the next reply.
      const footer = after(terminal, terminal.screen().indexOf("ompacted from", seen));
      expect(footer).toMatch(/\d+\.\d%\/128k/);
      expect(footer).not.toContain("?/128k");
    } finally {
      terminal.kill();
    }
  }, 90_000);
});
