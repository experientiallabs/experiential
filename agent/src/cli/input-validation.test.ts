/**
 * What exp-code refuses to send: a start model the gateway does not list, a missing or blank
 * prompt, stdin that is not text, and a prompt larger than the model's input budget. The built CLI
 * against the local fake gateway, and once in a pseudo-terminal.
 */

import { execFile, execFileSync } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { type FakeGateway, startFakeGateway } from "../testing/fake-gateway.js";
import { findPython, pause, Terminal } from "../testing/pty.js";

const packageDir = fileURLToPath(new URL("../..", import.meta.url));
// A build of its own, so this file never races the other end-to-end builds.
const outDir = join(packageDir, ".e2e-dist-input");
const cli = join(outDir, "cli.js");
const python = findPython();

let home: string;
let gateway: FakeGateway;

function runCli(args: string[], env: NodeJS.ProcessEnv = {}, input?: Buffer | string): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const child = execFile(
      process.execPath,
      [cli, ...args],
      {
        cwd: home,
        env: { PATH: process.env.PATH, HOME: home, EXP_CODE_HOME: join(home, ".exp-code"), EXP_GATEWAY_KEY: "local-key", EXP_CODE_BASE_URL: gateway.baseUrl, ...env },
        timeout: 60_000
      },
      (error, stdout, stderr) => resolve({ code: error ? (typeof error.code === "number" ? error.code : 1) : 0, stdout, stderr })
    );
    child.stdin?.end(input);
  });
}

beforeAll(async () => {
  execFileSync(process.execPath, [join(packageDir, "node_modules", "typescript", "bin", "tsc"), "-p", join(packageDir, "tsconfig.build.json"), "--outDir", outDir], { stdio: "inherit" });
  home = await mkdtemp(join(tmpdir(), "exp-code-input-"));
  gateway = await startFakeGateway([], [
    { id: "fake-model", object: "model", created: 0, owned_by: "exp" },
    { id: "small-window", object: "model", created: 0, owned_by: "exp", context_window_tokens: 20_000, maximum_output_tokens: 4_000 }
  ]);
}, 120_000);

afterAll(async () => {
  await gateway.close();
  await rm(home, { recursive: true, force: true });
  await rm(outDir, { recursive: true, force: true });
});

describe("print-mode input", () => {
  it("validates EXP_CODE_MODEL against the gateway listing like --model", async () => {
    const before = gateway.completions().length;
    gateway.script([{ text: "answered" }]);
    const result = await runCli(["-p", "hi"], { EXP_CODE_MODEL: "no-such-model" });
    expect(result.stderr).toContain("EXP_CODE_MODEL");
    expect(result.stderr).toContain("no-such-model");
    // It falls back to the default model with a notice, as a saved model the gateway no longer lists does.
    expect(result.code).toBe(0);
    expect(result.stdout).toContain("answered");
    const sent = gateway.completions().slice(before);
    expect(sent).toHaveLength(1);
    expect(sent[0]?.model).toBe("fake-model");
  });

  it("refuses -p with no prompt, an empty or a whitespace-only prompt", async () => {
    const before = gateway.completions().length;
    for (const [args, reason] of [
      [["-p"], "no prompt was given"],
      [["--mode", "json"], "no prompt was given"],
      [["-p", ""], "the prompt is empty"],
      [["-p", "   \n\t"], "the prompt is empty"],
      [["-p", "first", " "], "the prompt is empty"]
    ] as const) {
      const result = await runCli([...args]);
      expect(result.code, args.join(" ")).toBe(2);
      expect(result.stderr).toContain(reason);
      expect(result.stderr).toContain("Usage");
    }
    expect(gateway.completions()).toHaveLength(before);
  });

  it("refuses non-text stdin", async () => {
    const before = gateway.completions().length;
    const result = await runCli(["-p", "look at this"], {}, Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0x0d, 0xff, 0xfe]));
    expect(result.code).toBe(2);
    expect(result.stderr).toContain("stdin is not text");
    expect(gateway.completions()).toHaveLength(before);
    // Text, including non-ASCII text, is still sent.
    gateway.script([{ text: "read it" }]);
    const text = await runCli(["-p", "look at this"], {}, "café, naïve, 日本語\n");
    expect(text.code).toBe(0);
    expect(JSON.stringify(gateway.completions().at(-1))).toContain("日本語");
  });

  it("still sends a prompt that fits the model's input budget", async () => {
    gateway.script([{ text: "fits" }]);
    const result = await runCli(["--model", "small-window", "-p", "summarize"], {}, "word ".repeat(10_000));
    expect(result.code).toBe(0);
    expect(result.stdout).toContain("fits");
  });

  it("refuses a prompt larger than the model's input budget", async () => {
    const before = gateway.completions().length;
    const result = await runCli(["--model", "small-window", "-p", "summarize"], {}, "word ".repeat(200_000));
    expect(result.code).toBe(1);
    expect(result.stderr).toContain("This prompt is too large for small-window: about 250,002 tokens");
    expect(result.stderr).toContain("at most 16,000 tokens of input");
    expect(gateway.completions()).toHaveLength(before);
  });
});

describe.skipIf(python === undefined)("the terminal UI", () => {
  it("refuses a paste larger than the model's input budget, and keeps working", async () => {
    const root = mkdtempSync(join(home, "pty-"));
    const env = { PATH: process.env.PATH, HOME: root, EXP_CODE_HOME: join(root, ".exp-code"), EXP_GATEWAY_KEY: "k", EXP_CODE_BASE_URL: gateway.baseUrl, TERM: "xterm-256color", PI_OFFLINE: "1" };
    const terminal = new Terminal(cli, env, root, ["--model", "small-window"]);
    const before = gateway.completions().length;
    try {
      await terminal.waitForText("small-window");
      await pause(500);
      // A bracketed paste of about 20,000 tokens, then Enter.
      terminal.send(`\x1b[200~${"word ".repeat(16_000)}\x1b[201~`);
      await pause(500);
      terminal.send("\r");
      await terminal.waitForText("This prompt is too large for small-window");
      expect(gateway.completions()).toHaveLength(before);
      gateway.script([{ text: "small answer" }]);
      await terminal.type("hi\r");
      await terminal.waitForText("small answer");
      expect(gateway.completions()).toHaveLength(before + 1);
    } finally {
      terminal.kill();
    }
  }, 60_000);
});
