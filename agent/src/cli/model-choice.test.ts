import { mkdirSync, writeFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseArgs } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { defaultModelId, modelArgs, wantsDefaultModel } from "./model-choice.js";

let agentDir: string;
beforeEach(async () => {
  agentDir = await mkdtemp(join(tmpdir(), "exp-code-agent-"));
});
afterEach(() => rm(agentDir, { recursive: true, force: true }));

const wants = (args: string[]) => wantsDefaultModel(args, parseArgs(args), agentDir);

describe("default model", () => {
  it("is injected for plain prompts and every mode", () => {
    expect(wants([])).toBe(true);
    expect(wants(["-p", "hi"])).toBe(true);
    expect(wants(["--mode", "rpc"])).toBe(true);
    expect(wants(["--mode", "json", "hi"])).toBe(true);
  });

  it("is left alone for explicit models, resumed sessions, subcommands and info flags", () => {
    for (const args of [["--model", "x"], ["--provider", "p", "--model", "x"], ["-c"], ["--resume"], ["--session", "f"], ["install", "pkg"], ["--help"], ["--version"], ["--list-models"]]) {
      expect(wants(args), args.join(" ")).toBe(false);
    }
  });

  it("is left to Pi when a model was saved with /model", () => {
    mkdirSync(agentDir, { recursive: true });
    writeFileSync(join(agentDir, "settings.json"), JSON.stringify({ defaultProvider: "experiential", defaultModel: "saved" }));
    expect(wants([])).toBe(false);
  });

  it("prefers EXP_CODE_MODEL, else the first listed alias", () => {
    expect(defaultModelId(["a", "b"], undefined)).toBe("a");
    expect(defaultModelId(["a", "b"], " b ")).toBe("b");
    expect(defaultModelId([], undefined)).toBeUndefined();
    expect(modelArgs("a")).toEqual(["--model", "experiential/a"]);
  });
});
