import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseArgs } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  defaultModelId,
  directProviderIds,
  gatewayFallback,
  gatewayModelArgs,
  modelArgs,
  repointSavedModel,
  savedDirectModel,
  wantsDefaultModel
} from "./model-choice.js";

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

  it("prefers EXP_CODE_MODEL, else experiential-large-model when listed, else the first listed alias", () => {
    expect(defaultModelId(["a", "b"], undefined)).toBe("a");
    expect(defaultModelId(["a", "experiential-small-model", "experiential-large-model"], undefined)).toBe("experiential-large-model");
    expect(defaultModelId(["a", "experiential-large-model"], "a")).toBe("a");
    expect(defaultModelId(["a", "b"], " b ")).toBe("b");
    expect(defaultModelId([], undefined)).toBeUndefined();
    expect(modelArgs("a")).toEqual(["--model", "experiential/a"]);
  });
});

describe("selections that name a direct provider", () => {
  const direct = directProviderIds();
  const listed = ["gemini-4-neon", "experiential-large-model"];

  it("covers Pi's direct providers and never the gateway", () => {
    for (const id of ["google", "openai", "anthropic", "openrouter", "groq", "xai", "mistral", "amazon-bedrock", "google-vertex", "llama.cpp"]) expect(direct.has(id), id).toBe(true);
    expect(direct.has("experiential")).toBe(false);
  });

  it("move to the gateway model of the same id, or the default model", () => {
    expect(gatewayFallback("gemini-4-neon", listed, undefined)).toBe("gemini-4-neon");
    expect(gatewayFallback("gemini-4-neon:high", listed, undefined)).toBe("gemini-4-neon:high");
    expect(gatewayFallback("gemini-2.5-pro", listed, undefined)).toBe("experiential-large-model");
    expect(gatewayFallback("gemini-2.5-pro", listed, "gemini-4-neon")).toBe("gemini-4-neon");
    expect(gatewayFallback("gemini-2.5-pro", undefined, undefined)).toBe("gemini-2.5-pro");
  });

  it("rewrite --model, --provider and --models", () => {
    const resolve = (id: string) => gatewayFallback(id, listed, undefined);
    expect(gatewayModelArgs(["--model", "google/gemini-4-neon", "-p", "hi"], direct, resolve)).toEqual({
      args: ["--model", "experiential/gemini-4-neon", "-p", "hi"],
      moved: [{ from: "google/gemini-4-neon", to: "experiential/gemini-4-neon" }]
    });
    expect(gatewayModelArgs(["--model=openai/gpt-4o"], direct, resolve).args).toEqual(["--model=experiential/experiential-large-model"]);
    expect(gatewayModelArgs(["--provider", "google", "--model", "gemini-4-neon"], direct, resolve).args).toEqual(["--provider", "experiential", "--model", "gemini-4-neon"]);
    expect(gatewayModelArgs(["--models", "google/gemini-4-neon,experiential/x"], direct, resolve).args).toEqual(["--models", "experiential/gemini-4-neon,experiential/x"]);
    expect(gatewayModelArgs(["--model", "experiential/x", "--", "--model", "google/y"], direct, resolve)).toEqual({ args: ["--model", "experiential/x", "--", "--model", "google/y"], moved: [] });
  });

  it("repoint a saved google/gemini-4-neon default at the gateway's gemini-4-neon", () => {
    writeFileSync(join(agentDir, "settings.json"), JSON.stringify({ defaultProvider: "google", defaultModel: "gemini-4-neon", enabledModels: ["google/gemini-4-neon", "experiential/x"], theme: "dark" }));
    expect(savedDirectModel(agentDir, direct)).toBe("gemini-4-neon");
    repointSavedModel(agentDir, gatewayFallback("gemini-4-neon", listed, undefined), direct);
    expect(JSON.parse(readFileSync(join(agentDir, "settings.json"), "utf8"))).toEqual({
      defaultProvider: "experiential",
      defaultModel: "gemini-4-neon",
      enabledModels: ["experiential/gemini-4-neon", "experiential/x"],
      theme: "dark"
    });
    expect(savedDirectModel(agentDir, direct)).toBeUndefined();
  });
});
