import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { VERSION } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ensureSearchTools, loadToolsManager, markUpstreamChangelogSeen } from "./startup.js";

let agentDir: string;
beforeEach(async () => {
  agentDir = await mkdtemp(join(tmpdir(), "exp-code-startup-"));
});
afterEach(() => rm(agentDir, { recursive: true, force: true }));

describe("startup housekeeping", () => {
  it("loads the pinned tools manager", async () => {
    const tools = await loadToolsManager();
    expect(typeof tools.getToolPath).toBe("function");
    expect(typeof tools.ensureTool).toBe("function");
  });

  it("does nothing offline", async () => {
    const written: string[] = [];
    await ensureSearchTools(true, (text) => written.push(text));
    expect(written).toEqual([]);
  });

  it("records the bundled upstream version as seen, keeping other settings", () => {
    mkdirSync(agentDir, { recursive: true });
    const path = join(agentDir, "settings.json");
    writeFileSync(path, JSON.stringify({ lastChangelogVersion: "0.0.1", defaultModel: "m" }));
    markUpstreamChangelogSeen(agentDir);
    expect(JSON.parse(readFileSync(path, "utf8"))).toEqual({ lastChangelogVersion: VERSION, defaultModel: "m" });
  });

  it("leaves a missing settings file for the first start", () => {
    markUpstreamChangelogSeen(agentDir);
    expect(() => readFileSync(join(agentDir, "settings.json"))).toThrow();
  });
});
