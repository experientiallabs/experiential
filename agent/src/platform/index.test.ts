import { existsSync, readFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { createExpCodeExtension } from "../extension/index.js";
import type { GatewayConfig } from "../gateway/config.js";
import { createPlatformTools, PLATFORM_TOOL_NAMES, platformToolsEnabled, TRACES_SKILL_NAME, tracesSkillDir } from "./index.js";

const hosted: GatewayConfig = { kind: "hosted", baseUrl: "https://api.example/v1", apiKey: "xpl_k" };
const local: GatewayConfig = { kind: "local", baseUrl: "http://127.0.0.1:8000/v1", apiKey: "k" };

function fakePi() {
  const tools: string[] = [];
  const commands: string[] = [];
  const handlers = new Map<string, ((event: unknown, ctx: unknown) => unknown)[]>();
  const api = {
    registerTool: (tool: { name: string }) => tools.push(tool.name),
    registerCommand: (name: string) => commands.push(name),
    registerEntryRenderer: () => {},
    registerProvider: () => {},
    on: (event: string, handler: (event: unknown, ctx: unknown) => unknown) => handlers.set(event, [...(handlers.get(event) ?? []), handler])
  };
  return { api, tools, handlers };
}

const listing = { models: [{ id: "m" }] };

describe("platform tool defaults", () => {
  it("is on in the CLI unless the environment turns it off", () => {
    expect(platformToolsEnabled({})).toBe(true);
    expect(platformToolsEnabled({ EXP_CODE_PLATFORM_TOOLS: "1" })).toBe(true);
    for (const off of ["0", "false", "OFF", " no "]) expect(platformToolsEnabled({ EXP_CODE_PLATFORM_TOOLS: off })).toBe(false);
  });

  it("creates the five tools for a hosted gateway and none for a local one", () => {
    expect(createPlatformTools(hosted).map((tool) => tool.name)).toEqual([...PLATFORM_TOOL_NAMES]);
    expect(createPlatformTools(local)).toEqual([]);
    expect(createPlatformTools({ ...hosted, baseUrl: "https://api.example" })).toEqual([]);
  });

  it("registers the tools and the traces skill only when the extension is asked to", async () => {
    const off = fakePi();
    await createExpCodeExtension({ gateway: hosted, provider: listing })(off.api as never);
    expect(off.tools.filter((name) => name.startsWith("experiential_"))).toEqual([]);
    const offSkills = (await off.handlers.get("resources_discover")?.[0]?.({}, {})) as { skillPaths: string[] };
    expect(offSkills.skillPaths).toHaveLength(1);

    const root = await mkdtemp(join(tmpdir(), "exp-code-platform-skill-"));
    try {
      const on = fakePi();
      await createExpCodeExtension({ gateway: hosted, provider: listing, platformTools: true, skillsRoot: root })(on.api as never);
      expect(on.tools).toEqual(["todo_write", "ask_user", "web_fetch", "web_search", ...PLATFORM_TOOL_NAMES]);
      const onSkills = (await on.handlers.get("resources_discover")?.[0]?.({}, {})) as { skillPaths: string[] };
      expect(onSkills.skillPaths).toHaveLength(2);
      expect(readFileSync(join(onSkills.skillPaths[1]!, TRACES_SKILL_NAME, "SKILL.md"), "utf8")).toMatch(/^---\nname: experiential-traces\ndescription: .{40,}/);

      const localOn = fakePi();
      await createExpCodeExtension({ gateway: local, provider: listing, platformTools: true, skillsRoot: root })(localOn.api as never);
      expect(localOn.tools.filter((name) => name.startsWith("experiential_"))).toEqual([]);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("writes the traces skill to a content-addressed directory", async () => {
    const root = await mkdtemp(join(tmpdir(), "exp-code-platform-skill-"));
    try {
      const dir = tracesSkillDir(root);
      expect(existsSync(join(dir, TRACES_SKILL_NAME, "SKILL.md"))).toBe(true);
      expect(tracesSkillDir(root)).toBe(dir);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
