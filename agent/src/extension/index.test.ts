import { chmodSync, mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import type { GatewayConfig } from "../gateway/config.js";
import { createExpCodeExtension, createIdentityExtension } from "./index.js";
import { COWORK_SECTION, COWORK_SECTION_NAME, EXP_CODE_PREAMBLE, UPSTREAM_PREAMBLE } from "./prompt.js";

/** A minimal recorder for the parts of Pi's ExtensionAPI the extension uses. */
function fakePi() {
  const tools: string[] = [];
  const commands: string[] = [];
  const providers: { name: string; config: { models?: { id: string }[] } }[] = [];
  const handlers = new Map<string, ((event: unknown, ctx: unknown) => unknown)[]>();
  const api = {
    registerTool: (tool: { name: string }) => tools.push(tool.name),
    registerCommand: (name: string) => commands.push(name),
    registerEntryRenderer: () => {},
    registerProvider: (name: string, config: { models?: { id: string }[] }) => providers.push({ name, config }),
    on: (event: string, handler: (event: unknown, ctx: unknown) => unknown) => handlers.set(event, [...(handlers.get(event) ?? []), handler])
  };
  return { api, tools, commands, providers, handlers };
}

const gateway: GatewayConfig = { kind: "local", baseUrl: "http://127.0.0.1:1/v1", apiKey: "k" };

describe("exp-code extension", () => {
  it("registers the cowork tools, the gateway provider, the prompt section and the skills", async () => {
    const pi = fakePi();
    await createExpCodeExtension({ gateway, provider: { models: [{ id: "m1" }, { id: "m2" }] } })(pi.api as never);
    expect(pi.tools).toEqual(["todo_write", "ask_user", "web_fetch", "web_search"]);
    expect(pi.commands).toEqual(["usage", "cost"]);
    expect(pi.providers.map((provider) => [provider.name, provider.config.models?.map((model) => model.id)])).toEqual([["experiential", ["m1", "m2"]]]);
    const options = { sections: {} as Record<string, string> };
    for (const handler of pi.handlers.get("before_agent_start") ?? []) await handler({ systemPromptOptions: options }, {});
    expect(options.sections[COWORK_SECTION_NAME]).toBe(COWORK_SECTION);
    expect(options.sections.docs).toContain("exp-code README");
    const discovered = (await pi.handlers.get("resources_discover")?.[0]?.({}, {})) as { skillPaths: string[] };
    expect(discovered.skillPaths[0]).toMatch(/skills$/);
  });

  it("keeps the document skills when the traces skill cannot be written, and says why", async () => {
    const dir = mkdtempSync(join(tmpdir(), "exp-code-ext-"));
    try {
      // A root anyone can write to (and not sticky) is refused for the materialized traces skill.
      const open = join(dir, "open");
      mkdirSync(open);
      chmodSync(open, 0o777);
      const pi = fakePi();
      const hosted: GatewayConfig = { kind: "hosted", baseUrl: "https://api.example.test/v1", apiKey: "xpl_test" };
      await createExpCodeExtension({ gateway: hosted, provider: { models: [{ id: "m1" }] }, platformTools: true, skillsRoot: open })(pi.api as never);
      const notes: string[] = [];
      const discovered = (await pi.handlers.get("resources_discover")?.[0]?.({}, { hasUI: true, ui: { notify: (text: string) => notes.push(text) } })) as { skillPaths: string[] };
      expect(discovered.skillPaths).toHaveLength(1);
      expect(discovered.skillPaths[0]).toMatch(/skills$/);
      expect(notes).toEqual([expect.stringMatching(/^Some skills are unavailable: refusing skills root /)]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("names the agent exp-code in the preamble, leaving a custom prompt alone", async () => {
    const pi = fakePi();
    await createIdentityExtension()(pi.api as never);
    const handler = pi.handlers.get("before_agent_start")![0]!;
    expect(await handler({ systemPrompt: `${UPSTREAM_PREAMBLE}\n\n<cwd>\n/w\n</cwd>` }, {})).toEqual({ systemPrompt: `${EXP_CODE_PREAMBLE}\n\n<cwd>\n/w\n</cwd>` });
    expect(await handler({ systemPrompt: "Mine." }, {})).toBeUndefined();
  });

  it("runs without a gateway, warning interactive sessions once", async () => {
    const pi = fakePi();
    await createExpCodeExtension({ gateway: null, gatewayProblem: "no key" })(pi.api as never);
    expect(pi.tools).toEqual(["todo_write", "ask_user", "web_fetch"]);
    expect(pi.providers).toEqual([]);
    const notes: string[] = [];
    pi.handlers.get("session_start")?.[0]?.({}, { hasUI: true, ui: { notify: (text: string) => notes.push(text) } });
    expect(notes).toEqual(["no key"]);
  });

  it("registers the provider without models and reports why when the listing fails", async () => {
    const pi = fakePi();
    const errors: string[] = [];
    const fetch = (async () => new Response("{}", { status: 500 })) as typeof globalThis.fetch;
    await createExpCodeExtension({ gateway, provider: { fetch, ttlMs: 0, onListingError: (error) => errors.push(error.message) } })(pi.api as never);
    expect(pi.providers[0]?.config.models).toEqual([]);
    expect(errors[0]).toMatch(/refused the model listing \(500\)/);
  });
});
