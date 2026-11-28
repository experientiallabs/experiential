import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";
import { applyFilePatch, PI_PATCHES, PiPatchError, patchFor, patchSource } from "./pi-patches.js";

const modulesDir = fileURLToPath(new URL("../../node_modules/@earendil-works", import.meta.url));

/** The installed file a patch edits. */
function installedFile(pkg: string, file: string): string {
  return join(modulesDir, pkg, "dist", file);
}

describe("Pi patches", () => {
  it.each(PI_PATCHES.map((patch) => [`${patch.pkg}/${patch.file}`, patch] as const))("apply exactly to the installed %s", (_name, patch) => {
    const path = installedFile(patch.pkg, patch.file);
    const source = readFileSync(path, "utf8");
    const patched = applyFilePatch(patch, source);
    expect(patched).not.toBe(source);
    expect(patchSource(path, source)).toBe(patched);
  });

  it("rename the app, the terminal title and the project folder", () => {
    const patch = patchFor(installedFile("pi-coding-agent", "config.js"))!;
    const patched = applyFilePatch(patch, readFileSync(installedFile("pi-coding-agent", "config.js"), "utf8"));
    expect(patched).toContain('APP_NAME = "exp-code"');
    expect(patched).toContain('CONFIG_DIR_NAME = ".exp-code"');
    expect(patched).not.toContain('"π"');
  });

  it("register none of Pi's direct providers", () => {
    const file = installedFile("pi-coding-agent", "core/model-runtime.js");
    expect(patchSource(file, readFileSync(file, "utf8"))).not.toContain("builtinProviders()");
  });

  it("fail loudly when the text moved", () => {
    expect(() => applyFilePatch({ pkg: "x", file: "y.js", edits: [{ find: "absent", replace: "" }] }, "source")).toThrow(PiPatchError);
    expect(() => applyFilePatch({ pkg: "x", file: "y.js", edits: [{ find: "a", replace: "b" }] }, "a a")).toThrow(/expected 1/);
  });

  it("match module paths and file URLs, on any platform", () => {
    expect(patchFor("file:///x/node_modules/@earendil-works/pi-coding-agent/dist/config.js")?.file).toBe("config.js");
    expect(patchFor("C:\\x\\node_modules\\@earendil-works\\pi-tui\\dist\\tui-main-screen.js")?.pkg).toBe("pi-tui");
    expect(patchFor("/x/node_modules/@earendil-works/pi-coding-agent/dist/core/sdk.js")).toBeUndefined();
    expect(patchSource("/x/other.js", "same")).toBe("same");
  });
});

describe("command list and sign-in guidance", () => {
  const patched = (file: string) => patchSource(installedFile("pi-coding-agent", file), readFileSync(installedFile("pi-coding-agent", file), "utf8"));

  it("tags no exp-code command or bundled skill in the autocomplete", async () => {
    const { InteractiveMode } = await import("@earendil-works/pi-coding-agent");
    const tag = (sourceInfo: unknown) => (InteractiveMode.prototype as unknown as { getAutocompleteSourceTag(info: unknown): string | undefined }).getAutocompleteSourceTag(sourceInfo);
    // An exp-code command (inline extension) and a skill its extension discovers.
    expect(tag({ path: "<inline:exp-code-commands>", source: "inline", scope: "temporary" })).toBeUndefined();
    expect(tag({ path: "/pkg/skills/pdf/SKILL.md", source: "extension:inline:exp-code", scope: "temporary" })).toBeUndefined();
    // A user's own extension or skill keeps its tag.
    expect(tag({ path: "/home/u/.exp-code/agent/extensions/x.ts", source: "local", scope: "user" })).toBe("u");
    expect(tag({ path: "/repo/skill/SKILL.md", source: "cli", scope: "temporary" })).toBe("t");
  });

  it("describe /logout for exp-code", () => {
    expect(patched("core/slash-commands.js")).toContain('{ name: "logout", description: "Remove credentials saved by /login (exp-code logout forgets the Experiential key)" }');
  });

  it("point a missing key at exp-code login, not Pi's providers and docs", async () => {
    const source = patched("core/auth-guidance.js");
    expect(source).not.toContain("log into a provider");
    expect(source).not.toContain("providers.md");
    const { formatNoApiKeyFoundMessage } = (await import(`data:text/javascript,${encodeURIComponent(source.replace(/^import .*$/gm, ""))}`)) as {
      formatNoApiKeyFoundMessage(provider: string): string;
    };
    expect(formatNoApiKeyFoundMessage("unknown")).toBe(
      "No API key found for the selected model.\n\nRun `exp-code login` in a terminal to sign in to Experiential, or set EXP_API_KEY to an xpl_ key (EXP_GATEWAY_KEY for a local `exp` gateway).\n`exp-code login --status` shows which key is in use; `exp-code --list-models` lists the models it can call."
    );
  });
});

describe("settings, footer and model list", () => {
  const patched = (file: string) => patchSource(installedFile("pi-coding-agent", file), readFileSync(installedFile("pi-coding-agent", file), "utf8"));

  it("offer no setting that does nothing in exp-code", () => {
    const source = patched("modes/interactive/components/settings-selector.js");
    expect(source).not.toContain('id: "share-destination"');
    expect(source).not.toContain('id: "install-telemetry"');
    expect(source).not.toContain("Radius");
  });

  it("list a 1,050,000-token window as 1.05M, like the footer", () => {
    const source = patched("cli/list-models.js");
    const body = source.slice(source.indexOf("function formatTokenCount(count) {"), source.indexOf("/**\n * List available models, optionally"));
    const format = new Function(`${body}\nreturn formatTokenCount;`)() as (count: number) => string;
    expect([format(1_050_000), format(1_048_576), format(1_000_000), format(200_000), format(131_072)]).toEqual(["1.05M", "1.05M", "1M", "200K", "131.1K"]);
  });

  it("move the model to its own footer line rather than cut its name", async () => {
    const { FooterComponent, initTheme } = await import("@earendil-works/pi-coding-agent");
    initTheme("dark");
    const model = { provider: "experiential", id: "experiential-small-model", reasoning: true, contextWindow: 1_050_000 };
    const session = {
      state: { model, thinkingLevel: "low", messages: [] },
      model,
      sessionManager: {
        getEntries: () => [],
        getEntryCount: () => 0,
        getLeafId: () => null,
        getSessionId: () => "s",
        getCwd: () => "/private/tmp/exp-qa/work3",
        getSessionName: () => undefined
      },
      getContextUsage: () => ({ tokens: 0, contextWindow: 1_050_000, percent: 0 }),
      modelRuntime: { isUsingSubscription: () => false },
      routedModel: undefined
    };
    const data = { getGitBranch: () => null, getExtensionStatuses: () => new Map(), getAvailableProviderCount: () => 1 };
    const footer = new FooterComponent(session as never, data as never);
    const strip = (line: string) => line.replace(/\x1b\[[0-9;]*m/g, "");
    const narrow = footer.render(40).map(strip);
    expect(narrow.some((line) => line.endsWith("experiential-small-model • low"))).toBe(true);
    expect(narrow.every((line) => line.length <= 40)).toBe(true);
    const wide = footer.render(100).map(strip);
    expect(wide).toHaveLength(2);
    expect(wide[1]).toMatch(/\(auto\)\s+experiential-small-model • low$/);
  });
});

describe("Pi as tests load it", () => {
  it("is patched (the test setup installs the load hook)", async () => {
    const config = await import("@earendil-works/pi-coding-agent");
    expect(config.CONFIG_DIR_NAME).toBe(".exp-code");
  });
});

describe("context window and compaction", () => {
  // A gateway model as toProviderModel builds it: the whole window, and the output reservation.
  const sol = { provider: "experiential", id: "experiential-large-model", contextWindow: 1_050_000, maxTokens: 128_000 };

  it("compacts before the prompt reaches the window less the model's output reservation", async () => {
    const { SettingsManager, shouldCompact } = await import("@earendil-works/pi-coding-agent");
    const settings = SettingsManager.inMemory().getCompactionSettings(sol as never);
    // 1,050,000 - 128,000 = 922,000 is all the provider accepts as prompt; Pi's 16,384 headroom
    // sits below that.
    expect(settings.reserveTokens).toBe(128_000 + 16_384);
    expect(shouldCompact(905_616, sol.contextWindow, settings)).toBe(false);
    expect(shouldCompact(905_617, sol.contextWindow, settings)).toBe(true);
    // Unpatched Pi compacted only above 1,033,616, past the 922,000-token prompt budget.
    expect(shouldCompact(922_001, sol.contextWindow, settings)).toBe(true);
  });

  it("keeps a configured headroom, capped so a small window still holds a conversation", async () => {
    const { SettingsManager } = await import("@earendil-works/pi-coding-agent");
    expect(SettingsManager.inMemory({ compaction: { reserveTokens: 50_000 } }).getCompactionSettings(sol as never).reserveTokens).toBe(178_000);
    // 32k window, 16k output: a 16k prompt budget keeps a quarter of it as headroom.
    expect(SettingsManager.inMemory().getCompactionSettings({ ...sol, contextWindow: 32_768, maxTokens: 16_384 } as never).reserveTokens).toBe(16_384 + 4_096);
    expect(SettingsManager.inMemory().getCompactionSettings(undefined).reserveTokens).toBe(16_384);
  });

  it("shows a 1,050,000-token window as 1.05M in the footer", async () => {
    const footer = await import(pathToFileURL(installedFile("pi-coding-agent", "modes/interactive/components/footer.js")).href);
    expect(footer.formatTokens(1_050_000)).toBe("1.05M");
    expect(footer.formatTokens(1_048_576)).toBe("1.05M");
    expect(footer.formatTokens(1_000_000)).toBe("1M");
    expect(footer.formatTokens(128_000)).toBe("128k");
  });
});

describe("remembered choices", () => {
  const patched = (file: string) => patchSource(installedFile("pi-coding-agent", file), readFileSync(installedFile("pi-coding-agent", file), "utf8"));

  it("save a model picked with Enter, /model <name> or Ctrl+P, and a thinking level for its model", () => {
    const source = patched("modes/interactive/interactive-mode.js");
    expect(source).not.toContain("setModel(model, { persist: false })");
    expect(source).not.toContain("setModel(model, { persist })");
    expect(source).toContain("this.session.cycleModel(direction, { persist: true })");
    expect(source.match(/this\.rememberThinkingLevel\(\);/g)).toHaveLength(2);
    expect(source).toContain("this.settingsManager.setModelThinkingLevel(model.provider, model.id, this.session.thinkingLevel)");
    expect(patched("modes/interactive/components/model-selector.js")).toContain("to select (remembered for next time)");
  });
});
