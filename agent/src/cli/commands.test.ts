import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { BUILT_IN_COMMANDS, createSupportCommandsExtension, HELP_TEXT, newIssueUrl, releaseNotesUrls, unknownCommandNotice } from "./commands.js";
import { applyFilePatch, PI_PATCHES } from "./pi-patches.js";

type Handler = (args: string, ctx: unknown) => Promise<void>;
type InputHandler = (event: { text: string; source: string }, ctx: unknown) => { action: string };

/** Register the commands and return them with a recording UI context. */
function commands(open: (url: string) => Promise<boolean>, known: string[] = []) {
  const registered = new Map<string, { description: string; handler: Handler }>();
  const inputs: InputHandler[] = [];
  createSupportCommandsExtension({ version: "0.1.9", open })({
    registerCommand: (name: string, options: { description: string; handler: Handler }) => registered.set(name, options),
    on: (event: string, handler: InputHandler) => event === "input" && inputs.push(handler),
    getCommands: () => known.map((name) => ({ name }))
  } as never);
  const notified: string[] = [];
  const editor: string[] = [];
  let shutdowns = 0;
  const ctx = {
    model: { provider: "experiential", id: "gemini-4-neon" },
    hasUI: true,
    shutdown: () => void shutdowns++,
    ui: { notify: (line: string) => notified.push(line), setEditorText: (text: string) => editor.push(text) }
  };
  return { registered, inputs, notified, editor, ctx, shutdowns: () => shutdowns };
}

describe("/bug and /changelog", () => {
  it("open a new exp-code issue prefilled with version, OS, Node.js and model", async () => {
    const opened: string[] = [];
    const { registered, notified, ctx } = commands(async (url) => {
      opened.push(url);
      return true;
    });
    await registered.get("bug")!.handler("the picker froze", ctx);
    const url = new URL(opened[0]!);
    expect(`${url.origin}${url.pathname}`).toBe("https://github.com/experientiallabs/experiential/issues/new");
    expect(url.searchParams.get("title")).toBe("exp-code: the picker froze");
    const body = url.searchParams.get("body")!;
    expect(body).toContain("- exp-code: 0.1.9");
    expect(body).toContain("- Model: experiential/gemini-4-neon");
    expect(body).toMatch(/- OS: \S+/);
    expect(body).toContain(`- Node.js: ${process.version}`);
    expect(notified[0]).toBe(`Opened a new exp-code issue in your browser: ${opened[0]}`);
    expect(registered.get("bug")!.description).not.toMatch(/\bPi\b/);
  });

  it("print the issue link when no browser opens", async () => {
    const { registered, notified, ctx } = commands(async () => false);
    await registered.get("bug")!.handler("", ctx);
    expect(notified[0]).toMatch(/^Open this link to report the bug: https:\/\/github\.com\/experientiallabs\/experiential\/issues\/new\?/);
    expect(new URL(notified[0]!.split(": ")[1]!).searchParams.get("title")).toBe("exp-code: <short summary>");
  });

  it("link exp-code's own release notes", async () => {
    const { registered, notified, ctx } = commands(async () => true);
    await registered.get("changelog")!.handler("", ctx);
    expect(releaseNotesUrls("0.1.9").release).toBe("https://github.com/experientiallabs/experiential/releases/tag/exp-code-v0.1.9");
    expect(notified[0]).toContain("exp-code 0.1.9 release notes: https://github.com/experientiallabs/experiential/releases/tag/exp-code-v0.1.9");
    expect(newIssueUrl({ version: "1", os: "x", node: "v22" })).toContain("none+selected");
  });
});

describe("/help, /exit and unknown commands", () => {
  it("lists Pi's own interactive commands exactly", () => {
    const file = fileURLToPath(new URL("../../node_modules/@earendil-works/pi-coding-agent/dist/core/slash-commands.js", import.meta.url));
    const patch = PI_PATCHES.find((entry) => entry.file === "core/slash-commands.js")!;
    const names = [...applyFilePatch(patch, readFileSync(file, "utf8")).matchAll(/name: "([a-z-]+)"/g)].map((match) => match[1]);
    // `/debug` is handled by the terminal UI but not listed.
    expect([...BUILT_IN_COMMANDS].sort()).toEqual([...names, "debug"].sort());
    const ui = readFileSync(fileURLToPath(new URL("../../node_modules/@earendil-works/pi-coding-agent/dist/modes/interactive/interactive-mode.js", import.meta.url)), "utf8");
    expect(ui).toContain('if (text === "/debug") {');
  });

  it("leaves prompts, paths and every command the session runs alone", () => {
    const known = ["stash", "usage", "skill:pdf", "review"];
    for (const text of ["fix the bug", "/stash a draft", "/usage", "/skill:pdf make one", "/review src", "/src/app.ts fails to build", "/", "  ", "why does /model hang?"]) {
      expect(unknownCommandNotice(text, known)).toBeUndefined();
    }
  });

  it("names an unknown command and suggests the one a typo meant", () => {
    expect(unknownCommandNotice("/foo", [])).toBe("Unknown command /foo. Type / to see every command. To send this text to the model, add a word before it.");
    expect(unknownCommandNotice("/modle gpt", [])).toContain("Unknown command /modle. Did you mean /model?");
    expect(unknownCommandNotice("/Usage", ["usage"])).toContain("Did you mean /usage?");
    expect(unknownCommandNotice("/stas", ["stash"])).toContain("Did you mean /stash?");
    expect(unknownCommandNotice("/hotkeys please", [])).toBe("/hotkeys takes no arguments here; type /hotkeys on its own.");
  });

  it("keeps an unknown command from the model and puts it back in the editor", async () => {
    const { inputs, notified, editor, ctx } = commands(async () => true, ["stash", "exit"]);
    expect(inputs).toHaveLength(1);
    expect(inputs[0]!({ text: "/exitt", source: "interactive" }, ctx)).toEqual({ action: "handled" });
    expect(notified[0]).toContain("Unknown command /exitt. Did you mean /exit?");
    expect(editor).toEqual(["/exitt"]);
    expect(inputs[0]!({ text: "/stash x", source: "interactive" }, ctx)).toEqual({ action: "continue" });
    // Prompts from RPC, extensions or print mode are never second-guessed.
    expect(inputs[0]!({ text: "/exitt", source: "rpc" }, ctx)).toEqual({ action: "continue" });
    expect(inputs[0]!({ text: "/exitt", source: "interactive" }, { ...ctx, hasUI: false })).toEqual({ action: "continue" });
  });

  it("answer /help locally and quit on /exit", async () => {
    const { registered, notified, ctx, shutdowns } = commands(async () => true);
    await registered.get("help")!.handler("", ctx);
    expect(notified[0]).toBe(HELP_TEXT);
    expect(HELP_TEXT).toContain("https://github.com/experientiallabs/experiential/tree/main/agent#readme");
    expect(HELP_TEXT).not.toMatch(/\bPi\b/);
    await registered.get("exit")!.handler("", ctx);
    expect(shutdowns()).toBe(1);
  });
});
