import { describe, expect, it } from "vitest";
import { createSupportCommandsExtension, newIssueUrl, releaseNotesUrls } from "./commands.js";

type Handler = (args: string, ctx: unknown) => Promise<void>;

/** Register the commands and return them with a recording UI context. */
function commands(open: (url: string) => Promise<boolean>) {
  const registered = new Map<string, { description: string; handler: Handler }>();
  createSupportCommandsExtension({ version: "0.1.9", open })({ registerCommand: (name: string, options: { description: string; handler: Handler }) => registered.set(name, options) } as never);
  const notified: string[] = [];
  const ctx = { model: { provider: "experiential", id: "gemini-4-neon" }, ui: { notify: (line: string) => notified.push(line) } };
  return { registered, notified, ctx };
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
