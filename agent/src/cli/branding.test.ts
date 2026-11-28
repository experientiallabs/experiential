import { initTheme } from "@earendil-works/pi-coding-agent";
import { describe, expect, it } from "vitest";
import { EXPERIENTIAL_MARK, ExpCodeHeader, headerTop, markLines, createBrandingExtension } from "./branding.js";

const plain = { fg: (_color: string, text: string) => text, bold: (text: string) => text } as never;

describe("startup header", () => {
  it("draws the 12 by 12 mark as six half-block lines", () => {
    expect(EXPERIENTIAL_MARK.every((row) => row.length === 12)).toBe(true);
    const lines = markLines();
    expect(lines).toHaveLength(6);
    expect(lines[0]).toBe("▀▄▄     ▄▀▀█");
    expect(lines[5]).toBe("▄▄▀▀    ▀▄▄ ");
  });

  it("shows exp-code, Experiential and exp-code's own version beside the mark", () => {
    const top = headerTop(plain, { version: "9.9.9", showMark: true }).split("\n");
    expect(top[0]).toBe("▀▄▄     ▄▀▀█  exp-code v9.9.9");
    expect(top[1]).toContain("by Experiential");
    expect(headerTop(plain, { version: "9.9.9", showMark: false })).toBe("exp-code v9.9.9\nby Experiential");
  });

  it("names exp-code, never the upstream harness, collapsed and expanded", () => {
    initTheme();
    const header = new ExpCodeHeader(plain, { version: "1.0.0", showMark: true });
    for (const expanded of [false, true]) {
      header.setExpanded(expanded);
      const text = header.render(120).join("\n");
      expect(text).toContain("exp-code can explain its own features");
      expect(text).not.toMatch(/\bPi\b|\bpi\b/);
    }
  });
});

describe("update notice", () => {
  /** Run the extension's session_start handler with a recording UI. */
  function start(notice: Promise<string | undefined>, hasUI = true, notes: string[] = []) {
    const handlers: ((event: unknown, ctx: unknown) => void)[] = [];
    const notified: string[] = [];
    createBrandingExtension("1.0.0", notice, notes)({ on: (_name: string, handler: (event: unknown, ctx: unknown) => void) => handlers.push(handler) } as never);
    const ctx = { hasUI, ui: { setHeader: () => undefined, notify: (line: string) => notified.push(line) } };
    return { notified, fire: () => handlers.forEach((handler) => handler({}, ctx)) };
  }

  it("shows the notice once, when the check resolves", async () => {
    const { notified, fire } = start(Promise.resolve("exp-code 1.0.1 is available"));
    fire();
    fire();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(notified).toEqual(["exp-code 1.0.1 is available"]);
  });

  it("shows the startup notes once, before the update notice", async () => {
    const { notified, fire } = start(Promise.resolve("exp-code 1.0.1 is available"), true, ["the gateway no longer lists x, your saved model; using y"]);
    fire();
    fire();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(notified).toEqual(["the gateway no longer lists x, your saved model; using y", "exp-code 1.0.1 is available"]);
  });

  it("shows nothing without a newer release or without a UI", async () => {
    const none = start(Promise.resolve(undefined));
    none.fire();
    const headless = start(Promise.resolve("line"), false);
    headless.fire();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(none.notified).toEqual([]);
    expect(headless.notified).toEqual([]);
  });
});
