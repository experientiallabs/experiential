import { initTheme } from "@earendil-works/pi-coding-agent";
import { describe, expect, it } from "vitest";
import { EXPERIENTIAL_MARK, ExpCodeHeader, headerTop, markLines, rebrandHelp } from "./branding.js";

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

describe("help text", () => {
  it("names exp-code in usage, commands and examples, and keeps environment variable names", () => {
    const upstream = [
      "pi - AI coding assistant with read, bash, edit, write tools",
      "  pi [options] [--] [@files...] [messages...]",
      "  pi update [source|self|pi]   Update pi, extensions, or model catalogs",
      "  pi install <source> [-l]     Install extension source",
      '  pi -p "List all .ts files"',
      "  pi --export ~/.pi/agent/sessions/x.jsonl",
      "  PI_OFFLINE                       - Disable startup network operations"
    ].join("\n");
    expect(rebrandHelp(upstream).split("\n")).toEqual([
      "exp-code - Experiential's coding and cowork agent",
      "  exp-code [options] [--] [@files...] [messages...]",
      "  exp-code update [source]           Update installed extensions or model catalogs",
      "  exp-code install <source> [-l]     Install extension source",
      '  exp-code -p "List all .ts files"',
      "  exp-code --export ~/.exp-code/agent/sessions/x.jsonl",
      "  PI_OFFLINE                       - Disable startup network operations"
    ]);
  });
});
