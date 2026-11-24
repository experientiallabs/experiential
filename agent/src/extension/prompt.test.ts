import { describe, expect, it } from "vitest";
import { addConversationSection, addDocsSection, CONVERSATION_SECTION_NAME, docsSection, EXP_CODE_PREAMBLE, UPSTREAM_PREAMBLE, withExpCodePreamble } from "./prompt.js";

describe("conversation section", () => {
  it("adds a conversation's instructions as their own section, trimmed", () => {
    const options = { sections: {} as Record<string, string> };
    addConversationSection(options, "  Ask one question at a time.\n");
    expect(options.sections[CONVERSATION_SECTION_NAME]).toBe("Ask one question at a time.");
  });

  it("adds nothing for absent or blank instructions", () => {
    for (const instructions of [undefined, null, "", "   "]) {
      const options = { sections: {} as Record<string, string> };
      addConversationSection(options, instructions);
      expect(options.sections).toEqual({});
    }
  });
});

describe("exp-code identity", () => {
  it("swaps the upstream preamble for exp-code's and leaves a custom prompt alone", () => {
    expect(withExpCodePreamble(`${UPSTREAM_PREAMBLE}\n\n<tools>\nx\n</tools>`)).toBe(`${EXP_CODE_PREAMBLE}\n\n<tools>\nx\n</tools>`);
    expect(withExpCodePreamble("You are my reviewer.")).toBeUndefined();
    expect(EXP_CODE_PREAMBLE).toContain("exp-code");
    expect(EXP_CODE_PREAMBLE).not.toMatch(/\bpi\b/i);
  });

  it("points the docs section at exp-code's README, and skips it under a custom prompt", () => {
    const docs = docsSection("/pkg/README.md");
    expect(docs).toContain("exp-code README (usage, login, gateway, models, tools, modes): /pkg/README.md");
    expect(docs).not.toMatch(/\bPi\b/);
    const plain = { sections: {} as Record<string, string> };
    addDocsSection(plain);
    expect(plain.sections.docs).toContain("exp-code documentation");
    const custom = { sections: {} as Record<string, string>, customPrompt: "mine" };
    addDocsSection(custom);
    expect(custom.sections).toEqual({});
  });

  it("matches the pinned upstream default preamble", async () => {
    const { buildSystemPrompt } = (await import(new URL("./core/system-prompt.js", import.meta.resolve("@earendil-works/pi-coding-agent")).href)) as {
      buildSystemPrompt: (options: { cwd: string }) => string;
    };
    expect(buildSystemPrompt({ cwd: "/w" }).startsWith(UPSTREAM_PREAMBLE)).toBe(true);
  });
});
