import { describe, expect, it } from "vitest";
import { addConversationSection, CONVERSATION_SECTION_NAME } from "./prompt.js";

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
