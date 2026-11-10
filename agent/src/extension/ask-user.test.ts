import { describe, expect, it } from "vitest";
import { askThroughUi, createAskUserTool, TYPE_OWN_ANSWER } from "./ask-user.js";

function ui(select: string | undefined, input: string | undefined) {
  const calls: string[] = [];
  return {
    calls,
    select: async (title: string, options: string[]) => {
      calls.push(`select:${title}:${options.join("|")}`);
      return select;
    },
    input: async (title: string) => {
      calls.push(`input:${title}`);
      return input;
    }
  };
}

describe("ask_user", () => {
  it("offers the options plus free text, and falls through to input when chosen", async () => {
    const pick = ui("Q2", undefined);
    expect(await askThroughUi(pick, "Which?", ["Q1", "Q2"])).toBe("Q2");
    expect(pick.calls).toEqual([`select:Which?:Q1|Q2|${TYPE_OWN_ANSWER}`]);
    const typed = ui(TYPE_OWN_ANSWER, "Q3");
    expect(await askThroughUi(typed, "Which?", ["Q1"])).toBe("Q3");
    expect(await askThroughUi(ui(undefined, "free"), "Open?", [])).toBe("free");
  });

  it("returns the answer to the model when there is a UI", async () => {
    const result = await createAskUserTool().execute("c", { question: "Which?", options: ["A", "B"] }, undefined, undefined, { hasUI: true, ui: ui("B", undefined) } as never);
    expect(result.content[0]).toEqual({ type: "text", text: "The user answered: B" });
    expect(result.details).toEqual({ question: "Which?", options: ["A", "B"], answer: "B" });
    expect((result as { terminate?: boolean }).terminate).toBeUndefined();
  });

  it("ends the turn with the question in details when there is no UI", async () => {
    const result = await createAskUserTool().execute("c", { question: "Which?" }, undefined, undefined, { hasUI: false } as never);
    expect(result.details).toEqual({ question: "Which?", options: [] });
    expect((result as { terminate?: boolean }).terminate).toBe(true);
  });
});
