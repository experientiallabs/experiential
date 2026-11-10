/**
 * `ask_user`: ask the user one question.
 *
 * With an interactive surface (the terminal UI, or an RPC client that answers
 * `extension_ui_request`), the question is shown as a choice with a free-text option and the
 * answer goes straight back to the model. Without one (print mode, JSON mode, the headless
 * runner), the tool ends the turn instead: the question stays in the tool's `details` and the
 * user's next message is the answer.
 */

import { defineTool, type ExtensionUIContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

/** The choice that switches the dialog to free text. */
export const TYPE_OWN_ANSWER = "Type my own answer";

export interface AskUserDetails {
  question: string;
  options: string[];
  /** The user's answer, absent when the question ended the turn unanswered. */
  answer?: string;
}

const askSchema = Type.Object({
  question: Type.String({ description: "One clear question for the user" }),
  options: Type.Optional(Type.Array(Type.String(), { description: "Two to five short answers the user can pick (they can also type their own)" }))
});

/** Ask through the UI; resolves undefined when the user dismisses the dialog. */
export async function askThroughUi(ui: Pick<ExtensionUIContext, "select" | "input">, question: string, options: string[], signal?: AbortSignal): Promise<string | undefined> {
  if (options.length > 0) {
    const choice = await ui.select(question, [...options, TYPE_OWN_ANSWER], { signal });
    if (choice === undefined || choice !== TYPE_OWN_ANSWER) return choice;
  }
  return ui.input(question, "Your answer", { signal });
}

export function createAskUserTool() {
  return defineTool({
    name: "ask_user",
    label: "Question",
    description:
      "Ask the user a question and wait for the answer. Use it only when the request is genuinely ambiguous or a choice changes the result substantially (for example which file, which audience, which format). Do not ask for permission to do what was already asked. Offer options when you can.",
    promptSnippet: "Ask the user a question when the request is genuinely ambiguous",
    promptGuidelines: ["Ask with ask_user only when a choice substantially changes the result; otherwise make a sensible choice and say which."],
    parameters: askSchema,
    executionMode: "sequential",
    async execute(_toolCallId, { question, options }, signal, _onUpdate, ctx) {
      const choices = (options ?? []).filter((option) => option.trim() !== "");
      if (ctx?.hasUI) {
        const answer = await askThroughUi(ctx.ui, question, choices, signal);
        if (answer !== undefined && answer.trim() !== "") {
          const details: AskUserDetails = { question, options: choices, answer };
          return { content: [{ type: "text", text: `The user answered: ${answer}` }], details };
        }
        const details: AskUserDetails = { question, options: choices };
        return {
          content: [{ type: "text", text: "The user dismissed the question without answering. Make a sensible choice and say which." }],
          details
        };
      }
      const details: AskUserDetails = { question, options: choices };
      return {
        content: [{ type: "text", text: "Question shown to the user. Wait for their answer in the next message." }],
        details,
        terminate: true
      };
    }
  });
}
