/**
 * exp-code's parts of the system prompt.
 *
 * The upstream harness builds the prompt from structured sections (preamble, tools, rules, docs,
 * skills, working directory). exp-code states its own identity in the preamble, points the docs
 * section at its own README, and adds the cowork section: what a general-purpose coworker needs
 * beyond coding (deliverables as files, verifying them, explaining results to people who may not
 * be programmers). Document formats are covered by the bundled skills, loaded on demand.
 */

import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { getDocsPath, getExamplesPath } from "@earendil-works/pi-coding-agent";

export const COWORK_SECTION_NAME = "cowork";

export const COWORK_SECTION = [
  "You are exp-code, a capable general-purpose agent. You get real work done on the user's computer: code, analysis, documents, spreadsheets, slides, research, data processing, and scripts. Work autonomously until the task is done, then summarize what you did and where the results are.",
  "",
  "- Save deliverables (reports, spreadsheets, slides, charts, code) as files in the working directory and name them in your answer, as paths relative to the working directory.",
  "- For Word, Excel, PowerPoint, PDF, charts, and data analysis, load the matching skill before you start and follow it.",
  "- Use bash to run code, install packages, and inspect the environment. Prefer running a script over guessing its output.",
  "- Research before you ask. A named company, product, person, place or topic is something to look up, not to ask about: search the web (when web search is available), read the best sources, and check the working directory. Ask the user only for things you cannot find (their preferences, private details, or a choice between real alternatives), and when you do, say what you already found.",
  "- Verify your work: run what you wrote, open the files you produced, check the numbers, and fix errors before you finish.",
  "- The user may not be a programmer. Explain results in plain language and keep code out of the answer unless asked.",
  "- Be concise."
].join("\n");

/** Minimal shape of Pi's structured prompt options that this section edits. */
export interface PromptSections {
  sections: Record<string, string>;
}

/** The section holding a conversation's standing instructions (RunSpec.instructions). */
export const CONVERSATION_SECTION_NAME = "conversation";

/**
 * Add a conversation's standing instructions as their own system prompt section. Nothing is added
 * for empty instructions.
 */
export function addConversationSection(options: PromptSections, instructions: string | null | undefined): void {
  const text = instructions?.trim() ?? "";
  if (text !== "") options.sections[CONVERSATION_SECTION_NAME] = text;
}

/** Add the cowork section to Pi's structured system prompt options. */
export function addCoworkSection(options: PromptSections): void {
  options.sections[COWORK_SECTION_NAME] = COWORK_SECTION;
}

/** The upstream default preamble, which names the upstream harness. */
export const UPSTREAM_PREAMBLE =
  "You are a coding assistant running inside pi, a coding agent harness. Help users inspect files, run commands, and modify code.";

/** exp-code's preamble: the agent's name and maker. */
export const EXP_CODE_PREAMBLE =
  "You are exp-code, the coding and cowork agent made by Experiential. Help users inspect files, run commands, modify code, and produce finished work. If asked who or what you are, you are exp-code by Experiential.";

/** exp-code's README, when it ships next to this module (the npm package; not the runner bundle). */
export function expCodeReadmePath(): string | undefined {
  const path = fileURLToPath(new URL("../../README.md", import.meta.url));
  return existsSync(path) ? path : undefined;
}

/** The docs section: exp-code's README first, then the reference docs for extending it. */
export function docsSection(readmePath: string | undefined = expCodeReadmePath()): string {
  const lines = ["exp-code documentation (read only when the user asks about exp-code itself, how to use it, or how to extend it):"];
  if (readmePath !== undefined) lines.push(`- exp-code README (usage, login, gateway, models, tools, modes): ${readmePath}`);
  lines.push(
    `- Reference docs for extending exp-code: ${getDocsPath()}`,
    `- Examples: ${getExamplesPath()} (extensions, custom tools, SDK)`,
    "- Resolve docs/... under the reference docs and examples/... under Examples, not the current working directory",
    "- When asked about: extensions (docs/extensions.md, examples/extensions/), themes (docs/themes.md), skills (docs/skills.md), prompt templates (docs/prompt-templates.md), TUI components (docs/tui.md), keybindings (docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers (docs/custom-provider.md), adding models (docs/models.md), packages (docs/packages.md), environment variables (docs/environment-variables.md), MCP servers (docs/mcp.md)",
    "- exp-code keeps its user configuration in ~/.exp-code/agent (settings.json, extensions/, skills/, sessions/) where these docs say ~/.pi/agent, reads project files from .exp-code/ where they say .pi/, and names its environment variables EXP_CODE_* where they say PI_* (the README lists them); call it exp-code, not pi",
    "- Every model runs through the Experiential gateway: exp-code has no direct vendor providers, so the docs' provider API keys and /login providers do not apply",
    "- Before implementing changes to exp-code's own setup, read the relevant docs and examples in full."
  );
  return lines.join("\n");
}

/** Minimal shape of the prompt options the identity edit reads. */
export interface IdentityPromptOptions extends PromptSections {
  customPrompt?: string;
}

/**
 * Point the docs section at exp-code. Skipped when the user supplied their own system prompt,
 * which carries no docs section.
 */
export function addDocsSection(options: IdentityPromptOptions): void {
  if (options.customPrompt) return;
  options.sections.docs = docsSection();
}

/**
 * The rendered prompt with exp-code's preamble in place of the upstream one, or undefined when the
 * prompt does not start with the upstream preamble (a custom system prompt is left as written).
 */
export function withExpCodePreamble(systemPrompt: string): string | undefined {
  if (!systemPrompt.startsWith(UPSTREAM_PREAMBLE)) return undefined;
  return EXP_CODE_PREAMBLE + systemPrompt.slice(UPSTREAM_PREAMBLE.length);
}
