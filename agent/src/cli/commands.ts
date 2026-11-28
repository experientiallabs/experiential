/**
 * exp-code's own `/bug` and `/changelog` commands, in place of Pi's (removed in pi-patches.ts:
 * Pi's `/bug` uploaded a report to Pi's developers and its `/changelog` showed Pi's changelog),
 * plus `/help`, `/exit` and the unknown-command guard.
 *
 * `/bug [description]` opens a new GitHub issue on experientiallabs/experiential, prefilled with the
 * exp-code version, the operating system, Node.js and the session's model, and prints the same URL
 * (nothing is uploaded; the user reviews and submits the issue). `/changelog` links exp-code's
 * release notes for the running version and the list of exp-code releases.
 */

import { arch, platform, release } from "node:os";
import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { openBrowser } from "../auth/browser.js";

export const REPOSITORY_URL = "https://github.com/experientiallabs/experiential";

/** exp-code's documentation (the agent README). */
export const DOCS_URL = `${REPOSITORY_URL}/tree/main/agent#readme`;

/** What a bug report says about the environment. */
export interface BugReportFacts {
  version: string;
  os: string;
  node: string;
  model?: string;
}

/** The current environment's facts. */
export function bugReportFacts(version: string, model: string | undefined): BugReportFacts {
  return { version, os: `${platform()} ${release()} (${arch()})`, node: process.version, model };
}

/** A prefilled new-issue URL for exp-code. */
export function newIssueUrl(facts: BugReportFacts, description?: string): string {
  const summary = description?.trim();
  const body = [
    "**What happened**",
    "",
    summary || "",
    "",
    "**What you expected**",
    "",
    "",
    "**Environment**",
    "",
    `- exp-code: ${facts.version}`,
    `- OS: ${facts.os}`,
    `- Node.js: ${facts.node}`,
    `- Model: ${facts.model ?? "none selected"}`
  ].join("\n");
  const query = new URLSearchParams({ title: `exp-code: ${summary || "<short summary>"}`, body });
  return `${REPOSITORY_URL}/issues/new?${query.toString()}`;
}

/** Where the running version's release notes and every exp-code release are. */
export function releaseNotesUrls(version: string): { release: string; all: string } {
  return { release: `${REPOSITORY_URL}/releases/tag/exp-code-v${version}`, all: `${REPOSITORY_URL}/releases?q=exp-code&expanded=true` };
}

/**
 * The commands Pi's terminal UI handles itself (core/slash-commands.js as patched, plus the
 * unlisted `/debug`). It runs them only as typed exactly (or with an argument where they take
 * one), so one of them reaching a prompt was typed with an argument it does not take.
 * commands.test.ts keeps this list equal to the installed Pi's.
 */
export const BUILT_IN_COMMANDS: readonly string[] = [
  "settings", "model", "tree", "thinking", "scoped-models", "export", "import", "copy", "name", "session",
  "hotkeys", "fork", "clone", "trust", "login", "logout", "new", "compact", "resume", "reload", "quit", "debug"
];

/** What `/help` shows. */
export const HELP_TEXT = [
  "Type / to browse every command (Tab completes one), or ask exp-code itself how to use it.",
  "/model picks a model · /thinking sets the reasoning level · /usage shows time, tokens and cost",
  "/settings changes options · /hotkeys lists keyboard shortcuts · /new starts a new session · /quit (or /exit) quits",
  "!command runs a shell command (!!command keeps its output out of the conversation)",
  `Documentation: ${DOCS_URL}`
].join("\n");

/** Edit distance, for suggesting the command a typo meant. */
function distance(a: string, b: string): number {
  let previous = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const current = [i];
    for (let j = 1; j <= b.length; j++) current.push(Math.min(previous[j]! + 1, current[j - 1]! + 1, previous[j - 1]! + (a[i - 1] === b[j - 1] ? 0 : 1)));
    previous = current;
  }
  return previous[b.length]!;
}

/**
 * Why a submitted line that looks like a slash command is not one, or undefined when it should go
 * to the model. A line is a command when its first word is `/name` with no further slash in it, so
 * a path such as `/src/app.ts fails` is sent as written. `known` holds every command the session
 * runs (extension commands, prompt templates, `skill:` commands).
 */
export function unknownCommandNotice(text: string, known: Iterable<string>): string | undefined {
  const match = /^\/([^\s/]+)(?:\s|$)/.exec(text.trim());
  if (match === null) return undefined;
  const name = match[1]!;
  const names = new Set(known);
  if (names.has(name)) return undefined;
  if (BUILT_IN_COMMANDS.includes(name)) return `/${name} takes no arguments here; type /${name} on its own.`;
  const candidates = [...BUILT_IN_COMMANDS, ...names];
  const lower = name.toLowerCase();
  const suggestion = candidates.find((candidate) => candidate === lower) ?? candidates.find((candidate) => candidate.startsWith(lower)) ?? candidates.find((candidate) => distance(lower, candidate) <= (lower.length <= 4 ? 1 : 2));
  const hint = suggestion === undefined ? "" : ` Did you mean /${suggestion}?`;
  return `Unknown command /${name}.${hint} Type / to see every command. To send this text to the model, add a word before it.`;
}

export interface SupportCommandOptions {
  version: string;
  /** Opens a URL; defaults to the system browser. */
  open?: (url: string) => Promise<boolean>;
}

/**
 * The extension that registers `/bug`, `/changelog`, `/help` and `/exit`, and keeps a mistyped or
 * unknown `/command` from going to the model as a prompt (Pi sends any line it does not recognize,
 * so `/help` or `/exit` became a paid request the model answered). The guard runs on interactive
 * input only, before the prompt stash restores a draft, and puts the line back in the editor.
 */
export function createSupportCommandsExtension(options: SupportCommandOptions): ExtensionFactory {
  const open = options.open ?? ((url: string) => openBrowser(url));
  return (pi) => {
    pi.registerCommand("bug", {
      description: "Report an exp-code bug on GitHub (opens a prefilled issue)",
      handler: async (args, ctx) => {
        const model = ctx.model === undefined ? undefined : `${ctx.model.provider}/${ctx.model.id}`;
        const url = newIssueUrl(bugReportFacts(options.version, model), args);
        const opened = await open(url);
        ctx.ui.notify(`${opened ? "Opened a new exp-code issue in your browser" : "Open this link to report the bug"}: ${url}`, "info");
      }
    });
    pi.registerCommand("changelog", {
      description: "Show where exp-code's release notes are",
      handler: async (_args, ctx) => {
        const urls = releaseNotesUrls(options.version);
        ctx.ui.notify(`exp-code ${options.version} release notes: ${urls.release}\nAll exp-code releases: ${urls.all}`, "info");
      }
    });
    pi.registerCommand("help", {
      description: "Show the main commands and keys, and where the documentation is",
      handler: async (_args, ctx) => {
        ctx.ui.notify(HELP_TEXT, "info");
      }
    });
    pi.registerCommand("exit", {
      description: "Quit exp-code (same as /quit)",
      handler: async (_args, ctx) => {
        ctx.shutdown();
      }
    });
    pi.on("input", (event, ctx) => {
      if (event.source !== "interactive" || !ctx.hasUI) return { action: "continue" };
      const notice = unknownCommandNotice(event.text, pi.getCommands().map((command) => command.name));
      if (notice === undefined) return { action: "continue" };
      ctx.ui.notify(notice, "error");
      ctx.ui.setEditorText(event.text);
      return { action: "handled" };
    });
  };
}
