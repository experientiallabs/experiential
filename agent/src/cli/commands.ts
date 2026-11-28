/**
 * exp-code's own `/bug` and `/changelog` commands, in place of Pi's (removed in pi-patches.ts:
 * Pi's `/bug` uploaded a report to Pi's developers and its `/changelog` showed Pi's changelog).
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

export interface SupportCommandOptions {
  version: string;
  /** Opens a URL; defaults to the system browser. */
  open?: (url: string) => Promise<boolean>;
}

/** The extension that registers `/bug` and `/changelog`. */
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
  };
}
