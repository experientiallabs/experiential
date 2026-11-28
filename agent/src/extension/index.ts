/**
 * The exp-code extension: cowork tools, the cowork prompt section, the document skills, and
 * the Experiential gateway provider.
 *
 * The CLI and the headless runner load the same factory; they differ only in the options (where
 * shell commands run, whether the model list comes from the gateway or the run spec).
 */

import { type BashOperations, createLocalBashOperations, type ExtensionFactory } from "@earendil-works/pi-coding-agent";
import type { GatewayConfig } from "../gateway/config.js";
import { type GatewayProviderOptions, registerGatewayProvider } from "../gateway/provider.js";
import { createPlatformTools, tracesSkillDir } from "../platform/index.js";
import { createAskUserTool } from "./ask-user.js";
import { addCoworkSection, addDocsSection, withExpCodePreamble } from "./prompt.js";
import { registerUsageCommand } from "../usage/command.js";
import { resolveSkillsDir } from "./skills.js";
import { createTodoTool } from "./todo.js";
import { createWebFetchTool } from "./web-fetch.js";
import { createWebSearchTool } from "./web-search.js";

export interface ExpCodeExtensionOptions {
  /** The gateway to register as a provider; null runs without it (and so without models). */
  gateway: GatewayConfig | null;
  /** Why there is no gateway, shown once in interactive sessions. */
  gatewayProblem?: string;
  /** Provider registration options (listing cache, wire override, fixed model list). */
  provider?: GatewayProviderOptions;
  /** Shell operations for `web_fetch`; defaults to Pi's local bash backend. */
  shell?: BashOperations;
  /** Alias `web_search` uses when the session model is not a gateway model. */
  searchModel?: string;
  /** Where bundled skills are written when they are embedded (the runner bundle). */
  skillsRoot?: string;
  /**
   * Register the read-only Experiential platform tools (request log, usage, models, evals) for a
   * hosted gateway. Off unless set: they can read the organization's stored prompts, so a host
   * enables them only where a prompt injection cannot quietly exfiltrate traces (the CLI turns
   * them on, the runner only when asked; see platform/index.ts).
   */
  platformTools?: boolean;
}

/** Tool names this extension registers. */
export const EXP_CODE_TOOL_NAMES = ["todo_write", "ask_user", "web_fetch", "web_search"] as const;

export function createExpCodeExtension(options: ExpCodeExtensionOptions): ExtensionFactory {
  return async (pi) => {
    pi.registerTool(createTodoTool());
    pi.registerTool(createAskUserTool());
    pi.registerTool(createWebFetchTool({ shell: options.shell ?? createLocalBashOperations() }));
    if (options.gateway !== null) {
      pi.registerTool(createWebSearchTool({ gateway: options.gateway, model: options.searchModel, fetch: options.provider?.fetch }));
      await registerGatewayProvider(pi, options.gateway, options.provider);
    }
    const platformTools = options.platformTools === true && options.gateway !== null ? createPlatformTools(options.gateway, { fetch: options.provider?.fetch }) : [];
    for (const tool of platformTools) pi.registerTool(tool);
    pi.on("before_agent_start", (event) => {
      addCoworkSection(event.systemPromptOptions);
      addDocsSection(event.systemPromptOptions);
    });
    pi.on("resources_discover", () => {
      const skillPaths = [resolveSkillsDir(options.skillsRoot)];
      // The traces skill describes the platform tools, so it is offered only with them.
      if (platformTools.length > 0) skillPaths.push(tracesSkillDir(options.skillsRoot));
      return { skillPaths };
    });
    const problem = options.gatewayProblem;
    if (options.gateway === null && problem !== undefined) {
      pi.on("session_start", (_event, ctx) => {
        if (ctx.hasUI) ctx.ui.notify(problem, "warning");
      });
    }
    registerUsageCommand(pi, { gateway: options.gateway, fetch: options.provider?.fetch });
  };
}

/**
 * The extension that gives the agent exp-code's name in the system prompt preamble.
 *
 * The upstream preamble is not a replaceable section, so the rendered prompt is replaced, which
 * fixes its text for the turn. Load this factory LAST: a prompt edit from any extension loaded
 * after it would be lost.
 */
export function createIdentityExtension(): ExtensionFactory {
  return (pi) => {
    pi.on("before_agent_start", (event) => {
      const systemPrompt = withExpCodePreamble(event.systemPrompt);
      return systemPrompt === undefined ? undefined : { systemPrompt };
    });
  };
}
