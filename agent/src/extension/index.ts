/**
 * The exp-code Pi extension: cowork tools, the cowork prompt section, the document skills, and
 * the Experiential gateway provider.
 *
 * The CLI and the headless runner load the same factory; they differ only in the options (where
 * shell commands run, whether the model list comes from the gateway or the run spec).
 */

import { type BashOperations, createLocalBashOperations, type ExtensionFactory } from "@earendil-works/pi-coding-agent";
import type { GatewayConfig } from "../gateway/config.js";
import { type GatewayProviderOptions, registerGatewayProvider } from "../gateway/provider.js";
import { createAskUserTool } from "./ask-user.js";
import { addCoworkSection } from "./prompt.js";
import { resolveSkillsDir } from "./skills.js";
import { createTodoTool } from "./todo.js";
import { createWebFetchTool } from "./web-fetch.js";
import { createWebSearchTool } from "./web-search.js";

export interface ExpCodeExtensionOptions {
  /** The gateway to register as a provider; null runs without it (other Pi providers still work). */
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
    pi.on("before_agent_start", (event) => {
      addCoworkSection(event.systemPromptOptions);
    });
    pi.on("resources_discover", () => ({ skillPaths: [resolveSkillsDir(options.skillsRoot)] }));
    const problem = options.gatewayProblem;
    if (options.gateway === null && problem !== undefined) {
      pi.on("session_start", (_event, ctx) => {
        if (ctx.hasUI) ctx.ui.notify(problem, "warning");
      });
    }
  };
}
