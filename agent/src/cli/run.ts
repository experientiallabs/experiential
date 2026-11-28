/**
 * The exp-code command: the upstream agent CLI with the exp-code extension and the gateway preloaded.
 *
 * Every mode works unchanged (interactive, `-p` print, `--mode json`, `--mode rpc`), because
 * argument parsing and mode dispatch stay upstream. exp-code adds its extensions as hidden inline
 * factories, starts on a gateway model by default, and shows its own name, mark and version.
 */

import { join } from "node:path";
import { main, parseArgs } from "@earendil-works/pi-coding-agent";
import { readCredentials } from "../auth/credentials.js";
import { parseAuthCommand, runAuthCommand, runLoginInTerminal } from "../auth/login.js";
import { createExpCodeExtension, createIdentityExtension } from "../extension/index.js";
import { type GatewayResolution, resolveGateway } from "../gateway/config.js";
import { fetchGatewayModels, parseWireApi } from "../gateway/models.js";
import { platformToolsEnabled } from "../platform/index.js";
import { createPromptStashExtension } from "../stash/extension.js";
import { EXP_CODE_VERSION } from "../version.js";
import { createBrandingExtension } from "./branding.js";
import { createSupportCommandsExtension } from "./commands.js";
import { expCodeHome } from "./environment.js";
import {
  defaultModelId,
  directProviderIds,
  gatewayFallback,
  gatewayModelArgs,
  modelArgs,
  repointSavedModel,
  savedDirectModel,
  wantsDefaultModel
} from "./model-choice.js";
import { ensureSearchTools, markUpstreamChangelogSeen } from "./startup.js";
import { checkForUpdate, parseUpdateArgs, selfUpdate, UPDATE_HELP } from "./update.js";

/** Where exp-code's documentation lives. */
export const DOCS_URL = "https://github.com/experientiallabs/experiential/tree/main/agent#readme";

const HELP_PREFACE = `exp-code ${EXP_CODE_VERSION}: Experiential's coding and cowork agent.

Models come from the Experiential gateway. \`exp-code login\` signs in through your browser and
saves a key (\`exp-code login --status\` shows it, \`exp-code logout\` forgets it); EXP_API_KEY (hosted)
or EXP_GATEWAY_KEY (a local \`exp\` gateway) in the environment take precedence. EXP_CODE_MODEL picks the default model
(otherwise experiential-large-model when the gateway lists it), EXP_CODE_WIRE (completions, responses, messages) forces the wire API, and EXP_CODE_HOME moves the
configuration directory (default ~/.exp-code), and EXP_CODE_PLATFORM_TOOLS=0 turns off the tools that read
your organization's Experiential data (request log, usage, models, evals). Every model runs through the
gateway. \`exp-code update\` updates exp-code. Documentation: ${DOCS_URL}
`;

/** Run the CLI with the given arguments (without the node and script paths). */
export async function runCli(args: string[], env: NodeJS.ProcessEnv = process.env): Promise<void> {
  const auth = parseAuthCommand(args);
  if (auth !== undefined) {
    process.exitCode = await runAuthCommand(auth, env);
    return;
  }
  if (args[0] === "update") {
    process.exitCode = await runUpdateCommand(args.slice(1), env);
    return;
  }
  const parsed = parseArgs(args);
  if (parsed.version) {
    console.log(`exp-code ${EXP_CODE_VERSION}`);
    return;
  }
  if (parsed.help) process.stdout.write(`${HELP_PREFACE}\n`);

  const home = expCodeHome(env);
  const agentDir = join(home, "agent");
  let resolution: GatewayResolution = resolveGateway(env, await readCredentials(env));
  // An interactive start with no key anywhere offers to log in rather than starting keyless.
  if (resolution.missingKey === true && !parsed.help && isInteractiveStart(parsed) && (await confirmLogin())) {
    const code = await runLoginInTerminal(env, false);
    if (code !== 0) {
      process.exitCode = code;
      return;
    }
    resolution = resolveGateway(env, await readCredentials(env));
  }
  const wire = parseWireApi(env.EXP_CODE_WIRE);
  const cacheDir = join(home, "cache");
  const gateway = resolution.config;
  // The gateway's aliases, listed once and only when a choice below needs them.
  let listing: Promise<string[] | undefined> | undefined;
  const listedIds = (): Promise<string[] | undefined> => {
    listing ??=
      gateway === null
        ? Promise.resolve(undefined)
        : fetchGatewayModels(gateway, { cacheDir }).then(
            (models) => models.map((model) => model.id),
            (error: unknown) => {
              console.error(`exp-code: ${(error as Error).message}`);
              return undefined;
            }
          );
    return listing;
  };

  // Selections that name one of Pi's direct providers move to the gateway.
  const direct = directProviderIds();
  let finalArgs = args;
  if (gatewayModelArgs(args, direct).moved.length > 0) {
    const listed = await listedIds();
    const pointed = gatewayModelArgs(args, direct, (id) => gatewayFallback(id, listed, env.EXP_CODE_MODEL));
    for (const move of pointed.moved) console.error(`exp-code: every model runs through the Experiential gateway; using ${move.to} for ${move.from}`);
    finalArgs = pointed.args;
  }
  const savedDirect = savedDirectModel(agentDir, direct);
  if (savedDirect !== undefined) {
    const alias = gatewayFallback(savedDirect, await listedIds(), env.EXP_CODE_MODEL);
    repointSavedModel(agentDir, alias, direct);
  }

  if (gateway === null) {
    if (!parsed.help) console.error(`exp-code: ${resolution.problem}`);
  } else if (wantsDefaultModel(finalArgs, parsed, agentDir)) {
    const listed = await listedIds();
    const chosen = listed === undefined ? undefined : defaultModelId(listed, env.EXP_CODE_MODEL);
    if (chosen !== undefined) finalArgs = [...modelArgs(chosen), ...finalArgs];
  }
  process.title = "exp-code";
  const interactive = !parsed.help && isInteractiveStart(parsed);
  // Started now, shown in the UI when it resolves; it never delays startup.
  const updateNotice = interactive
    ? checkForUpdate({ env, home, currentVersion: EXP_CODE_VERSION, interactive, offline: parsed.offline === true })
    : Promise.resolve(undefined);
  if (interactive) {
    markUpstreamChangelogSeen(agentDir);
    await ensureSearchTools(parsed.offline === true);
  }
  await main(finalArgs, {
    extensionFactories: [
      {
        name: "exp-code",
        hidden: true,
        factory: createExpCodeExtension({
          gateway: resolution.config,
          gatewayProblem: resolution.problem,
          platformTools: platformToolsEnabled(env),
          provider: { cacheDir, wire, onListingError: (error) => console.error(`exp-code: ${error.message}`) }
        })
      },
      { name: "exp-code-branding", hidden: true, factory: createBrandingExtension(EXP_CODE_VERSION, updateNotice) },
      { name: "exp-code-commands", hidden: true, factory: createSupportCommandsExtension({ version: EXP_CODE_VERSION }) },
      { name: "exp-code-prompt-stash", hidden: true, factory: createPromptStashExtension({ agentDir }) },
      // Last, so its preamble swap sees every other prompt edit.
      { name: "exp-code-identity", hidden: true, factory: createIdentityExtension() }
    ]
  });
}

/** `exp-code update ...`: exp-code itself, then any extension or model update Pi performs. */
async function runUpdateCommand(args: string[], env: NodeJS.ProcessEnv): Promise<number> {
  const request = parseUpdateArgs(args);
  if (request.help) {
    process.stdout.write(UPDATE_HELP);
    return 0;
  }
  let code = 0;
  if (request.self) code = await selfUpdate({ env, currentVersion: EXP_CODE_VERSION });
  if (request.packageArgs !== undefined && code === 0) {
    await main(request.packageArgs);
    code = typeof process.exitCode === "number" ? process.exitCode : 0;
  }
  return code;
}

/** Whether this run opens the interactive terminal UI on a real terminal. */
function isInteractiveStart(parsed: ReturnType<typeof parseArgs>): boolean {
  if (parsed.print === true || (parsed.mode !== undefined && parsed.mode !== "text")) return false;
  return process.stdin.isTTY === true && process.stdout.isTTY === true;
}

/** Ask "Log in now? [Y/n]" on the terminal; Enter means yes. */
async function confirmLogin(): Promise<boolean> {
  const { createInterface } = await import("node:readline/promises");
  const rl = createInterface({ input: process.stdin, output: process.stdout });
  try {
    const answer = (await rl.question("No API key found. Log in now? [Y/n] ")).trim().toLowerCase();
    return answer === "" || answer === "y" || answer === "yes";
  } catch {
    return false;
  } finally {
    rl.close();
  }
}
