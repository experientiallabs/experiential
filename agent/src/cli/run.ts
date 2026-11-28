/**
 * The exp-code command: the upstream agent CLI with the exp-code extension and the gateway preloaded.
 *
 * Every mode works unchanged (interactive, `-p` print, `--mode json`, `--mode rpc`), because
 * argument parsing and mode dispatch stay upstream. exp-code adds its extensions as hidden inline
 * factories, starts on a gateway model by default, and shows its own name, mark and version.
 */

import { existsSync } from "node:fs";
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
import { createSupportCommandsExtension, DOCS_URL } from "./commands.js";
import type { EarlyInput } from "./early-input.js";
import { expCodeHome } from "./environment.js";
import {
  directProviderIds,
  gatewayFallback,
  gatewayModelArgs,
  modelArgs,
  repointSavedModel,
  savedDefaultModel,
  savedDirectModel,
  startModel,
  wantsDefaultModel
} from "./model-choice.js";
import { ensureSearchTools, markUpstreamChangelogSeen } from "./startup.js";
import { checkForUpdate, parseUpdateArgs, selfUpdate, UPDATE_HELP } from "./update.js";

/** Where exp-code's documentation lives. */
export { DOCS_URL } from "./commands.js";

const HELP_PREFACE = `exp-code ${EXP_CODE_VERSION}: Experiential's coding and cowork agent.

Models come from the Experiential gateway. \`exp-code login\` signs in through your browser and
saves a key (\`exp-code login --status\` shows it, \`exp-code logout\` forgets it); EXP_API_KEY (hosted)
or EXP_GATEWAY_KEY (a local \`exp\` gateway) in the environment take precedence. EXP_CODE_MODEL picks the start model for a run
(otherwise the model you last chose, else experiential-large-model when the gateway lists it), EXP_CODE_WIRE (completions, responses, messages) forces the wire API, and EXP_CODE_HOME moves the
configuration directory (default ~/.exp-code), and EXP_CODE_PLATFORM_TOOLS=0 turns off the tools that read
your organization's Experiential data (request log, usage, models, evals). Every model runs through the
gateway. \`exp-code update\` updates exp-code. Documentation: ${DOCS_URL}
`;

/** The subcommands exp-code runs itself or hands to Pi; any other bare first word is refused. */
export const SUBCOMMANDS: readonly string[] = ["login", "logout", "whoami", "update", "install", "remove", "list", "mcp", "help"];

/**
 * Why the arguments start with an unknown subcommand, or undefined when they do not. A first
 * argument that is one bare word (letters, digits, `-`, `_`; no spaces, path or option) and neither
 * a subcommand nor a file here would otherwise go to the model as the first message, and cost
 * money: `exp-code frobnicate` is a mistyped command far more often than a one-word prompt.
 */
export function unknownSubcommandMessage(args: readonly string[], exists: (path: string) => boolean = existsSync): string | undefined {
  const first = args[0];
  if (first === undefined || !/^[A-Za-z][A-Za-z0-9_-]*$/.test(first)) return undefined;
  if (SUBCOMMANDS.includes(first) || first === "auth" || first === "uninstall" || exists(first)) return undefined;
  return [
    `exp-code: unknown command "${first}".`,
    `Commands: ${SUBCOMMANDS.filter((name) => name !== "help").join(", ")} (exp-code --help shows every option).`,
    `To send a prompt, quote it: exp-code "${first} ..." (a one-word prompt: exp-code -- ${first}).`
  ].join("\n");
}

/** Why `--api-key` is refused, and what to use instead. */
export const API_KEY_FLAG_HELP =
  "--api-key is not supported: a key on the command line is visible to other local users. " +
  "Set EXP_API_KEY (EXP_GATEWAY_KEY for a local `exp` gateway) or run `exp-code login`.";

/** Run the CLI with the given arguments (without the node and script paths). */
export async function runCli(args: string[], env: NodeJS.ProcessEnv = process.env, earlyInput?: EarlyInput): Promise<void> {
  try {
    await runCliWith(args, env, earlyInput);
  } finally {
    // Anything but the terminal UI gets stdin back as it was.
    earlyInput?.release();
  }
}

async function runCliWith(args: string[], env: NodeJS.ProcessEnv, earlyInput: EarlyInput | undefined): Promise<void> {
  if (args.length === 1 && args[0] === "help") args = ["--help"];
  const unknown = unknownSubcommandMessage(args);
  if (unknown !== undefined) {
    console.error(unknown);
    process.exitCode = 2;
    return;
  }
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
  if (parsed.apiKey !== undefined && !parsed.help) {
    // Pi's --api-key would key the model calls but not the gateway's listing, and a key on the
    // command line is visible to every local user through the process list.
    console.error(`exp-code: ${API_KEY_FLAG_HELP}`);
    process.exitCode = 2;
    return;
  }

  const home = expCodeHome(env);
  const agentDir = join(home, "agent");
  let resolution: GatewayResolution = resolveGateway(env, await readCredentials(env));
  // An interactive start with no key anywhere offers to log in rather than starting keyless.
  if (resolution.missingKey === true && !parsed.help && isInteractiveStart(parsed)) earlyInput?.release();
  if (resolution.missingKey === true && !parsed.help && isInteractiveStart(parsed) && (await confirmLogin())) {
    const code = await runLoginInTerminal(env, false);
    if (code !== 0) {
      process.exitCode = code;
      return;
    }
    resolution = resolveGateway(env, await readCredentials(env));
  }
  let wire: ReturnType<typeof parseWireApi>;
  try {
    wire = parseWireApi(env.EXP_CODE_WIRE);
  } catch (error) {
    // A bad EXP_CODE_WIRE is a configuration mistake, not a crash with a stack trace.
    console.error(`exp-code: EXP_CODE_WIRE: ${(error as Error).message}`);
    process.exitCode = 2;
    return;
  }
  const cacheDir = join(home, "cache");
  const gateway = resolution.config;
  // One-line notes about the start model and the gateway: on stderr, or in the terminal UI, which
  // would cover a line printed before it starts. Each distinct note is shown once.
  const interactive = !parsed.help && isInteractiveStart(parsed);
  const startNotes: string[] = [];
  const noted = new Set<string>();
  const note = (line: string): void => {
    if (noted.has(line)) return;
    noted.add(line);
    if (interactive) startNotes.push(line);
    else console.error(`exp-code: ${line}`);
  };
  const onStaleListing = (error: Error, ageMs: number): void =>
    note(`${error.message} Using the model list cached ${describeAge(ageMs)} ago.`);

  // The gateway's aliases, listed once and only when a choice below needs them.
  let listing: Promise<string[] | undefined> | undefined;
  const listedIds = (): Promise<string[] | undefined> => {
    listing ??=
      gateway === null
        ? Promise.resolve(undefined)
        : fetchGatewayModels(gateway, { cacheDir, onStaleListing }).then(
            (models) => models.map((model) => model.id),
            (error: unknown) => {
              note((error as Error).message);
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
    for (const move of pointed.moved) note(`every model runs through the Experiential gateway; using ${move.to} for ${move.from}`);
    finalArgs = pointed.args;
  }
  const savedDirect = savedDirectModel(agentDir, direct);
  if (savedDirect !== undefined) {
    const alias = gatewayFallback(savedDirect, await listedIds(), env.EXP_CODE_MODEL);
    repointSavedModel(agentDir, alias, direct);
  }

  if (gateway === null) {
    if (!parsed.help) console.error(`exp-code: ${resolution.problem}`);
  } else if (wantsDefaultModel(finalArgs, parsed)) {
    // The listing is shared with the gateway provider's registration (cached in memory and on
    // disk), so checking the saved model against it costs no extra request.
    const start = startModel(env.EXP_CODE_MODEL?.trim() ? undefined : await listedIds(), env.EXP_CODE_MODEL, savedDefaultModel(agentDir));
    if (start.note !== undefined) note(start.note);
    if (start.id !== undefined) finalArgs = [...modelArgs(start.id), ...finalArgs];
  }
  process.title = "exp-code";
  // Started now, shown in the UI when it resolves; it never delays startup.
  const updateNotice = interactive
    ? checkForUpdate({ env, home, currentVersion: EXP_CODE_VERSION, interactive, offline: parsed.offline === true })
    : Promise.resolve(undefined);
  if (interactive) {
    markUpstreamChangelogSeen(agentDir);
    await ensureSearchTools(parsed.offline === true);
  }
  if (interactive) earlyInput?.handOff();
  else earlyInput?.release();
  await main(finalArgs, {
    extensionFactories: [
      {
        name: "exp-code",
        hidden: true,
        factory: createExpCodeExtension({
          gateway: resolution.config,
          gatewayProblem: resolution.problem,
          platformTools: platformToolsEnabled(env),
          provider: { cacheDir, wire, onStaleListing, onListingError: (error) => note(error.message) }
        })
      },
      { name: "exp-code-branding", hidden: true, factory: createBrandingExtension(EXP_CODE_VERSION, updateNotice, startNotes) },
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

/** A cache age in words: "40 seconds", "12 minutes", "3 hours", "2 days". */
export function describeAge(ms: number): string {
  const seconds = Math.round(ms / 1000);
  const [value, unit] =
    seconds < 90 ? [seconds, "second"] : seconds < 90 * 60 ? [Math.round(seconds / 60), "minute"] : seconds < 36 * 3600 ? [Math.round(seconds / 3600), "hour"] : [Math.round(seconds / 86400), "day"];
  return `${value} ${unit}${value === 1 ? "" : "s"}`;
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
