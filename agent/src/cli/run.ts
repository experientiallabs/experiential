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
import { EXP_CODE_VERSION } from "../version.js";
import { createBrandingExtension, rebrandHelp } from "./branding.js";
import { expCodeHome } from "./environment.js";
import { defaultModelId, modelArgs, wantsDefaultModel } from "./model-choice.js";
import { ensureSearchTools, markUpstreamChangelogSeen } from "./startup.js";

/** Where exp-code's documentation lives. */
export const DOCS_URL = "https://github.com/experientiallabs/experiential/tree/main/agent#readme";

const HELP_PREFACE = `exp-code ${EXP_CODE_VERSION}: Experiential's coding and cowork agent.

Models come from the Experiential gateway. \`exp-code login\` signs in through your browser and
saves a key (\`exp-code login --status\` shows it, \`exp-code logout\` forgets it); EXP_API_KEY (hosted)
or EXP_GATEWAY_KEY (a local \`exp\` gateway) in the environment take precedence. EXP_CODE_MODEL picks the default model
(otherwise experiential-large-model when the gateway lists it), EXP_CODE_WIRE (completions, responses, messages) forces the wire API, and EXP_CODE_HOME moves the
configuration directory (default ~/.exp-code). Documentation: ${DOCS_URL}
`;

/** Run the CLI with the given arguments (without the node and script paths). */
export async function runCli(args: string[], env: NodeJS.ProcessEnv = process.env): Promise<void> {
  const auth = parseAuthCommand(args);
  if (auth !== undefined) {
    process.exitCode = await runAuthCommand(auth, env);
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
  let finalArgs = args;
  if (resolution.config === null) {
    if (!parsed.help) console.error(`exp-code: ${resolution.problem}`);
  } else if (wantsDefaultModel(args, parsed, agentDir)) {
    try {
      const listed = await fetchGatewayModels(resolution.config, { cacheDir });
      const chosen = defaultModelId(
        listed.map((model) => model.id),
        env.EXP_CODE_MODEL
      );
      if (chosen !== undefined) finalArgs = [...modelArgs(chosen), ...args];
    } catch (error) {
      console.error(`exp-code: ${(error as Error).message}`);
    }
  }
  process.title = "exp-code";
  if (!parsed.help && isInteractiveStart(parsed)) {
    markUpstreamChangelogSeen(agentDir);
    await ensureSearchTools(parsed.offline === true);
  }
  const restoreOutput = wantsHelp(args) ? rewriteStdout(rebrandHelp) : () => {};
  try {
    await main(finalArgs, {
      extensionFactories: [
        {
          name: "exp-code",
          hidden: true,
          factory: createExpCodeExtension({
            gateway: resolution.config,
            gatewayProblem: resolution.problem,
            provider: { cacheDir, wire, onListingError: (error) => console.error(`exp-code: ${error.message}`) }
          })
        },
        { name: "exp-code-branding", hidden: true, factory: createBrandingExtension(EXP_CODE_VERSION) },
        // Last, so its preamble swap sees every other prompt edit.
        { name: "exp-code-identity", hidden: true, factory: createIdentityExtension() }
      ]
    });
  } finally {
    restoreOutput();
  }
}

/** Whether the arguments ask for help (`--help`/`-h`, including on a subcommand). */
function wantsHelp(args: string[]): boolean {
  return args.some((arg) => arg === "--help" || arg === "-h");
}

/** Pass everything written to stdout through `transform` until the returned restore is called. */
function rewriteStdout(transform: (text: string) => string): () => void {
  const original = process.stdout.write.bind(process.stdout);
  process.stdout.write = ((chunk: string | Uint8Array, ...rest: unknown[]) =>
    original(typeof chunk === "string" ? transform(chunk) : chunk, ...(rest as []))) as typeof process.stdout.write;
  return () => {
    process.stdout.write = original;
  };
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
