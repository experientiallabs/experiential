/**
 * The exp-code command: Pi's CLI with the exp-code extension and the gateway preloaded.
 *
 * Every Pi mode works unchanged (interactive, `-p` print, `--mode json`, `--mode rpc`), because
 * argument parsing and mode dispatch stay Pi's own. exp-code adds its extension as an inline
 * factory, starts on a gateway model by default, and reports its own version.
 */

import { join } from "node:path";
import { main, parseArgs, VERSION as PI_VERSION } from "@earendil-works/pi-coding-agent";
import { readCredentials } from "../auth/credentials.js";
import { parseAuthCommand, runAuthCommand, runLoginInTerminal } from "../auth/login.js";
import { createExpCodeExtension } from "../extension/index.js";
import { type GatewayResolution, resolveGateway } from "../gateway/config.js";
import { fetchGatewayModels, parseWireApi } from "../gateway/models.js";
import { EXP_CODE_VERSION } from "../version.js";
import { expCodeHome } from "./environment.js";
import { defaultModelId, modelArgs, wantsDefaultModel } from "./model-choice.js";

const HELP_PREFACE = `exp-code ${EXP_CODE_VERSION}: Experiential's coding and cowork agent, built on Pi ${PI_VERSION}.

Models come from the Experiential gateway. \`exp-code login\` signs in through your browser and
saves a key (\`exp-code login --status\` shows it, \`exp-code logout\` forgets it); EXP_API_KEY (hosted)
or EXP_GATEWAY_KEY (a local \`exp\` gateway) in the environment take precedence. EXP_CODE_MODEL picks the default model,
EXP_CODE_WIRE (completions, responses, messages) forces the wire API, and EXP_CODE_HOME moves the
configuration directory (default ~/.exp-code). The options below are Pi's; "pi" means exp-code.
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
    console.log(`exp-code ${EXP_CODE_VERSION} (pi-coding-agent ${PI_VERSION})`);
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
  await main(finalArgs, {
    extensionFactories: [
      {
        name: "exp-code",
        factory: createExpCodeExtension({
          gateway: resolution.config,
          gatewayProblem: resolution.problem,
          provider: { cacheDir, wire, onListingError: (error) => console.error(`exp-code: ${error.message}`) }
        })
      }
    ]
  });
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
