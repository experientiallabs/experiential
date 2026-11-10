/**
 * The exp-code command: Pi's CLI with the exp-code extension and the gateway preloaded.
 *
 * Every Pi mode works unchanged (interactive, `-p` print, `--mode json`, `--mode rpc`), because
 * argument parsing and mode dispatch stay Pi's own. exp-code adds its extension as an inline
 * factory, starts on a gateway model by default, and reports its own version.
 */

import { join } from "node:path";
import { main, parseArgs, VERSION as PI_VERSION } from "@earendil-works/pi-coding-agent";
import { createExpCodeExtension } from "../extension/index.js";
import { resolveGateway } from "../gateway/config.js";
import { fetchGatewayModels, parseWireApi } from "../gateway/models.js";
import { EXP_CODE_VERSION } from "../version.js";
import { expCodeHome } from "./environment.js";
import { defaultModelId, modelArgs, wantsDefaultModel } from "./model-choice.js";

const HELP_PREFACE = `exp-code ${EXP_CODE_VERSION}: Experiential's coding and cowork agent, built on Pi ${PI_VERSION}.

Models come from the Experiential gateway (EXP_API_KEY for hosted, EXP_GATEWAY_KEY for a local
\`exp\` gateway; \`exp code\` sets this up from \`exp login\`). EXP_CODE_MODEL picks the default model,
EXP_CODE_WIRE (completions, responses, messages) forces the wire API, and EXP_CODE_HOME moves the
configuration directory (default ~/.exp-code). The options below are Pi's; "pi" means exp-code.
`;

/** Run the CLI with the given arguments (without the node and script paths). */
export async function runCli(args: string[], env: NodeJS.ProcessEnv = process.env): Promise<void> {
  const parsed = parseArgs(args);
  if (parsed.version) {
    console.log(`exp-code ${EXP_CODE_VERSION} (pi-coding-agent ${PI_VERSION})`);
    return;
  }
  if (parsed.help) process.stdout.write(`${HELP_PREFACE}\n`);

  const home = expCodeHome(env);
  const agentDir = join(home, "agent");
  const resolution = resolveGateway(env);
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
