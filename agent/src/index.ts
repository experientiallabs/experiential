/** exp-code's library surface: the extension, the gateway provider, and the headless runner. */

export { applyCliEnvironment, expCodeHome } from "./cli/environment.js";
export { runCli } from "./cli/run.js";
export { createExpCodeExtension, EXP_CODE_TOOL_NAMES, type ExpCodeExtensionOptions } from "./extension/index.js";
export {
  GATEWAY_PROVIDER_ID,
  type GatewayConfig,
  type GatewayKind,
  HOSTED_GATEWAY_BASE_URL,
  LOCAL_GATEWAY_BASE_URL,
  resolveGateway
} from "./gateway/config.js";
export { chooseWireApi, fetchGatewayModels, type GatewayWireApi, type ListedModel } from "./gateway/models.js";
export { registerGatewayProvider } from "./gateway/provider.js";
export * from "./runner/protocol.js";
export { type RunnerOptions, runOnce, STOPPED_NOTE } from "./runner/runner.js";
export { EXP_CODE_VERSION } from "./version.js";
