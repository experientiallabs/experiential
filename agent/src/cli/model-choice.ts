/**
 * The default model for a CLI invocation.
 *
 * When the user names no model and is not resuming a session, exp-code starts on a gateway
 * model: `EXP_CODE_MODEL` when set, otherwise `experiential-large-model` when the gateway lists
 * it, otherwise the first alias the gateway lists. A model saved in exp-code's settings (chosen
 * earlier with `/model`) or named with `--model` always wins.
 *
 * Every model runs through the gateway: exp-code registers none of Pi's direct vendor providers
 * (see pi-patches.ts). A selection that still names one, such as `google/gemini-4-neon` saved by
 * an older release or typed with `--model`, is moved to the gateway model of the same id, or to
 * the default model when the gateway does not list that id.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { Args } from "@earendil-works/pi-coding-agent";
import { builtinProviders } from "@earendil-works/pi-ai/providers/all";
import { GATEWAY_PROVIDER_ID, LARGE_MODEL_ALIAS } from "../gateway/config.js";

/** Pi subcommands that take no model. */
const SUBCOMMANDS = new Set(["auth", "install", "remove", "uninstall", "update", "list", "config", "mcp"]);

/** Whether this invocation should get an injected `--model`. */
export function wantsDefaultModel(args: string[], parsed: Args, agentDir: string): boolean {
  if (args[0] !== undefined && SUBCOMMANDS.has(args[0])) return false;
  if (parsed.help || parsed.version || parsed.export !== undefined || parsed.listModels !== undefined) return false;
  if (parsed.model !== undefined || parsed.provider !== undefined || (parsed.models?.length ?? 0) > 0) return false;
  if (parsed.continue || parsed.resume || parsed.session !== undefined || parsed.sessionId !== undefined || parsed.fork !== undefined) return false;
  return savedDefaultModel(agentDir) === undefined;
}

/** The `defaultModel` saved in exp-code's global settings, if any. */
export function savedDefaultModel(agentDir: string): string | undefined {
  try {
    const settings = JSON.parse(readFileSync(join(agentDir, "settings.json"), "utf8")) as { defaultModel?: unknown };
    return typeof settings.defaultModel === "string" && settings.defaultModel !== "" ? settings.defaultModel : undefined;
  } catch {
    return undefined;
  }
}

/** Pick the alias: the configured one when set, else the large model when listed, else the first listed. */
export function defaultModelId(listed: string[], configured: string | undefined): string | undefined {
  const wanted = configured?.trim();
  if (wanted) return wanted;
  if (listed.includes(LARGE_MODEL_ALIAS)) return LARGE_MODEL_ALIAS;
  return listed[0];
}

/** The `--model` argument pair for a gateway alias. */
export function modelArgs(modelId: string): string[] {
  return ["--model", `${GATEWAY_PROVIDER_ID}/${modelId}`];
}

/** The ids of Pi's direct providers, none of which exp-code registers. */
export function directProviderIds(): Set<string> {
  return new Set([...builtinProviders().map((provider) => provider.id), "llama.cpp"]);
}

/** Split `provider/model` when the provider is one of Pi's direct providers. */
export function directModelRef(ref: string, direct: Set<string>): { provider: string; id: string } | undefined {
  const slash = ref.indexOf("/");
  if (slash <= 0) return undefined;
  const provider = ref.slice(0, slash);
  return direct.has(provider) ? { provider, id: ref.slice(slash + 1) } : undefined;
}

/** The gateway alias for a model id: the id itself when listed, else the default model. */
export function gatewayFallback(id: string, listed: string[] | undefined, configured: string | undefined): string {
  if (listed === undefined || listed.includes(id)) return id;
  const bare = id.replace(/:[a-z]+$/, "");
  if (bare !== id && listed.includes(bare)) return id;
  return defaultModelId(listed, configured) ?? id;
}

/**
 * Point model arguments that name a direct provider at the gateway: `--model google/x` becomes
 * `--model experiential/x`, `--provider google` becomes `--provider experiential`, and `--models`
 * patterns are moved the same way. `resolve` picks the gateway alias for a moved id.
 */
export function gatewayModelArgs(args: string[], direct: Set<string>, resolve: (id: string) => string = (id) => id): { args: string[]; moved: { from: string; to: string }[] } {
  const moved: { from: string; to: string }[] = [];
  const point = (value: string): string => {
    const ref = directModelRef(value, direct);
    if (ref === undefined) return value;
    const alias = resolve(ref.id);
    moved.push({ from: value, to: `${GATEWAY_PROVIDER_ID}/${alias}` });
    return `${GATEWAY_PROVIDER_ID}/${alias}`;
  };
  const out: string[] = [];
  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!;
    const [flag, inline] = arg.startsWith("--") && arg.includes("=") ? [arg.slice(0, arg.indexOf("=")), arg.slice(arg.indexOf("=") + 1)] : [arg, undefined];
    if (arg === "--") {
      out.push(...args.slice(i));
      break;
    }
    if (flag !== "--model" && flag !== "--provider" && flag !== "--models") {
      out.push(arg);
      continue;
    }
    const value = inline ?? args[i + 1];
    if (inline === undefined) i++;
    if (value === undefined) {
      out.push(arg);
      continue;
    }
    let next = value;
    if (flag === "--model") next = point(value);
    else if (flag === "--models") next = value.split(",").map(point).join(",");
    else if (direct.has(value)) {
      moved.push({ from: value, to: GATEWAY_PROVIDER_ID });
      next = GATEWAY_PROVIDER_ID;
    }
    out.push(...(inline === undefined ? [flag, next] : [`${flag}=${next}`]));
  }
  return { args: out, moved };
}

/** The saved default model, when it names a direct provider. */
export function savedDirectModel(agentDir: string, direct: Set<string>): string | undefined {
  const settings = readSettings(agentDir);
  const provider = settings?.defaultProvider;
  const model = settings?.defaultModel;
  return typeof provider === "string" && direct.has(provider) && typeof model === "string" && model !== "" ? model : undefined;
}

/**
 * Save `alias` on the gateway as the default model, and move scoped-model patterns (`enabledModels`)
 * off direct providers.
 */
export function repointSavedModel(agentDir: string, alias: string, direct: Set<string>): void {
  const settings = readSettings(agentDir);
  if (settings === undefined) return;
  settings.defaultProvider = GATEWAY_PROVIDER_ID;
  settings.defaultModel = alias;
  if (Array.isArray(settings.enabledModels)) {
    settings.enabledModels = settings.enabledModels.map((pattern) => {
      if (typeof pattern !== "string") return pattern;
      const ref = directModelRef(pattern, direct);
      return ref === undefined ? pattern : `${GATEWAY_PROVIDER_ID}/${ref.id}`;
    });
  }
  writeFileSync(join(agentDir, "settings.json"), `${JSON.stringify(settings, null, 2)}\n`);
}

function readSettings(agentDir: string): Record<string, unknown> & { enabledModels?: unknown[] } | undefined {
  try {
    const parsed: unknown = JSON.parse(readFileSync(join(agentDir, "settings.json"), "utf8"));
    return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed) ? (parsed as Record<string, unknown>) : undefined;
  } catch {
    return undefined;
  }
}
