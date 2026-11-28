/**
 * The model a CLI invocation starts on.
 *
 * When the user names no model and is not resuming a session, the start model is, in order:
 * `EXP_CODE_MODEL` when set and listed (for this run only, never saved); else the model saved in exp-code's
 * settings, which every explicit choice in the terminal UI writes (`/model`, Ctrl+P, see
 * pi-patches.ts); else `experiential-large-model` when the gateway lists it; else the first alias
 * it lists. A saved gateway model the gateway no longer lists falls back to that default with a
 * one-line note, and stays saved so it is used again once it is listed; an `EXP_CODE_MODEL` the
 * gateway does not list is passed over the same way, with a note. `--model` always wins and is
 * never saved either.
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

/** Whether this invocation picks its own start model (it names none and resumes no session). */
export function wantsDefaultModel(args: string[], parsed: Args): boolean {
  if (args[0] !== undefined && SUBCOMMANDS.has(args[0])) return false;
  if (parsed.help || parsed.version || parsed.export !== undefined || parsed.listModels !== undefined) return false;
  if (parsed.model !== undefined || parsed.provider !== undefined || (parsed.models?.length ?? 0) > 0) return false;
  if (parsed.continue || parsed.resume || parsed.session !== undefined || parsed.sessionId !== undefined || parsed.fork !== undefined) return false;
  return true;
}

/** The model saved in exp-code's global settings, if any. */
export function savedDefaultModel(agentDir: string): { provider: string | undefined; id: string } | undefined {
  const settings = readSettings(agentDir);
  const id = settings?.defaultModel;
  if (typeof id !== "string" || id === "") return undefined;
  return { provider: typeof settings?.defaultProvider === "string" ? settings.defaultProvider : undefined, id };
}

/** What to start on: a gateway alias to pass as `--model` (none leaves Pi on the saved model), and a note to print. */
export interface StartModel {
  id?: string;
  note?: string;
}

/**
 * Choose the start model (see the module comment). `listed` is the gateway's aliases, undefined
 * when the listing failed: a saved model is then left to Pi rather than replaced blind.
 */
export function startModel(listed: string[] | undefined, configured: string | undefined, saved: { provider: string | undefined; id: string } | undefined): StartModel {
  const wanted = configuredModelId(configured);
  // EXP_CODE_MODEL is checked against the listing as `--model` is: an alias the gateway does not
  // list would otherwise be sent as a custom model id and refused by the gateway (403) on every
  // call. It falls back the way a saved model does, to the saved model or the default, with a note.
  if (wanted !== undefined && (listed === undefined || isListed(wanted, listed))) return { id: wanted };
  const rejected = wanted === undefined ? undefined : `the gateway does not list ${wanted} (EXP_CODE_MODEL)`;
  const withNote = (start: StartModel, using: string | undefined): StartModel =>
    rejected === undefined ? start : { ...start, note: start.note ?? `${rejected}; using ${using ?? "the default model"}` };
  // A model a user's own extension provides is not the gateway's to judge.
  if (saved !== undefined && saved.provider !== undefined && saved.provider !== GATEWAY_PROVIDER_ID) return withNote({}, `${saved.provider}/${saved.id}, your saved model`);
  if (saved !== undefined) {
    if (listed === undefined || isListed(saved.id, listed)) return withNote({}, `${saved.id}, your saved model`);
    const fallback = defaultModelId(listed, undefined);
    if (fallback === undefined) return {};
    return { id: fallback, note: rejected !== undefined ? `${rejected}, nor ${saved.id}, your saved model; using ${fallback}` : `the gateway no longer lists ${saved.id}, your saved model; using ${fallback}` };
  }
  if (listed === undefined) return {};
  const fallback = defaultModelId(listed, undefined);
  return withNote({ id: fallback }, fallback);
}

/** The model EXP_CODE_MODEL names, without a leading `experiential/`; undefined when unset or blank. */
export function configuredModelId(configured: string | undefined): string | undefined {
  const wanted = configured?.trim().replace(new RegExp(`^${GATEWAY_PROVIDER_ID}/`), "");
  return wanted ? wanted : undefined;
}

/** Whether the gateway lists `id`, with or without a `:level` thinking suffix. */
function isListed(id: string, listed: string[]): boolean {
  return listed.includes(id) || listed.includes(id.replace(/:[a-z]+$/, ""));
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
  // An EXP_CODE_MODEL the gateway does not list is no fallback either.
  const wanted = configuredModelId(configured);
  return defaultModelId(listed, wanted !== undefined && isListed(wanted, listed) ? wanted : undefined) ?? id;
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
