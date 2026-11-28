/**
 * exp-code's own releases: the startup update notice and `exp-code update`.
 *
 * The interactive CLI asks the npm registry for the `latest` dist-tag of
 * `@experiential-labs/exp-code` at most once a day, in the background with a short timeout, and
 * shows one line when a newer release exists. The answer is cached in `<home>/cache/update-check.json`.
 * Any failure is silent. There is no check in the headless runner, in CI, without a terminal, when
 * offline, or with `EXP_CODE_NO_UPDATE_CHECK=1`.
 *
 * The update command matches how exp-code was installed: a global npm, pnpm or bun install, a run
 * through `npx`, the `exp code` launcher (which pins exp-code to the installed `experiential`
 * Python package and says how that package was installed in `EXP_CODE_LAUNCHER`), or a source
 * checkout.
 */

import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { trimTrailingSlashes } from "../gateway/config.js";
import { envFlag } from "./environment.js";

export const PACKAGE_NAME = "@experiential-labs/exp-code";
export const DEFAULT_REGISTRY = "https://registry.npmjs.org";
const CHECK_INTERVAL_MS = 24 * 60 * 60 * 1000;
const CHECK_TIMEOUT_MS = 3000;

// ---------------------------------------------------------------------------
// Versions
// ---------------------------------------------------------------------------

interface ParsedVersion {
  core: [number, number, number];
  prerelease: (number | string)[];
}

function parseVersion(version: string): ParsedVersion | undefined {
  const match = /^v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$/.exec(version.trim());
  if (match === null) return undefined;
  return {
    core: [Number(match[1]), Number(match[2]), Number(match[3])],
    prerelease: match[4] === undefined ? [] : match[4].split(".").map((part) => (/^\d+$/.test(part) ? Number(part) : part))
  };
}

/** Semantic-version order of two versions: negative, zero or positive; undefined when either is not a version. */
export function compareVersions(a: string, b: string): number | undefined {
  const left = parseVersion(a);
  const right = parseVersion(b);
  if (left === undefined || right === undefined) return undefined;
  for (let i = 0; i < 3; i++) if (left.core[i] !== right.core[i]) return left.core[i]! - right.core[i]!;
  // A release sorts after its prereleases.
  if (left.prerelease.length === 0 || right.prerelease.length === 0) return right.prerelease.length - left.prerelease.length;
  for (let i = 0; i < Math.max(left.prerelease.length, right.prerelease.length); i++) {
    const x = left.prerelease[i];
    const y = right.prerelease[i];
    if (x === undefined) return -1;
    if (y === undefined) return 1;
    if (x === y) continue;
    if (typeof x === "number" && typeof y === "number") return x - y;
    if (typeof x === "number") return -1;
    if (typeof y === "number") return 1;
    return x < y ? -1 : 1;
  }
  return 0;
}

/** Whether `latest` is a newer version than `current`. */
export function isNewer(latest: string, current: string): boolean {
  return (compareVersions(latest, current) ?? 0) > 0;
}

// ---------------------------------------------------------------------------
// Install method
// ---------------------------------------------------------------------------

export type InstallMethod =
  | { kind: "npm-global"; prefix?: string }
  | { kind: "pnpm-global" }
  | { kind: "bun-global" }
  | { kind: "npx" }
  | { kind: "launcher"; tool: "uv-tool" | "pipx" | "pip" }
  | { kind: "source"; dir: string }
  | { kind: "unknown" };

/** The directory this package runs from (the one holding its package.json). */
export function packageDir(): string {
  return fileURLToPath(new URL("../..", import.meta.url));
}

/** How this copy of exp-code was installed, from the launcher's variable and the package path. */
export function detectInstallMethod(env: NodeJS.ProcessEnv, dir: string = packageDir()): InstallMethod {
  const launcher = env.EXP_CODE_LAUNCHER;
  if (launcher === "uv-tool" || launcher === "pipx" || launcher === "pip") return { kind: "launcher", tool: launcher };
  const path = dir.replaceAll("\\", "/").replace(/\/+$/, "");
  if (!path.endsWith("/node_modules/@experiential-labs/exp-code")) return { kind: "source", dir };
  if (path.includes("/_npx/")) return { kind: "npx" };
  if (path.includes("/pnpm/global/") || path.includes("/.pnpm/")) return { kind: "pnpm-global" };
  if (path.includes("/.bun/install/global/")) return { kind: "bun-global" };
  // npm's global prefix: <prefix>/lib/node_modules on Unix, <prefix>/node_modules on Windows.
  const unix = /^(.*)\/lib\/node_modules\/@experiential-labs\/exp-code$/.exec(path);
  if (unix !== null) return { kind: "npm-global", prefix: unix[1] };
  const windows = /^(.*\/npm)\/node_modules\/@experiential-labs\/exp-code$/.exec(path);
  if (windows !== null) return { kind: "npm-global", prefix: windows[1] };
  return { kind: "unknown" };
}

/** A command exp-code can run for the user, and the text to show for it. */
export interface UpdateCommand {
  /** What `exp-code update` runs (an npm global update names the install's own prefix). */
  argv: string[];
  /** The command as the update notice shows it. */
  display: string;
  /** Whether `exp-code update` runs it; otherwise it only prints it. */
  runnable: boolean;
}

function command(argv: string[], runnable: boolean): UpdateCommand {
  return { argv, display: argv.join(" "), runnable };
}

/** The command that updates exp-code for an install method. */
export function updateCommand(method: InstallMethod): UpdateCommand {
  const latest = `${PACKAGE_NAME}@latest`;
  switch (method.kind) {
    case "npm-global": {
      const shown = command(["npm", "install", "-g", latest], true);
      return method.prefix === undefined ? shown : { ...shown, argv: ["npm", "install", "-g", "--prefix", method.prefix, latest] };
    }
    case "pnpm-global":
      return command(["pnpm", "add", "-g", latest], true);
    case "bun-global":
      return command(["bun", "add", "-g", latest], true);
    case "npx":
      return command(["npx", latest], false);
    case "launcher":
      switch (method.tool) {
        case "uv-tool":
          return command(["uv", "tool", "upgrade", "experiential"], true);
        case "pipx":
          return command(["pipx", "upgrade", "experiential"], true);
        case "pip":
          return command(["pip", "install", "-U", "experiential"], false);
      }
      break;
    case "source":
      return { argv: [], display: `git pull && npm ci && npm run build (in ${method.dir})`, runnable: false };
    case "unknown":
      return command(["npm", "install", "-g", latest], false);
  }
  throw new Error("unreachable");
}

// ---------------------------------------------------------------------------
// The daily check
// ---------------------------------------------------------------------------

interface CheckCache {
  checkedAt: number;
  latest?: string;
}

export function cachePath(home: string): string {
  return join(home, "cache", "update-check.json");
}

function readCache(path: string): CheckCache | undefined {
  try {
    const parsed = JSON.parse(readFileSync(path, "utf8")) as { checkedAt?: unknown; latest?: unknown };
    if (typeof parsed.checkedAt !== "number") return undefined;
    return { checkedAt: parsed.checkedAt, latest: typeof parsed.latest === "string" ? parsed.latest : undefined };
  } catch {
    return undefined;
  }
}

function writeCache(path: string, cache: CheckCache): void {
  try {
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, `${JSON.stringify(cache)}\n`);
  } catch {
    // An unwritable cache only means another check next start.
  }
}

/** The registry's `latest` version of exp-code. */
export async function fetchLatestVersion(registry: string, fetchImpl: typeof fetch = fetch, timeoutMs = CHECK_TIMEOUT_MS): Promise<string | undefined> {
  const url = `${trimTrailingSlashes(registry)}/-/package/${PACKAGE_NAME.replaceAll("/", "%2F")}/dist-tags`;
  const response = await fetchImpl(url, { headers: { accept: "application/json" }, signal: AbortSignal.timeout(timeoutMs) });
  if (!response.ok) return undefined;
  const latest = ((await response.json()) as { latest?: unknown }).latest;
  return typeof latest === "string" && parseVersion(latest) !== undefined ? latest : undefined;
}

export interface UpdateCheckOptions {
  env: NodeJS.ProcessEnv;
  home: string;
  currentVersion: string;
  /** Whether this run opens the interactive terminal UI on a real terminal. */
  interactive: boolean;
  offline?: boolean;
  fetch?: typeof fetch;
  now?: () => number;
  dir?: string;
}

/** Whether this run may check for a newer release at all. */
export function updateCheckAllowed(options: Pick<UpdateCheckOptions, "env" | "interactive" | "offline">): boolean {
  const { env } = options;
  if (!options.interactive || options.offline === true) return false;
  if (envFlag(env.EXP_CODE_NO_UPDATE_CHECK) || envFlag(env.EXP_CODE_OFFLINE)) return false;
  // CI services set CI (to true, 1, or a service name); an explicit false does not count.
  const ci = env.CI?.trim().toLowerCase();
  if (ci !== undefined && ci !== "" && ci !== "0" && ci !== "false") return false;
  return true;
}

/** The notice line for a newer release. */
export function updateNotice(latest: string, current: string, method: InstallMethod): string {
  return `exp-code ${latest} is available (you have ${current}). Update: ${updateCommand(method).display}`;
}

/**
 * Check for a newer release (at most once a day) and resolve to the notice line, or undefined.
 * Never rejects.
 */
export async function checkForUpdate(options: UpdateCheckOptions): Promise<string | undefined> {
  try {
    if (!updateCheckAllowed(options)) return undefined;
    const now = (options.now ?? Date.now)();
    const path = cachePath(options.home);
    let cache = readCache(path);
    if (cache === undefined || now - cache.checkedAt >= CHECK_INTERVAL_MS || now < cache.checkedAt) {
      const registry = options.env.EXP_CODE_NPM_REGISTRY?.trim() || DEFAULT_REGISTRY;
      let latest: string | undefined;
      try {
        latest = await fetchLatestVersion(registry, options.fetch);
      } catch {
        latest = undefined;
      }
      // A failed check still counts as today's check; the last answer stays.
      cache = { checkedAt: now, latest: latest ?? cache?.latest };
      writeCache(path, cache);
    }
    const latest = cache.latest;
    if (latest === undefined || !isNewer(latest, options.currentVersion)) return undefined;
    return updateNotice(latest, options.currentVersion, detectInstallMethod(options.env, options.dir));
  } catch {
    return undefined;
  }
}

// ---------------------------------------------------------------------------
// `exp-code update`
// ---------------------------------------------------------------------------

/** What `exp-code update ...` asks for: exp-code itself, then optionally Pi's package update. */
export interface UpdateRequest {
  self: boolean;
  /** Arguments for the extension/model update, or undefined when there is none. */
  packageArgs?: string[];
  help: boolean;
}

/**
 * Split `exp-code update` arguments. No target, `self` or `--self` update exp-code; `--all`
 * updates exp-code and then the installed extensions; anything else (`<source>`, `--extensions`,
 * `--extension <source>`, `--models`) is an extension or model update.
 */
export function parseUpdateArgs(args: string[]): UpdateRequest {
  if (args.includes("--help") || args.includes("-h")) return { self: false, help: true };
  const rest = args.filter((arg) => arg !== "self" && arg !== "--self" && arg !== "--all" && arg !== "--force");
  const shared: string[] = rest.filter((arg) => arg === "-a" || arg === "--approve" || arg === "-na" || arg === "--no-approve");
  const targets = rest.filter((arg) => !shared.includes(arg));
  if (args.includes("--all")) return { self: true, packageArgs: ["update", "--extensions", ...shared], help: false };
  if (targets.length === 0) return { self: true, help: false };
  return { self: args.includes("self") || args.includes("--self"), packageArgs: ["update", ...rest], help: false };
}

export const UPDATE_HELP = `Usage:
  exp-code update [source|self] [--self|--extensions|--models|--all] [--extension <source>] [--approve|--no-approve]

Update exp-code, installed extensions, or model catalogs.

Options:
  --self                  Update exp-code only (default when no target is given)
  --extensions            Update installed extensions only
  --models                Refresh model catalogs only
  --all                   Update exp-code and installed extensions
  --extension <source>    Update one extension only
  -a, --approve           Trust project-local files for this command
  -na, --no-approve       Ignore project-local files for this command

exp-code updates itself the way it was installed: npm, pnpm or bun global installs run the
package manager, \`exp code\` upgrades the experiential Python package (which pins exp-code), and
npx runs and source checkouts print what to do.
`;

export interface SelfUpdateOptions {
  env: NodeJS.ProcessEnv;
  currentVersion: string;
  dir?: string;
  fetch?: typeof fetch;
  run?: (argv: string[]) => number;
  log?: (line: string) => void;
}

function runCommand(argv: string[]): number {
  const [file, ...args] = argv;
  if (file === undefined) return 1;
  const result = spawnSync(file, args, { stdio: "inherit", shell: process.platform === "win32" });
  return result.status ?? 1;
}

/** Update exp-code itself; resolves to the exit code. */
export async function selfUpdate(options: SelfUpdateOptions): Promise<number> {
  const log = options.log ?? ((line) => console.log(line));
  const method = detectInstallMethod(options.env, options.dir);
  const update = updateCommand(method);
  let latest: string | undefined;
  try {
    latest = await fetchLatestVersion(options.env.EXP_CODE_NPM_REGISTRY?.trim() || DEFAULT_REGISTRY, options.fetch);
  } catch {
    latest = undefined;
  }
  if (latest !== undefined && !isNewer(latest, options.currentVersion)) {
    log(`exp-code ${options.currentVersion} is the latest release.`);
    return 0;
  }
  const target = latest === undefined ? "the latest exp-code" : `exp-code ${latest}`;
  if (!update.runnable) {
    log(`To update to ${target} (you have ${options.currentVersion}), run: ${update.display}`);
    if (method.kind === "launcher") log("The experiential package pins the exp-code release `exp code` runs.");
    return 0;
  }
  const exact = update.argv.join(" ");
  log(`Updating to ${target}: ${exact}`);
  const code = (options.run ?? runCommand)(update.argv);
  if (code !== 0) log(`The update failed (exit code ${code}). Run it yourself: ${exact}`);
  return code;
}
