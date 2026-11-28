import { readFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  cachePath,
  checkForUpdate,
  compareVersions,
  detectInstallMethod,
  fetchLatestVersion,
  isNewer,
  parseUpdateArgs,
  selfUpdate,
  updateCheckAllowed,
  updateCommand
} from "./update.js";

let home: string;
beforeEach(async () => {
  home = await mkdtemp(join(tmpdir(), "exp-code-update-"));
});
afterEach(() => rm(home, { recursive: true, force: true }));

const DAY = 24 * 60 * 60 * 1000;
const GLOBAL = "/usr/local/lib/node_modules/@experiential-labs/exp-code";

/** A fetch that answers the dist-tags request with `latest`, counting calls. */
function registry(latest: string | Error) {
  const calls: string[] = [];
  const fetchImpl = (async (url: string) => {
    calls.push(String(url));
    if (latest instanceof Error) throw latest;
    return new Response(JSON.stringify({ latest }), { status: 200 });
  }) as unknown as typeof fetch;
  return { calls, fetchImpl };
}

describe("version order", () => {
  it("compares semantic versions numerically, with prereleases before their release", () => {
    expect(isNewer("0.1.10", "0.1.9")).toBe(true);
    expect(isNewer("0.1.9", "0.1.10")).toBe(false);
    expect(isNewer("1.0.0", "0.99.99")).toBe(true);
    expect(compareVersions("0.1.9", "0.1.9")).toBe(0);
    expect(isNewer("0.2.0", "0.2.0-rc.1")).toBe(true);
    expect(isNewer("0.2.0-rc.2", "0.2.0-rc.10")).toBe(false);
    expect(isNewer("0.2.0-rc.1", "0.2.0-beta")).toBe(true);
    expect(isNewer("0.2.0-rc", "0.2.0-rc.1")).toBe(false);
    expect(compareVersions("v1.2.3+build", "1.2.3")).toBe(0);
    expect(compareVersions("latest", "1.0.0")).toBeUndefined();
    expect(isNewer("garbage", "1.0.0")).toBe(false);
  });
});

describe("install method", () => {
  it("is read from the package path, and from the launcher's variable", () => {
    expect(detectInstallMethod({}, GLOBAL)).toEqual({ kind: "npm-global", prefix: "/usr/local" });
    expect(detectInstallMethod({}, "/opt/homebrew/lib/node_modules/@experiential-labs/exp-code/")).toEqual({ kind: "npm-global", prefix: "/opt/homebrew" });
    expect(detectInstallMethod({}, "C:\\Users\\me\\AppData\\Roaming\\npm\\node_modules\\@experiential-labs\\exp-code")).toEqual({ kind: "npm-global", prefix: "C:/Users/me/AppData/Roaming/npm" });
    expect(detectInstallMethod({}, "/Users/me/.npm/_npx/1a2b/node_modules/@experiential-labs/exp-code")).toEqual({ kind: "npx" });
    expect(detectInstallMethod({}, "/Users/me/Library/pnpm/global/5/node_modules/@experiential-labs/exp-code")).toEqual({ kind: "pnpm-global" });
    expect(detectInstallMethod({}, "/Users/me/.bun/install/global/node_modules/@experiential-labs/exp-code")).toEqual({ kind: "bun-global" });
    expect(detectInstallMethod({}, "/work/app/node_modules/@experiential-labs/exp-code")).toEqual({ kind: "unknown" });
    expect(detectInstallMethod({}, "/src/experiential/agent")).toEqual({ kind: "source", dir: "/src/experiential/agent" });
    // `exp code` runs exp-code through npx but upgrades the experiential package.
    expect(detectInstallMethod({ EXP_CODE_LAUNCHER: "uv-tool" }, "/Users/me/.npm/_npx/1a2b/node_modules/@experiential-labs/exp-code")).toEqual({ kind: "launcher", tool: "uv-tool" });
    expect(detectInstallMethod({ EXP_CODE_LAUNCHER: "bogus" }, GLOBAL)).toMatchObject({ kind: "npm-global" });
  });

  it("gives each method its update command, and runs only those that are safe to run", () => {
    expect(updateCommand({ kind: "npm-global" })).toEqual({ argv: ["npm", "install", "-g", "@experiential-labs/exp-code@latest"], display: "npm install -g @experiential-labs/exp-code@latest", runnable: true });
    // It updates the install it runs from, even when another npm prefix is first on PATH.
    expect(updateCommand({ kind: "npm-global", prefix: "/usr/local" })).toMatchObject({
      argv: ["npm", "install", "-g", "--prefix", "/usr/local", "@experiential-labs/exp-code@latest"],
      display: "npm install -g @experiential-labs/exp-code@latest"
    });
    expect(updateCommand({ kind: "npx" })).toMatchObject({ display: "npx @experiential-labs/exp-code@latest", runnable: false });
    expect(updateCommand({ kind: "launcher", tool: "uv-tool" })).toMatchObject({ display: "uv tool upgrade experiential", runnable: true });
    expect(updateCommand({ kind: "launcher", tool: "pipx" })).toMatchObject({ display: "pipx upgrade experiential", runnable: true });
    expect(updateCommand({ kind: "launcher", tool: "pip" })).toMatchObject({ display: "pip install -U experiential", runnable: false });
    expect(updateCommand({ kind: "pnpm-global" }).display).toBe("pnpm add -g @experiential-labs/exp-code@latest");
    expect(updateCommand({ kind: "source", dir: "/src/agent" })).toMatchObject({ runnable: false });
    expect(updateCommand({ kind: "unknown" }).runnable).toBe(false);
  });
});

describe("update check", () => {
  const base = { home, currentVersion: "0.1.9", interactive: true, dir: GLOBAL };

  it("runs only for an interactive terminal start, outside CI, online, and not opted out", () => {
    expect(updateCheckAllowed({ env: {}, interactive: true })).toBe(true);
    expect(updateCheckAllowed({ env: {}, interactive: false })).toBe(false);
    expect(updateCheckAllowed({ env: {}, interactive: true, offline: true })).toBe(false);
    expect(updateCheckAllowed({ env: { EXP_CODE_OFFLINE: "1" }, interactive: true })).toBe(false);
    expect(updateCheckAllowed({ env: { EXP_CODE_NO_UPDATE_CHECK: "1" }, interactive: true })).toBe(false);
    expect(updateCheckAllowed({ env: { EXP_CODE_NO_UPDATE_CHECK: "true" }, interactive: true })).toBe(false);
    expect(updateCheckAllowed({ env: { CI: "true" }, interactive: true })).toBe(false);
    expect(updateCheckAllowed({ env: { CI: "github-actions" }, interactive: true })).toBe(false);
    expect(updateCheckAllowed({ env: { CI: "false" }, interactive: true })).toBe(true);
  });

  it("shows one line naming the new version and the install's update command", async () => {
    const { fetchImpl, calls } = registry("0.1.10");
    const line = await checkForUpdate({ ...base, home, env: {}, fetch: fetchImpl, now: () => 1000 });
    expect(line).toBe("exp-code 0.1.10 is available (you have 0.1.9). Update: npm install -g @experiential-labs/exp-code@latest");
    expect(calls).toEqual(["https://registry.npmjs.org/-/package/@experiential-labs%2Fexp-code/dist-tags"]);
  });

  it("asks the registry at most once a day and keeps the answer", async () => {
    const { fetchImpl, calls } = registry("0.1.10");
    await checkForUpdate({ ...base, home, env: {}, fetch: fetchImpl, now: () => 1000 });
    expect(JSON.parse(readFileSync(cachePath(home), "utf8"))).toEqual({ checkedAt: 1000, latest: "0.1.10" });
    const cached = await checkForUpdate({ ...base, home, env: {}, fetch: fetchImpl, now: () => 1000 + DAY - 1 });
    expect(cached).toContain("0.1.10 is available");
    expect(calls).toHaveLength(1);
    await checkForUpdate({ ...base, home, env: {}, fetch: fetchImpl, now: () => 1000 + DAY });
    expect(calls).toHaveLength(2);
  });

  it("is silent when up to date, on failures, and when gated, and counts a failed check as the day's check", async () => {
    expect(await checkForUpdate({ ...base, home, env: {}, fetch: registry("0.1.9").fetchImpl, now: () => 1 })).toBeUndefined();
    const failing = registry(new Error("offline"));
    expect(await checkForUpdate({ ...base, home, env: {}, fetch: failing.fetchImpl, now: () => 2 * DAY })).toBeUndefined();
    expect(await checkForUpdate({ ...base, home, env: {}, fetch: failing.fetchImpl, now: () => 2 * DAY + 5 })).toBeUndefined();
    expect(failing.calls).toHaveLength(1);
    const gated = registry("9.9.9");
    expect(await checkForUpdate({ ...base, home, env: { CI: "1" }, fetch: gated.fetchImpl })).toBeUndefined();
    expect(await checkForUpdate({ ...base, home, interactive: false, env: {}, fetch: gated.fetchImpl })).toBeUndefined();
    expect(gated.calls).toEqual([]);
  });

  it("reads another registry when EXP_CODE_NPM_REGISTRY is set, and rejects a non-version answer", async () => {
    const { fetchImpl, calls } = registry("not-a-version");
    expect(await fetchLatestVersion("https://npm.example/", fetchImpl)).toBeUndefined();
    expect(calls).toEqual(["https://npm.example/-/package/@experiential-labs%2Fexp-code/dist-tags"]);
  });
});

describe("exp-code update", () => {
  it("updates exp-code by default and hands only extension and model updates to Pi", () => {
    expect(parseUpdateArgs([])).toEqual({ self: true, help: false });
    expect(parseUpdateArgs(["self"])).toEqual({ self: true, help: false });
    expect(parseUpdateArgs(["--self"])).toEqual({ self: true, help: false });
    expect(parseUpdateArgs(["--all", "-a"])).toEqual({ self: true, packageArgs: ["update", "--extensions", "-a"], help: false });
    expect(parseUpdateArgs(["--extensions"])).toEqual({ self: false, packageArgs: ["update", "--extensions"], help: false });
    expect(parseUpdateArgs(["--models"])).toEqual({ self: false, packageArgs: ["update", "--models"], help: false });
    expect(parseUpdateArgs(["npm:foo"])).toEqual({ self: false, packageArgs: ["update", "npm:foo"], help: false });
    expect(parseUpdateArgs(["--help"]).help).toBe(true);
  });

  it("runs the install's command when it is safe to, and prints it otherwise", async () => {
    const ran: string[][] = [];
    const log: string[] = [];
    const run = (argv: string[]) => {
      ran.push(argv);
      return 0;
    };
    const fetchImpl = registry("0.1.10").fetchImpl;
    expect(await selfUpdate({ env: {}, currentVersion: "0.1.9", dir: GLOBAL, fetch: fetchImpl, run, log: (line) => log.push(line) })).toBe(0);
    expect(ran).toEqual([["npm", "install", "-g", "--prefix", "/usr/local", "@experiential-labs/exp-code@latest"]]);
    expect(log[0]).toBe("Updating to exp-code 0.1.10: npm install -g --prefix /usr/local @experiential-labs/exp-code@latest");

    ran.length = 0;
    log.length = 0;
    await selfUpdate({ env: {}, currentVersion: "0.1.9", dir: "/Users/me/.npm/_npx/1/node_modules/@experiential-labs/exp-code", fetch: fetchImpl, run, log: (line) => log.push(line) });
    expect(ran).toEqual([]);
    expect(log[0]).toBe("To update to exp-code 0.1.10 (you have 0.1.9), run: npx @experiential-labs/exp-code@latest");

    log.length = 0;
    await selfUpdate({ env: { EXP_CODE_LAUNCHER: "uv-tool" }, currentVersion: "0.1.9", dir: GLOBAL, fetch: fetchImpl, run, log: (line) => log.push(line) });
    expect(ran).toEqual([["uv", "tool", "upgrade", "experiential"]]);
  });

  it("does nothing when already on the latest release, and reports a failed command", async () => {
    const log: string[] = [];
    const run = () => 3;
    expect(await selfUpdate({ env: {}, currentVersion: "0.1.10", dir: GLOBAL, fetch: registry("0.1.10").fetchImpl, run, log: (line) => log.push(line) })).toBe(0);
    expect(log).toEqual(["exp-code 0.1.10 is the latest release."]);
    log.length = 0;
    expect(await selfUpdate({ env: {}, currentVersion: "0.1.9", dir: GLOBAL, fetch: registry(new Error("down")).fetchImpl, run, log: (line) => log.push(line) })).toBe(3);
    expect(log.at(-1)).toBe("The update failed (exit code 3). Run it yourself: npm install -g --prefix /usr/local @experiential-labs/exp-code@latest");
  });
});
