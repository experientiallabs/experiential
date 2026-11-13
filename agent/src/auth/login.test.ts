import { EventEmitter } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { type FakeGateway, startFakeGateway } from "../testing/fake-gateway.js";
import { readCredentials, saveCredentials } from "./credentials.js";
import { type AuthIo, isHostedKey, login, loginStatus, logout, parseAuthCommand } from "./login.js";
import type { MaskedInput } from "./masked-input.js";

const KEY = "xpl_testkey_0123456789";

/** A fake terminal: `type` delivers bytes once the prompt is reading. */
class FakeTty extends EventEmitter implements MaskedInput {
  isTTY: boolean;
  isRaw = false;
  rawModes: boolean[] = [];
  constructor(isTTY = true) {
    super();
    this.isTTY = isTTY;
  }
  setRawMode(mode: boolean) {
    this.isRaw = mode;
    this.rawModes.push(mode);
  }
  resume() {}
  pause() {}
  type(text: string) {
    this.emit("data", Buffer.from(text));
  }
}

let gateway: FakeGateway;
let home: string;
let output: string;
let errors: string;

function io(overrides: Partial<AuthIo> = {}): AuthIo {
  return {
    env: { EXP_CODE_HOME: join(home, ".exp-code"), EXP_CODE_BASE_URL: gateway.baseUrl, EXP_PLATFORM_URL: "https://platform.example" },
    stdin: new FakeTty(false),
    out: (t) => void (output += t),
    err: (t) => void (errors += t),
    hostname: () => "test-host",
    timeoutMs: 5_000,
    ...overrides
  };
}

/** A browser that approves at once: it follows the approval URL's port back to the listener. */
function approvingBrowser(token = KEY, stateOverride?: string): (url: string) => Promise<boolean> {
  return async (url) => {
    const params = new URL(url).searchParams;
    const callback = new URL(`http://127.0.0.1:${params.get("port")}/callback`);
    callback.searchParams.set("token", token);
    callback.searchParams.set("state", stateOverride ?? params.get("state") ?? "");
    await fetch(callback);
    return true;
  };
}

beforeAll(async () => {
  gateway = await startFakeGateway([], [{ id: "a" }, { id: "b" }]);
});

afterAll(async () => {
  await gateway.close();
});

beforeEach(async () => {
  home = await mkdtemp(join(tmpdir(), "exp-code-login-"));
  output = "";
  errors = "";
});

afterEach(async () => {
  await rm(home, { recursive: true, force: true });
});

describe("exp-code login", () => {
  it("asks for an exp-code key, verifies it on /models, and saves it", async () => {
    let requested = "";
    const options = io({
      openBrowser: async (url) => {
        requested = url;
        return approvingBrowser()(url);
      }
    });
    await expect(login(options, { noBrowser: false })).resolves.toBe(0);
    const url = new URL(requested);
    expect(url.origin + url.pathname).toBe("https://platform.example/cli/auth");
    expect(url.searchParams.get("name")).toBe("exp-code (test-host)");
    expect(output).toContain(requested);
    const check = gateway.requests.filter((r) => r.path === "/v1/models").at(-1)!;
    expect(check.headers.authorization).toBe(`Bearer ${KEY}`);
    const saved = await readCredentials(options.env);
    expect(saved).toMatchObject({ hostedKey: KEY, baseUrl: gateway.baseUrl, keyName: "exp-code (test-host)" });
    expect(output).toContain("Logged in.");
    expect(output).toContain("xpl_testkey…");
    expect(output).toContain("2 models available");
    expect(output).not.toContain(KEY);
  });

  it("does not save a key the gateway rejects", async () => {
    const rejecting: typeof fetch = async () => new Response("{}", { status: 401 });
    const options = io({ openBrowser: approvingBrowser(), fetch: rejecting });
    await expect(login(options, { noBrowser: false })).resolves.toBe(1);
    expect(errors).toContain("rejected");
    expect(errors).toContain("Nothing was saved");
    await expect(readCredentials(options.env)).resolves.toBeNull();
  });

  it("ignores a callback with the wrong state and times out", async () => {
    const options = io({ openBrowser: approvingBrowser(KEY, "forged"), timeoutMs: 200 });
    await expect(login(options, { noBrowser: false })).resolves.toBe(1);
    expect(errors).toContain("timed out");
    await expect(readCredentials(options.env)).resolves.toBeNull();
  });

  it("stops on Ctrl+C while waiting for the browser", async () => {
    const controller = new AbortController();
    const options = io({
      openBrowser: async () => {
        setTimeout(() => controller.abort(), 20);
        return true;
      },
      signal: controller.signal
    });
    await expect(login(options, { noBrowser: false })).resolves.toBe(130);
    expect(errors).toContain("cancelled");
  });

  it("refuses --no-browser without a terminal", async () => {
    await expect(login(io(), { noBrowser: true })).resolves.toBe(1);
    expect(errors).toContain("interactive terminal");
    expect(errors).toContain("EXP_API_KEY");
  });

  it("takes a pasted key with --no-browser, hidden, from a URL without a port", async () => {
    const tty = new FakeTty(true);
    const options = io({ stdin: tty });
    const done = login(options, { noBrowser: true });
    await new Promise((resolve) => setTimeout(resolve, 20));
    const url = new URL(output.match(/https:\/\/platform\.example\S+/)![0]);
    expect(url.searchParams.has("port")).toBe(false);
    expect(url.searchParams.get("state")).toMatch(/^[A-Za-z0-9_-]{32}$/);
    tty.type(`${KEY}\r`);
    await expect(done).resolves.toBe(0);
    expect(tty.rawModes).toEqual([true, false]);
    expect(output).not.toContain(KEY);
    await expect(readCredentials(options.env)).resolves.toMatchObject({ hostedKey: KEY });
  });

  it("treats Ctrl+C at the paste prompt as a cancel and restores the terminal", async () => {
    const tty = new FakeTty(true);
    const done = login(io({ stdin: tty }), { noBrowser: true });
    await new Promise((resolve) => setTimeout(resolve, 20));
    tty.type("xpl_par\u0003");
    await expect(done).resolves.toBe(130);
    expect(tty.isRaw).toBe(false);
  });

  it("lets the browser win while the paste prompt is open when no browser could be opened", async () => {
    const tty = new FakeTty(true);
    const options = io({
      stdin: tty,
      openBrowser: async (url) => {
        setTimeout(() => void approvingBrowser()(url), 30);
        return false;
      }
    });
    await expect(login(options, { noBrowser: false })).resolves.toBe(0);
    expect(output).toContain("paste an existing key");
    expect(tty.rawModes).toEqual([true, false]);
    await expect(readCredentials(options.env)).resolves.toMatchObject({ hostedKey: KEY, keyName: "exp-code (test-host)" });
  });

  it("rejects a pasted string that is not a key", async () => {
    const tty = new FakeTty(true);
    const done = login(io({ stdin: tty }), { noBrowser: true });
    await new Promise((resolve) => setTimeout(resolve, 20));
    tty.type("sk-not-ours\n");
    await expect(done).resolves.toBe(1);
    expect(errors).toContain("start with xpl_");
    expect(isHostedKey(KEY)).toBe(true);
    expect(isHostedKey("xpl_ short")).toBe(false);
  });
});

describe("exp-code logout and status", () => {
  it("logout deletes the saved key and points at revoking it on the platform", async () => {
    const env = { EXP_CODE_HOME: join(home, ".exp-code") };
    await saveCredentials(env, { hostedKey: KEY, savedAt: "t", keyName: "exp-code (h)" });
    await expect(logout({ env, out: (t) => void (output += t) })).resolves.toBe(0);
    await expect(readCredentials(env)).resolves.toBeNull();
    expect(output).toContain("still exists on the Experiential platform");
    expect(output).toContain("API keys page");
    output = "";
    await logout({ env, out: (t) => void (output += t) });
    expect(output).toContain("Not logged in");
  });

  it("status says where the key comes from, masked", async () => {
    const env: NodeJS.ProcessEnv = { EXP_CODE_HOME: join(home, ".exp-code") };
    const out = (t: string) => void (output += t);
    await expect(loginStatus({ env, out })).resolves.toBe(1);
    expect(output).toContain("Not logged in");

    output = "";
    await saveCredentials(env, { hostedKey: KEY, savedAt: "2026-11-13", keyName: "exp-code (h)" });
    await expect(loginStatus({ env, out })).resolves.toBe(0);
    expect(output).toContain("Using the saved login");
    expect(output).toContain("xpl_testkey…");
    expect(output).not.toContain(KEY);

    output = "";
    await loginStatus({ env: { ...env, EXP_API_KEY: "xpl_fromenvironment" }, out });
    expect(output).toContain("Using EXP_API_KEY from the environment: xpl_fromenv…");
    expect(output).toContain("not in use");
    expect(output).not.toContain("xpl_fromenvironment");
  });
});

describe("auth command parsing", () => {
  it("recognizes login, logout and whoami and leaves everything else to Pi", () => {
    expect(parseAuthCommand(["login"])).toEqual({ kind: "login", noBrowser: false });
    expect(parseAuthCommand(["login", "--no-browser"])).toEqual({ kind: "login", noBrowser: true });
    expect(parseAuthCommand(["login", "--status"])).toEqual({ kind: "status" });
    expect(parseAuthCommand(["whoami"])).toEqual({ kind: "status" });
    expect(parseAuthCommand(["logout"])).toEqual({ kind: "logout" });
    expect(parseAuthCommand(["login", "--bogus"])?.kind).toBe("error");
    expect(parseAuthCommand(["login", "--help"])?.kind).toBe("help");
    expect(parseAuthCommand(["-p", "login"])).toBeUndefined();
    expect(parseAuthCommand([])).toBeUndefined();
  });
});
