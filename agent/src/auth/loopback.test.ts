import { afterEach, describe, expect, it } from "vitest";
import { authorizeUrl, defaultKeyName, KEY_NAME_MAX_LENGTH, LoopbackLogin, newState, PLATFORM_DEFAULT_URL, platformUrl } from "./loopback.js";

const open: LoopbackLogin[] = [];

async function started(state = "s_expected-1"): Promise<LoopbackLogin> {
  const login = new LoopbackLogin("https://platform.example/", state);
  open.push(login);
  await login.start();
  return login;
}

function callback(login: LoopbackLogin, path: string): Promise<Response> {
  return fetch(`http://127.0.0.1:${login.port}${path}`, { redirect: "manual" });
}

afterEach(async () => {
  for (const login of open.splice(0)) await login.close();
});

describe("approval URL", () => {
  it("carries the state, port and key name on the platform's /cli/auth", () => {
    const url = new URL(authorizeUrl("https://platform.example/", { state: "abc", port: 8321, keyName: "exp-code (my mac)" }));
    expect(url.origin + url.pathname).toBe("https://platform.example/cli/auth");
    expect(Object.fromEntries(url.searchParams)).toEqual({ state: "abc", port: "8321", name: "exp-code (my mac)" });
  });

  it("omits the port for a paste-only login", () => {
    expect(new URL(authorizeUrl(PLATFORM_DEFAULT_URL, { state: "abc", keyName: "k" })).searchParams.has("port")).toBe(false);
  });

  it("reads the platform origin override", () => {
    expect(platformUrl({})).toBe(PLATFORM_DEFAULT_URL);
    expect(platformUrl({ EXP_PLATFORM_URL: " https://staging.example/ " })).toBe("https://staging.example");
  });

  it("names the key after the host, within the platform's 80 characters", () => {
    expect(defaultKeyName("mac-studio")).toBe("exp-code (mac-studio)");
    expect(defaultKeyName("")).toBe("exp-code");
    const long = defaultKeyName("h".repeat(200));
    expect(long).toHaveLength(KEY_NAME_MAX_LENGTH);
    expect(long.startsWith("exp-code (")).toBe(true);
  });

  it("makes url-safe states the size of token_urlsafe(24)", () => {
    const state = newState();
    expect(state).toMatch(/^[A-Za-z0-9_-]{32}$/);
    expect(newState()).not.toBe(state);
  });
});

describe("loopback listener", () => {
  it("binds loopback and puts its port in the URL", async () => {
    const login = await started();
    expect(new URL(login.url("k")).searchParams.get("port")).toBe(String(login.port));
  });

  it("accepts the matching callback once and hands back the key", async () => {
    const login = await started();
    const waiting = login.wait(5_000);
    const response = await callback(login, "/callback?token=xpl_good&state=s_expected-1");
    expect(response.status).toBe(200);
    expect(await response.text()).toContain("exp-code is logged in");
    await expect(waiting).resolves.toBe("xpl_good");
    // One-shot: a second valid-looking callback is refused and does not replace the key.
    expect((await callback(login, "/callback?token=xpl_other&state=s_expected-1")).status).toBe(400);
    expect(login.received()).toBe("xpl_good");
  });

  it("rejects a mismatched state and a non-xpl token", async () => {
    const login = await started();
    expect((await callback(login, "/callback?token=xpl_good&state=s_expected-2")).status).toBe(400);
    expect((await callback(login, "/callback?token=xpl_good&state=s_expected")).status).toBe(400);
    expect((await callback(login, "/callback?token=xpl_good")).status).toBe(400);
    expect((await callback(login, "/callback?token=sk_good&state=s_expected-1")).status).toBe(400);
    expect(login.received()).toBeNull();
  });

  it("ignores other paths and methods", async () => {
    const login = await started();
    expect((await callback(login, "/?token=xpl_good&state=s_expected-1")).status).toBe(404);
    expect((await callback(login, "/callback/x?token=xpl_good&state=s_expected-1")).status).toBe(404);
    const post = await fetch(`http://127.0.0.1:${login.port}/callback?token=xpl_good&state=s_expected-1`, { method: "POST" });
    expect(post.status).toBe(404);
    expect(login.received()).toBeNull();
  });

  it("times out without a callback, and close settles a pending wait", async () => {
    const login = await started();
    await expect(login.wait(50)).resolves.toBeNull();
    const pending = login.wait(60_000);
    await login.close();
    await expect(pending).resolves.toBeNull();
    await login.close();
  });

  it("stops waiting when aborted", async () => {
    const login = await started();
    const controller = new AbortController();
    const pending = login.wait(60_000, controller.signal);
    controller.abort();
    await expect(pending).resolves.toBeNull();
  });
});
