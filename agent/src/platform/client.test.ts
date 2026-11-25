import { afterEach, describe, expect, it } from "vitest";
import { type FakePlatform, startFakePlatform } from "../testing/fake-platform.js";
import { describeRefusal, PlatformClient, PlatformError, platformOrigin, redactKeys, withQuery } from "./client.js";

let platform: FakePlatform | undefined;
afterEach(async () => {
  await platform?.close();
  platform = undefined;
});

const KEY = "xpl_secret_key_123";

describe("platform client", () => {
  it("derives the platform origin from a hosted gateway only", () => {
    expect(platformOrigin({ kind: "hosted", baseUrl: "https://api.experientiallabs.ai/v1", apiKey: "k" })).toBe("https://api.experientiallabs.ai");
    expect(platformOrigin({ kind: "hosted", baseUrl: "https://gw.example/prefix/v1/", apiKey: "k" })).toBe("https://gw.example/prefix");
    expect(platformOrigin({ kind: "local", baseUrl: "http://127.0.0.1:8000/v1", apiKey: "k" })).toBeUndefined();
    expect(platformOrigin({ kind: "hosted", baseUrl: "https://gw.example/api", apiKey: "k" })).toBeUndefined();
  });

  it("reads whoami once per session and sends the key as a bearer", async () => {
    platform = await startFakePlatform();
    const client = new PlatformClient({ origin: platform.origin, apiKey: KEY });
    const [a, b] = await Promise.all([client.whoami(), client.whoami()]);
    expect(a.org_id).toBe("org-1111");
    expect(b).toBe(a);
    expect(await client.orgPath("/usage/requests")).toBe("/api/orgs/org-1111/usage/requests");
    expect(platform.calls("/api/whoami")).toHaveLength(1);
    expect(platform.requests[0]?.authorization).toBe(`Bearer ${KEY}`);
  });

  it("retries whoami after a failure", async () => {
    let calls = 0;
    platform = await startFakePlatform({
      "GET /api/whoami": () => (++calls === 1 ? { status: 503, body: { error: "busy" } } : { body: { org_id: "org-2" } })
    });
    const client = new PlatformClient({ origin: platform.origin, apiKey: KEY });
    await expect(client.whoami()).rejects.toThrow(/had a problem \(503\): busy/);
    expect((await client.whoami()).org_id).toBe("org-2");
  });

  it("explains each refusal status without echoing the key", async () => {
    const cases: [number, string, RegExp][] = [
      [401, JSON.stringify({ error: "Invalid API key" }), /did not accept this key \(401\).*exp-code login/],
      [403, JSON.stringify({ error: "Provisioning key required" }), /not allowed to read that \(403\): Provisioning key required.*provisioning key/],
      [404, JSON.stringify({ error: "Request not found: r1" }), /Not found \(404\): Request not found: r1/],
      [429, JSON.stringify({ error: { message: "slow down" } }), /rate limited.*slow down.*Retry in about 7 s/],
      [502, "upstream exploded", /had a problem \(502\): upstream exploded. Try again/],
      [422, JSON.stringify({ detail: [{ msg: "limit too big" }] }), /refused the request \(422\): limit too big/]
    ];
    for (const [status, body, pattern] of cases) {
      const error = describeRefusal(status, body, status === 429 ? "7" : null);
      expect(error).toBeInstanceOf(PlatformError);
      expect(error.status).toBe(status);
      expect(error.message).toMatch(pattern);
    }
    expect(describeRefusal(403, JSON.stringify({ error: `key ${KEY} is not allowed` }), null).message).not.toContain(KEY);
    expect(redactKeys("bearer xpladmin_abc-1 and xpl_def")).toBe("bearer [redacted key] and [redacted key]");
  });

  it("surfaces timeouts and unreachable hosts as platform errors", async () => {
    platform = await startFakePlatform({ "GET /slow": { delayMs: 500, body: {} } });
    const slow = new PlatformClient({ origin: platform.origin, apiKey: KEY, timeoutMs: 50 });
    await expect(slow.get("/slow")).rejects.toThrow(/did not answer within/);
    const gone = new PlatformClient({ origin: "http://127.0.0.1:1", apiKey: KEY });
    const error = (await gone.get("/x").catch((caught: unknown) => caught)) as PlatformError;
    expect(error).toBeInstanceOf(PlatformError);
    expect(error.message).toMatch(/Could not reach/);
    expect(error.message).not.toContain(KEY);
  });

  it("builds queries from defined values only", () => {
    expect(withQuery("/p", { a: "1", b: undefined, c: null, d: "", e: true, f: 0 })).toBe("/p?a=1&e=true&f=0");
    expect(withQuery("/p")).toBe("/p");
  });
});
