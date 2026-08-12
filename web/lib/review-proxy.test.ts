import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";

import { afterEach, describe, expect, it, vi } from "vitest";

import { LocalProxyError, proxyLocalReview, validateLocalWebRequest } from "@/lib/review-proxy";

describe("local review proxy boundary", () => {
  const originalApiUrl = process.env.WMO_REVIEW_API_URL;
  const originalWebPort = process.env.WMO_REVIEW_WEB_PORT;

  afterEach(() => {
    vi.unstubAllGlobals();
    restoreEnv("WMO_REVIEW_API_URL", originalApiUrl);
    restoreEnv("WMO_REVIEW_WEB_PORT", originalWebPort);
  });

  it("accepts only the exact loopback Host and same-origin browser headers", () => {
    expect(() =>
      validateLocalWebRequest(
        request({
          host: "127.0.0.1:3000",
          origin: "http://127.0.0.1:3000",
          referer: "http://127.0.0.1:3000/review"
        }),
        3000
      )
    ).not.toThrow();
    expect(() =>
      validateLocalWebRequest(
        request({ host: "[::1]:3000", origin: "http://[::1]:3000" }),
        3000
      )
    ).not.toThrow();

    const rejectedHeaders: Array<Record<string, string>> = [
      { host: "evil.example:3000" },
      { host: "192.168.1.20:3000" },
      { host: "127.0.0.1:3001" },
      { host: "127.0.0.1:3000", origin: "http://evil.example:3000" },
      { host: "127.0.0.1:3000", origin: "http://localhost:3000" },
      { host: "127.0.0.1:3000", referer: "http://127.0.0.1:3001/review" }
    ];
    for (const headers of rejectedHeaders) {
      expect(() => validateLocalWebRequest(request(headers), 3000)).toThrow(LocalProxyError);
    }
  });

  it("proxies only allowlisted paths to a configured loopback API without forwarding origin", async () => {
    process.env.WMO_REVIEW_API_URL = "http://127.0.0.1:8017";
    process.env.WMO_REVIEW_WEB_PORT = "3000";
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) =>
      new Response(JSON.stringify({ project_id: "support" }), {
        headers: { "Content-Type": "application/json" }
      })
    );
    vi.stubGlobal("fetch", fetchMock);
    const response = await proxyLocalReview(
      request({ host: "127.0.0.1:3000", origin: "http://127.0.0.1:3000" }),
      ["api", "review", "calibration", "report-1", "approve"],
      "POST"
    );

    expect(response.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledOnce();
    const [target, init] = fetchMock.mock.calls[0];
    expect(String(target)).toBe("http://127.0.0.1:8017/api/review/calibration/report-1/approve");
    expect(init?.headers).toEqual({ "Content-Type": "application/json" });
    await expect(
      proxyLocalReview(
        request({ host: "127.0.0.1:3000" }),
        ["api", "review", "../../raw"],
        "GET"
      )
    ).rejects.toThrow("not allowed");
  });

  it("hard-codes Next development and start commands to loopback", () => {
    const packageJson = JSON.parse(readFileSync(resolve(process.cwd(), "package.json"), "utf8"));
    expect(packageJson.scripts.dev).toBe("node scripts/next-local.mjs dev");
    expect(packageJson.scripts.start).toBe("node scripts/next-local.mjs start");
    const launcher = readFileSync(resolve(process.cwd(), "scripts/next-local.mjs"), "utf8");
    expect(launcher).toContain('"--hostname", "127.0.0.1"');
    const lanBind = spawnSync(
      process.execPath,
      ["scripts/next-local.mjs", "dev", "--hostname", "0.0.0.0"],
      { cwd: process.cwd(), encoding: "utf8" }
    );
    expect(lanBind.status).toBe(2);
    expect(lanBind.stderr).toContain("cannot be overridden");
  });
});

function request(headers: Record<string, string>): Request {
  return new Request("http://127.0.0.1:3000/review-api/api/review", {
    headers,
    method: "POST"
  });
}

function restoreEnv(name: string, value: string | undefined): void {
  if (value === undefined) {
    delete process.env[name];
  } else {
    process.env[name] = value;
  }
}
