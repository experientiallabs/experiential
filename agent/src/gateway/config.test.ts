import { describe, expect, it } from "vitest";
import { HOSTED_GATEWAY_BASE_URL, LOCAL_GATEWAY_BASE_URL, resolveGateway } from "./config.js";

describe("gateway resolution", () => {
  it("uses the hosted gateway with an xpl key", () => {
    expect(resolveGateway({ EXP_API_KEY: "xpl_1" })).toEqual({ config: { kind: "hosted", baseUrl: HOSTED_GATEWAY_BASE_URL, apiKey: "xpl_1" }, source: "env" });
  });

  it("uses the local gateway with only a local key", () => {
    expect(resolveGateway({ EXP_GATEWAY_KEY: "local" })).toEqual({ config: { kind: "local", baseUrl: LOCAL_GATEWAY_BASE_URL, apiKey: "local" }, source: "env" });
  });

  it("prefers hosted when both keys exist, unless local is requested", () => {
    expect(resolveGateway({ EXP_API_KEY: "h", EXP_GATEWAY_KEY: "l" }).config?.kind).toBe("hosted");
    expect(resolveGateway({ EXP_API_KEY: "h", EXP_GATEWAY_KEY: "l", EXP_CODE_GATEWAY: "local" }).config).toMatchObject({ kind: "local", apiKey: "l" });
  });

  it("honors a base URL override and trims trailing slashes", () => {
    expect(resolveGateway({ EXP_API_KEY: "h", EXP_CODE_BASE_URL: "https://gw.example/v1/" }).config?.baseUrl).toBe("https://gw.example/v1");
  });

  it("explains what to do when there is no key or a bad selection", () => {
    expect(resolveGateway({}).problem).toMatch(/EXP_API_KEY/);
    expect(resolveGateway({}).problem).toMatch(/^No Experiential gateway key found\. Run `exp-code login`/);
    expect(resolveGateway({}).missingKey).toBe(true);
    expect(resolveGateway({ EXP_CODE_GATEWAY: "cloud" }).missingKey).toBe(false);
    expect(resolveGateway({ EXP_CODE_GATEWAY: "local" }).problem).toMatch(/needs EXP_GATEWAY_KEY/);
    expect(resolveGateway({ EXP_CODE_GATEWAY: "cloud" }).problem).toMatch(/must be "hosted" or "local"/);
  });

  describe("with a saved login", () => {
    const saved = { hostedKey: "xpl_saved" };

    it("uses the saved key when the environment has none", () => {
      expect(resolveGateway({}, saved)).toEqual({ config: { kind: "hosted", baseUrl: HOSTED_GATEWAY_BASE_URL, apiKey: "xpl_saved" }, source: "saved" });
      expect(resolveGateway({ EXP_CODE_GATEWAY: "hosted" }, saved).source).toBe("saved");
    });

    it("lets either environment key win over it", () => {
      expect(resolveGateway({ EXP_API_KEY: "xpl_env" }, saved)).toMatchObject({ config: { apiKey: "xpl_env" }, source: "env" });
      expect(resolveGateway({ EXP_GATEWAY_KEY: "l" }, saved)).toMatchObject({ config: { kind: "local", apiKey: "l" }, source: "env" });
      expect(resolveGateway({ EXP_CODE_GATEWAY: "local" }, saved).problem).toMatch(/needs EXP_GATEWAY_KEY/);
    });

    it("uses the saved gateway URL unless EXP_CODE_BASE_URL overrides it", () => {
      const withBase = { hostedKey: "xpl_saved", baseUrl: "https://staging.example/v1/" };
      expect(resolveGateway({}, withBase).config?.baseUrl).toBe("https://staging.example/v1");
      expect(resolveGateway({ EXP_CODE_BASE_URL: "https://other.example/v1" }, withBase).config?.baseUrl).toBe("https://other.example/v1");
      expect(resolveGateway({ EXP_API_KEY: "xpl_env" }, withBase).config?.baseUrl).toBe(HOSTED_GATEWAY_BASE_URL);
    });
  });
});
