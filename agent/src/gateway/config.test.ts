import { describe, expect, it } from "vitest";
import { HOSTED_GATEWAY_BASE_URL, LOCAL_GATEWAY_BASE_URL, resolveGateway, unsendableCharacter } from "./config.js";

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

  it("refuses a key a header cannot carry, without repeating the key", () => {
    const resolution = resolveGateway({ EXP_API_KEY: "xpl_secretpart\nrest" });
    expect(resolution.config).toBeNull();
    expect(resolution.missingKey).toBe(false);
    expect(resolution.problem).toBe("EXP_API_KEY has a character a key cannot contain (a line break at position 15). Copy the key again.");
    expect(resolution.problem).not.toContain("secretpart");
    expect(resolveGateway({ EXP_API_KEY: "xpl_ab\u20accd" }).problem).toMatch(/a non-ASCII character at position 7/);
    expect(resolveGateway({ EXP_API_KEY: "xpl_a b" }).problem).toMatch(/a space at position 6/);
    expect(resolveGateway({}, { hostedKey: "xpl_saved\u0007" }).problem).toMatch(/^The key `exp-code login` saved has a character a key cannot contain \(a control character at position 10\)\. Copy the key again and run `exp-code login`\.$/);
    expect(unsendableCharacter("xpl_AZaz09-_.~")).toBeUndefined();
  });

  it("refuses a gateway address that is not an http or https URL", () => {
    expect(resolveGateway({ EXP_API_KEY: "xpl_1", EXP_CODE_BASE_URL: "not a url" }).problem).toBe(
      `EXP_CODE_BASE_URL must be an http:// or https:// URL such as ${HOSTED_GATEWAY_BASE_URL}, not "not a url".`
    );
    expect(resolveGateway({ EXP_API_KEY: "xpl_1", EXP_CODE_BASE_URL: "ftp://gw.example/v1" }).config).toBeNull();
    expect(resolveGateway({ EXP_API_KEY: "xpl_1", EXP_CODE_BASE_URL: "http://127.0.0.1:9/v1" }).config?.baseUrl).toBe("http://127.0.0.1:9/v1");
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
