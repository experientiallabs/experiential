import { chmod, mkdir, mkdtemp, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { credentialsPath, deleteCredentials, maskKey, parseCredentials, readCredentials, saveCredentials } from "./credentials.js";

let home: string;
let env: NodeJS.ProcessEnv;
const posix = process.platform !== "win32";

beforeEach(async () => {
  home = await mkdtemp(join(tmpdir(), "exp-code-creds-"));
  env = { EXP_CODE_HOME: join(home, ".exp-code") };
});

afterEach(async () => {
  await rm(home, { recursive: true, force: true });
});

const sample = { hostedKey: "xpl_abcdefghijk", savedAt: "2026-11-13T00:00:00.000Z", keyName: "exp-code (host)" };

describe("saved credentials", () => {
  it("lives in the exp-code agent directory", () => {
    expect(credentialsPath(env)).toBe(join(home, ".exp-code", "agent", "credentials.json"));
  });

  it("saves owner-only and reads back", async () => {
    const path = await saveCredentials(env, { ...sample, baseUrl: "https://gw.example/v1" });
    expect(path).toBe(credentialsPath(env));
    await expect(readCredentials(env)).resolves.toEqual({ ...sample, baseUrl: "https://gw.example/v1" });
    if (posix) {
      expect((await stat(path)).mode & 0o777).toBe(0o600);
      expect((await stat(join(home, ".exp-code", "agent"))).mode & 0o777).toBe(0o700);
    }
  });

  it("tightens an existing directory and replaces the file atomically", async () => {
    const dir = join(home, ".exp-code", "agent");
    await mkdir(dir, { recursive: true });
    await chmod(dir, 0o755);
    await writeFile(join(dir, "credentials.json"), "old", { mode: 0o644 });
    await saveCredentials(env, sample);
    expect(JSON.parse(await readFile(join(dir, "credentials.json"), "utf8"))).toEqual(sample);
    // No temporary file is left behind.
    expect((await readdir(dir)).filter((name) => name !== "credentials.json")).toEqual([]);
    if (posix) {
      expect((await stat(dir)).mode & 0o777).toBe(0o700);
      expect((await stat(join(dir, "credentials.json"))).mode & 0o777).toBe(0o600);
    }
  });

  it("reads a missing or malformed file as no login", async () => {
    await expect(readCredentials(env)).resolves.toBeNull();
    await mkdir(join(home, ".exp-code", "agent"), { recursive: true });
    await writeFile(credentialsPath(env), "{not json");
    await expect(readCredentials(env)).resolves.toBeNull();
    expect(parseCredentials({ hostedKey: "sk_live" })).toBeNull();
    expect(parseCredentials([])).toBeNull();
    expect(parseCredentials({ hostedKey: "xpl_x", baseUrl: 3 })).toBeNull();
    expect(parseCredentials({ hostedKey: "xpl_x" })).toEqual({ hostedKey: "xpl_x", savedAt: "", keyName: "" });
  });

  it("deletes the saved login", async () => {
    await expect(deleteCredentials(env)).resolves.toBe(false);
    await saveCredentials(env, sample);
    await expect(deleteCredentials(env)).resolves.toBe(true);
    await expect(readCredentials(env)).resolves.toBeNull();
  });

  it("masks keys to the prefix and seven characters", () => {
    expect(maskKey("xpl_abcdefghijklmnop")).toBe("xpl_abcdefg…");
    expect(maskKey("localvirtualkey")).toBe("localvi…");
  });
});
