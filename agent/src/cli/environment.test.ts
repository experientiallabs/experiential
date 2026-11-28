import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { applyCliEnvironment, expCodeHome, TRANSLATED_VARIABLES } from "./environment.js";

describe("CLI environment", () => {
  it("keeps exp-code's own config directory and turns Pi's telemetry and version check off", () => {
    const env: NodeJS.ProcessEnv = { PI_CODING_AGENT_DIR: "/somewhere/pi", PI_TELEMETRY: "1", PI_SKIP_VERSION_CHECK: "0" };
    applyCliEnvironment(env);
    expect(env.PI_CODING_AGENT_DIR).toBe(join(homedir(), ".exp-code", "agent"));
    expect(env.PI_TELEMETRY).toBe("0");
    expect(env.PI_SKIP_VERSION_CHECK).toBe("1");
    expect(env.EXP_CODE_AGENT).toBe("true");
    expect(env.AI_AGENT).toBe("exp-code");
    expect(env.PI_CODING_AGENT).toBeUndefined();
  });

  it("honors EXP_CODE_HOME", () => {
    const env: NodeJS.ProcessEnv = { EXP_CODE_HOME: "/x" };
    applyCliEnvironment(env);
    expect(expCodeHome(env)).toBe("/x");
    expect(env.PI_CODING_AGENT_DIR).toBe("/x/agent");
  });

  it("translates exp-code variables to the Pi names Pi reads", () => {
    const env: NodeJS.ProcessEnv = { EXP_CODE_OFFLINE: "1", EXP_CODE_SESSION_DIR: "/sessions", EXP_CODE_CACHE_RETENTION: "long", EXP_CODE_TRUE_COLOR: "" };
    applyCliEnvironment(env);
    expect(env.PI_OFFLINE).toBe("1");
    expect(env.PI_CODING_AGENT_SESSION_DIR).toBe("/sessions");
    expect(env.PI_CACHE_RETENTION).toBe("long");
    expect(env.PI_TRUE_COLOR).toBeUndefined();
  });

  it("never honors a PI_ name set by the user", () => {
    const env: NodeJS.ProcessEnv = { PI_OFFLINE: "1", PI_CODING_AGENT_SESSION_DIR: "/pi-sessions", PI_PACKAGE_DIR: "/nix", PI_SELF_UPDATE_COMMAND: "x", PI_MODEL: "m", RASPBERRY_PI_HOME: "kept" };
    applyCliEnvironment(env);
    expect(env.PI_OFFLINE).toBeUndefined();
    expect(env.PI_CODING_AGENT_SESSION_DIR).toBeUndefined();
    expect(env.PI_PACKAGE_DIR).toBeUndefined();
    expect(env.PI_SELF_UPDATE_COMMAND).toBeUndefined();
    expect(env.PI_MODEL).toBeUndefined();
    expect(env.RASPBERRY_PI_HOME).toBe("kept");
  });

  it("documents every translated variable in the README", () => {
    const readme = readFileSync(new URL("../../README.md", import.meta.url), "utf8");
    for (const variable of TRANSLATED_VARIABLES) expect(readme).toContain(`| \`${variable.name}\` |`);
    expect(readme).not.toMatch(/\bPI_[A-Z]/);
  });
});
