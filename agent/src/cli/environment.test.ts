import { homedir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { applyCliEnvironment, expCodeHome } from "./environment.js";

describe("CLI environment", () => {
  it("keeps exp-code's own config directory and defaults telemetry and version checks off", () => {
    const env: NodeJS.ProcessEnv = { PI_CODING_AGENT_DIR: "/somewhere/pi" };
    applyCliEnvironment(env);
    expect(env.PI_CODING_AGENT_DIR).toBe(join(homedir(), ".exp-code", "agent"));
    expect(env.PI_TELEMETRY).toBe("0");
    expect(env.PI_SKIP_VERSION_CHECK).toBe("1");
  });

  it("honors EXP_CODE_HOME and explicit telemetry choices", () => {
    const env: NodeJS.ProcessEnv = { EXP_CODE_HOME: "/x", PI_TELEMETRY: "1" };
    applyCliEnvironment(env);
    expect(expCodeHome(env)).toBe("/x");
    expect(env.PI_CODING_AGENT_DIR).toBe("/x/agent");
    expect(env.PI_TELEMETRY).toBe("1");
  });
});
