import { describe, expect, it } from "vitest";
import { parseRunnerArgs } from "./main.js";
import { SESSION_DIR, WORK_DIR } from "./protocol.js";

describe("runner arguments", () => {
  it("defaults to the protocol directories and the user account", () => {
    expect(parseRunnerArgs(["11111111-2222-3333-4444-555555555555"])).toEqual({
      runId: "11111111-2222-3333-4444-555555555555",
      sessionDir: SESSION_DIR,
      workDir: WORK_DIR,
      user: "user"
    });
  });

  it("accepts overrides and refuses bad run ids", () => {
    expect(parseRunnerArgs(["run-1", "--session-dir", "/s", "--work-dir", "/w", "--user", "agent"])).toEqual({ runId: "run-1", sessionDir: "/s", workDir: "/w", user: "agent" });
    expect(typeof parseRunnerArgs([])).toBe("string");
    expect(typeof parseRunnerArgs(["../etc"])).toBe("string");
    expect(typeof parseRunnerArgs(["a", "b"])).toBe("string");
  });
});
