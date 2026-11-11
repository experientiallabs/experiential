import { execFile, execFileSync } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { type FakeGateway, startFakeGateway } from "../testing/fake-gateway.js";
import { EXP_CODE_VERSION } from "../version.js";

const packageDir = fileURLToPath(new URL("../..", import.meta.url));
const cli = join(packageDir, "dist", "cli.js");

let home: string;
let gateway: FakeGateway;

/** Run the built CLI with stdin closed and an isolated home. */
function runCli(args: string[], env: NodeJS.ProcessEnv = {}): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const child = execFile(
      process.execPath,
      [cli, ...args],
      { cwd: home, env: { PATH: process.env.PATH, HOME: home, EXP_CODE_HOME: join(home, ".exp-code"), ...env }, timeout: 60_000 },
      (error, stdout, stderr) => resolve({ code: error ? (typeof error.code === "number" ? error.code : 1) : 0, stdout, stderr })
    );
    child.stdin?.end();
  });
}

beforeAll(async () => {
  execFileSync(process.execPath, [join(packageDir, "node_modules", "typescript", "bin", "tsc"), "-p", join(packageDir, "tsconfig.build.json")], { stdio: "inherit" });
  home = await mkdtemp(join(tmpdir(), "exp-code-cli-"));
  gateway = await startFakeGateway();
}, 120_000);

afterAll(async () => {
  await gateway.close();
  await rm(home, { recursive: true, force: true });
});

describe("exp-code CLI", () => {
  it("prints its version and help", async () => {
    expect(existsSync(cli)).toBe(true);
    const version = await runCli(["--version"]);
    expect(version.stdout).toMatch(new RegExp(`^exp-code ${EXP_CODE_VERSION.replaceAll(".", "\\.")} \\(pi-coding-agent \\d+\\.\\d+\\.\\d+\\)`));
    const help = await runCli(["--help"]);
    expect(help.code).toBe(0);
    expect(help.stdout).toContain("exp-code");
    expect(help.stdout).toContain("--mode");
  });

  it("answers a print-mode prompt through a local gateway, with the exp-code tools and prompt", async () => {
    gateway.script([{ text: "Hello from the gateway." }]);
    const result = await runCli(["-p", "say hello"], { EXP_GATEWAY_KEY: "local-key", EXP_CODE_BASE_URL: gateway.baseUrl });
    expect(result.stdout.trim()).toBe("Hello from the gateway.");
    const body = gateway.completions().at(-1)!;
    expect(body.model).toBe("fake-model");
    const tools = (body.tools as { function: { name: string } }[]).map((tool) => tool.function.name);
    expect(tools).toEqual(expect.arrayContaining(["read", "bash", "edit", "write", "todo_write", "ask_user", "web_fetch", "web_search"]));
    const system = JSON.stringify((body.messages as { content: unknown }[])[0]);
    expect(system).toContain("<cowork>");
    expect(system).toContain("xlsx");
    const listing = gateway.requests.find((request) => request.path === "/v1/models");
    expect(listing?.headers.authorization).toBe("Bearer local-key");
    // exp-code keeps its sessions under its own home, never Pi's.
    expect(readdirSync(join(home, ".exp-code", "agent", "sessions")).length).toBeGreaterThan(0);
    expect(existsSync(join(home, ".pi"))).toBe(false);
  });

  it("emits Pi's JSON event stream in --mode json", async () => {
    gateway.script([{ text: "json mode" }]);
    const result = await runCli(["--mode", "json", "hi"], { EXP_GATEWAY_KEY: "local-key", EXP_CODE_BASE_URL: gateway.baseUrl });
    const types = result.stdout
      .trim()
      .split("\n")
      .map((line) => (JSON.parse(line) as { type: string }).type);
    expect(types).toContain("agent_end");
  });

  it("explains how to provide a gateway key when there is none", async () => {
    const result = await runCli(["-p", "hi"]);
    expect(result.stderr).toMatch(/No Experiential gateway key found/);
  });
});
