import { execFile, execFileSync } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
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
    expect(version.stdout.trim()).toBe(`exp-code ${EXP_CODE_VERSION}`);
    const help = await runCli(["--help"]);
    expect(help.code).toBe(0);
    expect(help.stdout).toContain("exp-code");
    expect(help.stdout).toContain("--mode");
    expect(help.stdout).toContain("exp-code [options]");
    expect(help.stdout).not.toMatch(/(^|[\s`'"(])pi([\s`'")]|$)/m);
    expect(help.stdout).not.toMatch(/\bPi\b|pi\.dev|~\/\.pi|\bPI_[A-Z]/);
    expect(help.stdout).toContain("EXP_CODE_OFFLINE");
    expect(help.stdout).not.toContain("GEMINI_API_KEY");
    const update = await runCli(["update", "--help"]);
    expect(update.stdout).toContain("Update exp-code, installed extensions, or model catalogs.");
    expect(update.stdout).not.toMatch(/\bpi\b/i);
  });

  it("offers only the gateway's models, whatever vendor keys the environment holds", async () => {
    const env = { EXP_GATEWAY_KEY: "local-key", EXP_CODE_BASE_URL: gateway.baseUrl, GEMINI_API_KEY: "g", OPENAI_API_KEY: "o", ANTHROPIC_API_KEY: "a", OPENROUTER_API_KEY: "r", GROQ_API_KEY: "q" };
    const listed = await runCli(["--list-models"], env);
    const providers = listed.stdout
      .trim()
      .split("\n")
      .slice(1)
      .map((line) => line.split(/\s+/)[0]);
    expect(providers.length).toBeGreaterThan(0);
    expect(new Set(providers)).toEqual(new Set(["experiential"]));
  });

  it("moves a saved direct-provider model to the gateway model of the same id", async () => {
    const agentDir = join(home, ".exp-code", "agent");
    await mkdir(agentDir, { recursive: true });
    const settings = join(agentDir, "settings.json");
    await writeFile(settings, JSON.stringify({ defaultProvider: "google", defaultModel: "gemini-4-neon" }));
    // The five-minute listing cache of the other tests would answer with their model list.
    const cache = join(home, ".exp-code", "cache");
    await rm(cache, { recursive: true, force: true });
    gateway.setModels([
      { id: "experiential-large-model", object: "model", created: 0, owned_by: "exp" },
      { id: "gemini-4-neon", object: "model", created: 0, owned_by: "exp" }
    ]);
    try {
      gateway.script([{ text: "neon" }]);
      const env = { EXP_GATEWAY_KEY: "local-key", EXP_CODE_BASE_URL: gateway.baseUrl, GEMINI_API_KEY: "g" };
      expect((await runCli(["-p", "hi"], env)).stdout.trim()).toBe("neon");
      expect(gateway.completions().at(-1)!.model).toBe("gemini-4-neon");
      expect(JSON.parse(await readFile(settings, "utf8"))).toMatchObject({ defaultProvider: "experiential", defaultModel: "gemini-4-neon" });

      gateway.script([{ text: "moved" }]);
      const moved = await runCli(["--model", "google/gemini-4-neon", "-p", "hi"], env);
      expect(moved.stdout.trim()).toBe("moved");
      expect(moved.stderr).toContain("using experiential/gemini-4-neon for google/gemini-4-neon");
    } finally {
      await rm(settings, { force: true });
      await rm(cache, { recursive: true, force: true });
      gateway.setModels([{ id: "fake-model", object: "model", created: 0, owned_by: "exp" }]);
    }
  });

  it("reads exp-code's variables and ignores Pi's", async () => {
    const sessions = join(home, "exp-sessions");
    const ignored = join(home, "pi-sessions");
    gateway.script([{ text: "stored" }]);
    await runCli(["-p", "hi"], { EXP_GATEWAY_KEY: "local-key", EXP_CODE_BASE_URL: gateway.baseUrl, EXP_CODE_SESSION_DIR: sessions, PI_CODING_AGENT_SESSION_DIR: ignored });
    expect(readdirSync(sessions).length).toBeGreaterThan(0);
    expect(existsSync(ignored)).toBe(false);
  });

  it("never runs Pi's own self-update", async () => {
    const result = await runCli(["update", "pi"]);
    expect(result.stderr + result.stdout).not.toContain("@earendil-works");
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
    expect(system).toContain("You are exp-code");
    expect(system).not.toContain("running inside pi");
    const listing = gateway.requests.find((request) => request.path === "/v1/models");
    expect(listing?.headers.authorization).toBe("Bearer local-key");
    // exp-code keeps its sessions under its own home, never Pi's.
    expect(readdirSync(join(home, ".exp-code", "agent", "sessions")).length).toBeGreaterThan(0);
    expect(existsSync(join(home, ".pi"))).toBe(false);
  });

  it("gives a hosted session the Experiential platform tools unless EXP_CODE_PLATFORM_TOOLS=0", async () => {
    const names = () => ((gateway.completions().at(-1)!.tools as { function: { name: string } }[]) ?? []).map((tool) => tool.function.name);
    const hostedEnv = { EXP_API_KEY: "xpl_test", EXP_CODE_BASE_URL: gateway.baseUrl, EXP_CODE_MODEL: "fake-model" };
    gateway.script([{ text: "on" }]);
    expect((await runCli(["-p", "hi"], hostedEnv)).stdout.trim()).toBe("on");
    expect(names()).toEqual(expect.arrayContaining(["experiential_search_requests", "experiential_get_request", "experiential_usage", "experiential_models", "experiential_evals"]));
    gateway.script([{ text: "off" }]);
    expect((await runCli(["-p", "hi"], { ...hostedEnv, EXP_CODE_PLATFORM_TOOLS: "0" })).stdout.trim()).toBe("off");
    expect(names().filter((name) => name.startsWith("experiential_"))).toEqual([]);
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

  it("logs in, reports status, uses the saved key, and logs out", async () => {
    const status = await runCli(["login", "--status"]);
    expect(status.code).toBe(1);
    expect(status.stdout).toContain("Not logged in");

    const refused = await runCli(["login", "--no-browser"]);
    expect(refused.code).toBe(1);
    expect(refused.stderr).toContain("interactive terminal");

    const agentDir = join(home, ".exp-code", "agent");
    await mkdir(agentDir, { recursive: true });
    await writeFile(join(agentDir, "credentials.json"), JSON.stringify({ hostedKey: "xpl_savedkey_123456", baseUrl: gateway.baseUrl, savedAt: "now", keyName: "exp-code (ci)" }));
    const saved = await runCli(["whoami"]);
    expect(saved.code).toBe(0);
    expect(saved.stdout).toContain("Using the saved login");
    expect(saved.stdout).not.toContain("xpl_savedkey_123456");

    gateway.script([{ text: "Hello with the saved key." }]);
    const answer = await runCli(["-p", "say hello"]);
    expect(answer.stdout.trim()).toBe("Hello with the saved key.");
    expect(gateway.requests.filter((r) => r.path === "/v1/chat/completions").at(-1)!.headers.authorization).toBe("Bearer xpl_savedkey_123456");

    const out = await runCli(["logout"]);
    expect(out.code).toBe(0);
    expect(out.stdout).toContain("still exists on the Experiential platform");
    expect(existsSync(join(agentDir, "credentials.json"))).toBe(false);
  });
});
