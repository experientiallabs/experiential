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
function runCli(args: string[], env: NodeJS.ProcessEnv = {}, cwd?: string): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const child = execFile(
      process.execPath,
      [cli, ...args],
      { cwd: cwd ?? home, env: { PATH: process.env.PATH, HOME: home, EXP_CODE_HOME: join(home, ".exp-code"), ...env }, timeout: 60_000 },
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
    // Examples and commands name the gateway's models and exp-code's sign-in, not Pi's providers.
    expect(help.stdout).not.toMatch(/openai|github-copilot|sonnet|haiku|gpt-4o|radius|auth <command>|print-api-key/);
    expect(help.stdout).toContain("--model experiential-medium-model:high");
    expect(help.stdout).toContain("exp-code login --status, whoami");
    const update = await runCli(["update", "--help"]);
    expect(update.stdout).toContain("Update exp-code, installed extensions, or model catalogs.");
    expect(update.stdout).not.toMatch(/\bpi\b/i);
  });

  it("refuses an unknown subcommand instead of sending it to the model", async () => {
    const env = { EXP_GATEWAY_KEY: "local-key", EXP_CODE_BASE_URL: gateway.baseUrl };
    const before = gateway.completions().length;
    const unknown = await runCli(["frobnicate"], env);
    expect(unknown.code).toBe(2);
    expect(unknown.stderr).toContain('exp-code: unknown command "frobnicate".');
    expect(unknown.stderr).toContain("login, logout, whoami, update");
    expect(unknown.stderr).toContain("exp-code -- frobnicate");
    expect(gateway.completions()).toHaveLength(before);
    // A quoted prompt, a file and `--` still go to the model; `help` is --help.
    gateway.script([{ text: "one" }, { text: "two" }]);
    expect((await runCli(["fix the parser", "-p"], env)).stdout).toContain("one");
    expect((await runCli(["-p", "--", "frobnicate"], env)).stdout).toContain("two");
    expect((await runCli(["help"])).stdout).toContain("exp-code [options]");
    const { unknownSubcommandMessage } = await import("./run.js");
    expect(unknownSubcommandMessage(["notes"], (path) => path === "notes")).toBeUndefined();
    expect(unknownSubcommandMessage(["src/app.ts"])).toBeUndefined();
    expect(unknownSubcommandMessage(["login"])).toBeUndefined();
  });

  it("answers `auth` with the sign-in commands and a bad EXP_CODE_WIRE with one line", async () => {
    const auth = await runCli(["auth", "check", "--provider", "experiential"]);
    expect(auth.code).toBe(2);
    expect(auth.stderr).toMatch(/^exp-code has no `auth` command/);
    expect(auth.stderr).toContain("exp-code login --status");
    const wire = await runCli(["-p", "hi"], { EXP_GATEWAY_KEY: "local-key", EXP_CODE_BASE_URL: gateway.baseUrl, EXP_CODE_WIRE: "bogus" });
    expect(wire.code).toBe(2);
    expect(wire.stderr.trim()).toBe('exp-code: EXP_CODE_WIRE: Unknown wire API "bogus". Use completions, responses, or messages.');
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

  it("starts on the saved model and its thinking level; --model and EXP_CODE_MODEL are never saved", async () => {
    const agentDir = join(home, ".exp-code", "agent");
    await mkdir(agentDir, { recursive: true });
    const settings = join(agentDir, "settings.json");
    const saved = { defaultProvider: "experiential", defaultModel: "thinker", modelThinkingLevels: { "experiential/thinker": "high" } };
    await writeFile(settings, JSON.stringify(saved));
    const cache = join(home, ".exp-code", "cache");
    await rm(cache, { recursive: true, force: true });
    const model = (id: string) => ({ id, object: "model", created: 0, owned_by: "exp", context_window_tokens: 1_050_000, maximum_output_tokens: 128_000 });
    const thinker = { ...model("thinker"), supports_reasoning: true, supported_reasoning_efforts: ["none", "high", "max"] };
    gateway.setModels([model("experiential-large-model"), model("other"), thinker]);
    const env = { EXP_GATEWAY_KEY: "local-key", EXP_CODE_BASE_URL: gateway.baseUrl };
    const ask = async (args: string[], extra: NodeJS.ProcessEnv = {}) => {
      gateway.script([{ text: "ok" }]);
      const result = await runCli([...args, "-p", "hi"], { ...env, ...extra });
      return { ...result, body: gateway.completions().at(-1)! };
    };
    try {
      const start = await ask([]);
      expect(start.body.model).toBe("thinker");
      expect(start.body.reasoning_effort).toBe("high");

      expect((await ask(["--model", "experiential/other"])).body.model).toBe("other");
      expect((await ask([], { EXP_CODE_MODEL: "other" })).body.model).toBe("other");
      expect(JSON.parse(await readFile(settings, "utf8"))).toMatchObject(saved);

      // The gateway stops listing the saved model: the default model, with a note, and the saved
      // choice is kept for when it is listed again.
      await rm(cache, { recursive: true, force: true });
      gateway.setModels([model("experiential-large-model"), model("other")]);
      const gone = await ask([]);
      expect(gone.body.model).toBe("experiential-large-model");
      expect(gone.stderr).toContain("exp-code: the gateway no longer lists thinker, your saved model; using experiential-large-model");
      expect(JSON.parse(await readFile(settings, "utf8"))).toMatchObject(saved);
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

  it("says how to continue a session of another folder when nobody can answer the fork question", async () => {
    const env = { EXP_GATEWAY_KEY: "local-key", EXP_CODE_BASE_URL: gateway.baseUrl };
    const first = join(home, "first-project");
    const second = join(home, "second-project");
    await mkdir(first, { recursive: true });
    await mkdir(second, { recursive: true });
    gateway.script([{ text: "stored" }]);
    expect((await runCli(["-p", "hi"], env, first)).code).toBe(0);
    const sessionsRoot = join(home, ".exp-code", "agent", "sessions");
    const folder = readdirSync(sessionsRoot).find((name) => name.includes("first-project"))!;
    const id = readdirSync(join(sessionsRoot, folder))[0]!.replace(/\.jsonl$/, "").split("_").at(-1)!;
    // stdin is closed: the run used to end with Node's "unsettled top-level await" and code 13.
    const result = await runCli(["--session", id.slice(0, 8), "-p", "again"], env, second);
    expect(result.code).toBe(1);
    // Print mode keeps stdout for the answer, so the question and the advice go to stderr.
    expect(result.stderr).toContain("Fork this session into current directory? [y/N]");
    expect(result.stderr).toMatch(/Run exp-code in \S*\/first-project to continue the session there/);
    expect(result.stderr).toContain(`--fork ${id.slice(0, 8)}`);
    expect(result.stderr).not.toContain("top-level await");
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

  it("keeps the gateway key out of the commands the agent runs", async () => {
    const command = 'echo "key=[$EXP_GATEWAY_KEY] hosted=[$EXP_API_KEY] exp=[$EXPLABS_API_KEY] model=[$EXP_CODE_SESSION_MODEL] other=[$QA_OTHER]"';
    gateway.script([{ calls: [{ name: "bash", arguments: { command } }] }, { text: "done" }]);
    const result = await runCli(["-p", "show the environment"], {
      EXP_GATEWAY_KEY: "local-secret-key",
      EXPLABS_API_KEY: "xpl_launcher_secret",
      EXP_CODE_BASE_URL: gateway.baseUrl,
      QA_OTHER: "kept"
    });
    expect(result.stdout.trim()).toBe("done");
    const sent = JSON.stringify(gateway.completions().at(-1)!.messages);
    expect(sent).toContain("key=[] hosted=[] exp=[] model=[fake-model] other=[kept]");
    expect(sent).not.toContain("local-secret-key");
    expect(sent).not.toContain("xpl_launcher_secret");
    // The shell tool's guideline names the variables commands actually see.
    expect(sent).toContain("EXP_CODE_SESSION_* environment variables");
    expect(sent).not.toContain("PI_* environment");
  });

  it("runs a command-line prompt when piped stdin stays open and silent, and still reads input that arrives", async () => {
    const env = { PATH: process.env.PATH, HOME: home, EXP_CODE_HOME: join(home, ".exp-code"), EXP_GATEWAY_KEY: "local-key", EXP_CODE_BASE_URL: gateway.baseUrl };
    /** Run the CLI with a stdin pipe the caller controls. */
    const withStdin = (args: string[], feed: (stdin: NodeJS.WritableStream) => void) =>
      new Promise<{ code: number; stdout: string; stderr: string }>((resolve) => {
        const child = execFile(process.execPath, [cli, ...args], { cwd: home, env, timeout: 30_000 }, (error, stdout, stderr) =>
          resolve({ code: error ? (typeof error.code === "number" ? error.code : 1) : 0, stdout, stderr })
        );
        feed(child.stdin!);
      });

    // Before the fix this waited for an end of input that never came, and the run timed out.
    gateway.script([{ text: "ran without stdin" }]);
    let open: NodeJS.WritableStream | undefined;
    const silent = await withStdin(["-p", "say hello"], (stdin) => {
      open = stdin;
    });
    open?.end();
    expect(silent.code).toBe(0);
    expect(silent.stdout.trim()).toBe("ran without stdin");
    expect(silent.stderr).toContain("nothing arrived on stdin");

    gateway.script([{ text: "read the pipe" }]);
    const piped = await withStdin(["-p", "summarize"], (stdin) => {
      setTimeout(() => stdin.write("first part "), 200);
      setTimeout(() => stdin.end("second part"), 400);
    });
    expect(piped.stdout.trim()).toBe("read the pipe");
    expect(piped.stderr).not.toContain("nothing arrived on stdin");
    expect(JSON.stringify(gateway.completions().at(-1)!.messages)).toContain("first part second part");
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

  it("explains an unreachable gateway or an empty listing once, with exp-code's own guidance", async () => {
    const down = await runCli(["-p", "--no-session", "hi"], { EXP_API_KEY: "xpl_unreachable1", EXP_CODE_BASE_URL: "http://127.0.0.1:9/v1" });
    expect(down.code).not.toBe(0);
    const reasons = down.stderr.split("\n").filter((line) => line.includes("Could not reach the gateway at http://127.0.0.1:9/v1 to list models"));
    expect(reasons).toHaveLength(1);
    expect(down.stderr).not.toMatch(/models\.json|\/login to log into a provider|node_modules/);
    gateway.setModels([]);
    try {
      // Its own configuration directory: the other tests cached this gateway's listing.
      const empty = await runCli(["-p", "--no-session", "hi"], { EXP_GATEWAY_KEY: "local-key", EXP_CODE_BASE_URL: gateway.baseUrl, EXP_CODE_HOME: join(home, "empty-listing") });
      expect(empty.code).not.toBe(0);
      expect(empty.stderr).toContain(`exp-code: The gateway at ${gateway.baseUrl} lists no models this key can call.`);
      expect(empty.stderr).toContain("Run `exp-code login` in a terminal to sign in");
    } finally {
      gateway.setModels([{ id: "fake-model", object: "model", created: 0, owned_by: "exp" }]);
    }
  });

  it("refuses a key on the command line", async () => {
    const result = await runCli(["--api-key", "xpl_onthecommandline", "-p", "hi"], { EXP_GATEWAY_KEY: "local-key", EXP_CODE_BASE_URL: gateway.baseUrl });
    expect(result.code).toBe(2);
    expect(result.stderr).toContain("--api-key is not supported");
    expect(result.stderr).not.toContain("xpl_onthecommandline");
    expect((await runCli(["--help"])).stdout).not.toContain("--api-key");
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
