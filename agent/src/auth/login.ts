/**
 * `exp-code login`, `exp-code logout` and `exp-code login --status`.
 *
 * Login runs the same browser approval as the Python `exp login`: a loopback listener receives
 * an organization key from the Experiential platform's approval page. The key is checked against
 * the gateway (`GET <base>/models`) before it is saved, so a saved login always worked once.
 */

import { hostname as osHostname } from "node:os";
import { GATEWAY_ENV, HOSTED_GATEWAY_BASE_URL, trimBaseUrl } from "../gateway/config.js";
import { openBrowser } from "./browser.js";
import { credentialsPath, deleteCredentials, maskKey, readCredentials, saveCredentials } from "./credentials.js";
import { authorizeUrl, defaultKeyName, LOGIN_TIMEOUT_MS, LoopbackLogin, newState, platformUrl } from "./loopback.js";
import { LoginCancelled, type MaskedInput, readMaskedLine } from "./masked-input.js";

/** Everything login touches, injectable for tests. */
export interface AuthIo {
  env: NodeJS.ProcessEnv;
  stdin: MaskedInput;
  out: (text: string) => void;
  err: (text: string) => void;
  openBrowser?: (url: string) => Promise<boolean>;
  fetch?: typeof fetch;
  hostname?: () => string;
  timeoutMs?: number;
  /** Aborts the login (Ctrl+C outside the paste prompt). */
  signal?: AbortSignal;
}

const VERIFY_TIMEOUT_MS = 15_000;

/** Whether a string looks like an Experiential organization key. */
export function isHostedKey(value: string): boolean {
  return /^xpl_[A-Za-z0-9_-]{8,}$/.test(value);
}

/** Check a key against the gateway; resolves the number of models it can call. */
export async function verifyKey(baseUrl: string, key: string, fetchImpl: typeof fetch = fetch): Promise<number> {
  let response: Response;
  try {
    response = await fetchImpl(`${trimBaseUrl(baseUrl)}/models`, {
      headers: { authorization: `Bearer ${key}`, accept: "application/json" },
      signal: AbortSignal.timeout(VERIFY_TIMEOUT_MS)
    });
  } catch (error) {
    throw new Error(`Could not reach the gateway at ${baseUrl} to check the key: ${(error as Error).message}`);
  }
  if (response.status === 401 || response.status === 403) {
    throw new Error("The gateway rejected this key. It may have been revoked; run `exp-code login` again.");
  }
  if (!response.ok) throw new Error(`Could not check the key: the gateway answered HTTP ${response.status}.`);
  const body = (await response.json().catch(() => null)) as { data?: unknown } | null;
  return Array.isArray(body?.data) ? body.data.length : 0;
}

/** Run one login. Resolves the process exit code. */
export async function login(io: AuthIo, options: { noBrowser: boolean }): Promise<number> {
  const platform = platformUrl(io.env);
  const override = io.env[GATEWAY_ENV.baseUrl]?.trim() ?? "";
  const baseUrl = trimBaseUrl(override !== "" ? override : HOSTED_GATEWAY_BASE_URL);
  const keyName = defaultKeyName((io.hostname ?? osHostname)());
  const timeoutMs = io.timeoutMs ?? LOGIN_TIMEOUT_MS;
  const interactive = io.stdin.isTTY === true;

  let key: string | null = null;
  let pasted = false;
  try {
    if (options.noBrowser) {
      if (!interactive) {
        io.err(
          "exp-code login --no-browser reads the key from an interactive terminal, and this input is not one.\n" +
            `Set ${GATEWAY_ENV.hostedKey} to a key instead, or run \`exp-code login\` where a browser can open.\n`
        );
        return 1;
      }
      // Without a port the approval page shows the key to copy instead of handing it back.
      const url = authorizeUrl(platform, { state: newState(), keyName });
      io.out(`Open this URL in a browser, sign in and approve, then paste the key it shows:\n\n  ${url}\n\n`);
      key = await readMaskedLine(io.stdin, io.out, "API key: ", io.signal);
      pasted = true;
    } else {
      const listener = new LoopbackLogin(platform);
      try {
        await listener.start();
        const url = listener.url(keyName);
        io.out(`Opening the Experiential platform to approve exp-code. If no browser opens, visit:\n\n  ${url}\n\n`);
        const opened = await (io.openBrowser ?? openBrowser)(url);
        if (opened || !interactive) {
          io.out("Waiting for approval in your browser (Ctrl+C to cancel)...\n");
          key = await listener.wait(timeoutMs, io.signal);
        } else {
          io.out("Approve in a browser, or paste an existing key to continue.\n");
          const stopPrompt = new AbortController();
          const relay = () => stopPrompt.abort();
          io.signal?.addEventListener("abort", relay, { once: true });
          const fromBrowser = listener.wait(timeoutMs, io.signal).then((value) => {
            if (value !== null) stopPrompt.abort();
            return value;
          });
          const typed = await readMaskedLine(io.stdin, io.out, "API key: ", stopPrompt.signal);
          io.signal?.removeEventListener("abort", relay);
          key = listener.received() ?? typed;
          pasted = listener.received() === null && typed !== null;
          if (key === null && !stopPrompt.signal.aborted) key = await fromBrowser;
        }
      } finally {
        await listener.close();
      }
    }
  } catch (error) {
    if (error instanceof LoginCancelled) {
      io.err("Login cancelled.\n");
      return 130;
    }
    throw error;
  }

  if (io.signal?.aborted) {
    io.err("Login cancelled.\n");
    return 130;
  }
  if (key === null) {
    io.err(pasted || options.noBrowser ? "No key entered.\n" : "Login timed out waiting for the browser. Run `exp-code login` to try again.\n");
    return 1;
  }
  if (!isHostedKey(key)) {
    io.err("That is not an Experiential API key (they start with xpl_).\n");
    return 1;
  }
  let models: number;
  try {
    models = await verifyKey(baseUrl, key, io.fetch);
  } catch (error) {
    io.err(`${(error as Error).message}\nNothing was saved.\n`);
    return 1;
  }
  const savedName = pasted && !options.noBrowser ? "pasted key" : keyName;
  const path = await saveCredentials(io.env, {
    hostedKey: key,
    ...(override !== "" ? { baseUrl } : {}),
    savedAt: new Date().toISOString(),
    keyName: savedName
  });
  io.out(`Logged in. Saved key "${savedName}" (${maskKey(key)}) to ${path}; ${models} model${models === 1 ? "" : "s"} available.\n`);
  if ((io.env[GATEWAY_ENV.hostedKey]?.trim() ?? "") !== "") {
    io.out(`Note: ${GATEWAY_ENV.hostedKey} is set in this environment and is used instead of the saved login.\n`);
  }
  return 0;
}

/** Forget the saved login. */
export async function logout(io: Pick<AuthIo, "env" | "out">): Promise<number> {
  const removed = await deleteCredentials(io.env);
  if (!removed) {
    io.out("Not logged in; there was no saved key.\n");
    return 0;
  }
  io.out(
    "Logged out: the saved key was removed from this computer.\n" +
      "The key still exists on the Experiential platform; revoke it on your organization's API keys page if you no longer need it.\n"
  );
  if ((io.env[GATEWAY_ENV.hostedKey]?.trim() ?? "") !== "") {
    io.out(`${GATEWAY_ENV.hostedKey} is still set in this environment, so exp-code keeps using it.\n`);
  }
  return 0;
}

/** Say which key exp-code would use and where it comes from. Exit code 1 when there is none. */
export async function loginStatus(io: Pick<AuthIo, "env" | "out">): Promise<number> {
  const hosted = io.env[GATEWAY_ENV.hostedKey]?.trim() ?? "";
  const local = io.env[GATEWAY_ENV.localKey]?.trim() ?? "";
  const saved = await readCredentials(io.env);
  const path = credentialsPath(io.env);
  if (hosted !== "") io.out(`Using ${GATEWAY_ENV.hostedKey} from the environment: ${maskKey(hosted)}\n`);
  else if (local !== "") io.out(`Using ${GATEWAY_ENV.localKey} from the environment (local gateway): ${maskKey(local)}\n`);
  if (saved !== null) {
    const verb = hosted === "" && local === "" ? "Using the saved login" : "Also saved (not in use while the environment sets a key)";
    const name = saved.keyName !== "" ? ` "${saved.keyName}"` : "";
    const when = saved.savedAt !== "" ? `, saved ${saved.savedAt}` : "";
    const gateway = saved.baseUrl !== undefined ? `, gateway ${saved.baseUrl}` : "";
    io.out(`${verb}: key${name} ${maskKey(saved.hostedKey)}${when}${gateway}\n  ${path}\n`);
  }
  if (hosted === "" && local === "" && saved === null) {
    io.out(`Not logged in. Run \`exp-code login\`, or set ${GATEWAY_ENV.hostedKey}.\n`);
    return 1;
  }
  return 0;
}

/** Parse `login` / `logout` / `whoami` invocations; `undefined` when the arguments are not one. */
export type AuthCommand = { kind: "login"; noBrowser: boolean } | { kind: "logout" } | { kind: "status" } | { kind: "help"; text: string };

const LOGIN_HELP = `Usage:
  exp-code login              Approve exp-code in your browser and save the key
  exp-code login --no-browser Print the approval URL and paste the key it shows
  exp-code login --status     Show which key exp-code uses (also: exp-code whoami)
  exp-code logout             Remove the saved key from this computer

The key is saved in ~/.exp-code/agent/credentials.json (owner-only). EXP_API_KEY or
EXP_GATEWAY_KEY in the environment take precedence over it. EXP_PLATFORM_URL points login at
another platform origin, EXP_CODE_BASE_URL at another gateway.
`;

export function parseAuthCommand(args: readonly string[]): AuthCommand | { kind: "error"; message: string } | undefined {
  const [first, ...rest] = args;
  if (first === "whoami") return rest.length === 0 ? { kind: "status" } : { kind: "error", message: "exp-code whoami takes no options." };
  if (first === "logout") return rest.length === 0 ? { kind: "logout" } : { kind: "error", message: "exp-code logout takes no options." };
  if (first !== "login") return undefined;
  let noBrowser = false;
  let status = false;
  for (const option of rest) {
    if (option === "--no-browser") noBrowser = true;
    else if (option === "--status") status = true;
    else if (option === "--help" || option === "-h") return { kind: "help", text: LOGIN_HELP };
    else return { kind: "error", message: `Unknown option for exp-code login: ${option}\n\n${LOGIN_HELP}` };
  }
  if (status && noBrowser) return { kind: "error", message: "Use --status or --no-browser, not both." };
  return status ? { kind: "status" } : { kind: "login", noBrowser };
}

/** Run login with the real terminal: Ctrl+C aborts and the listener is always closed. */
export async function runLoginInTerminal(env: NodeJS.ProcessEnv, noBrowser: boolean): Promise<number> {
  const controller = new AbortController();
  const onSigint = () => controller.abort();
  process.once("SIGINT", onSigint);
  try {
    return await login(
      { env, stdin: process.stdin, out: (t) => process.stdout.write(t), err: (t) => process.stderr.write(t), signal: controller.signal },
      { noBrowser }
    );
  } finally {
    process.off("SIGINT", onSigint);
  }
}

/** Run a parsed auth command. Resolves the process exit code. */
export async function runAuthCommand(command: AuthCommand | { kind: "error"; message: string }, env: NodeJS.ProcessEnv): Promise<number> {
  const out = (text: string) => process.stdout.write(text);
  switch (command.kind) {
    case "error":
      process.stderr.write(`${command.message}\n`);
      return 2;
    case "help":
      out(command.text);
      return 0;
    case "status":
      return loginStatus({ env, out });
    case "logout":
      return logout({ env, out });
    case "login":
      return runLoginInTerminal(env, command.noBrowser);
  }
}
