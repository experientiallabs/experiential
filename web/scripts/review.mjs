import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repositoryDir = path.resolve(webDir, "..");
const options = parseOptions(process.argv.slice(2));
const adapterUrl = `http://127.0.0.1:${options.apiPort}`;

const adapter = spawn(
  "uv",
  [
    "run",
    "python",
    "-m",
    "wmo.review_server",
    "--root",
    options.root,
    "--project",
    options.project,
    "--host",
    "127.0.0.1",
    "--port",
    String(options.apiPort)
  ],
  { cwd: repositoryDir, stdio: "inherit" }
);
const web = spawn("npm", ["run", "dev", "--", "--port", String(options.port)], {
  cwd: webDir,
  env: { ...process.env, WMO_REVIEW_API_URL: adapterUrl },
  stdio: "inherit"
});

let stopping = false;

function stop(exitCode = 0) {
  if (stopping) {
    return;
  }
  stopping = true;
  adapter.kill("SIGTERM");
  web.kill("SIGTERM");
  process.exitCode = exitCode;
}

adapter.once("error", () => stop(1));
web.once("error", () => stop(1));
adapter.once("exit", (code) => {
  if (!stopping) {
    stop(code ?? 1);
  }
});
web.once("exit", (code) => {
  if (!stopping) {
    stop(code ?? 1);
  }
});
process.once("SIGINT", () => stop());
process.once("SIGTERM", () => stop());

function parseOptions(args) {
  const options = {
    root: path.join(repositoryDir, ".wmo"),
    project: "default",
    apiPort: 8017,
    port: 3000
  };
  for (let index = 0; index < args.length; index += 1) {
    const value = args[index];
    if (value === "--root") {
      options.root = requiredValue(args, ++index, value);
    } else if (value === "--project") {
      options.project = requiredValue(args, ++index, value);
    } else if (value === "--api-port") {
      options.apiPort = portValue(requiredValue(args, ++index, value), value);
    } else if (value === "--port") {
      options.port = portValue(requiredValue(args, ++index, value), value);
    } else {
      throw new Error(`Unknown option: ${value}`);
    }
  }
  return options;
}

function requiredValue(args, index, option) {
  const value = args[index];
  if (!value || value.startsWith("--")) {
    throw new Error(`${option} requires a value`);
  }
  return value;
}

function portValue(value, option) {
  const port = Number(value);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error(`${option} must be an integer from 1 through 65535`);
  }
  return port;
}
