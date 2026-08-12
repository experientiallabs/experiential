import { spawn } from "node:child_process";
import process from "node:process";

const [mode, ...arguments_] = process.argv.slice(2);
if (!new Set(["dev", "start"]).has(mode)) {
  fail("Usage: next-local.mjs <dev|start> [Next options]");
}
if (
  arguments_.some(
    (value) =>
      value === "--hostname" ||
      value.startsWith("--hostname=") ||
      value === "-H" ||
      value.startsWith("-H=")
  )
) {
  fail("The local review hostname is fixed to 127.0.0.1 and cannot be overridden.");
}

const next = spawn(
  process.execPath,
  ["node_modules/next/dist/bin/next", mode, ...arguments_, "--hostname", "127.0.0.1"],
  { stdio: "inherit" }
);
next.once("error", (error) => fail(error.message));
next.once("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
  } else {
    process.exit(code ?? 1);
  }
});

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(2);
}
