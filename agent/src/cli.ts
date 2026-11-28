#!/usr/bin/env node
/**
 * The `exp-code` binary. Pi's modules are patched as they load (pi-patches.ts) and the
 * environment is prepared before Pi is imported, so Pi reads exp-code's names and configuration
 * directory from its first access.
 */

import { captureEarlyInput, mayOpenTerminalUi } from "./cli/early-input.js";
import { applyCliEnvironment } from "./cli/environment.js";
import { installPiPatches } from "./cli/pi-patches.js";

const args = process.argv.slice(2);
// Keys typed while exp-code loads are kept for the terminal UI (early-input.ts).
const earlyInput = mayOpenTerminalUi(args, process.stdin, process.stdout) ? captureEarlyInput() : undefined;
installPiPatches();
applyCliEnvironment(process.env);
const { runCli } = await import("./cli/run.js");
await runCli(args, process.env, earlyInput);
