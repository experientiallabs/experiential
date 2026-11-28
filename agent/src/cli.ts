#!/usr/bin/env node
/**
 * The `exp-code` binary. Pi's modules are patched as they load (pi-patches.ts) and the
 * environment is prepared before Pi is imported, so Pi reads exp-code's names and configuration
 * directory from its first access.
 */

import { applyCliEnvironment } from "./cli/environment.js";
import { installPiPatches } from "./cli/pi-patches.js";

installPiPatches();
applyCliEnvironment(process.env);
const { runCli } = await import("./cli/run.js");
await runCli(process.argv.slice(2));
