#!/usr/bin/env node
/**
 * The `exp-code` binary. The environment is prepared before Pi is imported, so Pi reads
 * exp-code's configuration directory from its first access.
 */

import { applyCliEnvironment } from "./cli/environment.js";

applyCliEnvironment(process.env);
const { runCli } = await import("./cli/run.js");
await runCli(process.argv.slice(2));
