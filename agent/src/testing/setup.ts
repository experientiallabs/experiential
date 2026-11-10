/**
 * Test setup: remove provider credentials from the environment so no test can reach a real
 * model provider, whatever the developer's shell exports. `npm run test:live` (EXP_CODE_LIVE=1)
 * keeps EXP_API_KEY for the live smoke only, under a name no other code reads.
 */

if (process.env.EXP_CODE_LIVE === "1" && process.env.EXP_API_KEY) process.env.EXP_CODE_LIVE_API_KEY = process.env.EXP_API_KEY;

for (const name of Object.keys(process.env)) {
  if (/(_API_KEY|_ACCESS_KEY_ID|_SECRET_ACCESS_KEY|_SESSION_TOKEN|_AUTH_TOKEN)$/.test(name) && name !== "EXP_CODE_LIVE_API_KEY") {
    delete process.env[name];
  }
}
process.env.PI_OFFLINE = "1";
process.env.PI_TELEMETRY = "0";
process.env.PI_SKIP_VERSION_CHECK = "1";
