import { afterEach, describe, expect, it } from "vitest";
import { type FakePlatform, startFakePlatform } from "../testing/fake-platform.js";
import { PlatformClient } from "./client.js";
import { createEvalsTool } from "./evals.js";
import { exec, text } from "../testing/tool-results.js";

let platform: FakePlatform | undefined;
afterEach(async () => {
  await platform?.close();
  platform = undefined;
});

const ORG = "/api/orgs/org-1111/evaluations";
const TEST_ID = "11111111-2222-3333-4444-555555555555";
const TEST = {
  id: TEST_ID,
  name: "Luna vs Sol",
  visibility: "organization",
  reveal: "after_review",
  version: 1,
  questions: 12,
  mine: true,
  can_manage: true,
  closed: false,
  created_at: "2026-11-20T10:00:00Z",
  runs: [{ id: "r1", name: "Run 1", status: "running", completed: 5, total: 12, created_at: "x", mine: true, can_manage: true, submitted: false, scored: 3 }],
  configuration: { models: ["gpt-6-luna", "gpt-6-sol"] }
};

async function tool(routes: Parameters<typeof startFakePlatform>[0]) {
  platform = await startFakePlatform(routes);
  return createEvalsTool(new PlatformClient({ origin: platform.origin, apiKey: "xpl_k" }));
}

describe("experiential_evals", () => {
  it("lists imported results and A/B tests", async () => {
    const evals = await tool({
      [`GET ${ORG}/results`]: { body: { results: [{ report_id: "imported-1", title: "SWE subset", generated_at: "2026-11-24T00:00:00Z", models: 3, origin: { source: "file", label: "results.json", ref: {}, url: null, imported_by: null, imported_at: null } }] } },
      [`GET ${ORG}/ab-tests`]: { body: { tests: [TEST], models: [], generation: 1, capture_on: true, execution_ready: true } }
    });
    const output = text(await exec(evals, {}));
    expect(output).toContain("| 2026-11-24 00:00:00Z | SWE subset | 3 | file: results.json | imported-1 |");
    expect(output).toContain("| Luna vs Sol | gpt-6-luna vs gpt-6-sol | 12 | open | Run 1 running 5/12, 3 scored |");
  });

  it("reads one kind or one test", async () => {
    const evals = await tool({ [`GET ${ORG}/ab-tests/${TEST_ID}`]: { body: TEST }, [`GET ${ORG}/results`]: { body: { results: [] } } });
    expect(text(await exec(evals, { kind: "results" }))).toBe("Eval results: none imported.");
    expect(platform!.calls(`${ORG}/ab-tests`)).toHaveLength(0);
    expect(text(await exec(evals, { test_id: TEST_ID }))).toContain("Luna vs Sol");
    const missing = await exec(evals, { test_id: "99999999-2222-3333-4444-555555555555" });
    expect(text(missing)).toContain("No A/B test 99999999-2222-3333-4444-555555555555 is visible to this key");
    expect((await exec(evals, { test_id: "bad" })).isError).toBe(true);
  });

  it("names the provisioning key when every read is refused, and keeps a partial answer", async () => {
    const evals = await tool({ [`GET ${ORG}/results`]: { status: 403, body: { error: "Use the provisioning key" } }, [`GET ${ORG}/ab-tests`]: { status: 403, body: { error: "Use the provisioning key" } } });
    const refused = await exec(evals, {});
    expect(refused.isError).toBe(true);
    expect(text(refused)).toMatch(/403.*need the organization's provisioning key and access to Evals/);
    platform!.route(`GET ${ORG}/ab-tests`, { body: { tests: [] } });
    const partial = await exec(evals, {});
    expect(partial.isError).toBeUndefined();
    expect(text(partial)).toContain("Eval results unavailable:");
    expect(text(partial)).toContain("A/B tests: none.");
  });
});
