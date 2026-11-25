/**
 * `experiential_evals`: the organization's evaluation results and A/B (blind comparison) tests.
 *
 * Reads `GET /api/orgs/{org}/evaluations/results` (imported eval results) and
 * `GET /api/orgs/{org}/evaluations/ab-tests` (tests with their runs), or one test by id. Both need
 * the organization's provisioning key and access to Evals; the tool says so on a refusal.
 */

import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { type PlatformClient, PlatformError } from "./client.js";
import { failure, failureFrom, result, table, timestamp } from "./format.js";

interface ImportedResult {
  report_id: string;
  title: string;
  generated_at: string;
  models: number;
  origin?: { source?: string; label?: string | null; url?: string | null; imported_at?: string | null } | null;
}

interface AbRun {
  id: string;
  name: string;
  status: string;
  completed: number;
  total: number;
  created_at: string;
  scored: number;
}

interface AbTest {
  id: string;
  name: string;
  visibility: string;
  reveal: string;
  version: number;
  questions: number;
  closed: boolean;
  created_at: string;
  runs: AbRun[];
  configuration?: { models?: string[] } | null;
}

const evalsSchema = Type.Object({
  kind: Type.Optional(
    Type.Union([Type.Literal("all"), Type.Literal("results"), Type.Literal("ab_tests")], {
      description: "results (imported eval results), ab_tests (blind A/B comparison tests and their runs), or all (default)"
    })
  ),
  test_id: Type.Optional(Type.String({ description: "Read one A/B test by id (from a previous listing)" }))
});

function refusalHint(error: unknown): unknown {
  if (error instanceof PlatformError && (error.status === 403 || error.status === 404)) {
    return new PlatformError(`${error.message} Evaluations need the organization's provisioning key and access to Evals.`, error.status, error.code);
  }
  return error;
}

function resultsSection(results: ImportedResult[]): string {
  if (results.length === 0) return "Eval results: none imported.";
  return `Eval results (${results.length}, newest first):\n${table(
    ["generated (UTC)", "title", "models", "source", "report id"],
    results.map((row) => [timestamp(row.generated_at), row.title || "-", String(row.models), [row.origin?.source, row.origin?.label].filter(Boolean).join(": ") || "-", row.report_id])
  )}`;
}

function runsText(runs: AbRun[]): string {
  if (runs.length === 0) return "no runs";
  return runs.map((run) => `${run.name} ${run.status} ${run.completed}/${run.total}, ${run.scored} scored`).join("; ");
}

function testsSection(tests: AbTest[]): string {
  if (tests.length === 0) return "A/B tests: none.";
  return `A/B tests (${tests.length}):\n${table(
    ["created (UTC)", "name", "models", "questions", "state", "runs", "test id"],
    tests.map((test) => [
      timestamp(test.created_at),
      test.name,
      test.configuration?.models?.join(" vs ") ?? "-",
      String(test.questions),
      test.closed ? "closed" : "open",
      runsText(test.runs ?? []),
      test.id
    ])
  )}`;
}

export function createEvalsTool(client: PlatformClient) {
  return defineTool({
    name: "experiential_evals",
    label: "Experiential evals",
    description:
      "List this organization's Experiential evaluations: imported eval results (title, models compared, source) and blind A/B model comparison tests with their runs and progress, or one A/B test by id. Use it when the user asks how models compared in their evals or about an A/B test's status. Needs the organization's provisioning key.",
    promptSnippet: "List the organization's Experiential eval results and A/B tests",
    parameters: evalsSchema,
    annotations: { readOnlyHint: true, openWorldHint: true },
    async execute(_toolCallId, args, signal) {
      const kind = args.kind ?? "all";
      if (!["all", "results", "ab_tests"].includes(kind)) return failure("kind must be all, results or ab_tests.");
      try {
        if (args.test_id?.trim()) {
          const id = args.test_id.trim();
          if (!/^[0-9a-f-]{36}$/i.test(id)) return failure("test_id must be an A/B test id (a uuid).");
          const test = await client.get<AbTest>(await client.orgPath(`/evaluations/ab-tests/${id}`, signal), {}, signal).catch((error: unknown) => {
            if (error instanceof PlatformError && error.status === 404) throw new PlatformError(`No A/B test ${id} is visible to this key. List tests with kind=ab_tests.`, 404);
            throw refusalHint(error);
          });
          return result(testsSection([test]), { tests: [test] });
        }
        type Read<T> = { ok: true; value: T } | { ok: false; error: unknown } | undefined;
        const read = async <T>(wanted: boolean, suffix: string): Promise<Read<T>> => {
          if (!wanted) return undefined;
          try {
            return { ok: true, value: await client.get<T>(await client.orgPath(suffix, signal), {}, signal) };
          } catch (error) {
            return { ok: false, error: refusalHint(error) };
          }
        };
        const [results, tests] = await Promise.all([
          read<{ results?: ImportedResult[] }>(kind !== "ab_tests", "/evaluations/results"),
          read<{ tests?: AbTest[] }>(kind !== "results", "/evaluations/ab-tests")
        ]);
        const asked = [results, tests].filter((entry) => entry !== undefined);
        const failed = asked.filter((entry) => !entry.ok);
        if (failed.length === asked.length && failed[0] !== undefined && !failed[0].ok) throw failed[0].error;
        const message = (error: unknown) => (error instanceof Error ? error.message : String(error));
        const resultRows = results?.ok ? (results.value?.results ?? []) : [];
        const testRows = tests?.ok ? (tests.value?.tests ?? []) : [];
        const sections: string[] = [];
        if (results !== undefined) sections.push(results.ok ? resultsSection(resultRows) : `Eval results unavailable: ${message(results.error)}`);
        if (tests !== undefined) sections.push(tests.ok ? testsSection(testRows) : `A/B tests unavailable: ${message(tests.error)}`);
        return result(sections.join("\n\n"), { results: resultRows, tests: testRows });
      } catch (error) {
        return failureFrom(error, { results: [], tests: [] });
      }
    }
  });
}
