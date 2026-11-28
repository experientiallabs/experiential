import { describe, expect, it } from "vitest";
import { emptyFigures, type SessionUsage } from "./aggregate.js";
import { formatCents, formatCredits, formatDuration, formatNano, formatSpeed, renderUsageReport } from "./format.js";

describe("money", () => {
  it("formats nano-dollars like the platform, from the integer", () => {
    expect(formatNano(0)).toBe("$0");
    expect(formatNano(53_300_000)).toBe("$0.0533");
    expect(formatNano(912_000)).toBe("$0.000912");
    expect(formatNano(1_234_500_000)).toBe("$1.235");
    expect(formatNano(123_456_000_000)).toBe("$123.46");
    expect(formatNano(1_234_567_890_000)).toBe("$1,234.57");
    expect(formatNano(37)).toBe("$0.000000037");
    expect(formatNano(-53_300_000)).toBe("-$0.0533");
    expect(formatNano(Number.NaN)).toBe("-");
  });

  it("formats balances as cents and credits", () => {
    expect(formatCents(17_345_700_000)).toBe("$17.35");
    expect(formatCredits(17_345_700_000)).toBe("1,734.57 credits");
    expect(formatCredits(-5_000_000)).toBe("-0.50 credits");
  });
});

describe("durations and speed", () => {
  it("scales the unit", () => {
    expect(formatDuration(850)).toBe("850ms");
    expect(formatDuration(4200)).toBe("4.2s");
    expect(formatDuration(185_000)).toBe("3m 05s");
    expect(formatDuration(3_720_000)).toBe("1h 02m");
  });

  it("reports output tokens per second only when timed", () => {
    expect(formatSpeed({ timedMs: 2000, timedOutput: 100 })).toBe("50.0");
    expect(formatSpeed({ timedMs: 0, timedOutput: 0 })).toBe("-");
  });
});

function sessionUsage(overrides: Partial<SessionUsage> = {}): SessionUsage {
  return { sessionId: "01a21fa5-e567", sessionName: undefined, startedAt: 0, prompts: 0, totals: emptyFigures(), models: [], modelTimeMs: 0, toolTimeMs: 0, earlierCalls: 0, openedAt: 0, ...overrides };
}

const text = (lines: { text: string }[]) => lines.map((line) => line.text).join("\n");

describe("renderUsageReport", () => {
  it("says when no model calls were made", () => {
    const out = text(renderUsageReport(sessionUsage(), { now: 6000, context: { tokens: 0, contextWindow: 200_000, percent: 0, model: "m1" } }));
    expect(out).toContain("Usage · session 01a21fa5, started 1970-01-01 00:00Z");
    expect(out).toContain("No model calls yet");
    expect(out).toContain("0 / 200,000 tokens (0%) on m1");
    expect(out).not.toContain("Cost");
    expect(out).not.toContain("Model ");
  });

  it("lays out totals, the cost split and one aligned row per model", () => {
    const big = { ...emptyFigures(), calls: 1, input: 3100, output: 240, reasoning: 120, chargedNano: 53_300_000, timedMs: 409, timedOutput: 240 };
    const fast = { ...emptyFigures(), calls: 1, input: 400, cacheRead: 3800, output: 180, reasoning: 60, freeNano: 912_000, timedMs: 405, timedOutput: 180 };
    const totals = { ...emptyFigures(), calls: 2, input: 3500, cacheRead: 3800, output: 420, reasoning: 180, chargedNano: 53_300_000, freeNano: 912_000, byokListNano: 4_200_000 };
    const lines = renderUsageReport(
      sessionUsage({
        sessionName: "fix tests",
        prompts: 2,
        totals,
        modelTimeMs: 814,
        models: [
          { ...big, label: "experiential-large-model", provider: "experiential", model: "experiential-large-model" },
          { ...fast, label: "fast-model", provider: "experiential", model: "fast-model" }
        ]
      }),
      { now: 24_500, dashboardUrl: "https://platform.example/spend" }
    );
    const out = text(lines);
    expect(out).toContain('session 01a21fa5 "fix tests"');
    expect(out).toContain("24.5s since start · model 814ms");
    expect(out).toContain("2 prompts · 2 model calls");
    expect(out).toContain("7,300 input (3,800 cached) · 420 output (180 reasoning) · 7,720 total");
    expect(out).toContain("$0.0533 charged · $0.0042 on your own provider key (list value, not charged) · $0.000912 free (list value, not charged)");
    expect(out).toContain("History and per-request detail: https://platform.example/spend");
    const table = lines.filter((line) => line.style === "header" || /model\s+1\s/.test(line.text)).map((line) => line.text);
    expect(table).toEqual([
      "  Model                     Calls  Input  Cached  Output     Cost  Out tok/s",
      "  experiential-large-model      1  3,100       0     240  $0.0533      586.8",
      "  fast-model                    1  4,200   3,800     180     free      444.4"
    ]);
  });

  it("shows unpriced calls as '-' and marks partly priced rows", () => {
    const unpriced = { ...emptyFigures(), calls: 2, input: 10, unpricedCalls: 2 };
    const partly = { ...emptyFigures(), calls: 2, input: 10, unpricedCalls: 1, chargedNano: 10_000_000 };
    const totals = { ...emptyFigures(), calls: 4, input: 20, unpricedCalls: 3, chargedNano: 10_000_000 };
    const out = text(
      renderUsageReport(
        sessionUsage({ totals, models: [{ ...partly, label: "a", provider: "experiential", model: "a" }, { ...unpriced, label: "b", provider: "experiential", model: "b" }] }),
        { now: 1 }
      )
    );
    expect(out).toContain("$0.0100 charged · 3 unpriced calls not included");
    expect(out).toMatch(/a\s+2\s+10\s+0\s+0\s+\$0\.0100\*/);
    expect(out).toMatch(/b\s+2\s+10\s+0\s+0\s+-\s+-/);
    expect(out).toContain("* some calls on this model reported no price");
    const none = text(renderUsageReport(sessionUsage({ totals: unpriced }), { now: 1 }));
    expect(none).toContain("Cost          - (the gateway reported no price for these calls)");
  });

  it("names a resumed session, a streaming response and an unknown context", () => {
    const out = text(
      renderUsageReport(sessionUsage({ earlierCalls: 2, openedAt: 90_000, totals: { ...emptyFigures(), calls: 2 } }), {
        now: 100_000,
        streaming: true,
        context: { tokens: null, contextWindow: 128_000, percent: null }
      })
    );
    expect(out).toContain("Resumed session: includes 2 calls from before it was opened here.");
    expect(out).toContain("1m 40s since start · 10.0s since resumed");
    expect(out).toContain("A response is still streaming");
    expect(out).toContain("Context       unknown until the next response");
  });

  it("renders the organization section and its failures", () => {
    const ok = text(
      renderUsageReport(sessionUsage(), {
        now: 1,
        org: { kind: "available", orgName: "Acme", balanceNano: 17_345_700_000, today: { chargedNano: 1_234_500_000, requests: 42 }, notes: [] }
      })
    );
    expect(ok).toContain("Organization  Acme · 1,734.57 credits left ($17.35) · today (UTC) $1.235 charged over 42 requests");
    const partial = text(
      renderUsageReport(sessionUsage(), { now: 1, org: { kind: "available", orgName: "Acme", balanceNano: null, today: { chargedNano: 0, requests: 1 }, notes: ["Balance unavailable: nope"] } })
    );
    expect(partial).toContain("Acme · today (UTC) $0 charged over 1 request");
    expect(partial).toContain("Balance unavailable: nope");
    expect(text(renderUsageReport(sessionUsage(), { now: 1, org: { kind: "unavailable", reason: "not shown for a local exp gateway." } }))).toContain(
      "Organization: not shown for a local exp gateway."
    );
  });
});
