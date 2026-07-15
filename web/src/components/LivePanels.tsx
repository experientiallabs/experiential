"use client";

/**
 * The live session read-outs (scratchpad, usage) as mutually exclusive dropdowns. Only one is
 * open at a time so the panel never grows enough to scroll the page. The scratchpad section only
 * appears when the model actually wrote to it.
 */

import type { RunRecord } from "@/lib/types";

export type LiveState = { scratchpad: string; usage: RunRecord | null };
export type OpenPanel = "scratchpad" | "usage" | null;

function Section({
  id,
  label,
  summary,
  open,
  onToggle,
  children,
}: {
  id: OpenPanel;
  label: string;
  summary?: string;
  open: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-line">
      <button
        onClick={onToggle}
        aria-expanded={open}
        aria-controls={`panel-${id}`}
        className="flex w-full items-center justify-between gap-2 px-4 py-2.5"
      >
        <span className="mono-label">{label}</span>
        <span className="flex items-center gap-2">
          {summary && <span className="font-mono text-xs text-ink-soft">{summary}</span>}
          <span className={`text-ink-faint transition-transform ${open ? "rotate-90" : ""}`}>
            &rsaquo;
          </span>
        </span>
      </button>
      {open && (
        <div id={`panel-${id}`} className="border-t border-line px-4 py-3">
          {children}
        </div>
      )}
    </div>
  );
}

export function LivePanels({
  live,
  open,
  setOpen,
}: {
  live: LiveState | null;
  open: OpenPanel;
  setOpen: (p: OpenPanel) => void;
}) {
  if (!live) return null;
  const hasScratch = Boolean(live.scratchpad?.trim());
  const usage = live.usage;
  if (!hasScratch && !usage) return null;

  const toggle = (p: OpenPanel) => setOpen(open === p ? null : p);

  return (
    <div className="flex flex-col gap-2">
      {hasScratch && (
        <Section
          id="scratchpad"
          label="scratchpad"
          open={open === "scratchpad"}
          onToggle={() => toggle("scratchpad")}
        >
          <pre className="max-h-56 overflow-y-auto whitespace-pre-wrap font-mono text-xs text-ink-soft">
            {live.scratchpad}
          </pre>
        </Section>
      )}
      {usage && (
        <Section
          id="usage"
          label="session usage"
          summary={`${usage.total.calls} steps · $${usage.total.cost_usd.toFixed(4)}`}
          open={open === "usage"}
          onToggle={() => toggle("usage")}
        >
          <dl className="grid grid-cols-2 gap-y-1 text-xs">
            <dt className="text-ink-faint">steps</dt>
            <dd className="text-right tabular-nums">{usage.total.calls}</dd>
            <dt className="text-ink-faint">tokens</dt>
            <dd className="text-right tabular-nums">
              {(usage.total.input_tokens + usage.total.output_tokens).toLocaleString()}
            </dd>
            <dt className="text-ink-faint">cost</dt>
            <dd className="text-right tabular-nums">${usage.total.cost_usd.toFixed(4)}</dd>
            <dt className="text-ink-faint">wall clock</dt>
            <dd className="text-right tabular-nums">{usage.duration_seconds.toFixed(1)}s</dd>
          </dl>
        </Section>
      )}
    </div>
  );
}
