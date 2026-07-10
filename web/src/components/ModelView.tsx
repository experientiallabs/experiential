"use client";

/**
 * The standardized model page body: an interaction area (Play or Traces) with the model's card
 * beside it on desktop and collapsible below it on mobile. The live scratchpad and usage live in
 * the side panel as mutually exclusive dropdowns, so only one is ever open and the page does not
 * grow enough to scroll. Every world model plugs into this one interface via its index entry.
 */

import { useCallback, useEffect, useState } from "react";
import { FidelityGrid } from "@/components/FidelityGrid";
import { LivePanels, type LiveState, type OpenPanel } from "@/components/LivePanels";
import { isServeUp } from "@/lib/api";
import type { IndexEntry } from "@/lib/types";
import { ModelRecord } from "./ModelRecord";
import { Playground } from "./Playground";
import { ServeControls } from "./ServeControls";
import { ServeDownPanel } from "./ServeDownPanel";
import { TracesExplorer } from "./TracesExplorer";

type Tab = "play" | "traces";

export function ModelView({ entry, serveHint }: { entry: IndexEntry; serveHint: string }) {
  const [tab, setTab] = useState<Tab>("play");
  const [maxFidelity, setMaxFidelity] = useState(false);
  const [serveUp, setServeUp] = useState<boolean | null>(null);
  const [live, setLive] = useState<LiveState | null>(null);
  const [openPanel, setOpenPanel] = useState<OpenPanel>(null);
  // A max-fidelity serve is started with the extra flag (server-level, per WS-A3 #55), so it is
  // surfaced through the serve command the user copies rather than a per-session switch.
  const effectiveHint = maxFidelity ? `${serveHint} --max-fidelity` : serveHint;

  // When the live session goes away (ended, or the Play tab unmounted), collapse the panels so a
  // stale selection can't leave the remaining panel rendered-but-closed.
  const onLive = useCallback((next: LiveState | null) => {
    setLive(next);
    if (!next) setOpenPanel(null);
  }, []);

  useEffect(() => {
    isServeUp().then(setServeUp);
  }, []);

  const interaction =
    serveUp === false ? (
      <ServeDownPanel serveHint={effectiveHint} />
    ) : serveUp === null ? (
      <div className="rounded-xl border border-line p-6 text-sm text-ink-faint">
        Checking for a local backend...
      </div>
    ) : tab === "play" ? (
      <Playground entry={entry} onLive={onLive} />
    ) : (
      <TracesExplorer entry={entry} />
    );

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex rounded-lg border border-line p-0.5 text-sm">
          {(["play", "traces"] as const).map((key) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={`rounded-md px-4 py-1.5 transition-colors ${
                tab === key ? "bg-ink text-white" : "text-ink-soft hover:text-ink"
              }`}
            >
              {key === "traces" ? "Explore traces" : "Playground"}
            </button>
          ))}
        </div>
        <ServeControls
          serveHint={effectiveHint}
          maxFidelity={maxFidelity}
          onToggleMaxFidelity={() => setMaxFidelity((v) => !v)}
        />
      </div>

      {/* Max fidelity is a "powered up" mode: a teal wave fills the band when it is on. */}
      {maxFidelity && (
        <div className="flex items-center gap-3 rounded-lg border border-accent-teal/40 bg-surface px-3 py-1.5">
          <FidelityGrid className="h-6 flex-1" />
          <span className="mono-label shrink-0 text-ink-soft">max fidelity</span>
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_300px]">
        <div className="min-w-0">{interaction}</div>
        <aside className="hidden lg:block">
          <div className="sticky top-6 flex flex-col gap-3">
            <ModelRecord card={entry.card} />
            <LivePanels live={live} open={openPanel} setOpen={setOpenPanel} />
          </div>
        </aside>
      </div>

      {/* On mobile the card and live panels sit below the interaction, collapsed by default. */}
      <details className="lg:hidden">
        <summary className="mono-label cursor-pointer select-none py-2">model details</summary>
        <div className="flex flex-col gap-3">
          <ModelRecord card={entry.card} />
          <LivePanels live={live} open={openPanel} setOpen={setOpenPanel} />
        </div>
      </details>
    </div>
  );
}
