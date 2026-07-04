"use client";

/**
 * The standardized model page body: an interaction area (Play or Traces) with the model's card
 * beside it on desktop and collapsible below it on mobile. Every world model plugs into this one
 * interface via its index entry; there is no per-model UI code.
 */

import { useEffect, useState } from "react";
import { isServeUp } from "@/lib/api";
import type { IndexEntry, ModelCard } from "@/lib/types";
import { ModelRecord } from "./ModelRecord";
import { Playground } from "./Playground";
import { ServeControls } from "./ServeControls";
import { ServeDownPanel } from "./ServeDownPanel";
import { TracesExplorer } from "./TracesExplorer";

type Tab = "play" | "traces";

export function ModelView({
  entry,
  serveHint,
}: {
  entry: IndexEntry;
  serveHint: string;
}) {
  const card: ModelCard = entry.card;
  const [tab, setTab] = useState<Tab>("play");
  const [maxFidelity, setMaxFidelity] = useState(false);
  const [serveUp, setServeUp] = useState<boolean | null>(null);
  // A max-fidelity serve is started with the extra flag (server-level, per WS-A3 #55), so it is
  // surfaced through the serve command the user copies rather than a per-session switch.
  const effectiveHint = maxFidelity ? `${serveHint} --max-fidelity` : serveHint;

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
      <Playground entry={entry} />
    ) : (
      <TracesExplorer entry={entry} />
    );

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex rounded-lg border border-line p-0.5 text-sm">
          {(["play", "traces"] as const).map((key) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={`rounded-md px-4 py-1.5 capitalize transition-colors ${
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

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_300px]">
        <div className="min-w-0">{interaction}</div>
        <aside className="hidden lg:block">
          <div className="sticky top-6">
            <ModelRecord card={card} />
          </div>
        </aside>
      </div>

      {/* On mobile the card sits below the interaction, collapsed by default. */}
      <details className="lg:hidden">
        <summary className="mono-label cursor-pointer select-none py-2">model details</summary>
        <ModelRecord card={card} />
      </details>
    </div>
  );
}
