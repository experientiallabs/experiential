"use client";

/**
 * The standardized model page body, stackwise-style: a control bar toggling between the
 * interactive playground (default, front and center) and the card record, plus a copy button
 * for the serve command. Every world model plugs into this one interface via its card —
 * there is no per-model UI code.
 */

import { useState } from "react";
import type { ModelCard } from "@/lib/types";
import { ModelRecord } from "./ModelRecord";
import { Playground } from "./Playground";

function CopyServeCommand({ serveHint }: { serveHint: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => {
        navigator.clipboard.writeText(serveHint).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        });
      }}
      className="rounded-md border border-line px-3 py-1.5 font-mono text-xs text-ink-soft hover:border-accent"
      title={serveHint}
    >
      {copied ? "copied ✓" : "copy serve command"}
    </button>
  );
}

export function ModelView({
  card,
  heldOutAccuracy,
  serveHint,
}: {
  card: ModelCard;
  heldOutAccuracy: number | null;
  serveHint: string;
}) {
  const [view, setView] = useState<"playground" | "card">("playground");
  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between gap-3">
        <div className="flex rounded-md border border-line p-0.5 text-sm">
          {(["playground", "card"] as const).map((key) => (
            <button
              key={key}
              onClick={() => setView(key)}
              className={`rounded px-3 py-1 capitalize ${
                view === key ? "bg-ink text-white" : "text-ink-soft hover:text-ink"
              }`}
            >
              {key}
            </button>
          ))}
        </div>
        <CopyServeCommand serveHint={serveHint} />
      </div>
      {view === "playground" ? (
        <Playground name={card.name} task={card.task ?? null} serveHint={serveHint} />
      ) : (
        <div className="flex flex-col gap-8">
          <ModelRecord card={card} heldOutAccuracy={heldOutAccuracy} />
        </div>
      )}
    </div>
  );
}
