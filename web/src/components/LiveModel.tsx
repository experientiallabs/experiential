"use client";

/**
 * Fallback for models not in the generated index (e.g. just built via /build): fetch the card
 * from the live `wmh serve` API and render the same standardized view.
 */

import Link from "next/link";
import { useEffect, useState } from "react";
import { listModels } from "@/lib/api";
import type { ModelCard } from "@/lib/types";
import { ModelView } from "./ModelView";

type State =
  | { status: "loading" }
  | { status: "offline" } // backend unreachable
  | { status: "notfound" } // backend up, no such model
  | { status: "found"; card: ModelCard };

export function LiveModel({ name, serveHint }: { name: string; serveHint: string }) {
  const [state, setState] = useState<State>({ status: "loading" });

  useEffect(() => {
    listModels()
      .then((res) => {
        const entry = res.models.find((m) => m.name === name);
        if (!entry) {
          setState({ status: "notfound" });
          return;
        }
        setState({
          status: "found",
          card: entry.card ?? {
            schema_version: 1,
            name,
            title: name,
            description: "",
            corpus: { traces: null, steps: 0 },
            provider: "unknown",
            model_id: "unknown",
            tags: [],
          },
        });
      })
      // A rejected fetch means the backend is unreachable — NOT that the model is missing.
      .catch(() => setState({ status: "offline" }));
  }, [name]);

  if (state.status === "loading") {
    return (
      <div className="rounded-lg border border-line p-5 text-center text-sm text-ink-faint">
        Looking up {name} on the local backend…
      </div>
    );
  }
  if (state.status === "offline") {
    return (
      <div className="flex flex-col gap-3 rounded-lg border border-line bg-surface-sunk p-5">
        <div className="mono-label">backend offline</div>
        <p className="text-sm text-ink-soft">
          Can&apos;t reach a local <code className="font-mono">wmh serve</code>. Start one, then
          reload:
        </p>
        <pre className="overflow-x-auto rounded-md border border-line bg-surface p-3 font-mono text-xs">
          {serveHint}
        </pre>
      </div>
    );
  }
  if (state.status === "notfound") {
    return (
      <div className="flex flex-col items-center gap-3 rounded-lg border border-line p-5">
        <p className="text-sm text-ink-soft">
          No model named <code className="font-mono">{name}</code> in this gallery or on the
          local backend.
        </p>
        <Link href="/" className="text-sm text-accent hover:underline">
          ← Back to models
        </Link>
      </div>
    );
  }
  const { card } = state;
  return <ModelView card={card} heldOutAccuracy={null} serveHint={serveHint} />;
}
