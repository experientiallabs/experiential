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

export function LiveModel({ name, serveHint }: { name: string; serveHint: string }) {
  const [card, setCard] = useState<ModelCard | null | undefined>(undefined);

  useEffect(() => {
    listModels()
      .then((res) => {
        const entry = res.models.find((m) => m.name === name);
        setCard(
          entry?.card ??
            (entry
              ? {
                  schema_version: 1,
                  name,
                  title: name,
                  description: "",
                  corpus: { traces: null, steps: 0 },
                  provider: "unknown",
                  model_id: "unknown",
                  tags: [],
                }
              : null),
        );
      })
      .catch(() => setCard(null));
  }, [name]);

  if (card === undefined) {
    return (
      <div className="rounded-lg border border-line p-5 text-center text-sm text-ink-faint">
        Looking up {name} on the local backend…
      </div>
    );
  }
  if (card === null) {
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
  return <ModelView card={card} heldOutAccuracy={null} serveHint={serveHint} />;
}
