/**
 * Pure presentational block for one model's db record: title, tags, description, stat row,
 * fidelity provenance. Shared by the static page (index-backed) and the live-API fallback,
 * so a model renders identically wherever its card came from.
 */

import type { ModelCard } from "@/lib/types";

function pct(value: number | null | undefined): string {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function Stat({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="mono-label">{label}</div>
      <div className="truncate text-sm" title={title ?? value}>
        {value}
      </div>
    </div>
  );
}

export function ModelRecord({
  card,
  heldOutAccuracy,
}: {
  card: ModelCard;
  heldOutAccuracy: number | null;
}) {
  return (
    <>
      <header className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-semibold tracking-tight">{card.title}</h1>
          {card.tags.map((tag) => (
            <span
              key={tag}
              className="rounded-full border border-line px-2 py-0.5 text-xs text-ink-soft"
            >
              {tag}
            </span>
          ))}
        </div>
        <p className="max-w-3xl text-ink-soft">{card.description}</p>
      </header>

      <section className="grid grid-cols-2 gap-x-8 gap-y-4 rounded-lg border border-line p-5 sm:grid-cols-3 lg:grid-cols-6">
        <Stat
          label="fidelity"
          value={pct(card.fidelity?.score)}
          title={card.fidelity ? `${card.fidelity.suite} — ${card.fidelity.run_id ?? ""}` : undefined}
        />
        <Stat label="build accuracy" value={pct(heldOutAccuracy)} />
        <Stat
          label="corpus"
          value={`${card.corpus.traces ?? "—"} traces / ${card.corpus.steps} steps`}
        />
        <Stat label="serve LLM" value={card.model_id} />
        <Stat label="provider" value={card.provider} />
        <Stat label="built" value={card.built_at ? card.built_at.slice(0, 10) : "—"} />
      </section>
      {card.fidelity && (
        <p className="-mt-4 text-xs text-ink-faint">
          Fidelity measured by {card.fidelity.suite}
          {card.fidelity.std != null && <> (±{card.fidelity.std})</>}; provenance:{" "}
          <code className="font-mono">{card.fidelity.run_id}</code>
        </p>
      )}
    </>
  );
}
