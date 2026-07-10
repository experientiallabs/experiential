/**
 * The model's record, as a compact spec panel that sits beside the interaction. Shows the
 * numbers that describe the world model itself: reconstruction fidelity (with provenance),
 * corpus size, the serving LLM, and provenance. Build-time GEPA accuracy is intentionally not
 * shown here; it is a training-internal number that reads as a second, conflicting "accuracy".
 */

import type { ModelCard } from "@/lib/types";

function pct(value: number | null | undefined): string {
  return value == null ? "n/a" : `${(value * 100).toFixed(1)}%`;
}

function serveModelLabel(modelId: string, provider: string): string {
  if (modelId.includes("haiku")) return "Haiku 4.5";
  if (modelId.includes("opus")) return "Opus 4.8";
  if (modelId.includes("sonnet")) return "Sonnet 5";
  return provider;
}

function Row({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-t border-line py-2 first:border-t-0">
      <span className="mono-label shrink-0">{label}</span>
      <span className="truncate text-right text-sm text-ink" title={title ?? value}>
        {value}
      </span>
    </div>
  );
}

export function ModelRecord({ card }: { card: ModelCard }) {
  return (
    <div className="flex flex-col gap-4 rounded-xl border border-line p-5">
      {card.fidelity ? (
        <div className="flex flex-col gap-1">
          <span className="mono-label">reconstruction fidelity</span>
          <span className="text-2xl font-semibold tabular-nums tracking-tight">
            {pct(card.fidelity.score)}
          </span>
          <span className="text-xs leading-5 text-ink-faint">
            How closely the model&apos;s observations match the real environment&apos;s, on a
            held-out test slice
            {card.fidelity.std != null && ` (${"±"}${card.fidelity.std})`}.
          </span>
        </div>
      ) : (
        <div className="flex flex-col gap-1">
          <span className="mono-label">reconstruction fidelity</span>
          <span className="text-sm text-ink-faint">not yet evaluated</span>
        </div>
      )}

      <div className="flex flex-col">
        <Row
          label="corpus"
          value={`${card.corpus.traces ?? "n/a"} traces / ${card.corpus.steps} steps`}
        />
        <Row
          label="serve LLM"
          value={serveModelLabel(card.model_id, card.provider)}
          title={card.model_id}
        />
        <Row label="provider" value={card.provider} />
        {card.task && <Row label="task" value={card.task} />}
        {card.built_at && <Row label="built" value={card.built_at.slice(0, 10)} />}
        {card.license && <Row label="license" value={card.license} />}
      </div>

      {card.tags.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {card.tags.map((tag) => (
            <span
              key={tag}
              className="rounded-full border border-line px-2 py-0.5 text-xs text-ink-soft"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      {card.fidelity?.run_id && (
        <p className="text-[11px] leading-4 text-ink-faint">
          fidelity from <span className="font-mono">{card.fidelity.suite}</span>; provenance{" "}
          <span className="font-mono">{card.fidelity.run_id}</span>
        </p>
      )}
    </div>
  );
}
