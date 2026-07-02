import Link from "next/link";
import { RandomModelLink } from "@/components/RandomModelLink";
import { Wordmark } from "@/components/Wordmark";
import { allModels } from "@/lib/index-data";
import type { IndexEntry } from "@/lib/types";

/** The tile's "screenshot": real steps from the model's replay index as a mini terminal. */
function TerminalPreview({ entry }: { entry: IndexEntry }) {
  return (
    <div className="flex h-36 flex-col gap-1 overflow-hidden rounded-md border border-line bg-surface-sunk p-3 font-mono text-[10px] leading-4">
      {entry.preview.length === 0 ? (
        <span className="text-ink-faint">(no sample steps)</span>
      ) : (
        entry.preview.map((step, i) => (
          <div key={i} className="flex flex-col">
            <span className="truncate text-accent">&gt; {step.action}</span>
            <span className="line-clamp-2 text-ink-soft">{step.observation}</span>
          </div>
        ))
      )}
    </div>
  );
}

function Star({ filled }: { filled: boolean }) {
  return (
    <svg
      viewBox="0 0 16 16"
      className={`h-4 w-4 shrink-0 ${filled ? "fill-accent-amber" : "fill-line"}`}
      aria-hidden
    >
      <path d="M8 .25a.75.75 0 0 1 .673.418l1.882 3.815 4.21.612a.75.75 0 0 1 .416 1.279l-3.046 2.97.719 4.192a.75.75 0 0 1-1.088.791L8 12.347l-3.766 1.98a.75.75 0 0 1-1.088-.79l.72-4.194L.818 6.374a.75.75 0 0 1 .416-1.28l4.21-.611L7.327.668A.75.75 0 0 1 8 .25Z" />
    </svg>
  );
}

function ModelTile({ entry }: { entry: IndexEntry }) {
  const { card } = entry;
  return (
    <Link
      href={`/models/${encodeURIComponent(card.name)}`}
      className="flex flex-col gap-3 rounded-lg border border-line bg-surface p-5 shadow-sm transition-shadow hover:shadow-md"
    >
      <div className="flex items-center justify-between gap-2">
        <h2 className="truncate font-semibold tracking-tight">{card.title}</h2>
        {/* starred = has a measured fidelity number */}
        <Star filled={card.fidelity != null} />
      </div>
      <TerminalPreview entry={entry} />
      <p className="line-clamp-2 text-sm leading-6 text-ink-soft">
        {card.description || "No description yet."}
      </p>
    </Link>
  );
}

function BuildTile() {
  return (
    <Link
      href="/build"
      className="flex flex-col gap-3 rounded-lg border border-dashed border-line bg-surface p-5 transition-colors hover:border-accent"
    >
      <div className="flex items-center justify-between gap-2">
        <h2 className="font-semibold tracking-tight">Build your own</h2>
        <span className="text-lg leading-none text-accent">+</span>
      </div>
      <div className="flex h-36 items-center justify-center rounded-md border border-dashed border-line font-mono text-[10px] text-ink-faint">
        traces.otel.jsonl → world model
      </div>
      <p className="line-clamp-2 text-sm leading-6 text-ink-soft">
        Turn your own agent traces into a steppable environment.
      </p>
    </Link>
  );
}

export default function GalleryPage() {
  const models = allModels();
  return (
    <div className="flex flex-col gap-14">
      <section className="flex flex-col items-center gap-4 pt-14 text-center">
        <Wordmark size="lg" />
        <p className="text-lg text-ink-soft">The open source world model collection.</p>
        <RandomModelLink names={models.map((m) => m.card.name)} />
      </section>
      <section className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {models.map((entry) => (
          <ModelTile key={entry.card.name} entry={entry} />
        ))}
        <BuildTile />
      </section>
    </div>
  );
}
