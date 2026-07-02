import { LiveModel } from "@/components/LiveModel";
import { ModelView } from "@/components/ModelView";
import { Wordmark } from "@/components/Wordmark";
import { allModels, findModel, serveCommand } from "@/lib/index-data";

export function generateStaticParams() {
  return allModels().map((entry) => ({ name: entry.card.name }));
}

export default async function ModelPage({
  params,
}: {
  params: Promise<{ name: string }>;
}) {
  const { name } = await params;
  const decoded = decodeURIComponent(name);
  const entry = findModel(decoded);

  return (
    <div className="flex flex-col gap-8">
      <header className="flex flex-col items-center gap-3 pt-10 text-center">
        <Wordmark />
        <h1 className="text-2xl font-semibold tracking-tight">
          {entry ? entry.card.title : decoded}
        </h1>
        {entry && <p className="max-w-2xl text-ink-soft">{entry.card.description}</p>}
      </header>
      {entry ? (
        <ModelView
          card={entry.card}
          heldOutAccuracy={entry.held_out_accuracy}
          serveHint={serveCommand()}
        />
      ) : (
        // Not in the generated index — e.g. freshly built via /build; ask the live backend.
        <LiveModel name={decoded} serveHint={serveCommand()} />
      )}
    </div>
  );
}
