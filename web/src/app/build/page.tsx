import { BuildFlow } from "@/components/BuildFlow";
import { Wordmark } from "@/components/Wordmark";
import { serveCommand } from "@/lib/index-data";

export const metadata = { title: "Build your own — world-model-harness" };

export default function BuildPage() {
  return (
    <div className="flex flex-col gap-8">
      <header className="flex flex-col items-center gap-3 pt-10 text-center">
        <Wordmark />
        <h1 className="text-2xl font-semibold tracking-tight">Build your own world model</h1>
        <p className="max-w-2xl text-ink-soft">
          Point at your agent traces (OpenTelemetry GenAI JSONL), pick the LLM that serves the
          environment, and watch the build: ingest → split → index → optimize. The build runs on
          your local <code className="font-mono">wmh serve</code> — traces and keys stay on your
          machine.
        </p>
      </header>
      <BuildFlow serveHint={serveCommand()} />
    </div>
  );
}
