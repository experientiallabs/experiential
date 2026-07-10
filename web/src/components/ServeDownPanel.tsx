import { API_BASE } from "@/lib/api";

/** Shown when no local `wmh serve` answers: the exact command to start one. */
export function ServeDownPanel({ serveHint }: { serveHint: string }) {
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-line bg-surface-sunk p-6">
      <div className="mono-label">backend offline</div>
      <p className="text-sm text-ink-soft">
        No <code className="font-mono">wmh serve</code> is answering at{" "}
        <code className="font-mono">{API_BASE}</code>. From the repo root, run:
      </p>
      <pre className="overflow-x-auto rounded-lg border border-line bg-surface p-3 font-mono text-xs">
        {serveHint}
      </pre>
      <p className="text-xs text-ink-faint">
        Then reload this page. Your traces and provider keys never leave your machine.
      </p>
    </div>
  );
}
