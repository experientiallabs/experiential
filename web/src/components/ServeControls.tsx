"use client";

import { useState } from "react";

/**
 * The serve-side controls above the interaction: a max-fidelity toggle and a copy button for the
 * serve command. Max fidelity is a server-level mode (WS-A3, PR #55): turning it on here appends
 * `--max-fidelity` to the command you copy, so a fresh serve runs the reasoning + verification
 * extras. It is a no-op against a plain serve until that build lands.
 */
export function ServeControls({
  serveHint,
  maxFidelity,
  onToggleMaxFidelity,
}: {
  serveHint: string;
  maxFidelity: boolean;
  onToggleMaxFidelity: () => void;
}) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="flex items-center gap-2">
      <button
        onClick={onToggleMaxFidelity}
        role="switch"
        aria-checked={maxFidelity}
        title="Higher-fidelity mode: reasoning + self-verification. Requires a serve started with --max-fidelity (WS-A3)."
        className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs transition-colors ${
          maxFidelity
            ? "border-accent-teal bg-accent-teal/10 text-ink"
            : "border-line text-ink-soft hover:border-ink"
        }`}
      >
        <span
          className={`inline-block h-1.5 w-1.5 rounded-full ${
            maxFidelity ? "bg-accent-teal" : "bg-line"
          }`}
        />
        max fidelity
      </button>
      <button
        onClick={() => {
          navigator.clipboard?.writeText(serveHint).then(
            () => {
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            },
            () => {},
          );
        }}
        className="rounded-md border border-line px-3 py-1.5 font-mono text-xs text-ink-soft hover:border-ink"
        title={serveHint}
      >
        {copied ? "copied" : "copy serve command"}
      </button>
    </div>
  );
}
