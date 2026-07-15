/** A small inline spinner for pending states. */
export function Spinner({ className }: { className?: string }) {
  return (
    <span
      className={`inline-block h-3 w-3 shrink-0 animate-spin rounded-full border border-ink-faint border-t-transparent ${className ?? ""}`}
      aria-hidden
    />
  );
}
