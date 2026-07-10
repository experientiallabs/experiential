import Link from "next/link";
import { Logo } from "./Logo";

/** Centered logo + wordmark, stackwise-style; links back to the gallery. */
export function Wordmark({ size = "md" }: { size?: "md" | "lg" }) {
  const glyph = size === "lg" ? "h-7 w-7" : "h-5 w-5";
  const text = size === "lg" ? "text-2xl" : "text-lg";
  return (
    <Link href="/" className="flex items-center justify-center gap-2">
      <Logo className={`${glyph} text-ink`} />
      <span className={`${text} font-semibold tracking-tight`}>world-model-harness</span>
    </Link>
  );
}
