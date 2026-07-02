"use client";

import { useRouter } from "next/navigation";

/** Stackwise's "Take me to a random stack ->", for world models. */
export function RandomModelLink({ names }: { names: string[] }) {
  const router = useRouter();
  if (names.length === 0) return null;
  return (
    <button
      onClick={() => {
        const name = names[Math.floor(Math.random() * names.length)];
        router.push(`/models/${encodeURIComponent(name)}`);
      }}
      className="text-base font-semibold underline underline-offset-4 hover:text-accent"
    >
      Take me to a random world model -&gt;
    </button>
  );
}
