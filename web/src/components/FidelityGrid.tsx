"use client";

/**
 * An ambient grid of cells that pulse left to right in a wave, shown as a thin band when max
 * fidelity is on. Adapted from the resolute-agent onboarding grid; self-contained (measures its
 * container with a ResizeObserver) and tinted with the brand teal accent.
 */

import { useEffect, useRef, useState } from "react";

const CELL = 6;
const GAP = 3;
const WAVE = 14;
// Teal accent at rising alpha, level 0..4.
const COLORS = [
  "rgba(80,227,194,0.05)",
  "rgba(80,227,194,0.14)",
  "rgba(80,227,194,0.26)",
  "rgba(80,227,194,0.42)",
  "rgba(80,227,194,0.60)",
];

export function FidelityGrid({ className }: { className?: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const dims = useRef({ cols: 0, rows: 0 });
  const states = useRef<number[]>([]);
  const targets = useRef<number[]>([]);
  const waveCenter = useRef(-WAVE);
  const lastWave = useRef(0);
  const raf = useRef(0);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver(([e]) =>
      setSize({ w: e.contentRect.width, h: e.contentRect.height }),
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !size) return;
    const cols = Math.floor(size.w / (CELL + GAP));
    const rows = Math.max(1, Math.floor(size.h / (CELL + GAP)));
    if (cols === dims.current.cols && rows === dims.current.rows) return;
    dims.current = { cols, rows };
    canvas.width = cols * (CELL + GAP);
    canvas.height = rows * (CELL + GAP);
    states.current = new Array(cols * rows).fill(0);
    targets.current = new Array(cols * rows).fill(0);
  }, [size]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    function frame(time: number) {
      const { cols, rows } = dims.current;
      if (cols === 0 || rows === 0 || !ctx) {
        raf.current = requestAnimationFrame(frame);
        return;
      }
      if (time - lastWave.current > 2000) {
        lastWave.current = time;
        waveCenter.current = -WAVE;
      }
      waveCenter.current += 0.4;
      const total = cols * rows;
      for (let i = 0; i < total; i++) {
        const col = i % cols;
        const row = Math.floor(i / cols);
        const dist = Math.abs(Math.abs(col - waveCenter.current) + Math.sin(row * 0.9 + col * 0.3) * 2);
        if (dist < WAVE) {
          const intensity = 1 - dist / WAVE;
          const jitter = (Math.sin(i * 7.3 + time * 0.002) + 1) * 0.3;
          const level = Math.min(4, Math.floor((intensity + jitter) * 3));
          if (level > targets.current[i]) targets.current[i] = level;
        } else if (waveCenter.current > col + WAVE + 5) {
          targets.current[i] = Math.max(0, targets.current[i] - 0.01);
        }
        const cur = states.current[i];
        const tgt = targets.current[i];
        states.current[i] = cur < tgt ? Math.min(tgt, cur + 0.07) : Math.max(tgt, cur - 0.015);
      }
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      for (let i = 0; i < total; i++) {
        const level = Math.min(4, Math.max(0, Math.floor(states.current[i])));
        ctx.fillStyle = COLORS[level];
        ctx.beginPath();
        ctx.roundRect((i % cols) * (CELL + GAP), Math.floor(i / cols) * (CELL + GAP), CELL, CELL, 2);
        ctx.fill();
      }
      raf.current = requestAnimationFrame(frame);
    }
    raf.current = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf.current);
  }, []);

  return (
    <div ref={containerRef} className={`overflow-hidden ${className ?? ""}`}>
      <canvas ref={canvasRef} className="block" />
    </div>
  );
}
