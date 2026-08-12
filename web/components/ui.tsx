import { useId, type ButtonHTMLAttributes, type ReactNode } from "react";
import { clsx } from "clsx";

type ButtonVariant = "default" | "primary" | "danger" | "quiet";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  children: ReactNode;
  variant?: ButtonVariant;
};

export function buttonClassName(variant: ButtonVariant = "default", className?: string): string {
  return clsx(
    "inline-flex min-h-[38px] items-center justify-center gap-2 rounded-[var(--radius-md)] px-4 text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-55",
    variant === "primary" && "border border-ink bg-ink text-white hover:bg-[#252525]",
    variant === "danger" && "border border-danger bg-danger text-white hover:bg-[#bd2d2d]",
    variant === "quiet" && "border border-transparent bg-transparent text-muted hover:bg-hover hover:text-ink",
    variant === "default" && "border border-line-strong bg-surface text-ink hover:bg-hover",
    className
  );
}

export function Button({ children, className, variant = "default", ...props }: ButtonProps) {
  return (
    <button className={buttonClassName(variant, className)} {...props}>
      {children}
    </button>
  );
}

export function Card({
  children,
  className
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={clsx("rounded-[var(--radius-lg)] border border-line bg-surface p-[18px]", className)}>
      {children}
    </section>
  );
}

type ChipTone = "neutral" | "success" | "warning" | "danger" | "purple" | "blue";

const chipTone: Record<ChipTone, string> = {
  neutral: "bg-surface-subtle text-muted",
  success: "bg-success-soft text-success",
  warning: "bg-warning-soft text-warning",
  danger: "bg-danger-soft text-danger",
  purple: "bg-purple-soft text-purple",
  blue: "bg-blue-50 text-blue-700"
};

export function Chip({ label, tone = "neutral" }: { label: string; tone?: ChipTone }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-full px-[9px] py-[5px] font-mono text-[11px] font-semibold uppercase",
        chipTone[tone]
      )}
    >
      {label}
    </span>
  );
}

export function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="grid min-h-[220px] place-items-center rounded-[var(--radius-lg)] border border-dashed border-line-strong px-5 text-center">
      <div>
        <h2 className="m-0 text-sm font-medium text-[#474747]">{title}</h2>
        <p className="mt-2 max-w-xl text-[13px] leading-relaxed text-muted">{body}</p>
      </div>
    </div>
  );
}

export function Dialog({
  title,
  children,
  onClose
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
}) {
  const titleId = useId();
  return (
    <div
      aria-labelledby={titleId}
      aria-modal="true"
      className="fixed inset-0 z-50 grid place-items-center bg-ink/20 p-4"
      onMouseDown={onClose}
      role="dialog"
    >
      <section
        className="w-full max-w-xl rounded-[var(--radius-lg)] border border-line bg-surface p-5 shadow-xl"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between gap-4">
          <h2 className="m-0 text-base font-semibold text-ink" id={titleId}>
            {title}
          </h2>
          <button
            aria-label="Close dialog"
            className="rounded-[var(--radius-sm)] border-0 bg-transparent px-2 py-1 text-lg leading-none text-muted hover:bg-hover hover:text-ink"
            onClick={onClose}
            type="button"
          >
            ×
          </button>
        </div>
        {children}
      </section>
    </div>
  );
}
