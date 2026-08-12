import { clsx } from "clsx";
import { useEffect, useId, useRef, type ButtonHTMLAttributes, type ReactNode } from "react";
import { createPortal } from "react-dom";

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
  onClose,
  dismissible = true
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  dismissible?: boolean;
}) {
  const titleId = useId();
  const dialogId = useId();
  const overlayRef = useRef<HTMLDivElement>(null);
  const dialogRef = useRef<HTMLElement>(null);
  const onCloseRef = useRef(onClose);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    const overlay = overlayRef.current;
    const dialog = dialogRef.current;
    if (!overlay || !dialog) {
      return;
    }
    const priorFocus =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const background = Array.from(document.body.children).filter((item) => item !== overlay);
    const priorBackground = background.map((item) => ({
      element: item as HTMLElement,
      ariaHidden: item.getAttribute("aria-hidden"),
      inert: (item as HTMLElement).inert
    }));
    for (const item of priorBackground) {
      item.element.inert = true;
      item.element.setAttribute("aria-hidden", "true");
    }

    const controls = focusableControls(dialog);
    (controls[0] ?? dialog).focus();

    const protectModal = (event: Event) => {
      if (event.target instanceof Node && !overlay.contains(event.target)) {
        event.preventDefault();
        event.stopPropagation();
      }
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && dismissible) {
        event.preventDefault();
        event.stopPropagation();
        onCloseRef.current();
        return;
      }
      const currentControls = focusableControls(dialog);
      if (!(event.target instanceof Node) || !overlay.contains(event.target)) {
        protectModal(event);
        if (event.key === "Tab") {
          const boundary = event.shiftKey ? currentControls.at(-1) : currentControls[0];
          (boundary ?? dialog).focus();
        }
        return;
      }
      if (event.key !== "Tab") {
        return;
      }
      if (currentControls.length === 0) {
        event.preventDefault();
        dialog.focus();
        return;
      }
      const first = currentControls[0];
      const last = currentControls.at(-1) ?? first;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      } else if (!dialog.contains(document.activeElement)) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown, true);
    document.addEventListener("pointerdown", protectModal, true);
    document.addEventListener("click", protectModal, true);
    return () => {
      document.removeEventListener("keydown", onKeyDown, true);
      document.removeEventListener("pointerdown", protectModal, true);
      document.removeEventListener("click", protectModal, true);
      for (const item of priorBackground) {
        item.element.inert = item.inert;
        if (item.ariaHidden === null) {
          item.element.removeAttribute("aria-hidden");
        } else {
          item.element.setAttribute("aria-hidden", item.ariaHidden);
        }
      }
      if (priorFocus?.isConnected) {
        priorFocus.focus();
      }
    };
  }, [dismissible]);

  return createPortal(
    <div
      className="fixed inset-0 z-50 grid place-items-center bg-ink/20 p-4"
      onMouseDown={(event) => {
        if (dismissible && event.target === event.currentTarget) {
          onClose();
        }
      }}
      ref={overlayRef}
    >
      <section
        aria-labelledby={titleId}
        aria-modal="true"
        className="w-full max-w-xl rounded-[var(--radius-lg)] border border-line bg-surface p-5 shadow-xl"
        id={dialogId}
        ref={dialogRef}
        role="dialog"
        tabIndex={-1}
      >
        <div className="mb-4 flex items-start justify-between gap-4">
          <h2 className="m-0 text-base font-semibold text-ink" id={titleId}>
            {title}
          </h2>
          {dismissible ? (
            <button
              aria-label="Close dialog"
              className="rounded-[var(--radius-sm)] border-0 bg-transparent px-2 py-1 text-lg leading-none text-muted hover:bg-hover hover:text-ink"
              onClick={onClose}
              type="button"
            >
              ×
            </button>
          ) : null}
        </div>
        {children}
      </section>
    </div>,
    document.body
  );
}

function focusableControls(container: HTMLElement): HTMLElement[] {
  return Array.from(
    container.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
    )
  ).filter((item) => item.getAttribute("aria-hidden") !== "true");
}
