import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useState } from "react";

import { Button, Dialog } from "@/components/ui";

describe("Dialog", () => {
  afterEach(() => cleanup());

  it("enters and traps focus in both directions, then Escape restores prior focus", async () => {
    render(<DialogHarness />);
    const opener = screen.getByRole("button", { name: "Open dialog" });
    opener.focus();
    fireEvent.click(opener);

    const dialog = screen.getByRole("dialog", { name: "Confirm local action" });
    const close = screen.getByRole("button", { name: "Close dialog" });
    const confirm = screen.getByRole("button", { name: "Confirm action" });
    await waitFor(() => expect(close).toHaveFocus());
    expect(document.getElementById(dialog.getAttribute("aria-labelledby") ?? "")).toHaveTextContent(
      "Confirm local action"
    );

    confirm.focus();
    fireEvent.keyDown(confirm, { key: "Tab" });
    expect(close).toHaveFocus();
    fireEvent.keyDown(close, { key: "Tab", shiftKey: true });
    expect(confirm).toHaveFocus();
    fireEvent.keyDown(confirm, { key: "Escape" });

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
  });

  it("makes the background inert and blocks background pointer and keyboard actions", async () => {
    const backgroundAction = vi.fn();
    render(<DialogHarness onBackgroundAction={backgroundAction} />);
    const opener = screen.getByRole("button", { name: "Open dialog" });
    const background = screen.getByRole("button", { name: "Background action" });
    fireEvent.click(opener);
    await screen.findByRole("dialog");
    const applicationRoot = opener.closest("[aria-hidden='true']");

    expect(applicationRoot).toHaveAttribute("aria-hidden", "true");
    expect(applicationRoot).toHaveProperty("inert", true);
    fireEvent.click(background);
    fireEvent.keyDown(background, { key: "a" });
    expect(backgroundAction).not.toHaveBeenCalled();

    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
    expect(applicationRoot).not.toHaveAttribute("aria-hidden");
    expect((applicationRoot as HTMLElement | null)?.inert).toBeFalsy();
  });
});

function DialogHarness({ onBackgroundAction = () => undefined }: { onBackgroundAction?: () => void }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <Button onClick={() => setOpen(true)} type="button">
        Open dialog
      </Button>
      <Button onClick={onBackgroundAction} onKeyDown={onBackgroundAction} type="button">
        Background action
      </Button>
      {open ? (
        <Dialog onClose={() => setOpen(false)} title="Confirm local action">
          <Button type="button">Confirm action</Button>
        </Dialog>
      ) : null}
    </div>
  );
}
