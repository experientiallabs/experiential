import { mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { type HarnessOptions, StashHarness } from "../testing/stash-harness.js";
import { editorComposer } from "./composer.js";
import { stashedTexts } from "./extension.js";
import { STASH_KEYBINDING_ID } from "./keys.js";

let agentDir: string;
const cwd = "/work/project";
const live = new Set<number>();
const open: StashHarness[] = [];

beforeEach(() => {
  agentDir = mkdtempSync(join(tmpdir(), "exp-code-stash-ext-"));
  live.clear();
});
afterEach(async () => {
  for (const harness of open.splice(0)) await harness.shutdown().catch(() => {});
  rmSync(agentDir, { recursive: true, force: true });
});

/** Start an exp-code "process" with its own pid; it counts as running until `exit`. */
async function start(pid = 1000 + live.size, options: Partial<HarnessOptions> = {}): Promise<StashHarness> {
  live.add(pid);
  const harness = await StashHarness.start({ agentDir, cwd: options.cwd ?? cwd, store: { pid, host: "h", isAlive: (p) => live.has(p) }, ...options });
  open.push(harness);
  return harness;
}

async function exit(harness: StashHarness, pid: number): Promise<void> {
  await harness.shutdown();
  open.splice(open.indexOf(harness), 1);
  live.delete(pid);
}

const stashFiles = () => {
  try {
    return readdirSync(join(agentDir, "prompt-stash"));
  } catch {
    return [];
  }
};

describe("prompt stash in the terminal UI", () => {
  it("stashes on Ctrl+S, shows a hint and an indicator, and brings the draft back after the next send", async () => {
    const h = await start();
    h.type("refactor the parser\nthen add tests  ");
    h.ctrlS();
    expect(h.text()).toBe("");
    expect(h.notes.at(-1)).toMatch(/^Prompt stashed\. It comes back after you send a message, or press Ctrl\+S on an empty prompt\.$/);
    expect(h.indicator()[0]).toContain('Stashed: "refactor the parser ..."');
    h.type("what does main() do?");
    await h.submit();
    expect(h.sent).toEqual(["what does main() do?"]);
    expect(h.text()).toBe("refactor the parser\nthen add tests  ");
    expect(h.notes.at(-1)).toBe("Stashed prompt restored.");
    expect(h.indicator()).toEqual([]);
    // The cursor is back where it was (end of the second line here).
    expect(h.editor.getCursor()).toEqual({ line: 1, col: 16 });
  });

  it("restores the cursor position, not just the text", async () => {
    const h = await start();
    h.type("hello world");
    for (let i = 0; i < 6; i++) h.key("\x1b[D"); // Left x6
    expect(h.editor.getCursor()).toEqual({ line: 0, col: 5 });
    h.ctrlS();
    h.ctrlS();
    expect(h.text()).toBe("hello world");
    expect(h.editor.getCursor()).toEqual({ line: 0, col: 5 });
    h.type(",");
    expect(h.text()).toBe("hello, world");
  });

  it("double save stacks both drafts; Ctrl+S on empty pops the newest; nothing is duplicated", async () => {
    const h = await start();
    h.type("draft A");
    h.ctrlS();
    h.type("draft B");
    h.ctrlS();
    expect(h.text()).toBe("");
    expect(h.notes.at(-1)).toMatch(/^Prompt stashed \(2 stashed\)\./);
    expect(h.indicator()[0]).toContain('2 stashed: "draft B"');
    h.ctrlS();
    expect(h.text()).toBe("draft B");
    expect(h.notes.at(-1)).toBe("Stashed prompt restored. 1 more stashed.");
    await h.submit();
    expect(h.sent).toEqual(["draft B"]);
    expect(h.text()).toBe("draft A");
    expect(h.indicator()).toEqual([]);
  });

  it("two quick presses are a round trip", async () => {
    const h = await start();
    h.type("round trip");
    h.ctrlS();
    h.ctrlS();
    expect(h.text()).toBe("round trip");
    expect(h.indicator()).toEqual([]);
    expect(stashFiles()).toEqual([]);
  });

  it("does not overwrite a prompt typed again before the restore, and says the draft is still stashed", async () => {
    const h = await start();
    h.type("stashed");
    h.ctrlS();
    // The input event arrives after the user has already typed again.
    h.type("new text");
    await h.emit("input", { type: "input", text: "x", source: "interactive" });
    expect(h.text()).toBe("new text");
    expect(h.notes.at(-1)).toMatch(/^Your prompt is still stashed, since the prompt is not empty\./);
    expect(h.indicator()[0]).toContain('"stashed"');
  });

  it("ignores inputs that do not come from the person typing", async () => {
    const h = await start();
    h.type("mine");
    h.ctrlS();
    await h.emit("input", { type: "input", text: "from rpc", source: "rpc" });
    await h.emit("input", { type: "input", text: "from extension", source: "extension" });
    expect(h.text()).toBe("");
    expect(h.indicator()).toHaveLength(1);
  });

  it("restores after a steer while the agent is working", async () => {
    const h = await start();
    h.streaming = true;
    h.type("long draft");
    h.ctrlS();
    h.type("stop and use pnpm");
    await h.submit();
    expect(h.streamingBehaviors).toEqual(["steer"]);
    expect(h.text()).toBe("long draft");
  });

  it("restores after a ! command, which is a submit too", async () => {
    const h = await start();
    h.type("explain the diff");
    h.ctrlS();
    h.type("!git status");
    await h.submit();
    expect(h.bash).toEqual(["git status"]);
    expect(h.text()).toBe("explain the diff");
  });

  it("keeps the stash across built-in commands that open a picker", async () => {
    const h = await start();
    h.type("for later");
    h.ctrlS();
    h.type("/model");
    await h.submit();
    expect(h.builtins).toEqual(["/model"]);
    expect(h.text()).toBe("");
    expect(h.indicator()).toHaveLength(1);
    h.type("hi");
    await h.submit();
    expect(h.text()).toBe("for later");
  });

  it("survives Ctrl+C clear and Escape", async () => {
    const h = await start();
    h.type("precious");
    h.ctrlS();
    h.type("scratch");
    h.key("\x03");
    expect(h.text()).toBe("");
    h.key("\x1b");
    await new Promise((resolve) => setTimeout(resolve, 80)); // lone Escape is dispatched after a timeout
    expect(h.indicator()).toHaveLength(1);
    h.ctrlS();
    expect(h.text()).toBe("precious");
  });

  it("does nothing on an empty prompt with nothing stashed, and treats whitespace as empty", async () => {
    const h = await start();
    h.ctrlS();
    expect(h.notes.at(-1)).toBe("Nothing to stash: type a prompt first, then press Ctrl+S to set it aside.");
    h.type("   ");
    h.ctrlS();
    expect(h.text()).toBe("   ");
    expect(h.indicator()).toEqual([]);
  });

  it("keeps exact text: unicode, emoji, trailing spaces, and very long pastes", async () => {
    const h = await start();
    const text = "naïve café 🧪👩‍👩‍👧 日本語  \n\n  last line with trailing   ";
    h.paste(text);
    h.ctrlS();
    h.ctrlS();
    expect(h.text()).toBe(text);
    h.key("\x03");
    const long = Array.from({ length: 4000 }, (_, i) => `line ${i} ${"z".repeat(30)}`).join("\n");
    h.paste(long);
    expect(h.editor.getText()).toMatch(/^\[paste #\d+ \+4000 lines\]$/);
    h.ctrlS();
    expect(h.text()).toBe("");
    h.ctrlS();
    // It comes back collapsed, as it was shown, and expands to the same content.
    expect(h.editor.getText()).toMatch(/^\[paste #\d+ \+4000 lines\]$/);
    expect(h.text()).toBe(long);
    await h.submit();
    expect(h.sent.at(-1)).toBe(long);
  });

  it("keeps typed text around a collapsed paste, with the cursor between", async () => {
    const h = await start();
    h.type("before ");
    h.paste(Array.from({ length: 12 }, (_, i) => `p${i}`).join("\n"));
    h.type(" after");
    const shown = h.editor.getText();
    const full = h.text();
    h.key("\x1b[D");
    const cursor = h.editor.getCursor();
    h.ctrlS();
    h.ctrlS();
    expect(h.editor.getText()).toBe(shown);
    expect(h.text()).toBe(full);
    expect(h.editor.getCursor()).toEqual(cursor);
  });

  it("does not stash on control characters inside a bracketed paste", async () => {
    const h = await start();
    h.paste("a\x13b\x03c\x1b[Ad");
    expect(h.indicator()).toEqual([]);
    expect(h.text().startsWith("a")).toBe(true);
    const before = h.text();
    h.ctrlS();
    h.ctrlS();
    expect(h.text()).toBe(before);
  });

  it("undo after a stash cannot bring the stashed text back a second time", async () => {
    const h = await start();
    h.type("abc");
    h.ctrlS();
    for (let i = 0; i < 5; i++) h.undo();
    expect(h.text()).toBe("");
    h.type("x");
    h.undo();
    h.undo();
    expect(h.text()).toBe("");
    h.ctrlS();
    expect(h.text()).toBe("abc");
  });

  it("undo after a restore behaves exactly as if the draft had never left", async () => {
    const steps = async (stash: boolean) => {
      const h = await start();
      h.type("abc def");
      h.key("\x17"); // Ctrl+W: delete a word, an undo step of its own
      h.type("xyz");
      if (stash) {
        h.ctrlS();
        h.type("other");
        await h.submit(); // restores the draft
      }
      const seen = [h.text()];
      for (let i = 0; i < 4; i++) {
        h.undo();
        seen.push(h.text());
      }
      return seen;
    };
    const plain = await steps(false);
    expect(plain[0]).toBe("abc xyz");
    expect(await steps(true)).toEqual(plain);
  });

  it("stashing a message recalled with Up gives the hidden draft back", async () => {
    const h = await start();
    h.type("sent before");
    await h.submit();
    h.type("my draft");
    h.key("\x1b[H"); // Home: cursor at the start, so Up recalls history
    h.key("\x1b[A");
    expect(h.text()).toBe("sent before");
    h.ctrlS();
    expect(h.text()).toBe("my draft");
    h.key("\x03");
    h.ctrlS();
    expect(h.text()).toBe("sent before");
  });

  it("Up and Down with a stash present do not touch the stash", async () => {
    const h = await start();
    h.type("one");
    await h.submit();
    h.type("stashed");
    h.ctrlS();
    h.key("\x1b[A");
    expect(h.text()).toBe("one");
    h.key("\x1b[B");
    expect(h.text()).toBe("");
    expect(h.indicator()).toHaveLength(1);
  });

  it("stashes while an autocomplete list is open, closing it", async () => {
    const h = await start();
    h.editor.setAutocompleteProvider({
      async getSuggestions() {
        return { prefix: "/", items: [{ value: "/stash", label: "stash" }] };
      },
      applyCompletion: (lines: string[], line: number, col: number) => ({ lines, cursorLine: line, cursorCol: col })
    } as never);
    h.type("/st");
    await new Promise((resolve) => setTimeout(resolve, 50));
    h.ctrlS();
    expect(h.editor.isShowingAutocomplete()).toBe(false);
    expect(h.text()).toBe("");
    h.ctrlS();
    expect(h.text()).toBe("/st");
  });

  it("leaves Ctrl+S to whatever else has focus (a picker or dialog)", async () => {
    const h = await start();
    h.type("draft");
    const received: string[] = [];
    const picker = { render: () => [], invalidate() {}, handleInput: (data: string) => received.push(data) };
    h.tui.setFocus(picker);
    h.ctrlS();
    expect(received).toEqual(["\x13"]);
    expect(h.text()).toBe("draft");
    h.tui.setFocus(h.editor);
  });

  it("ignores key release and repeat reports", async () => {
    const h = await start();
    h.type("x");
    h.key("\x1b[115;5:3u"); // Kitty: Ctrl+S released
    h.key("\x1b[115;5:2u"); // Kitty: Ctrl+S repeated
    expect(h.indicator()).toEqual([]);
  });

  describe("/stash", () => {
    it("stashes text, pops, lists, drops, and explains itself", async () => {
      const h = await start();
      await h.command("/stash write the changelog");
      expect(h.notes.at(-1)).toMatch(/^Prompt stashed\./);
      expect(h.text()).toBe("");
      await h.command("/stash second   one");
      await h.command("/stash list");
      expect(h.notes.at(-1)).toBe('Stashed prompts, newest first (the first comes back next):\n1. "second   one"\n2. "write the changelog"');
      await h.command("/stash drop");
      expect(h.notes.at(-1)).toBe('Dropped the stashed prompt "second   one". 1 more stashed.');
      await h.command("/stash");
      expect(h.text()).toBe("write the changelog");
      h.key("\x03");
      await h.command("/stash pop");
      expect(h.notes.at(-1)).toMatch(/^Nothing to stash/);
      await h.command("/stash drop");
      expect(h.notes.at(-1)).toBe("Nothing is stashed.");
      await h.command("/stash list");
      expect(h.notes.at(-1)).toBe("Nothing is stashed.");
      await h.command("/stash help");
      expect(h.notes.at(-1)).toContain("exp-code.prompt.stash");
      // /stash is not a send, so it never triggers the restore.
      expect(h.sent).toEqual([]);
    });

    it("keeps multi-line /stash text exactly", async () => {
      const h = await start();
      await h.command("/stash line one\n  line two");
      await h.command("/stash");
      expect(h.text()).toBe("line one\n  line two");
    });

    it("says it only works in the terminal UI elsewhere", async () => {
      const notes: string[] = [];
      const commands = new Map<string, { handler: (args: string, ctx: unknown) => Promise<void> }>();
      const { createPromptStashExtension } = await import("./extension.js");
      await createPromptStashExtension({ agentDir, keys: ["ctrl+s"] })({ registerCommand: (n: string, c: never) => commands.set(n, c), on: () => {} } as never);
      await commands.get("stash")!.handler("x", { mode: "rpc", hasUI: true, ui: { notify: (m: string) => notes.push(m) } });
      expect(notes).toEqual(["The prompt stash works in the interactive terminal UI."]);
    });
  });

  describe("persistence", () => {
    it("survives exit: the next start in the folder puts the draft back and clears the file", async () => {
      const first = await start(1);
      first.type("unfinished thought");
      first.ctrlS();
      first.type("older");
      first.ctrlS();
      expect(stashFiles()).toHaveLength(1);
      await exit(first, 1);
      const second = await start(2);
      expect(second.text()).toBe("older");
      expect(second.notes).toContain("Restored the prompt you stashed in an earlier exp-code session. 1 more stashed.");
      expect(second.indicator()[0]).toContain('"unfinished thought"');
      second.ctrlS(); // stash "older" again: Ctrl+S with text stashes
      second.ctrlS();
      expect(second.text()).toBe("older");
      await second.submit();
      expect(second.text()).toBe("unfinished thought");
      second.key("\x03");
      expect(stashFiles()).toEqual([]);
    });

    it("survives a crash: the stash was already on disk", async () => {
      const first = await start(1);
      first.type("saved before the crash");
      first.ctrlS();
      live.delete(1); // killed, no shutdown
      open.splice(open.indexOf(first), 1);
      first.tui.stop();
      const second = await start(2);
      expect(second.text()).toBe("saved before the crash");
    });

    it("keeps two instances in one folder apart, with no lost writes", async () => {
      const a = await start(1);
      const b = await start(2);
      a.type("from A");
      a.ctrlS();
      b.type("from B");
      b.ctrlS();
      expect(stashFiles()).toHaveLength(2);
      // B does not see A's stash while A runs.
      b.ctrlS();
      expect(b.text()).toBe("from B");
      b.ctrlS();
      await exit(a, 1);
      const c = await start(3);
      expect(c.text()).toBe("from A");
      expect(b.indicator()[0]).toContain('"from B"');
      c.key("\x03");
      await exit(b, 2);
      const d = await start(4);
      expect(d.text()).toBe("from B");
    });

    it("ignores a corrupt stash file safely", async () => {
      const first = await start(1);
      first.type("x");
      first.ctrlS();
      const [name] = stashFiles();
      await exit(first, 1);
      writeFileSync(join(agentDir, "prompt-stash", name!), "\u0000garbage{");
      const second = await start(2);
      expect(second.text()).toBe("");
      expect(second.notes).toEqual([]);
      expect(stashFiles()).toEqual([`${name}.corrupt`]);
    });

    it("keeps the stash in memory and warns when the disk refuses it", async () => {
      writeFileSync(join(agentDir, "prompt-stash"), "a file where the directory should be");
      const h = await start();
      h.type("in memory only");
      h.ctrlS();
      expect(h.notes.at(-1)).toMatch(/^Could not save the stash to disk/);
      h.ctrlS();
      expect(h.text()).toBe("in memory only");
    });

    it("keeps the stash across /new and session switches in the same process", async () => {
      const first = await start(1);
      first.type("across sessions");
      first.ctrlS();
      await first.shutdown();
      open.splice(open.indexOf(first), 1);
      // Same process (pid 1 still running), same extension factory: a new session.
      const next = await StashHarness.start({ agentDir, cwd, factory: first.factory });
      open.push(next);
      expect(next.text()).toBe("");
      expect(next.indicator()[0]).toContain('"across sessions"');
      next.ctrlS();
      expect(next.text()).toBe("across sessions");
    });

    it("does not touch the stash outside the terminal UI (print, JSON, RPC)", async () => {
      const first = await start(1);
      first.type("for the TUI");
      first.ctrlS();
      await exit(first, 1);
      const handlers = new Map<string, (event: unknown, ctx: unknown) => unknown>();
      const { createPromptStashExtension } = await import("./extension.js");
      await createPromptStashExtension({ agentDir, keys: ["ctrl+s"], store: { pid: 9, host: "h", isAlive: () => false } })({
        registerCommand: () => {},
        on: (event: string, handler: (event: unknown, ctx: unknown) => unknown) => handlers.set(event, handler)
      } as never);
      handlers.get("session_start")!({}, { mode: "print", hasUI: false, cwd });
      expect(stashFiles()).toHaveLength(1);
      const later = await start(2);
      expect(later.text()).toBe("for the TUI");
    });
  });

  describe("rebinding", () => {
    it("uses the key from keybindings.json and leaves Ctrl+S alone", async () => {
      writeFileSync(join(agentDir, "keybindings.json"), JSON.stringify({ [STASH_KEYBINDING_ID]: "alt+s" }));
      const h = await start();
      h.type("rebound");
      h.ctrlS();
      expect(h.indicator()).toEqual([]);
      h.key("\x1bs"); // Alt+S
      expect(h.text()).toBe("");
      expect(h.notes.at(-1)).toContain("press Alt+S on an empty prompt");
    });

    it("works through /stash alone when no key is bound", async () => {
      writeFileSync(join(agentDir, "keybindings.json"), JSON.stringify({ [STASH_KEYBINDING_ID]: [] }));
      const h = await start();
      h.type("no key");
      h.ctrlS();
      expect(h.text()).toBe("no key");
      h.key("\x03");
      await h.command("/stash no key");
      expect(h.indicator()[0]).toContain("/stash brings it back");
      h.type("q");
      await h.submit();
      expect(h.text()).toBe("no key");
    });
  });
});

describe("prompt stash invariant", () => {
  // STASH_SEED=<n> runs another random sequence set; STASH_TRACE=<run> prints that run step by step.
  it("never loses or duplicates an unsent draft over random keystroke sequences", async () => {
    let seed = Number(process.env.STASH_SEED ?? 0xc0ffee);
    const random = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };
    const pick = <T>(items: readonly T[]): T => items[Math.floor(random() * items.length)]!;
    const ops = ["type", "type", "newline", "paste", "bigpaste", "ctrlS", "ctrlS", "ctrlS", "submit", "clear", "up", "down", "undo", "stashText", "pop", "drop", "steer", "bang", "model", "restart", "crash"] as const;
    const runs = 200;
    const length = 60;
    const counts = new Map<string, number>();
    for (let run = 0; run < runs; run++) {
      live.clear();
      let pid = 1;
      const folder = `/work/run-${run}`;
      let h = await start(pid, { cwd: folder });
      let token = 0;
      // Every line sent, /stash commands included, goes to history, so Up can show a copy of it.
      const inHistory = new Set<string>();
      const sent = new Set<string>(); // tokens that went out (copies may come back from history)
      let alive = new Set<string>(); // unsent tokens: each must exist exactly once
      // Tokens end in `z`, so a half-typed or half-undone token is never mistaken for another.
      const words = (text: string) => text.match(/[kp]\d+_\d+z/g) ?? [];
      const composerWords = () => {
        const draft = editorComposer(h.editor).read();
        // While a message recalled with Up is shown, the shown text is a copy from history and the
        // draft is the hidden one.
        return draft.hidden === undefined ? words(draft.text) : words(draft.hidden.text);
      };
      const visible = () => [...composerWords(), ...stashedTexts(h.factory, folder).flatMap(words)];
      for (let step = 0; step < length; step++) {
        const op = pick(ops);
        if (process.env.STASH_TRACE === String(run)) console.log(step, op, JSON.stringify(h.editor.getText()), JSON.stringify(stashedTexts(h.factory, folder)));
        const browsing = editorComposer(h.editor).read().hidden !== undefined;
        const created: string[] = [];
        const fresh = (prefix: string) => {
          const t = `${prefix}${run}_${token++}z`;
          created.push(t);
          return t;
        };
        // Ops that may legitimately discard: explicit clears, and (Pi's own behavior, not the
        // stash's) editing or sending while a recalled message hides the draft.
        let mayLose = false;
        let mayResurrect = false;
        // Insert at the end of the cursor's line, so a new token never splits an old one.
        if (op === "type" || op === "newline" || op === "paste" || op === "bigpaste") h.key("\x1b[F");
        switch (op) {
          case "type":
            mayLose = browsing;
            h.type(` ${fresh("k")} `);
            break;
          case "newline":
            mayLose = browsing;
            h.type("\n");
            break;
          case "paste":
            mayLose = browsing;
            h.paste(` ${fresh("k")} `);
            break;
          case "bigpaste": {
            mayLose = browsing;
            const t = fresh("k");
            h.paste(Array.from({ length: 12 }, (_, i) => (i === 5 ? ` ${t} ` : ` filler line ${i} `)).join("\n"));
            break;
          }
          case "ctrlS":
            h.ctrlS();
            break;
          case "submit":
          case "steer":
          case "bang": {
            // A recalled `/stash drop` is an explicit discard too.
            mayLose = browsing || h.text().trim().startsWith("/stash drop");
            h.streaming = op === "steer";
            const text = h.text();
            if (op === "bang" && text.trim() !== "" && !text.trimStart().startsWith("!") && !text.trimStart().startsWith("/")) h.editor.setText(`!${text}`);
            if (h.text().trim() !== "") for (const w of words(h.text())) sent.add(w);
            await h.submit();
            h.streaming = false;
            break;
          }
          case "clear":
            h.key("\x03");
            mayLose = true;
            break;
          case "up":
            h.key("\x1b[A");
            break;
          case "down":
            h.key("\x1b[B");
            break;
          case "undo":
            h.undo();
            mayLose = true;
            mayResurrect = true;
            break;
          case "stashText":
            if (h.text().trim() !== "") continue; // typed over the composer otherwise
            {
              const t = fresh("p");
              inHistory.add(t);
              await h.command(`/stash ${t}`);
            }
            break;
          case "pop":
            if (h.text().trim() !== "") continue;
            await h.command("/stash pop");
            break;
          case "drop":
            if (h.text().trim() !== "") continue;
            await h.command("/stash drop");
            mayLose = true;
            break;
          case "model":
            if (h.text().trim() !== "") continue;
            await h.command("/model");
            break;
          case "restart":
            // Ctrl+D exits only from an empty prompt.
            if (h.text() !== "") continue;
            await exit(h, pid);
            h = await start(++pid, { cwd: folder });
            break;
          case "crash":
            // A crash loses what is in the composer (Pi keeps no draft) but never the stash.
            for (const w of composerWords()) alive.delete(w);
            live.delete(pid);
            open.splice(open.indexOf(h), 1);
            h.tui.stop(); // the process is gone: no session_shutdown
            h = await start(++pid, { cwd: folder });
            break;
        }
        counts.set(op, (counts.get(op) ?? 0) + 1);
        const unsent = visible().filter((w) => !sent.has(w));
        const where = `run ${run} step ${step} ${op}: ${JSON.stringify(visible())}`;
        const counted = unsent.filter((w) => !inHistory.has(w));
        expect(new Set(counted).size, `duplicate after ${where}`).toBe(counted.length);
        for (const w of unsent) if (!alive.has(w) && !created.includes(w) && !inHistory.has(w) && !mayResurrect) throw new Error(`${w} appeared from nowhere after ${where}`);
        if (!mayLose) for (const w of alive) if (!sent.has(w) && !unsent.includes(w)) throw new Error(`${w} was lost after ${where}`);
        // A history copy is not a second draft: conserve one copy of it, not every copy.
        alive = new Set(unsent);
      }
      await exit(h, pid);
    }
    const total = [...counts.values()].reduce((a, b) => a + b, 0);
    expect(total).toBeGreaterThan(runs * length * 0.8);
    for (const op of ops) expect(counts.get(op) ?? 0, op).toBeGreaterThan(0);
    console.log(`prompt stash invariant: ${runs} runs, ${total} operations`, Object.fromEntries(counts));
  }, 180_000);
});
