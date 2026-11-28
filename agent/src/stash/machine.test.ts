import { describe, expect, it } from "vitest";
import { afterSubmit, type Draft, dropStash, type EntryFactory, isBlank, mergeStacks, popStash, pressStash, pushDraft, type StashEntry } from "./machine.js";

function factory(): EntryFactory {
  let n = 0;
  return (draft) => ({ ...draft, id: `e${++n}`, stashedAt: n });
}

const texts = (stack: readonly StashEntry[]) => stack.map((entry) => entry.text);

describe("prompt stash machine", () => {
  it("stashes the exact text and clears the composer", () => {
    const exact = "first line\n  indented\ttab? no: spaces   \n\nemoji 🧪👩‍👩‍👧 and ünïcødé  ";
    const t = pressStash([], { text: exact, cursor: { line: 1, col: 3 } }, factory());
    expect(t.composer).toBeNull();
    expect(t.stack).toHaveLength(1);
    expect(t.stack[0]).toMatchObject({ text: exact, cursor: { line: 1, col: 3 } });
    expect(t.outcome).toMatchObject({ kind: "stashed", depth: 1 });
  });

  it("stashes a very long draft unchanged", () => {
    const long = "x".repeat(300_000) + "\n" + "y ".repeat(50_000);
    const t = pressStash([], { text: long }, factory());
    expect(t.stack[0]!.text).toBe(long);
  });

  it("restores the newest stash into an empty composer, and does nothing with neither", () => {
    const make = factory();
    const one = pressStash([], { text: "a" }, make);
    const back = pressStash(one.stack, { text: "" }, make);
    expect(back.composer).toEqual({ text: "a" });
    expect(back.stack).toEqual([]);
    expect(back.outcome).toMatchObject({ kind: "restored", depth: 0 });
    expect(pressStash([], { text: "" }, make).outcome).toEqual({ kind: "nothing" });
  });

  it("double save keeps both drafts, newest on top, and pops them in reverse", () => {
    const make = factory();
    let t = pressStash([], { text: "A" }, make);
    t = pressStash(t.stack, { text: "B" }, make);
    expect(texts(t.stack)).toEqual(["A", "B"]);
    expect(t.composer).toBeNull();
    expect(t.outcome).toMatchObject({ kind: "stashed", depth: 2 });
    t = pressStash(t.stack, { text: "" }, make);
    expect(t.composer?.text).toBe("B");
    t = afterSubmit(t.stack, { text: "" });
    expect(t.composer?.text).toBe("A");
    expect(t.stack).toEqual([]);
  });

  it("two quick presses are a round trip with no copy left behind", () => {
    const make = factory();
    const draft: Draft = { text: "keep me", cursor: { line: 0, col: 2 } };
    const first = pressStash([], draft, make);
    const second = pressStash(first.stack, { text: "" }, make);
    expect(second.composer).toEqual(draft);
    expect(second.stack).toEqual([]);
  });

  it("treats whitespace-only input as empty: nothing to stash, and a restore replaces it", () => {
    expect(isBlank(" \n\t ")).toBe(true);
    const make = factory();
    expect(pressStash([], { text: "   \n " }, make).outcome).toEqual({ kind: "nothing" });
    const one = pressStash([], { text: "real" }, make);
    const back = pressStash(one.stack, { text: "  \n" }, make);
    expect(back.composer?.text).toBe("real");
  });

  it("after a submit, restores only into an empty composer and otherwise keeps the stash", () => {
    const one = pressStash([], { text: "later" }, factory());
    expect(afterSubmit(one.stack, { text: "" }).composer?.text).toBe("later");
    const busy = afterSubmit(one.stack, { text: "typing again" });
    expect(busy.composer).toBeUndefined();
    expect(texts(busy.stack)).toEqual(["later"]);
    expect(busy.outcome).toEqual({ kind: "kept", depth: 1 });
    expect(afterSubmit([], { text: "" })).toMatchObject({ composer: undefined, outcome: { kind: "nothing" } });
  });

  it("stashing a message recalled from history gives the hidden draft back", () => {
    const make = factory();
    const t = pressStash([], { text: "old message", undo: ["u"], hidden: { text: "my draft", cursor: { line: 0, col: 2 } } }, make);
    expect(texts(t.stack)).toEqual(["old message"]);
    expect(t.stack[0]!.undo).toBeUndefined();
    expect(t.stack[0]).not.toHaveProperty("hidden");
    expect(t.composer).toEqual({ text: "my draft", cursor: { line: 0, col: 2 } });
    expect(pressStash([], { text: "old", hidden: { text: "  " } }, make).composer).toBeNull();
  });

  it("carries the draft's undo history through stash and restore", () => {
    const make = factory();
    const t = pressStash([], { text: "abc", undo: ["s1", "s2"] }, make);
    expect(popStash(t.stack, { text: "" }).composer).toEqual({ text: "abc", undo: ["s1", "s2"] });
  });

  it("pushes /stash text without touching the composer, ignoring blank text", () => {
    const make = factory();
    const t = pushDraft([], { text: "  spaced  " }, make);
    expect(t.composer).toBeUndefined();
    expect(texts(t.stack)).toEqual(["  spaced  "]);
    expect(pushDraft(t.stack, { text: " " }, make)).toMatchObject({ outcome: { kind: "nothing" } });
  });

  it("drops only the newest draft", () => {
    const make = factory();
    const t = pushDraft(pushDraft([], { text: "a" }, make).stack, { text: "b" }, make);
    const { stack, dropped } = dropStash(t.stack);
    expect(dropped?.text).toBe("b");
    expect(texts(stack)).toEqual(["a"]);
    expect(dropStash([]).dropped).toBeUndefined();
  });

  it("merges adopted drafts once per id, oldest first", () => {
    const e = (id: string, at: number): StashEntry => ({ id, text: id, stashedAt: at });
    expect(texts(mergeStacks([e("c", 3)], [e("a", 1), e("c", 3), e("b", 2), e("a", 1)]))).toEqual(["a", "b", "c"]);
  });

  it("never loses or duplicates a draft over random operation sequences", () => {
    // A seeded generator keeps failures reproducible.
    let seed = 0x5eed;
    const random = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };
    const runs = 2000;
    let steps = 0;
    for (let run = 0; run < runs; run++) {
      const make = factory();
      let stack: StashEntry[] = [];
      let composer = "";
      let token = 0;
      const alive: string[] = []; // every token not yet sent or discarded
      for (let step = 0; step < 60; step++, steps++) {
        const op = Math.floor(random() * 6);
        if (op === 0) {
          const t = `t${run}_${token++}`;
          composer = composer === "" ? t : `${composer} ${t}`;
          alive.push(t);
        } else if (op === 1 || op === 5) {
          const t = pressStash(stack, { text: composer }, make);
          stack = t.stack;
          if (t.composer !== undefined) composer = t.composer?.text ?? "";
        } else if (op === 2) {
          if (isBlank(composer)) continue;
          for (const t of composer.split(" ")) alive.splice(alive.indexOf(t), 1);
          composer = "";
          const t = afterSubmit(stack, { text: composer });
          stack = t.stack;
          if (t.composer !== undefined) composer = t.composer?.text ?? "";
        } else if (op === 3) {
          const t = pushDraft(stack, { text: `p${run}_${token}` }, make);
          alive.push(`p${run}_${token++}`);
          stack = t.stack;
        } else {
          // Ctrl+C clear: an explicit discard of the composer only.
          if (!isBlank(composer)) for (const t of composer.split(" ")) alive.splice(alive.indexOf(t), 1);
          composer = "";
        }
        const everywhere = [...composer.split(" "), ...stack.flatMap((entry) => entry.text.split(" "))].filter((t) => t !== "");
        expect(everywhere.sort()).toEqual([...alive].sort());
      }
    }
    expect(steps).toBeGreaterThan(100_000);
  });
});
