import { CustomEditor } from "@earendil-works/pi-coding-agent";
import { Editor, KeybindingsManager, TUI_KEYBINDINGS, TuiMainScreen } from "@earendil-works/pi-tui";
import { describe, expect, it } from "vitest";
import { editorComposer, type EditorLike, expandMarkers, isMainEditor, textComposer } from "./composer.js";

const plain = (text: string) => text;
const theme = { borderColor: plain, selectList: { selectedPrefix: plain, selectedText: plain, description: plain, scrollInfo: plain, noMatch: plain } };
const tui = () => new TuiMainScreen({ columns: 80, rows: 24 } as never);

describe("composer access", () => {
  it("recognizes Pi's prompt editor and nothing else", () => {
    const prompt = new CustomEditor(tui(), theme as never, new KeybindingsManager(TUI_KEYBINDINGS) as never);
    expect(isMainEditor(prompt)).toBe(true);
    // A bare editor (as inside a dialog) has no app actions, so it is not the prompt.
    expect(isMainEditor(new Editor(tui(), theme as never))).toBe(false);
    expect(isMainEditor({ getText: () => "", setText: () => {} })).toBe(false);
    expect(isMainEditor(null)).toBe(false);
    expect(isMainEditor(undefined)).toBe(false);
  });

  it("round-trips text, cursor and collapsed pastes through the real editor", () => {
    const editor = new CustomEditor(tui(), theme as never, new KeybindingsManager(TUI_KEYBINDINGS) as never);
    const big = Array.from({ length: 15 }, (_, i) => `row ${i}`).join("\n");
    editor.handleInput("ab ");
    editor.handleInput(`\x1b[200~${big}\x1b[201~`);
    editor.handleInput(" cd");
    const composer = editorComposer(editor);
    const draft = composer.read();
    expect(draft.text).toBe(`ab ${big} cd`);
    expect(draft.collapsed?.text).toBe("ab [paste #1 +15 lines] cd");
    composer.write(null);
    expect(editor.getText()).toBe("");
    composer.write(draft);
    expect(editor.getText()).toBe("ab [paste #1 +15 lines] cd");
    expect(editor.getExpandedText()).toBe(`ab ${big} cd`);
    expect(editor.getCursor()).toEqual(draft.cursor);
  });

  it("falls back to the expanded text when the collapsed form no longer matches", () => {
    const editor = new CustomEditor(tui(), theme as never, new KeybindingsManager(TUI_KEYBINDINGS) as never);
    editorComposer(editor).write({ text: "the truth", collapsed: { text: "[paste #1 +11 lines]", pastes: [[1, "something else"]] }, cursor: { line: 0, col: 3 } });
    expect(editor.getText()).toBe("the truth");
    // The recorded cursor belonged to the collapsed form, so it is not applied.
    expect(editor.getCursor()).toEqual({ line: 0, col: 9 });
  });

  it("ignores a cursor outside the text", () => {
    const editor = new CustomEditor(tui(), theme as never, new KeybindingsManager(TUI_KEYBINDINGS) as never);
    editorComposer(editor).write({ text: "short", cursor: { line: 4, col: 99 } });
    expect(editor.getCursor()).toEqual({ line: 0, col: 5 });
  });

  it("works on an editor without Pi's internals, with plain text", () => {
    let text = "";
    const bare: EditorLike = { getText: () => text, setText: (next) => void (text = next) };
    const composer = editorComposer(bare);
    composer.write({ text: "x\ny", cursor: { line: 1, col: 0 }, undo: ["ignored"] });
    expect(composer.read()).toEqual({ text: "x\ny" });
    composer.write(null);
    expect(text).toBe("");
  });

  it("uses the extension UI's text methods when the editor is out of reach", () => {
    let text = "typed";
    const composer = textComposer({ getEditorText: () => text, setEditorText: (next) => void (text = next) });
    expect(composer.read()).toEqual({ text: "typed" });
    composer.write({ text: "back" });
    expect(text).toBe("back");
    composer.write(null);
    expect(text).toBe("");
  });

  it("expands paste markers like the editor does", () => {
    expect(expandMarkers("a [paste #2 +12 lines] b [paste #3 1500 chars] [paste #9]", [[2, "TWO"], [3, "THREE"]])).toBe("a TWO b THREE [paste #9]");
    expect(expandMarkers("$& and $1", [[1, "x"]])).toBe("$& and $1");
    expect(expandMarkers("[paste #1]", [[1, "$&$1"]])).toBe("$&$1");
  });
});
