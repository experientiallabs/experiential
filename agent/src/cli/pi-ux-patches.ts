/**
 * Terminal UI fixes in the pinned Pi packages: the prompt editor's autocomplete, input typed
 * while exp-code starts, `/export` file names, `/hotkeys`, the Ctrl+P model cycle and the context
 * figure after a compaction. pi-patches.ts applies these with its other edits (same exact-match
 * rules, same tests).
 */

import type { Edit, FilePatch } from "./pi-patches.js";

/** The process-wide key exp-code's own key bindings are listed under for `/hotkeys` (hotkeys.ts). */
export const HOTKEYS_KEY = 'Symbol.for("exp-code.hotkeys")';

/** The process-wide key the terminal's mode before exp-code's early input capture is kept under (early-input.ts). */
export const STDIN_WAS_RAW_KEY = 'Symbol.for("exp-code.stdin-was-raw")';

/** The models Ctrl+P cycles through when the gateway lists them, in this order. */
export const CYCLE_MODEL_IDS = ["experiential-small-model", "experiential-medium-model", "experiential-large-model"] as const;

const CYCLE_MODELS = `    /** exp-code: the Experiential models the gateway lists, which Ctrl+P cycles through with any scoped ones. */
    _expCycleModels() {
        const listed = this._modelRuntime.getAvailableSnapshot();
        return ${JSON.stringify(CYCLE_MODEL_IDS)}.map((id) => listed.find((model) => model.provider === "experiential" && model.id === id)).filter((model) => model !== undefined);
    }
    async cycleModel(direction = "forward", options = {}) {
        if (this._scopedModels.length > 0 || this._expCycleModels().length > 1) {
            return this._cycleScopedModel(direction, options);
        }`;

/**
 * The prompt editor (pi-tui components/editor.js). Autocomplete suggestions are computed
 * asynchronously, one request per keystroke, and a request whose text changed before it finished
 * is dropped, so the list on screen can belong to earlier text: keys that arrive together (a fast
 * typist, a paste-like burst, a loaded machine) left the `/` list (`settings` first) showing while
 * `export out1.jsonl` was typed, and Enter then applied `settings` to the end of the line
 * (`out1.jsonsettings`). Enter on a stale list now submits the line as typed, and Tab on one asks
 * again for the current text. A suggestion still completes when it was computed for the text as it
 * is (Tab, or Enter on a highlighted suggestion for the word being typed).
 *
 * An argument typed exactly (`/model experiential-small-model`, whose suggestion's label is that
 * id and whose value is `experiential/experiential-small-model`) is selected as the exact match and
 * runs on the first Enter, where Pi only completed it and waited for a second Enter.
 */
export const EDITOR_PATCH: FilePatch = {
  pkg: "pi-tui",
  file: "components/editor.js",
  edits: [
    {
      find: "            if (kb.matches(data, \"tui.input.tab\")) {\n                const selected = this.autocompleteList.getSelectedItem();",
      replace:
        "            if (kb.matches(data, \"tui.input.tab\") && !this.expAutocompleteCurrent()) {\n                // exp-code: the list on screen was computed for earlier text; ask again for this text.\n                this.cancelAutocomplete();\n                this.handleTabCompletion();\n                return;\n            }\n            if (kb.matches(data, \"tui.input.tab\")) {\n                const selected = this.autocompleteList.getSelectedItem();"
    },
    {
      find: "            if (kb.matches(data, \"tui.select.confirm\")) {\n                const selected = this.autocompleteList.getSelectedItem();\n                if (selected && this.autocompleteProvider) {\n                    this.pushUndoSnapshot();",
      replace:
        "            if (kb.matches(data, \"tui.select.confirm\") && !this.expAutocompleteCurrent()) {\n                // exp-code: the list on screen was computed for earlier text: submit the line as typed.\n                this.cancelAutocomplete();\n            }\n            else if (kb.matches(data, \"tui.select.confirm\")) {\n                const selected = this.autocompleteList.getSelectedItem();\n                const exact = selected !== undefined && !selected.label.endsWith(\"/\") && (selected.value === this.autocompletePrefix || selected.label === this.autocompletePrefix);\n                if (selected && this.autocompleteProvider) {\n                    this.pushUndoSnapshot();"
    },
    {
      find: "                    if (this.autocompletePrefix.startsWith(\"/\")) {\n                        this.cancelAutocomplete();\n                        // Fall through to submit\n                    }\n                    else {",
      replace:
        "                    if (this.autocompletePrefix.startsWith(\"/\") || exact) {\n                        // exp-code: an argument typed exactly also runs on this Enter.\n                        this.cancelAutocomplete();\n                        // Fall through to submit\n                    }\n                    else {"
    },
    { find: "            if (value === prefix) {\n                return i; // Exact match always wins", replace: "            if (value === prefix || items[i].label === prefix) {\n                return i; // Exact match always wins" },
    {
      find: "        this.autocompleteState = state;\n    }\n    cancelAutocompleteRequest() {",
      replace:
        "        this.autocompleteState = state;\n        this.expAutocompleteFor = this.expAutocompleteKey();\n    }\n    /** exp-code: the text and cursor a suggestion list is computed for. */\n    expAutocompleteKey() {\n        return `${this.state.cursorLine}:${this.state.cursorCol}:${this.getText()}`;\n    }\n    /** exp-code: whether the list on screen was computed for the text as it is now. */\n    expAutocompleteCurrent() {\n        return this.expAutocompleteFor === this.expAutocompleteKey();\n    }\n    cancelAutocompleteRequest() {"
    }
  ]
};

/**
 * Pi's terminal (pi-tui terminal.js) restores the mode it found when it stops. exp-code puts the
 * terminal in raw mode before Pi starts, to keep keys typed during startup (early-input.ts), so it
 * hands Pi the mode it found instead.
 */
export const TERMINAL_PATCH: FilePatch = {
  pkg: "pi-tui",
  file: "terminal.js",
  edits: [
    {
      find: "        this.wasRaw = process.stdin.isRaw || false;",
      replace: `        this.wasRaw = process[${STDIN_WAS_RAW_KEY}] ?? (process.stdin.isRaw || false);\n        delete process[${STDIN_WAS_RAW_KEY}];`
    }
  ]
};

/** Edits to pi-coding-agent's modes/interactive/interactive-mode.js. */
export const INTERACTIVE_UX_EDITS: Edit[] = [
  // Keys typed while startup completes go to the prompt as usual, but Pi answered Enter with
  // "Startup is still in progress" and kept the text, so a command typed early lost its Enter and
  // the next one was appended to it (`/compact` and `/usage` went to the model as one message).
  // From the first Enter on, everything typed is held and replayed key for key once the session is
  // ready. Ctrl+C and Ctrl+D act at once (and drop what was held); a dialog an extension opens
  // during startup gets its keys directly.
  {
    find: "        this.ui.start();\n        this.isInitialized = true;\n        this.programStatus.report();",
    replace:
      "        this.ui.start();\n        this.isInitialized = true;\n        this.expHeldInput = undefined;\n        this.expHoldInputRemove = this.ui.addInputListener((data) => {\n            if (this.expHoldInputRemove === undefined || this.ui.focusedComponent !== this.editor)\n                return undefined;\n            if (matchesKey(data, \"ctrl+c\") || matchesKey(data, \"ctrl+d\")) {\n                this.expHeldInput = undefined;\n                return undefined;\n            }\n            if (this.expHeldInput === undefined && !this.keybindings.matches(data, \"tui.input.submit\"))\n                return undefined;\n            (this.expHeldInput ??= []).push(data);\n            return { consume: true };\n        });\n        this.programStatus.report();"
  },
  {
    find: "        await this.rebindCurrentSession();\n        // Render initial messages AFTER showing loaded resources\n        this.renderInitialMessages();",
    replace: "        await this.rebindCurrentSession();\n        // Render initial messages AFTER showing loaded resources\n        this.renderInitialMessages();\n        this.expReplayHeldInput();"
  },
  {
    find: "    handleStartupSubmit(text) {",
    replace:
      "    /** exp-code: stop holding input and replay what was typed while startup completed. */\n    expReplayHeldInput() {\n        const held = this.expHeldInput ?? [];\n        this.expHeldInput = undefined;\n        this.expHoldInputRemove?.();\n        this.expHoldInputRemove = undefined;\n        for (const data of held)\n            this.ui.handleTerminalInput(data);\n    }\n    handleStartupSubmit(text) {"
  },
  // `/export` wrote HTML whatever the name said (`/export notes.txt` was an HTML file). It writes
  // `.html` or `.jsonl` and refuses any other name.
  {
    find: "        const outputPath = this.getPathCommandArgument(text, \"/export\");\n        try {",
    replace:
      "        const outputPath = this.getPathCommandArgument(text, \"/export\");\n        if (outputPath !== undefined && !/\\.(html|jsonl)$/.test(outputPath)) {\n            this.showError(`/export writes .html (a page to read) or .jsonl (the session, for /import) files, so ${outputPath} was not written. Name the file with one of those extensions, or give no name for an HTML file in the session folder.`);\n            return;\n        }\n        try {"
  },
  // `/hotkeys` lists exp-code's own keys (the prompt stash), which are not Pi shortcuts.
  {
    find: "| \\`!!\\` | Run bash command (excluded from context) |\n`;",
    replace: `| \\\`!!\\\` | Run bash command (excluded from context) |\n\${[...(process[${HOTKEYS_KEY}]?.values() ?? [])].map((row) => \`| \\\`\${row.key}\\\` | \${row.action} |\\n\`).join("")}\`;`
  }
];

/** Edits to pi-coding-agent's core/agent-session.js. */
export const AGENT_SESSION_UX_EDITS: Edit[] = [
  // Ctrl+P cycled through every model the gateway lists (about 300). It cycles through the three
  // Experiential models when the gateway lists them, then any models the user scoped (/scoped-models).
  { find: '    async cycleModel(direction = "forward", options = {}) {\n        if (this._scopedModels.length > 0) {\n            return this._cycleScopedModel(direction, options);\n        }', replace: CYCLE_MODELS },
  {
    find: "        const scopedModels = this._scopedModels.filter((scoped) => availableIds.has(`${scoped.model.provider}\\0${scoped.model.id}`));",
    replace:
      "        const cycle = [...this._expCycleModels().filter((model) => !this._scopedModels.some((scoped) => modelsAreEqual(scoped.model, model))).map((model) => ({ model, expCycleOnly: true })), ...this._scopedModels];\n        const scopedModels = cycle.filter((scoped) => availableIds.has(`${scoped.model.provider}\\0${scoped.model.id}`));"
  },
  // An Experiential model the cycle adds is not put into the user's scope when it is chosen.
  {
    find: "            this._addPersistedDefaultToNonEmptyScope(next.model);\n        }\n        // Apply thinking level for the new model.\n        // - Explicit",
    replace: "            if (!next.expCycleOnly)\n                this._addPersistedDefaultToNonEmptyScope(next.model);\n        }\n        // Apply thinking level for the new model.\n        // - Explicit"
  },
  // After a compaction the context figure was unknown ("?") until the next reply. It is the
  // estimate of the compacted context (summary and kept messages) until a reply reports usage.
  {
    from: "        const latestCompaction = getLatestCompactionEntry(branch);\n        if (latestCompaction) {",
    to: "        const estimate = estimateProjectedContextTokens(projection, branch);",
    replace: ""
  }
];
