/**
 * Source patches for the pinned Pi packages, applied as their modules load.
 *
 * Pi has no option for most of what exp-code needs to change: its name in user-facing text, the
 * project folder (`.pi/`), the direct vendor providers it enables from environment keys, and a
 * few commands that talk to Pi's own services. exp-code therefore rewrites a short list of exact
 * strings in Pi's published JavaScript. The CLI applies them with a Node module load hook
 * (`installPiPatches`, before Pi is imported) and the runner bundle applies the same table at
 * build time (scripts/bundle-runner.mjs), so every entry point runs the same patched code.
 *
 * Every edit names the exact text it replaces and how many times it occurs. An edit that does not
 * match exactly throws, so a Pi upgrade that moves the text fails the tests (pi-patches.test.ts
 * patches the installed files) instead of silently shipping an unpatched string.
 *
 * Internal names a user never sets stay Pi's (`PI_CODING_AGENT_DIR`, `PI_OFFLINE`, ...); the CLI
 * translates exp-code's variables to them before Pi loads (environment.ts).
 */

import { registerHooks } from "node:module";

/** One exact replacement: `find` must occur exactly `count` times (default 1). */
export interface ExactEdit {
  find: string;
  replace: string;
  count?: number;
}

/** Replace everything from `from` up to (not including) `to`; each must occur exactly once. */
export interface RangeEdit {
  from: string;
  to: string;
  replace: string;
}

export type Edit = ExactEdit | RangeEdit;

export interface FilePatch {
  /** Package directory under `@earendil-works/`. */
  pkg: string;
  /** File under the package's `dist/` directory. */
  file: string;
  edits: Edit[];
}

/** Thrown when an edit does not match the file it patches. */
export class PiPatchError extends Error {
  override name = "PiPatchError";
}

/**
 * Pi's `completeSummarization`, guarded (core/compaction/compaction.js). Reasoning models spend
 * output tokens thinking before they write, so a summary request that inherits the session's
 * thinking level and the whole output limit can end at the cap with only a character of summary
 * (a length stop Pi rejects), or stop with nothing usable (which Pi committed). The first attempt
 * asks for at most 32K tokens at no more than low reasoning; a truncated, empty or implausibly
 * short summary (under 400 characters, or under 1/50 of the summarized text when that is smaller)
 * is retried once with the model's whole output limit and its lowest reasoning level. A second
 * failure is returned as an error naming both, so every caller keeps what it would have replaced.
 */
const SUMMARY_GUARD = `const EXP_SUMMARY_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
/** exp-code: the first summary attempt's output limit, when the model allows that much. */
export const EXP_SUMMARY_OUTPUT_TOKENS = 32768;
function expSummaryRank(level) {
    return EXP_SUMMARY_LEVELS.indexOf(level);
}
/** exp-code: the reasoning level of the first attempt (at most low) and of the retry (the lowest). */
export function summaryThinkingLevels(model, requested) {
    const supported = model.reasoning ? getSupportedThinkingLevels(model) : ["off"];
    const lowest = supported.reduce((a, b) => (expSummaryRank(b) < expSummaryRank(a) ? b : a), supported[0] ?? "off");
    const cap = Math.min(expSummaryRank(requested ?? "off"), expSummaryRank("low"));
    const first = supported.filter((level) => expSummaryRank(level) <= cap).reduce((a, b) => (expSummaryRank(b) > expSummaryRank(a) ? b : a), lowest);
    return { first, retry: lowest };
}
function expSummaryInputChars(context) {
    let chars = 0;
    for (const message of context.messages) {
        if (typeof message.content === "string")
            chars += message.content.length;
        else if (Array.isArray(message.content))
            for (const block of message.content)
                if (block.type === "text")
                    chars += block.text.length;
    }
    return chars;
}
/** exp-code: why a finished summary must not be committed, or undefined when it may. */
export function getSummaryProblem(response, inputChars) {
    if (response.stopReason === "error" || response.stopReason === "aborted")
        return undefined;
    if (response.content.some((block) => block.type === "toolCall"))
        return undefined;
    if (response.stopReason === "length")
        return "the summary was cut off at the output limit";
    const text = contentText(response.content).trim();
    if (text === "")
        return "the summary came back empty";
    const minimum = Math.min(400, Math.floor(inputChars / 50));
    if (text.length < minimum)
        return \`the summary came back implausibly short (\${text.length} characters for \${inputChars} characters of conversation)\`;
    return undefined;
}
function expSummaryOptions(options, level, maxTokens) {
    const next = { ...options, maxTokens };
    if (level === "off")
        delete next.reasoning;
    else
        next.reasoning = level;
    return next;
}
/**
 * Shared choke point for every compaction/branch-summary summarization call. Each attempt is
 * wrapped in {@link retryAssistantCall} so transient stream drops honor the configured retry
 * policy. exp-code guards the answer: see getSummaryProblem and summaryThinkingLevels.
 */
export async function completeSummarization(model, context, options, streamFn, retry, callbacks) {
    // Avoid cache writes for one-off summaries. Reuse caller-supplied routing when available;
    // callers without a session ID, including branch summaries, receive a fresh routing ID.
    const requestOptions = {
        ...options,
        cacheRetention: "none",
        sessionId: options.sessionId ?? uuidv7(),
    };
    const produce = (attemptOptions) => async () => streamFn
        ? (await streamFn(model, context, attemptOptions)).result()
        : completeSimple(model, context, attemptOptions);
    const levels = summaryThinkingLevels(model, requestOptions.reasoning);
    const modelMax = model.maxTokens > 0 ? model.maxTokens : EXP_SUMMARY_OUTPUT_TOKENS;
    const firstMax = Math.min(modelMax, EXP_SUMMARY_OUTPUT_TOKENS);
    const retryMax = Math.max(firstMax, modelMax);
    const inputChars = expSummaryInputChars(context);
    const first = await retryAssistantCall(produce(expSummaryOptions(requestOptions, levels.first, firstMax)), retry, requestOptions.signal, callbacks);
    const firstProblem = getSummaryProblem(first, inputChars);
    if (firstProblem === undefined || requestOptions.signal?.aborted)
        return first;
    const second = await retryAssistantCall(produce(expSummaryOptions(requestOptions, levels.retry, retryMax)), retry, requestOptions.signal, callbacks);
    const usage = first.usage && second.usage ? combineUsage(first.usage, second.usage) : second.usage;
    const secondProblem = getSummaryProblem(second, inputChars);
    if (secondProblem === undefined)
        return { ...second, usage };
    const reasoning = levels.retry === "off" ? "reasoning off" : \`\${levels.retry} reasoning\`;
    return {
        ...second,
        usage,
        stopReason: "error",
        errorMessage: \`\${firstProblem}, and a retry with up to \${retryMax} output tokens and \${reasoning} failed too (\${secondProblem})\`,
    };
}
`;

/** The automatic-compaction pause (core/agent-session.js): keyed by session and model. */
const AUTO_COMPACTION_PAUSE_STATE = `    _overflowRecoveryAttempted = false;
    /** exp-code: the session and model whose automatic compaction failed; it is not retried for them. */
    _autoCompactionPausedFor = undefined;
    _autoCompactionKey(model) {
        return \`\${this.sessionManager.getSessionId()}\\0\${model.provider}/\${model.id}\`;
    }
    _autoCompactionPaused() {
        return this.model !== undefined && this._autoCompactionPausedFor === this._autoCompactionKey(this.model);
    }
    // Branch summarization state`;

/** What a compaction failure tells the user (core/agent-session.js). `model` is set for automatic compaction. */
const COMPACTION_FAILURE_ADVICE = `/** exp-code: what a failed compaction kept, and what to do next. */
function compactionFailureAdvice(message, model) {
    const stop = /[.!?]$/.test(message.trim()) ? "" : ".";
    // Failures before any summary was asked for (nothing to compact, no key) change nothing either.
    if (!/summarization failed/i.test(message))
        return model ? \`\${stop} Automatic compaction is paused for \${model.id} in this session; run /compact to try again.\` : "";
    const kept = \`\${stop} The conversation was kept unchanged.\`;
    const paused = model ? \` Automatic compaction is paused for \${model.id} in this session.\` : "";
    return \`\${kept}\${paused} Run /compact to try again, /model to switch to another model, or /new to start a fresh session.\`;
}`;

const HELP_ENVIRONMENT =`\${chalk.bold("Environment Variables:")}
  EXP_API_KEY                      - Hosted Experiential gateway key (or run exp-code login)
  EXP_GATEWAY_KEY                  - Key of a local exp gateway
  EXP_CODE_MODEL                   - Default model alias
  EXP_CODE_HOME                    - Configuration directory (default: ~/.exp-code)
  EXP_CODE_SESSION_DIR             - Session storage directory (overridden by --session-dir)
  EXP_CODE_OFFLINE                 - Disable startup network operations when set to 1/true/yes
  EXP_CODE_NO_UPDATE_CHECK         - Skip the daily check for a newer exp-code when set to 1
  The README lists every variable.

`;

/** Credential variables removed from the environment of every command the agent runs. */
export const AGENT_COMMAND_UNSET = ["EXP_API_KEY", "EXP_GATEWAY_KEY", "EXPLABS_API_KEY"] as const;

/** Pi's shell-tool guideline names its own variables; exp-code renames them (core/tools/bash.js). */
const SESSION_ENV_GUIDELINE = '"You can inspect PI_* environment variables for current model and session details."';
const EXP_SESSION_ENV_GUIDELINE = '"You can inspect the EXP_CODE_SESSION_* environment variables for the current model and session details."';
/**
 * Pi's `readPipedStdin` (main.js), bounded. Pi reads stdin to its end whenever stdin is not a
 * terminal, so a run whose stdin is a pipe that never closes (a CI step, an editor or a script
 * that spawns exp-code with a pipe and never writes to it) waited forever before its first model
 * call, printing nothing, even when the prompt was already on the command line. When the command
 * line carries a prompt or @file and no byte arrives within EXP_STDIN_WAIT_MS, exp-code stops
 * reading, says so on stderr, and runs the prompt alone. Input that starts arriving is still read
 * to its end, and a run with no prompt on the command line keeps waiting, as stdin is all it has.
 */
const READ_PIPED_STDIN = `/** exp-code: how long a run with a prompt waits for the first byte of piped stdin. */
const EXP_STDIN_WAIT_MS = 5000;
async function readPipedStdin(hasPrompt = false) {
    // If stdin is a TTY, we're running interactively - don't read stdin
    if (process.stdin.isTTY) {
        return undefined;
    }
    return new Promise((resolve) => {
        let data = "";
        let idle;
        const finish = (value) => {
            clearTimeout(idle);
            process.stdin.off("data", onData);
            process.stdin.off("end", onEnd);
            process.stdin.off("error", onEnd);
            resolve(value);
        };
        const onData = (chunk) => {
            data += chunk;
            clearTimeout(idle);
        };
        const onEnd = () => finish(data.trim() || undefined);
        process.stdin.setEncoding("utf8");
        process.stdin.on("data", onData);
        process.stdin.on("end", onEnd);
        process.stdin.on("error", onEnd);
        if (hasPrompt) {
            idle = setTimeout(() => {
                console.error(chalk.yellow(\`Warning: nothing arrived on stdin within \${EXP_STDIN_WAIT_MS / 1000}s, so exp-code runs the prompt without it. Redirect stdin from /dev/null (< /dev/null) when there is nothing to pipe.\`));
                process.stdin.destroy();
                finish(undefined);
            }, EXP_STDIN_WAIT_MS);
        }
        process.stdin.resume();
    });
}
`;

/** The model examples of `exp-code --help`, with gateway aliases. */
const HELP_MODEL_EXAMPLES = `  # Use another model (exp-code --list-models lists them)
  \${APP_NAME} --model experiential-small-model "Help me refactor this code"

  # Use a model with a thinking level
  \${APP_NAME} --model experiential-medium-model:high "Solve this complex problem"

  # Limit Ctrl+P model cycling to some models
  \${APP_NAME} --models experiential-small-model,experiential-medium-model,experiential-large-model

  # Cycle models matching a pattern
  \${APP_NAME} --models "experiential-*"

  # Cycle models with fixed thinking levels
  \${APP_NAME} --models experiential-medium-model:high,experiential-small-model:low

`;

/** The patch table for the pinned Pi release (1.5.1). */
export const PI_PATCHES: FilePatch[] = [
  {
    // The app name, terminal title and project folder. The agent and session directory variables
    // stay Pi's internal names: exp-code sets them from EXP_CODE_HOME and EXP_CODE_SESSION_DIR.
    pkg: "pi-coding-agent",
    file: "config.js",
    edits: [
      { find: 'export const APP_NAME = piConfigName || "pi";', replace: 'export const APP_NAME = "exp-code";' },
      { find: 'export const APP_TITLE = piConfigName ? APP_NAME : "π";', replace: "export const APP_TITLE = APP_NAME;" },
      { find: 'export const CONFIG_DIR_NAME = pkg.piConfig?.configDir || ".pi";', replace: 'export const CONFIG_DIR_NAME = ".exp-code";' },
      { find: "export const ENV_AGENT_DIR = `${APP_NAME.toUpperCase()}_CODING_AGENT_DIR`;", replace: 'export const ENV_AGENT_DIR = "PI_CODING_AGENT_DIR";' },
      {
        find: "export const ENV_SESSION_DIR = `${APP_NAME.toUpperCase()}_CODING_AGENT_SESSION_DIR`;",
        replace: 'export const ENV_SESSION_DIR = "PI_CODING_AGENT_SESSION_DIR";'
      }
    ]
  },
  {
    pkg: "pi-coding-agent",
    file: "cli/args.js",
    edits: [
      { find: "${chalk.bold(APP_NAME)} - AI coding assistant with read, bash, edit, write tools", replace: "${chalk.bold(APP_NAME)} - Experiential's coding and cowork agent" },
      { find: "update [source|self|pi]   Update pi, extensions, or model catalogs", replace: "update [source|self]      Update exp-code, extensions, or model catalogs" },
      // Pi's `auth` manages vendor provider credentials, which exp-code has none of (run.ts answers
      // it); the Experiential sign-in commands take its place.
      {
        find: "  ${APP_NAME} auth <command>            Log in or out, print credentials, or check provider readiness\n",
        replace:
          "  ${APP_NAME} login [--no-browser]      Sign in to Experiential and save a key\n  ${APP_NAME} login --status, whoami    Show which key exp-code uses\n  ${APP_NAME} logout                    Forget the saved key\n"
      },
      { find: "Show help for install/remove/uninstall/update/list/config/auth/mcp", replace: "Show help for install/remove/uninstall/update/list/config/login/mcp" },
      // Pi's examples name its direct providers and their models; exp-code's come from the gateway.
      { from: "  # Print a provider API key for an external client\n", to: "  # Interactive mode\n", replace: "" },
      { from: "  # Use different model\n", to: "  # Start with a specific thinking level\n", replace: HELP_MODEL_EXAMPLES },
      { find: "'mcp__radius__*'", replace: "'mcp__github__*'" },
      { find: "Supports globs (anthropic/*, *sonnet*) and fuzzy matching", replace: "Supports globs (experiential-*, *-small-*) and fuzzy matching" },
      { find: "(same as PI_OFFLINE=1)", replace: "(same as EXP_CODE_OFFLINE=1)" },
      // Pi lists the vendor key variables of its direct providers, which exp-code does not load.
      { from: '${chalk.bold("Environment Variables:")}', to: '${chalk.bold("Built-in Tool Names:")}', replace: HELP_ENVIRONMENT }
    ]
  },
  {
    pkg: "pi-coding-agent",
    file: "package-manager-cli.js",
    edits: [
      { find: "update [source|self|pi] [--self|", replace: "update [source|self] [--self|" },
      // `exp-code update` updates exp-code itself (update.ts) and hands Pi only extension and model
      // updates, so Pi's own self-update (of the Pi package) can never run.
      { find: 'const sourceIsSelf = source === "self" || source === "pi";', replace: 'const sourceIsSelf = source === "self";' },
      {
        find: 'function updateTargetIncludesSelf(target) {\n    return target.type === "all" || target.type === "self";',
        replace: "function updateTargetIncludesSelf(target) {\n    return false;"
      }
    ]
  },
  {
    pkg: "pi-coding-agent",
    file: "main.js",
    edits: [
      { find: '"Error: PI_STARTUP_BENCHMARK only supports interactive mode"', replace: '"Error: EXP_CODE_STARTUP_BENCHMARK only supports interactive mode"' },
      { from: "/**\n * Read all content from piped stdin.", to: "function reportDiagnostics(diagnostics) {", replace: READ_PIPED_STDIN },
      { find: "stdinContent = await readPipedStdin();", replace: "stdinContent = await readPipedStdin(parsed.messages.length > 0 || parsed.fileArgs.length > 0);" }
    ]
  },
  {
    pkg: "pi-coding-agent",
    file: "cli/setup.js",
    edits: [
      { find: 'process.env.PI_CODING_AGENT = "true";', replace: 'process.env.EXP_CODE_AGENT = "true";' },
      { find: 'process.env.AI_AGENT = "pi";', replace: 'process.env.AI_AGENT = "exp-code";' }
    ]
  },
  {
    // Every model runs through the Experiential gateway, which does the routing and fallback:
    // none of Pi's direct vendor providers (enabled by GEMINI_API_KEY, OPENAI_API_KEY, auth.json,
    // ...) is registered, and none of their catalogs is fetched.
    pkg: "pi-coding-agent",
    file: "core/model-runtime.js",
    edits: [{ find: "const providers = builtinProviderCatalog.builtinProviders().map((provider) => {", replace: "const providers = [].map((provider) => {" }]
  },
  {
    // The built-in llama.cpp extension registers another direct provider (`/llama`).
    pkg: "pi-coding-agent",
    file: "extensions/index.js",
    edits: [{ find: '    { name: "llama.cpp", factory: llamaExtension, builtin: true },\n', replace: "" }]
  },
  {
    // With every model on the gateway, the provider badge says nothing; models a user's own
    // extension registers keep theirs.
    pkg: "pi-coding-agent",
    file: "modes/interactive/components/model-selector.js",
    edits: [
      {
        find: '"Only showing models from configured providers. Use /login to add providers."',
        replace: '"Every model runs through the Experiential gateway."'
      },
      {
        find: 'const providerBadge = theme.fg("muted", `[${item.provider}]`);',
        replace: 'const providerBadge = item.provider === "experiential" ? "" : theme.fg("muted", ` [${item.provider}]`);'
      },
      { find: "${modelText}${nameText} ${providerBadge}${defaultBadge}", replace: "${modelText}${nameText}${providerBadge}${defaultBadge}" },
      // Enter remembers the model too (interactive-mode.js below), so the separate key is not offered.
      {
        find: '`  ${keyDisplayText("tui.select.confirm")} to select · ${keyDisplayText("app.models.save")} to set as default · ${keyDisplayText("tui.select.cancel")} to cancel`',
        replace: '`  ${keyDisplayText("tui.select.confirm")} to select (remembered for next time) · ${keyDisplayText("tui.select.cancel")} to cancel`'
      }
    ]
  },
  {
    pkg: "pi-coding-agent",
    file: "modes/interactive/components/scoped-models-selector.js",
    edits: [
      {
        find: 'const providerBadge = theme.fg("muted", item.model ? ` [${item.model.provider}]` : " [unavailable]");',
        replace: 'const providerBadge = item.model?.provider === "experiential" ? "" : theme.fg("muted", item.model ? ` [${item.model.provider}]` : " [unavailable]");'
      }
    ]
  },
  {
    pkg: "pi-coding-agent",
    file: "modes/interactive/components/settings-selector.js",
    edits: [
      { find: "return `${model.id} [${model.provider}]`;", replace: 'return model.provider === "experiential" ? model.id : `${model.id} [${model.provider}]`;' },
      {
        find: 'return `${model.id} ${theme.fg("muted", `[${model.provider}]`)}`;',
        replace: 'return model.provider === "experiential" ? model.id : `${model.id} ${theme.fg("muted", `[${model.provider}]`)}`;'
      },
      { find: "instead of Pi's drawn cursor", replace: "instead of the drawn cursor" },
      // `/share` is gone (see core/slash-commands.js), and Pi's install ping is always off
      // (environment.ts sets PI_TELEMETRY=0), so neither setting would do anything.
      {
        find: '            {\n                id: "share-destination",\n                label: "Session share destination",\n                description: "Where to upload sessions with /share (auto prefers Radius when authenticated, otherwise GitHub Gist)",\n                currentValue: config.sessionShareDestination,\n                values: ["auto", "gist", "radius"],\n            },\n',
        replace: ""
      },
      {
        find: '            {\n                id: "install-telemetry",\n                label: "Install telemetry",\n                description: "Send an anonymous version/update ping after changelog-detected updates",\n                currentValue: config.enableInstallTelemetry ? "true" : "false",\n                values: ["true", "false"],\n            },\n',
        replace: ""
      },
      { find: "${cycleThinkingKey} cycles in-session.`", replace: "${cycleThinkingKey} cycles them and remembers the level for the model.`" }
    ]
  },
  {
    // Enter remembers the level for the current model (interactive-mode.js below); the save key
    // also makes it the default for models without a remembered level.
    pkg: "pi-coding-agent",
    file: "modes/interactive/components/thinking-selector.js",
    edits: [
      { find: '`${keyDisplayText("app.thinking.cycle")} cycles thinking levels in-session`', replace: '`${keyDisplayText("app.thinking.cycle")} cycles thinking levels`' },
      {
        find: '`  ${keyDisplayText("tui.select.confirm")} to select · ${keyDisplayText("app.thinking.save")} to set as default · ${keyDisplayText("tui.select.cancel")} to cancel`',
        replace:
          '`  ${keyDisplayText("tui.select.confirm")} to select (remembered for this model) · ${keyDisplayText("app.thinking.save")} to also make it the default · ${keyDisplayText("tui.select.cancel")} to cancel`'
      }
    ]
  },
  {
    // `/share` uploads the session to Pi's services (a Radius artifact or a gist shown on pi.dev)
    // and is removed. `/bug` (which uploaded a report to Pi's developers) and `/changelog` (Pi's
    // changelog) are removed here and registered by exp-code's extension instead (commands.ts).
    pkg: "pi-coding-agent",
    file: "core/slash-commands.js",
    edits: [
      { find: '    { name: "share", description: "Share session via a Radius artifact or a GitHub gist" },\n', replace: "" },
      { find: '    { name: "bug", description: "Report a bug to the Pi developers", argumentHint: "<description>" },\n', replace: "" },
      { find: '    { name: "changelog", description: "Show changelog entries" },\n', replace: "" },
      // With no direct providers, /login is left for MCP servers that ask for a sign-in.
      {
        find: '{ name: "login", description: "Configure provider authentication", argumentHint: "<provider>" }',
        replace: '{ name: "login", description: "Sign in to an MCP server (exp-code login signs in to Experiential)", argumentHint: "<server>" }'
      },
      { find: '{ name: "logout", description: "Remove provider authentication" }', replace: '{ name: "logout", description: "Remove credentials saved by /login (exp-code logout forgets the Experiential key)" }' }
    ]
  },
  {
    pkg: "pi-coding-agent",
    file: "modes/interactive/interactive-mode.js",
    edits: [
      {
        find: "                await this.handleShareCommand();\n",
        replace: '                this.showStatus("exp-code has no /share: it would upload the session to a service outside Experiential. /export saves it as HTML or JSONL.");\n'
      },
      {
        find: "`If this looks like a ${APP_NAME} bug, /bug sends a report to the developers.`",
        replace: '"If this looks like an exp-code bug, /bug opens a prefilled GitHub issue."'
      },
      {
        find: `            if (text === "/bug" || text.startsWith("/bug ")) {
                const hint = text.slice("/bug".length).trim();
                this.editor.setText("");
                await this.handleBugCommand(hint ? hint : undefined);
                return;
            }
`,
        replace: ""
      },
      {
        find: `            if (text === "/changelog") {
                this.handleChangelogCommand();
                this.editor.setText("");
                return;
            }
`,
        replace: ""
      },
      {
        find: `            if (text === "/arminsayshi") {
                this.handleArminSaysHi();
                this.editor.setText("");
                return;
            }
            if (text === "/dementedelves") {
                this.handleDementedDelves();
                this.editor.setText("");
                return;
            }
`,
        replace: ""
      },
      {
        find: "`Pi can explain its own features and look up its docs. Ask it how to use or extend Pi.`",
        replace: '"exp-code can explain its own features. Ask it how to use it, or how to extend it with your own tools and skills."'
      },
      {
        find: "        if (!providerRef) {\n            this.showLoginAuthTypeSelector();\n            return;\n        }",
        replace:
          '        if (!providerRef) {\n            this.showStatus("exp-code signs in to Experiential with `exp-code login` in a terminal (or EXP_API_KEY). /login <server> signs in to an MCP server that asks for it.");\n            return;\n        }'
      },
      { find: "Pi works best with csi-u.", replace: "exp-code works best with csi-u." },
      // exp-code's own commands and skills come from inline extensions (run.ts); like Pi's built-in
      // ones they carry no source tag, where Pi would show "[t]" (temporary) before each description.
      {
        find: 'if (!sourceInfo || sourceInfo.source === "builtin") {',
        replace: 'if (!sourceInfo || sourceInfo.source === "builtin" || sourceInfo.source === "inline" || sourceInfo.source.startsWith("extension:inline:")) {'
      },
      // Pi's own new-version notice is off (PI_SKIP_VERSION_CHECK); its link still names exp-code's releases.
      { find: 'const changelogUrl = "https://pi.dev/changelog";', replace: 'const changelogUrl = "https://github.com/experientiallabs/experiential/releases?q=exp-code";' },
      { find: "`pi-clipboard-${crypto.randomUUID()}.${ext}`", replace: "`exp-code-clipboard-${crypto.randomUUID()}.${ext}`" },
      // Every explicit choice is remembered for the next start. Pi saves a model only through the
      // selector's "set as default" key and a thinking level only as the global default from its
      // selector; picking with Enter, `/model <name>`, Ctrl+P or Shift+Tab lasted one session.
      // A model choice is saved as the default model; a thinking level is saved for the model it
      // was chosen on (Pi's per-model thinking levels, applied when that model is selected again).
      // `--model` and EXP_CODE_MODEL select at startup and are never saved.
      { find: "await this.session.setModel(model, { persist });", replace: "await this.session.setModel(model, { persist: true });" },
      {
        find: "this.showStatus(persist ? `Default model: ${model.provider}/${model.id}` : `Model: ${model.id}`);",
        replace: "this.showStatus(`Model: ${model.id}`);"
      },
      { find: "await this.session.setModel(model, { persist: false });", replace: "await this.session.setModel(model, { persist: true });" },
      { find: "const result = await this.session.cycleModel(direction);", replace: "const result = await this.session.cycleModel(direction, { persist: true });" },
      {
        find: "this.showStatus(`Thinking level: ${newLevel}`);",
        replace: "this.rememberThinkingLevel();\n            this.showStatus(`Thinking level: ${newLevel}`);"
      },
      {
        find: "this.session.setThinkingLevel(level, { persist });",
        replace: "this.session.setThinkingLevel(level, { persist });\n            this.rememberThinkingLevel();"
      },
      {
        find: "    cycleThinkingLevel() {\n        const newLevel = this.session.cycleThinkingLevel();",
        replace:
          "    rememberThinkingLevel() {\n        const model = this.session.model;\n        if (model)\n            this.settingsManager.setModelThinkingLevel(model.provider, model.id, this.session.thinkingLevel);\n    }\n    cycleThinkingLevel() {\n        const newLevel = this.session.cycleThinkingLevel();"
      }
    ]
  },
  {
    // Auto-compaction runs when the context exceeds `contextWindow - reserveTokens`. A gateway
    // model's window is prompt plus its maximum output, and the provider counts that whole output
    // reservation inside the window whatever a request asks for, so the prompt budget is
    // `contextWindow - maxTokens`. Compacting against the whole window (1.05M less 16K) let a
    // session grow past that budget until every request, compaction's own included, was refused
    // with context_length_exceeded. The reserve therefore adds the model's output reservation to
    // Pi's headroom, the headroom capped at a quarter of the prompt budget so a small window still
    // leaves room for a conversation. 1,050,000 / 128,000 compacts above 905,616 tokens.
    pkg: "pi-coding-agent",
    file: "core/settings-manager.js",
    edits: [
      {
        find: '    getCompactionReserveTokens(model) {\n        return this.getCompactionTokenSetting("reserveTokens", model);\n    }',
        replace:
          '    getCompactionReserveTokens(model) {\n        const reserveTokens = this.getCompactionTokenSetting("reserveTokens", model);\n        if (!model || !(model.contextWindow > 0) || !(model.maxTokens > 0))\n            return reserveTokens;\n        const promptBudget = Math.max(0, model.contextWindow - model.maxTokens);\n        return model.maxTokens + Math.min(reserveTokens, Math.floor(promptBudget / 4));\n    }'
      }
    ]
  },
  {
    // A summary replaces the conversation it summarizes (compaction, a split turn's prefix, a
    // branch summary), so it is checked before Pi commits it. Every summary request goes through
    // `completeSummarization`, which now asks for a bounded output (32K tokens, or the model's
    // maximum when smaller) at low reasoning, rejects a truncated, empty or implausibly short
    // answer, and retries once with the model's whole output allowance and its lowest reasoning
    // level. When the retry fails too, it reports an error, so Pi keeps the conversation as it was.
    pkg: "pi-coding-agent",
    file: "core/compaction/compaction.js",
    edits: [
      { find: 'import { completeSimple } from "@earendil-works/pi-ai/compat";', replace: 'import { completeSimple, getSupportedThinkingLevels } from "@earendil-works/pi-ai/compat";' },
      { from: "/**\n * Shared choke point for every compaction/branch-summary summarization call. Wraps the", to: "/**\n * Generate a summary of the conversation using the LLM.", replace: SUMMARY_GUARD }
    ]
  },
  {
    pkg: "pi-coding-agent",
    file: "core/agent-session.js",
    edits: [
      {
        // A model without a remembered level starts at the default level (medium unless the user
        // saved another default), clamped to what the model accepts. Pi carried the previous model's
        // level instead, so a level clamped to "off" on a model that cannot think stayed "off" on
        // the next one, which then reasoned at its route's default while the footer said "thinking off".
        find: "        return this.settingsManager.getDefaultThinkingLevel() ?? this.thinkingLevel ?? DEFAULT_THINKING_LEVEL;\n    }\n    _clampThinkingLevel(",
        replace: "        return this.settingsManager.getDefaultThinkingLevel() ?? DEFAULT_THINKING_LEVEL;\n    }\n    _clampThinkingLevel("
      },
      // A failed automatic compaction keeps the conversation, says so with what to do next, and
      // pauses automatic compaction for that model in that session, so it is not retried (and
      // paid for) before every later model call. A successful compaction or another model resumes it.
      { find: "    _overflowRecoveryAttempted = false;\n    // Branch summarization state", replace: AUTO_COMPACTION_PAUSE_STATE },
      {
        find: "        const settings = this.settingsManager.getCompactionSettings(this.model);\n        if (!settings.enabled)\n            return false;\n        // Skip if message was aborted",
        replace: "        const settings = this.settingsManager.getCompactionSettings(this.model);\n        if (!settings.enabled || this._autoCompactionPaused())\n            return false;\n        // Skip if message was aborted"
      },
      {
        find: "        let cancelledByExtension = false;\n        try {\n            if (!model) {\n                return false;\n            }\n            const pathEntries = this.sessionManager.getBranch();",
        replace: "        let cancelledByExtension = false;\n        try {\n            if (!model || this._autoCompactionPaused()) {\n                return false;\n            }\n            const pathEntries = this.sessionManager.getBranch();"
      },
      {
        find: "            this.sessionManager.appendCompaction(summary, firstKeptEntryId, tokensBefore, details, fromExtension, usage);",
        replace: "            this.sessionManager.appendCompaction(summary, firstKeptEntryId, tokensBefore, details, fromExtension, usage);\n            this._autoCompactionPausedFor = undefined;",
        count: 2
      },
      {
        find: "            const errorMessage = aborted ? undefined : `Compaction failed: ${message}`;",
        replace: "            const errorMessage = aborted ? undefined : `Compaction failed: ${message}${compactionFailureAdvice(message, undefined)}`;"
      },
      {
        find: "                const errorMessage = aborted\n                    ? undefined\n                    : reason === \"overflow\"\n                        ? `Context overflow recovery failed: ${message}`\n                        : `Auto-compaction failed: ${message}`;",
        replace:
          "                if (!aborted && model)\n                    this._autoCompactionPausedFor = this._autoCompactionKey(model);\n                const errorMessage = aborted\n                    ? undefined\n                    : reason === \"overflow\"\n                        ? `Context overflow recovery failed: ${message}${compactionFailureAdvice(message, model)}`\n                        : `Auto-compaction failed: ${message}${compactionFailureAdvice(message, model)}`;"
      },
      { find: "export class AgentSession {", replace: `${COMPACTION_FAILURE_ADVICE}\nexport class AgentSession {` }
    ]
  },
  {
    // Token counts between 1M and 10M keep two significant decimals: a 1,050,000-token window reads
    // 1.05M, not 1.1M.
    pkg: "pi-coding-agent",
    file: "modes/interactive/components/footer.js",
    edits: [
      { find: "return `${(count / 1000000).toFixed(1)}M`;", replace: "return `${Number((count / 1000000).toFixed(2))}M`;" },
      // When the stats and the model do not fit on one line (a narrow terminal), Pi cut the model
      // name ("experiential-small-mo"). The model moves to a line of its own instead.
      {
        from: "            // Need to truncate right side\n            const availableForRight = width - statsLeftWidth - minPadding;",
        to: "        // Apply dim to each part separately.",
        replace: "            statsLine = statsLeft;\n            modelLine = truncateToWidth(rightSide, width, \"...\");\n        }\n"
      },
      { find: "        let statsLine;\n        if (totalNeeded <= width) {", replace: "        let statsLine;\n        let modelLine;\n        if (totalNeeded <= width) {" },
      { find: "        const lines = [pwdLine, dimStatsLeft + dimRemainder];", replace: "        const lines = [pwdLine, dimStatsLeft + dimRemainder];\n        if (modelLine !== undefined)\n            lines.push(theme.fg(\"dim\", \" \".repeat(Math.max(0, width - visibleWidth(modelLine))) + modelLine));" }
    ]
  },
  {
    // --list-models shows a window the way the footer does: 1,050,000 tokens is 1.05M, not 1.1M.
    pkg: "pi-coding-agent",
    file: "cli/list-models.js",
    edits: [{ find: "return millions % 1 === 0 ? `${millions}M` : `${millions.toFixed(1)}M`;", replace: "return `${Number(millions.toFixed(2))}M`;" }]
  },
  {
    // Variables a shell command sees about its session.
    pkg: "pi-coding-agent",
    file: "core/tools/bash.js",
    edits: [
      { find: "env.PI_SESSION_ID", replace: "env.EXP_CODE_SESSION_ID", count: 2 },
      { find: "env.PI_SESSION_FILE", replace: "env.EXP_CODE_SESSION_FILE", count: 2 },
      { find: "env.PI_PROVIDER", replace: "env.EXP_CODE_SESSION_PROVIDER", count: 2 },
      { find: "env.PI_MODEL", replace: "env.EXP_CODE_SESSION_MODEL", count: 2 },
      { find: "env.PI_REASONING_LEVEL", replace: "env.EXP_CODE_SESSION_REASONING_LEVEL", count: 2 },
      // The gateway key exp-code itself runs on stays out of the agent's commands: a test script,
      // build step or prompt-injected command the agent runs could otherwise read and send it.
      // Commands the user runs with `!` keep their whole environment.
      { find: "    delete env.EXP_CODE_SESSION_REASONING_LEVEL;\n", replace: `    delete env.EXP_CODE_SESSION_REASONING_LEVEL;\n${AGENT_COMMAND_UNSET.map((name) => `    delete env.${name};\n`).join("")}` },
      { find: SESSION_ENV_GUIDELINE, replace: EXP_SESSION_ENV_GUIDELINE }
    ]
  },
  { pkg: "pi-coding-agent", file: "core/tools/powershell.js", edits: [{ find: SESSION_ENV_GUIDELINE, replace: EXP_SESSION_ENV_GUIDELINE }] },
  {
    // Temporary files whose paths the model and the user see (truncated tool output and the like).
    pkg: "pi-coding-agent",
    file: "utils/output-files.js",
    edits: [{ find: "return join(tmpdir(), `${prefix}-", replace: 'return join(tmpdir(), `${prefix.replace(/^pi-/, "exp-code-")}-' }]
  },
  { pkg: "pi-coding-agent", file: "utils/clipboard.js", edits: [{ find: "`pi-wsl-clip-", replace: "`exp-code-wsl-clip-" }] },
  { pkg: "pi-coding-agent", file: "utils/clipboard-image.js", edits: [{ find: "`pi-wsl-clip-", replace: "`exp-code-wsl-clip-" }] },
  { pkg: "pi-coding-agent", file: "modes/interactive/external-editor.js", edits: [{ find: '"pi-editor-"', replace: '"exp-code-editor-"' }] },
  {
    pkg: "pi-tui",
    file: "tui-main-screen.js",
    edits: [
      { find: '"pi-tui-crash.log"', replace: '"exp-code-crash.log"' },
      { find: '"pi-tui-debug.log"', replace: '"exp-code-tui-debug.log"' }
    ]
  },
  {
    // What Pi says when there is no model or no key: it pointed at /login for vendor providers and
    // at Pi's provider docs inside node_modules. exp-code's models all come from the gateway key.
    pkg: "pi-coding-agent",
    file: "core/auth-guidance.js",
    edits: [
      {
        from: "    return [\n        \"Use /login to log into a provider via OAuth or API key. See:\",",
        to: "].join(\"\\n\");\n}\nexport function formatNoModelsAvailableMessage",
        replace:
          '    return [\n        "Run `exp-code login` in a terminal to sign in to Experiential, or set EXP_API_KEY to an xpl_ key (EXP_GATEWAY_KEY for a local `exp` gateway).",\n        "`exp-code login --status` shows which key is in use; `exp-code --list-models` lists the models it can call.",\n    '
      }
    ]
  },
  {
    pkg: "pi-coding-agent",
    file: "cli/auth-command.js",
    edits: [
      { find: "  pi auth ", replace: "  exp-code auth ", count: 5 },
      { find: "Credentials are stored in ~/.pi/agent/auth.json.", replace: "Credentials are stored in ~/.exp-code/agent/auth.json." }
    ]
  },
  { pkg: "pi-coding-agent", file: "extensions/mcp/cli.js", edits: [{ find: " or .pi/mcp.json.`", replace: " or .exp-code/mcp.json.`" }] },
  { pkg: "pi-coding-agent", file: "extensions/mcp/index.js", edits: [{ find: " or .pi/mcp.json.`", replace: " or .exp-code/mcp.json.`", count: 2 }] },
  {
    pkg: "pi-coding-agent",
    file: "modes/interactive/theme/theme-schema.js",
    edits: [
      { find: 'title: "Pi Coding Agent Theme"', replace: 'title: "exp-code Theme"' },
      { find: 'description: "Theme schema for Pi coding agent"', replace: 'description: "Theme schema for exp-code"' }
    ]
  },
  {
    // The client name MCP servers see.
    pkg: "pi-coding-agent",
    file: "extensions/mcp/runtime.js",
    edits: [{ find: 'name: "pi",\n            version: VERSION,', replace: 'name: "exp-code",\n            version: VERSION,' }]
  }
];

function occurrences(text: string, needle: string): number {
  let count = 0;
  for (let at = text.indexOf(needle); at !== -1; at = text.indexOf(needle, at + needle.length)) count++;
  return count;
}

/** Apply one file's edits; throws PiPatchError when an edit does not match exactly. */
export function applyFilePatch(patch: FilePatch, source: string): string {
  let text = source;
  for (const edit of patch.edits) {
    const where = `${patch.pkg}/dist/${patch.file}`;
    if ("find" in edit) {
      const expected = edit.count ?? 1;
      const found = occurrences(text, edit.find);
      if (found !== expected) throw new PiPatchError(`${where}: expected ${expected} of ${JSON.stringify(edit.find.slice(0, 80))}, found ${found}`);
      text = text.split(edit.find).join(edit.replace);
    } else {
      if (occurrences(text, edit.from) !== 1 || occurrences(text, edit.to) !== 1) throw new PiPatchError(`${where}: range ${JSON.stringify(edit.from)} is not unique`);
      const start = text.indexOf(edit.from);
      const end = text.indexOf(edit.to);
      if (end < start) throw new PiPatchError(`${where}: range ${JSON.stringify(edit.from)} ends before it starts`);
      text = text.slice(0, start) + edit.replace + text.slice(end);
    }
  }
  return text;
}

/** The patch for a module path or file URL, if exp-code patches it. */
export function patchFor(pathOrUrl: string): FilePatch | undefined {
  const path = pathOrUrl.replaceAll("\\", "/");
  return PI_PATCHES.find((patch) => path.endsWith(`/@earendil-works/${patch.pkg}/dist/${patch.file}`));
}

/** The module source with exp-code's patch applied (unchanged for files it does not patch). */
export function patchSource(pathOrUrl: string, source: string): string {
  const patch = patchFor(pathOrUrl);
  return patch === undefined ? source : applyFilePatch(patch, source);
}

let installed = false;

/** Patch Pi's modules as this process loads them. Call before anything imports Pi. */
export function installPiPatches(): void {
  if (installed) return;
  installed = true;
  registerHooks({
    load(url, context, nextLoad) {
      const result = nextLoad(url, context);
      if (patchFor(url) === undefined || result.source === undefined || result.source === null) return result;
      const source = typeof result.source === "string" ? result.source : Buffer.from(result.source as Uint8Array).toString("utf8");
      return { ...result, source: patchSource(url, source) };
    }
  });
}
