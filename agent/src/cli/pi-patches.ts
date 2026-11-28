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

const HELP_ENVIRONMENT = `\${chalk.bold("Environment Variables:")}
  EXP_API_KEY                      - Hosted Experiential gateway key (or run exp-code login)
  EXP_GATEWAY_KEY                  - Key of a local exp gateway
  EXP_CODE_MODEL                   - Default model alias
  EXP_CODE_HOME                    - Configuration directory (default: ~/.exp-code)
  EXP_CODE_SESSION_DIR             - Session storage directory (overridden by --session-dir)
  EXP_CODE_OFFLINE                 - Disable startup network operations when set to 1/true/yes
  EXP_CODE_NO_UPDATE_CHECK         - Skip the daily check for a newer exp-code when set to 1
  The README lists every variable.

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
    edits: [{ find: '"Error: PI_STARTUP_BENCHMARK only supports interactive mode"', replace: '"Error: EXP_CODE_STARTUP_BENCHMARK only supports interactive mode"' }]
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
      { find: "${modelText}${nameText} ${providerBadge}${defaultBadge}", replace: "${modelText}${nameText}${providerBadge}${defaultBadge}" }
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
      { find: "instead of Pi's drawn cursor", replace: "instead of the drawn cursor" }
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
      }
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
      // Pi's own new-version notice is off (PI_SKIP_VERSION_CHECK); its link still names exp-code's releases.
      { find: 'const changelogUrl = "https://pi.dev/changelog";', replace: 'const changelogUrl = "https://github.com/experientiallabs/experiential/releases?q=exp-code";' },
      { find: "`pi-clipboard-${crypto.randomUUID()}.${ext}`", replace: "`exp-code-clipboard-${crypto.randomUUID()}.${ext}`" }
    ]
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
      { find: "env.PI_REASONING_LEVEL", replace: "env.EXP_CODE_SESSION_REASONING_LEVEL", count: 2 }
    ]
  },
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
