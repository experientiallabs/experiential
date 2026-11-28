/**
 * Retry-policy edits in the pinned Pi packages: a reply that already streamed is never resent, and
 * automatic retries wait as long as the gateway asks (Retry-After). pi-patches.ts applies these
 * with its other edits (same exact-match rules, same tests).
 */

import type { FilePatch } from "./pi-patches.js";

/**
 * The wait a failed provider response asks for (pi-ai's utils/provider-retry.js), in ms:
 * `retry-after-ms`, then `retry-after` (seconds or an HTTP date), then the longest
 * `x-ratelimit-reset-requests` / `x-ratelimit-reset-tokens` whose window is exhausted (its
 * `x-ratelimit-remaining-*` is 0; the Experiential gateway sends these on a 429, as durations such
 * as `120s` or `6m0s`). Each provider stores it on the failed message as `retryAfterMs`, which the
 * session's retry policy reads (EXP_RETRY_POLICY).
 */
export const SERVER_RETRY_DELAY = `/** exp-code: a rate-limit reset duration (\`1h2m3.5s\`, \`120s\`, \`20ms\`, or bare seconds), in ms. */
function expResetDurationMs(text) {
    const value = text.trim();
    if (/^\\d+(\\.\\d+)?$/.test(value))
        return Number.parseFloat(value) * 1000;
    const units = { h: 3600000, m: 60000, s: 1000, ms: 1 };
    let total = 0;
    let rest = value;
    for (const match of value.matchAll(/(\\d+(?:\\.\\d+)?)(ms|h|m|s)/g)) {
        total += Number.parseFloat(match[1]) * units[match[2]];
        rest = rest.replace(match[0], "");
    }
    return rest === "" && value !== "" ? total : undefined;
}
/** exp-code: the wait a failed response asks for, in ms, or undefined when it names none. */
export function serverRetryDelayMs(error) {
    const headers = error?.headers;
    if (!(headers instanceof Headers))
        return undefined;
    const retryAfterMs = Number.parseFloat(headers.get("retry-after-ms") ?? "");
    if (Number.isFinite(retryAfterMs) && retryAfterMs >= 0)
        return retryAfterMs;
    const retryAfter = headers.get("retry-after");
    if (retryAfter) {
        const seconds = Number.parseFloat(retryAfter);
        const delayMs = /^\\s*\\d+(\\.\\d+)?\\s*$/.test(retryAfter) ? seconds * 1000 : Date.parse(retryAfter) - Date.now();
        if (Number.isFinite(delayMs))
            return Math.max(0, delayMs);
    }
    let reset;
    for (const window of ["requests", "tokens"]) {
        if (headers.get(\`x-ratelimit-remaining-\${window}\`)?.trim() !== "0")
            continue;
        const text = headers.get(\`x-ratelimit-reset-\${window}\`);
        const delayMs = text ? expResetDurationMs(text) : undefined;
        if (delayMs !== undefined && (reset === undefined || delayMs > reset))
            reset = delayMs;
    }
    return reset;
}
`;

/** What each provider's error handler adds: the wait the failed response asked for. */
export const STORE_RETRY_AFTER = `
            const expRetryAfterMs = serverRetryDelayMs(error);
            if (expRetryAfterMs !== undefined)
                output.retryAfterMs = expRetryAfterMs;`;

/** The import the patched providers use. */
export const RETRY_IMPORT = 'import { retryProviderRequest } from "../utils/provider-retry.js";';

/**
 * exp-code's automatic retry policy for a failed reply (core/agent-session.js), applied when the
 * reply ends, before it is saved, shown or retried, so its error says what happened:
 * - A reply that failed after it streamed anything (text, reasoning or a tool call) is not sent
 *   again: a retry re-sends and re-bills the whole prompt and throws away what arrived. It is kept
 *   (shown, and its text sent with the next prompt so "continue" picks up from it) and marked as
 *   interrupted. Its tool calls never ran, as they had not finished arriving.
 * - A failure before any content is retried as Pi does (up to retry.maxRetries), waiting what the
 *   gateway asked for (Retry-After, or an exhausted rate-limit window) instead of the fixed backoff.
 * - A wait longer than retry.maxAgentDelayMs (60 s by default) is not waited out: the error says
 *   how long and until when, and the user sends the message again then.
 * Only this session-level retry runs: the provider-level one (retry.provider.maxRetries) stays off,
 * so a failure is retried once per policy, with the retry shown in the UI and every attempt saved
 * (and counted in /usage and the footer).
 */
export const EXP_RETRY_POLICY = `/** exp-code: whether a failed reply streamed any content before it failed. */
function expStreamedContent(message) {
    return message.content.some((block) => block.type === "toolCall" ||
        (block.type === "text" && block.text.trim() !== "") ||
        (block.type === "thinking" && (block.thinking ?? "").trim() !== ""));
}
/** exp-code: a wait in words: "40 seconds", "3 minutes", "1 hour", "2 hours 30 minutes". */
function expWaitWords(ms) {
    const words = (value, unit) => \`\${value} \${unit}\${value === 1 ? "" : "s"}\`;
    const seconds = Math.ceil(ms / 1000);
    if (seconds < 90)
        return words(seconds, "second");
    const minutes = Math.ceil(seconds / 60);
    if (minutes < 60)
        return words(minutes, "minute");
    const hours = Math.floor(minutes / 60);
    return minutes % 60 === 0 ? words(hours, "hour") : \`\${words(hours, "hour")} \${words(minutes % 60, "minute")}\`;
}
function expSentence(text) {
    const trimmed = (text ?? "").trimEnd();
    return trimmed === "" ? "The request failed." : /[.!?)]$/.test(trimmed) ? trimmed : \`\${trimmed}.\`;
}
/**
 * exp-code: decide whether a failed reply may be retried automatically and why not, recorded on
 * the message as \`expRetry\` (and in its error text) before it is saved.
 */
export function applyRetryPolicy(message, settings, now = Date.now()) {
    if (message.role !== "assistant" || message.stopReason !== "error" || message.expRetry !== undefined || !settings.enabled)
        return;
    if (expStreamedContent(message)) {
        message.expRetry = { allowed: false, reason: "interrupted" };
        message.errorMessage = \`\${expSentence(message.errorMessage)} The reply was interrupted after it started. What arrived is kept, and exp-code did not send the request again (that would bill the whole prompt again); say "continue" to pick up where it stopped, or ask again.\`;
        return;
    }
    const wait = message.retryAfterMs;
    if (typeof wait !== "number" || !Number.isFinite(wait))
        return;
    const cap = settings.maxAgentDelayMs ?? 60000;
    if (wait > cap) {
        const until = new Date(now + wait).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        message.expRetry = { allowed: false, reason: "wait", waitMs: wait };
        message.errorMessage = \`\${expSentence(message.errorMessage)} The gateway asked to wait \${expWaitWords(wait)} before trying again (until \${until}), so exp-code did not retry. Send your message again after that.\`;
        return;
    }
    message.expRetry = { allowed: true, delayMs: Math.max(0, Math.ceil(wait)) };
}
`;

/** The provider edits: each of the three wires records the wait a failed response asked for. */
export const RETRY_PROVIDER_PATCHES: FilePatch[] = [
  {
    // The wait a failed response asks for (see SERVER_RETRY_DELAY), for the three wires exp-code
    // speaks (completions, responses, messages).
    pkg: "pi-ai",
    file: "utils/provider-retry.js",
    edits: [{ find: "export async function retryProviderRequest(request, options = {}) {", replace: `${SERVER_RETRY_DELAY}export async function retryProviderRequest(request, options = {}) {` }]
  },
  {
    pkg: "pi-ai",
    file: "api/openai-completions.js",
    edits: [
      { find: RETRY_IMPORT, replace: 'import { retryProviderRequest, serverRetryDelayMs } from "../utils/provider-retry.js";' },
      { find: "            output.errorMessage = formatProviderError(normalizeProviderError(error));", replace: `            output.errorMessage = formatProviderError(normalizeProviderError(error));${STORE_RETRY_AFTER}` }
    ]
  },
  {
    pkg: "pi-ai",
    file: "api/openai-responses.js",
    edits: [
      { find: RETRY_IMPORT, replace: 'import { retryProviderRequest, serverRetryDelayMs } from "../utils/provider-retry.js";' },
      {
        find: "            output.stopReason = options?.signal?.aborted ? \"aborted\" : \"error\";\n            const errorMessage = formatProviderError(normalizeProviderError(error), `${model.provider === \"openai\" ? \"OpenAI\" : model.provider} API error`);",
        replace: `            output.stopReason = options?.signal?.aborted ? "aborted" : "error";${STORE_RETRY_AFTER}\n            const errorMessage = formatProviderError(normalizeProviderError(error), \`\${model.provider === "openai" ? "OpenAI" : model.provider} API error\`);`
      }
    ]
  },
  {
    pkg: "pi-ai",
    file: "api/anthropic-messages.js",
    edits: [
      { find: RETRY_IMPORT, replace: 'import { retryProviderRequest, serverRetryDelayMs } from "../utils/provider-retry.js";' },
      {
        find: "            output.errorMessage = error instanceof Error ? withCauseDetail(error) : JSON.stringify(error);",
        replace: `            output.errorMessage = error instanceof Error ? withCauseDetail(error) : JSON.stringify(error);${STORE_RETRY_AFTER}`
      }
    ]
  }
];
