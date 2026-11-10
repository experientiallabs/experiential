/** Gateway failures mapped to codes a control plane can branch on, with user-facing words. */

export const MODEL_UNAVAILABLE_MESSAGE =
  "This model is not available to your account. Choose another model and send your message again.";

export const RUN_LIMIT_MESSAGE = "The agent reached its limit for one message. Send another message to continue.";

export const RESTART_LIMIT_MESSAGE = "The agent stopped after restarting too many times. Send another message to try again.";

export const INTERRUPTED_LIMIT_MESSAGE = "The model call was interrupted too many times. Send another message to try again.";

/** Map a model error message to a code, when it is one the UI should treat specially. */
export function errorCode(message: string | undefined): string | undefined {
  if (message === undefined) return undefined;
  // The gateway's refusal of an alias the key may not call, or one that does not exist.
  if (/^(403|404)\b/.test(message) && /model/i.test(message)) return "model_unavailable";
  if (/^402\b/.test(message) || /insufficient|credits/i.test(message)) return "insufficient_credits";
  if (/^429\b/.test(message)) return "rate_limited";
  return undefined;
}

/** The error event body for a failed model call. */
export function modelErrorEvent(message: string | undefined): { message: string; code?: string } {
  const code = errorCode(message);
  const text = code === "model_unavailable" ? MODEL_UNAVAILABLE_MESSAGE : (message ?? "The model call failed.");
  return code === undefined ? { message: text } : { message: text, code };
}
