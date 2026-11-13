/**
 * Which conversation and turn a gateway call belongs to.
 *
 * A hosted run sends two headers the Experiential gateway already records on every request:
 * `X-Explabs-Session-Id`, its documented conversation identity (stored hashed, so the gateway
 * links every call of one conversation), and `X-Explabs-Tags`, its call attribution (stored as
 * written, so the platform's Observability can name the conversation and the turn and group a
 * whole conversation's calls). The gateway strips both before the model provider sees the request.
 */

/** The conversation and the turn (one run of it) a call belongs to. */
export interface ConversationTrace {
  conversationId: string;
  turnId: string;
}

/** The `X-Explabs-Tags` keys the platform reads. */
export const CONVERSATION_TAG = "conversation";
export const TURN_TAG = "turn";

/** The gateway headers that attribute a call to its conversation and turn. */
export function conversationTraceHeaders(trace: ConversationTrace): Record<string, string> {
  return {
    "X-Explabs-Session-Id": trace.conversationId,
    "X-Explabs-Tags": JSON.stringify({ [CONVERSATION_TAG]: trace.conversationId, [TURN_TAG]: trace.turnId })
  };
}
