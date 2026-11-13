/* Explicit in-process native accounting callbacks for exp_gateway_native. */
#ifndef EXP_GATEWAY_NATIVE_CALLBACKS_H
#define EXP_GATEWAY_NATIVE_CALLBACKS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define EXP_GATEWAY_CALLBACK_CAPSULE "exp_gateway_native.callbacks.v1"
#define EXP_GATEWAY_CALLBACK_ABI_VERSION 1

/* Method IDs. The methods field uses bit (method_id - 1). */
#define EXP_GATEWAY_START_ATTEMPT 1
#define EXP_GATEWAY_SETTLE 2
#define EXP_GATEWAY_ABANDON 3
#define EXP_GATEWAY_PRICE 4
#define EXP_GATEWAY_FLEET_ACQUIRE 5
#define EXP_GATEWAY_FLEET_RELEASE 6
#define EXP_GATEWAY_ENFORCE_OUTPUT_SEGMENT 7
#define EXP_GATEWAY_ENFORCE_OUTPUT 8
#define EXP_GATEWAY_GUARDRAIL_INPUT_STATUS 9
#define EXP_GATEWAY_SIGN_DISPATCH 10
#define EXP_GATEWAY_SEAL_REASONING_CONTENT 11
#define EXP_GATEWAY_REMEMBER 12
#define EXP_GATEWAY_TOOL_SEARCH_ROUND 13
#define EXP_GATEWAY_PREPARE_EXPLICIT_CACHE 14
#define EXP_GATEWAY_FINISH_EXPLICIT_CACHE 15
#define EXP_GATEWAY_RENEW_EXPLICIT_CACHE 16
/* Durable host cache authority, selection, claim and observation operations.
 * This method is called by the native accounting owner, not by HTTP routes. */
#define EXP_GATEWAY_CACHE_TRANSACTION 17
/* Preserve a meter observed during cancellation without replaying financial effects. */
#define EXP_GATEWAY_RECORD_LATE_METER 18
/* Bounded content-free online moderation evidence, delivered off the serving path. */
#define EXP_GATEWAY_RECORD_INPUT_MODERATION 19

/* Write exactly once, synchronously, before invoke returns. The receiver copies
 * at most 16 MiB. A false result must be treated as a failed callback. */
typedef bool (*exp_gateway_write_result)(void *, const uint8_t *, size_t);

/* All functions must be thread safe and must not unwind or enter Python.
 * invoke returns 0 for a JSON result, 1 for a sanitized PublicError JSON object,
 * and any other value for an internal error. Native failure never falls back to
 * the Python target. A selected method owns its complete callback contract,
 * including admission, cancellation, retries, terminal accounting and recovery.
 * Borrowed input and sink pointers must not outlive invoke. Retain/release own
 * the context independently of the capsule's own reference. */
typedef struct {
    uint32_t version;
    uint32_t size;
    uint64_t methods;
    void *context;
    void (*retain)(void *);
    void (*release)(void *);
    uint32_t (*invoke)(void *, uint32_t, const uint8_t *, size_t,
                       exp_gateway_write_result, void *);
} exp_gateway_callbacks_v1;

#endif
