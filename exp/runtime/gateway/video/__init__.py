"""The asynchronous ``/v1/videos`` lane: contracts, provider clients, job store, and engine.

Video generation is a job, not a call: the caller creates a job, polls it, and
downloads the rendered asset. The engine admits a create through the same
authority, route, and ledger seams as every synchronous surface, holds the
attempt's reservation open while the provider renders, and settles it exactly
once when the job reaches a terminal state, whichever process observes it.
"""
