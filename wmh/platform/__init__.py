"""Client for the hosted platform: login credentials, HTTP client, and push/pull transfer.

`wmh login` connects this machine to a platform account (an org-scoped API key minted in the
browser or pasted from the keys page); `wmh push`/`wmh pull` then round-trip world-model bundles
and harness docs against the platform's registry surface.
"""
