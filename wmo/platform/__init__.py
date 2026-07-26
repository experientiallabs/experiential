"""Client for the hosted platform: login credentials, HTTP client, and push/pull transfer.

`wmo login` connects this machine to a platform account (an org-scoped API key minted in the
browser or pasted from the keys page); `wmo push`/`wmo pull` then round-trip world-model bundles
and harness docs against the platform's registry surface.
"""
