"""Ground-truth harness evaluation through Harbor (optional `harbor` extra).

This subpackage imports the `harbor` PyPI package at module scope and is therefore imported
lazily by its consumers, exactly like the e2b extra: `import wmh` must succeed without it.
"""
