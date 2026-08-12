"""No tests: `environment.py` declares the `Env` protocol and nothing else.

There is nothing here a test could hold. Asserting the protocol lists `reset`, `step`, and
`close` restates the declaration, and exercising a double defined in this file would test the
double. No production code does `isinstance(x, Env)` either, so even the runtime-checkable
behavior is unused: `ty` is what binds implementers to this contract.

The behavior the contract describes is covered where it is implemented, against real state:
`wmo/simulation/evaluation/closed_loop_test.py` for the world-model environment, and
`wmo/runtime/episode_test.py` for the loop that drives reset/step/close in order and snapshots the
live state view.

This file exists because AGENTS.md rule 2 pairs every module with a suite. An empty suite is the
honest answer when a module has no behavior of its own: it says "nothing to assert here, and here
is where the behavior is proven" instead of parking green assertions over a bare declaration.
"""
