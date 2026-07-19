"""Run tracking, cost accounting, and hard-budget admission.

Instrument at the provider boundary (`MeteredProvider`) so the world model, GEPA, and the judge are
all metered without changes to those modules. `RunTracker` aggregates `UsageEvent`s into
`UsageTotals` (priced via `wmh.tracking.pricing`) plus a wall-clock duration from an injectable
`Clock`; `RunRecord`s persist under `.wmh/runs/`.

Paid experiments use `SpendLedger` and `BudgetedProvider` separately from descriptive run
tracking. The ledger reserves a conservative call ceiling before dispatch, which makes its hard
cap an admission rule instead of a report generated after spend has already occurred.
"""

from wmh.tracking.budget import (
    BudgetAccount,
    BudgetAccountBinding,
    BudgetBreachError,
    BudgetedProvider,
    BudgetExceededError,
    BudgetIntegrityError,
    BudgetLedgerAuthority,
    BudgetPolicy,
    BudgetScope,
    ExternalSpendAuthority,
    ProviderCostMeter,
    SpendLedger,
    TimedResourceBudget,
    TimedResourceBudgetAccount,
    TimedResourceClass,
    TimedResourceCostMeter,
    TimedResourceReservation,
    TimedResourceRole,
    TokenPriceCeiling,
    bind_budget_account,
    bind_timed_resource_account,
    bootstrap_budget_ledger,
    nano_usd_from_usd,
    open_shared_spend_ledger,
    orphaned_timed_resource_requires_reap,
    reconcile_orphaned_timed_resource,
    resolve_budget_account,
    resolve_timed_resource_account,
    validate_timed_resource_class,
)
from wmh.tracking.clock import Clock, SystemClock
from wmh.tracking.metered import MeteredProvider, classify_build_call
from wmh.tracking.pricing import ModelPrice, cost_usd, price_for
from wmh.tracking.store import load_runs, save_run
from wmh.tracking.tracker import (
    Phase,
    RunRecord,
    RunTracker,
    UsageEvent,
    UsageTotals,
)

__all__ = [
    "BudgetAccount",
    "BudgetAccountBinding",
    "BudgetBreachError",
    "BudgetExceededError",
    "BudgetIntegrityError",
    "BudgetLedgerAuthority",
    "BudgetPolicy",
    "BudgetScope",
    "BudgetedProvider",
    "bind_budget_account",
    "bind_timed_resource_account",
    "bootstrap_budget_ledger",
    "Clock",
    "ExternalSpendAuthority",
    "SystemClock",
    "MeteredProvider",
    "ProviderCostMeter",
    "classify_build_call",
    "ModelPrice",
    "cost_usd",
    "price_for",
    "load_runs",
    "save_run",
    "Phase",
    "RunRecord",
    "RunTracker",
    "SpendLedger",
    "TimedResourceBudget",
    "TimedResourceBudgetAccount",
    "TimedResourceClass",
    "TimedResourceCostMeter",
    "TimedResourceRole",
    "TimedResourceReservation",
    "UsageEvent",
    "UsageTotals",
    "TokenPriceCeiling",
    "nano_usd_from_usd",
    "open_shared_spend_ledger",
    "orphaned_timed_resource_requires_reap",
    "reconcile_orphaned_timed_resource",
    "resolve_budget_account",
    "resolve_timed_resource_account",
    "validate_timed_resource_class",
]
