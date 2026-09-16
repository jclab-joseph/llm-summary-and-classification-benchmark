"""Pre-run cost estimation and the in-run budget guard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from llmbench.benchmarks.base import Task
from llmbench.pricing import ResolvedPricing, estimate_cost_usd

__all__ = ["BenchmarkEstimate", "estimate_tasks", "BudgetGuard"]


@dataclass(slots=True)
class BenchmarkEstimate:
    """What a run would cost *in addition to* what is already cached."""

    model_id: str
    pricing: ResolvedPricing
    total_tasks: int = 0
    cached_tasks: int = 0
    pending_tasks: int = 0
    total_cases: int = 0
    cached_cases: int = 0
    pending_cases: int = 0
    est_input_tokens: int = 0
    est_output_tokens: int = 0
    est_input_cost: float = 0.0
    est_output_cost: float = 0.0
    est_total_cost: float = 0.0
    # What the same run would have cost with an empty cache.
    est_total_cost_without_cache: float = 0.0
    per_benchmark: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def estimated_cache_savings(self) -> float:
        return max(0.0, self.est_total_cost_without_cache - self.est_total_cost)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "pricing": self.pricing.to_dict(),
            "total_tasks": self.total_tasks,
            "cached_tasks": self.cached_tasks,
            "pending_tasks": self.pending_tasks,
            "total_cases": self.total_cases,
            "cached_cases": self.cached_cases,
            "pending_cases": self.pending_cases,
            "est_input_tokens": self.est_input_tokens,
            "est_output_tokens": self.est_output_tokens,
            "est_input_cost": self.est_input_cost,
            "est_output_cost": self.est_output_cost,
            "est_total_cost": self.est_total_cost,
            "est_total_cost_without_cache": self.est_total_cost_without_cache,
            "estimated_cache_savings": self.estimated_cache_savings,
            "per_benchmark": self.per_benchmark,
            "skipped": self.skipped,
        }


def estimate_tasks(
    tasks: Sequence[Task],
    cached_keys: Iterable[str],
    pricing: ResolvedPricing,
    *,
    model_id: str,
    skipped: dict[str, str] | None = None,
) -> BenchmarkEstimate:
    """Estimate the cost of the *pending* tasks only.

    Cache hits are excluded from the fresh estimate, which is the number the
    budget guard checks. The all-misses figure is kept alongside so the report
    can show what the cache saved.
    """
    cached = set(cached_keys)
    estimate = BenchmarkEstimate(model_id=model_id, pricing=pricing, skipped=dict(skipped or {}))

    per_benchmark: dict[str, dict[str, Any]] = {}
    all_input = all_output = 0

    for task in tasks:
        stats = per_benchmark.setdefault(
            task.benchmark,
            {
                "total_tasks": 0,
                "cached_tasks": 0,
                "pending_tasks": 0,
                "total_cases": 0,
                "cached_cases": 0,
                "pending_cases": 0,
                "est_input_tokens": 0,
                "est_output_tokens": 0,
                "est_cost": 0.0,
                "per_language": {},
            },
        )
        lang_stats = stats["per_language"].setdefault(
            task.language, {"total_tasks": 0, "cached_tasks": 0, "pending_tasks": 0, "total_cases": 0, "pending_cases": 0}
        )

        stats["total_tasks"] += 1
        stats["total_cases"] += task.case_count
        lang_stats["total_tasks"] += 1
        lang_stats["total_cases"] += task.case_count
        estimate.total_tasks += 1
        estimate.total_cases += task.case_count
        all_input += task.est_input_tokens
        all_output += task.est_output_tokens

        if task.cache_key in cached:
            stats["cached_tasks"] += 1
            stats["cached_cases"] += task.case_count
            lang_stats["cached_tasks"] += 1
            estimate.cached_tasks += 1
            estimate.cached_cases += task.case_count
            continue

        stats["pending_tasks"] += 1
        stats["pending_cases"] += task.case_count
        lang_stats["pending_tasks"] += 1
        lang_stats["pending_cases"] += task.case_count
        estimate.pending_tasks += 1
        estimate.pending_cases += task.case_count
        stats["est_input_tokens"] += task.est_input_tokens
        stats["est_output_tokens"] += task.est_output_tokens
        estimate.est_input_tokens += task.est_input_tokens
        estimate.est_output_tokens += task.est_output_tokens

    costs = estimate_cost_usd(
        pricing, input_tokens=estimate.est_input_tokens, output_tokens=estimate.est_output_tokens
    )
    estimate.est_input_cost = costs["input_cost"]
    estimate.est_output_cost = costs["output_cost"]
    estimate.est_total_cost = costs["total_cost"]
    estimate.est_total_cost_without_cache = estimate_cost_usd(
        pricing, input_tokens=all_input, output_tokens=all_output
    )["total_cost"]

    for name, stats in per_benchmark.items():
        stats["est_cost"] = estimate_cost_usd(
            pricing,
            input_tokens=stats["est_input_tokens"],
            output_tokens=stats["est_output_tokens"],
        )["total_cost"]
    estimate.per_benchmark = per_benchmark
    return estimate


class BudgetGuard:
    """Tracks cumulative `usage.cost` and stops issuing requests before overrun.

    The guard is intentionally conservative: it refuses to *start* a request when
    the already-billed spend plus that request's estimated cost would cross the
    budget, so a run stops with partial results rather than an overspend.
    """

    def __init__(
        self,
        budget_usd: float,
        *,
        stop_margin: float = 0.98,
        allow_over_budget: bool = False,
    ) -> None:
        self.budget_usd = budget_usd
        self.stop_margin = stop_margin
        self.allow_over_budget = allow_over_budget
        self.spent_usd = 0.0
        self.stopped = False
        self.stop_reason: str | None = None

    @property
    def limit(self) -> float:
        return self.budget_usd * self.stop_margin

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget_usd - self.spent_usd)

    def preflight(self, estimated_total_cost: float) -> tuple[bool, str]:
        """Check the whole-run estimate before any request is made."""
        if self.allow_over_budget:
            return True, "OVER BUDGET ALLOWED (--allow-over-budget)"
        if estimated_total_cost > self.budget_usd:
            return (
                False,
                f"estimated fresh cost ${estimated_total_cost:.4f} exceeds budget "
                f"${self.budget_usd:.2f}; re-run with --allow-over-budget or --budget-usd",
            )
        return True, "SAFE TO RUN"

    def can_spend(self, estimated_cost: float) -> bool:
        if self.allow_over_budget:
            return True
        if self.stopped:
            return False
        if self.spent_usd + estimated_cost > self.limit:
            self.stopped = True
            self.stop_reason = (
                f"budget guard stopped the run: billed ${self.spent_usd:.4f} of "
                f"${self.budget_usd:.2f} (stop margin {self.stop_margin:.0%})"
            )
            return False
        return True

    def record(self, actual_cost: float | None) -> None:
        """Record the authoritative billed cost of a completed attempt."""
        if actual_cost:
            self.spent_usd += float(actual_cost)
