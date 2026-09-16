"""Budget estimation and the in-run spend guard."""

from __future__ import annotations

import pytest

from llmbench.budget import BudgetGuard, estimate_tasks
from llmbench.benchmarks.summarization import build_summarization_tasks
from llmbench.datasets.manifest import read_manifest
from llmbench.pricing import PricingService, ResolvedPricing, estimate_cost_usd


PRICING = ResolvedPricing(
    model_id="google/gemini-2.5-flash-lite",
    input_per_million=0.10,
    output_per_million=0.40,
    cached_input_per_million=0.025,
    retrieved_at="test",
    source="config",
)


def test_estimate_excludes_cache_hits(prepared):
    model = prepared.require_model("google/gemini-2.5-flash-lite")
    manifest = read_manifest(prepared.manifest_dir / prepared.benchmark.benchmarks.summarization.manifest)
    tasks = build_summarization_tasks(prepared, model, manifest).tasks

    none_cached = estimate_tasks(tasks, [], PRICING, model_id=model.model_id)
    half_cached = estimate_tasks(
        tasks, [t.cache_key for t in tasks[: len(tasks) // 2]], PRICING, model_id=model.model_id
    )

    assert none_cached.pending_tasks == len(tasks)
    assert half_cached.pending_tasks == len(tasks) - len(tasks) // 2
    assert half_cached.est_total_cost < none_cached.est_total_cost
    assert half_cached.estimated_cache_savings > 0

    all_cached = estimate_tasks(tasks, [t.cache_key for t in tasks], PRICING, model_id=model.model_id)
    assert all_cached.pending_tasks == 0
    assert all_cached.est_total_cost == 0.0


def test_cost_estimate_uses_cached_input_rate():
    plain = estimate_cost_usd(PRICING, input_tokens=1_000_000, output_tokens=0)
    cached = estimate_cost_usd(PRICING, input_tokens=1_000_000, output_tokens=0, cached_input_tokens=1_000_000)
    assert plain["total_cost"] == pytest.approx(0.10)
    assert cached["total_cost"] == pytest.approx(0.025)


def test_preflight_blocks_when_over_budget():
    """(8) A run that would exceed the budget does not start."""
    guard = BudgetGuard(1.00)
    ok, reason = guard.preflight(1.50)
    assert not ok
    assert "exceeds budget" in reason

    ok, reason = guard.preflight(0.50)
    assert ok
    assert reason == "SAFE TO RUN"


def test_allow_over_budget_override():
    guard = BudgetGuard(1.00, allow_over_budget=True)
    ok, reason = guard.preflight(50.0)
    assert ok
    assert "OVER BUDGET ALLOWED" in reason
    assert guard.can_spend(1000.0)


def test_guard_stops_mid_run_and_records_reason():
    guard = BudgetGuard(1.00, stop_margin=1.0)
    for _ in range(10):
        assert guard.can_spend(0.05)
        guard.record(0.05)

    assert guard.spent_usd == pytest.approx(0.50)
    guard.record(0.49)
    assert guard.can_spend(0.005)
    guard.record(0.005)
    assert not guard.can_spend(0.10)
    assert guard.stopped
    assert "budget guard stopped the run" in guard.stop_reason


def test_guard_uses_authoritative_usage_cost_only():
    guard = BudgetGuard(1.00)
    guard.record(None)
    guard.record(0.0)
    assert guard.spent_usd == 0.0


def test_reconciliation_flags_mismatch(prepared):
    service = PricingService(prepared, None)
    close = service.reconcile(
        PRICING, prompt_tokens=1_000_000, completion_tokens=0, cached_tokens=0, actual_cost=0.1001
    )
    assert close["matches_within_tolerance"]
    assert close["authoritative"] == "openrouter usage.cost"

    far = service.reconcile(
        PRICING, prompt_tokens=1_000_000, completion_tokens=0, cached_tokens=0, actual_cost=0.5
    )
    assert not far["matches_within_tolerance"]
    assert far["actual_usage_cost"] == 0.5
