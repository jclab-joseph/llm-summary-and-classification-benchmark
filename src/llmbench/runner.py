"""Benchmark orchestration: plan -> (dry run | execute) -> score -> persist."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Sequence

from llmbench.benchmarks.base import Task, TaskGroup
from llmbench.benchmarks.classification import (
    build_classification_tasks,
    score_classification,
    uses_structured_output,
)
from llmbench.benchmarks.hallucination import build_hallucination_tasks, score_hallucination
from llmbench.benchmarks.summarization import build_summarization_tasks, score_summarization
from llmbench.budget import BenchmarkEstimate, BudgetGuard, estimate_tasks
from llmbench.cache.metric_cache import MetricCache
from llmbench.config.loader import AppConfig
from llmbench.config.schema import ModelConfig
from llmbench.core.errors import BenchmarkError
from llmbench.core.reproducibility import environment_snapshot, utc_now_iso
from llmbench.failures import FailureGuard, FailureRecord
from llmbench.logging_setup import get_logger
from llmbench.datasets.manifest import Manifest
from llmbench.datasets.prepare import load_manifests
from llmbench.db.registry import register_manifests
from llmbench.db.store import Store
from llmbench.openrouter.client import OpenRouterClient
from llmbench.openrouter.types import GenerationRequest, GenerationResponse, InferenceStatus
from llmbench.pricing import PricingService, ResolvedPricing, estimate_cost_usd
from llmbench.prompts.registry import PROMPT_VERSIONS

__all__ = ["RunPlan", "RunOutcome", "BenchmarkRunner", "ALL_BENCHMARKS"]

ALL_BENCHMARKS = ("summarization", "hallucination", "classification")

log = get_logger("runner")


@dataclass(slots=True)
class RunPlan:
    model: ModelConfig
    benchmarks: list[str]
    languages: list[str]
    groups: dict[str, TaskGroup]
    tasks: list[Task]
    cached_keys: set[str]
    pending: list[Task]
    estimate: BenchmarkEstimate
    skipped: dict[str, str] = field(default_factory=dict)
    alias_resolution: str | None = None

    @property
    def cached_tasks(self) -> int:
        return self.estimate.cached_tasks

    def cached_cost(self, store: Store) -> float:
        """Authoritative cost already paid for the cache hits in this plan."""
        total = 0.0
        for task in self.tasks:
            if task.cache_key not in self.cached_keys:
                continue
            row = store.get_result(task.cache_key)
            if row is not None and row.usage_cost:
                total += float(row.usage_cost)
        return total


@dataclass(slots=True)
class RunOutcome:
    run_id: str
    model: ModelConfig
    status: str
    plan: RunPlan
    api_calls: int = 0
    api_attempts: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    fresh_cost_usd: float = 0.0
    fresh_input_tokens: int = 0
    fresh_output_tokens: int = 0
    failures: dict[str, int] = field(default_factory=dict)
    failure_summary: dict[str, Any] = field(default_factory=dict)
    aborted: bool = False
    skipped_by_failure_guard: int = 0
    skipped_by_budget: int = 0
    preflight: dict[str, Any] = field(default_factory=dict)
    log_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    reproducibility: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    judge: dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    def __init__(
        self,
        cfg: AppConfig,
        store: Store,
        *,
        client_factory: Callable[[], OpenRouterClient] | None = None,
        on_progress: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.pricing = PricingService(cfg, store)
        self._client_factory = client_factory
        self.on_progress = on_progress or (lambda event, payload: None)
        self._manifests: dict[str, Manifest | None] | None = None

    # ------------------------------------------------------------------ #
    # planning
    # ------------------------------------------------------------------ #
    def manifests(self, *, refresh: bool = False) -> dict[str, Manifest | None]:
        if self._manifests is None or refresh:
            self._manifests = load_manifests(self.cfg)
        return self._manifests

    def build_plan(
        self,
        model: ModelConfig,
        *,
        benchmarks: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
        pricing: ResolvedPricing | None = None,
        retry_failed: bool = False,
        force: bool = False,
    ) -> RunPlan:
        selected = list(benchmarks) if benchmarks else list(ALL_BENCHMARKS)
        unknown = [b for b in selected if b not in ALL_BENCHMARKS]
        if unknown:
            raise BenchmarkError(f"unknown benchmark(s): {unknown}. Available: {list(ALL_BENCHMARKS)}")
        langs = list(languages) if languages else ["en", "ko"]

        manifests = self.manifests()
        alias_resolution = (
            self.store.last_resolved_model(model.model_id) if model.uses_alias else None
        )

        groups: dict[str, TaskGroup] = {}
        skipped: dict[str, str] = {}

        if "summarization" in selected:
            manifest = manifests.get("summarization")
            if manifest is None:
                skipped["summarization"] = "manifest missing - run `benchmark prepare`"
            else:
                groups["summarization"] = build_summarization_tasks(
                    self.cfg, model, manifest, languages=langs, alias_resolution=alias_resolution
                )

        if "hallucination" in selected:
            halu = {"en": manifests.get("hallucination:en"), "ko": manifests.get("hallucination:ko")}
            group = build_hallucination_tasks(
                self.cfg, model, halu, languages=langs, alias_resolution=alias_resolution
            )
            groups["hallucination"] = group
            for language in group.skipped:
                skipped[f"hallucination:{language}"] = (
                    "AI-Hub data not prepared (set AIHUB_FACTUALITY_DATA_PATH)"
                    if language == "ko"
                    else "manifest missing - run `benchmark prepare`"
                )

        if "classification" in selected:
            manifest = manifests.get("classification")
            if manifest is None:
                skipped["classification"] = "manifest missing - run `benchmark prepare`"
            else:
                groups["classification"] = build_classification_tasks(
                    self.cfg, model, manifest, languages=langs, alias_resolution=alias_resolution
                )

        tasks = [task for group in groups.values() for task in group.tasks]
        cached_keys = set() if force else self.store.existing_success_keys(t.cache_key for t in tasks)

        if force:
            pending = list(tasks)
        else:
            failed = self.store.failed_keys(model.model_id)
            pending = []
            for task in tasks:
                if task.cache_key in cached_keys:
                    continue
                if task.cache_key in failed and not retry_failed:
                    # A previously failed case is not retried silently; the CLI
                    # points at --retry-failed so the decision stays explicit.
                    continue
                pending.append(task)

        resolved_pricing = pricing or self.pricing.resolve(model.model_id)
        estimate = estimate_tasks(
            tasks,
            {t.cache_key for t in tasks if t.cache_key in cached_keys},
            resolved_pricing,
            model_id=model.model_id,
            skipped=skipped,
        )
        # `pending` can be narrower than "not cached" (see the retry rules above).
        estimate.pending_tasks = len(pending)
        estimate.pending_cases = sum(t.case_count for t in pending)
        estimate.est_input_tokens = sum(t.est_input_tokens for t in pending)
        estimate.est_output_tokens = sum(t.est_output_tokens for t in pending)
        costs = estimate_cost_usd(
            resolved_pricing,
            input_tokens=estimate.est_input_tokens,
            output_tokens=estimate.est_output_tokens,
        )
        estimate.est_input_cost = costs["input_cost"]
        estimate.est_output_cost = costs["output_cost"]
        estimate.est_total_cost = costs["total_cost"]

        return RunPlan(
            model=model,
            benchmarks=selected,
            languages=langs,
            groups=groups,
            tasks=tasks,
            cached_keys=cached_keys,
            pending=pending,
            estimate=estimate,
            skipped=skipped,
            alias_resolution=alias_resolution,
        )

    # ------------------------------------------------------------------ #
    # execution
    # ------------------------------------------------------------------ #
    def _request_for(self, model: ModelConfig, task: Task) -> GenerationRequest:
        """Build the OpenRouter request, omitting parameters the model rejects.

        `routing.require_parameters` is what stops an endpoint from silently
        ignoring `temperature=0`; the price of that guarantee is that we must not
        send a parameter the model does not advertise, or OpenRouter answers 404
        for every endpoint.
        """
        generation = self.cfg.benchmark.generation
        return GenerationRequest(
            model=model.model_id,
            messages=task.prompt.messages,
            max_tokens=task.max_output_tokens,
            temperature=generation.temperature if model.supports("temperature") else None,
            top_p=generation.top_p if model.supports("top_p") else None,
            seed=generation.seed if model.supports("seed") else None,
            reasoning=model.reasoning.to_request_payload() if model.supports("reasoning") else None,
            provider=model.routing.to_request_payload(),
            response_format=task.response_format if model.supports_json_schema() else None,
        )

    def _persist(
        self,
        *,
        run_id: str,
        model: ModelConfig,
        task: Task,
        response: GenerationResponse,
    ) -> None:
        usage = response.usage
        self.store.save_result(
            {
                "cache_key": task.cache_key,
                "run_id": run_id,
                "api_provider": "openrouter",
                "model_id": model.model_id,
                "requested_model": model.model_id,
                "resolved_model": response.resolved_model,
                "upstream_provider": response.upstream_provider,
                "routing_config_hash": model.routing.config_hash,
                "benchmark": task.benchmark,
                "benchmark_version": task.benchmark_version,
                "dataset_revision": task.dataset_revision,
                "split": task.split,
                "sample_id": task.task_id,
                "language": task.language,
                "sample_content_hash": task.sample_content_hash,
                "status": str(response.status),
                "output_text": response.text,
                "finish_reason": response.finish_reason,
                "request_id": response.request_id,
                "prompt_tokens": usage.prompt_tokens,
                "cached_tokens": usage.cached_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
                "completion_tokens": usage.completion_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "total_tokens": usage.total_tokens,
                "usage_cost": usage.cost,
                "usage_json": usage.to_dict(),
                "raw_response_json": response.raw,
                "key_material_json": task.key_material,
                "attempts": response.attempts,
                "latency_ms": response.latency_ms,
                "error": response.error,
            }
        )
        # Every physical attempt, including billed failures.
        self.store.save_attempts(
            {
                "run_id": run_id,
                "cache_key": task.cache_key,
                "model_id": model.model_id,
                "benchmark": task.benchmark,
                "sample_id": task.task_id,
                "attempt_no": record.attempt_no,
                "status": str(record.status),
                "request_id": record.request_id,
                "resolved_model": record.resolved_model,
                "upstream_provider": record.upstream_provider,
                "http_status": record.http_status,
                "prompt_tokens": record.usage.prompt_tokens,
                "cached_tokens": record.usage.cached_tokens,
                "cache_write_tokens": record.usage.cache_write_tokens,
                "completion_tokens": record.usage.completion_tokens,
                "reasoning_tokens": record.usage.reasoning_tokens,
                "total_tokens": record.usage.total_tokens,
                "usage_cost": record.usage.cost,
                "latency_ms": record.latency_ms,
                "error": record.error,
            }
            for record in response.attempt_records
        )

    async def _execute(
        self,
        run_id: str,
        model: ModelConfig,
        plan: RunPlan,
        client: OpenRouterClient,
        guard: BudgetGuard,
        failures: FailureGuard,
        outcome: RunOutcome,
    ) -> None:
        semaphore = asyncio.Semaphore(self.cfg.benchmark.concurrency.max_parallel_requests)
        per_task_estimate = (
            plan.estimate.est_total_cost / len(plan.pending) if plan.pending else 0.0
        )
        lock = asyncio.Lock()
        completed = 0
        total = len(plan.pending)

        async def run_one(task: Task) -> None:
            nonlocal completed
            async with semaphore:
                async with lock:
                    # Tasks skipped by either guard are deliberately NOT persisted:
                    # they were never attempted, so leaving no row lets the next run
                    # pick them up normally instead of needing --retry-failed.
                    if failures.tripped:
                        outcome.skipped_by_failure_guard += 1
                        return
                    allowed = guard.can_spend(per_task_estimate)
                if not allowed:
                    async with lock:
                        outcome.skipped_by_budget += 1
                    return

                response = await client.generate(self._request_for(model, task))

                async with lock:
                    outcome.api_calls += 1
                    outcome.api_attempts += len(response.attempt_records)
                    for record in response.attempt_records:
                        guard.record(record.usage.cost)
                        outcome.fresh_cost_usd += record.usage.billed_cost
                    outcome.fresh_input_tokens += response.usage.prompt_tokens
                    outcome.fresh_output_tokens += response.usage.completion_tokens

                    if response.ok:
                        failures.record_success()
                    else:
                        key = str(response.status)
                        outcome.failures[key] = outcome.failures.get(key, 0) + 1
                        http_status = next(
                            (r.http_status for r in reversed(response.attempt_records) if r.http_status),
                            None,
                        )
                        failures.record_failure(
                            FailureRecord(
                                status=key,
                                error=response.error,
                                http_status=http_status,
                                sample_id=task.task_id,
                            )
                        )
                        log.warning(
                            "case failed benchmark=%s sample=%s status=%s http=%s: %s",
                            task.benchmark,
                            task.task_id,
                            key,
                            http_status,
                            (response.error or "")[:300],
                        )
                        if failures.tripped and not outcome.aborted:
                            outcome.aborted = True
                            log.error("failure guard tripped: %s", failures.reason)

                    completed += 1
                    self.on_progress(
                        "task",
                        {
                            "completed": completed,
                            "total": total,
                            "benchmark": task.benchmark,
                            "status": str(response.status),
                            "fresh_cost": outcome.fresh_cost_usd,
                            "failures": failures.failures,
                        },
                    )
                self._persist(run_id=run_id, model=model, task=task, response=response)

        await asyncio.gather(*(run_one(task) for task in plan.pending))

    # ------------------------------------------------------------------ #
    # pre-run capability check
    # ------------------------------------------------------------------ #
    def _parameters_in_use(self, model: ModelConfig, *, include_response_format: bool) -> set[str]:
        """Which request parameters this run would actually send."""
        generation = self.cfg.benchmark.generation
        sending = set()
        if model.supports("temperature"):
            sending.add("temperature")
        if model.supports("top_p"):
            sending.add("top_p")
        if model.supports("reasoning"):
            sending.add("reasoning")
        if generation.seed is not None and model.supports("seed"):
            sending.add("seed")
        if include_response_format and model.supports_json_schema():
            # A strict schema needs both capabilities, so both are "in use".
            sending.update({"response_format", "structured_outputs"})
        return sending

    def preflight_capabilities(
        self,
        model: ModelConfig,
        metadata: dict[str, Any] | None,
        *,
        include_response_format: bool = False,
    ) -> dict[str, Any]:  # noqa: D401
        """Compare configured parameters against OpenRouter's live metadata.

        This is what turns "1,300 identical 404s" into "stop, fix one line of
        config". It costs nothing extra: the /models response was already fetched
        for pricing.
        """
        # A model that cannot serve a strict schema cannot run the structured
        # condition at all. Letting it through would send the structured *prompt*
        # without the schema and file the result as if it had been constrained.
        unsupported_condition = (
            [
                {
                    "parameter": "structured_outputs",
                    "configured": False,
                    "advertised": False,
                    "severity": "error",
                    "hint": (
                        f"{model.model_id} cannot serve a strict json_schema response, so it "
                        "cannot run classification with --classification-mode json_schema. "
                        "Run it in text mode, or disable it for this comparison."
                    ),
                }
            ]
            if include_response_format and not model.supports_json_schema()
            else []
        )

        if not metadata:
            return {
                "checked": False,
                "reason": "live model metadata unavailable",
                "blocking": unsupported_condition,
            }

        entry = metadata.get(model.model_id)
        if entry is None:
            return {
                "checked": False,
                "reason": f"{model.model_id} is not listed by OpenRouter",
                "blocking": unsupported_condition,
            }

        supported = entry.get("supported_parameters") or []
        if not supported:
            return {
                "checked": False,
                "reason": "OpenRouter listed no supported_parameters",
                "blocking": unsupported_condition,
            }

        sending = self._parameters_in_use(model, include_response_format=include_response_format)
        problems = model.capability_mismatches(supported)
        blocking = unsupported_condition + [
            p for p in problems if p["severity"] == "error" and p["parameter"] in sending
        ]
        informational = [p for p in problems if p not in blocking]
        return {
            "checked": True,
            "supported_parameters": sorted(supported),
            "sending": sorted(sending),
            "blocking": blocking,
            "informational": informational,
            "require_parameters": model.routing.require_parameters,
        }

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #
    def _collect_results(self, plan: RunPlan) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for task in plan.tasks:
            row = self.store.get_result(task.cache_key)
            if row is None:
                continue
            out[task.task_id] = {
                "cache_key": row.cache_key,
                "status": row.status,
                "output_text": row.output_text or "",
                "usage_cost": row.usage_cost,
                "language": row.language,
                "benchmark": row.benchmark,
            }
        return out

    def collect_results(self, plan: RunPlan) -> dict[str, dict[str, Any]]:
        """Public view of the cached results covering this plan's tasks."""
        return self._collect_results(plan)

    def score(self, plan: RunPlan, *, use_metric_cache: bool = True) -> dict[str, Any]:
        results = self._collect_results(plan)
        manifests = self.manifests()
        metric_cache = MetricCache(self.store, model_id=plan.model.model_id, enabled=use_metric_cache)
        metrics: dict[str, Any] = {}

        if "summarization" in plan.groups and manifests.get("summarization") is not None:
            metrics["summarization"] = score_summarization(
                self.cfg,
                manifests["summarization"],
                results,
                metric_cache=metric_cache,
                languages=plan.languages,
            )
        if "hallucination" in plan.groups:
            metrics["hallucination"] = score_hallucination(
                self.cfg,
                {"en": manifests.get("hallucination:en"), "ko": manifests.get("hallucination:ko")},
                results,
                metric_cache=metric_cache,
                languages=plan.languages,
            )
        if "classification" in plan.groups and manifests.get("classification") is not None:
            metrics["classification"] = score_classification(
                self.cfg,
                manifests["classification"],
                results,
                metric_cache=metric_cache,
                languages=plan.languages,
            )

        metrics["_metric_cache"] = {"hits": metric_cache.hits, "misses": metric_cache.misses}
        return metrics

    # ------------------------------------------------------------------ #
    # cost reporting
    # ------------------------------------------------------------------ #
    def cost_report(self, plan: RunPlan, outcome: RunOutcome, pricing: ResolvedPricing) -> dict[str, Any]:
        model_id = plan.model.model_id
        by_benchmark = self.store.cost_by_benchmark(model_id)
        attempts = self.store.attempt_cost_summary(model_id)
        usage = self.store.usage_totals(model_id)
        judge_cost = self.store.judge_cost(model_id)

        # Cost already paid for the cache hits that this run reused: exactly what
        # re-running them would have cost again.
        cache_hit_cost = plan.cached_cost(self.store)
        stored_cost = attempts["total_cost"]
        without_cache = cache_hit_cost + outcome.fresh_cost_usd

        reconciliation = self.pricing.reconcile(
            pricing,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            cached_tokens=usage["cached_tokens"],
            actual_cost=stored_cost,
        )

        return {
            "pricing_snapshot": pricing.to_dict(),
            "authoritative_source": "openrouter usage.cost",
            "by_benchmark": by_benchmark,
            "summarization_cost": by_benchmark.get("summarization", 0.0),
            "hallucination_cost": by_benchmark.get("hallucination", 0.0),
            "classification_cost": by_benchmark.get("classification", 0.0),
            "retry_error_cost": attempts["error_cost"],
            "judge_cost": judge_cost,
            "total_openrouter_cost": stored_cost + judge_cost,
            "fresh_cost_this_run": outcome.fresh_cost_usd,
            "stored_benchmark_cost": stored_cost,
            "estimated_cost_without_cache": without_cache,
            "estimated_cache_savings": max(0.0, without_cache - outcome.fresh_cost_usd),
            "attempts": attempts,
            "reconciliation": reconciliation,
        }

    # ------------------------------------------------------------------ #
    # top-level entry point
    # ------------------------------------------------------------------ #
    async def run(
        self,
        model: ModelConfig,
        *,
        benchmarks: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
        budget_usd: float | None = None,
        allow_over_budget: bool = False,
        dry_run: bool = False,
        retry_failed: bool = False,
        force: bool = False,
        refresh_pricing: bool = True,
        skip_preflight: bool = False,
        client: OpenRouterClient | None = None,
        use_metric_cache: bool = True,
    ) -> RunOutcome:
        own_client = client is None
        client = client or self._make_client(dry_run=dry_run)

        try:
            pricing = self.pricing.resolve(model.model_id)
            metadata: dict[str, Any] = {}
            if refresh_pricing:
                live = await self.pricing.refresh(client, [model.model_id])
                pricing = live.get(model.model_id, pricing)
                metadata = self.pricing.last_metadata
                if self.pricing.last_refresh_error:
                    log.warning(
                        "live model metadata unavailable (%s); using config/pricing.yaml",
                        self.pricing.last_refresh_error,
                    )

            plan = self.build_plan(
                model,
                benchmarks=benchmarks,
                languages=languages,
                pricing=pricing,
                retry_failed=retry_failed,
                force=force,
            )

            budget = budget_usd if budget_usd is not None else self.cfg.benchmark.budget.default_usd
            guard = BudgetGuard(
                budget,
                stop_margin=self.cfg.benchmark.budget.stop_margin,
                allow_over_budget=allow_over_budget,
            )
            ok, reason = guard.preflight(plan.estimate.est_total_cost)

            # `response_format` only reaches the API when classification runs in
            # structured-output mode, so only then is a model without support a
            # blocking problem.
            selected_benchmarks = list(benchmarks) if benchmarks else list(ALL_BENCHMARKS)
            needs_response_format = (
                "classification" in selected_benchmarks and uses_structured_output(self.cfg)
            )
            preflight = self.preflight_capabilities(
                model, metadata, include_response_format=needs_response_format
            )
            blocking = preflight.get("blocking") or []
            for problem in preflight.get("informational") or []:
                log.info("capability note for %s: %s", model.model_id, problem["hint"])

            run_id = f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
            outcome = RunOutcome(
                run_id=run_id,
                model=model,
                status="DRY_RUN" if dry_run else ("BLOCKED" if not ok else "RUNNING"),
                plan=plan,
                cache_hits=plan.estimate.cached_tasks,
                cache_misses=len(plan.pending),
            )
            outcome.reproducibility = self._reproducibility(plan, pricing)
            outcome.preflight = preflight

            if blocking and not skip_preflight:
                names = ", ".join(p["parameter"] for p in blocking)
                outcome.status = "BLOCKED"
                outcome.notes.append(
                    f"{model.model_id} does not support: {names}. With "
                    "routing.require_parameters=true OpenRouter would reject every request "
                    "with HTTP 404 'No endpoints found that can handle the requested "
                    "parameters'. Fix config/models.yaml (see `benchmark models --check`) "
                    "or re-run with --skip-preflight."
                )
                for problem in blocking:
                    outcome.notes.append(problem["hint"])
                log.error("preflight blocked %s: unsupported parameters %s", model.model_id, names)
                outcome.cost = {
                    "pricing_snapshot": pricing.to_dict(),
                    "estimated_fresh_cost": plan.estimate.est_total_cost,
                    "budget_usd": budget,
                    "budget_status": "NOT STARTED",
                    "budget_reason": "blocked by the pre-run capability check",
                }
                return outcome

            if dry_run:
                outcome.cost = {
                    "pricing_snapshot": pricing.to_dict(),
                    "estimated_fresh_input_cost": plan.estimate.est_input_cost,
                    "estimated_fresh_output_cost": plan.estimate.est_output_cost,
                    "estimated_fresh_cost": plan.estimate.est_total_cost,
                    "budget_usd": budget,
                    "budget_status": "SAFE TO RUN" if ok else "OVER BUDGET",
                    "budget_reason": reason,
                }
                outcome.notes.append("dry run: no OpenRouter inference request was made")
                for problem in blocking:
                    outcome.notes.append(f"capability problem: {problem['hint']}")
                return outcome

            if not ok:
                outcome.notes.append(reason)
                outcome.cost = {
                    "pricing_snapshot": pricing.to_dict(),
                    "estimated_fresh_cost": plan.estimate.est_total_cost,
                    "budget_usd": budget,
                    "budget_status": "OVER BUDGET",
                    "budget_reason": reason,
                }
                return outcome

            # Mirror the frozen manifests into `benchmarks` / `samples` so results
            # can be joined back to the exact sample they came from.
            register_manifests(self.cfg, self.store, self.manifests())
            self.store.upsert_model(
                model_id=model.model_id,
                display_name=model.display_name,
                vendor=model.vendor,
                role=model.role,
                config=model.model_dump(),
                routing_config_hash=model.routing.config_hash,
            )
            self.store.create_run(
                run_id=run_id,
                model_id=model.model_id,
                benchmarks=list(plan.benchmarks),
                languages=list(plan.languages),
                status="RUNNING",
                dry_run=False,
                budget_usd=budget,
                config_json=self._run_config(plan, pricing),
                environment_json=environment_snapshot(),
                git_commit=outcome.reproducibility.get("git_commit"),
                benchmark_version=self.cfg.benchmark.benchmark_version,
            )

            failure_guard_cfg = self.cfg.benchmark.failure_guard
            failures = FailureGuard(
                enabled=failure_guard_cfg.enabled,
                max_consecutive_failures=failure_guard_cfg.max_consecutive_failures,
                max_failure_rate=failure_guard_cfg.max_failure_rate,
                min_attempts_before_rate_check=failure_guard_cfg.min_attempts_before_rate_check,
            )
            log.info(
                "run %s model=%s benchmarks=%s pending=%d cached=%d budget=$%.2f",
                run_id,
                model.model_id,
                ",".join(plan.benchmarks),
                len(plan.pending),
                plan.estimate.cached_tasks,
                budget,
            )

            if plan.pending:
                await self._execute(run_id, model, plan, client, guard, failures, outcome)

            outcome.failure_summary = failures.summary()
            if failures.tripped:
                outcome.status = "ABORTED"
                if failures.reason:
                    outcome.notes.append(failures.reason)
                for message, count in failures.top_errors(3):
                    outcome.notes.append(f"[{count}x] {message}")
                outcome.notes.append(
                    "Nothing was stored for the cases that were skipped, so a re-run picks "
                    "them up normally once the cause is fixed."
                )
            elif guard.stopped:
                outcome.status = "BUDGET_STOPPED"
            else:
                outcome.status = "COMPLETED"
            if guard.stopped and guard.stop_reason:
                outcome.notes.append(guard.stop_reason)
            log.info(
                "run %s finished status=%s calls=%d failures=%d fresh_cost=$%.6f",
                run_id,
                outcome.status,
                outcome.api_calls,
                failures.failures,
                outcome.fresh_cost_usd,
            )

            outcome.metrics = self.score(plan, use_metric_cache=use_metric_cache)
            outcome.cost = self.cost_report(plan, outcome, pricing)
            outcome.usage = self.store.usage_totals(model.model_id)

            self.store.finish_run(
                run_id,
                status=outcome.status,
                fresh_cost_usd=outcome.fresh_cost_usd,
                judge_cost_usd=outcome.cost.get("judge_cost", 0.0),
                api_calls=outcome.api_calls,
                cache_hits=outcome.cache_hits,
                cache_misses=outcome.cache_misses,
                finished_at=datetime.now(UTC),
                notes="; ".join(outcome.notes) or None,
            )
            return outcome
        finally:
            if own_client:
                await client.aclose()

    # ------------------------------------------------------------------ #
    def _make_client(self, *, dry_run: bool) -> OpenRouterClient:
        if self._client_factory is not None:
            client = self._client_factory()
            client.dry_run = dry_run
            return client
        retry = self.cfg.benchmark.retry
        return OpenRouterClient(
            timeout=self.cfg.benchmark.concurrency.request_timeout_seconds,
            max_attempts=retry.max_attempts,
            initial_backoff=retry.initial_backoff_seconds,
            max_backoff=retry.max_backoff_seconds,
            jitter=retry.jitter_seconds,
            dry_run=dry_run,
        )

    def _run_config(self, plan: RunPlan, pricing: ResolvedPricing) -> dict[str, Any]:
        return {
            "benchmarks": plan.benchmarks,
            "languages": plan.languages,
            "generation": self.cfg.benchmark.generation.cache_material(),
            "truncation": self.cfg.benchmark.truncation.cache_material(),
            "routing": plan.model.routing.cache_material(),
            "reasoning": plan.model.reasoning.cache_material(),
            "prompt_versions": PROMPT_VERSIONS,
            "structured_output": self.cfg.benchmark.structured_output.model_dump(),
            "pricing_snapshot": pricing.to_dict(),
        }

    def _reproducibility(self, plan: RunPlan, pricing: ResolvedPricing) -> dict[str, Any]:
        manifests = self.manifests()
        return {
            **environment_snapshot(),
            "executed_at": utc_now_iso(),
            "api_provider": "openrouter",
            "model_id": plan.model.model_id,
            "requested_model": plan.model.model_id,
            "alias_resolution": plan.alias_resolution,
            "uses_alias": plan.model.uses_alias,
            "routing_config": plan.model.routing.cache_material(),
            "routing_config_hash": plan.model.routing.config_hash,
            "reasoning_config": plan.model.reasoning.cache_material(),
            "generation": self.cfg.benchmark.generation.cache_material(),
            "truncation_policy": self.cfg.benchmark.truncation.cache_material(),
            "structured_output": self.cfg.benchmark.structured_output.model_dump(),
            "prompt_versions": PROMPT_VERSIONS,
            "metric_versions": {
                "rouge": self.cfg.benchmark.metrics.rouge.version,
                "chrf": self.cfg.benchmark.metrics.chrf.version,
                "bertscore": self.cfg.benchmark.metrics.bertscore.version,
                "surface_facts": self.cfg.benchmark.metrics.surface_facts.version,
                "classification_metrics": self.cfg.benchmark.metrics.classification_metrics.version,
                "hallucination_metrics": self.cfg.benchmark.metrics.hallucination_metrics.version,
            },
            "manifests": {
                key: (
                    {
                        "path": str(manifest.path.name),
                        "hash": manifest.meta.manifest_hash,
                        "dataset_revision": manifest.meta.dataset_revision,
                        "counts": manifest.meta.counts,
                    }
                    if manifest is not None
                    else None
                )
                for key, manifest in manifests.items()
            },
            "pricing_snapshot": pricing.to_dict(),
        }
