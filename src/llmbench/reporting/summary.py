"""Per-model result file: results/models/{safe_slug}/summary.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llmbench.config.loader import AppConfig
from llmbench.core.reproducibility import utc_now_iso
from llmbench.runner import RunOutcome

__all__ = [
    "build_model_summary",
    "write_model_summary",
    "model_result_dir",
    "run_variant",
    "summary_filename",
    "BASELINE_CLASSIFICATION_MODE",
]


BASELINE_CLASSIFICATION_MODE = "text"


def model_result_dir(cfg: AppConfig, model_id: str) -> Path:
    return cfg.results_dir / "models" / model_id.replace("/", "__")


def run_variant(cfg: AppConfig) -> dict[str, str]:
    """The condition a run was executed under, beyond the model itself."""
    return {"classification_mode": cfg.benchmark.structured_output.classification_mode}


def summary_filename(variant: dict[str, str]) -> str:
    """Baseline keeps `summary.json`; other conditions get their own file.

    Without this a structured-output run would overwrite the text-mode numbers
    and the comparison the run exists to make would be gone.
    """
    mode = variant.get("classification_mode", BASELINE_CLASSIFICATION_MODE)
    if mode == BASELINE_CLASSIFICATION_MODE:
        return "summary.json"
    return f"summary.classification-{mode}.json"


def build_model_summary(cfg: AppConfig, outcome: RunOutcome) -> dict[str, Any]:
    """Everything about one model's run, minus the raw OpenRouter payloads.

    Raw responses stay in the database; dumping them into the report would make
    it unreadable and would duplicate megabytes of text per model.
    """
    plan = outcome.plan
    metrics = outcome.metrics or {}
    usage = outcome.usage or {}

    return {
        "schema_version": 2,
        "generated_at": utc_now_iso(),
        "run_id": outcome.run_id,
        "variant": run_variant(cfg),
        "status": outcome.status,
        "api_provider": "openrouter",
        "model": {
            "model_id": plan.model.model_id,
            "display_name": plan.model.display_name,
            "vendor": plan.model.vendor,
            "safe_slug": plan.model.safe_slug,
            "max_output_tokens": plan.model.max_output_tokens,
            "reasoning": plan.model.reasoning.cache_material(),
            "routing": plan.model.routing.cache_material(),
            "routing_config_hash": plan.model.routing.config_hash,
            "uses_alias": plan.model.uses_alias,
        },
        "scope": {
            "benchmarks": plan.benchmarks,
            "languages": plan.languages,
            "total_tasks": plan.estimate.total_tasks,
            "total_cases": plan.estimate.total_cases,
            "skipped": plan.skipped,
        },
        "cache": {
            "cache_hits": outcome.cache_hits,
            "cache_misses": outcome.cache_misses,
            "openrouter_api_calls": outcome.api_calls,
            "openrouter_http_attempts": outcome.api_attempts,
            "metric_cache": metrics.get("_metric_cache", {}),
        },
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "cached_tokens": usage.get("cached_tokens", 0),
            "cache_write_tokens": usage.get("cache_write_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": usage.get("reasoning_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "successful_cases": usage.get("cases", 0),
        },
        "cost": outcome.cost,
        "failures": outcome.failures,
        "failure_guard": outcome.failure_summary,
        "preflight": outcome.preflight,
        "skipped": {
            "by_failure_guard": outcome.skipped_by_failure_guard,
            "by_budget": outcome.skipped_by_budget,
        },
        "log_path": outcome.log_path,
        "notes": outcome.notes,
        "metrics": {key: value for key, value in metrics.items() if not key.startswith("_")},
        "judge": outcome.judge,
        "reproducibility": outcome.reproducibility,
    }


def write_model_summary(cfg: AppConfig, outcome: RunOutcome) -> Path:
    directory = model_result_dir(cfg, outcome.plan.model.model_id)
    directory.mkdir(parents=True, exist_ok=True)
    payload = build_model_summary(cfg, outcome)
    path = directory / summary_filename(payload["variant"])
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
