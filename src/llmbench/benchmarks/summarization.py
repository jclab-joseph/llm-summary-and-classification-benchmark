"""XL-Sum summarization benchmark: task construction and scoring."""

from __future__ import annotations

from typing import Any, Sequence

from llmbench.benchmarks.base import Task, TaskGroup, make_task
from llmbench.config.loader import AppConfig
from llmbench.config.schema import ModelConfig
from llmbench.core.text import estimate_tokens
from llmbench.datasets.manifest import Manifest
from llmbench.metrics.base import inference_result_hash
from llmbench.metrics.bertscore import BertScoreEvaluator
from llmbench.metrics.summarization import (
    ChrfEvaluator,
    RougeEvaluator,
    is_refusal,
    summarization_aggregate,
)
from llmbench.metrics.surface_facts import SurfaceFactEvaluator, surface_fact_aggregate
from llmbench.prompts.registry import summarization_prompt

__all__ = ["build_summarization_tasks", "score_summarization"]


def build_summarization_tasks(
    cfg: AppConfig,
    model: ModelConfig,
    manifest: Manifest,
    *,
    languages: Sequence[str] | None = None,
    alias_resolution: str | None = None,
) -> TaskGroup:
    bench = cfg.benchmark.benchmarks.summarization
    answer_tokens = cfg.benchmark.generation.max_output_tokens.summarization
    wanted = set(languages) if languages else None

    tasks: list[Task] = []
    for record in manifest.records:
        if wanted and record.language not in wanted:
            continue
        prompt = summarization_prompt(record.language, record.payload["source"])
        tasks.append(
            make_task(
                cfg=cfg,
                model=model,
                benchmark=bench.name,
                benchmark_version=bench.version,
                language=record.language,
                task_id=record.sample_id,
                sample_ids=[record.sample_id],
                dataset_revision=record.dataset_revision,
                split=record.split,
                sample_content_hash=record.source_hash,
                prompt=prompt,
                answer_tokens=answer_tokens,
                alias_resolution=alias_resolution,
                meta={"reference_hash": record.reference_hash},
            )
        )
    return TaskGroup(benchmark=bench.name, tasks=tasks, languages=sorted({t.language for t in tasks}))


def score_summarization(
    cfg: AppConfig,
    manifest: Manifest,
    results: dict[str, dict[str, Any]],
    *,
    metric_cache=None,
    languages: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score every generated summary, reusing cached metric values where possible.

    ``results`` maps task_id -> {status, output_text, cache_key}. ``metric_cache``
    is an optional object exposing ``get(evaluator, inference_hash, reference_hash)``
    and ``put(...)``; it is what keeps a metric-only change from recomputing
    everything.
    """
    metrics_cfg = cfg.benchmark.metrics
    rouge = RougeEvaluator(metrics_cfg.rouge)
    chrf = ChrfEvaluator(metrics_cfg.chrf)
    surface = SurfaceFactEvaluator(metrics_cfg.surface_facts)
    bertscore = BertScoreEvaluator(metrics_cfg.bertscore)

    wanted = set(languages) if languages else None
    rows: list[dict[str, Any]] = []
    pending_bert: list[int] = []

    for record in manifest.records:
        if wanted and record.language not in wanted:
            continue
        result = results.get(record.sample_id)
        if result is None:
            continue
        output = result.get("output_text", "") or ""
        status = result.get("status", "MISSING")
        inf_hash = inference_result_hash(result.get("cache_key", record.sample_id), output)

        sample_metrics: dict[str, Any] = {}
        if status == "SUCCESS":
            for evaluator_obj, fn in (
                (rouge.evaluator, lambda: rouge.score(record.payload["reference"], output)),
                (chrf.evaluator, lambda: chrf.score(record.payload["reference"], output)),
                (
                    surface.evaluator,
                    lambda: surface.score(record.payload["source"], output),
                ),
            ):
                value = None
                if metric_cache is not None:
                    value = metric_cache.get(
                        evaluator=evaluator_obj,
                        inference_result_hash=inf_hash,
                        reference_hash=record.reference_hash,
                    )
                if value is None:
                    value = fn()
                    if metric_cache is not None:
                        metric_cache.put(
                            evaluator=evaluator_obj,
                            inference_result_hash=inf_hash,
                            reference_hash=record.reference_hash,
                            value=value,
                            benchmark="summarization",
                            language=record.language,
                        )
                sample_metrics.update(value)

        row = {
            "sample_id": record.sample_id,
            "language": record.language,
            "status": status,
            "output_text": output,
            "metrics": sample_metrics,
            "output_tokens_est": estimate_tokens(output),
            "source_tokens_est": record.payload.get("source_tokens_est")
            or estimate_tokens(record.payload["source"]),
            "reference": record.payload["reference"],
            "reference_hash": record.reference_hash,
            "inference_result_hash": inf_hash,
            "refusal": is_refusal(output) if status == "SUCCESS" else False,
        }
        rows.append(row)
        if status == "SUCCESS" and output.strip():
            pending_bert.append(len(rows) - 1)

    # --- BERTScore: cached per sample, computed in one batch for the misses --- #
    if bertscore.enabled and pending_bert:
        need_compute: list[int] = []
        for idx in pending_bert:
            row = rows[idx]
            cached = (
                metric_cache.get(
                    evaluator=bertscore.evaluator,
                    inference_result_hash=row["inference_result_hash"],
                    reference_hash=row["reference_hash"],
                )
                if metric_cache is not None
                else None
            )
            if cached is not None:
                row["metrics"]["bertscore_f1"] = cached.get("bertscore_f1")
            else:
                need_compute.append(idx)
        if need_compute:
            scores = bertscore.score_batch(
                [rows[i]["reference"] for i in need_compute],
                [rows[i]["output_text"] for i in need_compute],
            )
            for idx, score in zip(need_compute, scores, strict=True):
                rows[idx]["metrics"]["bertscore_f1"] = score
                if metric_cache is not None and score is not None:
                    metric_cache.put(
                        evaluator=bertscore.evaluator,
                        inference_result_hash=rows[idx]["inference_result_hash"],
                        reference_hash=rows[idx]["reference_hash"],
                        value={"bertscore_f1": score},
                        benchmark="summarization",
                        language=rows[idx]["language"],
                    )

    per_language: dict[str, Any] = {}
    for language in sorted({r["language"] for r in rows}):
        lang_rows = [r for r in rows if r["language"] == language]
        aggregate = summarization_aggregate(lang_rows)
        aggregate["surface_fact_diagnostic"] = surface_fact_aggregate(
            [r["metrics"] for r in lang_rows if r["status"] == "SUCCESS"]
        )
        per_language[language] = aggregate

    return {
        "benchmark": "summarization",
        "version": cfg.benchmark.benchmarks.summarization.version,
        "manifest_hash": manifest.meta.manifest_hash,
        "dataset_revision": manifest.meta.dataset_revision,
        "per_language": per_language,
        "evaluators": {
            "rouge": {"version": rouge.evaluator.version, "config": rouge.evaluator.config},
            "chrf": {"version": chrf.evaluator.version, "config": chrf.evaluator.config},
            "surface_facts": {"version": surface.evaluator.version, "config": surface.evaluator.config},
            "bertscore": bertscore.describe(),
        },
    }
