"""Summary hallucination benchmark (HaluEval EN + AI-Hub KO)."""

from __future__ import annotations

from typing import Any, Sequence

from llmbench.benchmarks.base import Task, TaskGroup, make_task
from llmbench.config.loader import AppConfig
from llmbench.config.schema import ModelConfig
from llmbench.core.canonical import hash_obj
from llmbench.datasets.manifest import Manifest
from llmbench.metrics.base import inference_result_hash
from llmbench.metrics.hallucination import (
    HALLUCINATION_EVALUATOR,
    hallucination_scores,
    parse_hallucination_label,
)
from llmbench.prompts.registry import hallucination_prompt

__all__ = ["build_hallucination_tasks", "score_hallucination"]


def build_hallucination_tasks(
    cfg: AppConfig,
    model: ModelConfig,
    manifests: dict[str, Manifest | None],
    *,
    languages: Sequence[str] | None = None,
    alias_resolution: str | None = None,
) -> TaskGroup:
    """``manifests`` maps language -> manifest (None when the source is unavailable)."""
    bench = cfg.benchmark.benchmarks.hallucination
    max_tokens = cfg.benchmark.generation.max_output_tokens.hallucination
    wanted = set(languages) if languages else None

    tasks: list[Task] = []
    skipped: dict[str, str] = {}
    for language, manifest in sorted(manifests.items()):
        if wanted and language not in wanted:
            continue
        if manifest is None:
            skipped[language] = "SKIPPED"
            continue
        for record in manifest.records:
            prompt = hallucination_prompt(
                record.language,
                record.payload["source"],
                record.payload["candidate_summary"],
            )
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
                    # Both the source and the candidate summary determine the case.
                    sample_content_hash=hash_obj(
                        {
                            "source": record.source_hash,
                            "candidate": record.payload["candidate_hash"],
                        }
                    ),
                    prompt=prompt,
                    max_output_tokens=max_tokens,
                    alias_resolution=alias_resolution,
                    meta={"gold_label": record.payload["gold_label"]},
                )
            )
    return TaskGroup(
        benchmark=bench.name,
        tasks=tasks,
        languages=sorted({t.language for t in tasks}),
        skipped=skipped,
    )


def score_hallucination(
    cfg: AppConfig,
    manifests: dict[str, Manifest | None],
    results: dict[str, dict[str, Any]],
    *,
    metric_cache=None,
    languages: Sequence[str] | None = None,
) -> dict[str, Any]:
    wanted = set(languages) if languages else None
    per_language: dict[str, Any] = {}
    manifest_hashes: dict[str, str] = {}

    for language, manifest in sorted(manifests.items()):
        if wanted and language not in wanted:
            continue
        if manifest is None:
            per_language[language] = {"status": "SKIPPED", "cases": 0}
            continue
        manifest_hashes[language] = manifest.meta.manifest_hash

        rows: list[dict[str, Any]] = []
        for record in manifest.records:
            result = results.get(record.sample_id)
            if result is None:
                continue
            output = result.get("output_text", "") or ""
            status = result.get("status", "MISSING")
            inf_hash = inference_result_hash(result.get("cache_key", record.sample_id), output)
            predicted = None
            if status == "SUCCESS":
                cached = (
                    metric_cache.get(
                        evaluator=HALLUCINATION_EVALUATOR,
                        inference_result_hash=inf_hash,
                        reference_hash=record.reference_hash,
                    )
                    if metric_cache is not None
                    else None
                )
                if cached is None:
                    predicted = parse_hallucination_label(output)
                    if metric_cache is not None:
                        metric_cache.put(
                            evaluator=HALLUCINATION_EVALUATOR,
                            inference_result_hash=inf_hash,
                            reference_hash=record.reference_hash,
                            value={"predicted": predicted},
                            benchmark="hallucination",
                            language=language,
                        )
                else:
                    predicted = cached.get("predicted")
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "language": language,
                    "status": status,
                    "output_text": output,
                    "gold_label": record.payload["gold_label"],
                    "predicted": predicted,
                    "document_id": record.payload.get("document_id"),
                }
            )

        scores = hallucination_scores(rows)
        scores["status"] = "OK" if rows else "NO_RESULTS"
        scores["documents"] = len({r["document_id"] for r in rows if r["document_id"]})
        per_language[language] = scores

    return {
        "benchmark": "hallucination",
        "version": cfg.benchmark.benchmarks.hallucination.version,
        "manifest_hashes": manifest_hashes,
        "per_language": per_language,
        "evaluators": {
            "hallucination_metrics": {
                "version": HALLUCINATION_EVALUATOR.version,
                "config": HALLUCINATION_EVALUATOR.config,
            }
        },
    }
