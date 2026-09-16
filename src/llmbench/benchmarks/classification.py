"""MASSIVE intent classification benchmark (batched requests)."""

from __future__ import annotations

from typing import Any, Sequence

from llmbench.benchmarks.base import Task, TaskGroup, make_task
from llmbench.config.loader import AppConfig
from llmbench.config.schema import ModelConfig
from llmbench.core.canonical import hash_obj
from llmbench.datasets.manifest import Manifest, ManifestRecord
from llmbench.metrics.base import inference_result_hash
from llmbench.metrics.classification import (
    CLASSIFICATION_EVALUATOR,
    classification_scores,
    cross_lingual_consistency,
    parse_batch_answer,
)
from llmbench.prompts.registry import classification_prompt

__all__ = ["build_classification_tasks", "score_classification", "batch_records", "label_space_of"]


def label_space_of(manifest: Manifest) -> list[str]:
    """The frozen 60-intent label space, identical for EN and KO."""
    labels = manifest.meta.extra.get("label_space")
    if labels:
        return list(labels)
    return sorted({r.label for r in manifest.records if r.label})


def batch_records(records: Sequence[ManifestRecord], batch_size: int) -> list[list[ManifestRecord]]:
    """Deterministic batching: manifest order, fixed chunk size."""
    return [list(records[i : i + batch_size]) for i in range(0, len(records), batch_size)]


def build_classification_tasks(
    cfg: AppConfig,
    model: ModelConfig,
    manifest: Manifest,
    *,
    languages: Sequence[str] | None = None,
    alias_resolution: str | None = None,
) -> TaskGroup:
    bench = cfg.benchmark.benchmarks.classification
    answer_tokens = cfg.benchmark.generation.max_output_tokens.classification
    wanted = set(languages) if languages else None
    labels = label_space_of(manifest)

    tasks: list[Task] = []
    for language in manifest.languages():
        if wanted and language not in wanted:
            continue
        records = manifest.by_language(language)
        for batch_index, batch in enumerate(batch_records(records, bench.batch_size)):
            prompt = classification_prompt(language, [r.payload["text"] for r in batch], labels)
            tasks.append(
                make_task(
                    cfg=cfg,
                    model=model,
                    benchmark=bench.name,
                    benchmark_version=bench.version,
                    language=language,
                    task_id=f"massive-batch:{language}:{batch_index:04d}",
                    sample_ids=[r.sample_id for r in batch],
                    dataset_revision=manifest.meta.dataset_revision,
                    split=manifest.meta.split,
                    sample_content_hash=hash_obj([r.source_hash for r in batch]),
                    prompt=prompt,
                    answer_tokens=answer_tokens,
                    alias_resolution=alias_resolution,
                    meta={
                        "batch_index": batch_index,
                        "batch_size": len(batch),
                        "label_count": len(labels),
                    },
                )
            )
    return TaskGroup(benchmark=bench.name, tasks=tasks, languages=sorted({t.language for t in tasks}))


def score_classification(
    cfg: AppConfig,
    manifest: Manifest,
    results: dict[str, dict[str, Any]],
    *,
    metric_cache=None,
    languages: Sequence[str] | None = None,
) -> dict[str, Any]:
    bench = cfg.benchmark.benchmarks.classification
    labels = label_space_of(manifest)
    wanted = set(languages) if languages else None

    rows_by_language: dict[str, list[dict[str, Any]]] = {}
    for language in manifest.languages():
        if wanted and language not in wanted:
            continue
        records = manifest.by_language(language)
        rows: list[dict[str, Any]] = []
        for batch_index, batch in enumerate(batch_records(records, bench.batch_size)):
            task_id = f"massive-batch:{language}:{batch_index:04d}"
            result = results.get(task_id)
            if result is None:
                continue
            output = result.get("output_text", "") or ""
            status = result.get("status", "MISSING")
            inf_hash = inference_result_hash(result.get("cache_key", task_id), output)
            reference_hash = hash_obj([r.reference_hash for r in batch])

            parsed: dict[int, int | None]
            cached = (
                metric_cache.get(
                    evaluator=CLASSIFICATION_EVALUATOR,
                    inference_result_hash=inf_hash,
                    reference_hash=reference_hash,
                )
                if metric_cache is not None
                else None
            )
            if cached is not None:
                parsed = {int(k): v for k, v in cached["answers"].items()}
            elif status == "SUCCESS":
                parsed = parse_batch_answer(output, len(batch), len(labels))
                if metric_cache is not None:
                    metric_cache.put(
                        evaluator=CLASSIFICATION_EVALUATOR,
                        inference_result_hash=inf_hash,
                        reference_hash=reference_hash,
                        value={"answers": {str(k): v for k, v in parsed.items()}},
                        benchmark="classification",
                        language=language,
                    )
            else:
                parsed = {i: None for i in range(1, len(batch) + 1)}

            for position, record in enumerate(batch, start=1):
                label_index = parsed.get(position)
                predicted = labels[label_index] if label_index is not None else None
                rows.append(
                    {
                        "sample_id": record.sample_id,
                        "semantic_id": record.payload["semantic_id"],
                        "language": language,
                        "status": status,
                        "gold": record.label,
                        "predicted": predicted,
                        "batch_task_id": task_id,
                    }
                )
        rows_by_language[language] = rows

    per_language = {
        language: classification_scores(rows, labels) for language, rows in rows_by_language.items()
    }

    overall: dict[str, Any] = {}
    all_rows = [r for rows in rows_by_language.values() for r in rows]
    if all_rows:
        overall = classification_scores(all_rows, labels)

    consistency: dict[str, Any] = {}
    if "en" in rows_by_language and "ko" in rows_by_language:
        consistency = cross_lingual_consistency(rows_by_language["en"], rows_by_language["ko"])

    return {
        "benchmark": "classification",
        "version": bench.version,
        "manifest_hash": manifest.meta.manifest_hash,
        "dataset_revision": manifest.meta.dataset_revision,
        "label_space_size": len(labels),
        "batch_size": bench.batch_size,
        "per_language": per_language,
        "overall": overall,
        "cross_lingual": consistency,
        "evaluators": {
            "classification_metrics": {
                "version": CLASSIFICATION_EVALUATOR.version,
                "config": CLASSIFICATION_EVALUATOR.config,
            }
        },
    }
