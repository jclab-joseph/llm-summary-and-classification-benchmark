"""Persistent metric cache, independent of the inference cache."""

from __future__ import annotations

from typing import Any

from llmbench.cache.keys import build_metric_key
from llmbench.db.store import Store
from llmbench.metrics.base import Evaluator

__all__ = ["MetricCache"]


class MetricCache:
    """Get/put for evaluator outputs keyed by (result, evaluator, reference).

    The model slug is *not* part of the key material (the inference result hash
    already pins the model's output), and neither is anything about pricing or
    reporting.
    """

    def __init__(self, store: Store, *, model_id: str, enabled: bool = True) -> None:
        self.store = store
        self.model_id = model_id
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def get(
        self,
        *,
        evaluator: Evaluator,
        inference_result_hash: str,
        reference_hash: str = "",
        scope: str = "sample",
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        key, _ = build_metric_key(
            inference_result_hash=inference_result_hash,
            evaluator=evaluator.name,
            evaluator_version=evaluator.version,
            evaluator_config=evaluator.config,
            reference_hash=reference_hash,
            scope=scope,
        )
        row = self.store.get_metric(key)
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return dict(row.value_json)

    def put(
        self,
        *,
        evaluator: Evaluator,
        inference_result_hash: str,
        value: dict[str, Any],
        reference_hash: str = "",
        scope: str = "sample",
        benchmark: str = "",
        language: str = "",
    ) -> None:
        if not self.enabled:
            return
        key, _ = build_metric_key(
            inference_result_hash=inference_result_hash,
            evaluator=evaluator.name,
            evaluator_version=evaluator.version,
            evaluator_config=evaluator.config,
            reference_hash=reference_hash,
            scope=scope,
        )
        self.store.save_metric(
            {
                "metric_cache_key": key,
                "inference_result_hash": inference_result_hash,
                "model_id": self.model_id,
                "benchmark": benchmark,
                "language": language,
                "scope": scope,
                "evaluator": evaluator.name,
                "evaluator_version": evaluator.version,
                "evaluator_config_hash": evaluator.config_hash,
                "reference_hash": reference_hash,
                "value_json": value,
            }
        )
