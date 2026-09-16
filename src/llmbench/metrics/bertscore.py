"""Multilingual BERTScore F1 (optional extra).

`bert-score` pulls in torch + transformers, which is a multi-GB install, so it
lives behind the `bertscore` extra. When it is not installed the benchmark still
runs end to end and the metric is reported as `null` with an explicit reason --
never silently as 0.0, which would be indistinguishable from a bad model.
"""

from __future__ import annotations

from typing import Any, Sequence

from llmbench.config.schema import BertScoreMetricConfig
from llmbench.metrics.base import Evaluator

__all__ = ["BertScoreEvaluator", "bertscore_available"]


def bertscore_available() -> bool:
    try:
        import bert_score  # noqa: F401
    except Exception:
        return False
    return True


class BertScoreEvaluator:
    def __init__(self, config: BertScoreMetricConfig) -> None:
        self.config = config
        self.evaluator = Evaluator(
            name="bertscore",
            version=config.version,
            config={
                "model_type": config.model_type,
                "num_layers": config.num_layers,
                "idf": config.idf,
                # batch_size deliberately excluded: it changes throughput, not the score.
            },
        )
        self._scorer = None
        self.unavailable_reason: str | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _get_scorer(self):
        if self._scorer is None:
            from bert_score import BERTScorer

            self._scorer = BERTScorer(
                model_type=self.config.model_type,
                num_layers=self.config.num_layers,
                idf=self.config.idf,
                rescale_with_baseline=False,
                batch_size=self.config.batch_size,
            )
        return self._scorer

    def score_batch(
        self,
        references: Sequence[str],
        predictions: Sequence[str],
    ) -> list[float | None]:
        """Score a whole batch at once (BERTScore is far cheaper batched)."""
        if not references:
            return []
        if not self.config.enabled:
            self.unavailable_reason = "disabled in config/benchmark.yaml (metrics.bertscore.enabled=false)"
            return [None] * len(references)
        if not bertscore_available():
            self.unavailable_reason = (
                "bert-score is not installed; run `uv sync --extra bertscore` to enable it"
            )
            return [None] * len(references)
        try:
            scorer = self._get_scorer()
            _, _, f1 = scorer.score(list(predictions), list(references))
            return [float(v) for v in f1.tolist()]
        except Exception as exc:  # pragma: no cover - depends on the optional extra
            self.unavailable_reason = f"BERTScore failed: {type(exc).__name__}: {exc}"
            return [None] * len(references)

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "available": bertscore_available(),
            "model_type": self.config.model_type,
            "num_layers": self.config.num_layers,
            "version": self.config.version,
            "unavailable_reason": self.unavailable_reason,
        }
