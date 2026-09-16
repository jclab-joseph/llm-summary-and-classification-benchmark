"""Summarization metrics: ROUGE-Lsum, chrF++, length/compression, refusal detection."""

from __future__ import annotations

import re
import statistics
from functools import lru_cache
from typing import Any, Sequence

from llmbench.config.schema import ChrfMetricConfig, RougeMetricConfig
from llmbench.core.text import estimate_tokens, multilingual_tokens, split_sentences
from llmbench.metrics.base import Evaluator

__all__ = [
    "RougeEvaluator",
    "ChrfEvaluator",
    "is_refusal",
    "summarization_aggregate",
    "MultilingualTokenizer",
]


class MultilingualTokenizer:
    """Adapter exposing the project tokenizer to `rouge_score`.

    `rouge_score`'s default tokenizer strips every character outside `[a-z0-9]`,
    which reduces Korean summaries to the empty string and reports ROUGE 0.0 for
    every model. This tokenizer keeps Latin words whole and splits Hangul/CJK
    into characters.
    """

    def __init__(self, lowercase: bool = True) -> None:
        self.lowercase = lowercase

    def tokenize(self, text: str) -> list[str]:
        return multilingual_tokens(text, lowercase=self.lowercase)


@lru_cache(maxsize=8)
def _rouge_scorer(use_stemmer: bool):
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeLsum"],
        use_stemmer=use_stemmer,
        tokenizer=MultilingualTokenizer(),
        split_summaries=False,
    )


def _sentence_segmented(text: str) -> str:
    """ROUGE-Lsum treats newlines as sentence boundaries."""
    sents = split_sentences(text)
    return "\n".join(sents) if sents else text


class RougeEvaluator:
    """ROUGE-1 / ROUGE-2 / ROUGE-Lsum with a multilingual tokenizer."""

    def __init__(self, config: RougeMetricConfig) -> None:
        self.config = config
        self.evaluator = Evaluator(
            name="rouge",
            version=config.version,
            config={
                "tokenizer": config.tokenizer,
                "use_stemmer": config.use_stemmer,
                "variants": ["rouge1", "rouge2", "rougeLsum"],
            },
        )

    def score(self, reference: str, prediction: str) -> dict[str, float]:
        if not prediction.strip():
            return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeLsum_f": 0.0, "rougeLsum_p": 0.0, "rougeLsum_r": 0.0}
        scorer = _rouge_scorer(self.config.use_stemmer)
        scores = scorer.score(_sentence_segmented(reference), _sentence_segmented(prediction))
        lsum = scores["rougeLsum"]
        return {
            "rouge1_f": float(scores["rouge1"].fmeasure),
            "rouge2_f": float(scores["rouge2"].fmeasure),
            "rougeLsum_f": float(lsum.fmeasure),
            "rougeLsum_p": float(lsum.precision),
            "rougeLsum_r": float(lsum.recall),
        }


class ChrfEvaluator:
    """chrF++ (character n-grams + word bigrams) via sacrebleu."""

    def __init__(self, config: ChrfMetricConfig) -> None:
        self.config = config
        self.evaluator = Evaluator(
            name="chrf",
            version=config.version,
            config={
                "word_order": config.word_order,
                "char_order": config.char_order,
                "beta": config.beta,
            },
        )
        self._metric = None

    def _get_metric(self):
        if self._metric is None:
            from sacrebleu.metrics import CHRF

            self._metric = CHRF(
                char_order=self.config.char_order,
                word_order=self.config.word_order,
                beta=self.config.beta,
            )
        return self._metric

    def score(self, reference: str, prediction: str) -> dict[str, float]:
        if not prediction.strip():
            return {"chrf": 0.0}
        result = self._get_metric().sentence_score(prediction, [reference])
        return {"chrf": float(result.score)}


# --------------------------------------------------------------------------- #
# Refusal / empty-output detection
# --------------------------------------------------------------------------- #
_REFUSAL_PATTERNS = [
    r"\bi (?:can(?:not|'t)|am unable to|won't)\b",
    r"\bi'm (?:sorry|unable)\b",
    r"\bas an ai\b",
    r"\bi do not have (?:access|enough information)\b",
    r"\bunable to (?:summarize|comply|assist)\b",
    r"죄송(?:하지만|합니다)",
    r"요약(?:할 수 없|해 드릴 수 없|이 불가)",
    r"(?:답변|응답)(?:할 수 없|이 불가)",
    r"저는 (?:인공지능|ai)",
    r"제공(?:할 수 없|이 불가)",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(text: str) -> bool:
    """Conservative refusal heuristic.

    Only short outputs count: a long, otherwise-valid summary that merely
    contains the words "I'm sorry" is not a refusal.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if estimate_tokens(stripped) > 80:
        return False
    return bool(_REFUSAL_RE.search(stripped))


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def summarization_aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-sample summarization scores for one language.

    ``rows`` carry per-sample metric values plus the bookkeeping fields the
    report needs (status, output text, source/reference length).
    """
    total = len(rows)
    if total == 0:
        return {"cases": 0}

    scored = [r for r in rows if r["status"] == "SUCCESS"]
    empty = [r for r in scored if not r["output_text"].strip()]
    refusals = [r for r in scored if r.get("refusal")]
    # Empty and refusal outputs stay in the denominator: a model that refuses
    # 20% of the time should not get a free pass on quality.
    non_empty = [r for r in scored if r["output_text"].strip()]

    def metric(key: str) -> float:
        # Skip None: a metric that could not be computed for a sample must not be
        # silently averaged in as 0.0.
        values = [r["metrics"].get(key) for r in scored]
        return _mean([float(v) for v in values if v is not None])

    # Surface fact support is a *micro* average -- supported facts over all
    # extracted facts -- exactly as the diagnostic is defined. Summaries that
    # contain no extractable fact contribute nothing rather than a spurious 1.0.
    facts_total = sum(int(r["metrics"].get("facts_total") or 0) for r in scored)
    facts_supported = sum(int(r["metrics"].get("facts_supported") or 0) for r in scored)

    compressions = [
        r["output_tokens_est"] / r["source_tokens_est"]
        for r in non_empty
        if r.get("source_tokens_est")
    ]
    bert_values = [
        float(r["metrics"]["bertscore_f1"])
        for r in scored
        if r["metrics"].get("bertscore_f1") is not None
    ]

    return {
        "cases": total,
        "scored_cases": len(scored),
        "failed_cases": total - len(scored),
        "rouge1_f": metric("rouge1_f"),
        "rouge2_f": metric("rouge2_f"),
        "rougeLsum_f": metric("rougeLsum_f"),
        "chrf": metric("chrf"),
        "bertscore_f1": _mean(bert_values) if bert_values else None,
        "bertscore_scored_cases": len(bert_values),
        "surface_fact_support_precision": (facts_supported / facts_total) if facts_total else None,
        "compression_ratio": _mean(compressions),
        "output_tokens_mean": _mean([float(r["output_tokens_est"]) for r in scored]),
        "output_chars_mean": _mean([float(len(r["output_text"])) for r in scored]),
        "empty_output_rate": len(empty) / total,
        "refusal_rate": len(refusals) / total,
    }
