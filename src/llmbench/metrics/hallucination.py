"""Metrics for the binary SUPPORTED / HALLUCINATED factual-consistency task."""

from __future__ import annotations

import re
from typing import Any, Sequence

from llmbench.metrics.base import Evaluator

__all__ = [
    "HALLUCINATION_EVALUATOR",
    "parse_hallucination_label",
    "hallucination_scores",
    "LABELS",
]

LABELS = ("SUPPORTED", "HALLUCINATED")

HALLUCINATION_EVALUATOR = Evaluator(
    name="hallucination_metrics",
    version="halumetrics-1.0",
    config={"labels": list(LABELS), "positive_class": "HALLUCINATED"},
)

_LABEL_RE = re.compile(r"\b(SUPPORTED|HALLUCINATED)\b", re.IGNORECASE)
_KO_HINTS = (
    ("HALLUCINATED", ("환각", "뒷받침되지 않", "지지되지 않", "사실이 아", "불일치")),
    ("SUPPORTED", ("뒷받침", "지지됨", "일치", "사실임")),
)


def parse_hallucination_label(text: str) -> str | None:
    """Extract the label from a model reply, or None if the output is invalid.

    Tolerant of surrounding punctuation/whitespace and of the model echoing
    "Label:", but never guesses: an ambiguous reply is counted as invalid rather
    than silently scored.
    """
    if not text:
        return None
    matches = _LABEL_RE.findall(text)
    if matches:
        found = {m.upper() for m in matches}
        if len(found) == 1:
            return found.pop()
        # Both labels present (e.g. the model echoed the instructions): take the
        # last one only if it terminates the output, otherwise treat as invalid.
        tail = text.strip().upper().rstrip(".。!?\"'` \n")
        for label in LABELS:
            if tail.endswith(label):
                return label
        return None
    upper = text.strip().upper()
    if upper in {"SUPPORT", "SUPPORTS", "TRUE", "YES"}:
        return "SUPPORTED"
    if upper in {"HALLUCINATION", "UNSUPPORTED", "FALSE", "NO"}:
        return "HALLUCINATED"
    for label, hints in _KO_HINTS:
        if any(hint in text for hint in hints):
            return label
    return None


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def hallucination_scores(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy / macro-F1 / per-class P-R-F for one language.

    Invalid outputs are *not* dropped: they count as wrong for accuracy, and are
    reported separately as `invalid_output_rate`. Dropping them would reward a
    model for being unparseable.
    """
    total = len(rows)
    if total == 0:
        return {"cases": 0}

    scored = [r for r in rows if r["status"] == "SUCCESS"]
    predictions = [(r["gold_label"], parse_hallucination_label(r["output_text"])) for r in scored]
    invalid = sum(1 for _, pred in predictions if pred is None)
    valid = [(gold, pred) for gold, pred in predictions if pred is not None]

    correct = sum(1 for gold, pred in valid if gold == pred)
    accuracy = correct / total if total else 0.0

    per_class: dict[str, dict[str, float]] = {}
    f1s = []
    for label in LABELS:
        tp = sum(1 for gold, pred in valid if gold == label and pred == label)
        fp = sum(1 for gold, pred in valid if gold != label and pred == label)
        fn = sum(1 for gold, pred in predictions if gold == label and pred != label)
        precision, recall, f1 = _prf(tp, fp, fn)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for gold, _ in predictions if gold == label),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        f1s.append(f1)

    return {
        "cases": total,
        "scored_cases": len(scored),
        "failed_cases": total - len(scored),
        "valid_outputs": len(valid),
        "invalid_outputs": invalid,
        "invalid_output_rate": invalid / total if total else 0.0,
        "accuracy": accuracy,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "hallucination_precision": per_class["HALLUCINATED"]["precision"],
        "hallucination_recall": per_class["HALLUCINATED"]["recall"],
        "hallucination_f1": per_class["HALLUCINATED"]["f1"],
        "supported_recall": per_class["SUPPORTED"]["recall"],
        "supported_precision": per_class["SUPPORTED"]["precision"],
        "supported_f1": per_class["SUPPORTED"]["f1"],
        "per_class": per_class,
    }
