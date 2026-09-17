"""MASSIVE intent-classification metrics, including EN/KO cross-lingual consistency."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from llmbench.metrics.base import Evaluator

__all__ = [
    "CLASSIFICATION_EVALUATOR",
    "parse_batch_answer",
    "parse_batch_json_answer",
    "classification_scores",
    "cross_lingual_consistency",
]

CLASSIFICATION_EVALUATOR = Evaluator(
    name="classification_metrics",
    version="clsmetrics-1.0",
    config={"averaging": "macro", "invalid_counts_as_wrong": True},
)

_LINE_RE = re.compile(r"^\s*[\*\-•]?\s*(\d{1,3})\s*[:\.\)\-]\s*(\d{1,3})\s*$")
_LOOSE_RE = re.compile(r"(\d{1,3})\s*[:\.\)\-]\s*(\d{1,3})")


def parse_batch_answer(text: str, batch_size: int, label_count: int) -> dict[int, int | None]:
    """Parse the `<item>: <intent>` block a batched classification request returns.

    Returns a mapping of 1-based item index -> 0-based label index (or None when
    the model gave no usable answer for that item).
    """
    answers: dict[int, int | None] = {i: None for i in range(1, batch_size + 1)}
    if not text:
        return answers

    def record(item_raw: str, label_raw: str) -> None:
        item = int(item_raw)
        label = int(label_raw)
        if not (1 <= item <= batch_size):
            return
        if not (1 <= label <= label_count):
            return
        if answers.get(item) is None:
            answers[item] = label - 1

    for line in text.splitlines():
        match = _LINE_RE.match(line)
        if match:
            record(match.group(1), match.group(2))
    if all(v is None for v in answers.values()):
        # Fall back to a looser scan for models that inline the answers.
        for match in _LOOSE_RE.finditer(text):
            record(match.group(1), match.group(2))
    return answers


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_batch_json_answer(
    text: str,
    batch_size: int,
    label_space: Sequence[str],
) -> dict[int, int | None]:
    """Parse the structured-output reply: `{"answers": ["intent", ...]}`.

    A length mismatch fails the **whole batch**. Position is the only link
    between an answer and its utterance, so a short or long array means every
    mapping after the first gap is a guess -- scoring those would invent data.
    They are reported as invalid instead.
    """
    answers: dict[int, int | None] = {i: None for i in range(1, batch_size + 1)}
    if not text:
        return answers

    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate.split("\n", 1)[-1] if "\n" in candidate else candidate
    match = _JSON_BLOCK.search(candidate)
    if not match:
        return answers
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return answers
    if not isinstance(data, dict):
        return answers

    values = data.get("answers")
    if not isinstance(values, list) or len(values) != batch_size:
        return answers

    index = {label: position for position, label in enumerate(label_space)}
    for position, value in enumerate(values, start=1):
        if isinstance(value, str) and value in index:
            answers[position] = index[value]
    return answers


def _macro_f1(golds: Sequence[str], preds: Sequence[str | None], labels: Sequence[str]) -> float:
    """Macro-F1 over the *frozen* label space.

    Averaging over all labels in the manifest label space (not only the ones the
    model happened to predict) keeps the number comparable across models.
    """
    present = [label for label in labels if any(g == label for g in golds)]
    if not present:
        return 0.0
    f1s = []
    for label in present:
        tp = sum(1 for g, p in zip(golds, preds, strict=True) if g == label and p == label)
        fp = sum(1 for g, p in zip(golds, preds, strict=True) if g != label and p == label)
        fn = sum(1 for g, p in zip(golds, preds, strict=True) if g == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return sum(f1s) / len(f1s)


def classification_scores(rows: Sequence[dict[str, Any]], label_space: Sequence[str]) -> dict[str, Any]:
    """Accuracy and macro-F1 for one language.

    ``rows`` are per-utterance: ``{sample_id, semantic_id, gold, predicted, status}``.
    An unparseable or missing answer counts as wrong.
    """
    total = len(rows)
    if total == 0:
        return {"cases": 0}
    golds = [r["gold"] for r in rows]
    preds = [r.get("predicted") for r in rows]
    invalid = sum(1 for p in preds if p is None)
    correct = sum(1 for g, p in zip(golds, preds, strict=True) if g == p)
    return {
        "cases": total,
        "accuracy": correct / total,
        "macro_f1": _macro_f1(golds, preds, label_space),
        "correct": correct,
        "invalid_outputs": invalid,
        "invalid_output_rate": invalid / total,
        "failed_cases": sum(1 for r in rows if r["status"] != "SUCCESS"),
        "label_space_size": len(label_space),
    }


def cross_lingual_consistency(
    en_rows: Sequence[dict[str, Any]],
    ko_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Agreement between the EN and KO predictions for the same semantic id."""
    en = {r["semantic_id"]: r for r in en_rows}
    ko = {r["semantic_id"]: r for r in ko_rows}
    shared = sorted(en.keys() & ko.keys())
    if not shared:
        return {"paired_cases": 0}

    both_correct = en_only = ko_only = both_wrong = same_prediction = 0
    for sid in shared:
        e, k = en[sid], ko[sid]
        e_ok = e.get("predicted") == e["gold"]
        k_ok = k.get("predicted") == k["gold"]
        if e.get("predicted") is not None and e.get("predicted") == k.get("predicted"):
            same_prediction += 1
        if e_ok and k_ok:
            both_correct += 1
        elif e_ok:
            en_only += 1
        elif k_ok:
            ko_only += 1
        else:
            both_wrong += 1

    n = len(shared)
    return {
        "paired_cases": n,
        # Agreement on the *same* predicted intent, right or wrong: measures how
        # stable the model is across languages independently of accuracy.
        "prediction_agreement": same_prediction / n,
        # Agreement on correctness -- the headline "EN-KO consistency" column.
        "consistency": (both_correct + both_wrong) / n,
        "both_correct": both_correct,
        "both_correct_rate": both_correct / n,
        "en_only_correct": en_only,
        "en_only_correct_rate": en_only / n,
        "ko_only_correct": ko_only,
        "ko_only_correct_rate": ko_only / n,
        "both_wrong": both_wrong,
        "both_wrong_rate": both_wrong / n,
    }
