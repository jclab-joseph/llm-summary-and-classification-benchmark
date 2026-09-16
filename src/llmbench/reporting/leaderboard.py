"""Leaderboard assembly (CSV / JSON / Markdown).

Metrics of different kinds are never collapsed into one composite score: a
summarization ROUGE and a classification accuracy do not belong on the same
axis, and averaging them would hide exactly the trade-offs this benchmark exists
to expose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llmbench.config.loader import AppConfig

__all__ = [
    "LEADERBOARD_COLUMNS",
    "load_model_summaries",
    "build_leaderboard",
    "write_leaderboard",
    "model_detail_rows",
]

LEADERBOARD_COLUMNS = [
    ("model", "Model"),
    ("input_per_million", "Input $/M"),
    ("output_per_million", "Output $/M"),
    ("summary_en", "Summary EN"),
    ("summary_ko", "Summary KO"),
    ("halu_en_f1", "Halu EN F1"),
    ("halu_ko_f1", "Halu KO F1"),
    ("cls_en", "Classification EN"),
    ("cls_ko", "Classification KO"),
    ("en_ko_consistency", "EN-KO Consistency"),
    ("input_tokens", "Input Tokens"),
    ("output_tokens", "Output Tokens"),
    ("openrouter_cost", "OpenRouter Cost"),
    ("judge_cost", "Judge Cost"),
]


def load_model_summaries(cfg: AppConfig) -> list[dict[str, Any]]:
    root = cfg.results_dir / "models"
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.glob("*/summary.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return out


def _get(data: dict[str, Any], *path: str, default: Any = None) -> Any:
    node: Any = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def build_leaderboard(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        metrics = summary.get("metrics", {})
        cost = summary.get("cost", {})
        usage = summary.get("usage", {})
        pricing = _get(cost, "pricing_snapshot", default={}) or {}

        rows.append(
            {
                "model": _get(summary, "model", "model_id", default=""),
                "display_name": _get(summary, "model", "display_name", default=""),
                "vendor": _get(summary, "model", "vendor", default=""),
                "status": summary.get("status", ""),
                "input_per_million": pricing.get("input_per_million"),
                "output_per_million": pricing.get("output_per_million"),
                "cached_input_per_million": pricing.get("cached_input_per_million"),
                "pricing_source": pricing.get("source"),
                "pricing_retrieved_at": pricing.get("retrieved_at"),
                # Summarization headline = ROUGE-Lsum F (EN / KO reported separately).
                "summary_en": _get(metrics, "summarization", "per_language", "en", "rougeLsum_f"),
                "summary_ko": _get(metrics, "summarization", "per_language", "ko", "rougeLsum_f"),
                "summary_en_chrf": _get(metrics, "summarization", "per_language", "en", "chrf"),
                "summary_ko_chrf": _get(metrics, "summarization", "per_language", "ko", "chrf"),
                "summary_en_bertscore": _get(metrics, "summarization", "per_language", "en", "bertscore_f1"),
                "summary_ko_bertscore": _get(metrics, "summarization", "per_language", "ko", "bertscore_f1"),
                "summary_en_surface_facts": _get(
                    metrics, "summarization", "per_language", "en", "surface_fact_support_precision"
                ),
                "summary_ko_surface_facts": _get(
                    metrics, "summarization", "per_language", "ko", "surface_fact_support_precision"
                ),
                "halu_en_f1": _get(metrics, "hallucination", "per_language", "en", "hallucination_f1"),
                "halu_ko_f1": _get(metrics, "hallucination", "per_language", "ko", "hallucination_f1"),
                "halu_en_accuracy": _get(metrics, "hallucination", "per_language", "en", "accuracy"),
                "halu_ko_accuracy": _get(metrics, "hallucination", "per_language", "ko", "accuracy"),
                "halu_ko_status": _get(metrics, "hallucination", "per_language", "ko", "status"),
                "cls_en": _get(metrics, "classification", "per_language", "en", "accuracy"),
                "cls_ko": _get(metrics, "classification", "per_language", "ko", "accuracy"),
                "cls_en_macro_f1": _get(metrics, "classification", "per_language", "en", "macro_f1"),
                "cls_ko_macro_f1": _get(metrics, "classification", "per_language", "ko", "macro_f1"),
                "cls_overall_accuracy": _get(metrics, "classification", "overall", "accuracy"),
                "cls_overall_macro_f1": _get(metrics, "classification", "overall", "macro_f1"),
                "en_ko_consistency": _get(metrics, "classification", "cross_lingual", "consistency"),
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cached_tokens": usage.get("cached_tokens", 0),
                "reasoning_tokens": usage.get("reasoning_tokens", 0),
                "openrouter_cost": cost.get("stored_benchmark_cost", 0.0),
                "judge_cost": cost.get("judge_cost", 0.0),
                "fresh_cost_this_run": cost.get("fresh_cost_this_run", 0.0),
                "estimated_cache_savings": cost.get("estimated_cache_savings", 0.0),
            }
        )
    rows.sort(key=lambda r: (-(r["cls_overall_accuracy"] or 0.0), r["model"]))
    return rows


def _fmt(value: Any, spec: str) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return format(value, spec)
    return str(value)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [label for _, label in LEADERBOARD_COLUMNS]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |")
    for row in rows:
        cells = [
            str(row["model"]),
            _fmt(row["input_per_million"], ".4f"),
            _fmt(row["output_per_million"], ".4f"),
            _fmt(row["summary_en"], ".4f"),
            _fmt(row["summary_ko"], ".4f"),
            _fmt(row["halu_en_f1"], ".4f"),
            _fmt(row["halu_ko_f1"], ".4f"),
            _fmt(row["cls_en"], ".4f"),
            _fmt(row["cls_ko"], ".4f"),
            _fmt(row["en_ko_consistency"], ".4f"),
            f"{int(row['input_tokens'] or 0):,}",
            f"{int(row['output_tokens'] or 0):,}",
            _fmt(row["openrouter_cost"], ".6f"),
            _fmt(row["judge_cost"], ".6f"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def model_detail_rows(summary: dict[str, Any]) -> dict[str, Any]:
    """The per-model detail block shown under the leaderboard."""
    metrics = summary.get("metrics", {})
    detail: dict[str, Any] = {"model": _get(summary, "model", "model_id", default="")}

    summarization = {}
    for language in ("en", "ko"):
        block = _get(metrics, "summarization", "per_language", language, default=None)
        if not block:
            continue
        summarization[language] = {
            "ROUGE-Lsum": block.get("rougeLsum_f"),
            "BERTScore F1": block.get("bertscore_f1"),
            "chrF++": block.get("chrf"),
            "Surface Fact Support (diagnostic)": block.get("surface_fact_support_precision"),
            "Compression ratio": block.get("compression_ratio"),
            "Output tokens (mean)": block.get("output_tokens_mean"),
            "Empty output rate": block.get("empty_output_rate"),
            "Refusal rate": block.get("refusal_rate"),
        }
    detail["summarization"] = summarization

    hallucination = {}
    for language in ("en", "ko"):
        block = _get(metrics, "hallucination", "per_language", language, default=None)
        if not block:
            continue
        hallucination[language] = {
            "Accuracy": block.get("accuracy"),
            "Macro-F1": block.get("macro_f1"),
            "Hallucination Precision": block.get("hallucination_precision"),
            "Hallucination Recall": block.get("hallucination_recall"),
            "Hallucination F1": block.get("hallucination_f1"),
            "Supported Recall": block.get("supported_recall"),
            "Invalid output rate": block.get("invalid_output_rate"),
            "Status": block.get("status", "OK"),
        }
    detail["hallucination"] = hallucination

    classification = {}
    for language in ("en", "ko"):
        block = _get(metrics, "classification", "per_language", language, default=None)
        if not block:
            continue
        classification[language] = {
            "Accuracy": block.get("accuracy"),
            "Macro-F1": block.get("macro_f1"),
            "Invalid output rate": block.get("invalid_output_rate"),
        }
    cross = _get(metrics, "classification", "cross_lingual", default={}) or {}
    detail["classification"] = classification
    detail["cross_lingual"] = cross
    detail["cost"] = summary.get("cost", {})
    detail["usage"] = summary.get("usage", {})
    detail["judge"] = summary.get("judge", {})
    return detail


def write_leaderboard(cfg: AppConfig, rows: list[dict[str, Any]]) -> dict[str, Path]:
    import pandas as pd

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    csv_columns = [key for key, _ in LEADERBOARD_COLUMNS]
    extra = [k for k in (rows[0].keys() if rows else []) if k not in csv_columns]
    frame = pd.DataFrame(rows, columns=csv_columns + extra) if rows else pd.DataFrame(columns=csv_columns)

    csv_path = cfg.results_dir / "leaderboard.csv"
    frame.to_csv(csv_path, index=False)
    paths["csv"] = csv_path

    json_path = cfg.results_dir / "leaderboard.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths["json"] = json_path

    md_path = cfg.results_dir / "leaderboard.md"
    body = [
        "# LLM Summarization / Hallucination / Classification Benchmark",
        "",
        "All models are called through **OpenRouter**. Costs are the authoritative",
        "`usage.cost` values returned by OpenRouter, not pricing-table estimates.",
        "",
        "Metrics of different kinds are reported side by side and never merged into a",
        "single composite score.",
        "",
        _markdown_table(rows) if rows else "_No model results yet. Run `benchmark run --all`._",
        "",
    ]
    md_path.write_text("\n".join(body), encoding="utf-8")
    paths["markdown"] = md_path
    return paths
