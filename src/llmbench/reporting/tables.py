"""Flat, tabular views of the per-model summaries.

Both RESULT.md and the spreadsheet are built from these, so a column can never
mean one thing in the document and another in the workbook.
"""

from __future__ import annotations

from typing import Any, Sequence

__all__ = [
    "TABLES",
    "build_tables",
    "leaderboard_rows",
    "summarization_rows",
    "hallucination_rows",
    "classification_rows",
    "cross_lingual_rows",
    "classification_mode_rows",
    "surface_fact_rows",
    "cost_rows",
    "usage_rows",
    "judge_rows",
    "run_rows",
]

LANGUAGES = ("en", "ko")


def _get(data: Any, *path: str, default: Any = None) -> Any:
    node = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _model_id(summary: dict[str, Any]) -> str:
    return _get(summary, "model", "model_id", default="")


def _display(summary: dict[str, Any]) -> str:
    return _get(summary, "model", "display_name", default="") or _model_id(summary)


def leaderboard_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The leaderboard as already assembled by `build_leaderboard`."""
    return [dict(row) for row in rows]


def summarization_rows(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for summary in summaries:
        for language in LANGUAGES:
            block = _get(summary, "metrics", "summarization", "per_language", language)
            if not block:
                continue
            out.append(
                {
                    "model": _model_id(summary),
                    "display_name": _display(summary),
                    "language": language,
                    "cases": block.get("cases"),
                    "scored_cases": block.get("scored_cases"),
                    "rouge_lsum_f": block.get("rougeLsum_f"),
                    "rouge1_f": block.get("rouge1_f"),
                    "rouge2_f": block.get("rouge2_f"),
                    "chrf_pp": block.get("chrf"),
                    "bertscore_f1": block.get("bertscore_f1"),
                    "surface_fact_support": block.get("surface_fact_support_precision"),
                    "compression_ratio": block.get("compression_ratio"),
                    "output_tokens_mean": block.get("output_tokens_mean"),
                    "empty_output_rate": block.get("empty_output_rate"),
                    "refusal_rate": block.get("refusal_rate"),
                    "failed_cases": block.get("failed_cases"),
                }
            )
    return out


def hallucination_rows(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for summary in summaries:
        for language in LANGUAGES:
            block = _get(summary, "metrics", "hallucination", "per_language", language)
            if not block:
                continue
            out.append(
                {
                    "model": _model_id(summary),
                    "display_name": _display(summary),
                    "language": language,
                    "status": block.get("status", "OK"),
                    "cases": block.get("cases"),
                    "documents": block.get("documents"),
                    "accuracy": block.get("accuracy"),
                    "macro_f1": block.get("macro_f1"),
                    "hallucination_precision": block.get("hallucination_precision"),
                    "hallucination_recall": block.get("hallucination_recall"),
                    "hallucination_f1": block.get("hallucination_f1"),
                    "supported_recall": block.get("supported_recall"),
                    "supported_f1": block.get("supported_f1"),
                    "invalid_output_rate": block.get("invalid_output_rate"),
                    "failed_cases": block.get("failed_cases"),
                }
            )
    return out


def classification_rows(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for summary in summaries:
        block = _get(summary, "metrics", "classification")
        if not block:
            continue
        for language in LANGUAGES:
            stats = _get(block, "per_language", language)
            if not stats:
                continue
            out.append(
                {
                    "model": _model_id(summary),
                    "display_name": _display(summary),
                    "scope": language,
                    "cases": stats.get("cases"),
                    "accuracy": stats.get("accuracy"),
                    "macro_f1": stats.get("macro_f1"),
                    "correct": stats.get("correct"),
                    "invalid_output_rate": stats.get("invalid_output_rate"),
                    "label_space_size": stats.get("label_space_size"),
                }
            )
        overall = block.get("overall") or {}
        if overall:
            out.append(
                {
                    "model": _model_id(summary),
                    "display_name": _display(summary),
                    "scope": "overall",
                    "cases": overall.get("cases"),
                    "accuracy": overall.get("accuracy"),
                    "macro_f1": overall.get("macro_f1"),
                    "correct": overall.get("correct"),
                    "invalid_output_rate": overall.get("invalid_output_rate"),
                    "label_space_size": overall.get("label_space_size"),
                }
            )
    return out


def cross_lingual_rows(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for summary in summaries:
        cross = _get(summary, "metrics", "classification", "cross_lingual")
        if not cross or not cross.get("paired_cases"):
            continue
        out.append(
            {
                "model": _model_id(summary),
                "display_name": _display(summary),
                "paired_cases": cross.get("paired_cases"),
                "consistency": cross.get("consistency"),
                "prediction_agreement": cross.get("prediction_agreement"),
                "both_correct": cross.get("both_correct"),
                "both_correct_rate": cross.get("both_correct_rate"),
                "en_only_correct": cross.get("en_only_correct"),
                "en_only_correct_rate": cross.get("en_only_correct_rate"),
                "ko_only_correct": cross.get("ko_only_correct"),
                "ko_only_correct_rate": cross.get("ko_only_correct_rate"),
                "both_wrong": cross.get("both_wrong"),
                "both_wrong_rate": cross.get("both_wrong_rate"),
            }
        )
    return out


def surface_fact_rows(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The deterministic diagnostic, broken out per language."""
    out: list[dict[str, Any]] = []
    for summary in summaries:
        for language in LANGUAGES:
            block = _get(
                summary, "metrics", "summarization", "per_language", language, "surface_fact_diagnostic"
            )
            if not block:
                continue
            out.append(
                {
                    "model": _model_id(summary),
                    "display_name": _display(summary),
                    "language": language,
                    "facts_total": block.get("facts_total"),
                    "facts_supported": block.get("facts_supported"),
                    "support_precision": block.get("surface_fact_support_precision"),
                    "number_mismatch": block.get("number_mismatch"),
                    "percentage_mismatch": block.get("percentage_mismatch"),
                    "date_mismatch": block.get("date_mismatch"),
                    "currency_mismatch": block.get("currency_mismatch"),
                    "unsupported_named_entity": block.get("unsupported_named_entity"),
                    "summaries_with_facts": block.get("summaries_with_facts"),
                }
            )
    return out


def cost_rows(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for summary in summaries:
        cost = summary.get("cost") or {}
        pricing = cost.get("pricing_snapshot") or {}
        reconciliation = cost.get("reconciliation") or {}
        out.append(
            {
                "model": _model_id(summary),
                "display_name": _display(summary),
                "input_per_million": pricing.get("input_per_million"),
                "cached_input_per_million": pricing.get("cached_input_per_million"),
                "output_per_million": pricing.get("output_per_million"),
                "pricing_source": pricing.get("source"),
                "pricing_retrieved_at": pricing.get("retrieved_at"),
                "summarization_cost": cost.get("summarization_cost"),
                "hallucination_cost": cost.get("hallucination_cost"),
                "classification_cost": cost.get("classification_cost"),
                "retry_error_cost": cost.get("retry_error_cost"),
                "judge_cost": cost.get("judge_cost"),
                "total_openrouter_cost": cost.get("total_openrouter_cost"),
                "fresh_cost_this_run": cost.get("fresh_cost_this_run"),
                "stored_benchmark_cost": cost.get("stored_benchmark_cost"),
                "estimated_cost_without_cache": cost.get("estimated_cost_without_cache"),
                "estimated_cache_savings": cost.get("estimated_cache_savings"),
                "recomputed_from_pricing_table": reconciliation.get("recomputed_from_pricing_table"),
                "reconciliation_ok": reconciliation.get("matches_within_tolerance"),
            }
        )
    return out


def usage_rows(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for summary in summaries:
        usage = summary.get("usage") or {}
        cache = summary.get("cache") or {}
        out.append(
            {
                "model": _model_id(summary),
                "display_name": _display(summary),
                "successful_cases": usage.get("successful_cases"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": usage.get("cached_tokens"),
                "cache_write_tokens": usage.get("cache_write_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "cache_hits": cache.get("cache_hits"),
                "cache_misses": cache.get("cache_misses"),
                "openrouter_api_calls": cache.get("openrouter_api_calls"),
                "openrouter_http_attempts": cache.get("openrouter_http_attempts"),
            }
        )
    return out


def judge_rows(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for summary in summaries:
        judge = summary.get("judge") or {}
        if not judge.get("enabled"):
            continue
        scores = judge.get("scores") or {}
        out.append(
            {
                "model": _model_id(summary),
                "judge_model": judge.get("judge_model"),
                "profile": judge.get("profile"),
                "self_judge": judge.get("self_judge"),
                "judged_cases": scores.get("judged_cases"),
                "factual_consistency": scores.get("factual_consistency"),
                "key_information_coverage": scores.get("key_information_coverage"),
                "conciseness": scores.get("conciseness"),
                "overall_quality": scores.get("overall_quality"),
                "invalid_outputs": judge.get("invalid_outputs"),
                "judge_cost_usd": judge.get("judge_cost_usd"),
                "warnings": "; ".join(judge.get("warnings") or []),
            }
        )
    return out


def run_rows(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reproducibility metadata: what produced each model's numbers."""
    out: list[dict[str, Any]] = []
    for summary in summaries:
        repro = summary.get("reproducibility") or {}
        manifests = repro.get("manifests") or {}
        scope = summary.get("scope") or {}
        out.append(
            {
                "model": _model_id(summary),
                "status": summary.get("status"),
                "run_id": summary.get("run_id"),
                "generated_at": summary.get("generated_at"),
                "executed_at": repro.get("executed_at"),
                "benchmark_version": repro.get("benchmark_version"),
                "git_commit": repro.get("git_commit"),
                "git_dirty": repro.get("git_dirty"),
                "python_version": repro.get("python_version"),
                "benchmarks": ", ".join(scope.get("benchmarks") or []),
                "languages": ", ".join(scope.get("languages") or []),
                "total_cases": scope.get("total_cases"),
                "routing_config_hash": repro.get("routing_config_hash"),
                "resolved_model": repro.get("alias_resolution"),
                "xlsum_manifest": _get(manifests, "summarization", "hash", default=""),
                "halueval_manifest": _get(manifests, "hallucination:en", "hash", default=""),
                "aihub_manifest": _get(manifests, "hallucination:ko", "hash", default=""),
                "massive_manifest": _get(manifests, "classification", "hash", default=""),
                "skipped": "; ".join(f"{k}: {v}" for k, v in (scope.get("skipped") or {}).items()),
                "notes": "; ".join(summary.get("notes") or []),
            }
        )
    return out


def classification_mode_rows(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Free-text vs structured output, per model and language.

    The text mode measures "can it classify AND follow the batch format"; the
    structured mode constrains the answer to the label space, so it isolates the
    classification itself. The delta is how much of a model's score was format
    compliance.
    """
    from llmbench.reporting.leaderboard import classification_mode_of

    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for summary in summaries:
        block = _get(summary, "metrics", "classification")
        if not block:
            continue
        mode = classification_mode_of(summary)
        model_id = _model_id(summary)
        for language in LANGUAGES:
            stats = _get(block, "per_language", language)
            if not stats:
                continue
            by_model.setdefault(f"{model_id}\u0000{language}", {})[mode] = stats

    rows: list[dict[str, Any]] = []
    for key, modes in sorted(by_model.items()):
        if len(modes) < 2:
            continue
        model_id, language = key.split("\u0000")
        text = modes.get("text") or {}
        structured = modes.get("json_schema") or {}

        def delta(field: str) -> float | None:
            a, b = structured.get(field), text.get(field)
            return None if a is None or b is None else a - b

        rows.append(
            {
                "model": model_id,
                "language": language,
                "text_accuracy": text.get("accuracy"),
                "json_schema_accuracy": structured.get("accuracy"),
                "accuracy_delta": delta("accuracy"),
                "text_macro_f1": text.get("macro_f1"),
                "json_schema_macro_f1": structured.get("macro_f1"),
                "macro_f1_delta": delta("macro_f1"),
                "text_invalid_rate": text.get("invalid_output_rate"),
                "json_schema_invalid_rate": structured.get("invalid_output_rate"),
                "invalid_rate_delta": delta("invalid_output_rate"),
            }
        )
    rows.sort(key=lambda r: -(r["accuracy_delta"] or 0.0))
    return rows


# Sheet name -> (builder, title). Order is the order of the workbook sheets and
# of the RESULT.md sections.
TABLES: dict[str, str] = {
    "Leaderboard": "Leaderboard",
    "Summarization": "Summarization (XL-Sum)",
    "Hallucination": "Hallucination detection",
    "Classification": "Classification (MASSIVE)",
    "CrossLingual": "EN/KO cross-lingual consistency",
    "ClassificationModes": "Free-text vs structured output",
    "SurfaceFacts": "Surface fact support (diagnostic)",
    "Cost": "Cost (OpenRouter usage.cost)",
    "Usage": "Token usage and cache",
    "Judge": "LLM-as-a-Judge (advisory)",
    "Runs": "Reproducibility",
}


def build_tables(
    leaderboard: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    *,
    all_summaries: Sequence[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """All tabular views, keyed by sheet/section name.

    ``summaries`` is one per model (the baseline condition); ``all_summaries``
    additionally carries the other run conditions, which only the mode
    comparison needs.
    """
    return {
        "Leaderboard": leaderboard_rows(leaderboard),
        "Summarization": summarization_rows(summaries),
        "Hallucination": hallucination_rows(summaries),
        "Classification": classification_rows(summaries),
        "CrossLingual": cross_lingual_rows(summaries),
        "ClassificationModes": classification_mode_rows(all_summaries or summaries),
        "SurfaceFacts": surface_fact_rows(summaries),
        "Cost": cost_rows(summaries),
        "Usage": usage_rows(summaries),
        "Judge": judge_rows(summaries),
        "Runs": run_rows(summaries),
    }
