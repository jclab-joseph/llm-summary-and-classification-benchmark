"""RESULT.md, the Excel workbook and the shared table views."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmbench.reporting.build import generate_reports
from llmbench.reporting.leaderboard import build_leaderboard
from llmbench.reporting.result_doc import render_result_markdown
from llmbench.reporting.summary import model_result_dir
from llmbench.reporting.tables import build_tables


def _summary(
    model_id: str = "google/gemini-2.5-flash-lite",
    *,
    status: str = "COMPLETED",
    ko_hallucination: bool = True,
    judge: bool = False,
) -> dict:
    halu_ko = (
        {
            "status": "OK",
            "cases": 400,
            "documents": 200,
            "accuracy": 0.61,
            "macro_f1": 0.6,
            "hallucination_precision": 0.62,
            "hallucination_recall": 0.58,
            "hallucination_f1": 0.6,
            "supported_recall": 0.64,
            "invalid_output_rate": 0.0,
            "failed_cases": 0,
        }
        if ko_hallucination
        else {"status": "SKIPPED", "cases": 0}
    )
    return {
        "schema_version": 1,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "run_id": "run-test",
        "status": status,
        "api_provider": "openrouter",
        "model": {
            "model_id": model_id,
            "display_name": "Test Model",
            "vendor": "test",
            "safe_slug": model_id.replace("/", "__"),
        },
        "scope": {
            "benchmarks": ["summarization", "hallucination", "classification"],
            "languages": ["en", "ko"],
            "total_cases": 3200,
            "skipped": {},
        },
        "cache": {
            "cache_hits": 1300,
            "cache_misses": 0,
            "openrouter_api_calls": 0,
            "openrouter_http_attempts": 0,
        },
        "usage": {
            "prompt_tokens": 1_097_914,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "completion_tokens": 54_986,
            "reasoning_tokens": 0,
            "total_tokens": 1_152_900,
            "successful_cases": 1300,
        },
        "cost": {
            "pricing_snapshot": {
                "model_id": model_id,
                "input_per_million": 0.1,
                "output_per_million": 0.4,
                # Deliberately None: an all-None column used to crash the workbook.
                "cached_input_per_million": None,
                "retrieved_at": "2026-01-01T00:00:00+00:00",
                "source": "openrouter",
            },
            "summarization_cost": 0.05,
            "hallucination_cost": 0.07,
            "classification_cost": 0.011786,
            "retry_error_cost": 0.0,
            "judge_cost": 0.0,
            "total_openrouter_cost": 0.131786,
            "fresh_cost_this_run": 0.0,
            "stored_benchmark_cost": 0.131786,
            "estimated_cost_without_cache": 0.131786,
            "estimated_cache_savings": 0.131786,
            "reconciliation": {
                "recomputed_from_pricing_table": 0.131,
                "actual_usage_cost": 0.131786,
                "matches_within_tolerance": True,
            },
        },
        "failures": {},
        "notes": [],
        "metrics": {
            "summarization": {
                "per_language": {
                    language: {
                        "cases": 200,
                        "scored_cases": 200,
                        "rougeLsum_f": 0.1624 if language == "en" else 0.2287,
                        "rouge1_f": 0.2,
                        "rouge2_f": 0.05,
                        "chrf": 24.5,
                        "bertscore_f1": None,
                        "surface_fact_support_precision": 0.82,
                        "compression_ratio": 0.09,
                        "output_tokens_mean": 41.0,
                        "empty_output_rate": 0.0,
                        "refusal_rate": 0.0,
                        "failed_cases": 0,
                        "surface_fact_diagnostic": {
                            "facts_total": 900,
                            "facts_supported": 740,
                            "surface_fact_support_precision": 0.822,
                            "number_mismatch": 40,
                            "percentage_mismatch": 10,
                            "date_mismatch": 30,
                            "currency_mismatch": 5,
                            "unsupported_named_entity": 75,
                            "summaries_with_facts": 190,
                        },
                    }
                    for language in ("en", "ko")
                }
            },
            "hallucination": {
                "per_language": {
                    "en": {
                        "status": "OK",
                        "cases": 400,
                        "documents": 200,
                        "accuracy": 0.65,
                        "macro_f1": 0.64,
                        "hallucination_precision": 0.66,
                        "hallucination_recall": 0.63,
                        "hallucination_f1": 0.6476,
                        "supported_recall": 0.67,
                        "invalid_output_rate": 0.0,
                        "failed_cases": 0,
                    },
                    "ko": halu_ko,
                }
            },
            "classification": {
                "per_language": {
                    "en": {"cases": 1000, "accuracy": 0.843, "macro_f1": 0.81, "correct": 843,
                           "invalid_output_rate": 0.0, "label_space_size": 60},
                    "ko": {"cases": 1000, "accuracy": 0.823, "macro_f1": 0.79, "correct": 823,
                           "invalid_output_rate": 0.0, "label_space_size": 60},
                },
                "overall": {"cases": 2000, "accuracy": 0.833, "macro_f1": 0.8, "correct": 1666,
                            "invalid_output_rate": 0.0, "label_space_size": 60},
                "cross_lingual": {
                    "paired_cases": 1000,
                    "consistency": 0.886,
                    "prediction_agreement": 0.85,
                    "both_correct": 800,
                    "both_correct_rate": 0.8,
                    "en_only_correct": 43,
                    "en_only_correct_rate": 0.043,
                    "ko_only_correct": 71,
                    "ko_only_correct_rate": 0.071,
                    "both_wrong": 86,
                    "both_wrong_rate": 0.086,
                },
            },
        },
        "judge": (
            {
                "enabled": True,
                "status": "COMPLETED",
                "profile": "cheap",
                "judge_model": "openai/gpt-5.6-luna",
                "self_judge": False,
                "warnings": [],
                "invalid_outputs": 0,
                "judge_cost_usd": 0.004,
                "scores": {
                    "judged_cases": 200,
                    "factual_consistency": 4.1,
                    "key_information_coverage": 3.8,
                    "conciseness": 4.4,
                    "overall_quality": 4.0,
                },
            }
            if judge
            else {}
        ),
        "reproducibility": {
            "benchmark_version": "1.0.0",
            "executed_at": "2026-01-01T00:00:00+00:00",
            "git_commit": "a" * 40,
            "git_dirty": False,
            "python_version": "3.12.3",
            "routing_config_hash": "b" * 64,
            "alias_resolution": None,
            "manifests": {
                "summarization": {"hash": "c" * 64, "dataset_revision": "rev", "counts": {}, "path": "x"},
                "hallucination:en": {"hash": "d" * 64, "dataset_revision": "rev", "counts": {}, "path": "x"},
                "hallucination:ko": None if not ko_hallucination else
                    {"hash": "e" * 64, "dataset_revision": "local:Validation", "counts": {}, "path": "x"},
                "classification": {"hash": "f" * 64, "dataset_revision": "rev", "counts": {}, "path": "x"},
            },
        },
    }


def _write(project, summaries: list[dict]) -> None:
    for summary in summaries:
        directory = model_result_dir(project, summary["model"]["model_id"])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )


@pytest.fixture
def results(project):
    _write(project, [_summary(), _summary("openai/gpt-5.6-luna", ko_hallucination=False)])
    return project


# --------------------------------------------------------------------------- #
# generate_reports
# --------------------------------------------------------------------------- #
def test_generate_reports_writes_every_artifact(results):
    bundle = generate_reports(results)

    assert set(bundle.paths) >= {"csv", "json", "markdown", "html", "excel", "result_md"}
    for name, path in bundle.paths.items():
        assert path.is_file(), name
        assert path.stat().st_size > 0, name
    assert bundle.paths["result_md"] == results.result_markdown_path
    assert bundle.paths["excel"] == results.result_excel_path
    assert bundle.model_count == 2


def test_generate_reports_is_idempotent(results):
    first = generate_reports(results).paths["result_md"].read_text(encoding="utf-8")
    second = generate_reports(results).paths["result_md"].read_text(encoding="utf-8")
    # Only the generation timestamp may differ.
    strip = lambda text: "\n".join(l for l in text.splitlines() if not l.startswith("생성:"))
    assert strip(first) == strip(second)


# --------------------------------------------------------------------------- #
# RESULT.md
# --------------------------------------------------------------------------- #
def test_result_markdown_contains_every_section(results):
    document = generate_reports(results).paths["result_md"].read_text(encoding="utf-8")

    for heading in (
        "## 리더보드",
        "## 요약 (XL-Sum)",
        "## 환각 탐지",
        "## 분류 (MASSIVE)",
        "## EN/KO 교차 일관성",
        "## Surface fact support (진단)",
        "## 비용 (OpenRouter usage.cost)",
        "## 토큰 사용량 / 캐시",
        "## 재현성",
    ):
        assert heading in document, heading

    assert "google/gemini-2.5-flash-lite" in document
    assert "자동 생성" in document


def test_result_markdown_keeps_the_caveats(results):
    """The caveats are the part that makes the numbers safe to quote."""
    document = generate_reports(results).paths["result_md"].read_text(encoding="utf-8")

    assert "usage.cost" in document
    assert "진단 지표" in document
    assert "종합 점수로 합치지 않습니다" in document


def test_result_markdown_reports_actual_cost(results):
    document = generate_reports(results).paths["result_md"].read_text(encoding="utf-8")
    # The leaderboard cost cell is the stored usage.cost, to 6 decimals.
    assert "$0.131786" in document


def test_result_markdown_marks_missing_values(results):
    """A skipped benchmark shows an em dash, never a misleading 0."""
    document = generate_reports(results).paths["result_md"].read_text(encoding="utf-8")
    luna_row = next(l for l in document.splitlines() if l.startswith("| openai/gpt-5.6-luna |"))
    assert "—" in luna_row
    assert "SKIPPED" in document


def test_result_markdown_without_results(project):
    document = render_result_markdown([], [])
    assert "## 결과 없음" in document
    assert "benchmark run --all" in document


def test_notes_section_lists_aborted_runs(project):
    summary = _summary(status="ABORTED")
    summary["notes"] = ["10 consecutive failures (limit 10); aborting", "[16x] Reasoning is mandatory"]
    _write(project, [summary])

    document = generate_reports(project).paths["result_md"].read_text(encoding="utf-8")
    assert "## 비고" in document
    assert "Reasoning is mandatory" in document
    assert "ABORTED" in document


def test_judge_section_only_appears_when_used(project):
    _write(project, [_summary()])
    assert "LLM-as-a-Judge" not in generate_reports(project).paths["result_md"].read_text(encoding="utf-8")

    _write(project, [_summary(judge=True)])
    document = generate_reports(project).paths["result_md"].read_text(encoding="utf-8")
    assert "LLM-as-a-Judge" in document
    assert "openai/gpt-5.6-luna" in document


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #
def test_workbook_sheets_and_values(results):
    openpyxl = pytest.importorskip("openpyxl")
    path = generate_reports(results).paths["excel"]
    workbook = openpyxl.load_workbook(path)

    assert workbook.sheetnames == [
        "About", "Leaderboard", "Summarization", "Hallucination", "Classification",
        "CrossLingual", "SurfaceFacts", "Cost", "Usage", "Judge", "Runs",
    ]

    leaderboard = workbook["Leaderboard"]
    headers = [cell.value for cell in leaderboard[1]]
    assert "model" in headers and "openrouter_cost" in headers
    cost_column = headers.index("openrouter_cost") + 1
    costs = [leaderboard.cell(row=r, column=cost_column).value for r in range(2, leaderboard.max_row + 1)]
    assert pytest.approx(0.131786) in costs

    # Header row frozen and bold on every sheet.
    for name in workbook.sheetnames:
        sheet = workbook[name]
        assert sheet.freeze_panes == "A2"
        assert sheet[1][0].font.bold


def test_workbook_about_sheet_carries_the_caveats(results):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.load_workbook(generate_reports(results).paths["excel"])
    text = "\n".join(
        str(cell.value) for row in workbook["About"].iter_rows() for cell in row if cell.value
    )
    assert "usage.cost" in text
    assert "diagnostic" in text
    assert "OpenRouter" in text


def test_workbook_handles_all_none_columns(project):
    """Regression: an all-None column is object dtype and broke column sizing."""
    openpyxl = pytest.importorskip("openpyxl")
    summary = _summary()
    for language in ("en", "ko"):
        summary["metrics"]["summarization"]["per_language"][language]["bertscore_f1"] = None
    summary["cost"]["pricing_snapshot"]["cached_input_per_million"] = None
    _write(project, [summary])

    path = generate_reports(project).paths["excel"]
    sheet = openpyxl.load_workbook(path)["Summarization"]
    headers = [cell.value for cell in sheet[1]]
    column = headers.index("bertscore_f1") + 1
    assert sheet.cell(row=2, column=column).value is None


def test_workbook_number_formats(results):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.load_workbook(generate_reports(results).paths["excel"])
    sheet = workbook["Cost"]
    headers = [cell.value for cell in sheet[1]]
    column = headers.index("total_openrouter_cost") + 1
    assert sheet.cell(row=2, column=column).number_format == "$#,##0.000000"


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def test_tables_are_shared_by_both_outputs():
    summaries = [_summary()]
    tables = build_tables(build_leaderboard(summaries), summaries)

    assert {r["language"] for r in tables["Summarization"]} == {"en", "ko"}
    assert {r["scope"] for r in tables["Classification"]} == {"en", "ko", "overall"}
    assert tables["CrossLingual"][0]["paired_cases"] == 1000
    assert tables["Cost"][0]["total_openrouter_cost"] == pytest.approx(0.131786)
    assert tables["Usage"][0]["prompt_tokens"] == 1_097_914
    assert tables["SurfaceFacts"][0]["unsupported_named_entity"] == 75
    assert tables["Runs"][0]["git_commit"] == "a" * 40
    assert tables["Judge"] == []


def test_skipped_language_is_reported_not_dropped():
    summaries = [_summary(ko_hallucination=False)]
    tables = build_tables(build_leaderboard(summaries), summaries)
    ko = next(r for r in tables["Hallucination"] if r["language"] == "ko")
    assert ko["status"] == "SKIPPED"
    assert ko["accuracy"] is None
