"""RESULT.md -- the human-readable result document.

Generated, never hand-edited. It is the artifact people read in a pull request or
paste into a chat, so it leads with the caveats that make the numbers safe to
quote and keeps every metric family in its own table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from llmbench.config.loader import AppConfig
from llmbench.core.reproducibility import utc_now_iso
from llmbench.reporting.tables import build_tables
from llmbench.version import BENCHMARK_VERSION

__all__ = ["write_result_markdown", "render_result_markdown"]

Column = tuple[str, str, str]  # (field, label, format code)


def _fmt(value: Any, code: str) -> str:
    if value is None or value == "":
        return "—"
    if code == "s":
        return str(value).replace("|", "\\|")
    if code == "int":
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return str(value)
    if code == "f4":
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)
    if code == "f2":
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)
    if code == "money":
        try:
            return f"${float(value):,.6f}"
        except (TypeError, ValueError):
            return str(value)
    if code == "price":
        try:
            return f"${float(value):,.4f}"
        except (TypeError, ValueError):
            return str(value)
    if code == "bool":
        return "예" if value else "아니오"
    return str(value)


def _table(rows: Sequence[dict[str, Any]], columns: Sequence[Column]) -> str:
    header = "| " + " | ".join(label for _, label, _ in columns) + " |"
    align = "| " + " | ".join("---" if code == "s" else "---:" for _, _, code in columns) + " |"
    lines = [header, align]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(field), code) for field, _, code in columns) + " |")
    return "\n".join(lines)


def _section(
    title: str,
    rows: Sequence[dict[str, Any]],
    columns: Sequence[Column],
    *,
    note: str | None = None,
    empty: str = "_결과 없음._",
) -> list[str]:
    body = [f"## {title}", ""]
    if note:
        body += [note, ""]
    body += [_table(rows, columns) if rows else empty, ""]
    return body


LEADERBOARD_COLUMNS: list[Column] = [
    ("model", "Model", "s"),
    ("input_per_million", "Input $/M", "price"),
    ("output_per_million", "Output $/M", "price"),
    ("summary_en", "Summary EN", "f4"),
    ("summary_ko", "Summary KO", "f4"),
    ("halu_en_f1", "Halu EN F1", "f4"),
    ("halu_ko_f1", "Halu KO F1", "f4"),
    ("cls_en", "Cls EN", "f4"),
    ("cls_ko", "Cls KO", "f4"),
    ("en_ko_consistency", "EN-KO", "f4"),
    ("input_tokens", "Input Tokens", "int"),
    ("output_tokens", "Output Tokens", "int"),
    ("openrouter_cost", "OpenRouter Cost", "money"),
    ("judge_cost", "Judge Cost", "money"),
]

SUMMARIZATION_COLUMNS: list[Column] = [
    ("model", "Model", "s"),
    ("language", "Lang", "s"),
    ("cases", "Cases", "int"),
    ("rouge_lsum_f", "ROUGE-Lsum", "f4"),
    ("chrf_pp", "chrF++", "f2"),
    ("bertscore_f1", "BERTScore", "f4"),
    ("surface_fact_support", "Surface facts*", "f4"),
    ("compression_ratio", "Compression", "f4"),
    ("output_tokens_mean", "Out tokens", "f2"),
    ("empty_output_rate", "Empty", "f4"),
    ("refusal_rate", "Refusal", "f4"),
]

HALLUCINATION_COLUMNS: list[Column] = [
    ("model", "Model", "s"),
    ("language", "Lang", "s"),
    ("status", "Status", "s"),
    ("cases", "Cases", "int"),
    ("accuracy", "Accuracy", "f4"),
    ("macro_f1", "Macro-F1", "f4"),
    ("hallucination_precision", "Halu P", "f4"),
    ("hallucination_recall", "Halu R", "f4"),
    ("hallucination_f1", "Halu F1", "f4"),
    ("supported_recall", "Supported R", "f4"),
    ("invalid_output_rate", "Invalid", "f4"),
]

CLASSIFICATION_COLUMNS: list[Column] = [
    ("model", "Model", "s"),
    ("scope", "Scope", "s"),
    ("cases", "Cases", "int"),
    ("accuracy", "Accuracy", "f4"),
    ("macro_f1", "Macro-F1", "f4"),
    ("invalid_output_rate", "Invalid", "f4"),
]

CROSS_LINGUAL_COLUMNS: list[Column] = [
    ("model", "Model", "s"),
    ("paired_cases", "Pairs", "int"),
    ("consistency", "Consistency", "f4"),
    ("prediction_agreement", "Pred. agreement", "f4"),
    ("both_correct_rate", "Both correct", "f4"),
    ("en_only_correct_rate", "EN only", "f4"),
    ("ko_only_correct_rate", "KO only", "f4"),
    ("both_wrong_rate", "Both wrong", "f4"),
]

SURFACE_FACT_COLUMNS: list[Column] = [
    ("model", "Model", "s"),
    ("language", "Lang", "s"),
    ("facts_total", "Facts", "int"),
    ("support_precision", "Support", "f4"),
    ("number_mismatch", "Number", "int"),
    ("percentage_mismatch", "Percent", "int"),
    ("date_mismatch", "Date", "int"),
    ("currency_mismatch", "Currency", "int"),
    ("unsupported_named_entity", "Entity", "int"),
]

COST_COLUMNS: list[Column] = [
    ("model", "Model", "s"),
    ("summarization_cost", "Summarization", "money"),
    ("hallucination_cost", "Hallucination", "money"),
    ("classification_cost", "Classification", "money"),
    ("retry_error_cost", "Retry/error", "money"),
    ("judge_cost", "Judge", "money"),
    ("total_openrouter_cost", "Total", "money"),
    ("fresh_cost_this_run", "Fresh this run", "money"),
    ("estimated_cache_savings", "Cache savings", "money"),
]

USAGE_COLUMNS: list[Column] = [
    ("model", "Model", "s"),
    ("successful_cases", "Cases", "int"),
    ("prompt_tokens", "Prompt", "int"),
    ("cached_tokens", "Cached", "int"),
    ("completion_tokens", "Completion", "int"),
    ("reasoning_tokens", "Reasoning", "int"),
    ("total_tokens", "Total", "int"),
    ("openrouter_api_calls", "API calls", "int"),
    ("cache_hits", "Cache hits", "int"),
]

JUDGE_COLUMNS: list[Column] = [
    ("model", "Model", "s"),
    ("judge_model", "Judge", "s"),
    ("self_judge", "Self-judge", "bool"),
    ("judged_cases", "Cases", "int"),
    ("factual_consistency", "Factual", "f4"),
    ("key_information_coverage", "Coverage", "f4"),
    ("conciseness", "Conciseness", "f4"),
    ("overall_quality", "Overall", "f4"),
    ("judge_cost_usd", "Judge cost", "money"),
]

RUN_COLUMNS: list[Column] = [
    ("model", "Model", "s"),
    ("status", "Status", "s"),
    ("executed_at", "실행 시각(UTC)", "s"),
    ("benchmark_version", "Bench ver.", "s"),
    ("git_commit", "Git commit", "s"),
    ("total_cases", "Cases", "int"),
    ("xlsum_manifest", "XL-Sum", "s"),
    ("halueval_manifest", "HaluEval", "s"),
    ("aihub_manifest", "AI-Hub", "s"),
    ("massive_manifest", "MASSIVE", "s"),
]


def _short_hashes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        copy = dict(row)
        for key in ("git_commit", "xlsum_manifest", "halueval_manifest", "aihub_manifest", "massive_manifest"):
            value = copy.get(key)
            if isinstance(value, str) and len(value) > 12:
                copy[key] = value[:12]
        out.append(copy)
    return out


def render_result_markdown(
    leaderboard: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    *,
    excel_path: Path | None = None,
    root: Path | None = None,
) -> str:
    tables = build_tables(leaderboard, summaries)
    generated = utc_now_iso()

    def rel(path: Path | None) -> str:
        if path is None:
            return "results/benchmark_results.xlsx"
        if root is not None:
            try:
                return str(path.relative_to(root))
            except ValueError:
                pass
        return str(path)

    lines: list[str] = [
        "# 벤치마크 결과",
        "",
        "<!-- 이 파일은 `benchmark report` 가 자동 생성합니다. 직접 수정하지 마세요. -->",
        "",
        f"생성: `{generated}` · benchmark version `{BENCHMARK_VERSION}` · API provider: **OpenRouter**",
        "",
        f"같은 데이터를 담은 스프레드시트: [`{rel(excel_path)}`]({rel(excel_path)})",
        "",
    ]

    if not summaries:
        lines += [
            "## 결과 없음",
            "",
            "아직 실행된 모델이 없습니다. `benchmark run --all` 을 실행한 뒤 `benchmark report` 를 다시 실행하세요.",
            "",
        ]
        return "\n".join(lines)

    lines += [
        "## 숫자를 읽기 전에",
        "",
        "- 비용은 전부 OpenRouter가 반환한 **실제 청구액 `usage.cost`** 입니다. 가격표 추정치가 아닙니다.",
        "- 성격이 다른 지표를 하나의 종합 점수로 합치지 않습니다. 열끼리 따로 보세요.",
        "- **Surface fact support 는 진단 지표** 이지 환각 지표가 아닙니다. 표면적 사실만 대조합니다.",
        "- EN/KO 환각 점수는 출처가 달라(HaluEval vs AI-Hub) 서로 직접 비교 대상이 아닙니다.",
        "- 한국어 ROUGE 는 문자 단위 토큰화라 영어 ROUGE 와 직접 비교할 수 없습니다.",
        "- 빈 칸(`—`)은 값이 0이라는 뜻이 아니라 **계산되지 않았다**는 뜻입니다.",
        "",
        *_section("리더보드", tables["Leaderboard"], LEADERBOARD_COLUMNS),
        *_section(
            "요약 (XL-Sum)",
            tables["Summarization"],
            SUMMARIZATION_COLUMNS,
            note="`*` Surface facts 는 진단 지표입니다.",
        ),
        *_section(
            "환각 탐지",
            tables["Hallucination"],
            HALLUCINATION_COLUMNS,
            note="한국어는 AI-Hub 데이터가 준비되지 않으면 `SKIPPED` 로 표시됩니다.",
        ),
        *_section("분류 (MASSIVE)", tables["Classification"], CLASSIFICATION_COLUMNS),
        *_section(
            "EN/KO 교차 일관성",
            tables["CrossLingual"],
            CROSS_LINGUAL_COLUMNS,
            note="같은 semantic id 를 공유하는 EN/KO 쌍 기준입니다.",
        ),
        *_section(
            "Surface fact support (진단)",
            tables["SurfaceFacts"],
            SURFACE_FACT_COLUMNS,
            note="숫자·백분율·날짜·통화·개체명을 원문과 대조한 결과입니다. 완전한 환각 지표가 아닙니다.",
        ),
        *_section(
            "비용 (OpenRouter usage.cost)",
            tables["Cost"],
            COST_COLUMNS,
            note="`Fresh this run` 은 마지막 실행에서 새로 청구된 금액, `Total` 은 이 모델 캐시를 만드는 데 지금까지 지불한 총액입니다.",
        ),
        *_section("토큰 사용량 / 캐시", tables["Usage"], USAGE_COLUMNS),
    ]

    if tables["Judge"]:
        lines += _section(
            "LLM-as-a-Judge (참고용)",
            tables["Judge"],
            JUDGE_COLUMNS,
            note="기본 지표가 아닙니다. self-judge 인 경우 신뢰하지 마세요.",
        )

    lines += _section(
        "재현성",
        _short_hashes(tables["Runs"]),
        RUN_COLUMNS,
        note="manifest 해시가 같으면 모든 모델이 완전히 동일한 케이스를 받았다는 뜻입니다.",
    )

    notes = [row for row in tables["Runs"] if row.get("notes") or row.get("skipped")]
    if notes:
        lines += ["## 비고", ""]
        for row in notes:
            details = [d for d in (row.get("skipped"), row.get("notes")) if d]
            lines.append(f"- **{row['model']}**: {' / '.join(details)}")
        lines.append("")

    return "\n".join(lines)


def write_result_markdown(
    cfg: AppConfig,
    leaderboard: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    *,
    path: Path | None = None,
    excel_path: Path | None = None,
) -> Path:
    target = path or cfg.result_markdown_path
    target.parent.mkdir(parents=True, exist_ok=True)
    document = render_result_markdown(
        leaderboard,
        summaries,
        excel_path=excel_path or cfg.result_excel_path,
        root=cfg.root,
    )
    target.write_text(document + "\n", encoding="utf-8")
    return target
