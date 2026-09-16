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


# (지표, 방향, 설명). 방향: "up" 높을수록 좋음, "down" 낮을수록 좋음, "info" 해석용.
Legend = tuple[str, str, str]

_ARROW = {"up": "↑", "down": "↓", "info": "—"}
LEGEND_KEY = "↑ 높을수록 좋음 · ↓ 낮을수록 좋음 · — 좋고 나쁨이 아니라 해석용"


def _legend_table(legend: Sequence[Legend]) -> str:
    lines = ["| 지표 | 방향 | 의미 |", "| --- | :--: | --- |"]
    for name, direction, meaning in legend:
        lines.append(f"| {name} | {_ARROW[direction]} | {meaning} |")
    return "\n".join(lines)


def _section(
    title: str,
    rows: Sequence[dict[str, Any]],
    columns: Sequence[Column],
    *,
    about: str | None = None,
    legend: Sequence[Legend] | None = None,
    note: str | None = None,
    empty: str = "_결과 없음._",
) -> list[str]:
    body = [f"## {title}", ""]
    if about:
        body += [about, ""]
    body += [_table(rows, columns) if rows else empty, ""]
    if legend:
        body += ["<details><summary>지표 설명</summary>", "", _legend_table(legend), "", "</details>", ""]
    if note:
        body += [note, ""]
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


# --------------------------------------------------------------------------- #
# 벤치마크 설명과 지표 범례
# --------------------------------------------------------------------------- #
BENCHMARK_OVERVIEW = """세 가지 능력을 영어와 한국어에서 각각 측정합니다. 모델 하나당 3,200 케이스,
OpenRouter 요청 1,300건입니다.

| 벤치마크 | 데이터셋 | 과제 | 케이스 |
|---|---|---|---:|
| 요약 | XL-Sum (BBC 뉴스) | 기사를 1~3문장으로 요약 | EN 200 + KO 200 |
| 환각 탐지 | HaluEval (EN) / AI-Hub 추상요약 사실성 검증 (KO) | 요약이 원문에 의해 뒷받침되는지 `SUPPORTED`/`HALLUCINATED` 판정 | EN 400 + KO 400 |
| 분류 | MASSIVE | 발화를 60개 인텐트 중 하나로 분류 (20건씩 묶어 요청) | EN 1,000 + KO 1,000 |

모든 모델이 동일한 frozen manifest(같은 샘플, 같은 전처리, 같은 프롬프트)를 받습니다."""

SUMMARIZATION_ABOUT = """XL-Sum(BBC 뉴스)의 기사를 1~3문장으로 요약시키고, 기사에 딸린 사람이 쓴
참조 요약과 비교합니다. 원문은 약 1,800토큰으로 잘라 모든 모델에 동일하게 제공합니다."""

SUMMARIZATION_LEGEND: list[Legend] = [
    ("Cases", "info", "채점 대상 케이스 수."),
    ("ROUGE-Lsum", "up", "참조 요약과의 최장 공통 부분열 기반 F1 (0~1). 문장 단위로 계산. 한국어는 문자 단위 토큰화라 영어 값과 직접 비교 불가."),
    ("chrF++", "up", "문자 n-gram + 단어 bigram F-score (0~100). 어형 변화에 강해 한국어에 비교적 공정."),
    ("BERTScore", "up", "다국어 임베딩 기반 의미 유사도 F1 (0~1). 표현이 달라도 뜻이 같으면 점수를 줍니다. 선택 설치."),
    ("Surface facts", "up", "요약에서 뽑은 표면적 사실 중 원문에서 확인되는 비율 (0~1). 진단 지표."),
    ("Compression", "info", "출력 길이 ÷ 원문 길이. 낮으면 압축적, 높으면 장황. 낮다고 무조건 좋은 것은 아닙니다."),
    ("Out tokens", "info", "평균 출력 길이(추정 토큰)."),
    ("Empty", "down", "빈 출력 비율."),
    ("Refusal", "down", "거부 응답 비율. 빈 출력과 거부는 분모에서 빼지 않습니다."),
]

HALLUCINATION_ABOUT = """원문과 후보 요약을 함께 주고 요약의 모든 사실적 주장이 원문에 의해
뒷받침되는지 판단시킵니다. 문서 하나당 두 케이스(충실한 요약 = `SUPPORTED`, 오류가 있는 요약 =
`HALLUCINATED`)라 레이블이 정확히 반반입니다. 따라서 **무작위 추측의 정확도는 0.5** 입니다."""

HALLUCINATION_LEGEND: list[Legend] = [
    ("Status", "info", "`OK` 또는 `SKIPPED`(해당 언어 데이터 미준비)."),
    ("Cases", "info", "케이스 수. 문서 수 × 2."),
    ("Accuracy", "up", "전체 정답률 (0~1). 레이블이 반반이므로 0.5가 무작위 수준."),
    ("Macro-F1", "up", "두 레이블 F1의 단순 평균 (0~1). 한쪽 레이블만 찍는 모델을 걸러냅니다."),
    ("Halu P", "up", "HALLUCINATED 라고 답한 것 중 실제로 맞은 비율. 낮으면 멀쩡한 요약을 환각이라고 의심."),
    ("Halu R", "up", "실제 HALLUCINATED 중 잡아낸 비율. 낮으면 환각을 놓침."),
    ("Halu F1", "up", "Halu P 와 Halu R 의 조화평균. 환각 탐지 성능의 대표값."),
    ("Supported R", "up", "실제 SUPPORTED 중 맞힌 비율. 이 값만 낮으면 과도하게 의심하는 모델."),
    ("Invalid", "down", "레이블로 파싱할 수 없는 출력 비율. **오답으로 계산됩니다.**"),
]

CLASSIFICATION_ABOUT = """MASSIVE 음성비서 발화를 60개 인텐트 중 하나로 분류시킵니다. 비용 절감을
위해 20건씩 묶어 한 요청으로 보내고, `<번호>: <인텐트 번호>` 형식으로 답하게 합니다.
**무작위 추측의 정확도는 1/60 ≈ 0.017** 입니다."""

CLASSIFICATION_LEGEND: list[Legend] = [
    ("Scope", "info", "`en` / `ko` / `overall`(두 언어 합산)."),
    ("Cases", "info", "분류한 발화 수."),
    ("Accuracy", "up", "정답률 (0~1)."),
    ("Macro-F1", "up", "60개 인텐트별 F1의 평균 (0~1). 드문 인텐트를 무시하는 모델에 불리하게 작동합니다."),
    ("Invalid", "down", "형식을 못 지켜 답을 못 읽은 비율. 오답으로 계산됩니다. 배치 형식을 못 지키면 여기가 올라갑니다."),
]

CROSS_LINGUAL_ABOUT = """분류 벤치마크의 EN/KO 쌍은 **같은 semantic id** 를 공유합니다. 즉 뜻이 같은
발화를 두 언어로 물어본 것이라, 두 언어 사이의 성능 편차를 직접 볼 수 있습니다."""

CROSS_LINGUAL_LEGEND: list[Legend] = [
    ("Pairs", "info", "비교한 EN/KO 쌍의 수."),
    ("Consistency", "up", "두 언어에서 정오답이 일치한 비율(둘 다 맞음 + 둘 다 틀림). 높을수록 언어에 따라 흔들리지 않음."),
    ("Pred. agreement", "up", "정답 여부와 무관하게 **같은 인텐트** 를 예측한 비율. 정확도와 분리된 안정성 지표."),
    ("Both correct", "up", "두 언어 모두 맞힌 비율."),
    ("EN only", "down", "영어만 맞힌 비율. 높으면 한국어가 약함."),
    ("KO only", "down", "한국어만 맞힌 비율. 높으면 영어가 약함."),
    ("Both wrong", "down", "두 언어 모두 틀린 비율."),
]

SURFACE_FACT_ABOUT = """생성된 요약에서 숫자·백분율·날짜·통화·개체명을 규칙 기반으로 추출해 원문과
대조합니다. **LLM 을 쓰지 않는 결정적 진단** 이며, 완전한 환각 지표가 아닙니다. 의미 수준의
왜곡("부인했다" vs "확인했다")이나 누락은 잡지 못하고, 개체명은 정밀도 위주로 적게 뽑습니다."""

SURFACE_FACT_LEGEND: list[Legend] = [
    ("Facts", "info", "요약에서 추출된 사실의 총 개수. 적으면 아래 비율의 신뢰도가 낮습니다."),
    ("Support", "up", "그중 원문에서 확인된 비율 (0~1)."),
    ("Number", "down", "원문에 없는 숫자 건수."),
    ("Percent", "down", "원문과 다른 백분율 건수. (원문 12% → 요약 21% 같은 경우)"),
    ("Date", "down", "원문에 없는 날짜 건수."),
    ("Currency", "down", "원문에 없는 금액 건수."),
    ("Entity", "down", "원문에 없는 고유명사 건수."),
]

COST_ABOUT = """모두 OpenRouter 가 응답에 실어 보낸 **실제 청구액 `usage.cost`** 입니다. 가격표로
다시 계산한 값이 아닙니다."""

COST_LEGEND: list[Legend] = [
    ("Summarization / Hallucination / Classification", "down", "벤치마크별 누적 청구액."),
    ("Retry/error", "down", "실패했지만 과금된 시도의 합. 0이 아니면 재시도가 돈을 쓴 것."),
    ("Judge", "down", "LLM-as-a-Judge 비용. 기본 예산과 별도로 집계합니다."),
    ("Total", "down", "이 모델 캐시를 만드는 데 지금까지 지불한 총액."),
    ("Fresh this run", "info", "마지막 실행에서 **새로** 청구된 금액. 캐시가 다 맞으면 $0."),
    ("Cache savings", "up", "캐시 덕분에 다시 내지 않은 금액."),
]

USAGE_ABOUT = """토큰 수는 OpenRouter 가 보고한 값입니다. 같은 케이스를 푸는 데 토큰을 적게 쓰면
싸지만, 출력 토큰이 지나치게 적으면 요약이 잘렸다는 뜻일 수도 있습니다."""

USAGE_LEGEND: list[Legend] = [
    ("Cases", "info", "성공적으로 저장된 케이스 수."),
    ("Prompt / Completion / Total", "info", "입력·출력·합계 토큰."),
    ("Cached", "up", "프롬프트 캐시로 할인된 입력 토큰."),
    ("Reasoning", "info", "thinking 토큰. thinking 을 끌 수 없는 모델만 0이 아닙니다. 과금 대상입니다."),
    ("API calls", "down", "실제로 OpenRouter 로 나간 요청 수. 재실행에서 0이면 캐시가 완전히 맞은 것."),
    ("Cache hits", "up", "캐시로 처리해 요청하지 않은 수."),
]

JUDGE_LEGEND: list[Legend] = [
    ("Self-judge", "down", "`예` 면 모델이 자기 출력을 채점한 것. 신뢰하지 마세요."),
    ("Factual / Coverage / Conciseness / Overall", "up", "각 1~5 정수 채점의 평균."),
    ("Judge cost", "down", "judge 호출의 실제 청구액."),
]

RUN_ABOUT = """결과를 재현하는 데 필요한 정보입니다. **manifest 해시가 모든 모델에서 같으면**
완전히 동일한 케이스로 비교했다는 뜻입니다."""


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
        "## 벤치마크 구성",
        "",
        BENCHMARK_OVERVIEW,
        "",
        "## 숫자를 읽기 전에",
        "",
        f"각 표 아래 `지표 설명` 을 펼치면 지표의 의미와 방향이 나옵니다 ({LEGEND_KEY}).",
        "",
        "- 비용은 전부 OpenRouter가 반환한 **실제 청구액 `usage.cost`** 입니다. 가격표 추정치가 아닙니다.",
        "- 성격이 다른 지표를 하나의 종합 점수로 합치지 않습니다. 열끼리 따로 보세요.",
        "- **Surface fact support 는 진단 지표** 이지 환각 지표가 아닙니다. 표면적 사실만 대조합니다.",
        "- EN/KO 환각 점수는 출처가 달라(HaluEval vs AI-Hub) 서로 직접 비교 대상이 아닙니다.",
        "- 한국어 ROUGE 는 문자 단위 토큰화라 영어 ROUGE 와 직접 비교할 수 없습니다.",
        "- 빈 칸(`—`)은 값이 0이라는 뜻이 아니라 **계산되지 않았다**는 뜻입니다.",
        "",
        *_section(
            "리더보드",
            tables["Leaderboard"],
            LEADERBOARD_COLUMNS,
            about=(
                "각 벤치마크의 대표 지표만 모은 표입니다. `Summary` 는 ROUGE-Lsum, `Halu` 는 "
                "HALLUCINATED F1, `Cls` 는 정확도, `EN-KO` 는 교차 일관성입니다. 자세한 값은 "
                "아래 벤치마크별 표를 보세요. 정렬 기준은 분류 전체 정확도입니다."
            ),
            legend=[
                ("Input $/M, Output $/M", "down", "100만 토큰당 가격(USD). 실행 시점 OpenRouter 가격 스냅샷."),
                ("Summary EN / KO", "up", "요약 ROUGE-Lsum F1 (0~1)."),
                ("Halu EN / KO F1", "up", "환각 탐지에서 HALLUCINATED 를 양성으로 본 F1 (0~1)."),
                ("Cls EN / KO", "up", "분류 정확도 (0~1). 무작위는 약 0.017."),
                ("EN-KO", "up", "두 언어에서 정오답이 일치한 비율 (0~1)."),
                ("Input / Output Tokens", "info", "누적 토큰 수."),
                ("OpenRouter Cost", "down", "누적 실제 청구액."),
                ("Judge Cost", "down", "judge 사용 시의 청구액. 기본 예산과 별도."),
            ],
        ),
        *_section(
            "요약 (XL-Sum)",
            tables["Summarization"],
            SUMMARIZATION_COLUMNS,
            about=SUMMARIZATION_ABOUT,
            legend=SUMMARIZATION_LEGEND,
            note="`*` Surface facts 는 진단 지표입니다.",
        ),
        *_section(
            "환각 탐지",
            tables["Hallucination"],
            HALLUCINATION_COLUMNS,
            about=HALLUCINATION_ABOUT,
            legend=HALLUCINATION_LEGEND,
            note="한국어는 AI-Hub 데이터가 준비되지 않으면 `SKIPPED` 로 표시됩니다.",
        ),
        *_section(
            "분류 (MASSIVE)",
            tables["Classification"],
            CLASSIFICATION_COLUMNS,
            about=CLASSIFICATION_ABOUT,
            legend=CLASSIFICATION_LEGEND,
        ),
        *_section(
            "EN/KO 교차 일관성",
            tables["CrossLingual"],
            CROSS_LINGUAL_COLUMNS,
            about=CROSS_LINGUAL_ABOUT,
            legend=CROSS_LINGUAL_LEGEND,
        ),
        *_section(
            "Surface fact support (진단)",
            tables["SurfaceFacts"],
            SURFACE_FACT_COLUMNS,
            about=SURFACE_FACT_ABOUT,
            legend=SURFACE_FACT_LEGEND,
        ),
        *_section(
            "비용 (OpenRouter usage.cost)",
            tables["Cost"],
            COST_COLUMNS,
            about=COST_ABOUT,
            legend=COST_LEGEND,
        ),
        *_section(
            "토큰 사용량 / 캐시",
            tables["Usage"],
            USAGE_COLUMNS,
            about=USAGE_ABOUT,
            legend=USAGE_LEGEND,
        ),
    ]

    if tables["Judge"]:
        lines += _section(
            "LLM-as-a-Judge (참고용)",
            tables["Judge"],
            JUDGE_COLUMNS,
            about=(
                "요약 200건(EN 100 / KO 100)을 다른 모델이 1~5점으로 채점한 결과입니다. "
                "**기본 지표가 아닙니다.** 위의 결정적 지표를 보조하는 참고용입니다."
            ),
            legend=JUDGE_LEGEND,
        )

    lines += _section(
        "재현성",
        _short_hashes(tables["Runs"]),
        RUN_COLUMNS,
        about=RUN_ABOUT,
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
