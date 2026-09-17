"""Static HTML report (results/report.html)."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from llmbench.config.loader import AppConfig
from llmbench.core.reproducibility import utc_now_iso
from llmbench.reporting.leaderboard import LEADERBOARD_COLUMNS, model_detail_rows
from llmbench.reporting.tables import results_timestamp

__all__ = ["render_html_report"]

_STYLE = """
:root {
  --bg: #ffffff; --fg: #16181d; --muted: #5b6270; --line: #e3e6ec;
  --accent: #2f5fd0; --chip: #f2f4f8; --warn: #a2620a;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #14161a; --fg: #e8eaee; --muted: #9aa3b2; --line: #2a2e36;
          --accent: #7aa2f7; --chip: #1e222a; --warn: #e0a458; }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 32px 16px 64px; background: var(--bg); color: var(--fg);
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans KR", sans-serif; }
main { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 19px; margin: 40px 0 10px; }
h3 { font-size: 15px; margin: 22px 0 6px; }
p.sub { color: var(--muted); margin: 0 0 20px; }
.note { background: var(--chip); border-left: 3px solid var(--accent); padding: 10px 14px;
  border-radius: 6px; margin: 16px 0; color: var(--muted); font-size: 13.5px; }
.warn { border-left-color: var(--warn); }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; margin: 8px 0 4px; }
th, td { border-bottom: 1px solid var(--line); padding: 7px 9px; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { color: var(--muted); font-weight: 600; border-bottom: 2px solid var(--line); }
tbody tr:hover { background: var(--chip); }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
.chip { display: inline-block; background: var(--chip); border-radius: 999px; padding: 2px 10px;
  font-size: 12px; color: var(--muted); margin-right: 6px; }
.kv { display: grid; grid-template-columns: minmax(180px, 280px) 1fr; gap: 2px 16px; font-size: 13.5px; }
.kv div:nth-child(odd) { color: var(--muted); }
details { border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px; margin: 10px 0; }
summary { cursor: pointer; font-weight: 600; }
footer { margin-top: 48px; color: var(--muted); font-size: 12.5px; }
@media (max-width: 640px) { body { padding: 20px 16px 48px; } .kv { grid-template-columns: 1fr; } }
"""


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _kv(pairs: list[tuple[str, Any]], digits: int = 6) -> str:
    cells = "".join(
        f"<div>{html.escape(str(k))}</div><div>{_fmt(v, digits)}</div>" for k, v in pairs
    )
    return f'<div class="kv">{cells}</div>'


def render_html_report(cfg: AppConfig, rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> Path:
    leaderboard_rows = [
        [
            html.escape(str(row["model"])),
            html.escape(str(row.get("api_provider", "openrouter"))),
            _fmt(row["input_per_million"]),
            _fmt(row["output_per_million"]),
            _fmt(row["summary_en"]),
            _fmt(row["summary_ko"]),
            _fmt(row["halu_en_f1"]),
            _fmt(row["halu_ko_f1"]),
            _fmt(row["cls_en"]),
            _fmt(row["cls_ko"]),
            _fmt(row["en_ko_consistency"]),
            _fmt(int(row["input_tokens"] or 0)),
            _fmt(int(row["output_tokens"] or 0)),
            _fmt(row["openrouter_cost"], 6),
            _fmt(row["judge_cost"], 6),
        ]
        for row in rows
    ]

    sections = []
    for summary in summaries:
        detail = model_detail_rows(summary)
        model_id = detail["model"]
        cost = detail.get("cost", {}) or {}
        usage = detail.get("usage", {}) or {}
        pricing = cost.get("pricing_snapshot", {}) or {}
        judge = detail.get("judge", {}) or {}

        blocks = [f"<h3>{html.escape(model_id)}</h3>"]
        blocks.append(
            '<p><span class="chip">status: '
            + html.escape(str(summary.get("status", "")))
            + '</span><span class="chip">run: '
            + html.escape(str(summary.get("run_id", "")))
            + '</span><span class="chip">api: openrouter</span></p>'
        )

        blocks.append("<h4>OpenRouter Pricing Snapshot</h4>")
        blocks.append(
            _kv(
                [
                    ("Input $/M", pricing.get("input_per_million")),
                    ("Cached Input $/M", pricing.get("cached_input_per_million")),
                    ("Output $/M", pricing.get("output_per_million")),
                    ("Pricing retrieved at", pricing.get("retrieved_at")),
                    ("Pricing source", pricing.get("source")),
                ]
            )
        )

        blocks.append("<h4>Actual Usage</h4>")
        blocks.append(
            _kv(
                [
                    ("Prompt Tokens", usage.get("prompt_tokens")),
                    ("Cached Tokens", usage.get("cached_tokens")),
                    ("Cache Write Tokens", usage.get("cache_write_tokens")),
                    ("Completion Tokens", usage.get("completion_tokens")),
                    ("Reasoning Tokens", usage.get("reasoning_tokens")),
                    ("Total Tokens", usage.get("total_tokens")),
                ]
            )
        )

        blocks.append("<h4>Actual Cost (OpenRouter usage.cost)</h4>")
        blocks.append(
            _kv(
                [
                    ("Summarization cost", cost.get("summarization_cost")),
                    ("Hallucination cost", cost.get("hallucination_cost")),
                    ("Classification cost", cost.get("classification_cost")),
                    ("Retry/error cost", cost.get("retry_error_cost")),
                    ("Judge cost", cost.get("judge_cost")),
                    ("Total OpenRouter cost", cost.get("total_openrouter_cost")),
                    ("Fresh cost this run", cost.get("fresh_cost_this_run")),
                    ("Stored benchmark cost", cost.get("stored_benchmark_cost")),
                    ("Estimated cost without cache", cost.get("estimated_cost_without_cache")),
                    ("Estimated cache savings", cost.get("estimated_cache_savings")),
                ]
            )
        )

        for title, block in (
            ("Summarization", detail.get("summarization") or {}),
            ("Hallucination", detail.get("hallucination") or {}),
            ("Classification", detail.get("classification") or {}),
        ):
            if not block:
                continue
            languages = sorted(block)
            metric_names: list[str] = []
            for language in languages:
                for name in block[language]:
                    if name not in metric_names:
                        metric_names.append(name)
            table_rows = [
                [html.escape(name)] + [_fmt(block[language].get(name)) for language in languages]
                for name in metric_names
            ]
            blocks.append(f"<h4>{title}</h4>")
            blocks.append(_table(["Metric"] + [lang.upper() for lang in languages], table_rows))

        cross = detail.get("cross_lingual") or {}
        if cross:
            blocks.append("<h4>Cross-lingual (EN/KO paired)</h4>")
            blocks.append(
                _kv(
                    [
                        ("Paired cases", cross.get("paired_cases")),
                        ("Consistency", cross.get("consistency")),
                        ("Prediction agreement", cross.get("prediction_agreement")),
                        ("Both correct", cross.get("both_correct_rate")),
                        ("EN only correct", cross.get("en_only_correct_rate")),
                        ("KO only correct", cross.get("ko_only_correct_rate")),
                        ("Both wrong", cross.get("both_wrong_rate")),
                    ],
                    digits=4,
                )
            )

        if judge and judge.get("enabled"):
            warn_class = " warn" if judge.get("self_judge") else ""
            warnings = "<br>".join(html.escape(w) for w in judge.get("warnings", []))
            blocks.append("<h4>LLM-as-a-Judge (advisory)</h4>")
            if warnings:
                blocks.append(f'<div class="note{warn_class}">{warnings}</div>')
            scores = judge.get("scores", {}) or {}
            blocks.append(
                _kv(
                    [
                        ("Judge model", judge.get("judge_model")),
                        ("Judged cases", scores.get("judged_cases")),
                        ("Factual consistency", scores.get("factual_consistency")),
                        ("Key information coverage", scores.get("key_information_coverage")),
                        ("Conciseness", scores.get("conciseness")),
                        ("Overall quality", scores.get("overall_quality")),
                        ("Judge cost (usage.cost)", judge.get("judge_cost_usd")),
                    ],
                    digits=4,
                )
            )

        repro = summary.get("reproducibility", {}) or {}
        blocks.append(
            "<details><summary>Reproducibility</summary><pre>"
            + html.escape(json.dumps(repro, ensure_ascii=False, indent=2))
            + "</pre></details>"
        )
        sections.append("<section>" + "".join(blocks) + "</section>")

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Benchmark Report</title>
<style>{_STYLE}</style>
</head>
<body>
<main>
<h1>LLM Summarization / Hallucination / Classification Benchmark</h1>
<p class="sub">Results as of {html.escape(results_timestamp(summaries))} &middot; every model called through OpenRouter</p>

<div class="note">
Costs shown are the authoritative <code>usage.cost</code> values returned by OpenRouter.
Pricing-table figures are estimates used for budgeting only. Surface fact support is a
<strong>diagnostic</strong>, not a complete hallucination metric.
</div>

<h2>Leaderboard</h2>
{_table([label for _, label in LEADERBOARD_COLUMNS], leaderboard_rows) if leaderboard_rows
  else "<p class='sub'>No model results yet. Run <code>benchmark run --all</code>.</p>"}

<h2>Model detail</h2>
{"".join(sections) if sections else "<p class='sub'>Nothing to show yet.</p>"}

<footer>
Different metric families are reported separately by design; this report never combines
them into a single composite score.
</footer>
</main>
</body>
</html>
"""
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.results_dir / "report.html"
    path.write_text(document, encoding="utf-8")
    return path
