"""One entry point that regenerates every result artifact.

`benchmark report`, the end of `benchmark run` and the CI workflow all call this,
so the files can never drift apart depending on which one produced them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmbench.config.loader import AppConfig
from llmbench.logging_setup import get_logger
from llmbench.reporting.excel import write_excel
from llmbench.reporting.html import render_html_report
from llmbench.reporting.leaderboard import (
    baseline_summaries,
    build_leaderboard,
    load_model_summaries,
    write_leaderboard,
)
from llmbench.reporting.result_doc import write_result_markdown

__all__ = ["ReportBundle", "generate_reports"]

log = get_logger("reporting")


@dataclass(slots=True)
class ReportBundle:
    paths: dict[str, Path]
    rows: list[dict[str, Any]] = field(default_factory=list)
    summaries: list[dict[str, Any]] = field(default_factory=list)
    all_summaries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def model_count(self) -> int:
        return len(self.summaries)


def generate_reports(cfg: AppConfig) -> ReportBundle:
    """Rebuild leaderboard.{csv,json,md}, report.html, RESULT.md and the workbook."""
    all_summaries = load_model_summaries(cfg)
    summaries = baseline_summaries(all_summaries)
    rows = build_leaderboard(summaries)

    paths: dict[str, Path] = dict(write_leaderboard(cfg, rows))
    paths["html"] = render_html_report(cfg, rows, summaries)
    paths["excel"] = write_excel(cfg, rows, summaries, all_summaries=all_summaries)
    paths["result_md"] = write_result_markdown(
        cfg, rows, summaries, excel_path=paths["excel"], all_summaries=all_summaries
    )

    log.info(
        "regenerated reports for %d model(s): %s",
        len(summaries),
        ", ".join(sorted(paths)),
    )
    return ReportBundle(paths=paths, rows=rows, summaries=summaries, all_summaries=list(all_summaries))
