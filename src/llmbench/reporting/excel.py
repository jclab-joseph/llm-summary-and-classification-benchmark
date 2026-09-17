"""Excel workbook export (results/benchmark_results.xlsx).

One sheet per view, plus an `About` sheet that carries the caveats. A
spreadsheet gets copied into slide decks, so the caveats travel with it: costs
are OpenRouter's billed `usage.cost`, the surface-fact column is a diagnostic,
and nothing here is a composite score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from llmbench.config.loader import AppConfig
from llmbench.core.reproducibility import utc_now_iso
from llmbench.reporting.result_doc import (
    CLASSIFICATION_LEGEND,
    CLASSIFICATION_MODES_LEGEND,
    COST_LEGEND,
    CROSS_LINGUAL_LEGEND,
    HALLUCINATION_LEGEND,
    JUDGE_LEGEND,
    LEGEND_KEY,
    SUMMARIZATION_LEGEND,
    SURFACE_FACT_LEGEND,
    USAGE_LEGEND,
    _ARROW,
)
from llmbench.reporting.tables import TABLES, build_tables, results_timestamp
from llmbench.version import BENCHMARK_VERSION

__all__ = ["write_excel"]

# Column-name suffix -> Excel number format.
_RATE_COLUMNS = (
    "accuracy", "macro_f1", "_f1", "_f", "recall", "precision", "rate", "consistency",
    "agreement", "ratio", "support", "chrf_pp", "bertscore_f1", "conciseness",
    "coverage", "quality",
)
_COST_COLUMNS = ("cost", "per_million", "savings")
_INT_COLUMNS = (
    "cases", "tokens", "count", "correct", "documents", "mismatch", "entity",
    "calls", "attempts", "hits", "misses", "total", "size", "facts",
)

def _about_rows(summaries: Sequence[dict[str, Any]]) -> list[tuple[str, str]]:
    return [
        ("Results as of (UTC)", results_timestamp(summaries)),
        *_ABOUT,
    ]


_ABOUT = [
    ("Benchmark", "LLM summarization / hallucination / classification benchmark"),
    ("Benchmark version", BENCHMARK_VERSION),
    ("API provider", "OpenRouter (every model, including judges)"),
    ("", ""),
    ("Cost", "All cost columns are OpenRouter's billed usage.cost, not pricing-table estimates."),
    ("Pricing columns", "Estimates only, used for budgeting and for validating what was billed."),
    ("Surface fact support", "A deterministic diagnostic, NOT a complete hallucination metric."),
    ("Composite scores", "Deliberately absent: metric families are not averaged together."),
    ("EN vs KO hallucination", "Different sources (HaluEval vs AI-Hub); compare within a language."),
    ("Korean ROUGE", "Character-level tokenization; not comparable with word-level English ROUGE."),
    ("Empty metric cells", "Mean the metric was not computed (e.g. BERTScore extra not installed)."),
    ("", ""),
    ("Metric directions", LEGEND_KEY),
    ("Legend sheet", "Explains every metric column: what it means and which way is better."),
]


# Sheet -> its metric legend, so the workbook explains its own columns exactly
# the way RESULT.md does (both read the same definitions).
_SHEET_LEGENDS = (
    ("Summarization", SUMMARIZATION_LEGEND),
    ("Hallucination", HALLUCINATION_LEGEND),
    ("Classification", CLASSIFICATION_LEGEND),
    ("ClassificationModes", CLASSIFICATION_MODES_LEGEND),
    ("CrossLingual", CROSS_LINGUAL_LEGEND),
    ("SurfaceFacts", SURFACE_FACT_LEGEND),
    ("Cost", COST_LEGEND),
    ("Usage", USAGE_LEGEND),
    ("Judge", JUDGE_LEGEND),
)


def _legend_rows() -> list[tuple[str, str, str, str]]:
    rows = []
    for sheet, legend in _SHEET_LEGENDS:
        for name, direction, meaning in legend:
            rows.append((sheet, name, _ARROW[direction], meaning))
    return rows


def _number_format(column: str) -> str | None:
    name = column.lower()
    if any(token in name for token in _COST_COLUMNS):
        return "$#,##0.000000"
    if any(name.endswith(token) or token in name for token in _RATE_COLUMNS):
        return "0.0000"
    if any(token in name for token in _INT_COLUMNS):
        return "#,##0"
    return None


def _cell_text(value: Any) -> str:
    """Render a cell for width measurement.

    An all-None column lands as object dtype holding None, and `astype(str)`
    leaves those as float NaN rather than a string, so this cannot go through
    pandas' own conversion.
    """
    import pandas as pd

    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _autosize(worksheet, frame) -> None:
    from openpyxl.utils import get_column_letter

    for index, column in enumerate(frame.columns, start=1):
        widths = [len(str(column))]
        widths += [len(_cell_text(value)) for value in frame[column].tolist()[:200]]
        worksheet.column_dimensions[get_column_letter(index)].width = min(max(max(widths) + 2, 10), 46)


def write_excel(
    cfg: AppConfig,
    leaderboard: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    *,
    path: Path | None = None,
    all_summaries: Sequence[dict[str, Any]] | None = None,
) -> Path:
    """Write every result view to one workbook. Returns the path."""
    import pandas as pd
    from openpyxl.styles import Alignment, Font

    target = path or cfg.result_excel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    tables = build_tables(leaderboard, summaries, all_summaries=all_summaries)

    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        about = pd.DataFrame(_about_rows(summaries), columns=["Field", "Value"])
        about.to_excel(writer, sheet_name="About", index=False)

        legend = pd.DataFrame(_legend_rows(), columns=["Sheet", "지표", "방향", "의미"])
        legend.to_excel(writer, sheet_name="Legend", index=False)

        written: list[tuple[str, Any]] = [("About", about), ("Legend", legend)]
        for sheet, rows in tables.items():
            # An empty view still gets a sheet, so the workbook's shape does not
            # change depending on whether the judge happened to run.
            frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["model"])
            frame.to_excel(writer, sheet_name=sheet, index=False)
            written.append((sheet, frame))

        header_font = Font(bold=True)
        for sheet, frame in written:
            worksheet = writer.sheets[sheet]
            for cell in worksheet[1]:
                cell.font = header_font
                cell.alignment = Alignment(vertical="center", wrap_text=False)
            worksheet.freeze_panes = "A2"
            if sheet != "About":
                worksheet.auto_filter.ref = worksheet.dimensions
            _autosize(worksheet, frame)

            for index, column in enumerate(frame.columns, start=1):
                fmt = _number_format(str(column))
                if not fmt:
                    continue
                for row in range(2, worksheet.max_row + 1):
                    worksheet.cell(row=row, column=index).number_format = fmt

        legend_sheet = writer.sheets["Legend"]
        legend_sheet.column_dimensions["D"].width = 90
        for row in range(2, legend_sheet.max_row + 1):
            legend_sheet.cell(row=row, column=4).alignment = Alignment(wrap_text=True, vertical="top")

        # Long descriptions on the About sheet need room to breathe.
        about_sheet = writer.sheets["About"]
        about_sheet.column_dimensions["A"].width = 26
        about_sheet.column_dimensions["B"].width = 96
        for row in range(2, about_sheet.max_row + 1):
            about_sheet.cell(row=row, column=2).alignment = Alignment(wrap_text=True, vertical="top")

    _normalize_workbook(target, results_timestamp(summaries))
    return target


def _normalize_workbook(path: Path, stamp: str) -> None:
    """Make the .xlsx byte-identical for identical results.

    openpyxl stamps the workbook's core properties with "now" and the zip
    entries with the current time, so a regeneration that changed nothing still
    produces a different file -- which would have the results workflow commit a
    new spreadsheet on every run. Both are rewritten to a fixed value derived
    from the results themselves.
    """
    import shutil
    import zipfile
    from datetime import datetime

    try:
        moment = datetime.fromisoformat(stamp)
        fixed = (moment.year, moment.month, moment.day, moment.hour, moment.minute, moment.second)
    except (TypeError, ValueError):
        fixed = (1980, 1, 1, 0, 0, 0)

    core = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"'
        ' xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/"'
        ' xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:creator>llmbench</dc:creator><cp:lastModifiedBy>llmbench</cp:lastModifiedBy>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{stamp}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{stamp}</dcterms:modified>'
        "</cp:coreProperties>"
    )

    temporary = path.with_suffix(".xlsx.tmp")
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(
        temporary, "w", zipfile.ZIP_DEFLATED
    ) as target:
        for name in sorted(source.namelist()):
            data = core.encode("utf-8") if name == "docProps/core.xml" else source.read(name)
            info = zipfile.ZipInfo(name, date_time=fixed)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            target.writestr(info, data)
    shutil.move(temporary, path)


def sheet_titles() -> dict[str, str]:
    return dict(TABLES)
