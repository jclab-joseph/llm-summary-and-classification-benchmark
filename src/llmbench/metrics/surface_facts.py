"""Deterministic surface-fact support diagnostic (no LLM involved).

This is NOT a hallucination metric. It is a cheap, fully reproducible
*diagnostic*: it extracts surface-level factual atoms (numbers, percentages,
dates, currency amounts, high-precision named-entity candidates) from a
generated summary and checks whether each one is recoverable from the source
document after normalization.

    surface_fact_support_precision = supported extracted facts / all extracted facts

Known limits, stated plainly because the number is easy to over-read:
* it cannot see paraphrase-level unfaithfulness ("denied" vs "confirmed");
* it cannot see omission;
* entity candidates are restricted to high-precision patterns, so entity recall
  is low by design -- a clean score does not mean a faithful summary.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from llmbench.config.schema import SurfaceFactsMetricConfig
from llmbench.metrics.base import Evaluator

__all__ = ["SurfaceFactEvaluator", "extract_facts", "Fact", "FACT_CATEGORIES"]

FACT_CATEGORIES = ("number", "percentage", "date", "currency", "entity")

_NUM = r"\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?"
_PERCENT_RE = re.compile(rf"({_NUM})\s*(?:%|퍼센트|percent\b|pct\b)", re.IGNORECASE)
_CURRENCY_SYMBOLS = {"$": "USD", "US$": "USD", "£": "GBP", "€": "EUR", "¥": "JPY", "₩": "KRW", "元": "CNY"}
_CURRENCY_PREFIX_RE = re.compile(rf"(US\$|[$£€¥₩])\s*({_NUM})\s*(억|만|천|백|million|billion|trillion)?", re.IGNORECASE)
_CURRENCY_SUFFIX_RE = re.compile(rf"({_NUM})\s*(억|만|천|백)?\s*(원|달러|유로|엔|파운드|위안|dollars?|euros?|pounds?|yen|won)", re.IGNORECASE)
_KO_UNIT_RE = re.compile(rf"({_NUM})\s*(조|억|만|천|백)")
_EN_SCALE_RE = re.compile(rf"({_NUM})\s+(million|billion|trillion|thousand)", re.IGNORECASE)
_PLAIN_NUM_RE = re.compile(_NUM)

_MONTHS = (
    "january february march april may june july august september october november december "
    "jan feb mar apr jun jul aug sep sept oct nov dec"
).split()
_MONTH_INDEX = {
    **{m: i % 12 + 1 for i, m in enumerate(_MONTHS[:12], start=0)},
    **{
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    },
}
_ISO_DATE_RE = re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_KO_DATE_RE = re.compile(r"(?:(\d{4})\s*년)?\s*(?:(\d{1,2})\s*월)?\s*(?:(\d{1,2})\s*일)")
_KO_YEARMONTH_RE = re.compile(r"(?:(\d{4})\s*년)\s*(?:(\d{1,2})\s*월)?")
_EN_DATE_RE = re.compile(
    r"\b(" + "|".join(sorted(_MONTH_INDEX, key=len, reverse=True)) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?\b",
    re.IGNORECASE,
)
_EN_DAY_MONTH_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(sorted(_MONTH_INDEX, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2}|21\d{2})\b")

_KO_SCALES = {"조": 10**12, "억": 10**8, "만": 10**4, "천": 10**3, "백": 10**2}
_EN_SCALES = {"thousand": 10**3, "million": 10**6, "billion": 10**9, "trillion": 10**12}

# High-precision Korean proper-noun suffixes. Restricting candidates this way
# keeps entity precision usable without a morphological analyzer or an NER model
# (either of which would make the diagnostic non-deterministic across installs).
_KO_ENTITY_SUFFIXES = (
    "위원회", "대학교", "연구소", "연구원", "협회", "재단", "공사", "공단", "은행", "항공", "그룹",
    "병원", "공항", "경찰서", "검찰청", "대통령", "총리", "장관", "의회", "정부", "대학", "학교",
    "신문", "방송", "통신", "기업", "회사", "부처", "본부", "센터", "구단", "리그",
)
_KO_JOSA = (
    "에서는", "에게서", "으로는", "이라고", "라고는", "에서도", "부터는", "까지는",
    "에서", "에게", "으로", "라고", "부터", "까지", "처럼", "보다", "한테", "와의", "과의",
    "은", "는", "이", "가", "을", "를", "에", "의", "도", "만", "와", "과", "로", "며",
)
_EN_SENTENCE_STARTERS = {
    "the", "a", "an", "in", "on", "at", "it", "this", "that", "these", "those", "but", "and",
    "he", "she", "they", "we", "you", "his", "her", "their", "its", "there", "when", "while",
    "after", "before", "however", "meanwhile", "according", "as", "for", "to", "of", "by",
    "if", "so", "then", "now", "also", "more", "most", "some", "no", "not", "one", "two",
}
# Calendar words are capitalized but are dates, not entities: counting them twice
# would double-penalize a single wrong date.
_ENTITY_STOPWORDS = set(_MONTHS) | {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}
_LATIN_ENTITY_RE = re.compile(r"\b([A-Z][\w'&.-]*(?:\s+[A-Z][\w'&.-]*)*)\b")
_HANGUL_TOKEN_RE = re.compile(r"[가-힣]+")


@dataclass(frozen=True, slots=True)
class Fact:
    category: str
    surface: str
    normalized: str
    value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "surface": self.surface,
            "normalized": self.normalized,
            "value": self.value,
        }


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def _num(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _scaled(raw: str, scale_token: str | None) -> float | None:
    base = _num(raw)
    if base is None:
        return None
    if not scale_token:
        return base
    token = scale_token.lower()
    return base * _KO_SCALES.get(scale_token, _EN_SCALES.get(token, 1))


def _fmt(value: float) -> str:
    return f"{value:.6g}"


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def _extract_percentages(text: str) -> list[Fact]:
    out = []
    for match in _PERCENT_RE.finditer(text):
        value = _num(match.group(1))
        if value is not None:
            out.append(Fact("percentage", match.group(0).strip(), _fmt(value), value))
    return out


def _extract_currency(text: str) -> list[Fact]:
    out = []
    for match in _CURRENCY_PREFIX_RE.finditer(text):
        value = _scaled(match.group(2), match.group(3))
        if value is None:
            continue
        code = _CURRENCY_SYMBOLS.get(match.group(1).upper(), _CURRENCY_SYMBOLS.get(match.group(1), "?"))
        out.append(Fact("currency", match.group(0).strip(), f"{code}:{_fmt(value)}", value))
    for match in _CURRENCY_SUFFIX_RE.finditer(text):
        value = _scaled(match.group(1), match.group(2))
        if value is None:
            continue
        unit = match.group(3).lower()
        code = {
            "원": "KRW", "won": "KRW", "달러": "USD", "dollar": "USD", "dollars": "USD",
            "유로": "EUR", "euro": "EUR", "euros": "EUR", "엔": "JPY", "yen": "JPY",
            "파운드": "GBP", "pound": "GBP", "pounds": "GBP", "위안": "CNY",
        }.get(unit, "?")
        out.append(Fact("currency", match.group(0).strip(), f"{code}:{_fmt(value)}", value))
    return out


def _extract_dates(text: str) -> list[Fact]:
    out: list[Fact] = []
    for match in _ISO_DATE_RE.finditer(text):
        y, m, d = match.group(1), int(match.group(2)), int(match.group(3))
        out.append(Fact("date", match.group(0), f"{y}-{m:02d}-{d:02d}"))
    for match in _KO_DATE_RE.finditer(text):
        if not match.group(0).strip():
            continue
        y, m, d = match.group(1), match.group(2), match.group(3)
        if not d:
            continue
        parts = [y or "----", f"{int(m):02d}" if m else "--", f"{int(d):02d}"]
        out.append(Fact("date", match.group(0).strip(), "-".join(parts)))
    for match in _KO_YEARMONTH_RE.finditer(text):
        y, m = match.group(1), match.group(2)
        if not y:
            continue
        out.append(Fact("date", match.group(0).strip(), f"{y}-{int(m):02d}" if m else f"{y}"))
    for match in _EN_DATE_RE.finditer(text):
        month = _MONTH_INDEX[match.group(1).lower()]
        day = int(match.group(2))
        year = match.group(3) or "----"
        out.append(Fact("date", match.group(0), f"{year}-{month:02d}-{day:02d}"))
    for match in _EN_DAY_MONTH_RE.finditer(text):
        month = _MONTH_INDEX[match.group(2).lower()]
        day = int(match.group(1))
        out.append(Fact("date", match.group(0), f"-----{month:02d}-{day:02d}"))
    for match in _YEAR_RE.finditer(text):
        out.append(Fact("date", match.group(0), match.group(1)))
    # De-duplicate identical normalized dates while keeping order.
    seen: set[str] = set()
    unique = []
    for fact in out:
        if fact.normalized in seen:
            continue
        seen.add(fact.normalized)
        unique.append(fact)
    return unique


def _consumed_spans(text: str, patterns: Iterable[re.Pattern[str]]) -> list[tuple[int, int]]:
    spans = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            spans.append(match.span())
    return spans


def _extract_numbers(text: str) -> list[Fact]:
    """Plain numbers that are not already covered by a richer category."""
    spans = _consumed_spans(
        text,
        (_PERCENT_RE, _CURRENCY_PREFIX_RE, _CURRENCY_SUFFIX_RE, _ISO_DATE_RE, _KO_DATE_RE, _KO_YEARMONTH_RE, _EN_DATE_RE, _EN_DAY_MONTH_RE, _YEAR_RE),
    )

    def covered(start: int, end: int) -> bool:
        return any(s <= start and end <= e for s, e in spans)

    out: list[Fact] = []
    handled: list[tuple[int, int]] = []
    for pattern in (_KO_UNIT_RE, _EN_SCALE_RE):
        for match in pattern.finditer(text):
            if covered(*match.span()):
                continue
            value = _scaled(match.group(1), match.group(2))
            if value is None:
                continue
            handled.append(match.span())
            out.append(Fact("number", match.group(0).strip(), _fmt(value), value))
    for match in _PLAIN_NUM_RE.finditer(text):
        start, end = match.span()
        if covered(start, end) or any(s <= start and end <= e for s, e in handled):
            continue
        value = _num(match.group(0))
        if value is None:
            continue
        out.append(Fact("number", match.group(0), _fmt(value), value))
    return out


def _strip_josa(token: str) -> str:
    for josa in sorted(_KO_JOSA, key=len, reverse=True):
        if token.endswith(josa) and len(token) - len(josa) >= 2:
            return token[: -len(josa)]
    return token


def _extract_entities(text: str) -> list[Fact]:
    out: list[Fact] = []
    seen: set[str] = set()

    # Latin-script proper nouns (present in both EN and KO news text).
    for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", text):
        for match in _LATIN_ENTITY_RE.finditer(sentence):
            surface = match.group(1).strip(" .")
            if len(surface) < 2:
                continue
            words = surface.split()
            if len(words) == 1:
                lowered = surface.lower()
                if lowered in _EN_SENTENCE_STARTERS or lowered in _ENTITY_STOPWORDS:
                    continue
                if surface.isupper() and len(surface) < 2:
                    continue
            key = surface.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(Fact("entity", surface, key))

    # High-precision Korean proper-noun candidates.
    for token in _HANGUL_TOKEN_RE.findall(text):
        stem = _strip_josa(token)
        if len(stem) < 2:
            continue
        if not stem.endswith(_KO_ENTITY_SUFFIXES):
            continue
        key = stem
        if key in seen:
            continue
        seen.add(key)
        out.append(Fact("entity", token, key))

    return out


def extract_facts(text: str, config: SurfaceFactsMetricConfig) -> list[Fact]:
    """Extract every enabled surface fact category from ``text``."""
    text = _norm(text)
    facts: list[Fact] = []
    if config.check_percentages:
        facts.extend(_extract_percentages(text))
    if config.check_currency:
        facts.extend(_extract_currency(text))
    if config.check_dates:
        facts.extend(_extract_dates(text))
    if config.check_numbers:
        facts.extend(_extract_numbers(text))
    if config.check_entities:
        facts.extend(_extract_entities(text))
    return facts


# --------------------------------------------------------------------------- #
# support checking
# --------------------------------------------------------------------------- #
class _SourceIndex:
    """Normalized view of the source used to decide whether a fact is supported."""

    def __init__(self, source: str, config: SurfaceFactsMetricConfig) -> None:
        self.text = _norm(source)
        self.lower = self.text.lower()
        facts = extract_facts(source, config)
        self.by_category: dict[str, set[str]] = {c: set() for c in FACT_CATEGORIES}
        self.values: dict[str, set[float]] = {c: set() for c in FACT_CATEGORIES}
        for fact in facts:
            self.by_category[fact.category].add(fact.normalized)
            if fact.value is not None:
                self.values[fact.category].add(fact.value)
        self.all_values = set().union(*self.values.values()) if self.values else set()

    def supports(self, fact: Fact, tolerance: float) -> bool:
        if fact.category == "entity":
            return fact.normalized in self.lower or fact.surface.lower() in self.lower
        if fact.category == "date":
            if fact.normalized in self.by_category["date"]:
                return True
            # A bare year in the summary is supported by any source date in that year.
            if len(fact.normalized) == 4:
                return any(d.startswith(fact.normalized) for d in self.by_category["date"])
            year = fact.normalized.split("-")[0]
            if year != "----" and year in self.by_category["date"]:
                return True
            return fact.surface.lower() in self.lower
        if fact.value is None:
            return fact.normalized in self.by_category.get(fact.category, set())
        if fact.category == "currency":
            code = fact.normalized.split(":")[0]
            same_code = {v for v in self.by_category["currency"] if v.startswith(f"{code}:")}
            if fact.normalized in same_code:
                return True
            return _value_in(fact.value, self.values["currency"], tolerance)
        if fact.category == "percentage":
            if _value_in(fact.value, self.values["percentage"], tolerance):
                return True
            return _value_in(fact.value, self.values["number"], tolerance)
        return _value_in(fact.value, self.all_values, tolerance)


def _value_in(value: float, pool: Iterable[float], tolerance: float) -> bool:
    for candidate in pool:
        if tolerance <= 0:
            if candidate == value:
                return True
        elif abs(candidate - value) <= tolerance * max(1.0, abs(value)):
            return True
    return False


class SurfaceFactEvaluator:
    """Per-sample surface fact support, plus per-category mismatch counts."""

    def __init__(self, config: SurfaceFactsMetricConfig) -> None:
        self.config = config
        self.evaluator = Evaluator(
            name="surface_facts",
            version=config.version,
            config=config.model_dump(),
        )

    def score(self, source: str, summary: str) -> dict[str, Any]:
        facts = extract_facts(summary, self.config)
        index = _SourceIndex(source, self.config)

        supported = 0
        mismatches = {f"{c}_mismatch": 0 for c in FACT_CATEGORIES if c != "entity"}
        mismatches["unsupported_named_entity"] = 0
        per_category: dict[str, dict[str, int]] = {
            c: {"total": 0, "supported": 0} for c in FACT_CATEGORIES
        }
        unsupported_examples: list[dict[str, Any]] = []

        for fact in facts:
            per_category[fact.category]["total"] += 1
            if index.supports(fact, self.config.numeric_tolerance):
                supported += 1
                per_category[fact.category]["supported"] += 1
            else:
                key = "unsupported_named_entity" if fact.category == "entity" else f"{fact.category}_mismatch"
                mismatches[key] += 1
                if len(unsupported_examples) < 5:
                    unsupported_examples.append(fact.to_dict())

        total = len(facts)
        return {
            "surface_fact_support_precision": (supported / total) if total else None,
            "facts_total": total,
            "facts_supported": supported,
            "per_category": per_category,
            "unsupported_examples": unsupported_examples,
            **mismatches,
        }


def surface_fact_aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Corpus-level (micro) aggregation of the diagnostic."""
    total = sum(int(r.get("facts_total", 0) or 0) for r in rows)
    supported = sum(int(r.get("facts_supported", 0) or 0) for r in rows)
    out: dict[str, Any] = {
        "diagnostic": True,
        "note": "Surface-level check only; not a complete hallucination metric.",
        "facts_total": total,
        "facts_supported": supported,
        "surface_fact_support_precision": (supported / total) if total else None,
        "summaries_with_facts": sum(1 for r in rows if (r.get("facts_total") or 0) > 0),
    }
    for key in ("number_mismatch", "percentage_mismatch", "date_mismatch", "currency_mismatch", "unsupported_named_entity"):
        out[key] = sum(int(r.get(key, 0) or 0) for r in rows)
    per_category: dict[str, dict[str, int]] = {c: {"total": 0, "supported": 0} for c in FACT_CATEGORIES}
    for row in rows:
        for category, stats in (row.get("per_category") or {}).items():
            per_category[category]["total"] += int(stats.get("total", 0))
            per_category[category]["supported"] += int(stats.get("supported", 0))
    out["per_category"] = {
        category: {
            **stats,
            "support_rate": (stats["supported"] / stats["total"]) if stats["total"] else None,
        }
        for category, stats in per_category.items()
    }
    return out
