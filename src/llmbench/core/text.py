"""Language-aware text utilities shared by preprocessing, metrics and estimation.

Everything here is deterministic and model-independent: the same source text is
truncated, tokenized and split identically for every model in the benchmark.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "detect_script_profile",
    "split_sentences",
    "multilingual_tokens",
    "truncate_to_tokens",
    "estimate_tokens",
    "char_class_counts",
]

_HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
_CJK = re.compile(r"[一-鿿぀-ヿ豈-﫿]")
_WORD = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[^\W\d_]+", re.UNICODE)
_CJK_OR_HANGUL_CHAR = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏一-鿿぀-ヿ]")

# Sentence terminators for Latin + CJK punctuation.
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？…])[\s​]+|\n+")


def char_class_counts(text: str) -> dict[str, int]:
    """Count characters per script class; used by the token estimator."""
    hangul = cjk = latin = other = 0
    for ch in text:
        if _HANGUL.match(ch):
            hangul += 1
        elif _CJK.match(ch):
            cjk += 1
        elif ch.isspace():
            continue
        elif ch.isascii():
            latin += 1
        else:
            other += 1
    return {"hangul": hangul, "cjk": cjk, "latin": latin, "other": other}


def detect_script_profile(text: str) -> str:
    """Return ``"ko"``, ``"cjk"`` or ``"latin"`` for the dominant script."""
    counts = char_class_counts(text)
    total = sum(counts.values()) or 1
    if counts["hangul"] / total > 0.15:
        return "ko"
    if counts["cjk"] / total > 0.15:
        return "cjk"
    return "latin"


def split_sentences(text: str) -> list[str]:
    """Deterministic sentence splitter (needed for ROUGE-Lsum).

    Intentionally simple and rule-based: a learned splitter would make the metric
    depend on an external model checkpoint.
    """
    text = unicodedata.normalize("NFC", text).strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text)]
    return [p for p in parts if p]


def multilingual_tokens(text: str, lowercase: bool = True) -> list[str]:
    """Tokenizer used by ROUGE.

    * Latin / digits -> word tokens
    * Hangul / CJK   -> single characters

    Character-level tokenization for Hangul is the standard workaround for
    ROUGE on Korean: the reference `rouge_score` tokenizer strips every
    non-ASCII-alphanumeric character, which would delete Korean text entirely
    and report a flat 0.0.
    """
    if lowercase:
        text = text.lower()
    text = unicodedata.normalize("NFC", text)
    tokens: list[str] = []
    for match in _WORD.finditer(text):
        word = match.group(0)
        if _CJK_OR_HANGUL_CHAR.search(word):
            # Mixed or pure CJK/Hangul run -> explode into characters, keeping
            # embedded Latin/digit runs together.
            buf = ""
            for ch in word:
                if _CJK_OR_HANGUL_CHAR.match(ch):
                    if buf:
                        tokens.append(buf)
                        buf = ""
                    tokens.append(ch)
                else:
                    buf += ch
            if buf:
                tokens.append(buf)
        else:
            tokens.append(word)
    return tokens


# Heuristic characters-per-token ratios. These are only used for *estimates*
# (dry-run / budget guard); billed token counts always come from OpenRouter.
_CHARS_PER_TOKEN = {"hangul": 1.5, "cjk": 1.0, "latin": 4.0, "other": 2.5}


def estimate_tokens(text: str, *, overhead: int = 0) -> int:
    """Script-aware token count estimate.

    Deliberately conservative (rounds up) so the budget guard never under-counts.
    """
    if not text:
        return overhead
    counts = char_class_counts(text)
    est = 0.0
    for key, n in counts.items():
        est += n / _CHARS_PER_TOKEN[key]
    # Whitespace/punctuation glue tokens.
    est += text.count("\n") * 0.5
    return int(est + 0.999) + overhead


def truncate_to_tokens(text: str, max_tokens: int, *, word_boundary: bool = True) -> tuple[str, bool]:
    """Canonical truncation to approximately ``max_tokens`` estimated tokens.

    Returns ``(text, truncated)``. Identical for every model: truncation is a
    property of the frozen sample, never of the model being evaluated.
    """
    if max_tokens <= 0:
        return text, False
    if estimate_tokens(text) <= max_tokens:
        return text, False

    profile = detect_script_profile(text)
    ratio = _CHARS_PER_TOKEN["hangul"] if profile == "ko" else (
        _CHARS_PER_TOKEN["cjk"] if profile == "cjk" else _CHARS_PER_TOKEN["latin"]
    )
    budget_chars = int(max_tokens * ratio)

    cut = text[:budget_chars]
    # Binary-search down until the estimate fits (the mixed-script estimate is
    # not exactly linear in characters).
    while cut and estimate_tokens(cut) > max_tokens:
        cut = cut[: int(len(cut) * 0.95)]

    if word_boundary and cut and profile == "latin":
        idx = cut.rfind(" ")
        if idx > 0:
            cut = cut[:idx]
    return cut.strip(), True
