"""Dataset readers producing normalized records for the prepare step."""

from __future__ import annotations

import gzip
import json
import re
import tarfile
import unicodedata
import zipfile
from difflib import SequenceMatcher
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from llmbench.config.schema import DatasetSources
from llmbench.core.canonical import normalize_text
from llmbench.core.errors import DatasetError

__all__ = [
    "read_xlsum",
    "read_halueval_summarization",
    "read_massive",
    "read_aihub_factuality",
    "AIHUB_FIELD_ALIASES",
    "AIHUB_ERROR_TYPES",
    "AIHUB_FACTUAL_ERROR_TYPES",
    "summary_has_broken_prefix",
]


# --------------------------------------------------------------------------- #
# XL-Sum  (csebuetnlp/xlsum, tar.bz2 of {lang}_{split}.jsonl)
# --------------------------------------------------------------------------- #
def read_xlsum(archive: Path, *, language: str, split: str) -> Iterator[dict[str, Any]]:
    member_suffix = f"_{split}.jsonl"
    try:
        with tarfile.open(archive, "r:bz2") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(member_suffix)]
            if not members:
                names = [m.name for m in tar.getmembers()]
                raise DatasetError(f"no *{member_suffix} member in {archive} (found {names})")
            handle = tar.extractfile(members[0])
            if handle is None:
                raise DatasetError(f"cannot read {members[0].name} from {archive}")
            for line in handle:
                row = json.loads(line.decode("utf-8"))
                yield {
                    "id": row["id"],
                    "language": language,
                    "title": normalize_text(row.get("title", "")),
                    "source": normalize_text(row["text"]),
                    "reference": normalize_text(row["summary"]),
                    "url": row.get("url", ""),
                }
    except (tarfile.TarError, OSError) as exc:
        raise DatasetError(f"cannot open XL-Sum archive {archive}: {exc}") from exc


# --------------------------------------------------------------------------- #
# HaluEval summarization (parquet: document / right_summary / hallucinated_summary)
# --------------------------------------------------------------------------- #
def read_halueval_summarization(parquet_path: Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    columns = set(table.column_names)
    required = {"document", "right_summary", "hallucinated_summary"}
    if not required.issubset(columns):
        raise DatasetError(
            f"HaluEval parquet {parquet_path} is missing columns {sorted(required - columns)}"
        )
    docs = table.column("document").to_pylist()
    right = table.column("right_summary").to_pylist()
    wrong = table.column("hallucinated_summary").to_pylist()
    for idx, (doc, ok, bad) in enumerate(zip(docs, right, wrong, strict=True)):
        if not doc or not ok or not bad:
            continue
        yield {
            "id": f"halueval-{idx:05d}",
            "index": idx,
            "source": normalize_text(doc),
            "right_summary": normalize_text(ok),
            "hallucinated_summary": normalize_text(bad),
        }


# --------------------------------------------------------------------------- #
# MASSIVE intent classification (mteb/amazon_massive_intent, gzipped JSONL)
# --------------------------------------------------------------------------- #
def read_massive(path: Path, *, language: str) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" or path.name.endswith(".json.gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = normalize_text(row.get("text", ""))
            if not text:
                continue
            yield {
                # `id` is the MASSIVE semantic id: the same value denotes the same
                # utterance across every locale, which is what makes the EN/KO
                # cross-lingual consistency metric meaningful.
                "id": str(row["id"]),
                "language": language,
                "text": text,
                "intent": row.get("label_text") or row.get("label"),
            }


# --------------------------------------------------------------------------- #
# AI-Hub 추상 요약 사실성 검증 (licence-gated: always local, never downloaded)
# --------------------------------------------------------------------------- #
AIHUB_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "sample_id", "doc_id", "document_id", "문서번호", "no", "idx"),
    "source": ("source", "document", "passage", "context", "text", "article", "input", "원문", "본문"),
    "erroneous_summary": (
        "erroneous_summary",
        "original_summary",
        "hallucinated_summary",
        "error_summary",
        "wrong_summary",
        "summary",
        "오류요약",
        "원본요약",
        "요약문",
    ),
    "corrected_summary": (
        "corrected_summary",
        "revised_summary",
        "correct_summary",
        "fixed_summary",
        "gold_summary",
        "수정요약",
        "교정요약",
        "수정문",
    ),
    "error_type": ("error_type", "error_types", "type", "오류유형", "오류_유형", "errortype", "label"),
}


def _pick(row: dict[str, Any], key: str) -> Any:
    lowered = {str(k).lower().replace(" ", "_"): v for k, v in row.items()}
    for alias in AIHUB_FIELD_ALIASES[key]:
        if alias in row:
            return row[alias]
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def _iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        if isinstance(data, list):
            yield from (row for row in data if isinstance(row, dict))
        return
    if stripped.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            # A wrapper object: look for the first list-of-dicts value.
            for value in data.values():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    yield from value
                    return
            yield data
            return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            yield row


AIHUB_ERROR_TYPES = {
    "type1": "[1] 한글 맞춤법, 띄어쓰기 오류",
    "type2": "[2] 단어 선택 오류",
    "type3": "[3] 비문",
    "type4": "[4] 미완성 또는 불완전한 문장",
    "type5": "[5] 키워드 또는 중요 내용 오류",
    "type6": "[6] 유사한 내용 반복",
}
# Only type5 is a *content* error. A spelling or repetition fix does not make a
# summary factually unsupported, so the other types cannot be used to build
# SUPPORTED / HALLUCINATED pairs.
AIHUB_FACTUAL_ERROR_TYPES = frozenset({"type5"})

# A handful of AI-Hub source documents carry a leftover "summary |" prefix.
_AIHUB_SOURCE_PREFIX = re.compile(r"^\s*summary\s*\|\s*", re.IGNORECASE)


def _span_kind(sub: str, correction: str) -> str:
    """Classify one annotated error span.

    ``type5`` ("키워드 또는 중요 내용 오류") mixes two very different things:

    * the summary states something the source does not -- a genuine
      unsupported claim; and
    * the summary *omits* key content, which the annotator fixed by adding it.

    Only the first kind makes a summary HALLUCINATED for a factual-consistency
    task: an omission leaves every remaining claim supported. When the erroneous
    span is contained verbatim in its correction, the correction only added
    material, so the span is an omission.
    """
    sub_n = re.sub(r"\s+", "", unicodedata.normalize("NFC", sub or ""))
    cor_n = re.sub(r"\s+", "", unicodedata.normalize("NFC", correction or ""))
    if not sub_n or not cor_n or sub_n == cor_n:
        return "unknown"

    # A character diff, not a substring test: the annotator frequently inserts
    # the missing clause in the *middle* of the span, so "따라 보험사들이" ->
    # "따라 보험업 규제가 대폭 완화되어 보험사들이" is an insertion even though the
    # original is not a contiguous substring of the correction.
    opcodes = {op for op, *_ in SequenceMatcher(None, sub_n, cor_n, autojunk=False).get_opcodes()}
    opcodes.discard("equal")
    if opcodes == {"insert"}:
        return "omission"
    if opcodes == {"delete"}:
        # The erroneous summary carried extra material the source does not
        # support -- a textbook hallucination.
        return "extraneous"
    return "alteration"


_HANGUL_CHAR = re.compile(r"[\uac00-\ud7a3]")


def summary_has_broken_prefix(summary: str, source: str) -> bool:
    """Detect a summary whose first syllable was lost upstream.

    Some AI-Hub records begin mid-word -- "럼프 대통령과" for "트럼프 대통령과",
    "은경 환경부" for "김은경 환경부". The tell is that the summary's opening
    characters never start a word in the source, but do appear there preceded by
    another Hangul syllable. It is rare in machine summaries (~0.5%) and common
    in the small human-written subset (~48%).
    """
    prefix = summary[:4]
    if len(prefix) < 4 or not _HANGUL_CHAR.match(prefix[:1]):
        return False
    if re.search(r"(?:^|[\s\"'(\[])" + re.escape(prefix), source):
        return False
    return bool(re.search(r"[\uac00-\ud7a3]" + re.escape(prefix), source))


def _parse_aihub_native(row: dict[str, Any]) -> dict[str, Any] | None:
    """Parse one record in AI-Hub 157 (추상요약 사실성 검증) label format.

    Shape::

        {"annotation": {"target": "machine_abstractive_summary",
                        "original_summary": {<target>: {"text": ...}},
                        "corrected_summary": {"corrected_type5": {"text": ..., ...},
                                              "corrected_all": {"text": ...}}},
         "original_text": "..."}

    Returns None when the record is not in this format (e.g. a 원천데이터 file
    with no corrections), so the caller can fall back to the generic reader.
    """
    annotation = row.get("annotation")
    source = row.get("original_text")
    if not isinstance(annotation, dict) or not source:
        return None
    corrected = annotation.get("corrected_summary")
    originals = annotation.get("original_summary")
    if not isinstance(corrected, dict) or not isinstance(originals, dict):
        return None

    target = annotation.get("target") or ""
    original = originals.get(target) or {}
    erroneous = original.get("text") if isinstance(original, dict) else None
    if not erroneous:
        return None

    present = {
        key.removeprefix("corrected_"): value
        for key, value in corrected.items()
        if key.startswith("corrected_type") and isinstance(value, dict) and value.get("text")
    }
    if not present:
        return None

    span_kinds: list[str] = []
    for value in present.values():
        for error in value.get("errors") or []:
            if isinstance(error, dict):
                span_kinds.append(_span_kind(error.get("sub", ""), error.get("correction", "")))

    if len(present) == 1:
        # A single error type: its own correction is the fully faithful summary.
        only_type = next(iter(present))
        corrected_text = present[only_type]["text"]
    else:
        # Several error types: only `corrected_all` fixes all of them, so a
        # single-type correction would still leave the summary unsupported.
        corrected_all = corrected.get("corrected_all") or {}
        corrected_text = corrected_all.get("text") if isinstance(corrected_all, dict) else None
        if not corrected_text:
            return None

    return {
        "source": normalize_text(_AIHUB_SOURCE_PREFIX.sub("", str(source))),
        "erroneous_summary": normalize_text(str(erroneous)),
        "corrected_summary": normalize_text(str(corrected_text)),
        "error_types": sorted(present),
        "correction_labels": sorted(
            str(value.get("correction_type") or AIHUB_ERROR_TYPES.get(key, key))
            for key, value in present.items()
        ),
        "error_span_kinds": span_kinds,
        # Every annotated span only adds missing content: the erroneous summary
        # is incomplete, not unsupported, so it is not a hallucination example.
        "is_omission_only": bool(span_kinds) and set(span_kinds) == {"omission"},
        "summary_target": target,
        "summary_kind": "machine" if target.startswith("machine") else "human",
        "document_type": annotation.get("type", ""),
        "schema": "aihub-157",
    }


def _parse_aihub_generic(row: dict[str, Any]) -> dict[str, Any] | None:
    """Fallback for user-converted exports keyed by the documented field names."""
    source = _pick(row, "source")
    erroneous = _pick(row, "erroneous_summary")
    corrected = _pick(row, "corrected_summary")
    if not (source and erroneous and corrected):
        return None

    raw_types = _pick(row, "error_type")
    if isinstance(raw_types, str):
        types = [t.strip() for t in raw_types.replace(";", ",").split(",") if t.strip()]
    elif isinstance(raw_types, (list, tuple)):
        types = [str(t).strip() for t in raw_types if str(t).strip()]
    elif raw_types is None:
        types = []
    else:
        types = [str(raw_types)]

    return {
        "source": normalize_text(str(source)),
        "erroneous_summary": normalize_text(str(erroneous)),
        "corrected_summary": normalize_text(str(corrected)),
        "error_types": types,
        "correction_labels": types,
        "error_span_kinds": [],
        "is_omission_only": False,
        "summary_target": "",
        "summary_kind": "",
        "document_type": "",
        "schema": "generic",
    }


def _iter_aihub_containers(root: Path, split: str | None) -> Iterator[tuple[str, Path, str | None]]:
    """Yield ``(kind, path, split)`` for every readable data container under root.

    ``kind`` is ``"zip"`` or ``"json"``. AI-Hub ships the data as .zip archives of
    per-document JSON; those are read in place so nobody has to extract 300 MB
    to run the benchmark.
    """
    if root.is_file():
        kind = "zip" if root.suffix.lower() == ".zip" else "json"
        yield kind, root, split
        return

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".zip", ".json", ".jsonl"}:
            continue
        parts = {part.lower() for part in path.relative_to(root).parts}
        detected = None
        for candidate in ("training", "validation"):
            if candidate in parts:
                detected = candidate.capitalize()
        if split and detected and detected.lower() != split.lower():
            continue
        yield ("zip" if suffix == ".zip" else "json"), path, detected


def read_aihub_factuality(
    root: Path,
    *,
    preferred_error_type: str = "type5",
    split: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Read a *locally prepared* AI-Hub factuality dataset.

    The AI-Hub licence forbids redistribution and automated scraping, so this
    reader only ever touches the directory the user points
    ``AIHUB_FACTUALITY_DATA_PATH`` at. It understands the distributed layout
    (``…/02.라벨링데이터/*.zip`` holding one JSON per document) directly, and
    falls back to lenient field-name matching for hand-converted exports with the
    keys ``id / source / erroneous_summary / corrected_summary / error_type``.
    """
    if not root.exists():
        raise DatasetError(f"AI-Hub data path does not exist: {root}")

    containers = list(_iter_aihub_containers(root, split))
    if not containers:
        raise DatasetError(
            f"no .zip/.json/.jsonl data found under {root}"
            + (f" for split {split!r}" if split else "")
        )

    seen_ids: set[str] = set()
    found = 0

    def emit(record: dict[str, Any], raw_id: str, origin: str, detected_split: str | None) -> dict[str, Any] | None:
        parsed = _parse_aihub_native(record) or _parse_aihub_generic(record)
        if parsed is None:
            return None
        sample_id = f"{detected_split}/{raw_id}" if detected_split else raw_id
        suffix = 1
        unique = sample_id
        while unique in seen_ids:
            suffix += 1
            unique = f"{sample_id}#{suffix}"
        seen_ids.add(unique)

        types = parsed["error_types"]
        parsed.update(
            {
                "id": unique,
                "split": detected_split or "local",
                "source_file": origin,
                "has_preferred_type": preferred_error_type in types,
                "only_preferred_type": types == [preferred_error_type],
                "is_factual_error": bool(types) and set(types) <= AIHUB_FACTUAL_ERROR_TYPES,
            }
        )
        return parsed

    for kind, path, detected_split in containers:
        origin = str(path.relative_to(root)) if root.is_dir() else path.name
        if kind == "zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    for name in sorted(archive.namelist()):
                        if not name.lower().endswith(".json"):
                            continue
                        try:
                            row = json.loads(archive.read(name).decode("utf-8-sig"))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
                        if not isinstance(row, dict):
                            continue
                        raw_id = Path(name).stem
                        record = emit(row, raw_id, f"{origin}!{name.lstrip('/')}", detected_split)
                        if record is not None:
                            found += 1
                            yield record
            except zipfile.BadZipFile:
                continue
        else:
            for index, row in enumerate(_iter_json_records(path)):
                raw_id = _pick(row, "id")
                raw_id = str(raw_id) if raw_id is not None else f"{path.stem}-{index:06d}"
                record = emit(row, raw_id, origin, detected_split)
                if record is not None:
                    found += 1
                    yield record

    if not found:
        raise DatasetError(
            f"found {len(containers)} data file(s) under {root} but none contained a usable "
            "(source, erroneous summary, corrected summary) triple"
        )


def dataset_source_summary(sources: DatasetSources) -> dict[str, Any]:
    """Human-readable provenance block recorded in manifests and reports."""
    return {
        "xlsum": {"repo": sources.xlsum.repo, "revision": sources.xlsum.revision},
        "halueval": {"repo": sources.halueval.repo, "revision": sources.halueval.revision},
        "massive": {"repo": sources.massive.repo, "revision": sources.massive.revision},
        "aihub": {"repo": "local", "revision": sources.aihub.revision},
    }
