"""`benchmark prepare`: build the frozen manifests.

Sampling is deterministic and happens exactly once. Every model then sees byte-
identical inputs, because the canonical preprocessing (normalization and the
1,800-token source truncation) is applied *here*, not per model at run time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from llmbench.config.loader import AppConfig
from llmbench.core.canonical import hash_text, sha256_hex
from llmbench.core.errors import DatasetError
from llmbench.core.text import estimate_tokens, truncate_to_tokens
from llmbench.datasets.hf import download_dataset_file
from llmbench.datasets.manifest import (
    Manifest,
    ManifestRecord,
    manifest_exists,
    read_manifest,
    write_manifest,
)
from llmbench.datasets.sources import (
    read_aihub_factuality,
    summary_has_broken_prefix,
    read_halueval_summarization,
    read_massive,
    read_xlsum,
)

__all__ = ["PrepareResult", "prepare_all", "MASSIVE_INTENT_COUNT"]

MASSIVE_INTENT_COUNT = 60

# XL-Sum documents shorter than this are dropped: summarizing a 3-sentence stub
# is not a discriminative task and would compress the score range.
_MIN_SOURCE_TOKENS = 120
_MIN_REFERENCE_TOKENS = 8
_HALUEVAL_MIN_SOURCE_TOKENS = 100


@dataclass(slots=True)
class PrepareResult:
    name: str
    status: str  # CREATED | EXISTS | SKIPPED | FAILED
    path: Path | None = None
    counts: dict[str, int] = field(default_factory=dict)
    manifest_hash: str = ""
    dataset_revision: str = ""
    detail: str = ""


def _rank(seed: int, key: str) -> str:
    """Stable pseudo-random ordering key, independent of input order."""
    return sha256_hex(f"{seed}:{key}")


def _deterministic_take(
    rows: Sequence[dict[str, Any]],
    n: int,
    *,
    seed: int,
    key_fn: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: (_rank(seed, key_fn(r)), key_fn(r)))
    return ordered[:n]


# --------------------------------------------------------------------------- #
# XL-Sum -> summarization manifest
# --------------------------------------------------------------------------- #
def prepare_xlsum(cfg: AppConfig, *, force: bool, progress=None) -> PrepareResult:
    bench = cfg.benchmark.benchmarks.summarization
    path = cfg.manifest_dir / bench.manifest
    if manifest_exists(path) and not force:
        manifest = read_manifest(path)
        return PrepareResult(
            "summarization",
            "EXISTS",
            path,
            dict(manifest.meta.counts),
            manifest.meta.manifest_hash,
            manifest.meta.dataset_revision,
        )

    src = cfg.benchmark.dataset_sources.xlsum
    trunc = cfg.benchmark.truncation
    seed = cfg.benchmark.sampling.seed
    records: list[ManifestRecord] = []

    for language, wanted in (("en", bench.cases.en), ("ko", bench.cases.ko)):
        rel = getattr(src.files, language)
        archive = download_dataset_file(
            src.repo, src.revision, rel, cfg.raw_data_dir, progress=progress
        )
        candidates = [
            row
            for row in read_xlsum(archive, language=language, split=src.split)
            if row["source"]
            and row["reference"]
            and estimate_tokens(row["source"]) >= _MIN_SOURCE_TOKENS
            and estimate_tokens(row["reference"]) >= _MIN_REFERENCE_TOKENS
        ]
        if len(candidates) < wanted:
            raise DatasetError(
                f"XL-Sum {language}/{src.split} only has {len(candidates)} usable rows, need {wanted}"
            )
        chosen = _deterministic_take(candidates, wanted, seed=seed, key_fn=lambda r: str(r["id"]))
        chosen.sort(key=lambda r: str(r["id"]))

        for row in chosen:
            source, truncated = truncate_to_tokens(
                row["source"],
                trunc.summarization_max_source_tokens,
                word_boundary=trunc.word_boundary,
            )
            reference = row["reference"]
            records.append(
                ManifestRecord(
                    benchmark="summarization",
                    language=language,
                    split=src.split,
                    dataset_revision=src.revision,
                    sample_id=f"xlsum:{language}:{row['id']}",
                    source_hash=hash_text(source),
                    reference_hash=hash_text(reference),
                    payload={
                        "source": source,
                        "reference": reference,
                        "title": row["title"],
                        "url": row["url"],
                        "truncated": truncated,
                        "raw_source_hash": hash_text(row["source"]),
                        "source_tokens_est": estimate_tokens(source),
                        "reference_tokens_est": estimate_tokens(reference),
                    },
                )
            )

    manifest = write_manifest(
        path,
        records,
        benchmark_version=bench.version,
        dataset_repo=src.repo,
        dataset_revision=src.revision,
        split=src.split,
        seed=seed,
        preprocessing={
            "normalization": "NFC + whitespace collapse",
            "truncation_policy_version": trunc.version,
            "max_source_tokens": trunc.summarization_max_source_tokens,
            "min_source_tokens": _MIN_SOURCE_TOKENS,
        },
    )
    return PrepareResult(
        "summarization",
        "CREATED",
        path,
        dict(manifest.meta.counts),
        manifest.meta.manifest_hash,
        src.revision,
    )


# --------------------------------------------------------------------------- #
# HaluEval -> English hallucination manifest
# --------------------------------------------------------------------------- #
def prepare_halueval(cfg: AppConfig, *, force: bool, progress=None) -> PrepareResult:
    bench = cfg.benchmark.benchmarks.hallucination
    path = cfg.manifest_dir / bench.manifests.en
    if manifest_exists(path) and not force:
        manifest = read_manifest(path)
        return PrepareResult(
            "hallucination:en",
            "EXISTS",
            path,
            dict(manifest.meta.counts),
            manifest.meta.manifest_hash,
            manifest.meta.dataset_revision,
        )

    src = cfg.benchmark.dataset_sources.halueval
    trunc = cfg.benchmark.truncation
    seed = cfg.benchmark.sampling.seed
    wanted = bench.documents.en

    parquet = download_dataset_file(src.repo, src.revision, src.file, cfg.raw_data_dir, progress=progress)
    candidates = [
        row
        for row in read_halueval_summarization(parquet)
        if estimate_tokens(row["source"]) >= _HALUEVAL_MIN_SOURCE_TOKENS
        and row["right_summary"] != row["hallucinated_summary"]
    ]
    if len(candidates) < wanted:
        raise DatasetError(f"HaluEval summarization has {len(candidates)} usable rows, need {wanted}")

    chosen = _deterministic_take(candidates, wanted, seed=seed, key_fn=lambda r: r["id"])
    chosen.sort(key=lambda r: r["id"])

    records: list[ManifestRecord] = []
    for row in chosen:
        source, truncated = truncate_to_tokens(
            row["source"], trunc.hallucination_max_source_tokens, word_boundary=trunc.word_boundary
        )
        # Each document yields exactly two cases: the faithful summary and the
        # hallucinated one. The pair shares a document_id so per-document
        # analysis stays possible.
        for label, candidate in (
            ("SUPPORTED", row["right_summary"]),
            ("HALLUCINATED", row["hallucinated_summary"]),
        ):
            records.append(
                ManifestRecord(
                    benchmark="hallucination",
                    language="en",
                    split=src.split,
                    dataset_revision=src.revision,
                    sample_id=f"halueval:en:{row['id']}:{label.lower()}",
                    source_hash=hash_text(source),
                    reference_hash=hash_text(label),
                    label=label,
                    payload={
                        "document_id": row["id"],
                        "source": source,
                        "candidate_summary": candidate,
                        "gold_label": label,
                        "truncated": truncated,
                        "candidate_hash": hash_text(candidate),
                        "source_tokens_est": estimate_tokens(source),
                    },
                )
            )

    manifest = write_manifest(
        path,
        records,
        benchmark_version=bench.version,
        dataset_repo=src.repo,
        dataset_revision=src.revision,
        split=src.split,
        seed=seed,
        preprocessing={
            "truncation_policy_version": trunc.version,
            "max_source_tokens": trunc.hallucination_max_source_tokens,
            "documents": wanted,
            "cases_per_document": 2,
        },
    )
    return PrepareResult(
        "hallucination:en",
        "CREATED",
        path,
        dict(manifest.meta.counts),
        manifest.meta.manifest_hash,
        src.revision,
    )


# --------------------------------------------------------------------------- #
# AI-Hub -> Korean hallucination manifest (optional, never downloaded)
# --------------------------------------------------------------------------- #
def prepare_aihub(cfg: AppConfig, *, force: bool, progress=None) -> PrepareResult:
    bench = cfg.benchmark.benchmarks.hallucination
    src = cfg.benchmark.dataset_sources.aihub
    path = cfg.manifest_dir / bench.manifests.ko
    if manifest_exists(path) and not force:
        manifest = read_manifest(path)
        return PrepareResult(
            "hallucination:ko",
            "EXISTS",
            path,
            dict(manifest.meta.counts),
            manifest.meta.manifest_hash,
            manifest.meta.dataset_revision,
        )

    raw_path = os.environ.get(src.env_var)
    if not raw_path:
        return PrepareResult(
            "hallucination:ko",
            "SKIPPED",
            None,
            {},
            "",
            src.revision,
            detail=(
                f"{src.env_var} is not set. AI-Hub 추상 요약 사실성 검증 데이터는 라이선스 때문에 "
                "자동으로 내려받지 않습니다. 직접 내려받아 경로를 지정하세요."
            ),
        )

    root = Path(raw_path).expanduser()
    revision = f"{src.revision}:{src.split}" if src.split else src.revision

    def skipped(detail: str) -> PrepareResult:
        return PrepareResult("hallucination:ko", "SKIPPED", None, {}, "", revision, detail=detail)

    try:
        rows = list(
            read_aihub_factuality(
                root, preferred_error_type=src.preferred_error_type, split=src.split
            )
        )
    except DatasetError as exc:
        return skipped(str(exc))

    if not rows:
        return skipped(f"no usable records under {root}")

    wanted = bench.documents.ko
    seed = cfg.benchmark.sampling.seed
    trunc = cfg.benchmark.truncation

    # Only *content* errors can produce a genuine SUPPORTED / HALLUCINATED pair.
    # An AI-Hub spelling or repetition fix leaves both summaries equally faithful,
    # so those documents are excluded rather than silently mislabelled.
    usable = [
        row
        for row in rows
        if row["erroneous_summary"] != row["corrected_summary"]
        and row["source"]
        and (row["is_factual_error"] or src.allow_non_factual_fallback)
    ]
    excluded_non_factual = len(rows) - len(usable)

    # Type 5 ("키워드 또는 중요 내용 오류") covers both wrong content and *missing*
    # content. An omission-only document would be labelled HALLUCINATED while
    # every claim it makes is still supported, so it is excluded by default.
    if src.exclude_omission_only_errors:
        kept = [row for row in usable if not row.get("is_omission_only")]
        excluded_omission_only = len(usable) - len(kept)
        usable = kept
    else:
        excluded_omission_only = 0

    if src.exclude_broken_prefix_summaries:
        kept = [
            row
            for row in usable
            if not summary_has_broken_prefix(row["erroneous_summary"], row["source"])
            and not summary_has_broken_prefix(row["corrected_summary"], row["source"])
        ]
        excluded_broken_prefix = len(usable) - len(kept)
        usable = kept
    else:
        excluded_broken_prefix = 0

    if src.require_untruncated_source:
        # A SUPPORTED label is unverifiable if the supporting sentence was cut off
        # by the source truncation budget, so keep only sources that survive it.
        fitting = [
            row
            for row in usable
            if estimate_tokens(row["source"]) <= trunc.hallucination_max_source_tokens
        ]
        truncated_out = len(usable) - len(fitting)
        usable = fitting
    else:
        truncated_out = 0

    # Preference order (spec section 7): documents whose *only* error is the
    # content-factuality type, then documents that merely include it.
    tier_only = [r for r in usable if r["only_preferred_type"]]
    tier_has = [r for r in usable if r["has_preferred_type"] and not r["only_preferred_type"]]
    tier_rest = [r for r in usable if not r["has_preferred_type"]]

    chosen: list[dict[str, Any]] = []
    for tier in (tier_only, tier_has, tier_rest):
        if len(chosen) >= wanted:
            break
        need = wanted - len(chosen)
        chosen.extend(_deterministic_take(tier, need, seed=seed, key_fn=lambda r: str(r["id"])))

    if len(chosen) < wanted:
        return skipped(
            f"only {len(chosen)} usable AI-Hub documents found under {root} "
            f"(need {wanted}); {excluded_non_factual} excluded as non-factual error types, "
            f"{excluded_omission_only} excluded as omission-only, "
            f"{excluded_broken_prefix} excluded for a broken summary prefix, "
            f"{truncated_out} excluded for exceeding {trunc.hallucination_max_source_tokens} source tokens"
        )

    chosen.sort(key=lambda r: str(r["id"]))
    records: list[ManifestRecord] = []
    for row in chosen:
        source, truncated = truncate_to_tokens(
            row["source"], trunc.hallucination_max_source_tokens, word_boundary=trunc.word_boundary
        )
        # HALLUCINATED = the annotated erroneous summary; SUPPORTED = its correction.
        for label, candidate in (
            ("HALLUCINATED", row["erroneous_summary"]),
            ("SUPPORTED", row["corrected_summary"]),
        ):
            records.append(
                ManifestRecord(
                    benchmark="hallucination",
                    language="ko",
                    split=row.get("split", "local"),
                    dataset_revision=revision,
                    sample_id=f"aihub:ko:{row['id']}:{label.lower()}",
                    source_hash=hash_text(source),
                    reference_hash=hash_text(label),
                    label=label,
                    payload={
                        "document_id": str(row["id"]),
                        "source": source,
                        "candidate_summary": candidate,
                        "gold_label": label,
                        "truncated": truncated,
                        "candidate_hash": hash_text(candidate),
                        "error_types": row["error_types"],
                        "correction_labels": row.get("correction_labels", []),
                        "error_span_kinds": row.get("error_span_kinds", []),
                        "only_preferred_type": row["only_preferred_type"],
                        "summary_kind": row.get("summary_kind", ""),
                        "document_type": row.get("document_type", ""),
                        "source_tokens_est": estimate_tokens(source),
                    },
                )
            )

    manifest = write_manifest(
        path,
        records,
        benchmark_version=bench.version,
        dataset_repo="AI-Hub 157 추상요약 사실성 검증 (local)",
        dataset_revision=revision,
        split=src.split or "local",
        seed=seed,
        preprocessing={
            "truncation_policy_version": trunc.version,
            "max_source_tokens": trunc.hallucination_max_source_tokens,
            "preferred_error_type": src.preferred_error_type,
            "factual_error_types_only": not src.allow_non_factual_fallback,
            "exclude_omission_only_errors": src.exclude_omission_only_errors,
            "exclude_broken_prefix_summaries": src.exclude_broken_prefix_summaries,
            "require_untruncated_source": src.require_untruncated_source,
            "documents": wanted,
            "cases_per_document": 2,
        },
        extra={
            "source_path": str(root),
            "candidate_documents": len(usable),
            "excluded_non_factual_error_types": excluded_non_factual,
            "excluded_omission_only_documents": excluded_omission_only,
            "excluded_broken_prefix_documents": excluded_broken_prefix,
            "excluded_oversized_sources": truncated_out,
            "only_preferred_type_documents": sum(1 for r in chosen if r["only_preferred_type"]),
            "summary_kinds": sorted({r.get("summary_kind", "") for r in chosen}),
        },
    )
    return PrepareResult(
        "hallucination:ko",
        "CREATED",
        path,
        dict(manifest.meta.counts),
        manifest.meta.manifest_hash,
        revision,
    )


# --------------------------------------------------------------------------- #
# MASSIVE -> classification manifest
# --------------------------------------------------------------------------- #
def _massive_label_space(cfg: AppConfig, rows_by_lang: dict[str, list[dict[str, Any]]], progress=None) -> list[str]:
    """Freeze the 60-intent label space, falling back to train if test is short."""
    labels = {row["intent"] for rows in rows_by_lang.values() for row in rows}
    if len(labels) >= MASSIVE_INTENT_COUNT:
        return sorted(labels)
    src = cfg.benchmark.dataset_sources.massive
    for language in ("en", "ko"):
        rel = getattr(src.files, language).replace(f"{src.split}/", "train/")
        try:
            path = download_dataset_file(src.repo, src.revision, rel, cfg.raw_data_dir, progress=progress)
        except DatasetError:
            continue
        labels.update(row["intent"] for row in read_massive(path, language=language))
    return sorted(labels)


def prepare_massive(cfg: AppConfig, *, force: bool, progress=None) -> PrepareResult:
    bench = cfg.benchmark.benchmarks.classification
    path = cfg.manifest_dir / bench.manifest
    if manifest_exists(path) and not force:
        manifest = read_manifest(path)
        return PrepareResult(
            "classification",
            "EXISTS",
            path,
            dict(manifest.meta.counts),
            manifest.meta.manifest_hash,
            manifest.meta.dataset_revision,
        )

    src = cfg.benchmark.dataset_sources.massive
    seed = cfg.benchmark.sampling.seed
    rows_by_lang: dict[str, list[dict[str, Any]]] = {}
    for language in ("en", "ko"):
        rel = getattr(src.files, language)
        file = download_dataset_file(src.repo, src.revision, rel, cfg.raw_data_dir, progress=progress)
        rows_by_lang[language] = list(read_massive(file, language=language))

    en_by_id = {row["id"]: row for row in rows_by_lang["en"]}
    ko_by_id = {row["id"]: row for row in rows_by_lang["ko"]}
    # Only ids present in BOTH locales with the SAME gold intent can support a
    # cross-lingual consistency metric.
    paired_ids = sorted(
        sid for sid in en_by_id.keys() & ko_by_id.keys() if en_by_id[sid]["intent"] == ko_by_id[sid]["intent"]
    )
    wanted = bench.cases.en
    if bench.cases.en != bench.cases.ko:
        raise DatasetError("classification requires the same EN and KO case count (paired design)")
    if len(paired_ids) < wanted:
        raise DatasetError(
            f"MASSIVE {src.split} has only {len(paired_ids)} aligned EN/KO ids, need {wanted}"
        )

    label_space = _massive_label_space(cfg, rows_by_lang, progress=progress)

    # Stratify across intents so every intent appears, then fill deterministically.
    by_intent: dict[str, list[str]] = {}
    for sid in paired_ids:
        by_intent.setdefault(en_by_id[sid]["intent"], []).append(sid)
    for intent in by_intent:
        by_intent[intent].sort(key=lambda sid: (_rank(seed, f"{intent}:{sid}"), sid))

    selected: list[str] = []
    seen: set[str] = set()
    intents_cycle = sorted(by_intent)
    round_idx = 0
    while len(selected) < wanted:
        added = 0
        for intent in intents_cycle:
            if len(selected) >= wanted:
                break
            bucket = by_intent[intent]
            if round_idx < len(bucket):
                sid = bucket[round_idx]
                if sid not in seen:
                    seen.add(sid)
                    selected.append(sid)
                    added += 1
        if added == 0:
            break
        round_idx += 1
    if len(selected) < wanted:
        raise DatasetError(f"could only select {len(selected)} MASSIVE pairs, need {wanted}")

    selected.sort(key=lambda sid: (_rank(seed, f"order:{sid}"), sid))

    records: list[ManifestRecord] = []
    for language, table in (("en", en_by_id), ("ko", ko_by_id)):
        for sid in selected:
            row = table[sid]
            records.append(
                ManifestRecord(
                    benchmark="classification",
                    language=language,
                    split=src.split,
                    dataset_revision=src.revision,
                    sample_id=f"massive:{language}:{sid}",
                    source_hash=hash_text(row["text"]),
                    reference_hash=hash_text(row["intent"]),
                    label=row["intent"],
                    payload={
                        "semantic_id": sid,
                        "text": row["text"],
                        "intent": row["intent"],
                    },
                )
            )

    manifest = write_manifest(
        path,
        records,
        benchmark_version=bench.version,
        dataset_repo=src.repo,
        dataset_revision=src.revision,
        split=src.split,
        seed=seed,
        preprocessing={
            "selection": "intent-stratified deterministic round-robin over EN/KO aligned ids",
            "batch_size": bench.batch_size,
        },
        extra={
            "label_space": label_space,
            "label_count": len(label_space),
            "paired_semantic_ids": selected,
        },
    )
    return PrepareResult(
        "classification",
        "CREATED",
        path,
        dict(manifest.meta.counts),
        manifest.meta.manifest_hash,
        src.revision,
    )


# --------------------------------------------------------------------------- #
# Judge subset manifest (frozen, derived from the XL-Sum manifest)
# --------------------------------------------------------------------------- #
def prepare_judge_subset(cfg: AppConfig, *, force: bool) -> PrepareResult:
    judge = cfg.judge.judge
    path = cfg.manifest_dir / judge.manifest
    if manifest_exists(path) and not force:
        manifest = read_manifest(path)
        return PrepareResult(
            "judge",
            "EXISTS",
            path,
            dict(manifest.meta.counts),
            manifest.meta.manifest_hash,
            manifest.meta.dataset_revision,
        )

    xlsum_path = cfg.manifest_dir / cfg.benchmark.benchmarks.summarization.manifest
    if not manifest_exists(xlsum_path):
        return PrepareResult("judge", "SKIPPED", None, {}, "", "", detail="XL-Sum manifest not prepared yet")

    xlsum = read_manifest(xlsum_path)
    seed = cfg.benchmark.sampling.seed
    records: list[ManifestRecord] = []
    for language, wanted in (("en", judge.samples.en), ("ko", judge.samples.ko)):
        pool = [r.to_dict() for r in xlsum.by_language(language)]
        chosen = _deterministic_take(pool, wanted, seed=seed + 1, key_fn=lambda r: r["sample_id"])
        chosen.sort(key=lambda r: r["sample_id"])
        for row in chosen:
            rec = ManifestRecord.from_dict(row)
            rec.benchmark = "judge"
            records.append(rec)

    manifest = write_manifest(
        path,
        records,
        benchmark_version=judge.prompt_version,
        dataset_repo=cfg.benchmark.dataset_sources.xlsum.repo,
        dataset_revision=xlsum.meta.dataset_revision,
        split=xlsum.meta.split,
        seed=seed + 1,
        preprocessing={"derived_from": str(xlsum_path.name), "source_manifest_hash": xlsum.meta.manifest_hash},
    )
    return PrepareResult(
        "judge", "CREATED", path, dict(manifest.meta.counts), manifest.meta.manifest_hash, xlsum.meta.dataset_revision
    )


def prepare_all(
    cfg: AppConfig,
    *,
    force: bool = False,
    only: Iterable[str] | None = None,
    progress=None,
) -> list[PrepareResult]:
    cfg.ensure_dirs()
    steps: list[tuple[str, Callable[[], PrepareResult]]] = [
        ("summarization", lambda: prepare_xlsum(cfg, force=force, progress=progress)),
        ("hallucination", lambda: prepare_halueval(cfg, force=force, progress=progress)),
        ("hallucination", lambda: prepare_aihub(cfg, force=force, progress=progress)),
        ("classification", lambda: prepare_massive(cfg, force=force, progress=progress)),
        ("judge", lambda: prepare_judge_subset(cfg, force=force)),
    ]
    wanted = set(only) if only else None
    results: list[PrepareResult] = []
    for name, step in steps:
        if wanted and name not in wanted:
            continue
        try:
            results.append(step())
        except (DatasetError, OSError) as exc:
            results.append(PrepareResult(name, "FAILED", None, {}, "", "", detail=str(exc)))
    return results


def load_manifests(cfg: AppConfig) -> dict[str, Manifest | None]:
    """Load every prepared manifest; missing optional ones come back as None."""
    out: dict[str, Manifest | None] = {}
    mapping = {
        "summarization": cfg.benchmark.benchmarks.summarization.manifest,
        "hallucination:en": cfg.benchmark.benchmarks.hallucination.manifests.en,
        "hallucination:ko": cfg.benchmark.benchmarks.hallucination.manifests.ko,
        "classification": cfg.benchmark.benchmarks.classification.manifest,
        "judge": cfg.judge.judge.manifest,
    }
    for key, filename in mapping.items():
        path = cfg.manifest_dir / filename
        out[key] = read_manifest(path) if manifest_exists(path) else None
    return out
