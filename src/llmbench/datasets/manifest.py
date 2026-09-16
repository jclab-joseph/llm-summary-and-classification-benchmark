"""Frozen benchmark manifests.

A manifest is the contract that makes cross-model comparison fair: the exact
sample IDs, the exact canonical (already truncated) text, and a hash over the
whole thing. Once written it is never re-sampled -- only `prepare --force`
regenerates it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llmbench.core.canonical import canonical_json, hash_obj
from llmbench.core.errors import ManifestError
from llmbench.core.reproducibility import utc_now_iso

__all__ = [
    "ManifestRecord",
    "ManifestMeta",
    "Manifest",
    "read_manifest",
    "write_manifest",
    "manifest_exists",
]


@dataclass(slots=True)
class ManifestRecord:
    """One frozen benchmark case."""

    benchmark: str
    language: str
    split: str
    dataset_revision: str
    sample_id: str
    source_hash: str
    reference_hash: str
    payload: dict[str, Any] = field(default_factory=dict)
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManifestRecord:
        return cls(
            benchmark=data["benchmark"],
            language=data["language"],
            split=data["split"],
            dataset_revision=data["dataset_revision"],
            sample_id=data["sample_id"],
            source_hash=data["source_hash"],
            reference_hash=data["reference_hash"],
            payload=data.get("payload", {}),
            label=data.get("label"),
        )


@dataclass(slots=True)
class ManifestMeta:
    benchmark: str
    benchmark_version: str
    dataset_revision: str
    dataset_repo: str
    split: str
    seed: int
    counts: dict[str, int]
    manifest_hash: str
    created_at: str
    preprocessing: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Manifest:
    path: Path
    records: list[ManifestRecord]
    meta: ManifestMeta

    def __len__(self) -> int:
        return len(self.records)

    def by_language(self, language: str | None) -> list[ManifestRecord]:
        if language in (None, "all", "both"):
            return list(self.records)
        return [r for r in self.records if r.language == language]

    def languages(self) -> list[str]:
        return sorted({r.language for r in self.records})

    def get(self, sample_id: str) -> ManifestRecord | None:
        for record in self.records:
            if record.sample_id == sample_id:
                return record
        return None

    def index(self) -> dict[str, ManifestRecord]:
        return {r.sample_id: r for r in self.records}


def compute_manifest_hash(records: list[ManifestRecord]) -> str:
    """Order-sensitive hash of the full record list (payload included)."""
    return hash_obj([r.to_dict() for r in records])


def meta_path_for(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def manifest_exists(path: Path) -> bool:
    return path.is_file() and meta_path_for(path).is_file()


def write_manifest(
    path: Path,
    records: list[ManifestRecord],
    *,
    benchmark_version: str,
    dataset_repo: str,
    dataset_revision: str,
    split: str,
    seed: int,
    preprocessing: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Manifest:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_hash = compute_manifest_hash(records)
    counts: dict[str, int] = {}
    for record in records:
        counts[record.language] = counts.get(record.language, 0) + 1
    counts["total"] = len(records)

    benchmark = records[0].benchmark if records else path.stem
    meta = ManifestMeta(
        benchmark=benchmark,
        benchmark_version=benchmark_version,
        dataset_revision=dataset_revision,
        dataset_repo=dataset_repo,
        split=split,
        seed=seed,
        counts=counts,
        manifest_hash=manifest_hash,
        created_at=utc_now_iso(),
        preprocessing=preprocessing or {},
        extra=extra or {},
    )

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(canonical_json(record.to_dict()) + "\n")
    tmp.replace(path)

    meta_path_for(path).write_text(
        json.dumps(meta.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return Manifest(path=path, records=records, meta=meta)


def read_manifest(path: Path) -> Manifest:
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}. Run `benchmark prepare` first.")
    meta_path = meta_path_for(path)
    if not meta_path.is_file():
        raise ManifestError(f"manifest metadata not found: {meta_path}. Run `benchmark prepare --force`.")

    records: list[ManifestRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(ManifestRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ManifestError(f"{path}:{line_no} is not a valid manifest record: {exc}") from exc

    meta = ManifestMeta(**json.loads(meta_path.read_text(encoding="utf-8")))
    actual = compute_manifest_hash(records)
    if actual != meta.manifest_hash:
        raise ManifestError(
            f"manifest hash mismatch for {path}: recorded {meta.manifest_hash[:12]}, "
            f"computed {actual[:12]}. The frozen manifest has been modified."
        )
    return Manifest(path=path, records=records, meta=meta)
