"""Registration of frozen manifests into the `benchmarks` / `samples` tables.

The manifests on disk are the source of truth; these tables are the queryable
mirror, so a stored inference result can be joined back to the exact sample text,
gold label and content hashes it was produced from.
"""

from __future__ import annotations

from typing import Any

from llmbench.config.loader import AppConfig
from llmbench.datasets.manifest import Manifest
from llmbench.db.store import Store

__all__ = ["register_manifests"]


def register_manifests(
    cfg: AppConfig,
    store: Store,
    manifests: dict[str, Manifest | None],
) -> dict[str, Any]:
    """Idempotently record every prepared manifest and its samples."""
    registered: dict[str, Any] = {}
    for key, manifest in sorted(manifests.items()):
        if manifest is None:
            continue
        languages = manifest.languages()
        benchmark_row_id = store.upsert_benchmark(
            name=manifest.meta.benchmark,
            version=manifest.meta.benchmark_version,
            language=languages[0] if len(languages) == 1 else "multi",
            dataset_revision=manifest.meta.dataset_revision,
            manifest_path=str(manifest.path),
            manifest_hash=manifest.meta.manifest_hash,
            sample_count=len(manifest.records),
        )
        inserted = store.upsert_samples(
            benchmark_row_id,
            (
                {
                    "sample_id": record.sample_id,
                    "language": record.language,
                    "split": record.split,
                    "dataset_revision": record.dataset_revision,
                    "source_hash": record.source_hash,
                    "reference_hash": record.reference_hash,
                    "label": record.label,
                    "payload": record.payload,
                }
                for record in manifest.records
            ),
        )
        registered[key] = {
            "benchmark_row_id": benchmark_row_id,
            "manifest_hash": manifest.meta.manifest_hash,
            "samples_total": len(manifest.records),
            "samples_inserted": inserted,
        }
    return registered
