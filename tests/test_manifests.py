"""Frozen-manifest guarantees."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from llmbench.core.errors import ManifestError
from llmbench.datasets.manifest import compute_manifest_hash, read_manifest, write_manifest
from llmbench.datasets.prepare import _deterministic_take, prepare_all

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"


def test_deterministic_take_is_order_independent():
    rows = [{"id": f"item-{i}"} for i in range(200)]
    shuffled = rows[:]
    random.Random(7).shuffle(shuffled)
    a = _deterministic_take(rows, 20, seed=42, key_fn=lambda r: r["id"])
    b = _deterministic_take(shuffled, 20, seed=42, key_fn=lambda r: r["id"])
    assert a == b


def test_deterministic_take_depends_on_seed():
    rows = [{"id": f"item-{i}"} for i in range(200)]
    a = _deterministic_take(rows, 20, seed=1, key_fn=lambda r: r["id"])
    b = _deterministic_take(rows, 20, seed=2, key_fn=lambda r: r["id"])
    assert a != b


def test_manifest_roundtrip_and_hash(prepared):
    path = prepared.manifest_dir / prepared.benchmark.benchmarks.summarization.manifest
    manifest = read_manifest(path)
    assert manifest.meta.manifest_hash == compute_manifest_hash(manifest.records)
    assert manifest.meta.counts["total"] == len(manifest.records)


def test_tampered_manifest_is_rejected(prepared):
    """A frozen manifest that no longer matches its recorded hash must not load."""
    path = prepared.manifest_dir / prepared.benchmark.benchmarks.summarization.manifest
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["payload"]["source"] += " tampered"
    lines[0] = json.dumps(row, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="manifest hash mismatch"):
        read_manifest(path)


def test_rerunning_prepare_does_not_resample(prepared):
    """(10) Existing manifests are reused, never re-sampled."""
    before = {
        name: read_manifest(prepared.manifest_dir / name).meta.manifest_hash
        for name in (
            prepared.benchmark.benchmarks.summarization.manifest,
            prepared.benchmark.benchmarks.hallucination.manifests.en,
            prepared.benchmark.benchmarks.classification.manifest,
        )
    }

    results = prepare_all(prepared, force=False)
    statuses = {r.name: r.status for r in results}
    assert statuses["summarization"] == "EXISTS"
    assert statuses["classification"] == "EXISTS"

    after = {
        name: read_manifest(prepared.manifest_dir / name).meta.manifest_hash
        for name in before
    }
    assert before == after


def test_massive_en_ko_semantic_ids_align(prepared):
    """(9) Every MASSIVE case exists in both languages with the same gold intent."""
    manifest = read_manifest(prepared.manifest_dir / prepared.benchmark.benchmarks.classification.manifest)
    en = {r.payload["semantic_id"]: r for r in manifest.by_language("en")}
    ko = {r.payload["semantic_id"]: r for r in manifest.by_language("ko")}

    assert en.keys() == ko.keys()
    assert len(en) == len(ko)
    for semantic_id, record in en.items():
        assert record.label == ko[semantic_id].label
        assert record.payload["text"] != ko[semantic_id].payload["text"]


@pytest.mark.skipif(
    not (REAL_MANIFEST_DIR / "massive_v1.jsonl").is_file(),
    reason="real manifests not prepared (run `benchmark prepare`)",
)
def test_real_manifests_are_consistent():
    """The manifests actually shipped by `benchmark prepare` satisfy the spec."""
    xlsum = read_manifest(REAL_MANIFEST_DIR / "xlsum_v1.jsonl")
    assert xlsum.meta.counts == {"en": 200, "ko": 200, "total": 400}
    assert max(r.payload["source_tokens_est"] for r in xlsum.records) <= 1800

    halueval = read_manifest(REAL_MANIFEST_DIR / "halueval_summary_v1.jsonl")
    assert halueval.meta.counts["total"] == 400
    labels = [r.label for r in halueval.records]
    assert labels.count("SUPPORTED") == 200
    assert labels.count("HALLUCINATED") == 200

    massive = read_manifest(REAL_MANIFEST_DIR / "massive_v1.jsonl")
    assert massive.meta.counts == {"en": 1000, "ko": 1000, "total": 2000}
    assert massive.meta.extra["label_count"] == 60
    en = {r.payload["semantic_id"]: r.label for r in massive.by_language("en")}
    ko = {r.payload["semantic_id"]: r.label for r in massive.by_language("ko")}
    assert en == ko
