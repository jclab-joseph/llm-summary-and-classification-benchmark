"""AI-Hub 추상요약 사실성 검증 reader and manifest construction.

Uses synthetic fixtures in the exact shape AI-Hub distributes (a .zip of one
JSON per document) plus the documented hand-converted format, so nothing here
depends on the licence-gated data being present.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from llmbench.core.errors import DatasetError
from llmbench.datasets.manifest import read_manifest
from llmbench.datasets.prepare import prepare_aihub
from llmbench.datasets.sources import (
    AIHUB_ERROR_TYPES,
    read_aihub_factuality,
    summary_has_broken_prefix,
)

SOURCE = (
    "도널드 트럼프 대통령과 시진핑 주석이 1일 정상회담에서 무역전쟁 휴전을 선언했다. "
    "양국은 90일간 추가 관세를 부과하지 않기로 합의했다. 시장은 안도의 한숨을 내쉬었다. "
    "다만 중국 제조 2025를 둘러싼 이견은 그대로 남아 있다는 분석이 나온다."
)


def _doc(
    *,
    target: str = "machine_abstractive_summary",
    erroneous: str,
    corrections: dict[str, tuple[str, list[tuple[str, str]]]],
    source: str = SOURCE,
) -> dict:
    corrected = {f"corrected_type{i}": None for i in range(1, 7)}
    for type_key, (text, spans) in corrections.items():
        corrected[type_key] = {
            "correction_type": AIHUB_ERROR_TYPES[type_key.removeprefix("corrected_")],
            "text": text,
            "errors": [
                {"error_id": f"{i}", "begin": 0, "end": 1, "sub": sub, "type": "x", "correction": cor}
                for i, (sub, cor) in enumerate(spans)
            ],
        }
    corrected["corrected_all"] = {"corrected_type": "all", "text": next(iter(corrections.values()))[0]}
    return {
        "annotation": {
            "target": target,
            "type": "사실전달형",
            "original_summary": {target: {"text": erroneous, "author": {"type": "machine"}}},
            "corrected_summary": corrected,
        },
        "original_text": source,
    }


DOCS = {
    # Genuine content error: the summary states a wrong number.
    "SM00000001": _doc(
        erroneous="트럼프 대통령과 시진핑 주석이 1일 정상회담에서 180일간 관세 유예에 합의했다.",
        corrections={"corrected_type5": ("트럼프 대통령과 시진핑 주석이 1일 정상회담에서 90일간 관세 유예에 합의했다.", [("180일간", "90일간")])},
    ),
    # Omission only: the correction merely inserts the missing clause.
    "SM00000002": _doc(
        erroneous="트럼프 대통령과 시진핑 주석이 정상회담에서 합의했다.",
        corrections={"corrected_type5": ("트럼프 대통령과 시진핑 주석이 1일 무역전쟁 휴전 정상회담에서 합의했다.", [("주석이 정상회담에서", "주석이 1일 무역전쟁 휴전 정상회담에서")])},
    ),
    # Non-factual error type: a spelling fix cannot make a summary unsupported.
    "SM00000003": _doc(
        erroneous="트럼프 대통령과 시진핑 주석이 1일 정상회담에서 90일간 관세 유예에 합의햇다.",
        corrections={"corrected_type1": ("트럼프 대통령과 시진핑 주석이 1일 정상회담에서 90일간 관세 유예에 합의했다.", [("합의햇다", "합의했다")])},
    ),
    # Content error, but the summary lost its first syllable upstream.
    "SM00000004": _doc(
        erroneous="럼프 대통령과 시진핑 주석이 1일 정상회담에서 180일간 관세 유예에 합의했다.",
        corrections={"corrected_type5": ("럼프 대통령과 시진핑 주석이 1일 정상회담에서 90일간 관세 유예에 합의했다.", [("180일간", "90일간")])},
    ),
}


@pytest.fixture
def aihub_root(tmp_path: Path) -> Path:
    """A directory shaped like the real AI-Hub distribution."""
    labels = tmp_path / "157-추상요약사실성검증데이터" / "01-1.정식개방데이터" / "Validation" / "02.라벨링데이터"
    labels.mkdir(parents=True)
    archive = labels / "VL_기계요약문_사실전달형_corrected_type5.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for doc_id, payload in DOCS.items():
            zf.writestr(f"/{doc_id}.json", json.dumps(payload, ensure_ascii=False))
    # A 원천데이터 archive with no corrections must be skipped silently.
    sources = tmp_path / "157-추상요약사실성검증데이터" / "01-1.정식개방데이터" / "Validation" / "01.원천데이터"
    sources.mkdir(parents=True)
    with zipfile.ZipFile(sources / "VS_기계요약문_사실전달형.zip", "w") as zf:
        zf.writestr("/SM00000001.json", json.dumps({"original_text": SOURCE}, ensure_ascii=False))
    return tmp_path


def read_all(root: Path, **kwargs):
    return {r["id"]: r for r in read_aihub_factuality(root, **kwargs)}


def test_reads_native_zip_layout(aihub_root):
    rows = read_all(aihub_root, preferred_error_type="type5", split="Validation")

    assert set(rows) == {f"Validation/{doc}" for doc in DOCS}
    row = rows["Validation/SM00000001"]
    assert row["schema"] == "aihub-157"
    assert row["error_types"] == ["type5"]
    assert row["is_factual_error"] is True
    assert row["only_preferred_type"] is True
    assert row["summary_kind"] == "machine"
    assert "180일간" in row["erroneous_summary"]
    assert "90일간" in row["corrected_summary"]
    assert row["source"] == " ".join(SOURCE.split())


def test_classifies_error_spans(aihub_root):
    rows = read_all(aihub_root, preferred_error_type="type5", split="Validation")

    assert rows["Validation/SM00000001"]["error_span_kinds"] == ["alteration"]
    assert rows["Validation/SM00000001"]["is_omission_only"] is False
    # Inserted in the middle of the span, so a substring test would miss it.
    assert rows["Validation/SM00000002"]["error_span_kinds"] == ["omission"]
    assert rows["Validation/SM00000002"]["is_omission_only"] is True


def test_non_factual_error_types_are_flagged(aihub_root):
    rows = read_all(aihub_root, preferred_error_type="type5", split="Validation")
    spelling = rows["Validation/SM00000003"]
    assert spelling["error_types"] == ["type1"]
    assert spelling["is_factual_error"] is False
    assert spelling["has_preferred_type"] is False


def test_split_filter(aihub_root):
    assert read_all(aihub_root, split="Validation")
    with pytest.raises(DatasetError, match="no .zip/.json/.jsonl data found"):
        read_all(aihub_root, split="Training")


def test_missing_path_raises(tmp_path):
    with pytest.raises(DatasetError, match="does not exist"):
        read_all(tmp_path / "nope")


def test_generic_converted_format_is_still_supported(tmp_path):
    path = tmp_path / "converted.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                {
                    "id": "doc-1",
                    "source": SOURCE,
                    "erroneous_summary": "180일간 관세 유예에 합의했다.",
                    "corrected_summary": "90일간 관세 유예에 합의했다.",
                    "error_type": "type5",
                },
                {"id": "doc-2", "source": SOURCE},  # incomplete -> skipped
            )
        ),
        encoding="utf-8",
    )
    rows = read_all(tmp_path)
    assert set(rows) == {"doc-1"}
    assert rows["doc-1"]["schema"] == "generic"
    assert rows["doc-1"]["is_factual_error"] is True


@pytest.mark.parametrize(
    "summary,expected",
    [
        ("럼프 대통령과 시진핑 주석", True),
        ("트럼프 대통령과 시진핑 주석", False),
        ("시장은 안도의 한숨을 내쉬었다", False),
        ("짧음", False),
    ],
)
def test_broken_prefix_detection(summary, expected):
    assert summary_has_broken_prefix(summary, SOURCE) is expected


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #
def _prepare(project, aihub_root, monkeypatch, documents=1, **overrides):
    monkeypatch.setenv("AIHUB_FACTUALITY_DATA_PATH", str(aihub_root))
    project.benchmark.benchmarks.hallucination.documents.ko = documents
    for key, value in overrides.items():
        setattr(project.benchmark.dataset_sources.aihub, key, value)
    return prepare_aihub(project, force=True)


def test_prepare_builds_supported_hallucinated_pairs(project, aihub_root, monkeypatch):
    result = _prepare(project, aihub_root, monkeypatch, documents=1)

    assert result.status == "CREATED"
    assert result.counts == {"ko": 2, "total": 2}
    assert result.dataset_revision == "local:Validation"

    manifest = read_manifest(result.path)
    labels = {r.label: r for r in manifest.records}
    assert set(labels) == {"SUPPORTED", "HALLUCINATED"}
    assert "180일간" in labels["HALLUCINATED"].payload["candidate_summary"]
    assert "90일간" in labels["SUPPORTED"].payload["candidate_summary"]
    # Both cases share one source document.
    assert len({r.source_hash for r in manifest.records}) == 1
    assert manifest.meta.preprocessing["preferred_error_type"] == "type5"


def test_prepare_excludes_unusable_documents(project, aihub_root, monkeypatch):
    """Only the one genuinely-factual, well-formed document survives the filters."""
    result = _prepare(project, aihub_root, monkeypatch, documents=1)
    extra = read_manifest(result.path).meta.extra

    assert extra["candidate_documents"] == 1
    assert extra["excluded_non_factual_error_types"] == 1  # the type1 spelling fix
    assert extra["excluded_omission_only_documents"] == 1
    assert extra["excluded_broken_prefix_documents"] == 1


def test_prepare_skips_when_not_enough_documents(project, aihub_root, monkeypatch):
    result = _prepare(project, aihub_root, monkeypatch, documents=200)
    assert result.status == "SKIPPED"
    assert "only 1 usable AI-Hub documents" in result.detail
    assert "excluded as non-factual error types" in result.detail


def test_prepare_skips_without_env_var(project, monkeypatch):
    monkeypatch.delenv("AIHUB_FACTUALITY_DATA_PATH", raising=False)
    result = prepare_aihub(project, force=True)
    assert result.status == "SKIPPED"
    assert "AIHUB_FACTUALITY_DATA_PATH" in result.detail


def test_prepare_is_deterministic(project, aihub_root, monkeypatch):
    first = _prepare(project, aihub_root, monkeypatch, documents=1)
    second = _prepare(project, aihub_root, monkeypatch, documents=1)
    assert first.manifest_hash == second.manifest_hash


def test_filters_can_be_disabled(project, aihub_root, monkeypatch):
    result = _prepare(
        project,
        aihub_root,
        monkeypatch,
        documents=4,
        allow_non_factual_fallback=True,
        exclude_omission_only_errors=False,
        exclude_broken_prefix_summaries=False,
    )
    assert result.status == "CREATED"
    assert result.counts["total"] == 8
