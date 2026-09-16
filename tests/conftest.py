"""Shared fixtures: an isolated project root with tiny frozen manifests."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llmbench.config.loader import AppConfig, load_config  # noqa: E402
from llmbench.core.canonical import hash_text  # noqa: E402
from llmbench.datasets.manifest import ManifestRecord, write_manifest  # noqa: E402
from llmbench.db.store import Store  # noqa: E402
from llmbench.openrouter.client import OpenRouterClient  # noqa: E402
from mock_openrouter import MockOpenRouter  # noqa: E402

EN_SOURCE = (
    "The transport ministry said on 12 March 2021 that rail passenger numbers rose 12% "
    "last year, the first increase since 2016. Officials in Manchester told the BBC that "
    "a further 400 carriages would enter service. The ministry added that ticket revenue "
    "reached 1.5 billion pounds over the same period, and that a review would report back "
    "in the autumn."
)
EN_REFERENCE = "Rail passenger numbers rose 12% last year, the transport ministry said."

KO_SOURCE = (
    "국토교통부는 2021년 3월 12일 지난해 철도 여객 수가 12% 증가했다고 밝혔다. 이는 2016년 이후 "
    "첫 증가다. 서울시와 국토교통부 관계자는 올해 안에 객차 400량을 추가 투입하겠다고 설명했다. "
    "같은 기간 승차권 수입은 1조 5천억 원에 이르렀다."
)
KO_REFERENCE = "국토교통부는 지난해 철도 여객 수가 12% 증가했다고 밝혔다."

INTENTS = ["alarm_set", "audio_volume_mute", "iot_hue_lightchange", "weather_query"]


def _write_summarization_manifest(cfg: AppConfig, n_per_language: int = 3) -> None:
    records = []
    for language, source, reference in (("en", EN_SOURCE, EN_REFERENCE), ("ko", KO_SOURCE, KO_REFERENCE)):
        for i in range(n_per_language):
            text = f"{source} Item {i}."
            records.append(
                ManifestRecord(
                    benchmark="summarization",
                    language=language,
                    split="test",
                    dataset_revision="test-rev-1",
                    sample_id=f"xlsum:{language}:{i:03d}",
                    source_hash=hash_text(text),
                    reference_hash=hash_text(reference),
                    payload={
                        "source": text,
                        "reference": reference,
                        "title": "t",
                        "url": "",
                        "truncated": False,
                        "raw_source_hash": hash_text(text),
                        "source_tokens_est": 120,
                        "reference_tokens_est": 20,
                    },
                )
            )
    write_manifest(
        cfg.manifest_dir / cfg.benchmark.benchmarks.summarization.manifest,
        records,
        benchmark_version="1.0.0",
        dataset_repo="test/xlsum",
        dataset_revision="test-rev-1",
        split="test",
        seed=cfg.benchmark.sampling.seed,
    )


def _write_halueval_manifest(cfg: AppConfig, n_documents: int = 3) -> None:
    records = []
    for i in range(n_documents):
        source = f"{EN_SOURCE} Document {i}."
        for label, candidate in (
            ("SUPPORTED", "Rail passenger numbers rose 12% last year."),
            ("HALLUCINATED", "Rail passenger numbers fell 40% last year."),
        ):
            records.append(
                ManifestRecord(
                    benchmark="hallucination",
                    language="en",
                    split="data",
                    dataset_revision="halueval-rev-1",
                    sample_id=f"halueval:en:{i:03d}:{label.lower()}",
                    source_hash=hash_text(source),
                    reference_hash=hash_text(label),
                    label=label,
                    payload={
                        "document_id": f"{i:03d}",
                        "source": source,
                        "candidate_summary": candidate,
                        "gold_label": label,
                        "truncated": False,
                        "candidate_hash": hash_text(candidate),
                        "source_tokens_est": 120,
                    },
                )
            )
    write_manifest(
        cfg.manifest_dir / cfg.benchmark.benchmarks.hallucination.manifests.en,
        records,
        benchmark_version="1.0.0",
        dataset_repo="test/halueval",
        dataset_revision="halueval-rev-1",
        split="data",
        seed=cfg.benchmark.sampling.seed,
    )


def _write_aihub_manifest(cfg: AppConfig, n_documents: int = 3) -> None:
    """Korean hallucination manifest in the shape `prepare_aihub` produces."""
    records = []
    for i in range(n_documents):
        source = f"{KO_SOURCE} 문서 {i}."
        for label, candidate in (
            ("HALLUCINATED", "국토교통부는 지난해 철도 여객 수가 40% 감소했다고 밝혔다."),
            ("SUPPORTED", "국토교통부는 지난해 철도 여객 수가 12% 증가했다고 밝혔다."),
        ):
            records.append(
                ManifestRecord(
                    benchmark="hallucination",
                    language="ko",
                    split="Validation",
                    dataset_revision="local:Validation",
                    sample_id=f"aihub:ko:Validation/SM{i:08d}:{label.lower()}",
                    source_hash=hash_text(source),
                    reference_hash=hash_text(label),
                    label=label,
                    payload={
                        "document_id": f"Validation/SM{i:08d}",
                        "source": source,
                        "candidate_summary": candidate,
                        "gold_label": label,
                        "truncated": False,
                        "candidate_hash": hash_text(candidate),
                        "error_types": ["type5"],
                        "correction_labels": ["[5] 키워드 또는 중요 내용 오류"],
                        "error_span_kinds": ["alteration"],
                        "only_preferred_type": True,
                        "summary_kind": "machine",
                        "document_type": "사실전달형",
                        "source_tokens_est": 120,
                    },
                )
            )
    write_manifest(
        cfg.manifest_dir / cfg.benchmark.benchmarks.hallucination.manifests.ko,
        records,
        benchmark_version="1.0.0",
        dataset_repo="AI-Hub 157 추상요약 사실성 검증 (local)",
        dataset_revision="local:Validation",
        split="Validation",
        seed=cfg.benchmark.sampling.seed,
    )


def _write_massive_manifest(cfg: AppConfig, n_pairs: int = 8) -> None:
    records = []
    for language, template in (("en", "wake me up at {i} am"), ("ko", "{i}시에 깨워줘")):
        for i in range(n_pairs):
            intent = INTENTS[i % len(INTENTS)]
            text = template.format(i=i + 1)
            records.append(
                ManifestRecord(
                    benchmark="classification",
                    language=language,
                    split="test",
                    dataset_revision="massive-rev-1",
                    sample_id=f"massive:{language}:{i}",
                    source_hash=hash_text(text),
                    reference_hash=hash_text(intent),
                    label=intent,
                    payload={"semantic_id": str(i), "text": text, "intent": intent},
                )
            )
    write_manifest(
        cfg.manifest_dir / cfg.benchmark.benchmarks.classification.manifest,
        records,
        benchmark_version="1.0.0",
        dataset_repo="test/massive",
        dataset_revision="massive-rev-1",
        split="test",
        seed=cfg.benchmark.sampling.seed,
        extra={"label_space": INTENTS, "label_count": len(INTENTS)},
    )


def _write_judge_manifest(cfg: AppConfig) -> None:
    from llmbench.datasets.manifest import read_manifest

    xlsum = read_manifest(cfg.manifest_dir / cfg.benchmark.benchmarks.summarization.manifest)
    records = []
    for record in xlsum.records[:2]:
        clone = ManifestRecord.from_dict(record.to_dict())
        clone.benchmark = "judge"
        records.append(clone)
    write_manifest(
        cfg.manifest_dir / cfg.judge.judge.manifest,
        records,
        benchmark_version="judge-1",
        dataset_repo="test/xlsum",
        dataset_revision="test-rev-1",
        split="test",
        seed=cfg.benchmark.sampling.seed + 1,
    )


@pytest.fixture
def project(tmp_path: Path) -> AppConfig:
    """An isolated copy of config/ with an empty data/ and results/ tree."""
    config_dir = tmp_path / "config"
    shutil.copytree(PROJECT_ROOT / "config", config_dir)
    cfg = load_config(config_dir, root=tmp_path)
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def prepared(project: AppConfig) -> AppConfig:
    """Project fixture with small frozen manifests already written."""
    _write_summarization_manifest(project)
    _write_halueval_manifest(project)
    _write_massive_manifest(project)
    _write_judge_manifest(project)
    return project


@pytest.fixture
def prepared_with_ko(prepared: AppConfig) -> AppConfig:
    """Adds the optional Korean hallucination manifest (AI-Hub data present)."""
    _write_aihub_manifest(prepared)
    return prepared


@pytest.fixture
def store(prepared: AppConfig) -> Store:
    store = Store(prepared.cache_db_path)
    yield store
    store.close()


@pytest.fixture
def mock_openrouter() -> MockOpenRouter:
    return MockOpenRouter()


@pytest.fixture
def client_factory(mock_openrouter: MockOpenRouter):
    def factory(**kwargs: Any) -> OpenRouterClient:
        params = {
            "api_key": "test-key",
            "transport": mock_openrouter.transport(),
            "max_attempts": 3,
            "initial_backoff": 0.0,
            "jitter": 0.0,
            "sleep": _no_sleep,
        }
        params.update(kwargs)
        return OpenRouterClient(**params)

    return factory


async def _no_sleep(_seconds: float) -> None:
    return None
