"""Persistent benchmark database (SQLite via SQLAlchemy).

The store survives process exit: a second run of the same benchmark finds every
previously successful inference here and issues zero OpenRouter calls.

``inference_attempts`` (one row per physical HTTP attempt, successes and
failures alike) is kept separate from ``inference_results`` (one row per cache
key, the reusable outcome).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(200))
    vendor: Mapped[str] = mapped_column(String(100))
    api_provider: Mapped[str] = mapped_column(String(50), default="openrouter")
    role: Mapped[str] = mapped_column(String(20), default="target")
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    routing_config_hash: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Benchmark(Base):
    __tablename__ = "benchmarks"
    __table_args__ = (UniqueConstraint("name", "version", "manifest_hash", name="uq_benchmark_ident"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), index=True)
    version: Mapped[str] = mapped_column(String(40))
    language: Mapped[str] = mapped_column(String(10), default="multi")
    dataset_revision: Mapped[str] = mapped_column(String(120), default="")
    manifest_path: Mapped[str] = mapped_column(String(400), default="")
    manifest_hash: Mapped[str] = mapped_column(String(64), default="")
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    samples: Mapped[list["Sample"]] = relationship(back_populates="benchmark")


class Sample(Base):
    __tablename__ = "samples"
    __table_args__ = (UniqueConstraint("benchmark_id", "sample_id", name="uq_sample_ident"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    benchmark_id: Mapped[int] = mapped_column(ForeignKey("benchmarks.id"), index=True)
    sample_id: Mapped[str] = mapped_column(String(200), index=True)
    language: Mapped[str] = mapped_column(String(10))
    split: Mapped[str] = mapped_column(String(40), default="test")
    dataset_revision: Mapped[str] = mapped_column(String(120), default="")
    source_hash: Mapped[str] = mapped_column(String(64), default="")
    reference_hash: Mapped[str] = mapped_column(String(64), default="")
    label: Mapped[str | None] = mapped_column(String(120), nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    benchmark: Mapped[Benchmark] = relationship(back_populates="samples")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    model_id: Mapped[str] = mapped_column(String(200), index=True)
    benchmarks: Mapped[list[str]] = mapped_column(JSON, default=list)
    languages: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(40), default="RUNNING")
    dry_run: Mapped[bool] = mapped_column(Boolean, default=False)
    budget_usd: Mapped[float] = mapped_column(Float, default=1.0)
    fresh_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    judge_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    api_calls: Mapped[int] = mapped_column(Integer, default=0)
    cache_hits: Mapped[int] = mapped_column(Integer, default=0)
    cache_misses: Mapped[int] = mapped_column(Integer, default=0)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    environment_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    git_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    benchmark_version: Mapped[str] = mapped_column(String(40), default="")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class InferenceAttempt(Base):
    """One physical OpenRouter HTTP attempt (successful or not)."""

    __tablename__ = "inference_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    cache_key: Mapped[str] = mapped_column(String(64), index=True)
    model_id: Mapped[str] = mapped_column(String(200), index=True)
    benchmark: Mapped[str] = mapped_column(String(80), default="")
    sample_id: Mapped[str] = mapped_column(String(200), default="")
    attempt_no: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(40), index=True)
    request_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    resolved_model: Mapped[str | None] = mapped_column(String(200), nullable=True)
    upstream_provider: Mapped[str | None] = mapped_column(String(120), nullable=True)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    reasoning_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    # Authoritative billed cost of THIS attempt, including failed attempts.
    usage_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


Index("ix_attempt_model_status", InferenceAttempt.model_id, InferenceAttempt.status)


class InferenceResult(Base):
    """The cacheable outcome of one benchmark case, keyed by the inference cache key."""

    __tablename__ = "inference_results"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    api_provider: Mapped[str] = mapped_column(String(40), default="openrouter")
    model_id: Mapped[str] = mapped_column(String(200), index=True)
    requested_model: Mapped[str] = mapped_column(String(200), default="")
    resolved_model: Mapped[str | None] = mapped_column(String(200), nullable=True)
    upstream_provider: Mapped[str | None] = mapped_column(String(120), nullable=True)
    routing_config_hash: Mapped[str] = mapped_column(String(64), default="")
    benchmark: Mapped[str] = mapped_column(String(80), index=True)
    benchmark_version: Mapped[str] = mapped_column(String(40), default="")
    dataset_revision: Mapped[str] = mapped_column(String(120), default="")
    split: Mapped[str] = mapped_column(String(40), default="test")
    sample_id: Mapped[str] = mapped_column(String(200), index=True)
    language: Mapped[str] = mapped_column(String(10), index=True)
    sample_content_hash: Mapped[str] = mapped_column(String(64), default="")
    status: Mapped[str] = mapped_column(String(40), index=True)
    output_text: Mapped[str] = mapped_column(Text, default="")
    finish_reason: Mapped[str | None] = mapped_column(String(60), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    reasoning_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    usage_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    usage_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    raw_response_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    key_material_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


Index("ix_result_model_benchmark_status", InferenceResult.model_id, InferenceResult.benchmark, InferenceResult.status)


class MetricResult(Base):
    """Cached evaluator output. Independent of the inference cache on purpose."""

    __tablename__ = "metric_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    metric_cache_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    inference_result_hash: Mapped[str] = mapped_column(String(64), index=True)
    model_id: Mapped[str] = mapped_column(String(200), index=True)
    benchmark: Mapped[str] = mapped_column(String(80), index=True)
    language: Mapped[str] = mapped_column(String(10), default="")
    scope: Mapped[str] = mapped_column(String(40), default="sample")  # sample | aggregate
    evaluator: Mapped[str] = mapped_column(String(80), index=True)
    evaluator_version: Mapped[str] = mapped_column(String(40))
    evaluator_config_hash: Mapped[str] = mapped_column(String(64))
    reference_hash: Mapped[str] = mapped_column(String(64), default="")
    value_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class PricingSnapshot(Base):
    __tablename__ = "pricing_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str] = mapped_column(String(200), index=True)
    input_per_million: Mapped[float | None] = mapped_column(Float, nullable=True)
    output_per_million: Mapped[float | None] = mapped_column(Float, nullable=True)
    cached_input_per_million: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    source: Mapped[str] = mapped_column(String(40), default="openrouter")
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class JudgeResult(Base):
    __tablename__ = "judge_results"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    judge_model: Mapped[str] = mapped_column(String(200), index=True)
    judge_profile: Mapped[str] = mapped_column(String(40), default="")
    target_model: Mapped[str] = mapped_column(String(200), index=True)
    benchmark: Mapped[str] = mapped_column(String(80), default="summarization")
    sample_id: Mapped[str] = mapped_column(String(200), index=True)
    language: Mapped[str] = mapped_column(String(10), default="")
    status: Mapped[str] = mapped_column(String(40), default="SUCCESS")
    self_judge: Mapped[bool] = mapped_column(Boolean, default=False)
    scores_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    usage_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_response_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    key_material_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
