"""Thin persistence facade over the SQLite benchmark database."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from llmbench.db.schema import (
    Base,
    Benchmark,
    InferenceAttempt,
    InferenceResult,
    JudgeResult,
    MetricResult,
    Model,
    PricingSnapshot,
    Run,
    Sample,
)

__all__ = ["Store"]


class Store:
    """Owns the SQLAlchemy engine and exposes the queries the benchmark needs."""

    def __init__(self, db_path: str | Path, *, echo: bool = False) -> None:
        self.db_path = Path(db_path)
        if str(db_path) == ":memory:":
            url = "sqlite+pysqlite:///:memory:"
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite+pysqlite:///{self.db_path}"
        self.engine = create_engine(url, echo=echo, future=True)
        # WAL keeps concurrent readers cheap and survives process exit.
        if str(db_path) != ":memory:":
            with self.engine.connect() as conn:
                conn.exec_driver_sql("PRAGMA journal_mode=WAL")
                conn.exec_driver_sql("PRAGMA synchronous=NORMAL")
                conn.commit()
        self._sessionmaker = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        sess = self._sessionmaker()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    def close(self) -> None:
        self.engine.dispose()

    # ------------------------------------------------------------------ #
    # models / benchmarks / samples
    # ------------------------------------------------------------------ #
    def upsert_model(
        self,
        *,
        model_id: str,
        display_name: str,
        vendor: str,
        role: str,
        config: dict[str, Any],
        routing_config_hash: str,
    ) -> None:
        with self.session() as sess:
            row = sess.scalar(select(Model).where(Model.model_id == model_id))
            if row is None:
                row = Model(model_id=model_id)
                sess.add(row)
            row.display_name = display_name
            row.vendor = vendor
            row.role = role
            row.api_provider = "openrouter"
            row.config_json = config
            row.routing_config_hash = routing_config_hash

    def upsert_benchmark(
        self,
        *,
        name: str,
        version: str,
        language: str,
        dataset_revision: str,
        manifest_path: str,
        manifest_hash: str,
        sample_count: int,
    ) -> int:
        with self.session() as sess:
            row = sess.scalar(
                select(Benchmark).where(
                    Benchmark.name == name,
                    Benchmark.version == version,
                    Benchmark.manifest_hash == manifest_hash,
                )
            )
            if row is None:
                row = Benchmark(name=name, version=version, manifest_hash=manifest_hash)
                sess.add(row)
            row.language = language
            row.dataset_revision = dataset_revision
            row.manifest_path = manifest_path
            row.sample_count = sample_count
            sess.flush()
            return row.id

    def upsert_samples(self, benchmark_row_id: int, samples: Iterable[dict[str, Any]]) -> int:
        count = 0
        with self.session() as sess:
            existing = {
                sid
                for (sid,) in sess.execute(
                    select(Sample.sample_id).where(Sample.benchmark_id == benchmark_row_id)
                )
            }
            for sample in samples:
                if sample["sample_id"] in existing:
                    continue
                sess.add(
                    Sample(
                        benchmark_id=benchmark_row_id,
                        sample_id=sample["sample_id"],
                        language=sample.get("language", ""),
                        split=sample.get("split", "test"),
                        dataset_revision=sample.get("dataset_revision", ""),
                        source_hash=sample.get("source_hash", ""),
                        reference_hash=sample.get("reference_hash", ""),
                        label=sample.get("label"),
                        payload_json=sample.get("payload", {}),
                    )
                )
                count += 1
        return count

    # ------------------------------------------------------------------ #
    # runs
    # ------------------------------------------------------------------ #
    def create_run(self, **kwargs: Any) -> None:
        with self.session() as sess:
            sess.add(Run(**kwargs))

    def finish_run(self, run_id: str, **updates: Any) -> None:
        with self.session() as sess:
            row = sess.scalar(select(Run).where(Run.run_id == run_id))
            if row is None:
                return
            for key, value in updates.items():
                setattr(row, key, value)
            if row.finished_at is None:
                row.finished_at = datetime.now(UTC)

    def get_run(self, run_id: str) -> Run | None:
        with self.session() as sess:
            return sess.scalar(select(Run).where(Run.run_id == run_id))

    def latest_run(self, model_id: str) -> Run | None:
        with self.session() as sess:
            return sess.scalar(
                select(Run).where(Run.model_id == model_id).order_by(Run.started_at.desc()).limit(1)
            )

    # ------------------------------------------------------------------ #
    # inference cache
    # ------------------------------------------------------------------ #
    def get_result(self, cache_key: str) -> InferenceResult | None:
        with self.session() as sess:
            return sess.get(InferenceResult, cache_key)

    def get_successful_result(self, cache_key: str) -> InferenceResult | None:
        row = self.get_result(cache_key)
        if row is not None and row.status == "SUCCESS":
            return row
        return None

    def existing_success_keys(self, cache_keys: Iterable[str]) -> set[str]:
        keys = list(cache_keys)
        found: set[str] = set()
        with self.session() as sess:
            for i in range(0, len(keys), 500):
                chunk = keys[i : i + 500]
                rows = sess.execute(
                    select(InferenceResult.cache_key).where(
                        InferenceResult.cache_key.in_(chunk),
                        InferenceResult.status == "SUCCESS",
                    )
                )
                found.update(k for (k,) in rows)
        return found

    def save_result(self, payload: dict[str, Any]) -> None:
        with self.session() as sess:
            row = sess.get(InferenceResult, payload["cache_key"])
            if row is None:
                row = InferenceResult(cache_key=payload["cache_key"])
                sess.add(row)
            elif row.status == "SUCCESS" and payload.get("status") != "SUCCESS":
                # Never let a later failure clobber a cached success.
                return
            for key, value in payload.items():
                setattr(row, key, value)

    def save_attempts(self, rows: Iterable[dict[str, Any]]) -> None:
        with self.session() as sess:
            for row in rows:
                sess.add(InferenceAttempt(**row))

    def results_for(
        self,
        model_id: str,
        benchmark: str | None = None,
        *,
        statuses: Iterable[str] | None = None,
    ) -> list[InferenceResult]:
        with self.session() as sess:
            stmt = select(InferenceResult).where(InferenceResult.model_id == model_id)
            if benchmark:
                stmt = stmt.where(InferenceResult.benchmark == benchmark)
            if statuses:
                stmt = stmt.where(InferenceResult.status.in_(list(statuses)))
            return list(sess.scalars(stmt))

    def delete_results(self, cache_keys: Iterable[str]) -> int:
        keys = list(cache_keys)
        if not keys:
            return 0
        with self.session() as sess:
            result = sess.execute(delete(InferenceResult).where(InferenceResult.cache_key.in_(keys)))
            return int(result.rowcount or 0)

    def failed_keys(self, model_id: str, benchmark: str | None = None) -> set[str]:
        with self.session() as sess:
            stmt = select(InferenceResult.cache_key).where(
                InferenceResult.model_id == model_id,
                InferenceResult.status != "SUCCESS",
            )
            if benchmark:
                stmt = stmt.where(InferenceResult.benchmark == benchmark)
            return {k for (k,) in sess.execute(stmt)}

    # ------------------------------------------------------------------ #
    # metric cache
    # ------------------------------------------------------------------ #
    def get_metric(self, metric_cache_key: str) -> MetricResult | None:
        with self.session() as sess:
            return sess.scalar(
                select(MetricResult).where(MetricResult.metric_cache_key == metric_cache_key)
            )

    def save_metric(self, payload: dict[str, Any]) -> None:
        with self.session() as sess:
            row = sess.scalar(
                select(MetricResult).where(MetricResult.metric_cache_key == payload["metric_cache_key"])
            )
            if row is None:
                row = MetricResult(metric_cache_key=payload["metric_cache_key"])
                sess.add(row)
            for key, value in payload.items():
                setattr(row, key, value)

    # ------------------------------------------------------------------ #
    # pricing
    # ------------------------------------------------------------------ #
    def save_pricing_snapshot(self, payload: dict[str, Any]) -> None:
        with self.session() as sess:
            sess.add(PricingSnapshot(**payload))

    def latest_pricing(self, model_id: str) -> PricingSnapshot | None:
        with self.session() as sess:
            return sess.scalar(
                select(PricingSnapshot)
                .where(PricingSnapshot.model_id == model_id)
                .order_by(PricingSnapshot.retrieved_at.desc(), PricingSnapshot.id.desc())
                .limit(1)
            )

    # ------------------------------------------------------------------ #
    # judge
    # ------------------------------------------------------------------ #
    def get_judge_result(self, cache_key: str) -> JudgeResult | None:
        with self.session() as sess:
            return sess.get(JudgeResult, cache_key)

    def save_judge_result(self, payload: dict[str, Any]) -> None:
        with self.session() as sess:
            row = sess.get(JudgeResult, payload["cache_key"])
            if row is None:
                row = JudgeResult(cache_key=payload["cache_key"])
                sess.add(row)
            for key, value in payload.items():
                setattr(row, key, value)

    def judge_results_for(self, target_model: str, judge_model: str | None = None) -> list[JudgeResult]:
        with self.session() as sess:
            stmt = select(JudgeResult).where(JudgeResult.target_model == target_model)
            if judge_model:
                stmt = stmt.where(JudgeResult.judge_model == judge_model)
            return list(sess.scalars(stmt))

    # ------------------------------------------------------------------ #
    # aggregates for reporting / `cache stats`
    # ------------------------------------------------------------------ #
    def cost_by_benchmark(self, model_id: str) -> dict[str, float]:
        """Sum of the authoritative usage.cost, grouped by benchmark."""
        with self.session() as sess:
            rows = sess.execute(
                select(InferenceResult.benchmark, func.sum(InferenceResult.usage_cost))
                .where(InferenceResult.model_id == model_id)
                .group_by(InferenceResult.benchmark)
            )
            return {name: float(total or 0.0) for name, total in rows}

    def attempt_cost_summary(self, model_id: str) -> dict[str, Any]:
        """Cost split between successful cases and retries/errors."""
        with self.session() as sess:
            rows = sess.execute(
                select(
                    InferenceAttempt.status,
                    func.count(InferenceAttempt.id),
                    func.sum(InferenceAttempt.usage_cost),
                )
                .where(InferenceAttempt.model_id == model_id)
                .group_by(InferenceAttempt.status)
            )
            by_status = {
                status: {"count": int(count or 0), "usage_cost": float(cost or 0.0)}
                for status, count, cost in rows
            }
        success = by_status.get("SUCCESS", {"count": 0, "usage_cost": 0.0})
        error_cost = sum(v["usage_cost"] for k, v in by_status.items() if k != "SUCCESS")
        error_count = sum(v["count"] for k, v in by_status.items() if k != "SUCCESS")
        return {
            "by_status": by_status,
            "success_attempts": success["count"],
            "success_cost": success["usage_cost"],
            "error_attempts": error_count,
            "error_cost": error_cost,
            "total_attempts": success["count"] + error_count,
            "total_cost": success["usage_cost"] + error_cost,
        }

    def usage_totals(self, model_id: str) -> dict[str, Any]:
        with self.session() as sess:
            row = sess.execute(
                select(
                    func.count(InferenceResult.cache_key),
                    func.sum(InferenceResult.prompt_tokens),
                    func.sum(InferenceResult.cached_tokens),
                    func.sum(InferenceResult.cache_write_tokens),
                    func.sum(InferenceResult.completion_tokens),
                    func.sum(InferenceResult.reasoning_tokens),
                    func.sum(InferenceResult.total_tokens),
                    func.sum(InferenceResult.usage_cost),
                ).where(InferenceResult.model_id == model_id, InferenceResult.status == "SUCCESS")
            ).one()
        keys = (
            "cases",
            "prompt_tokens",
            "cached_tokens",
            "cache_write_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "total_tokens",
            "usage_cost",
        )
        out = {k: (v or 0) for k, v in zip(keys, row, strict=True)}
        out["cases"] = int(out["cases"])
        out["usage_cost"] = float(out["usage_cost"])
        for k in keys[1:-1]:
            out[k] = int(out[k])
        return out

    def judge_cost(self, target_model: str) -> float:
        with self.session() as sess:
            total = sess.scalar(
                select(func.sum(JudgeResult.usage_cost)).where(JudgeResult.target_model == target_model)
            )
        return float(total or 0.0)

    def global_stats(self) -> dict[str, Any]:
        with self.session() as sess:
            results = sess.scalar(select(func.count(InferenceResult.cache_key))) or 0
            successes = (
                sess.scalar(
                    select(func.count(InferenceResult.cache_key)).where(
                        InferenceResult.status == "SUCCESS"
                    )
                )
                or 0
            )
            attempts = sess.scalar(select(func.count(InferenceAttempt.id))) or 0
            metrics = sess.scalar(select(func.count(MetricResult.id))) or 0
            judges = sess.scalar(select(func.count(JudgeResult.cache_key))) or 0
            cost = sess.scalar(select(func.sum(InferenceAttempt.usage_cost))) or 0.0
            per_model = sess.execute(
                select(
                    InferenceResult.model_id,
                    InferenceResult.benchmark,
                    InferenceResult.status,
                    func.count(InferenceResult.cache_key),
                    func.sum(InferenceResult.usage_cost),
                ).group_by(InferenceResult.model_id, InferenceResult.benchmark, InferenceResult.status)
            ).all()
        return {
            "inference_results": int(results),
            "successful_results": int(successes),
            "inference_attempts": int(attempts),
            "metric_results": int(metrics),
            "judge_results": int(judges),
            "total_usage_cost": float(cost),
            "per_model": [
                {
                    "model_id": m,
                    "benchmark": b,
                    "status": s,
                    "count": int(c),
                    "usage_cost": float(cost or 0.0),
                }
                for m, b, s, c, cost in per_model
            ],
        }

    def last_resolved_model(self, model_id: str) -> str | None:
        """Most recent `resolved_model` OpenRouter reported for this slug.

        Used to pin alias-based slugs: if the alias starts resolving to a new
        upstream model, the cache key changes and the benchmark is treated as a
        new revision instead of silently mixing two models.
        """
        with self.session() as sess:
            return sess.scalar(
                select(InferenceResult.resolved_model)
                .where(
                    InferenceResult.model_id == model_id,
                    InferenceResult.status == "SUCCESS",
                    InferenceResult.resolved_model.is_not(None),
                )
                .order_by(InferenceResult.created_at.desc())
                .limit(1)
            )

    def distinct_model_ids(self) -> list[str]:
        with self.session() as sess:
            return [m for (m,) in sess.execute(select(InferenceResult.model_id).distinct())]
