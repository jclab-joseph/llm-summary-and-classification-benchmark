"""Failure guard: stop a run that is failing systematically.

A misconfiguration -- an unsupported request parameter, a revoked key, a model
that no longer exists -- fails identically on every case. Without a circuit
breaker the runner happily issues all 1,300 requests, reports 1,300 errors and
tells you nothing about why. This stops after a handful and keeps the reason.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

__all__ = ["FailureGuard", "FailureRecord"]


@dataclass(slots=True)
class FailureRecord:
    status: str
    error: str | None
    http_status: int | None = None
    sample_id: str = ""


class FailureGuard:
    """Trips on a run of consecutive failures, or on a sustained failure rate."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_consecutive_failures: int = 10,
        max_failure_rate: float = 0.5,
        min_attempts_before_rate_check: int = 20,
    ) -> None:
        self.enabled = enabled
        self.max_consecutive_failures = max_consecutive_failures
        self.max_failure_rate = max_failure_rate
        self.min_attempts_before_rate_check = min_attempts_before_rate_check

        self.attempts = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.tripped = False
        self.reason: str | None = None
        self.error_counts: Counter[str] = Counter()
        self.status_counts: Counter[str] = Counter()
        self.first_failure: FailureRecord | None = None

    # ------------------------------------------------------------------ #
    @property
    def failure_rate(self) -> float:
        return self.failures / self.attempts if self.attempts else 0.0

    def record_success(self) -> None:
        self.attempts += 1
        self.consecutive_failures = 0

    def record_failure(self, record: FailureRecord) -> None:
        self.attempts += 1
        self.failures += 1
        self.consecutive_failures += 1
        self.status_counts[record.status] += 1
        self.error_counts[self._normalize(record.error)] += 1
        if self.first_failure is None:
            self.first_failure = record
        if self.enabled and not self.tripped:
            self._check()

    def record(self, *, ok: bool, record: FailureRecord | None = None) -> None:
        if ok:
            self.record_success()
        else:
            self.record_failure(record or FailureRecord(status="UNKNOWN", error=None))

    # ------------------------------------------------------------------ #
    def _check(self) -> None:
        if (
            self.max_consecutive_failures
            and self.consecutive_failures >= self.max_consecutive_failures
        ):
            self.tripped = True
            self.reason = (
                f"{self.consecutive_failures} consecutive failures "
                f"(limit {self.max_consecutive_failures}); aborting before the rest of the run"
            )
            return
        if (
            self.attempts >= self.min_attempts_before_rate_check
            and self.failure_rate > self.max_failure_rate
        ):
            self.tripped = True
            self.reason = (
                f"{self.failures}/{self.attempts} requests failed "
                f"({self.failure_rate:.0%} > {self.max_failure_rate:.0%}); aborting the run"
            )

    @staticmethod
    def _normalize(error: str | None) -> str:
        if not error:
            return "(no error message)"
        return " ".join(error.split())[:300]

    def top_errors(self, limit: int = 3) -> list[tuple[str, int]]:
        return self.error_counts.most_common(limit)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "tripped": self.tripped,
            "reason": self.reason,
            "attempts": self.attempts,
            "failures": self.failures,
            "failure_rate": self.failure_rate,
            "consecutive_failures": self.consecutive_failures,
            "status_counts": dict(self.status_counts),
            "top_errors": [{"message": msg, "count": n} for msg, n in self.top_errors(5)],
            "first_failure": (
                {
                    "status": self.first_failure.status,
                    "error": self.first_failure.error,
                    "http_status": self.first_failure.http_status,
                    "sample_id": self.first_failure.sample_id,
                }
                if self.first_failure
                else None
            ),
        }
