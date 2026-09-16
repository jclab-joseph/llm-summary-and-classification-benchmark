"""Failure guard, capability preflight and logging."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from llmbench.failures import FailureGuard, FailureRecord
from llmbench.logging_setup import configure_logging, get_logger
from llmbench.runner import BenchmarkRunner

MODEL = "google/gemini-2.5-flash-lite"
LUNA = "openai/gpt-5.6-luna"


def failure(status: str = "OPENROUTER_ERROR", error: str = "boom") -> FailureRecord:
    return FailureRecord(status=status, error=error, http_status=404, sample_id="s")


# --------------------------------------------------------------------------- #
# FailureGuard
# --------------------------------------------------------------------------- #
def test_trips_on_consecutive_failures():
    guard = FailureGuard(max_consecutive_failures=3, min_attempts_before_rate_check=1000)
    for _ in range(2):
        guard.record_failure(failure())
    assert not guard.tripped

    guard.record_failure(failure())
    assert guard.tripped
    assert "3 consecutive failures" in guard.reason


def test_success_resets_the_consecutive_counter():
    guard = FailureGuard(max_consecutive_failures=3, min_attempts_before_rate_check=1000)
    guard.record_failure(failure())
    guard.record_failure(failure())
    guard.record_success()
    guard.record_failure(failure())
    assert not guard.tripped
    assert guard.consecutive_failures == 1


def test_trips_on_sustained_failure_rate():
    guard = FailureGuard(max_consecutive_failures=0, max_failure_rate=0.5, min_attempts_before_rate_check=10)
    for _ in range(10):
        guard.record_success()
    for _ in range(10):
        guard.record_failure(failure())
    # Exactly 50% is not "> 50%", so a merely flaky run is not aborted.
    assert guard.failure_rate == pytest.approx(0.5)
    assert not guard.tripped

    guard.record_failure(failure())
    assert guard.tripped
    assert "aborting the run" in guard.reason


def test_disabled_guard_never_trips():
    guard = FailureGuard(enabled=False, max_consecutive_failures=1)
    for _ in range(50):
        guard.record_failure(failure())
    assert not guard.tripped
    assert guard.reason is None


def test_summary_keeps_error_messages():
    guard = FailureGuard()
    guard.record_failure(failure(error="No endpoints found that can handle the requested parameters."))
    guard.record_failure(failure(error="No endpoints found that can handle the requested parameters."))
    guard.record_failure(failure(error="rate limited"))

    summary = guard.summary()
    assert summary["failures"] == 3
    assert summary["top_errors"][0]["count"] == 2
    assert "No endpoints found" in summary["top_errors"][0]["message"]
    assert summary["first_failure"]["http_status"] == 404


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def test_log_file_is_written(tmp_path: Path):
    path = configure_logging(log_dir=tmp_path, verbosity=0, console=False)
    assert path is not None and path.parent == tmp_path

    get_logger("test").warning("something broke: %s", "details")
    for handler in logging.getLogger("llmbench").handlers:
        handler.flush()

    contents = path.read_text(encoding="utf-8")
    assert "something broke: details" in contents
    assert "WARNING" in contents


def test_unwritable_log_destination_does_not_raise(tmp_path: Path):
    blocked = tmp_path / "file.txt"
    blocked.write_text("x", encoding="utf-8")
    assert configure_logging(log_file=blocked / "nested" / "run.log", console=False) is None


# --------------------------------------------------------------------------- #
# capability preflight
# --------------------------------------------------------------------------- #
async def test_preflight_blocks_unsupported_parameters(prepared, store, client_factory, mock_openrouter):
    """A model that rejects `temperature` must not burn the whole run on 404s."""
    runner = BenchmarkRunner(prepared, store, client_factory=lambda: client_factory())
    luna = prepared.require_model(LUNA)
    # Pretend the config is stale: claim temperature support the model lacks.
    luna.parameters.temperature = True
    luna.parameters.top_p = True

    outcome = await runner.run(luna, benchmarks=["summarization"])

    assert outcome.status == "BLOCKED"
    assert mock_openrouter.chat_calls == 0, "nothing may be sent when preflight fails"
    assert outcome.api_calls == 0
    assert {p["parameter"] for p in outcome.preflight["blocking"]} == {"temperature", "top_p"}
    assert any("parameters.temperature: false" in note for note in outcome.notes)
    assert any("404" in note for note in outcome.notes)


async def test_preflight_passes_for_correctly_configured_model(prepared, store, client_factory):
    runner = BenchmarkRunner(prepared, store, client_factory=lambda: client_factory())
    luna = prepared.require_model(LUNA)
    assert luna.parameters.unsupported() == ["stop", "temperature", "top_p"]

    outcome = await runner.run(luna, benchmarks=["summarization"])

    assert outcome.status == "COMPLETED"
    assert outcome.preflight["checked"] is True
    assert outcome.preflight["blocking"] == []


async def test_unsupported_parameters_are_not_sent(prepared, store, client_factory, mock_openrouter):
    runner = BenchmarkRunner(prepared, store, client_factory=lambda: client_factory())
    await runner.run(prepared.require_model(LUNA), benchmarks=["summarization"])

    assert mock_openrouter.requests
    for payload in mock_openrouter.requests:
        assert "temperature" not in payload
        assert "top_p" not in payload
        assert payload["max_tokens"] > 0
        # The parameters it does support are still sent.
        assert "reasoning" in payload


async def test_supported_parameters_are_still_sent(prepared, store, client_factory, mock_openrouter):
    runner = BenchmarkRunner(prepared, store, client_factory=lambda: client_factory())
    await runner.run(prepared.require_model(MODEL), benchmarks=["summarization"])

    for payload in mock_openrouter.requests:
        assert payload["temperature"] == 0.0
        assert payload["top_p"] == 1.0


async def test_skip_preflight_override(prepared, store, client_factory, mock_openrouter):
    runner = BenchmarkRunner(prepared, store, client_factory=lambda: client_factory())
    luna = prepared.require_model(LUNA)
    luna.parameters.temperature = True
    prepared.benchmark.failure_guard.max_consecutive_failures = 3

    outcome = await runner.run(luna, benchmarks=["summarization"], skip_preflight=True)

    # It runs, hits the real 404s, and the failure guard stops it early.
    assert outcome.status == "ABORTED"
    assert mock_openrouter.chat_calls > 0


def test_cache_key_reflects_omitted_parameters(prepared):
    """Dropping a parameter changes what was asked, so it changes the key."""
    from llmbench.benchmarks.summarization import build_summarization_tasks
    from llmbench.datasets.manifest import read_manifest

    manifest = read_manifest(
        prepared.manifest_dir / prepared.benchmark.benchmarks.summarization.manifest
    )
    luna = prepared.require_model(LUNA)
    without = [t.cache_key for t in build_summarization_tasks(prepared, luna, manifest).tasks]
    material = build_summarization_tasks(prepared, luna, manifest).tasks[0].key_material
    assert material["temperature"] is None
    assert material["top_p"] is None

    luna.parameters.temperature = True
    luna.parameters.top_p = True
    with_params = [t.cache_key for t in build_summarization_tasks(prepared, luna, manifest).tasks]
    assert without != with_params


# --------------------------------------------------------------------------- #
# abort behaviour
# --------------------------------------------------------------------------- #
async def test_run_aborts_after_repeated_failures(prepared, store, client_factory, mock_openrouter):
    mock_openrouter.fail_first_n = 10_000
    mock_openrouter.fail_status = 500
    prepared.benchmark.failure_guard.max_consecutive_failures = 3
    prepared.benchmark.concurrency.max_parallel_requests = 1

    runner = BenchmarkRunner(prepared, store, client_factory=lambda: client_factory())
    outcome = await runner.run(prepared.require_model(MODEL))

    assert outcome.status == "ABORTED"
    assert outcome.aborted
    assert outcome.api_calls < len(outcome.plan.tasks), "the guard must stop the run early"
    assert outcome.skipped_by_failure_guard > 0
    assert outcome.failure_summary["tripped"] is True
    assert outcome.failure_summary["top_errors"]


async def test_skipped_cases_are_not_persisted(prepared, store, client_factory, mock_openrouter):
    """A skipped case stores nothing, so a later run retries it without a flag."""
    mock_openrouter.fail_first_n = 10_000
    prepared.benchmark.failure_guard.max_consecutive_failures = 2
    prepared.benchmark.concurrency.max_parallel_requests = 1

    runner = BenchmarkRunner(prepared, store, client_factory=lambda: client_factory())
    aborted = await runner.run(prepared.require_model(MODEL), benchmarks=["summarization"])
    assert aborted.status == "ABORTED"

    attempted = aborted.api_calls
    stored = store.results_for(MODEL, "summarization")
    assert len(stored) == attempted

    # Fix the cause and re-run: everything still pending is picked up, no flags.
    mock_openrouter.fail_first_n = 0
    mock_openrouter.reset_counters()
    recovered = await runner.run(
        prepared.require_model(MODEL), benchmarks=["summarization"], retry_failed=True
    )
    assert recovered.status == "COMPLETED"
    assert not store.failed_keys(MODEL, "summarization")


async def test_guard_can_be_disabled(prepared, store, client_factory, mock_openrouter):
    mock_openrouter.fail_first_n = 10_000
    prepared.benchmark.failure_guard.enabled = False

    runner = BenchmarkRunner(prepared, store, client_factory=lambda: client_factory())
    outcome = await runner.run(prepared.require_model(MODEL), benchmarks=["summarization"])

    assert outcome.status == "COMPLETED"
    assert outcome.api_calls == len(outcome.plan.tasks)
    assert outcome.failure_summary["tripped"] is False


def test_prune_logs_keeps_the_newest(tmp_path: Path):
    from llmbench.logging_setup import prune_logs

    for stamp in ("20250101T000000", "20250102T000000", "20250103T000000"):
        (tmp_path / f"benchmark-{stamp}.log").write_text("x", encoding="utf-8")
    (tmp_path / "unrelated.txt").write_text("keep me", encoding="utf-8")

    assert prune_logs(tmp_path, keep_last=2) == 1
    remaining = sorted(p.name for p in tmp_path.glob("benchmark-*.log"))
    assert remaining == ["benchmark-20250102T000000.log", "benchmark-20250103T000000.log"]
    assert (tmp_path / "unrelated.txt").exists()
