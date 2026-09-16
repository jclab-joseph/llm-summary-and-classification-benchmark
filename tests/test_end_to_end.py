"""Full benchmark run against a mock OpenRouter transport.

The real client, runner, cache, metrics and report code all execute; only the
HTTP layer is replaced.
"""

from __future__ import annotations

import json

import pytest

from llmbench.reporting.leaderboard import build_leaderboard, load_model_summaries, write_leaderboard
from llmbench.reporting.html import render_html_report
from llmbench.reporting.summary import write_model_summary
from llmbench.runner import BenchmarkRunner

MODEL = "google/gemini-2.5-flash-lite"


@pytest.fixture
def runner(prepared, store, client_factory):
    return BenchmarkRunner(prepared, store, client_factory=lambda: client_factory())


async def test_full_run_then_cached_rerun(prepared, store, runner, mock_openrouter):
    model = prepared.require_model(MODEL)

    first = await runner.run(model)
    assert first.status == "COMPLETED"
    expected_calls = len(first.plan.tasks)
    assert expected_calls > 0
    assert first.api_calls == expected_calls
    assert mock_openrouter.chat_calls == expected_calls
    assert first.cache_hits == 0

    # (14) The reported cost is exactly the sum of OpenRouter's usage.cost.
    assert first.fresh_cost_usd == pytest.approx(mock_openrouter.total_cost)
    assert first.cost["stored_benchmark_cost"] == pytest.approx(mock_openrouter.total_cost)
    assert first.cost["total_openrouter_cost"] == pytest.approx(mock_openrouter.total_cost)
    assert first.cost["authoritative_source"] == "openrouter usage.cost"
    by_benchmark = first.cost["by_benchmark"]
    assert sum(by_benchmark.values()) == pytest.approx(mock_openrouter.total_cost)

    metrics = first.metrics
    assert set(metrics) >= {"summarization", "hallucination", "classification"}
    assert metrics["summarization"]["per_language"]["en"]["rougeLsum_f"] > 0
    assert metrics["summarization"]["per_language"]["ko"]["rougeLsum_f"] > 0
    assert metrics["classification"]["cross_lingual"]["paired_cases"] > 0
    assert metrics["hallucination"]["per_language"]["en"]["cases"] > 0
    assert metrics["hallucination"]["per_language"]["ko"]["status"] == "SKIPPED"

    # ---- second identical run: (1) + (13) zero inference calls ---- #
    mock_openrouter.reset_counters()
    second = await runner.run(model)

    assert mock_openrouter.chat_calls == 0, "a cached benchmark must issue no inference requests"
    assert second.api_calls == 0
    assert second.fresh_cost_usd == 0.0
    assert second.cost["fresh_cost_this_run"] == 0.0
    assert second.cache_hits == expected_calls
    assert second.cache_misses == 0

    # Stored cost is unchanged; savings equal what the cache avoided re-paying.
    assert second.cost["stored_benchmark_cost"] == pytest.approx(first.cost["stored_benchmark_cost"])
    assert second.cost["estimated_cost_without_cache"] == pytest.approx(mock_openrouter.total_cost)
    assert second.cost["estimated_cache_savings"] == pytest.approx(mock_openrouter.total_cost)

    # Identical inputs must produce identical scores.
    assert (
        second.metrics["summarization"]["per_language"]["en"]["rougeLsum_f"]
        == first.metrics["summarization"]["per_language"]["en"]["rougeLsum_f"]
    )


async def test_dry_run_never_calls_the_inference_api(prepared, runner, mock_openrouter):
    """(11) A dry run plans and prices, but issues no inference request."""
    model = prepared.require_model(MODEL)
    outcome = await runner.run(model, dry_run=True)

    assert outcome.status == "DRY_RUN"
    assert mock_openrouter.chat_calls == 0
    assert mock_openrouter.models_calls >= 1, "pricing metadata lookup is allowed in a dry run"
    assert outcome.plan.estimate.pending_tasks == len(outcome.plan.tasks)
    assert outcome.cost["estimated_fresh_cost"] > 0
    assert outcome.cost["budget_status"] == "SAFE TO RUN"
    assert "no OpenRouter inference request" in " ".join(outcome.notes)


async def test_judge_disabled_by_default_issues_no_judge_requests(prepared, runner, mock_openrouter):
    """(12) With the judge off, no judge request is ever made."""
    model = prepared.require_model(MODEL)
    assert prepared.judge.judge.enabled is False

    outcome = await runner.run(model, benchmarks=["summarization"])
    judge_requests = [
        r for r in mock_openrouter.requests
        if any("Generated summary" in m["content"] or "생성된 요약" in m["content"] for m in r["messages"])
    ]
    assert judge_requests == []
    assert outcome.judge == {}
    assert "judge" not in outcome.cost


async def test_judge_runs_and_caches_when_enabled(prepared, store, runner, client_factory, mock_openrouter):
    from llmbench.judge.runner import JudgeRunner
    from llmbench.openrouter.client import OpenRouterClient

    model = prepared.require_model(MODEL)
    outcome = await runner.run(model, benchmarks=["summarization"])
    results = runner.collect_results(outcome.plan)

    judge_runner = JudgeRunner(prepared, store)
    manifest = runner.manifests()["judge"]
    client: OpenRouterClient = client_factory()
    try:
        first = await judge_runner.run(
            profile="cheap",
            target_model=model,
            manifest=manifest,
            summarization_results=results,
            client=client,
        )
        assert first.status == "COMPLETED"
        assert first.api_calls == first.requested_samples > 0
        assert first.cost_usd > 0
        assert first.scores["judged_cases"] > 0

        calls_before = mock_openrouter.chat_calls
        second = await judge_runner.run(
            profile="cheap",
            target_model=model,
            manifest=manifest,
            summarization_results=results,
            client=client,
        )
        assert mock_openrouter.chat_calls == calls_before, "judge results must be cached"
        assert second.cache_hits == second.requested_samples
        assert second.api_calls == 0
    finally:
        await client.aclose()


async def test_self_judge_warning(prepared, store, runner, client_factory):
    from llmbench.judge.runner import SELF_JUDGE_WARNING, JudgeRunner

    luna = prepared.require_model("openai/gpt-5.6-luna")
    outcome = await runner.run(luna, benchmarks=["summarization"])
    results = runner.collect_results(outcome.plan)

    judge_runner = JudgeRunner(prepared, store)
    client = client_factory()
    try:
        judged = await judge_runner.run(
            profile="cheap",  # config/judge.yaml maps `cheap` to openai/gpt-5.6-luna
            target_model=luna,
            manifest=runner.manifests()["judge"],
            summarization_results=results,
            client=client,
        )
    finally:
        await client.aclose()

    assert judged.self_judge
    assert any(SELF_JUDGE_WARNING in w for w in judged.warnings)
    assert judged.to_dict()["primary_factuality_source"].startswith("deterministic")


async def test_budget_guard_blocks_expensive_run(prepared, runner):
    """(8) The guard refuses to start when the estimate exceeds the budget."""
    model = prepared.require_model(MODEL)
    outcome = await runner.run(model, budget_usd=0.0000001)

    assert outcome.status == "BLOCKED"
    assert outcome.api_calls == 0
    assert outcome.cost["budget_status"] == "OVER BUDGET"


async def test_failed_attempts_are_recorded_with_cost(prepared, store, runner, mock_openrouter):
    mock_openrouter.fail_first_n = 3
    mock_openrouter.fail_cost = 0.000002
    model = prepared.require_model(MODEL)

    outcome = await runner.run(model, benchmarks=["classification"])
    summary = store.attempt_cost_summary(model.model_id)

    assert summary["error_attempts"] == 3
    assert summary["error_cost"] == pytest.approx(3 * 0.000002)
    assert outcome.cost["retry_error_cost"] == pytest.approx(3 * 0.000002)
    # Total billed includes the failed attempts.
    assert summary["total_cost"] == pytest.approx(mock_openrouter.total_cost)
    assert outcome.fresh_cost_usd == pytest.approx(mock_openrouter.total_cost)


async def test_reports_are_written_and_match_usage_cost(prepared, runner, mock_openrouter):
    model = prepared.require_model(MODEL)
    outcome = await runner.run(model)
    path = write_model_summary(prepared, outcome)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["api_provider"] == "openrouter"
    assert payload["model"]["model_id"] == MODEL
    assert payload["cost"]["stored_benchmark_cost"] == pytest.approx(mock_openrouter.total_cost)
    assert payload["reproducibility"]["prompt_versions"]["summarization"]
    assert payload["reproducibility"]["manifests"]["summarization"]["hash"]
    assert "raw_response_json" not in json.dumps(payload)

    summaries = load_model_summaries(prepared)
    rows = build_leaderboard(summaries)
    assert rows and rows[0]["model"] == MODEL
    assert rows[0]["openrouter_cost"] == pytest.approx(mock_openrouter.total_cost)

    paths = write_leaderboard(prepared, rows)
    html_path = render_html_report(prepared, rows, summaries)
    assert paths["csv"].is_file() and paths["json"].is_file() and paths["markdown"].is_file()
    assert html_path.is_file()
    assert MODEL in html_path.read_text(encoding="utf-8")
    assert "usage.cost" in html_path.read_text(encoding="utf-8")


async def test_force_reruns_cached_cases(prepared, runner, mock_openrouter):
    model = prepared.require_model(MODEL)
    first = await runner.run(model, benchmarks=["summarization"])
    mock_openrouter.reset_counters()

    cached = await runner.run(model, benchmarks=["summarization"])
    assert mock_openrouter.chat_calls == 0

    forced = await runner.run(model, benchmarks=["summarization"], force=True)
    assert mock_openrouter.chat_calls == len(first.plan.tasks)
    assert forced.api_calls == len(first.plan.tasks)


async def test_failed_cases_need_retry_flag(prepared, store, runner, mock_openrouter):
    mock_openrouter.fail_first_n = 999  # every attempt fails
    model = prepared.require_model(MODEL)

    first = await runner.run(model, benchmarks=["summarization"])
    assert first.failures
    failed_before = len(store.failed_keys(model.model_id))
    assert failed_before > 0

    mock_openrouter.fail_first_n = 0
    mock_openrouter.reset_counters()
    # Default behaviour: a stored failure is not silently retried.
    await runner.run(model, benchmarks=["summarization"])
    assert mock_openrouter.chat_calls == 0

    retried = await runner.run(model, benchmarks=["summarization"], retry_failed=True)
    assert mock_openrouter.chat_calls == failed_before
    assert retried.api_calls == failed_before
    assert not store.failed_keys(model.model_id)


async def test_metric_cache_reuses_inference_when_metric_version_changes(prepared, runner, mock_openrouter):
    """(6) Bumping an evaluator version recomputes metrics, not inference."""
    model = prepared.require_model(MODEL)
    await runner.run(model, benchmarks=["summarization"])
    mock_openrouter.reset_counters()

    prepared.benchmark.metrics.rouge.version = "rouge-2.0"
    outcome = await runner.run(model, benchmarks=["summarization"])

    assert mock_openrouter.chat_calls == 0
    assert outcome.metrics["summarization"]["per_language"]["en"]["rougeLsum_f"] > 0
    assert outcome.metrics["summarization"]["evaluators"]["rouge"]["version"] == "rouge-2.0"


async def test_manifests_are_mirrored_into_the_database(prepared, store, runner):
    """`benchmarks` and `samples` mirror the frozen manifests for offline joins."""
    from sqlalchemy import select

    from llmbench.db.schema import Benchmark, Sample

    model = prepared.require_model(MODEL)
    await runner.run(model)

    with store.session() as session:
        benchmarks = list(session.scalars(select(Benchmark)))
        samples = list(session.scalars(select(Sample)))

    names = {b.name for b in benchmarks}
    assert {"summarization", "hallucination", "classification"} <= names
    for benchmark in benchmarks:
        assert benchmark.manifest_hash
        assert benchmark.dataset_revision

    assert samples
    by_id = {s.sample_id: s for s in samples}
    manifest = runner.manifests()["summarization"]
    for record in manifest.records:
        stored = by_id[record.sample_id]
        assert stored.source_hash == record.source_hash
        assert stored.reference_hash == record.reference_hash
        assert stored.payload_json["source"] == record.payload["source"]

    # Re-running must not duplicate rows.
    await runner.run(model)
    with store.session() as session:
        assert len(list(session.scalars(select(Sample)))) == len(samples)


async def test_korean_hallucination_runs_when_aihub_data_is_present(
    prepared_with_ko, store, client_factory, mock_openrouter
):
    """With the AI-Hub manifest prepared, KO is scored instead of SKIPPED."""
    runner = BenchmarkRunner(prepared_with_ko, store, client_factory=lambda: client_factory())
    model = prepared_with_ko.require_model(MODEL)

    outcome = await runner.run(model, benchmarks=["hallucination"])
    per_language = outcome.metrics["hallucination"]["per_language"]

    assert per_language["ko"]["status"] == "OK"
    assert per_language["ko"]["cases"] == 6
    assert per_language["ko"]["documents"] == 3
    assert per_language["en"]["cases"] == 6
    assert outcome.metrics["hallucination"]["manifest_hashes"]["ko"]

    # Korean prompts must be Korean, and must carry the Korean source.
    ko_requests = [
        r for r in mock_openrouter.requests
        if any("후보 요약문" in m["content"] for m in r["messages"])
    ]
    assert len(ko_requests) == 6
    assert all(r["max_tokens"] == 8 for r in ko_requests)

    # Second run: the KO cases are cached too.
    mock_openrouter.reset_counters()
    await runner.run(model, benchmarks=["hallucination"])
    assert mock_openrouter.chat_calls == 0
