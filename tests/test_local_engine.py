"""Local (non-OpenRouter) classification track.

A fake engine stands in for llama.cpp so the suite needs no weights and no
optional extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pytest

from llmbench.benchmarks.classification import (
    LOCAL_MODE,
    build_classification_tasks,
    classification_output_mode,
    score_classification,
    structured_output_version,
)
from llmbench.config.schema import LocalModelConfig
from llmbench.datasets.manifest import read_manifest
from llmbench.local.client import LocalEngineClient
from llmbench.local.download import local_model_path
from llmbench.local.engine import LocalResult, build_engine
from llmbench.metrics.classification import parse_single_label
from llmbench.openrouter.types import GenerationRequest, InferenceStatus
from llmbench.runner import BenchmarkRunner

LOCAL_SLUG = "local/qwen2.5-1.5b-instruct-q8-gguf"
LABELS = ["alarm_set", "audio_volume_mute", "iot_hue_lightchange", "weather_query"]


class FakeEngine:
    """Always answers with a candidate, like a grammar-constrained decoder."""

    def __init__(self, model_path: Path, runtime: dict[str, Any]) -> None:
        self.model_path = Path(model_path)
        self.runtime = runtime
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    def classify(self, *, system: str, user: str, candidates: Sequence[str]) -> LocalResult:
        self.calls.append((system, user))
        # Deterministic pick from the candidate set.
        index = sum(map(ord, user)) % len(candidates)
        return LocalResult(
            label=candidates[index], prompt_tokens=len(system) // 4, completion_tokens=3, latency_ms=1
        )

    def describe(self) -> dict[str, Any]:
        return {"engine": "fake", "model_file": self.model_path.name}

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def local_model(prepared) -> LocalModelConfig:
    return prepared.require_model(LOCAL_SLUG)


@pytest.fixture
def fake_engine(tmp_path) -> FakeEngine:
    weights = tmp_path / "fake.gguf"
    weights.write_bytes(b"0")
    return FakeEngine(weights, {})


@pytest.fixture
def local_client(local_model, fake_engine):
    client = LocalEngineClient(
        engine="fake",
        model_id=local_model.model_id,
        model_path=fake_engine.model_path,
        runtime=local_model.runtime.model_dump(),
    )
    # The runner closes its client, so the engine is held by the fixture too.
    client._engine = fake_engine
    return client


@pytest.fixture
def local_runner(prepared, store, local_client):
    return BenchmarkRunner(prepared, store, client_factory=lambda: local_client)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def test_local_models_are_separate_from_the_openrouter_config(prepared):
    assert {m.provider for m in prepared.models.models} == {"openrouter"}
    assert prepared.local_models.models
    assert all(m.provider == "local" for m in prepared.local_models.models)


def test_weights_are_pinned_to_a_revision(prepared, local_model):
    path = local_model_path(prepared, local_model)
    assert local_model.source.revision[:12] in path.name
    assert local_model.source.filename in path.name


def test_engine_identity_is_part_of_the_cache_key(prepared, local_model):
    """Different weights or decoding settings are a different benchmark run."""
    manifest = read_manifest(prepared.manifest_dir / prepared.benchmark.benchmarks.classification.manifest)
    before = [t.cache_key for t in build_classification_tasks(prepared, local_model, manifest).tasks]
    assert before
    assert build_classification_tasks(prepared, local_model, manifest).tasks[0].key_material["extra"][
        "local_engine"
    ]["repo"]

    local_model.runtime.temperature = 0.7
    assert [t.cache_key for t in build_classification_tasks(prepared, local_model, manifest).tasks] != before


def test_throughput_settings_do_not_change_the_cache_key(prepared, local_model):
    """A different thread count must not re-run the whole benchmark."""
    manifest = read_manifest(prepared.manifest_dir / prepared.benchmark.benchmarks.classification.manifest)
    before = [t.cache_key for t in build_classification_tasks(prepared, local_model, manifest).tasks]

    local_model.runtime.n_threads = 4
    local_model.runtime.n_batch = 128
    assert [t.cache_key for t in build_classification_tasks(prepared, local_model, manifest).tasks] == before


def test_unknown_engine_is_rejected(tmp_path):
    from llmbench.core.errors import BenchmarkError

    with pytest.raises(BenchmarkError, match="unknown local engine"):
        build_engine("nope", tmp_path / "x.gguf", {})


# --------------------------------------------------------------------------- #
# task shape
# --------------------------------------------------------------------------- #
def test_local_tasks_are_one_utterance_each(prepared, local_model):
    manifest = read_manifest(prepared.manifest_dir / prepared.benchmark.benchmarks.classification.manifest)
    tasks = build_classification_tasks(prepared, local_model, manifest).tasks

    assert len(tasks) == len(manifest.records)
    for task in tasks:
        assert task.case_count == 1
        assert task.meta["batch_size"] == 1
        assert task.meta["output_mode"] == LOCAL_MODE
        # The candidate set is what the engine constrains decoding to.
        assert task.candidates and len(task.candidates) == task.meta["label_count"]
        assert task.response_format is None


def test_local_mode_is_independent_of_the_hosted_setting(prepared, local_model):
    for hosted in ("text", "json_schema"):
        prepared.benchmark.structured_output.classification_mode = hosted
        assert classification_output_mode(prepared, local_model) == LOCAL_MODE
        assert structured_output_version(prepared, local_model).endswith(LOCAL_MODE)


def test_local_keys_never_collide_with_hosted_keys(prepared, local_model):
    manifest = read_manifest(prepared.manifest_dir / prepared.benchmark.benchmarks.classification.manifest)
    hosted = prepared.require_model("google/gemini-2.5-flash-lite")

    local_keys = {t.cache_key for t in build_classification_tasks(prepared, local_model, manifest).tasks}
    hosted_keys = {t.cache_key for t in build_classification_tasks(prepared, hosted, manifest).tasks}
    assert local_keys.isdisjoint(hosted_keys)


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #
async def test_client_returns_a_candidate_and_no_cost(local_client):
    response = await local_client.generate(
        GenerationRequest(
            model="local/x",
            messages=[
                {"role": "system", "content": "intent list " * 40},
                {"role": "user", "content": "wake me up"},
            ],
            max_tokens=32,
            candidates=LABELS,
        )
    )

    assert response.status is InferenceStatus.SUCCESS
    assert response.text in LABELS
    assert response.upstream_provider.startswith("local:")
    # Nothing was billed. `None` is not `0.0`.
    assert response.usage.cost is None
    assert response.usage.prompt_tokens > 0


async def test_client_refuses_without_candidates(local_client):
    response = await local_client.generate(
        GenerationRequest(model="local/x", messages=[{"role": "user", "content": "u"}], max_tokens=8)
    )
    assert not response.ok
    assert "candidate set" in response.error


async def test_client_reports_engine_failure_as_an_error(local_client):
    def boom(**kwargs):
        raise RuntimeError("engine exploded")

    local_client._engine.classify = boom
    response = await local_client.generate(
        GenerationRequest(
            model="local/x",
            messages=[{"role": "user", "content": "u"}],
            max_tokens=8,
            candidates=LABELS,
        )
    )
    assert not response.ok
    assert "engine exploded" in response.error


async def test_client_refuses_to_run_during_a_dry_run(local_client):
    from llmbench.core.errors import DryRunViolation

    local_client.dry_run = True
    with pytest.raises(DryRunViolation):
        await local_client.generate(
            GenerationRequest(
                model="local/x", messages=[{"role": "user", "content": "u"}], max_tokens=8, candidates=LABELS
            )
        )


def test_client_exposes_no_pricing(local_client):
    import asyncio

    assert asyncio.run(local_client.get_model_metadata()) == {}
    assert asyncio.run(local_client.get_model_pricing()) == {}


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [("alarm_set", 0), (" alarm_set \n", 0), ("`weather_query`", 3), ("not_a_label", None), ("", None)],
)
def test_single_label_parsing(text, expected):
    assert parse_single_label(text, LABELS) == expected


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
async def test_local_run_scores_and_caches(prepared, store, local_runner, local_client, fake_engine):
    model = prepared.require_model(LOCAL_SLUG)

    outcome = await local_runner.run(model, benchmarks=["classification"])

    assert outcome.status == "COMPLETED"
    block = outcome.metrics["classification"]
    assert block["output_mode"] == LOCAL_MODE
    assert block["batch_size"] == 1
    assert block["per_language"]["en"]["cases"] > 0
    # A constrained decoder cannot produce an unparseable answer.
    assert block["per_language"]["en"]["invalid_output_rate"] == 0.0

    calls = len(fake_engine.calls)
    assert calls == len(outcome.plan.tasks)

    await local_runner.run(model, benchmarks=["classification"])
    assert len(fake_engine.calls) == calls, "the second run must hit the cache"


async def test_local_run_is_not_billed(prepared, store, local_runner):
    from llmbench.reporting.summary import build_model_summary

    model = prepared.require_model(LOCAL_SLUG)
    outcome = await local_runner.run(model, benchmarks=["classification"])
    summary = build_model_summary(prepared, outcome)

    assert summary["api_provider"] == "local"
    assert summary["cost"]["total_openrouter_cost"] is None
    assert summary["cost"]["fresh_cost_this_run"] is None
    assert "not applicable" in summary["cost"]["billing"]


async def test_local_models_only_run_classification(prepared, store, local_runner):
    model = prepared.require_model(LOCAL_SLUG)
    outcome = await local_runner.run(model)

    assert set(outcome.plan.benchmarks) == {"classification"}
    assert "summarization" in outcome.plan.skipped
    assert "classification benchmark only" in outcome.plan.skipped["summarization"]


async def test_local_run_needs_no_api_key(prepared, store, local_runner, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    outcome = await local_runner.run(prepared.require_model(LOCAL_SLUG), benchmarks=["classification"])
    assert outcome.status == "COMPLETED"


def test_gpu_offload_is_part_of_the_cache_key(prepared, local_model):
    """CPU and CUDA kernels disagree on ~1% of answers, so they are not the same run."""
    manifest = read_manifest(prepared.manifest_dir / prepared.benchmark.benchmarks.classification.manifest)
    local_model.runtime.n_gpu_layers = 0
    cpu_keys = [t.cache_key for t in build_classification_tasks(prepared, local_model, manifest).tasks]

    local_model.runtime.n_gpu_layers = 20
    gpu_keys = [t.cache_key for t in build_classification_tasks(prepared, local_model, manifest).tasks]
    assert cpu_keys != gpu_keys
