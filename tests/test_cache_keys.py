"""Cache-key semantics: what must invalidate inference, and what must not."""

from __future__ import annotations

import copy

import pytest

from llmbench.benchmarks.summarization import build_summarization_tasks
from llmbench.cache.keys import build_inference_key, build_metric_key
from llmbench.datasets.manifest import read_manifest


BASE = dict(
    model_id="google/gemini-2.5-flash-lite",
    routing_config_hash="routing-hash-a",
    benchmark="summarization",
    benchmark_version="1.0.0",
    dataset_revision="rev-1",
    split="test",
    sample_id="xlsum:en:001",
    sample_content_hash="content-hash",
    system_prompt="You are a summarizer.",
    user_prompt="Summarize: hello world",
    prompt_template_version="sum-1",
    temperature=0.0,
    top_p=1.0,
    max_output_tokens=200,
    reasoning={"enabled": False, "effort": None, "max_tokens": None, "exclude": True},
)


def key(**overrides):
    params = {**BASE, **overrides}
    return build_inference_key(**params)[0]


def test_identical_request_produces_identical_key():
    """(1) The same request hits the cache."""
    assert key() == key()


def test_key_is_independent_of_dict_ordering():
    reordered = {k: BASE[k] for k in reversed(list(BASE))}
    assert build_inference_key(**reordered)[0] == key()


def test_prompt_change_misses():
    """(3) A changed prompt is a different question -> cache miss."""
    assert key(user_prompt="Summarize this instead: hello world") != key()
    assert key(system_prompt="You are a different summarizer.") != key()
    assert key(prompt_template_version="sum-2") != key()


def test_model_slug_change_misses():
    """(4) A different OpenRouter slug is a different model -> cache miss."""
    assert key(model_id="openai/gpt-5.6-luna") != key()


def test_routing_config_change_misses():
    """(5) Routing decides which upstream endpoint answers -> cache miss."""
    assert key(routing_config_hash="routing-hash-b") != key()


def test_generation_parameters_change_misses():
    assert key(temperature=0.7) != key()
    assert key(top_p=0.9) != key()
    assert key(max_output_tokens=256) != key()
    assert key(seed=42) != key()
    assert key(reasoning={"enabled": True, "effort": "low", "max_tokens": None, "exclude": True}) != key()
    assert key(structured_output_schema_version="so-2") != key()
    assert key(truncation_policy_version="trunc-2") != key()
    assert key(generation_config_version="gen-2") != key()


def test_sample_identity_change_misses():
    assert key(sample_id="xlsum:en:002") != key()
    assert key(sample_content_hash="other-content") != key()
    assert key(dataset_revision="rev-2") != key()
    assert key(split="validation") != key()
    assert key(benchmark_version="1.1.0") != key()


def test_alias_resolution_change_misses():
    """A `:latest` alias resolving to a new model is a new benchmark revision."""
    assert key(alias_resolution="gemini-2.5-flash-lite-20250101") != key()


def test_price_is_not_part_of_the_key(prepared, store):
    """(2) Changing pricing must never cause a single new inference request."""
    model = prepared.require_model("google/gemini-2.5-flash-lite")
    manifest = read_manifest(prepared.manifest_dir / prepared.benchmark.benchmarks.summarization.manifest)
    before = [t.cache_key for t in build_summarization_tasks(prepared, model, manifest).tasks]

    # Mutate the pricing table as aggressively as possible.
    prepared.pricing.models["google/gemini-2.5-flash-lite"].input_per_million = 99.0
    prepared.pricing.models["google/gemini-2.5-flash-lite"].output_per_million = 123.0
    prepared.pricing.constraints.max_input_per_million = 500.0

    after = [t.cache_key for t in build_summarization_tasks(prepared, model, manifest).tasks]
    assert before == after


def test_report_and_metric_config_are_not_part_of_the_key(prepared):
    """Metric configuration belongs to the metric cache, not the inference cache."""
    model = prepared.require_model("google/gemini-2.5-flash-lite")
    manifest = read_manifest(prepared.manifest_dir / prepared.benchmark.benchmarks.summarization.manifest)
    before = [t.cache_key for t in build_summarization_tasks(prepared, model, manifest).tasks]

    prepared.benchmark.metrics.bertscore.model_type = "xlm-roberta-large"
    prepared.benchmark.metrics.bertscore.version = "bertscore-2.0"
    prepared.benchmark.metrics.rouge.version = "rouge-9.9"

    after = [t.cache_key for t in build_summarization_tasks(prepared, model, manifest).tasks]
    assert before == after


def test_metric_key_changes_with_evaluator_version():
    """(6) A new evaluator version recomputes metrics without touching inference."""
    base = dict(
        inference_result_hash="result-hash",
        evaluator="bertscore",
        evaluator_config={"model_type": "bert-base-multilingual-cased"},
        reference_hash="ref-hash",
    )
    k1, _ = build_metric_key(evaluator_version="bertscore-1.0", **base)
    k2, _ = build_metric_key(evaluator_version="bertscore-2.0", **base)
    k3, _ = build_metric_key(
        evaluator_version="bertscore-1.0",
        **{**base, "evaluator_config": {"model_type": "xlm-roberta-large"}},
    )
    assert k1 != k2
    assert k1 != k3
    # ...and the inference key is untouched by any of it.
    assert key() == key()
