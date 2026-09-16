"""Cache key construction.

Two independent caches, on purpose:

* **Inference cache** -- keyed by everything that changes what the model was
  actually asked to produce. Price changes, report formatting and metric
  configuration are deliberately excluded, so a price update never triggers a
  single new OpenRouter call.
* **Metric cache** -- keyed by the inference result plus the evaluator identity.
  Swapping the BERTScore checkpoint recomputes metrics and reuses inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from llmbench.core.canonical import hash_obj, hash_text

__all__ = [
    "InferenceKeyMaterial",
    "build_inference_key",
    "build_metric_key",
    "build_judge_key",
    "KEY_SCHEMA_VERSION",
]

# Version of the *shape* of the key material. Bumping it invalidates every key.
KEY_SCHEMA_VERSION = "key-1"


@dataclass(frozen=True, slots=True)
class InferenceKeyMaterial:
    """The exact, fully-explicit set of facts an inference cache key hashes.

    Anything absent from this dataclass cannot influence the cache key -- which
    is the point: `usage.cost`, pricing tables, report options and metric config
    are structurally unable to cause re-inference.
    """

    key_schema_version: str

    # provider / model identity
    api_provider: str
    model_id: str
    routing_config_hash: str
    alias_resolution: str | None

    # benchmark / sample identity
    benchmark: str
    benchmark_version: str
    dataset_revision: str
    split: str
    sample_id: str
    sample_content_hash: str

    # prompt identity
    prompt_template_version: str
    system_prompt_hash: str
    user_prompt_hash: str

    # generation identity (None = parameter not sent, because the model does
    # not accept it -- the key has to describe what was actually requested)
    temperature: float | None
    top_p: float | None
    max_output_tokens: int
    seed: int | None
    reasoning: dict[str, Any] | None
    tools_enabled: bool
    web_search_enabled: bool
    structured_output_schema_version: str
    truncation_policy_version: str
    generation_config_version: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def cache_key(self) -> str:
        return hash_obj(self.to_dict())


def build_inference_key(
    *,
    model_id: str,
    routing_config_hash: str,
    benchmark: str,
    benchmark_version: str,
    dataset_revision: str,
    split: str,
    sample_id: str,
    sample_content_hash: str,
    system_prompt: str,
    user_prompt: str,
    prompt_template_version: str,
    temperature: float | None,
    top_p: float | None,
    max_output_tokens: int,
    reasoning: dict[str, Any] | None,
    seed: int | None = None,
    tools_enabled: bool = False,
    web_search_enabled: bool = False,
    structured_output_schema_version: str = "so-1",
    truncation_policy_version: str = "trunc-1",
    generation_config_version: str = "gen-1",
    alias_resolution: str | None = None,
    api_provider: str = "openrouter",
    extra: dict[str, Any] | None = None,
) -> tuple[str, InferenceKeyMaterial]:
    """Return ``(cache_key, key_material)`` for one benchmark case."""
    material = InferenceKeyMaterial(
        key_schema_version=KEY_SCHEMA_VERSION,
        api_provider=api_provider,
        model_id=model_id,
        routing_config_hash=routing_config_hash,
        alias_resolution=alias_resolution,
        benchmark=benchmark,
        benchmark_version=benchmark_version,
        dataset_revision=dataset_revision,
        split=split,
        sample_id=sample_id,
        sample_content_hash=sample_content_hash,
        prompt_template_version=prompt_template_version,
        system_prompt_hash=hash_text(system_prompt),
        user_prompt_hash=hash_text(user_prompt),
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        seed=seed,
        reasoning=reasoning,
        tools_enabled=tools_enabled,
        web_search_enabled=web_search_enabled,
        structured_output_schema_version=structured_output_schema_version,
        truncation_policy_version=truncation_policy_version,
        generation_config_version=generation_config_version,
        extra=extra or {},
    )
    return material.cache_key, material


def build_metric_key(
    *,
    inference_result_hash: str,
    evaluator: str,
    evaluator_version: str,
    evaluator_config: dict[str, Any],
    reference_hash: str = "",
    scope: str = "sample",
) -> tuple[str, dict[str, Any]]:
    """Return ``(metric_cache_key, material)``.

    Note what is *not* here: the model slug, the pricing table, the report
    format. A metric is a pure function of (inference output, evaluator,
    reference).
    """
    material = {
        "key_schema_version": KEY_SCHEMA_VERSION,
        "inference_result_hash": inference_result_hash,
        "evaluator": evaluator,
        "evaluator_version": evaluator_version,
        "evaluator_config_hash": hash_obj(evaluator_config),
        "reference_hash": reference_hash,
        "scope": scope,
    }
    return hash_obj(material), material


def build_judge_key(
    *,
    judge_model_id: str,
    judge_routing_config_hash: str,
    target_model_id: str,
    sample_id: str,
    summary_hash: str,
    source_hash: str,
    reference_hash: str,
    judge_prompt_version: str,
    rubric_version: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float | None,
    top_p: float | None,
    max_output_tokens: int,
    reasoning: dict[str, Any] | None,
    structured_output_schema_version: str,
) -> tuple[str, dict[str, Any]]:
    """Judge results are cached with the same discipline as inference."""
    material = {
        "key_schema_version": KEY_SCHEMA_VERSION,
        "api_provider": "openrouter",
        "judge_model_id": judge_model_id,
        "judge_routing_config_hash": judge_routing_config_hash,
        "target_model_id": target_model_id,
        "sample_id": sample_id,
        "summary_hash": summary_hash,
        "source_hash": source_hash,
        "reference_hash": reference_hash,
        "judge_prompt_version": judge_prompt_version,
        "rubric_version": rubric_version,
        "system_prompt_hash": hash_text(system_prompt),
        "user_prompt_hash": hash_text(user_prompt),
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
        "reasoning": reasoning,
        "structured_output_schema_version": structured_output_schema_version,
    }
    return hash_obj(material), material
