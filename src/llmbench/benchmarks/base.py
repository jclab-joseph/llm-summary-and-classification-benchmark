"""Shared task representation.

A *task* is one OpenRouter request. For summarization and hallucination that is
one benchmark case; for classification it is a batch of 20 cases, which is where
most of the cost saving comes from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llmbench.config.loader import AppConfig
from llmbench.config.schema import ModelConfig
from llmbench.core.text import estimate_tokens
from llmbench.cache.keys import build_inference_key
from llmbench.prompts.registry import RenderedPrompt

__all__ = ["Task", "TaskGroup", "make_task"]


@dataclass(slots=True)
class Task:
    """One cacheable OpenRouter request."""

    benchmark: str
    benchmark_version: str
    language: str
    task_id: str
    sample_ids: list[str]
    dataset_revision: str
    split: str
    sample_content_hash: str
    prompt: RenderedPrompt
    max_output_tokens: int
    cache_key: str
    key_material: dict[str, Any]
    est_input_tokens: int
    est_output_tokens: int
    response_format: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def case_count(self) -> int:
        return len(self.sample_ids)


@dataclass(slots=True)
class TaskGroup:
    benchmark: str
    tasks: list[Task]
    languages: list[str]
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def case_count(self) -> int:
        return sum(t.case_count for t in self.tasks)


def make_task(
    *,
    cfg: AppConfig,
    model: ModelConfig,
    benchmark: str,
    benchmark_version: str,
    language: str,
    task_id: str,
    sample_ids: list[str],
    dataset_revision: str,
    split: str,
    sample_content_hash: str,
    prompt: RenderedPrompt,
    answer_tokens: int,
    response_format: dict[str, Any] | None = None,
    alias_resolution: str | None = None,
    meta: dict[str, Any] | None = None,
) -> Task:
    """Build a task and its inference cache key in one place.

    Every benchmark goes through here, so the key material can never drift
    between benchmarks.
    """
    generation = cfg.benchmark.generation
    # A thinking model needs room for reasoning tokens on top of the answer.
    max_output_tokens = model.output_token_budget(answer_tokens)
    # Parameters the model does not accept are omitted from the request, so they
    # are omitted from the key material too.
    temperature = generation.temperature if model.supports("temperature") else None
    top_p = generation.top_p if model.supports("top_p") else None
    seed = generation.seed if model.supports("seed") else None
    reasoning = model.reasoning.cache_material() if model.supports("reasoning") else None
    cache_key, material = build_inference_key(
        model_id=model.model_id,
        routing_config_hash=model.routing.config_hash,
        benchmark=benchmark,
        benchmark_version=benchmark_version,
        dataset_revision=dataset_revision,
        split=split,
        sample_id=task_id,
        sample_content_hash=sample_content_hash,
        system_prompt=prompt.system,
        user_prompt=prompt.user,
        prompt_template_version=prompt.template_version,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
        seed=seed,
        tools_enabled=generation.tools_enabled,
        web_search_enabled=generation.web_search_enabled,
        structured_output_schema_version=cfg.benchmark.structured_output.version,
        truncation_policy_version=cfg.benchmark.truncation.version,
        generation_config_version=generation.version,
        alias_resolution=alias_resolution,
    )
    return Task(
        benchmark=benchmark,
        benchmark_version=benchmark_version,
        language=language,
        task_id=task_id,
        sample_ids=sample_ids,
        dataset_revision=dataset_revision,
        split=split,
        sample_content_hash=sample_content_hash,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        cache_key=cache_key,
        key_material=material.to_dict(),
        est_input_tokens=estimate_tokens(prompt.system) + estimate_tokens(prompt.user) + 8,
        est_output_tokens=model.estimated_output_tokens(answer_tokens),
        response_format=response_format,
        meta=meta or {},
    )
