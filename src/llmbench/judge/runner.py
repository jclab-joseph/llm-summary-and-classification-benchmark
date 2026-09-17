"""Optional LLM-as-a-Judge pass.

Off by default: no headline number in this benchmark comes from a model's
opinion. When enabled, the judge is reached through OpenRouter like every other
model -- there is no direct vendor API call anywhere in this project.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence

from llmbench.cache.keys import build_judge_key
from llmbench.config.loader import AppConfig
from llmbench.config.schema import ModelConfig
from llmbench.core.canonical import hash_text
from llmbench.datasets.manifest import Manifest
from llmbench.db.store import Store
from llmbench.openrouter.client import OpenRouterClient
from llmbench.openrouter.types import GenerationRequest, InferenceStatus
from llmbench.prompts.registry import JUDGE_JSON_SCHEMA, judge_prompt

__all__ = ["JudgeRunner", "JudgeOutcome", "SELF_JUDGE_WARNING"]

SELF_JUDGE_WARNING = "SELF_JUDGE_WARNING"

_DIMENSIONS = ("factual_consistency", "key_information_coverage", "conciseness", "overall_quality")
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(slots=True)
class JudgeOutcome:
    enabled: bool
    profile: str | None = None
    judge_model: str | None = None
    target_model: str | None = None
    self_judge: bool = False
    warnings: list[str] = field(default_factory=list)
    requested_samples: int = 0
    cache_hits: int = 0
    api_calls: int = 0
    invalid_outputs: int = 0
    cost_usd: float = 0.0
    scores: dict[str, Any] = field(default_factory=dict)
    status: str = "DISABLED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "profile": self.profile,
            "judge_model": self.judge_model,
            "target_model": self.target_model,
            "self_judge": self.self_judge,
            "warnings": self.warnings,
            "requested_samples": self.requested_samples,
            "cache_hits": self.cache_hits,
            "api_calls": self.api_calls,
            "invalid_outputs": self.invalid_outputs,
            "judge_cost_usd": self.cost_usd,
            "scores": self.scores,
            "primary_factuality_source": "deterministic metrics (judge is advisory only)",
        }


def parse_judgement(text: str) -> dict[str, Any] | None:
    """Parse the judge's JSON reply, tolerating surrounding prose/code fences."""
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate.split("\n", 1)[-1] if "\n" in candidate else candidate
    match = _JSON_BLOCK.search(candidate)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, Any] = {}
    for key in _DIMENSIONS:
        value = data.get(key)
        if not isinstance(value, (int, float)):
            return None
        score = int(value)
        if not 1 <= score <= 5:
            return None
        out[key] = score
    out["rationale"] = str(data.get("rationale", ""))[:500]
    return out


class JudgeRunner:
    def __init__(self, cfg: AppConfig, store: Store) -> None:
        self.cfg = cfg
        self.store = store

    def _generation(self):
        return self.cfg.judge.judge.generation

    def build_cases(
        self,
        manifest: Manifest,
        target_model: ModelConfig,
        summarization_results: dict[str, dict[str, Any]],
        *,
        languages: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """One judge case per generated summary in the frozen judge manifest."""
        wanted = set(languages) if languages else None
        cases: list[dict[str, Any]] = []
        for record in manifest.records:
            if wanted and record.language not in wanted:
                continue
            result = summarization_results.get(record.sample_id)
            if result is None or result.get("status") != "SUCCESS":
                continue
            generated = result.get("output_text", "")
            if not generated.strip():
                continue
            cases.append(
                {
                    "record": record,
                    "generated": generated,
                    "summary_hash": hash_text(generated),
                }
            )
        if limit is not None:
            # Deterministic truncation: manifest order, never a random subset.
            cases = cases[:limit]
        return cases

    async def run(
        self,
        *,
        profile: str,
        target_model: ModelConfig,
        manifest: Manifest,
        summarization_results: dict[str, dict[str, Any]],
        client: OpenRouterClient,
        languages: Sequence[str] | None = None,
        samples: int | None = None,
        run_id: str | None = None,
    ) -> JudgeOutcome:
        judge_cfg = self.cfg.judge.judge
        judge_model = self.cfg.judge_model(profile)
        outcome = JudgeOutcome(
            enabled=True,
            profile=profile,
            judge_model=judge_model.model_id,
            target_model=target_model.model_id,
            status="RUNNING",
        )

        if judge_model.model_id == target_model.model_id:
            outcome.self_judge = True
            outcome.warnings.append(
                f"{SELF_JUDGE_WARNING}: {judge_model.model_id} is judging its own output; "
                "these scores are reported as advisory only and never as the primary "
                "factuality score."
            )

        cases = self.build_cases(
            manifest, target_model, summarization_results, languages=languages, limit=samples
        )
        outcome.requested_samples = len(cases)
        if not cases:
            outcome.status = "NO_SUMMARIES"
            return outcome

        generation = self._generation()
        response_format = (
            JUDGE_JSON_SCHEMA if self.cfg.benchmark.structured_output.judge_mode == "json_schema" else None
        )
        semaphore = asyncio.Semaphore(self.cfg.benchmark.concurrency.max_parallel_requests)
        lock = asyncio.Lock()
        judgements: list[dict[str, Any]] = []

        async def judge_one(case: dict[str, Any]) -> None:
            record = case["record"]
            prompt = judge_prompt(
                record.language,
                record.payload["source"],
                record.payload["reference"],
                case["generated"],
            )
            judge_temperature = generation.temperature if judge_model.supports("temperature") else None
            judge_top_p = generation.top_p if judge_model.supports("top_p") else None
            judge_reasoning = (
                generation.reasoning.cache_material() if judge_model.supports("reasoning") else None
            )
            cache_key, material = build_judge_key(
                judge_model_id=judge_model.model_id,
                judge_routing_config_hash=judge_model.routing.config_hash,
                target_model_id=target_model.model_id,
                sample_id=record.sample_id,
                summary_hash=case["summary_hash"],
                source_hash=record.source_hash,
                reference_hash=record.reference_hash,
                judge_prompt_version=judge_cfg.prompt_version,
                rubric_version=judge_cfg.rubric_version,
                system_prompt=prompt.system,
                user_prompt=prompt.user,
                temperature=judge_temperature,
                top_p=judge_top_p,
                max_output_tokens=judge_model.output_token_budget(generation.max_output_tokens),
                reasoning=judge_reasoning,
                structured_output_schema_version=self.cfg.benchmark.structured_output.version,
            )

            cached = self.store.get_judge_result(cache_key)
            if cached is not None and cached.status == "SUCCESS":
                async with lock:
                    outcome.cache_hits += 1
                    judgements.append({**cached.scores_json, "language": record.language})
                return

            async with semaphore:
                response = await client.generate(
                    GenerationRequest(
                        model=judge_model.model_id,
                        messages=prompt.messages,
                        max_tokens=judge_model.output_token_budget(generation.max_output_tokens),
                        temperature=(
                            generation.temperature if judge_model.supports("temperature") else None
                        ),
                        top_p=generation.top_p if judge_model.supports("top_p") else None,
                        reasoning=(
                            generation.reasoning.to_request_payload()
                            if judge_model.supports("reasoning")
                            else None
                        ),
                        provider=judge_model.routing.to_request_payload(),
                        response_format=(
                            response_format if judge_model.supports_json_schema() else None
                        ),
                    )
                )

            parsed = parse_judgement(response.text) if response.ok else None
            status = (
                str(response.status)
                if not response.ok
                else (str(InferenceStatus.SUCCESS) if parsed else str(InferenceStatus.INVALID_OUTPUT))
            )
            self.store.save_judge_result(
                {
                    "cache_key": cache_key,
                    "run_id": run_id,
                    "judge_model": judge_model.model_id,
                    "judge_profile": profile,
                    "target_model": target_model.model_id,
                    "benchmark": "summarization",
                    "sample_id": record.sample_id,
                    "language": record.language,
                    "status": status,
                    "self_judge": outcome.self_judge,
                    "scores_json": parsed or {},
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "usage_cost": response.usage.cost,
                    "raw_response_json": response.raw,
                    "key_material_json": material,
                    "latency_ms": response.latency_ms,
                    "error": response.error,
                }
            )
            async with lock:
                outcome.api_calls += 1
                for record_attempt in response.attempt_records:
                    outcome.cost_usd += record_attempt.usage.billed_cost
                if parsed:
                    judgements.append({**parsed, "language": record.language})
                else:
                    outcome.invalid_outputs += 1

        await asyncio.gather(*(judge_one(case) for case in cases))

        outcome.scores = self._aggregate(judgements)
        outcome.status = "COMPLETED"
        return outcome

    @staticmethod
    def _aggregate(judgements: Sequence[dict[str, Any]]) -> dict[str, Any]:
        def mean_for(rows: Sequence[dict[str, Any]], key: str) -> float | None:
            values = [float(r[key]) for r in rows if key in r]
            return float(statistics.fmean(values)) if values else None

        out: dict[str, Any] = {"judged_cases": len(judgements), "per_language": {}}
        for dimension in _DIMENSIONS:
            out[dimension] = mean_for(judgements, dimension)
        for language in sorted({r.get("language", "") for r in judgements}):
            rows = [r for r in judgements if r.get("language") == language]
            out["per_language"][language] = {
                "judged_cases": len(rows),
                **{dimension: mean_for(rows, dimension) for dimension in _DIMENSIONS},
            }
        return out
