"""Pydantic models for every configuration file.

Design rule from the spec: prices are never hardcoded in code, and the benchmark
core never branches on model vendor. Vendor-specific behaviour is expressed as
configuration on the model entry (reasoning, max_output_tokens, routing).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llmbench.core.canonical import hash_obj


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ReasoningConfig(StrictModel):
    """Reasoning/thinking configuration. Disabled by default for this benchmark.

    Some models refuse to run with thinking off ("Reasoning is mandatory for this
    endpoint and cannot be disabled"). For those, enable it at the lowest effort
    -- the spec asks for reasoning "disabled or minimized" -- and give the request
    headroom, because reasoning tokens are counted against `max_tokens`. Without
    headroom the hallucination benchmark's 8-token answer budget leaves the model
    no room to think, and it returns nothing at all.
    """

    enabled: bool = False
    effort: Literal["low", "medium", "high"] | None = None
    max_tokens: int | None = None
    exclude: bool = True  # do not stream reasoning back; we never score it
    # Extra output tokens allowed on top of the answer budget when thinking is on.
    output_token_headroom: int = 0
    # Typical reasoning length, used only for the pre-run cost estimate. The
    # headroom is a ceiling; billing follows what was actually generated.
    estimated_output_tokens: int = 0

    def to_request_payload(self) -> dict[str, Any] | None:
        """OpenRouter `reasoning` request field, or None when disabled."""
        if not self.enabled:
            # OpenRouter disables thinking for hybrid models with enabled=false.
            return {"enabled": False, "exclude": True}
        payload: dict[str, Any] = {"enabled": True, "exclude": self.exclude}
        if self.effort:
            payload["effort"] = self.effort
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        return payload

    def cache_material(self) -> dict[str, Any]:
        # `output_token_headroom` is deliberately absent: its only effect is on
        # max_output_tokens, which the key already records. `estimated_output_tokens`
        # never reaches the API at all.
        return {
            "enabled": self.enabled,
            "effort": self.effort,
            "max_tokens": self.max_tokens,
            "exclude": self.exclude,
        }


class RoutingConfig(StrictModel):
    """OpenRouter provider-routing configuration.

    This is reproducibility-relevant: the same slug can be served by different
    upstream endpoints, so routing settings are part of the inference cache key.
    """

    mode: Literal["default", "pinned", "restricted"] = "default"
    provider_order: list[str] | None = None
    allow_fallbacks: bool = True
    only: list[str] | None = None
    ignore: list[str] | None = None
    require_parameters: bool = True
    data_collection: Literal["allow", "deny"] = "allow"
    sort: Literal["price", "throughput", "latency"] | None = None
    quantizations: list[str] | None = None

    @model_validator(mode="after")
    def _check_mode(self) -> RoutingConfig:
        if self.mode == "pinned" and not self.provider_order:
            raise ValueError("routing.mode='pinned' requires provider_order")
        if self.mode == "restricted" and not (self.only or self.ignore):
            raise ValueError("routing.mode='restricted' requires 'only' or 'ignore'")
        return self

    def to_request_payload(self) -> dict[str, Any] | None:
        """The OpenRouter `provider` request field."""
        payload: dict[str, Any] = {}
        if self.provider_order:
            payload["order"] = list(self.provider_order)
        if self.only:
            payload["only"] = list(self.only)
        if self.ignore:
            payload["ignore"] = list(self.ignore)
        if self.quantizations:
            payload["quantizations"] = list(self.quantizations)
        if self.sort:
            payload["sort"] = self.sort
        payload["allow_fallbacks"] = self.allow_fallbacks
        payload["require_parameters"] = self.require_parameters
        payload["data_collection"] = self.data_collection
        return payload or None

    def cache_material(self) -> dict[str, Any]:
        """Fully-expanded routing view, so defaults hash identically everywhere."""
        return {
            "mode": self.mode,
            "provider_order": list(self.provider_order) if self.provider_order else None,
            "allow_fallbacks": self.allow_fallbacks,
            "only": list(self.only) if self.only else None,
            "ignore": list(self.ignore) if self.ignore else None,
            "require_parameters": self.require_parameters,
            "data_collection": self.data_collection,
            "sort": self.sort,
            "quantizations": list(self.quantizations) if self.quantizations else None,
        }

    @property
    def config_hash(self) -> str:
        return hash_obj(self.cache_material())


class ModelParameters(StrictModel):
    """Which request parameters this OpenRouter model actually accepts.

    OpenRouter publishes a `supported_parameters` list per model. With
    `routing.require_parameters: true` (our default, which is what stops an
    endpoint from silently ignoring `temperature=0`), sending a parameter the
    model does not support makes OpenRouter reject the request with
    HTTP 404 "No endpoints found that can handle the requested parameters".

    So unsupported parameters are omitted from the request instead. They are
    omitted from the cache key material too, because the key must describe what
    was really sent.
    """

    temperature: bool = True
    top_p: bool = True
    seed: bool = True
    stop: bool = True
    reasoning: bool = True
    response_format: bool = True

    def as_dict(self) -> dict[str, bool]:
        return self.model_dump()

    def unsupported(self) -> list[str]:
        return sorted(name for name, supported in self.model_dump().items() if not supported)


# OpenRouter's `supported_parameters` entry names, mapped to our flags. A model
# is treated as supporting a flag when ANY of the listed names is present.
OPENROUTER_PARAMETER_NAMES: dict[str, tuple[str, ...]] = {
    "temperature": ("temperature",),
    "top_p": ("top_p",),
    "seed": ("seed",),
    "stop": ("stop",),
    "reasoning": ("reasoning", "include_reasoning", "reasoning_effort"),
    "response_format": ("response_format", "structured_outputs"),
}


class ModelConfig(StrictModel):
    provider: Literal["openrouter"] = "openrouter"
    model_id: str
    display_name: str
    vendor: str
    enabled: bool = True
    role: Literal["target", "cheap", "strong"] = "target"
    max_output_tokens: int = 256
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    parameters: ModelParameters = Field(default_factory=ModelParameters)

    @field_validator("model_id")
    @classmethod
    def _slug_shape(cls, v: str) -> str:
        if "/" not in v:
            raise ValueError(f"'{v}' is not an OpenRouter model slug (expected 'vendor/model')")
        return v

    @property
    def safe_slug(self) -> str:
        """Filesystem-safe form of the slug, e.g. google__gemini-2.5-flash-lite."""
        return self.model_id.replace("/", "__")

    def supports(self, parameter: str) -> bool:
        return bool(getattr(self.parameters, parameter, True))

    def output_token_budget(self, answer_tokens: int) -> int:
        """`max_tokens` to send for a request whose answer should fit in ``answer_tokens``.

        Reasoning tokens count against the same budget, so a thinking model gets
        the configured headroom on top. ``max_output_tokens`` is the hard ceiling.
        """
        budget = answer_tokens
        if self.reasoning.enabled:
            budget += self.reasoning.output_token_headroom
        return min(budget, self.max_output_tokens)

    def estimated_output_tokens(self, answer_tokens: int) -> int:
        """Expected output length, for the pre-run cost estimate only."""
        estimate = answer_tokens
        if self.reasoning.enabled:
            estimate += self.reasoning.estimated_output_tokens
        return min(estimate, self.output_token_budget(answer_tokens))

    def capability_mismatches(self, supported_parameters: list[str]) -> list[dict[str, Any]]:
        """Compare the configured flags against OpenRouter's live metadata.

        Returns one entry per parameter we would send that the model does not
        advertise -- the exact condition that produces the 404 above.
        """
        live = {name.lower() for name in supported_parameters}
        problems: list[dict[str, Any]] = []
        for flag, names in OPENROUTER_PARAMETER_NAMES.items():
            configured = self.supports(flag)
            advertised = any(name in live for name in names)
            if configured and not advertised:
                problems.append(
                    {
                        "parameter": flag,
                        "configured": True,
                        "advertised": False,
                        "severity": "error",
                        "hint": f"set parameters.{flag}: false for {self.model_id} in config/models.yaml",
                    }
                )
            elif not configured and advertised:
                problems.append(
                    {
                        "parameter": flag,
                        "configured": False,
                        "advertised": True,
                        "severity": "warning",
                        "hint": f"{self.model_id} now supports {flag}; parameters.{flag}: true would use it",
                    }
                )
        return problems

    @property
    def uses_alias(self) -> bool:
        """`:latest`-style aliases make the resolved model non-deterministic."""
        tail = self.model_id.rsplit("/", 1)[-1]
        return tail.endswith(":latest") or tail in {"latest"} or ":latest" in self.model_id


class ModelsConfig(StrictModel):
    version: int = 1
    models: list[ModelConfig]
    judges: list[ModelConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _all_openrouter(self) -> ModelsConfig:
        for m in [*self.models, *self.judges]:
            if m.provider != "openrouter":
                raise ValueError(
                    f"model {m.model_id} has provider={m.provider!r}; only 'openrouter' is supported"
                )
        return self

    def by_id(self, model_id: str) -> ModelConfig | None:
        for m in [*self.models, *self.judges]:
            if m.model_id == model_id:
                return m
        return None

    def enabled_models(self) -> list[ModelConfig]:
        return [m for m in self.models if m.enabled]

    def judge_by_role(self, role: str) -> ModelConfig | None:
        for m in self.judges:
            if m.role == role:
                return m
        return None


class PricingEntry(StrictModel):
    input_per_million: float | None = None
    output_per_million: float | None = None
    cached_input_per_million: float | None = None

    @property
    def complete(self) -> bool:
        return self.input_per_million is not None and self.output_per_million is not None


class PricingConstraints(StrictModel):
    max_input_per_million: float
    max_output_per_million: float


class PricingConfig(StrictModel):
    version: int = 1
    currency: str = "USD"
    source: str = "config"
    constraints: PricingConstraints
    models: dict[str, PricingEntry]

    def get(self, model_id: str) -> PricingEntry:
        return self.models.get(model_id, PricingEntry())

    def within_constraints(self, model_id: str) -> bool | None:
        entry = self.get(model_id)
        if not entry.complete:
            return None
        return (
            entry.input_per_million <= self.constraints.max_input_per_million
            and entry.output_per_million <= self.constraints.max_output_per_million
        )


# --------------------------------------------------------------------------- #
# benchmark.yaml
# --------------------------------------------------------------------------- #
class PathsConfig(StrictModel):
    manifests: str = "data/manifests"
    cache_db: str = "data/cache/benchmark.db"
    raw_data: str = "data/raw"
    results: str = "results"
    result_markdown: str = "RESULT.md"
    result_excel: str = "results/benchmark_results.xlsx"


class SamplingConfig(StrictModel):
    seed: int = 20240917


class MaxOutputTokens(StrictModel):
    summarization: int = 200
    hallucination: int = 8
    classification: int = 320


class GenerationConfig(StrictModel):
    version: str = "gen-1"
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    tools_enabled: bool = False
    web_search_enabled: bool = False
    max_output_tokens: MaxOutputTokens = Field(default_factory=MaxOutputTokens)

    def cache_material(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "tools_enabled": self.tools_enabled,
            "web_search_enabled": self.web_search_enabled,
        }


class TruncationConfig(StrictModel):
    version: str = "trunc-1"
    summarization_max_source_tokens: int = 1800
    hallucination_max_source_tokens: int = 1600
    word_boundary: bool = True

    def cache_material(self) -> dict[str, Any]:
        return self.model_dump()


class StructuredOutputConfig(StrictModel):
    version: str = "so-1"
    classification_mode: Literal["text", "json_schema"] = "text"
    judge_mode: Literal["text", "json_schema"] = "json_schema"


class ConcurrencyConfig(StrictModel):
    max_parallel_requests: int = 8
    request_timeout_seconds: float = 180.0


class RetryConfig(StrictModel):
    max_attempts: int = 4
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    jitter_seconds: float = 0.5


class FailureGuardConfig(StrictModel):
    enabled: bool = True
    max_consecutive_failures: int = 10
    max_failure_rate: float = 0.5
    min_attempts_before_rate_check: int = 20


class LoggingConfig(StrictModel):
    directory: str = "data/logs"
    keep_last: int = 50


class BudgetConfig(StrictModel):
    default_usd: float = 1.00
    stop_margin: float = 0.98


class CostConfig(StrictModel):
    authoritative_source: str = "openrouter_usage_cost"
    reconciliation_tolerance_usd: float = 5e-6
    reconciliation_relative_tolerance: float = 0.02


class TokenEnvelopeEntry(StrictModel):
    input: int
    output: int


class BenchmarkCases(StrictModel):
    en: int
    ko: int


class SummarizationBenchmarkConfig(StrictModel):
    name: str = "summarization"
    version: str = "1.0.0"
    manifest: str = "xlsum_v1.jsonl"
    cases: BenchmarkCases


class HallucinationManifests(StrictModel):
    en: str
    ko: str


class HallucinationBenchmarkConfig(StrictModel):
    name: str = "hallucination"
    version: str = "1.0.0"
    manifests: HallucinationManifests
    documents: BenchmarkCases


class ClassificationBenchmarkConfig(StrictModel):
    name: str = "classification"
    version: str = "1.0.0"
    manifest: str = "massive_v1.jsonl"
    cases: BenchmarkCases
    batch_size: int = 20


class BenchmarksConfig(StrictModel):
    summarization: SummarizationBenchmarkConfig
    hallucination: HallucinationBenchmarkConfig
    classification: ClassificationBenchmarkConfig


class RougeMetricConfig(StrictModel):
    version: str = "rouge-1.1"
    tokenizer: str = "multilingual-v1"
    use_stemmer: bool = False


class ChrfMetricConfig(StrictModel):
    version: str = "chrf-1.0"
    word_order: int = 2
    char_order: int = 6
    beta: int = 2


class BertScoreMetricConfig(StrictModel):
    version: str = "bertscore-1.0"
    enabled: bool = True
    model_type: str = "bert-base-multilingual-cased"
    num_layers: int = 9
    batch_size: int = 32
    idf: bool = False


class SurfaceFactsMetricConfig(StrictModel):
    version: str = "surface-facts-1.1"
    check_numbers: bool = True
    check_percentages: bool = True
    check_dates: bool = True
    check_currency: bool = True
    check_entities: bool = True
    numeric_tolerance: float = 0.0


class SimpleMetricConfig(StrictModel):
    version: str


class MetricsConfig(StrictModel):
    rouge: RougeMetricConfig = Field(default_factory=RougeMetricConfig)
    chrf: ChrfMetricConfig = Field(default_factory=ChrfMetricConfig)
    bertscore: BertScoreMetricConfig = Field(default_factory=BertScoreMetricConfig)
    surface_facts: SurfaceFactsMetricConfig = Field(default_factory=SurfaceFactsMetricConfig)
    classification_metrics: SimpleMetricConfig = Field(
        default_factory=lambda: SimpleMetricConfig(version="clsmetrics-1.0")
    )
    hallucination_metrics: SimpleMetricConfig = Field(
        default_factory=lambda: SimpleMetricConfig(version="halumetrics-1.0")
    )


class XlsumSourceFiles(StrictModel):
    en: str
    ko: str


class XlsumSource(StrictModel):
    repo: str
    revision: str
    files: XlsumSourceFiles
    split: str = "test"


class HaluEvalSource(StrictModel):
    repo: str
    revision: str
    config: str
    split: str
    file: str


class MassiveSource(StrictModel):
    repo: str
    revision: str
    split: str
    files: XlsumSourceFiles


class AihubSource(StrictModel):
    env_var: str = "AIHUB_FACTUALITY_DATA_PATH"
    revision: str = "local"
    preferred_error_type: str = "type5"
    allow_non_factual_fallback: bool = False
    exclude_omission_only_errors: bool = True
    exclude_broken_prefix_summaries: bool = True
    split: Literal["Training", "Validation"] | None = "Validation"
    require_untruncated_source: bool = True


class DatasetSources(StrictModel):
    xlsum: XlsumSource
    halueval: HaluEvalSource
    massive: MassiveSource
    aihub: AihubSource


class TokenEnvelope(StrictModel):
    summarization: TokenEnvelopeEntry
    hallucination: TokenEnvelopeEntry
    classification: TokenEnvelopeEntry


class BenchmarkConfig(StrictModel):
    version: int = 1
    benchmark_version: str = "1.0.0"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    truncation: TruncationConfig = Field(default_factory=TruncationConfig)
    structured_output: StructuredOutputConfig = Field(default_factory=StructuredOutputConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    failure_guard: FailureGuardConfig = Field(default_factory=FailureGuardConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    token_envelope: TokenEnvelope
    benchmarks: BenchmarksConfig
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    dataset_sources: DatasetSources


# --------------------------------------------------------------------------- #
# judge.yaml
# --------------------------------------------------------------------------- #
class JudgeSamples(StrictModel):
    total: int = 200
    en: int = 100
    ko: int = 100


class JudgeGeneration(StrictModel):
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 512
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)


class JudgeProfile(StrictModel):
    model_id: str


class JudgeSettings(StrictModel):
    enabled: bool = False
    prompt_version: str = "judge-1"
    rubric_version: str = "rubric-1"
    samples: JudgeSamples = Field(default_factory=JudgeSamples)
    manifest: str = "judge_xlsum_v1.jsonl"
    generation: JudgeGeneration = Field(default_factory=JudgeGeneration)
    profiles: dict[str, JudgeProfile] = Field(default_factory=dict)
    self_judge_policy: str = "warn_and_demote"


class JudgeConfig(StrictModel):
    version: int = 1
    judge: JudgeSettings = Field(default_factory=JudgeSettings)
