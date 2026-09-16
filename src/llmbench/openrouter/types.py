"""Data types for the single OpenRouter provider adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "InferenceStatus",
    "Usage",
    "ModelPricing",
    "GenerationRequest",
    "GenerationResponse",
    "AttemptRecord",
]


class InferenceStatus(StrEnum):
    """Outcome of one benchmark case (section 20)."""

    SUCCESS = "SUCCESS"
    OPENROUTER_ERROR = "OPENROUTER_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    PARSE_ERROR = "PARSE_ERROR"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    BUDGET_STOPPED = "BUDGET_STOPPED"
    SKIPPED = "SKIPPED"

    @property
    def is_terminal_success(self) -> bool:
        return self is InferenceStatus.SUCCESS

    @property
    def is_retryable(self) -> bool:
        return self in {
            InferenceStatus.OPENROUTER_ERROR,
            InferenceStatus.RATE_LIMIT,
            InferenceStatus.PARSE_ERROR,
            InferenceStatus.INVALID_OUTPUT,
            InferenceStatus.BUDGET_STOPPED,
        }


@dataclass(slots=True)
class Usage:
    """Usage block of an OpenRouter response.

    ``cost`` is the authoritative billed amount for the request. Everything else
    is recorded verbatim for auditing.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float | None = None
    cost_details: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def billed_cost(self) -> float:
        """usage.cost, or 0.0 when OpenRouter reported none."""
        return float(self.cost) if self.cost is not None else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost": self.cost,
            "cost_details": self.cost_details,
            "raw": self.raw,
        }

    def __add__(self, other: Usage) -> Usage:
        cost: float | None
        if self.cost is None and other.cost is None:
            cost = None
        else:
            cost = self.billed_cost + other.billed_cost
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cost=cost,
            cost_details={},
            raw={},
        )


@dataclass(slots=True)
class ModelPricing:
    """A pricing snapshot for one OpenRouter model slug (USD / 1M tokens)."""

    model_id: str
    input_per_million: float | None
    output_per_million: float | None
    cached_input_per_million: float | None
    retrieved_at: str
    source: str = "openrouter"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.input_per_million is not None and self.output_per_million is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "input_per_million": self.input_per_million,
            "output_per_million": self.output_per_million,
            "cached_input_per_million": self.cached_input_per_million,
            "retrieved_at": self.retrieved_at,
            "source": self.source,
        }


@dataclass(slots=True)
class AttemptRecord:
    """One physical HTTP attempt against OpenRouter."""

    attempt_no: int
    status: InferenceStatus
    request_id: str | None = None
    resolved_model: str | None = None
    upstream_provider: str | None = None
    usage: Usage = field(default_factory=Usage)
    latency_ms: int = 0
    http_status: int | None = None
    error: str | None = None


@dataclass(slots=True)
class GenerationRequest:
    """One chat-completions request. Provider-agnostic fields only."""

    model: str
    messages: list[dict[str, str]]
    max_tokens: int
    # None means "this model does not accept the parameter, so do not send it".
    # Sending an unsupported parameter with provider.require_parameters=true makes
    # OpenRouter return 404 "No endpoints found that can handle the requested
    # parameters" for every endpoint.
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    seed: int | None = None
    stop: list[str] | None = None
    reasoning: dict[str, Any] | None = None
    provider: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            # Ask OpenRouter to include the authoritative billed cost inline so we
            # never have to reconstruct it from a pricing table.
            "usage": {"include": True},
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.stop:
            payload["stop"] = self.stop
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        if self.provider is not None:
            payload["provider"] = self.provider
        if self.response_format is not None:
            payload["response_format"] = self.response_format
        return payload


@dataclass(slots=True)
class GenerationResponse:
    """Normalized OpenRouter response."""

    status: InferenceStatus
    text: str
    usage: Usage
    requested_model: str
    resolved_model: str | None = None
    upstream_provider: str | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    latency_ms: int = 0
    attempts: int = 1
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    # One entry per physical HTTP attempt, including the failed ones. Failed
    # attempts can still have been billed, so their usage.cost is preserved.
    attempt_records: list[AttemptRecord] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status is InferenceStatus.SUCCESS
