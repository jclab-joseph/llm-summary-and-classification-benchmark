"""The single provider adapter in this project.

There is deliberately no ``OpenAIProvider`` / ``GoogleProvider`` / ``MistralProvider``.
Every model -- target or judge -- is reached through this one class, and vendor
differences are expressed as configuration in ``config/models.yaml``.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Any

import httpx

from llmbench.core.errors import DryRunViolation
from llmbench.logging_setup import get_logger
from llmbench.openrouter.types import (
    AttemptRecord,
    GenerationRequest,
    GenerationResponse,
    InferenceStatus,
    ModelPricing,
    Usage,
)

__all__ = ["OpenRouterClient", "DEFAULT_BASE_URL"]

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

log = get_logger("openrouter")

_RETRYABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class OpenRouterClient:
    """Async OpenRouter client with retries, usage/cost parsing and a dry-run guard.

    ``transport`` exists so tests can inject an ``httpx.MockTransport`` and drive a
    full end-to-end benchmark without touching the network.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 180.0,
        max_attempts: int = 4,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        jitter: float = 0.5,
        http_referer: str | None = None,
        app_name: str | None = None,
        dry_run: bool = False,
        sleep: Any = asyncio.sleep,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_attempts = max(1, int(max_attempts))
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.jitter = jitter
        self.http_referer = http_referer if http_referer is not None else os.environ.get("OPENROUTER_HTTP_REFERER")
        self.app_name = app_name if app_name is not None else os.environ.get("OPENROUTER_APP_NAME", "llm-benchmark")
        self.dry_run = dry_run
        self._sleep = sleep
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # An empty bearer token is an illegal header value, and the public
        # /models endpoint needs no auth at all -- so `benchmark pricing` and
        # `--dry-run` still work before a key is configured.
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers(),
                timeout=self.timeout,
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> OpenRouterClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ #
    # parsing helpers (pure, unit-testable)
    # ------------------------------------------------------------------ #
    @staticmethod
    def parse_usage(raw: dict[str, Any] | None) -> Usage:
        """Extract the usage block, preserving OpenRouter's own fields verbatim."""
        usage_raw = (raw or {}).get("usage") or {}
        if not isinstance(usage_raw, dict):
            return Usage()
        prompt_details = usage_raw.get("prompt_tokens_details") or {}
        completion_details = usage_raw.get("completion_tokens_details") or {}
        if not isinstance(prompt_details, dict):
            prompt_details = {}
        if not isinstance(completion_details, dict):
            completion_details = {}
        cost_details = usage_raw.get("cost_details") or {}
        if not isinstance(cost_details, dict):
            cost_details = {}

        prompt_tokens = _to_int(usage_raw.get("prompt_tokens"))
        completion_tokens = _to_int(usage_raw.get("completion_tokens"))
        total = _to_int(usage_raw.get("total_tokens")) or (prompt_tokens + completion_tokens)

        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            cached_tokens=_to_int(prompt_details.get("cached_tokens")),
            cache_write_tokens=_to_int(
                prompt_details.get("cache_write_tokens", prompt_details.get("cache_creation_tokens"))
            ),
            reasoning_tokens=_to_int(completion_details.get("reasoning_tokens")),
            cost=_to_float(usage_raw.get("cost")),
            cost_details=cost_details,
            raw=usage_raw,
        )

    @staticmethod
    def parse_actual_cost(raw: dict[str, Any] | None) -> float | None:
        """`usage.cost` -- the authoritative billed amount for this request."""
        return OpenRouterClient.parse_usage(raw).cost

    @staticmethod
    def _extract_text(raw: dict[str, Any]) -> tuple[str, str | None]:
        choices = raw.get("choices") or []
        if not choices:
            return "", None
        choice = choices[0] or {}
        finish_reason = choice.get("finish_reason") or choice.get("native_finish_reason")
        message = choice.get("message") or {}
        content = message.get("content")
        if content is None:
            content = choice.get("text")
        if isinstance(content, list):
            # Some upstreams return content parts.
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part.get("text") or "")
                else:
                    parts.append(str(part))
            content = "".join(parts)
        return (content or ""), finish_reason

    def normalize_response(
        self,
        raw: dict[str, Any],
        *,
        requested_model: str,
        latency_ms: int = 0,
    ) -> GenerationResponse:
        """Turn a raw OpenRouter body into the project's canonical response shape."""
        usage = self.parse_usage(raw)
        text, finish_reason = self._extract_text(raw)
        provider = raw.get("provider")
        if isinstance(provider, dict):
            provider = provider.get("name")

        error = raw.get("error")
        if error:
            message = error.get("message") if isinstance(error, dict) else str(error)
            code = error.get("code") if isinstance(error, dict) else None
            status = (
                InferenceStatus.RATE_LIMIT
                if code in (429, "429", "rate_limit_exceeded")
                else InferenceStatus.OPENROUTER_ERROR
            )
            return GenerationResponse(
                status=status,
                text="",
                usage=usage,
                requested_model=requested_model,
                resolved_model=raw.get("model"),
                upstream_provider=provider,
                request_id=raw.get("id"),
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                error=str(message),
                raw=raw,
            )

        return GenerationResponse(
            status=InferenceStatus.SUCCESS,
            text=text,
            usage=usage,
            requested_model=requested_model,
            resolved_model=raw.get("model"),
            upstream_provider=provider,
            request_id=raw.get("id"),
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            raw=raw,
        )

    # ------------------------------------------------------------------ #
    # generation
    # ------------------------------------------------------------------ #
    def _backoff_delay(self, attempt_no: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self.max_backoff)
        delay = min(self.initial_backoff * (2 ** (attempt_no - 1)), self.max_backoff)
        return delay + random.uniform(0, self.jitter)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Execute one generation, retrying transient failures.

        Every physical attempt is recorded in ``response.attempt_records`` -- including
        failed ones, whose ``usage.cost`` is preserved so billed-but-failed calls are
        never lost from the cost accounting.
        """
        if self.dry_run:
            raise DryRunViolation(
                "OpenRouterClient.generate() called during a dry run; "
                "dry runs must never touch the inference API"
            )

        payload = request.to_payload()
        records: list[AttemptRecord] = []
        last: GenerationResponse | None = None

        for attempt_no in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            retry_after: float | None = None
            try:
                http_response = await self.client.post("/chat/completions", json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                log.warning(
                    "transport error model=%s attempt=%d/%d: %s",
                    request.model,
                    attempt_no,
                    self.max_attempts,
                    f"{type(exc).__name__}: {exc}",
                )
                records.append(
                    AttemptRecord(
                        attempt_no=attempt_no,
                        status=InferenceStatus.OPENROUTER_ERROR,
                        latency_ms=latency_ms,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                last = GenerationResponse(
                    status=InferenceStatus.OPENROUTER_ERROR,
                    text="",
                    usage=Usage(),
                    requested_model=request.model,
                    latency_ms=latency_ms,
                    error=f"{type(exc).__name__}: {exc}",
                )
                if attempt_no < self.max_attempts:
                    await self._sleep(self._backoff_delay(attempt_no, None))
                    continue
                break

            latency_ms = int((time.perf_counter() - started) * 1000)
            http_status = http_response.status_code
            try:
                raw = http_response.json()
            except ValueError:
                raw = {}

            if http_status >= 400 or (isinstance(raw, dict) and raw.get("error")):
                usage = self.parse_usage(raw if isinstance(raw, dict) else {})
                status = (
                    InferenceStatus.RATE_LIMIT
                    if http_status == 429
                    else InferenceStatus.OPENROUTER_ERROR
                )
                err_body = raw.get("error") if isinstance(raw, dict) else None
                message = (
                    err_body.get("message")
                    if isinstance(err_body, dict)
                    else (http_response.text[:500] if http_response.text else f"HTTP {http_status}")
                )
                log.warning(
                    "openrouter error model=%s http=%s attempt=%d/%d cost=%s: %s",
                    request.model,
                    http_status,
                    attempt_no,
                    self.max_attempts,
                    usage.cost,
                    str(message)[:400],
                )
                records.append(
                    AttemptRecord(
                        attempt_no=attempt_no,
                        status=status,
                        request_id=raw.get("id") if isinstance(raw, dict) else None,
                        usage=usage,
                        latency_ms=latency_ms,
                        http_status=http_status,
                        error=str(message),
                    )
                )
                last = GenerationResponse(
                    status=status,
                    text="",
                    usage=usage,
                    requested_model=request.model,
                    request_id=raw.get("id") if isinstance(raw, dict) else None,
                    latency_ms=latency_ms,
                    error=str(message),
                    raw=raw if isinstance(raw, dict) else {},
                )
                retryable = http_status in _RETRYABLE_HTTP or http_status == 0
                if retryable and attempt_no < self.max_attempts:
                    header = http_response.headers.get("retry-after")
                    retry_after = _to_float(header)
                    await self._sleep(self._backoff_delay(attempt_no, retry_after))
                    continue
                break

            if not isinstance(raw, dict):
                records.append(
                    AttemptRecord(
                        attempt_no=attempt_no,
                        status=InferenceStatus.PARSE_ERROR,
                        latency_ms=latency_ms,
                        http_status=http_status,
                        error="response body was not a JSON object",
                    )
                )
                last = GenerationResponse(
                    status=InferenceStatus.PARSE_ERROR,
                    text="",
                    usage=Usage(),
                    requested_model=request.model,
                    latency_ms=latency_ms,
                    error="response body was not a JSON object",
                )
                break

            response = self.normalize_response(raw, requested_model=request.model, latency_ms=latency_ms)
            log.debug(
                "ok model=%s resolved=%s provider=%s prompt=%d completion=%d cost=%s latency=%dms",
                request.model,
                response.resolved_model,
                response.upstream_provider,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.cost,
                latency_ms,
            )
            records.append(
                AttemptRecord(
                    attempt_no=attempt_no,
                    status=response.status,
                    request_id=response.request_id,
                    resolved_model=response.resolved_model,
                    upstream_provider=response.upstream_provider,
                    usage=response.usage,
                    latency_ms=latency_ms,
                    http_status=http_status,
                    error=response.error,
                )
            )
            response.attempts = attempt_no
            response.attempt_records = records
            return response

        assert last is not None
        last.attempts = len(records)
        last.attempt_records = records
        return last

    # ------------------------------------------------------------------ #
    # pricing metadata
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pricing_from_entry(entry: dict[str, Any], retrieved_at: str) -> ModelPricing:
        """OpenRouter reports USD *per token*; the benchmark works per 1M tokens."""
        pricing = entry.get("pricing") or {}
        prompt = _to_float(pricing.get("prompt"))
        completion = _to_float(pricing.get("completion"))
        cached = _to_float(pricing.get("input_cache_read"))
        scale = 1_000_000.0

        def per_million(value: float | None) -> float | None:
            # Rounded because 1e-7 * 1e6 is 0.09999999999999999 in binary floats,
            # which would otherwise leak into snapshots and reports.
            return None if value is None else round(value * scale, 10)

        return ModelPricing(
            model_id=entry.get("id", ""),
            input_per_million=per_million(prompt),
            output_per_million=per_million(completion),
            cached_input_per_million=per_million(cached),
            retrieved_at=retrieved_at,
            source="openrouter",
            raw=pricing,
        )

    async def get_model_metadata(self, model_ids: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """Fetch /models metadata: pricing plus the supported-parameter list.

        One call serves both the budget estimate and the pre-run capability check,
        so a run never pays for the same metadata twice. Allowed during a dry run:
        it is metadata, not inference.
        """
        from llmbench.core.reproducibility import utc_now_iso

        retrieved_at = utc_now_iso()
        response = await self.client.get("/models")
        response.raise_for_status()
        body = response.json()
        entries = body.get("data") or []
        wanted = set(model_ids) if model_ids else None
        out: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id")
            if not model_id:
                continue
            if wanted is not None and model_id not in wanted:
                continue
            supported = entry.get("supported_parameters") or []
            out[model_id] = {
                "pricing": self._pricing_from_entry(entry, retrieved_at),
                "supported_parameters": [str(name) for name in supported],
                "context_length": entry.get("context_length"),
                "retrieved_at": retrieved_at,
            }
        log.debug("fetched metadata for %d model(s)", len(out))
        return out

    async def get_model_pricing(self, model_ids: list[str] | None = None) -> dict[str, ModelPricing]:
        """Current pricing metadata, keyed by OpenRouter model slug."""
        metadata = await self.get_model_metadata(model_ids)
        return {model_id: entry["pricing"] for model_id, entry in metadata.items()}
