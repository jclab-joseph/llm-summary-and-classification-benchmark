"""OpenRouter adapter behaviour, exercised through a mock HTTP transport."""

from __future__ import annotations

import httpx
import pytest

from llmbench.core.errors import DryRunViolation
from llmbench.openrouter.client import OpenRouterClient
from llmbench.openrouter.types import GenerationRequest, InferenceStatus
from mock_openrouter import MockOpenRouter


def make_request(model: str = "google/gemini-2.5-flash-lite") -> GenerationRequest:
    return GenerationRequest(
        model=model,
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "Summarize: hello."}],
        max_tokens=64,
    )


async def test_generate_parses_usage_and_cost(client_factory, mock_openrouter):
    async with client_factory() as client:
        response = await client.generate(make_request())

    assert response.status is InferenceStatus.SUCCESS
    assert response.text
    assert response.upstream_provider == "MockProvider"
    assert response.resolved_model == "google/gemini-2.5-flash-lite"
    assert response.request_id
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.cost == pytest.approx(mock_openrouter.total_cost)
    assert response.usage.cost_details


async def test_usage_request_flag_is_sent(client_factory, mock_openrouter):
    async with client_factory() as client:
        await client.generate(make_request())
    assert mock_openrouter.requests[0]["usage"] == {"include": True}


async def test_no_vendor_api_key_is_required(monkeypatch, mock_openrouter):
    """(15) Only OPENROUTER_API_KEY is ever used."""
    for vendor_key in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "MISTRAL_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(vendor_key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    client = OpenRouterClient(transport=mock_openrouter.transport())
    try:
        response = await client.generate(make_request())
    finally:
        await client.aclose()

    assert response.ok
    headers = client._headers()
    assert headers["Authorization"] == "Bearer or-test-key"
    assert not any(h.lower().startswith(("openai", "x-goog", "anthropic", "mistral")) for h in headers)


async def test_retry_with_backoff_then_success(client_factory, mock_openrouter):
    mock_openrouter.fail_first_n = 2
    async with client_factory() as client:
        response = await client.generate(make_request())

    assert response.ok
    assert response.attempts == 3
    assert len(response.attempt_records) == 3
    assert [str(r.status) for r in response.attempt_records[:2]] == ["RATE_LIMIT", "RATE_LIMIT"]


async def test_failed_attempt_cost_is_preserved(client_factory, mock_openrouter):
    """(7) A billed-but-failed attempt keeps its usage.cost."""
    mock_openrouter.fail_first_n = 1
    mock_openrouter.fail_cost = 0.000004
    async with client_factory() as client:
        response = await client.generate(make_request())

    assert response.ok
    failed = response.attempt_records[0]
    assert str(failed.status) == "RATE_LIMIT"
    assert failed.usage.cost == pytest.approx(0.000004)
    total = sum(record.usage.billed_cost for record in response.attempt_records)
    assert total == pytest.approx(mock_openrouter.total_cost)


async def test_exhausted_retries_returns_error_status(client_factory, mock_openrouter):
    mock_openrouter.fail_first_n = 99
    async with client_factory(max_attempts=2) as client:
        response = await client.generate(make_request())

    assert response.status is InferenceStatus.RATE_LIMIT
    assert response.attempts == 2
    assert response.error


async def test_non_retryable_http_error_stops_immediately(mock_openrouter):
    mock_openrouter.fail_first_n = 99
    mock_openrouter.fail_status = 400
    client = OpenRouterClient(api_key="k", transport=mock_openrouter.transport(), max_attempts=4)
    try:
        response = await client.generate(make_request())
    finally:
        await client.aclose()

    assert response.status is InferenceStatus.OPENROUTER_ERROR
    assert response.attempts == 1


async def test_dry_run_client_refuses_to_generate(client_factory):
    async with client_factory() as client:
        client.dry_run = True
        with pytest.raises(DryRunViolation):
            await client.generate(make_request())


async def test_get_model_pricing_converts_to_per_million(client_factory):
    async with client_factory() as client:
        pricing = await client.get_model_pricing(["google/gemini-2.5-flash-lite"])

    entry = pricing["google/gemini-2.5-flash-lite"]
    assert entry.input_per_million == pytest.approx(0.10)
    assert entry.output_per_million == pytest.approx(0.40)
    assert entry.cached_input_per_million == pytest.approx(0.025)
    assert entry.source == "openrouter"


def test_parse_usage_handles_missing_blocks():
    assert OpenRouterClient.parse_usage(None).prompt_tokens == 0
    assert OpenRouterClient.parse_usage({}).cost is None
    usage = OpenRouterClient.parse_usage(
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 2},
                "completion_tokens_details": {"reasoning_tokens": 3},
                "cost": 0.5,
            }
        }
    )
    assert (usage.total_tokens, usage.cached_tokens, usage.cache_write_tokens, usage.reasoning_tokens) == (15, 4, 2, 3)
    assert OpenRouterClient.parse_actual_cost({"usage": {"cost": 0.25}}) == 0.25


async def test_transport_error_is_retried(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(
            200,
            json={
                "id": "gen-1",
                "model": "m",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
            },
        )

    async def no_sleep(_):
        return None

    client = OpenRouterClient(
        api_key="k", transport=httpx.MockTransport(handler), max_attempts=3, sleep=no_sleep
    )
    try:
        response = await client.generate(make_request())
    finally:
        await client.aclose()
    assert response.ok
    assert calls["n"] == 2


async def test_no_api_key_still_allows_public_metadata(mock_openrouter, monkeypatch):
    """An empty bearer token is an illegal header; it must simply be omitted.

    `benchmark pricing` and `--dry-run` are expected to work before a key is
    configured, and OpenRouter's /models endpoint needs no auth.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = OpenRouterClient(api_key="", transport=mock_openrouter.transport())
    try:
        assert "Authorization" not in client._headers()
        pricing = await client.get_model_pricing()
        assert pricing
    finally:
        await client.aclose()
