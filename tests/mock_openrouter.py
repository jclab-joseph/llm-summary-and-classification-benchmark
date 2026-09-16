"""An in-process mock of the OpenRouter HTTP API.

This is deliberately a mock *transport*, not a mock provider class: the real
`OpenRouterClient` -- with its real request building, retry loop, usage parsing
and cost accounting -- runs unchanged against it. A fake provider object would
test nothing about the code that actually talks to OpenRouter.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

# USD per token, chosen so that costs are small but exactly representable.
MOCK_INPUT_RATE = 0.10 / 1_000_000
MOCK_OUTPUT_RATE = 0.40 / 1_000_000

MOCK_PRICING = {
    "google/gemini-2.5-flash-lite": {"prompt": "0.0000001", "completion": "0.0000004", "input_cache_read": "0.000000025"},
    "openai/gpt-5.6-luna": {"prompt": "0.0000002", "completion": "0.0000012"},
    "openai/gpt-5.6-sol": {"prompt": "0.00000125", "completion": "0.00001"},
    "mistralai/ministral-3b-2512": {"prompt": "0.0000001", "completion": "0.0000001"},
}

_FULL_PARAMETERS = [
    "max_tokens", "temperature", "top_p", "seed", "stop",
    "reasoning", "include_reasoning", "response_format", "structured_outputs",
]
# Mirrors the real API: the OpenAI reasoning models advertise no temperature/top_p,
# which is exactly what produced the 404 this guard exists to prevent.
_NO_SAMPLING = ["max_tokens", "seed", "reasoning", "include_reasoning", "response_format", "structured_outputs"]

MOCK_SUPPORTED_PARAMETERS = {
    "google/gemini-2.5-flash-lite": _FULL_PARAMETERS,
    "openai/gpt-5.6-luna": _NO_SAMPLING,
    "openai/gpt-5.6-sol": _NO_SAMPLING,
    "mistralai/ministral-3b-2512": ["max_tokens", "temperature", "top_p", "seed", "stop", "response_format"],
}


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _first_sentences(text: str, count: int = 2) -> str:
    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return " ".join(parts[:count]).strip()


@dataclass
class MockOpenRouter:
    """Scriptable OpenRouter stand-in.

    ``fail_first_n`` makes the first N chat requests fail so retry behaviour and
    billed-but-failed attempts can be exercised.
    """

    fail_first_n: int = 0
    fail_status: int = 429
    fail_cost: float | None = None
    error_for_sample: dict[str, int] = field(default_factory=dict)
    chat_calls: int = 0
    models_calls: int = 0
    requests: list[dict[str, Any]] = field(default_factory=list)
    total_cost: float = 0.0

    # ------------------------------------------------------------------ #
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def reset_counters(self) -> None:
        self.chat_calls = 0
        self.models_calls = 0
        self.requests.clear()

    # ------------------------------------------------------------------ #
    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            self.models_calls += 1
            return httpx.Response(200, json={"data": [
                {
                    "id": model_id,
                    "name": model_id,
                    "pricing": pricing,
                    "context_length": 128000,
                    "supported_parameters": MOCK_SUPPORTED_PARAMETERS.get(model_id, _FULL_PARAMETERS),
                }
                for model_id, pricing in MOCK_PRICING.items()
            ]})

        if not request.url.path.endswith("/chat/completions"):
            return httpx.Response(404, json={"error": {"message": "not found", "code": 404}})

        payload = json.loads(request.content.decode("utf-8"))
        self.requests.append(payload)
        self.chat_calls += 1

        # Reproduce OpenRouter's behaviour with provider.require_parameters=true:
        # an unsupported parameter eliminates every endpoint and yields a 404.
        provider = payload.get("provider") or {}
        if provider.get("require_parameters"):
            supported = set(MOCK_SUPPORTED_PARAMETERS.get(payload["model"], _FULL_PARAMETERS))
            sent = {k for k in ("temperature", "top_p", "seed", "stop", "reasoning", "response_format") if k in payload}
            unsupported = sorted(sent - supported)
            if unsupported:
                return httpx.Response(
                    404,
                    json={
                        "error": {
                            "message": (
                                "No endpoints found that can handle the requested parameters. "
                                f"(unsupported: {', '.join(unsupported)})"
                            ),
                            "code": 404,
                        }
                    },
                )

        # A vendor key must never be required: only the OpenRouter bearer token
        # is allowed to appear on the wire.
        for header in request.headers:
            assert not header.lower().startswith(
                ("openai-", "x-goog-", "anthropic-", "mistral-")
            ), f"vendor-specific header leaked to OpenRouter: {header}"

        if self.chat_calls <= self.fail_first_n:
            body: dict[str, Any] = {
                "error": {"message": "mock rate limit", "code": self.fail_status},
                "id": f"gen-fail-{self.chat_calls}",
            }
            if self.fail_cost:
                body["usage"] = {
                    "prompt_tokens": 100,
                    "completion_tokens": 0,
                    "total_tokens": 100,
                    "cost": self.fail_cost,
                    "cost_details": {"upstream_inference_cost": self.fail_cost},
                }
                self.total_cost += self.fail_cost
            return httpx.Response(self.fail_status, json=body)

        messages = payload["messages"]
        user = next(m["content"] for m in messages if m["role"] == "user")
        content = self._answer(user, payload)

        prompt_tokens = sum(_approx_tokens(m["content"]) for m in messages)
        completion_tokens = min(_approx_tokens(content), payload["max_tokens"])
        cost = round(prompt_tokens * MOCK_INPUT_RATE + completion_tokens * MOCK_OUTPUT_RATE, 10)
        self.total_cost += cost

        return httpx.Response(
            200,
            json={
                "id": f"gen-{_stable_int(user):08x}",
                "object": "chat.completion",
                "created": 1750000000,
                "model": payload["model"],
                "provider": "MockProvider",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "native_finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                    "completion_tokens_details": {"reasoning_tokens": 0},
                    "cost": cost,
                    "cost_details": {"upstream_inference_cost": cost},
                },
            },
        )

    # ------------------------------------------------------------------ #
    def _answer(self, user: str, payload: dict[str, Any]) -> str:
        if "SUPPORTED" in user and "HALLUCINATED" in user:
            return "SUPPORTED" if _stable_int(user) % 2 == 0 else "HALLUCINATED"

        if "Intents:" in user or "인텐트 목록:" in user:
            return self._classification_answer(user)

        if "Generated summary:" in user or "생성된 요약:" in user:
            seed = _stable_int(user)
            scores = {
                "factual_consistency": seed % 5 + 1,
                "key_information_coverage": (seed // 5) % 5 + 1,
                "conciseness": (seed // 25) % 5 + 1,
                "overall_quality": (seed // 125) % 5 + 1,
                "rationale": "mock judgement",
            }
            return json.dumps(scores)

        # Summarization: extractive lead, which is a realistic weak baseline.
        body = user.split("Article:\n", 1)[-1].split("기사:\n", 1)[-1]
        body = body.rsplit("\n\nSummary:", 1)[0].rsplit("\n\n요약:", 1)[0]
        return _first_sentences(body, 2) or "No summary available."

    @staticmethod
    def _classification_answer(user: str) -> str:
        label_block = user.split("Intents:\n", 1)[-1].split("인텐트 목록:\n", 1)[-1]
        label_block = label_block.split("\n\n", 1)[0]
        label_count = len([line for line in label_block.splitlines() if re.match(r"^\d+: ", line)])
        item_block = user.split("Utterances:\n", 1)[-1].split("발화:\n", 1)[-1]
        item_block = item_block.split("\n\n", 1)[0]
        items = [line for line in item_block.splitlines() if re.match(r"^\d+: ", line)]
        lines = []
        for index, item in enumerate(items, start=1):
            choice = (_stable_int(item) % max(1, label_count)) + 1
            lines.append(f"{index}: {choice}")
        return "\n".join(lines)
