"""A local engine behind the same interface the runner uses for OpenRouter.

Presenting the engine as a client is what lets local models reuse the whole
pipeline unchanged -- the inference cache, the attempt log, the scoring, the
reports -- without the runner growing a second code path.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from llmbench.core.errors import DryRunViolation
from llmbench.local.engine import LocalEngine, build_engine
from llmbench.logging_setup import get_logger
from llmbench.openrouter.types import (
    AttemptRecord,
    GenerationRequest,
    GenerationResponse,
    InferenceStatus,
    Usage,
)

__all__ = ["LocalEngineClient"]

log = get_logger("local.client")


class LocalEngineClient:
    """Runs constrained classification on a local engine.

    Cost is `None`, not `0.0`: nothing was billed, which is a different statement
    from "billed zero", and the reports render it as "not applicable" rather than
    as a free hosted model.
    """

    def __init__(
        self,
        *,
        engine: str,
        model_id: str,
        model_path: Path,
        runtime: dict[str, Any],
        dry_run: bool = False,
    ) -> None:
        self.engine_name = engine
        self.model_id = model_id
        self.model_path = Path(model_path)
        self.runtime = runtime
        self.dry_run = dry_run
        self._engine: LocalEngine | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    @property
    def engine(self) -> LocalEngine:
        if self._engine is None:
            self._engine = build_engine(self.engine_name, self.model_path, self.runtime)
        return self._engine

    async def aclose(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    async def __aenter__(self) -> LocalEngineClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def get_model_metadata(self, model_ids: list[str] | None = None) -> dict[str, Any]:
        """No hosted metadata exists for a local model, and no pricing applies."""
        return {}

    async def get_model_pricing(self, model_ids: list[str] | None = None) -> dict[str, Any]:
        return {}

    # ------------------------------------------------------------------ #
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self.dry_run:
            raise DryRunViolation(
                "LocalEngineClient.generate() called during a dry run; "
                "dry runs must never run inference"
            )
        if not request.candidates:
            return self._failure(
                request,
                "the local engine only supports constrained classification, "
                "which needs a candidate set",
            )

        system = next((m["content"] for m in request.messages if m["role"] == "system"), "")
        user = next((m["content"] for m in request.messages if m["role"] == "user"), "")

        started = time.perf_counter()
        try:
            # llama.cpp holds one mutable context, so calls are serialized. The
            # runner's concurrency setting still applies to hosted models.
            async with self._lock:
                result = await asyncio.to_thread(
                    self.engine.classify, system=system, user=user, candidates=list(request.candidates)
                )
        except Exception as exc:  # engine failures must not look like model output
            log.warning("local engine error model=%s: %s", self.model_id, exc)
            return self._failure(request, f"{type(exc).__name__}: {exc}")

        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
            cost=None,  # nothing was billed; see the class docstring
            raw={"local": True, "engine": self.engine_name},
        )
        response = GenerationResponse(
            status=InferenceStatus.SUCCESS,
            text=result.label,
            usage=usage,
            requested_model=request.model,
            resolved_model=self.model_path.name,
            upstream_provider=f"local:{self.engine_name}",
            request_id=None,
            finish_reason="stop",
            latency_ms=latency_ms,
            raw={"local": True, "label": result.label},
        )
        response.attempt_records = [
            AttemptRecord(
                attempt_no=1,
                status=InferenceStatus.SUCCESS,
                resolved_model=response.resolved_model,
                upstream_provider=response.upstream_provider,
                usage=usage,
                latency_ms=latency_ms,
            )
        ]
        return response

    def _failure(self, request: GenerationRequest, error: str) -> GenerationResponse:
        response = GenerationResponse(
            status=InferenceStatus.OPENROUTER_ERROR,
            text="",
            usage=Usage(cost=None),
            requested_model=request.model,
            upstream_provider=f"local:{self.engine_name}",
            error=error,
        )
        response.attempt_records = [
            AttemptRecord(attempt_no=1, status=InferenceStatus.OPENROUTER_ERROR, error=error)
        ]
        return response
