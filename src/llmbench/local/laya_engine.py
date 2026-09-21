"""Laya: a non-autoregressive decision model as a local classification backend.

https://huggingface.co/convaiinnovations/laya-multilingual

Unlike the other local engines this is not a causal LM at all. It is an mmBERT
encoder with a decision head that scores every option at its own `[MASK]` token
and softmaxes over that question's options, so the answer comes out of one
forward pass with a probability attached -- there is no text to generate and none
to parse.

That makes it the same *guarantee* the other engines buy differently: a
constrained decoder masks generation to the label set, this one never generates.
It is a different *shape* of request, though: it takes a state and a typed
question rather than a chat prompt, so its models are configured with
`prompt_style: state`.

**Read its score with the vendor's own caveat.** The model card says to keep
`choice` questions under about 20 options, because options share a fixed
256-token head budget and a larger label space leaves only a few tokens per
label. This benchmark asks for one of 60 intents -- three times that -- so the
number here is outside the documented envelope and understates what the model
does at the size it was built for.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Sequence

from llmbench.core.errors import BenchmarkError
from llmbench.local.engine import LocalResult
from llmbench.logging_setup import get_logger

__all__ = ["LayaEngine", "MAX_DOCUMENTED_OPTIONS"]

log = get_logger("local.laya")

#: The model card's documented ceiling for a `choice` question.
MAX_DOCUMENTED_OPTIONS = 20


class LayaEngine:
    """Scores the label set in a single forward pass."""

    def __init__(self, model_path: Path, runtime: dict[str, Any]) -> None:
        # transformers probes for TensorFlow at import and its abseil runtime can
        # deadlock model construction; the vendor documents this exact fix.
        os.environ.setdefault("USE_TF", "0")
        try:
            import laya
        except ImportError as exc:  # pragma: no cover - optional extra
            raise BenchmarkError(
                "the laya package is not installed; run `uv sync --extra laya`"
            ) from exc

        self.repo = str(model_path)
        self.runtime = dict(runtime)
        self.question_id = str(runtime.get("question_id", "intent"))
        self._warned_options = False

        device = runtime.get("device") or None
        log.info("loading %s (device=%s)", self.repo, device or "auto")
        started = time.perf_counter()
        self._agent = laya.load(self.repo, device=device)
        self._laya_version = getattr(laya, "__version__", "unknown")
        log.info("loaded in %.1fs", time.perf_counter() - started)

    # ------------------------------------------------------------------ #
    def classify(self, *, system: str, user: str, candidates: Sequence[str]) -> LocalResult:
        if not candidates:
            raise BenchmarkError("a constrained classification needs at least one candidate")
        if len(candidates) > MAX_DOCUMENTED_OPTIONS and not self._warned_options:
            self._warned_options = True
            log.warning(
                "%d options: the model card documents a ceiling of ~%d for a choice question, "
                "so this score is outside the supported envelope",
                len(candidates),
                MAX_DOCUMENTED_OPTIONS,
            )

        started = time.perf_counter()
        question = {
            self.question_id: {
                "type": "choice",
                "instructions": system,
                # The label name is its own description: it is all the generative
                # models get too, so neither side is handed extra context.
                "criteria": {label: label.replace("_", " ") for label in candidates},
            }
        }
        try:
            output = self._agent.predict({"utterance": user}, question)
        except Exception as exc:
            raise BenchmarkError(f"laya.predict failed: {type(exc).__name__}: {exc}") from exc

        answer = (output.get("answers") or {}).get(self.question_id) or {}
        label = answer.get("choice")
        if label not in set(candidates):
            raise BenchmarkError(f"laya returned an option outside the label set: {label!r}")

        usage = output.get("usage") or {}
        return LocalResult(
            label=label,
            prompt_tokens=int(usage.get("tokens") or usage.get("input_tokens") or 0),
            # Nothing is generated; the answer is a distribution over the options.
            completion_tokens=0,
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw={
                "probability": answer.get("confidence"),
                "action": answer.get("action"),
                "decoding": "non-autoregressive decision head",
                "options": len(candidates),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "engine": "laya",
            "engine_version": self._laya_version,
            "checkpoint": self.repo,
            "decoding": "non-autoregressive (option markers + per-question softmax)",
            "documented_option_ceiling": MAX_DOCUMENTED_OPTIONS,
            "runtime": {key: self.runtime.get(key) for key in ("device", "question_id")},
        }

    def close(self) -> None:
        self._agent = None
