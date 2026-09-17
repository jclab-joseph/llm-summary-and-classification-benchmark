"""GGUF inference via llama.cpp, with the answer constrained by a GBNF grammar."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from llmbench.core.errors import BenchmarkError
from llmbench.local.engine import LocalResult
from llmbench.logging_setup import get_logger

__all__ = ["LlamaCppEngine", "candidate_grammar"]

log = get_logger("local.llama_cpp")


def candidate_grammar(candidates: Sequence[str]) -> str:
    """GBNF allowing exactly one of ``candidates`` and nothing else.

    Decoding is masked to the grammar at every step, so the model cannot emit a
    label that does not exist, cannot add prose around it, and cannot run long.
    """
    if not candidates:
        raise BenchmarkError("a constrained classification needs at least one candidate")
    alternatives = " | ".join('"' + c.replace("\\", "\\\\").replace('"', '\\"') + '"' for c in candidates)
    return f"root ::= {alternatives}"


class LlamaCppEngine:
    """Local GGUF engine.

    The label list lives in the system message and the utterance in the user
    message, so every request shares a long identical prefix that llama.cpp
    reuses from its KV cache instead of re-prefilling.
    """

    def __init__(self, model_path: Path, runtime: dict[str, Any]) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - optional extra
            raise BenchmarkError(
                "llama-cpp-python is not installed; run `uv sync --extra local`"
            ) from exc

        if not Path(model_path).is_file():
            raise BenchmarkError(f"local model file not found: {model_path}")

        self.model_path = Path(model_path)
        self.runtime = dict(runtime)
        self._grammars: dict[tuple[str, ...], Any] = {}

        log.info("loading %s (n_ctx=%s, n_gpu_layers=%s)", self.model_path.name,
                 runtime.get("n_ctx"), runtime.get("n_gpu_layers"))
        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=int(runtime.get("n_ctx", 4096)),
            n_threads=runtime.get("n_threads"),
            n_gpu_layers=int(runtime.get("n_gpu_layers", 0)),
            n_batch=int(runtime.get("n_batch", 512)),
            seed=int(runtime.get("seed", 0)),
            logits_all=False,
            verbose=False,
        )

    # ------------------------------------------------------------------ #
    def _grammar(self, candidates: Sequence[str]):
        from llama_cpp import LlamaGrammar

        key = tuple(candidates)
        grammar = self._grammars.get(key)
        if grammar is None:
            grammar = LlamaGrammar.from_string(candidate_grammar(candidates), verbose=False)
            self._grammars[key] = grammar
        return grammar

    def classify(self, *, system: str, user: str, candidates: Sequence[str]) -> LocalResult:
        started = time.perf_counter()
        # A label is short; the grammar stops generation on its own, and this cap
        # is only a guard against a pathological tokenization.
        max_tokens = int(self.runtime.get("max_label_tokens", 32))
        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            grammar=self._grammar(candidates),
            temperature=float(self.runtime.get("temperature", 0.0)),
            top_p=float(self.runtime.get("top_p", 1.0)),
            max_tokens=max_tokens,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        label = (response["choices"][0]["message"]["content"] or "").strip()
        usage = response.get("usage") or {}
        return LocalResult(
            label=label,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=latency_ms,
            raw=response,
        )

    def describe(self) -> dict[str, Any]:
        import llama_cpp

        return {
            "engine": "llama_cpp",
            "engine_version": llama_cpp.__version__,
            "model_file": self.model_path.name,
            "decoding": "grammar-constrained",
            "runtime": {
                key: self.runtime.get(key)
                for key in ("n_ctx", "n_threads", "n_gpu_layers", "n_batch", "seed", "temperature", "top_p")
            },
        }

    def close(self) -> None:
        self._llm = None
        self._grammars.clear()
