"""Local inference engines and the constrained-classification contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from llmbench.core.errors import BenchmarkError

__all__ = ["LocalResult", "LocalEngine", "build_engine", "ENGINES"]


@dataclass(slots=True)
class LocalResult:
    """One constrained classification."""

    label: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LocalEngine(Protocol):
    """Chooses one of ``candidates`` for a prompt, by construction.

    "By construction" is the point: the engine is not asked to write a label and
    then parsed, it is constrained to emit one of the candidates, so an
    unparseable answer is impossible. That is the technique the hosted
    `json_schema` mode buys from the provider, run locally instead.
    """

    def classify(self, *, system: str, user: str, candidates: Sequence[str]) -> LocalResult: ...

    def describe(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


ENGINES: dict[str, str] = {
    # Grammar-guided generation: the baseline the parallel engine is measured
    # against. The answer is generated token by token, masked to valid labels.
    "llama_cpp": "llmbench.local.llama_cpp_engine:LlamaCppEngine",
    # Parallel constrained decoding: the answer is read out of the logits.
    "llama_cpp_pcd": "llmbench.local.pcd_engine:ParallelConstrainedEngine",
}


def build_engine(engine: str, model_path: Path, runtime: dict[str, Any]) -> LocalEngine:
    """Instantiate the configured engine.

    Engines are resolved lazily so the heavy runtime is imported only when a
    local model is actually run.
    """
    target = ENGINES.get(engine)
    if target is None:
        raise BenchmarkError(f"unknown local engine '{engine}'. Available: {sorted(ENGINES)}")
    module_name, _, attribute = target.partition(":")
    from importlib import import_module

    try:
        module = import_module(module_name)
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise BenchmarkError(
            f"local engine '{engine}' is not installed: {exc}. "
            "Install it with `uv sync --extra local`."
        ) from exc
    return getattr(module, attribute)(model_path, runtime)
