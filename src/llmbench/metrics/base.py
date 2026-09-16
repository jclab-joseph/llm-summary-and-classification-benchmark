"""Evaluator identity -- the half of the metric cache key that is not the output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llmbench.core.canonical import hash_obj, hash_text

__all__ = ["Evaluator", "inference_result_hash"]


@dataclass(frozen=True, slots=True)
class Evaluator:
    """Name + version + config of one metric implementation.

    Bumping ``version`` (e.g. after switching BERTScore checkpoint) invalidates
    the metric cache while leaving every cached inference untouched.
    """

    name: str
    version: str
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        return hash_obj(self.config)


def inference_result_hash(cache_key: str, output_text: str) -> str:
    """Identity of a scored model output.

    Includes the inference cache key so two models that happen to emit the same
    string still get separate metric rows, and the output hash so a re-run that
    produced different text is not silently scored from cache.
    """
    return hash_obj({"inference_cache_key": cache_key, "output_hash": hash_text(output_text)})
