"""Parallel Constrained Decoding over a GGUF model.

The technique comes from the "Parallel Constrained Decoding" engine published at
https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD. Instead of *generating* the
answer token by token and hoping it lands on a valid label, it:

1. prefills the shared context once into a KV cache;
2. reads the logits at that position and slices them to the candidate tokens
   ("sub-vocabulary logit slicing"), so the whole label set is evaluated from a
   single forward pass rather than one pass per token of a generated answer;
3. walks a token tree for candidates that share a prefix, continuing from the
   cached state ("token tree disambiguation");
4. turns the candidate scores into calibrated probabilities with a softmax over
   the candidate slice alone.

Two things differ from the reference implementation, both because the benchmark's
label space breaks its assumptions:

* The reference compares candidates by their **first token only**. On MASSIVE's
  60 intents that is fatal -- `alarm_set`, `alarm_query` and `alarm_remove` all
  begin with token 56780, so they are indistinguishable and argmax over the
  duplicated ids silently returns whichever came first. Its Linux/PyTorch backend
  does not implement the disambiguation step at all, and its MLX backend falls
  back to free generation plus fuzzy string matching.
* Here the tree is walked to the point where a candidate is uniquely determined,
  accumulating log-probabilities, so candidates are ranked on **complete
  sequences**. That also makes the ranking non-greedy: a label whose first token
  is not the argmax can still win on total probability, which grammar-guided
  decoding can never do.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from llmbench.core.errors import BenchmarkError
from llmbench.local.engine import LocalResult
from llmbench.logging_setup import get_logger

__all__ = ["ParallelConstrainedEngine", "TokenTree", "build_token_tree"]

log = get_logger("local.pcd")


@dataclass(slots=True)
class _Search:
    """Mutable state of one best-first tree walk."""

    scores: dict[int, float] = field(default_factory=dict)
    # Upper bounds for candidates whose branch was pruned; used only so the
    # calibrated probabilities still cover every candidate.
    bounds: dict[int, float] = field(default_factory=dict)
    best: int | None = None
    best_score: float = -math.inf
    pruned: int = 0


@dataclass(slots=True)
class TokenTree:
    """Prefix tree over the candidates' token sequences."""

    children: dict[int, "TokenTree"] = field(default_factory=dict)
    # Candidate index when this node completes exactly one candidate.
    candidate: int | None = None
    # Candidates still reachable below this node.
    reachable: list[int] = field(default_factory=list)

    @property
    def is_branch(self) -> bool:
        return bool(self.children)

    def count_nodes(self) -> int:
        return 1 + sum(child.count_nodes() for child in self.children.values())

    def count_forward_passes(self) -> int:
        """Forwards needed to score every candidate: one per node with children.

        The root is free -- its logits come from the prefill everyone shares.
        """
        if not self.children:
            return 0
        return sum(1 + child.count_forward_passes() for child in self.children.values() if child.children)


def build_token_tree(tokenized: Sequence[Sequence[int]]) -> TokenTree:
    """Build the tree, stopping each candidate as soon as it is unambiguous.

    Once no other candidate shares the path there is nothing left to compare, so
    the remaining tokens are forced and contribute no ranking information. That
    is what keeps this to a handful of forward passes instead of one per token of
    every label.
    """
    root = TokenTree(reachable=list(range(len(tokenized))))

    def insert(node: TokenTree, tokens: Sequence[int], index: int, depth: int) -> None:
        if depth >= len(tokens):
            node.candidate = index
            return
        token = tokens[depth]
        child = node.children.get(token)
        if child is None:
            child = TokenTree()
            node.children[token] = child
        child.reachable.append(index)
        insert(child, tokens, index, depth + 1)

    for index, tokens in enumerate(tokenized):
        if not tokens:
            raise BenchmarkError("a candidate tokenized to nothing")
        insert(root, tokens, index, 0)

    def prune(node: TokenTree) -> None:
        for child in list(node.children.values()):
            if len(child.reachable) == 1 and child.candidate is None:
                child.candidate = child.reachable[0]
                child.children.clear()
            else:
                prune(child)

    prune(root)
    return root


class ParallelConstrainedEngine:
    """GGUF backend that scores the candidate set instead of generating it."""

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
        self._trees: dict[tuple[str, ...], tuple[TokenTree, list[list[int]]]] = {}
        self._prefix_tokens: list[int] = []
        self.telemetry: dict[str, Any] = {"prefill_tokens": 0, "forward_passes": 0, "prefix_reused": 0}

        log.info("loading %s for parallel constrained decoding", self.model_path.name)
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
    def _tree_for(self, candidates: Sequence[str]) -> tuple[TokenTree, list[list[int]]]:
        key = tuple(candidates)
        cached = self._trees.get(key)
        if cached is None:
            tokenized = [
                list(self._llm.tokenize(c.encode("utf-8"), add_bos=False, special=False))
                for c in candidates
            ]
            tree = build_token_tree(tokenized)
            log.info(
                "token tree: %d candidates, %d nodes, %d forward passes per utterance",
                len(candidates),
                tree.count_nodes(),
                tree.count_forward_passes(),
            )
            cached = (tree, tokenized)
            self._trees[key] = cached
        return cached

    def _logits(self) -> np.ndarray:
        """Logits for the last evaluated position, read from the context.

        Not `Llama.scores`: with `logits_all=False` llama-cpp-python 0.3.x stops
        populating that array entirely (the write is commented out because
        sampling moved inside the sampler), so reading it returns zeros and every
        candidate scores identically.
        """
        return np.ctypeslib.as_array(
            self._llm._ctx.get_logits(), shape=(self._llm._n_vocab,)
        ).astype(np.float32, copy=True)

    def _prefill(self, tokens: list[int]) -> np.ndarray:
        """Evaluate the prompt, reusing the KV cache for the shared prefix.

        Every utterance in a run carries the same system message -- the whole
        label catalog -- so the reusable prefix is most of the prompt.
        """
        shared = 0
        for a, b in zip(self._prefix_tokens, tokens):
            if a != b:
                break
            shared += 1
        # Always re-evaluate at least the final token so fresh logits exist.
        shared = min(shared, len(tokens) - 1)

        if shared > 0:
            self._llm._ctx.kv_cache_seq_rm(-1, shared, -1)
            self._llm.n_tokens = shared
        else:
            self._llm.reset()

        self.telemetry["prefix_reused"] += shared
        self.telemetry["prefill_tokens"] += len(tokens) - shared
        self._llm.eval(tokens[shared:])
        self._prefix_tokens = list(tokens)
        return self._logits()

    def _walk(self, node: TokenTree, logits: np.ndarray, base: float, state: _Search) -> None:
        """Accumulate log-probabilities down the tree from the cached state.

        Branches are explored best-first and pruned against the best complete
        candidate found so far. Log-probabilities are non-positive, so a node's
        score is an upper bound on every candidate beneath it: a branch that
        already scores below a finished candidate cannot contain the winner.
        The argmax stays exact while the number of forward passes collapses.
        """
        # Sub-vocabulary slicing: normalize over the whole vocabulary once, then
        # read only the candidate tokens out of it.
        shifted = logits - logits.max()
        log_norm = math.log(float(np.exp(shifted).sum()))

        ranked = sorted(
            ((float(shifted[token]) - log_norm, token, child) for token, child in node.children.items()),
            key=lambda item: item[0],
            reverse=True,
        )

        for delta, token, child in ranked:
            score = base + delta
            if score <= state.best_score:
                # Upper bound already loses; record the bound for calibration.
                for index in child.reachable:
                    state.bounds[index] = max(state.bounds.get(index, -math.inf), score)
                state.pruned += 1
                continue

            if child.candidate is not None:
                state.scores[child.candidate] = score
                if score > state.best_score:
                    state.best_score = score
                    state.best = child.candidate
            if not child.children:
                continue

            anchor = self._llm.n_tokens
            self._llm.eval([token])
            self.telemetry["forward_passes"] += 1
            child_logits = self._logits()
            self._walk(child, child_logits, score, state)
            # Rewind to this node so the next sibling starts from the same state.
            self._llm._ctx.kv_cache_seq_rm(-1, anchor, -1)
            self._llm.n_tokens = anchor

    # ------------------------------------------------------------------ #
    def classify(self, *, system: str, user: str, candidates: Sequence[str]) -> LocalResult:
        if not candidates:
            raise BenchmarkError("a constrained classification needs at least one candidate")

        started = time.perf_counter()
        tree, _ = self._tree_for(candidates)
        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        tokens = list(self._llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True))
        forwards_before = self.telemetry["forward_passes"]

        logits = self._prefill(tokens)
        state = _Search()
        self._walk(tree, logits, 0.0, state)

        if state.best is None:
            raise BenchmarkError("the token tree produced no candidate scores")

        # Calibrated probabilities over the candidate slice alone. Pruned
        # candidates contribute their upper bound, which can only understate the
        # winner's confidence -- never overstate it.
        best = state.best
        values = np.array(
            [state.scores.get(i, state.bounds.get(i, -math.inf)) for i in range(len(candidates))],
            dtype=np.float64,
        )
        exp = np.exp(values - values.max())
        probs = exp / exp.sum()

        latency_ms = int((time.perf_counter() - started) * 1000)
        return LocalResult(
            label=candidates[best],
            prompt_tokens=len(tokens),
            # Nothing is generated: the answer is read out of the logits.
            completion_tokens=0,
            latency_ms=latency_ms,
            raw={
                "probability": float(probs[best]),
                "forward_passes": self.telemetry["forward_passes"] - forwards_before,
                "pruned_branches": state.pruned,
                "decoding": "parallel-constrained",
            },
        )

    def describe(self) -> dict[str, Any]:
        import llama_cpp

        return {
            "engine": "llama_cpp_pcd",
            "engine_version": llama_cpp.__version__,
            "model_file": self.model_path.name,
            "decoding": "parallel-constrained (prefill + logit slicing + token tree)",
            "telemetry": dict(self.telemetry),
            "runtime": {
                key: self.runtime.get(key)
                for key in ("n_ctx", "n_threads", "n_gpu_layers", "n_batch", "seed", "temperature", "top_p")
            },
        }

    def close(self) -> None:
        self._llm = None
        self._trees.clear()
