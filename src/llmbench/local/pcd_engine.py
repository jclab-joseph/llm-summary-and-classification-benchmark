"""Parallel Constrained Decoding over a GGUF model.

The technique comes from the "Parallel Constrained Decoding" engine published at
https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD. Instead of *generating* the
answer token by token and hoping it lands on a valid label, it:

1. prefills the shared context once into a KV cache;
2. reads the logits at that position and slices them to the candidate tokens
   ("sub-vocabulary logit slicing"), so the whole label set is evaluated from a
   single forward pass rather than one pass per token of a generated answer;
3. walks a token tree for candidates that share a prefix, broadcasting the
   cached prefix into one KV sequence per branch and evaluating a whole tree
   depth in a single batched decode ("KV-cache broadcasting" plus "token tree
   disambiguation");
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

__all__ = ["ParallelConstrainedEngine", "TokenTree", "build_token_tree", "ENGINE_VERSION"]

#: Bumped when the decoding algorithm changes. It is part of the inference cache
#: key, so a change re-runs the benchmark instead of pooling results produced by
#: two different decoders.
ENGINE_VERSION = "pcd-2"

log = get_logger("local.pcd")


def _template_error(message: str) -> None:
    raise BenchmarkError(f"chat template raised: {message}")


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

    #: Sequence 0 always holds the shared prompt prefix; branches use 1..n_seq_max-1.
    PROMPT_SEQ = 0

    def __init__(self, model_path: Path, runtime: dict[str, Any]) -> None:
        try:
            import llama_cpp
            from llama_cpp import _internals
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
        self.telemetry: dict[str, Any] = {
            "prefill_tokens": 0,
            "decodes": 0,
            "branches_evaluated": 0,
            "prefix_reused": 0,
        }

        # The context is built here rather than through `llama_cpp.Llama` for one
        # reason: `Llama.__init__` never exposes `n_seq_max`, and llama.cpp
        # defaults it to 1. Broadcasting the prefix across branches needs more
        # than one sequence, and a context built directly can simply ask for them.
        self.n_seq_max = int(runtime.get("n_seq_max", 32))
        max_parallel = llama_cpp.llama_max_parallel_sequences()
        if self.n_seq_max > max_parallel:
            self.n_seq_max = max_parallel

        model_params = llama_cpp.llama_model_default_params()
        model_params.n_gpu_layers = int(runtime.get("n_gpu_layers", 0))

        log.info(
            "loading %s for parallel constrained decoding (n_seq_max=%d)",
            self.model_path.name,
            self.n_seq_max,
        )
        self._model = _internals.LlamaModel(
            path_model=str(self.model_path), params=model_params, verbose=False
        )

        context_params = llama_cpp.llama_context_default_params()
        context_params.n_ctx = int(runtime.get("n_ctx", 4096))
        context_params.n_batch = int(runtime.get("n_batch", 512))
        context_params.n_ubatch = min(context_params.n_batch, context_params.n_batch)
        context_params.n_seq_max = self.n_seq_max
        threads = runtime.get("n_threads")
        if threads:
            context_params.n_threads = int(threads)
            context_params.n_threads_batch = int(threads)
        if hasattr(context_params, "kv_unified"):
            # Branches share the prompt's cells instead of each holding a copy.
            context_params.kv_unified = True

        self._ctx = _internals.LlamaContext(model=self._model, params=context_params, verbose=False)
        self._batch = _internals.LlamaBatch(
            n_tokens=context_params.n_batch, embd=0, n_seq_max=self.n_seq_max, verbose=False
        )
        self._n_vocab = self._model.n_vocab()
        self._prompt_len = 0
        self._free_seqs: list[int] = list(range(1, self.n_seq_max))
        self._chat_template = self._load_chat_template()
        self._prefix_reuse = True
        self._prefix_tokens = []

    # ------------------------------------------------------------------ #
    # prompt formatting
    # ------------------------------------------------------------------ #
    def _load_chat_template(self):
        """Compile the chat template the GGUF itself carries.

        Hardcoding ChatML works for Qwen2.5 and quietly produces the wrong prompt
        for anything else. The high-level `Llama` wrapper resolves this from GGUF
        metadata; this engine builds its own context, so it has to do the same.
        """
        try:
            template = (self._model.metadata() or {}).get("tokenizer.chat_template")
        except Exception:  # pragma: no cover - metadata is optional
            template = None
        if not template:
            log.warning("%s has no chat template; falling back to ChatML", self.model_path.name)
            return None
        try:
            from jinja2 import Environment
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
            environment.globals["raise_exception"] = _template_error
            environment.globals["strftime_now"] = lambda fmt: ""
            return environment.from_string(template)
        except Exception as exc:  # pragma: no cover - depends on the template
            log.warning("could not compile the chat template (%s); falling back to ChatML", exc)
            return None

    def _render_prompt(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self._chat_template is not None:
            try:
                return self._chat_template.render(
                    messages=messages,
                    add_generation_prompt=True,
                    # Thinking models otherwise open a reasoning block the engine
                    # would then have to score a label inside of.
                    enable_thinking=False,
                    tools=None,
                    bos_token="",
                    eos_token="",
                )
            except Exception as exc:  # pragma: no cover - depends on the template
                log.warning("chat template failed (%s); falling back to ChatML", exc)
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _tree_for(self, candidates: Sequence[str]) -> tuple[TokenTree, list[list[int]]]:
        key = tuple(candidates)
        cached = self._trees.get(key)
        if cached is None:
            tokenized = [
                list(self._model.tokenize(c.encode("utf-8"), add_bos=False, special=False))
                for c in candidates
            ]
            tree = build_token_tree(tokenized)
            log.info(
                "token tree: %d candidates, %d nodes, depth-wise expansion",
                len(candidates),
                tree.count_nodes(),
            )
            cached = (tree, tokenized)
            self._trees[key] = cached
        return cached

    # ------------------------------------------------------------------ #
    # batched decoding
    # ------------------------------------------------------------------ #
    def _decode(self, items: Sequence[tuple[int, int, int, bool]]) -> list[np.ndarray]:
        """Decode ``(token, position, seq_id, want_logits)`` tuples in one pass.

        This is the broadcast step: every branch of the current tree depth lives
        in its own KV sequence sharing the prompt's cells, so one decode produces
        the logits for all of them instead of one decode per branch.

        Only the positions that are actually read ask for logits. Requesting them
        everywhere makes llama.cpp allocate `n_tokens x n_vocab` floats, which for
        a 600-token prefill over a 150k vocabulary is a 365 MB buffer and enough
        to make `llama_decode` fail outright.
        """
        if not items:
            return []
        batch = self._batch.batch
        batch.n_tokens = len(items)
        for index, (token, position, seq_id, want) in enumerate(items):
            batch.token[index] = token
            batch.pos[index] = position
            batch.n_seq_id[index] = 1
            batch.seq_id[index][0] = seq_id
            batch.logits[index] = want
        self._ctx.decode(self._batch)
        self.telemetry["decodes"] += 1
        return [
            np.ctypeslib.as_array(self._ctx.get_logits_ith(index), shape=(self._n_vocab,)).astype(
                np.float32, copy=True
            )
            for index, item in enumerate(items)
            if item[3]
        ]

    def _prefill(self, tokens: list[int]) -> tuple[np.ndarray, int, int]:
        """Evaluate the prompt into sequence 0. Returns (logits, seq, length).

        Every utterance in a run carries the same system message -- the whole
        label catalog -- so the shared prefix is most of the prompt. It is kept in
        sequence 0 and the differing tail is rewound and re-evaluated.

        Multimodal-RoPE models reject that: Qwen3.5 requires each sequence's
        positions to strictly increase, so llama.cpp refuses a batch restarting at
        a position the rewind just freed. Those models fall back to a full prefill.

        Copying the prefix into a fresh sequence would sidestep the position check
        and is *wrong*: M-RoPE positions are not a scalar the copy reconstructs,
        and the run completes with quietly degraded accuracy (measured: Qwen3.5-4B
        0.838 -> 0.750 on the same 80 utterances). Slower and right beats faster
        and silently wrong.
        """
        shared = 0
        if self._prefix_reuse:
            for a, b in zip(self._prefix_tokens, tokens):
                if a != b:
                    break
                shared += 1
            # Always re-evaluate at least the final token so fresh logits exist.
            shared = min(shared, len(tokens) - 1)

        try:
            logits = self._prefill_from(tokens, shared)
        except RuntimeError as exc:
            if not self._prefix_reuse or shared == 0:
                raise
            log.info(
                "%s rejects prefix reuse (%s); prefilling in full from now on",
                self.model_path.name,
                str(exc).strip().splitlines()[-1] if str(exc).strip() else exc,
            )
            self._prefix_reuse = False
            logits = self._prefill_from(tokens, 0)
        return logits, self.PROMPT_SEQ, len(tokens)

    def _prefill_from(self, tokens: list[int], shared: int) -> np.ndarray:
        self._ctx.kv_cache_seq_rm(self.PROMPT_SEQ, shared if shared else -1, -1)
        logits = self._decode_span(tokens, shared, self.PROMPT_SEQ)
        self._prefix_tokens = list(tokens)
        self._prompt_len = len(tokens)
        return logits

    def _decode_span(self, tokens: list[int], start: int, seq_id: int) -> np.ndarray:
        """Decode ``tokens[start:]`` into ``seq_id``, asking for the last logits only."""
        self.telemetry["prefix_reused"] += start
        self.telemetry["prefill_tokens"] += len(tokens) - start

        positions = list(range(start, len(tokens)))
        last = positions[-1]
        chunk = self._batch._n_tokens
        logits: list[np.ndarray] = []
        for offset in range(0, len(positions), chunk):
            window = positions[offset : offset + chunk]
            produced = self._decode(
                [(tokens[position], position, seq_id, position == last) for position in window]
            )
            if produced:
                logits = produced
        return logits[-1]

    # ------------------------------------------------------------------ #
    # tree walk
    # ------------------------------------------------------------------ #
    def _walk(
        self, root: TokenTree, root_logits: np.ndarray, state: _Search, seq_id: int, prompt_len: int
    ) -> None:
        """Expand the tree one depth at a time, one batched decode per depth.

        Branches are pruned against the best complete candidate found so far.
        Log-probabilities are non-positive, so a node's score is an upper bound on
        every candidate beneath it: a branch already scoring below a finished
        candidate cannot contain the winner. The argmax stays exact while whole
        subtrees go unevaluated.
        """
        # (node, accumulated score, seq id holding its prefix, next position)
        frontier: list[tuple[TokenTree, float, int, int]] = [(root, 0.0, seq_id, prompt_len)]
        frontier_logits = [root_logits]

        while frontier:
            # 1. Score every child of every frontier node, and finish candidates.
            pending: list[tuple[float, int, TokenTree, int, int]] = []
            for (node, base, seq_id, position), logits in zip(frontier, frontier_logits):
                shifted = logits - logits.max()
                log_norm = math.log(float(np.exp(shifted).sum()))
                for token, child in node.children.items():
                    score = base + float(shifted[token]) - log_norm
                    if child.candidate is not None:
                        state.scores[child.candidate] = score
                        if score > state.best_score:
                            state.best_score = score
                            state.best = child.candidate
                    if child.children:
                        pending.append((score, token, child, seq_id, position))

            # 2. Prune against the best complete candidate, then expand the rest.
            survivors = []
            for score, token, child, seq_id, position in pending:
                if score <= state.best_score:
                    for index in child.reachable:
                        state.bounds[index] = max(state.bounds.get(index, -math.inf), score)
                    state.pruned += 1
                    continue
                survivors.append((score, token, child, seq_id, position))

            if len(survivors) > self.n_seq_max - 1:
                # Truncating here would silently drop branches and could change
                # the argmax, so it fails instead. Sequences are cheap: raise
                # runtime.n_seq_max.
                raise BenchmarkError(
                    f"{len(survivors)} branches need evaluating but only "
                    f"{self.n_seq_max - 1} KV sequences are available; "
                    "raise runtime.n_seq_max for this model"
                )

            parents = {seq_id for _, _, _, seq_id, _ in survivors}
            items: list[tuple[int, int, int]] = []
            next_frontier: list[tuple[TokenTree, float, int, int]] = []
            for score, token, child, seq_id, position in survivors:
                branch_seq = self._take_seq()
                # Broadcast: the branch gets its own view of the parent's cells.
                self._ctx.kv_cache_seq_cp(seq_id, branch_seq, 0, position)
                items.append((token, position, branch_seq, True))
                next_frontier.append((child, score, branch_seq, position + 1))
            self.telemetry["branches_evaluated"] += len(items)

            frontier_logits = self._decode(items)
            # Parent sequences are no longer referenced once the copies exist.
            for parent in parents:
                if parent not in (self.PROMPT_SEQ, seq_id):
                    self._release_seq(parent)
            frontier = next_frontier

    def _take_seq(self) -> int:
        if not self._free_seqs:
            raise BenchmarkError(
                f"ran out of KV sequences (n_seq_max={self.n_seq_max}); "
                "raise runtime.n_seq_max for this model"
            )
        return self._free_seqs.pop()

    def _release_seq(self, seq_id: int) -> None:
        self._ctx.kv_cache_seq_rm(seq_id, -1, -1)
        self._free_seqs.append(seq_id)

    def classify(self, *, system: str, user: str, candidates: Sequence[str]) -> LocalResult:
        if not candidates:
            raise BenchmarkError("a constrained classification needs at least one candidate")

        started = time.perf_counter()
        tree, _ = self._tree_for(candidates)
        prompt = self._render_prompt(system, user)
        tokens = list(self._model.tokenize(prompt.encode("utf-8"), add_bos=True, special=True))
        decodes_before = self.telemetry["decodes"]
        branches_before = self.telemetry["branches_evaluated"]

        logits, working_seq, prompt_len = self._prefill(tokens)
        state = _Search()
        self._walk(tree, logits, state, working_seq, prompt_len)
        self._reset_branches()

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
                "decodes": self.telemetry["decodes"] - decodes_before,
                "branches_evaluated": self.telemetry["branches_evaluated"] - branches_before,
                "pruned_branches": state.pruned,
                "decoding": "parallel-constrained",
            },
        )

    def describe(self) -> dict[str, Any]:
        import llama_cpp

        return {
            "engine": "llama_cpp_pcd",
            "algorithm_version": ENGINE_VERSION,
            "engine_version": llama_cpp.__version__,
            "model_file": self.model_path.name,
            "decoding": "parallel-constrained (prefill + logit slicing + broadcast token tree)",
            "chat_template": "gguf" if self._chat_template is not None else "chatml-fallback",
            "prefix_reuse": self._prefix_reuse,
            "n_seq_max": self.n_seq_max,
            "telemetry": dict(self.telemetry),
            "runtime": {
                key: self.runtime.get(key)
                for key in ("n_ctx", "n_threads", "n_gpu_layers", "n_batch", "seed", "temperature", "top_p")
            },
        }

    def _reset_branches(self) -> None:
        """Drop every branch sequence, keeping the prompt prefix for reuse."""
        for seq_id in range(1, self.n_seq_max):
            self._ctx.kv_cache_seq_rm(seq_id, -1, -1)
        self._free_seqs = list(range(1, self.n_seq_max))

    def close(self) -> None:
        for resource in (getattr(self, "_batch", None), getattr(self, "_ctx", None), getattr(self, "_model", None)):
            if resource is not None:
                resource.close()
        self._batch = self._ctx = self._model = None
        self._trees.clear()
