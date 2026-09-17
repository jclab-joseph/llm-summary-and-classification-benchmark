"""Parallel constrained decoding: token tree, pruning and calibrated scores.

The scoring is driven through a stub llama.cpp context, so these run without
weights and without the optional extra.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pytest

from llmbench.local.pcd_engine import ParallelConstrainedEngine, TokenTree, build_token_tree

VOCAB = 16


class StubBatchStruct:
    """Mimics the fields of llama.cpp's `llama_batch` that the engine fills."""

    def __init__(self, capacity: int = 64) -> None:
        self.n_tokens = 0
        self.token = [0] * capacity
        self.pos = [0] * capacity
        self.n_seq_id = [0] * capacity
        self.seq_id = [[0] for _ in range(capacity)]
        self.logits = [False] * capacity


class StubBatch:
    def __init__(self, capacity: int = 64) -> None:
        self.batch = StubBatchStruct(capacity)
        self._n_tokens = capacity

    def close(self) -> None:
        pass


class StubContext:
    """Scripted logits keyed by the token path a sequence has accumulated."""

    def __init__(self, table: dict[tuple[int, ...], Sequence[float]]) -> None:
        self.table = table
        self.seq_paths: dict[int, tuple[int, ...]] = {0: ()}
        self.decodes = 0
        self.batched_sizes: list[int] = []
        self.copies: list[tuple[int, int]] = []
        self.removed: list[int] = []
        self._logits: list[np.ndarray] = []

    def logits_for(self, path: tuple[int, ...]) -> np.ndarray:
        return np.array(self.table.get(path, [0.0] * VOCAB), dtype=np.float32)

    def decode(self, batch: StubBatch) -> None:
        raw = batch.batch
        self.decodes += 1
        self.batched_sizes.append(raw.n_tokens)
        self._logits = []
        for index in range(raw.n_tokens):
            seq = raw.seq_id[index][0]
            path = self.seq_paths.get(seq, ()) + (raw.token[index],)
            self.seq_paths[seq] = path
            self._logits.append(self.logits_for(path))

    def get_logits_ith(self, index: int) -> np.ndarray:
        return self._logits[index]

    def kv_cache_seq_cp(self, src: int, dst: int, p0: int, p1: int) -> None:
        self.copies.append((src, dst))
        self.seq_paths[dst] = self.seq_paths.get(src, ())

    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int) -> None:
        self.removed.append(seq_id)
        if p0 <= 0:
            self.seq_paths[seq_id] = ()

    def close(self) -> None:
        pass


def make_engine(table, n_seq_max: int = 8) -> ParallelConstrainedEngine:
    engine = ParallelConstrainedEngine.__new__(ParallelConstrainedEngine)
    engine.model_path = __import__("pathlib").Path("stub.gguf")
    engine.runtime = {}
    engine._trees = {}
    engine._prefix_tokens = []
    engine.telemetry = {
        "prefill_tokens": 0,
        "decodes": 0,
        "branches_evaluated": 0,
        "prefix_reused": 0,
    }
    engine.n_seq_max = n_seq_max
    engine._free_seqs = list(range(1, n_seq_max))
    engine._ctx = StubContext(table)
    engine._batch = StubBatch()
    engine._n_vocab = VOCAB
    engine._prompt_len = 0
    return engine


def score_candidates(engine, tree, root_logits):
    from llmbench.local.pcd_engine import _Search

    state = _Search()
    engine._walk(tree, np.array(root_logits, dtype=np.float32), state)
    return state


# --------------------------------------------------------------------------- #
# token tree
# --------------------------------------------------------------------------- #
def test_tree_stops_as_soon_as_a_candidate_is_unambiguous():
    # "a b", "a c", "d" -> only the "a" branch needs a continuation step.
    tree = build_token_tree([[1, 2], [1, 3], [4]])

    assert set(tree.children) == {1, 4}
    assert tree.children[4].candidate == 2
    assert not tree.children[4].children, "a unique candidate needs no further tokens"
    assert set(tree.children[1].children) == {2, 3}
    assert tree.count_forward_passes() == 1


def test_tree_handles_a_candidate_that_prefixes_another():
    tree = build_token_tree([[1], [1, 2]])
    node = tree.children[1]
    assert node.candidate == 0
    assert 2 in node.children


def test_tree_rejects_an_empty_candidate():
    from llmbench.core.errors import BenchmarkError

    with pytest.raises(BenchmarkError, match="tokenized to nothing"):
        build_token_tree([[1], []])


def test_shared_first_token_is_the_normal_case_not_an_edge_case():
    """MASSIVE intents collide heavily: alarm_set / alarm_query / alarm_remove."""
    tokenized = [[10, 1], [10, 2], [10, 3], [20, 1]]
    tree = build_token_tree(tokenized)
    assert len(tree.children) == 2, "four candidates collapse to two first tokens"
    assert len(tree.children[10].children) == 3


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def test_scores_full_sequences_not_just_the_first_token():
    """The whole point: a first token that wins can still lose overall.

    Candidate 0 and 1 share first token 1; candidate 2 uses token 4. Token 1 has
    the highest logit at depth 0, but both of its continuations are unlikely, so
    the winner is candidate 2 -- which a first-token comparison could never find.
    """
    root = [0.0] * VOCAB
    root[1] = 2.0
    root[4] = 1.0
    after_one = [0.0] * VOCAB
    after_one[2] = -6.0
    after_one[3] = -6.0

    engine = make_engine({(1,): after_one})
    tree = build_token_tree([[1, 2], [1, 3], [4]])
    state = score_candidates(engine, tree, root)

    assert state.best == 2
    assert state.scores[2] > state.scores.get(0, -math.inf)


def test_pruning_preserves_the_argmax():
    """Pruning uses an admissible bound, so it cannot change the winner."""
    root = [0.0] * VOCAB
    root[1] = 5.0   # strong branch, needs a continuation step
    root[7] = 0.0   # a finished candidate, sets the bar
    root[4] = -8.0  # hopeless branch that would need expanding: must be pruned
    after_one = [0.0] * VOCAB
    after_one[2] = 3.0
    after_one[3] = 0.0

    tokenized = [[1, 2], [1, 3], [4, 5], [4, 6], [7]]
    engine = make_engine({(1,): after_one})
    state = score_candidates(engine, build_token_tree(tokenized), root)

    assert state.best == 0
    assert state.pruned >= 1
    # The pruned candidates still get an upper bound, for calibration.
    assert {2, 3} <= set(state.bounds) | set(state.scores)


def test_pruned_branch_is_never_evaluated():
    root = [0.0] * VOCAB
    root[1] = 10.0
    root[4] = -20.0
    after_one = [0.0] * VOCAB
    after_one[2] = 0.0
    after_one[3] = 0.0
    after_four = [0.0] * VOCAB
    after_four[5] = 50.0  # would win if it were ever reached

    tokenized = [[1, 2], [1, 3], [4, 5]]
    engine = make_engine({(1,): after_one, (4,): after_four})
    state = score_candidates(engine, build_token_tree(tokenized), root)

    assert state.best in (0, 1)
    assert engine._ctx.batched_sizes == [1], "only the surviving branch was expanded"


def test_sibling_branches_are_evaluated_in_one_batched_decode():
    root = [0.0] * VOCAB
    root[1] = 1.0
    root[4] = 0.9
    cont = [0.0] * VOCAB
    cont[2] = 0.0
    cont[3] = 0.0
    cont[5] = 0.0
    cont[6] = 0.0

    tokenized = [[1, 2], [1, 3], [4, 5], [4, 6]]
    engine = make_engine({(1,): cont, (4,): cont})
    score_candidates(engine, build_token_tree(tokenized), root)

    # This is the broadcast: two branches, one decode, not one decode each.
    assert engine._ctx.decodes == 1
    assert engine._ctx.batched_sizes == [2]
    # Each branch got its own sequence copied from the shared prompt prefix.
    assert sorted(engine._ctx.copies) == [(0, 6), (0, 7)]


def test_every_candidate_gets_a_score_or_a_bound():
    root = [0.0] * VOCAB
    root[1] = 1.0
    root[4] = 0.5
    root[7] = -30.0
    cont = [0.0] * VOCAB

    tokenized = [[1, 2], [1, 3], [4], [7, 8]]
    engine = make_engine({(1,): cont, (4,): cont, (7,): cont})
    state = score_candidates(engine, build_token_tree(tokenized), root)

    covered = set(state.scores) | set(state.bounds)
    assert covered == {0, 1, 2, 3}


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #
def test_both_decoders_are_registered():
    from llmbench.local.engine import ENGINES

    assert ENGINES["llama_cpp"].endswith("LlamaCppEngine")
    assert ENGINES["llama_cpp_pcd"].endswith("ParallelConstrainedEngine")


def test_config_ships_both_decoders(prepared):
    engines = {m.engine for m in prepared.local_models.models}
    assert engines == {"llama_cpp", "llama_cpp_pcd"}


def test_the_two_decoders_do_not_share_cache_keys(prepared):
    """Same weights, different decoding: different runs."""
    from llmbench.benchmarks.classification import build_classification_tasks
    from llmbench.datasets.manifest import read_manifest

    manifest = read_manifest(prepared.manifest_dir / prepared.benchmark.benchmarks.classification.manifest)
    by_engine = {}
    for model in prepared.local_models.models:
        by_engine[model.engine] = {
            t.cache_key for t in build_classification_tasks(prepared, model, manifest).tasks
        }
    assert by_engine["llama_cpp"].isdisjoint(by_engine["llama_cpp_pcd"])
