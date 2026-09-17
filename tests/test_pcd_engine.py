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


class StubContext:
    def __init__(self, owner: "StubLlama") -> None:
        self.owner = owner

    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int) -> None:
        self.owner.rewinds.append(p0)

    def get_logits(self):
        return self.owner.logits_for(tuple(self.owner.path))


class StubLlama:
    """Returns scripted logits keyed by the token path evaluated so far."""

    def __init__(self, table: dict[tuple[int, ...], Sequence[float]]) -> None:
        self.table = table
        self.path: list[int] = []
        self.n_tokens = 0
        self._n_vocab = VOCAB
        self._ctx = StubContext(self)
        self.evals = 0
        self.rewinds: list[int] = []

    def logits_for(self, path: tuple[int, ...]):
        values = self.table.get(path)
        if values is None:
            values = [0.0] * VOCAB
        return np.array(values, dtype=np.float32)

    def tokenize(self, text: bytes, add_bos: bool = False, special: bool = False):
        return [1, 2, 3]

    def eval(self, tokens: Sequence[int]) -> None:
        self.evals += 1
        self.path.extend(tokens)
        self.n_tokens += len(tokens)

    def reset(self) -> None:
        self.path.clear()
        self.n_tokens = 0


def make_engine(table, tokenized) -> ParallelConstrainedEngine:
    engine = ParallelConstrainedEngine.__new__(ParallelConstrainedEngine)
    engine.model_path = __import__("pathlib").Path("stub.gguf")
    engine.runtime = {}
    engine._trees = {}
    engine._prefix_tokens = []
    engine.telemetry = {"prefill_tokens": 0, "forward_passes": 0, "prefix_reused": 0}
    engine._llm = StubLlama(table)
    engine._tokenized = tokenized
    return engine


def score_candidates(engine, tree, root_logits) -> dict:
    from llmbench.local.pcd_engine import _Search

    state = _Search()
    engine._walk(tree, np.array(root_logits, dtype=np.float32), 0.0, state)
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

    engine = make_engine({(1,): after_one}, [[1, 2], [1, 3], [4]])
    tree = build_token_tree([[1, 2], [1, 3], [4]])
    state = score_candidates(engine, tree, root)

    assert state.best == 2
    assert state.scores[2] > state.scores.get(0, -math.inf)


def test_pruning_preserves_the_argmax():
    """Pruning uses an admissible bound, so it cannot change the winner."""
    root = [0.0] * VOCAB
    root[1] = 5.0   # strong branch, needs a continuation step
    root[4] = -8.0  # hopeless branch, must be pruned
    after_one = [0.0] * VOCAB
    after_one[2] = 3.0
    after_one[3] = 0.0

    tokenized = [[1, 2], [1, 3], [4]]
    engine = make_engine({(1,): after_one}, tokenized)
    state = score_candidates(engine, build_token_tree(tokenized), root)

    assert state.best == 0
    assert state.pruned >= 1
    # The pruned candidate still gets an upper bound for calibration.
    assert 2 in state.bounds or 2 in state.scores


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
    engine = make_engine({(1,): after_one, (4,): after_four}, tokenized)
    state = score_candidates(engine, build_token_tree(tokenized), root)

    assert state.best in (0, 1)
    assert engine._llm.evals == 1, "only the surviving branch was expanded"


def test_walk_rewinds_the_cache_for_each_sibling():
    root = [0.0] * VOCAB
    root[1] = 1.0
    root[4] = 0.9
    cont = [0.0] * VOCAB
    cont[2] = 0.0
    cont[3] = 0.0
    cont[5] = 0.0
    cont[6] = 0.0

    tokenized = [[1, 2], [1, 3], [4, 5], [4, 6]]
    engine = make_engine({(1,): cont, (4,): cont}, tokenized)
    score_candidates(engine, build_token_tree(tokenized), root)

    assert engine._llm.evals == 2
    # Both branches were rewound back to the shared prefix position.
    assert engine._llm.rewinds == [0, 0]


def test_every_candidate_gets_a_score_or_a_bound():
    root = [0.0] * VOCAB
    root[1] = 1.0
    root[4] = 0.5
    root[7] = -30.0
    cont = [0.0] * VOCAB

    tokenized = [[1, 2], [1, 3], [4], [7, 8]]
    engine = make_engine({(1,): cont, (4,): cont, (7,): cont}, tokenized)
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
