"""Metric behaviour, including the Korean-specific pitfalls."""

from __future__ import annotations

import pytest

from llmbench.config.schema import (
    ChrfMetricConfig,
    RougeMetricConfig,
    SurfaceFactsMetricConfig,
)
from llmbench.core.text import estimate_tokens, multilingual_tokens, split_sentences, truncate_to_tokens
from llmbench.metrics.classification import (
    classification_scores,
    cross_lingual_consistency,
    parse_batch_answer,
)
from llmbench.metrics.hallucination import hallucination_scores, parse_hallucination_label
from llmbench.metrics.summarization import ChrfEvaluator, RougeEvaluator, is_refusal
from llmbench.metrics.surface_facts import SurfaceFactEvaluator


# --------------------------------------------------------------------------- #
# tokenization / preprocessing
# --------------------------------------------------------------------------- #
def test_korean_rouge_is_not_zero():
    """The stock rouge-score tokenizer deletes Hangul; ours must not."""
    rouge = RougeEvaluator(RougeMetricConfig())
    scores = rouge.score("매출은 12% 증가했다. 회사는 성장했다.", "매출이 12% 늘었다.")
    assert scores["rougeLsum_f"] > 0.0
    assert scores["rouge1_f"] > 0.0


def test_rouge_perfect_match_is_one():
    rouge = RougeEvaluator(RougeMetricConfig())
    for text in ("The company said sales rose.", "회사는 매출이 늘었다고 밝혔다."):
        assert rouge.score(text, text)["rougeLsum_f"] == pytest.approx(1.0)


def test_rouge_empty_prediction_is_zero():
    rouge = RougeEvaluator(RougeMetricConfig())
    assert rouge.score("anything", "   ")["rougeLsum_f"] == 0.0


def test_chrf_handles_both_scripts():
    chrf = ChrfEvaluator(ChrfMetricConfig())
    assert chrf.score("매출은 12% 증가했다.", "매출은 12% 증가했다.")["chrf"] > 99
    assert chrf.score("Sales rose 12%.", "Sales rose 12%.")["chrf"] > 99


def test_multilingual_tokenizer_splits_hangul_by_character():
    assert multilingual_tokens("서울 2021") == ["서", "울", "2021"]
    assert multilingual_tokens("Hello World") == ["hello", "world"]


def test_sentence_splitter_handles_both_scripts():
    assert len(split_sentences("A first one. And a second one!")) == 2
    assert len(split_sentences("첫 문장이다. 두 번째 문장이다.")) == 2


def test_truncation_is_deterministic_and_bounded():
    text = "단어 " * 5000
    a, truncated_a = truncate_to_tokens(text, 100)
    b, truncated_b = truncate_to_tokens(text, 100)
    assert a == b
    assert truncated_a and truncated_b
    assert estimate_tokens(a) <= 100

    short = "짧은 문장."
    out, truncated = truncate_to_tokens(short, 100)
    assert out == short and not truncated


# --------------------------------------------------------------------------- #
# surface fact diagnostic
# --------------------------------------------------------------------------- #
@pytest.fixture
def surface():
    return SurfaceFactEvaluator(SurfaceFactsMetricConfig())


def test_korean_numeric_mismatch_is_detected(surface):
    """The spec's worked example: 12% in the source, 21% in the summary."""
    result = surface.score("매출은 12% 증가했다.", "매출은 21% 증가했다.")
    assert result["percentage_mismatch"] == 1
    assert result["surface_fact_support_precision"] == 0.0


def test_matching_numbers_are_supported(surface):
    result = surface.score("매출은 12% 증가했다.", "매출은 12% 증가했다고 한다.")
    assert result["percentage_mismatch"] == 0
    assert result["surface_fact_support_precision"] == 1.0


def test_currency_and_date_mismatch(surface):
    result = surface.score(
        "The ministry said on 12 March 2021 that revenue reached $1.5 billion.",
        "On 12 May 2022 revenue reached $2.5 billion.",
    )
    assert result["currency_mismatch"] >= 1
    assert result["date_mismatch"] >= 1


def test_unsupported_entity_is_flagged(surface):
    result = surface.score("Officials in Manchester spoke to the BBC.", "Officials in Manchester spoke to CNN.")
    assert result["unsupported_named_entity"] == 1


def test_no_facts_yields_null_precision(surface):
    result = surface.score("아무 숫자도 없는 원문이다.", "숫자가 없다.")
    assert result["facts_total"] == 0
    assert result["surface_fact_support_precision"] is None


# --------------------------------------------------------------------------- #
# hallucination labels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("SUPPORTED", "SUPPORTED"),
        ("HALLUCINATED", "HALLUCINATED"),
        ("  supported.\n", "SUPPORTED"),
        ("Label: HALLUCINATED", "HALLUCINATED"),
        ("**SUPPORTED**", "SUPPORTED"),
        ("YES", "SUPPORTED"),
        ("정답: 원문에 의해 뒷받침된다", "SUPPORTED"),
        ("", None),
        ("maybe", None),
    ],
)
def test_hallucination_label_parsing(text, expected):
    assert parse_hallucination_label(text) == expected


def test_ambiguous_output_counts_as_invalid():
    assert parse_hallucination_label("SUPPORTED or HALLUCINATED, hard to say") is None


def test_hallucination_scores_count_invalid_as_wrong():
    rows = [
        {"status": "SUCCESS", "gold_label": "SUPPORTED", "output_text": "SUPPORTED"},
        {"status": "SUCCESS", "gold_label": "HALLUCINATED", "output_text": "HALLUCINATED"},
        {"status": "SUCCESS", "gold_label": "HALLUCINATED", "output_text": "who knows"},
        {"status": "SUCCESS", "gold_label": "SUPPORTED", "output_text": "HALLUCINATED"},
    ]
    scores = hallucination_scores(rows)
    assert scores["cases"] == 4
    assert scores["accuracy"] == pytest.approx(0.5)
    assert scores["invalid_outputs"] == 1
    assert scores["invalid_output_rate"] == pytest.approx(0.25)
    assert scores["hallucination_recall"] == pytest.approx(0.5)
    assert 0 < scores["macro_f1"] < 1


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def test_batch_answer_parsing():
    parsed = parse_batch_answer("1: 3\n2: 7\n3: 1\n", batch_size=3, label_count=10)
    assert parsed == {1: 2, 2: 6, 3: 0}


def test_batch_answer_parsing_tolerates_noise_and_gaps():
    parsed = parse_batch_answer("- 1. 4\n2) 2\n", batch_size=3, label_count=10)
    assert parsed[1] == 3
    assert parsed[2] == 1
    assert parsed[3] is None


def test_batch_answer_rejects_out_of_range_labels():
    parsed = parse_batch_answer("1: 99\n2: 2", batch_size=2, label_count=4)
    assert parsed[1] is None
    assert parsed[2] == 1


def test_classification_scores_and_consistency():
    labels = ["a", "b", "c"]
    en = [
        {"sample_id": "e1", "semantic_id": "1", "status": "SUCCESS", "gold": "a", "predicted": "a"},
        {"sample_id": "e2", "semantic_id": "2", "status": "SUCCESS", "gold": "b", "predicted": "c"},
        {"sample_id": "e3", "semantic_id": "3", "status": "SUCCESS", "gold": "c", "predicted": None},
    ]
    ko = [
        {"sample_id": "k1", "semantic_id": "1", "status": "SUCCESS", "gold": "a", "predicted": "a"},
        {"sample_id": "k2", "semantic_id": "2", "status": "SUCCESS", "gold": "b", "predicted": "b"},
        {"sample_id": "k3", "semantic_id": "3", "status": "SUCCESS", "gold": "c", "predicted": "a"},
    ]

    en_scores = classification_scores(en, labels)
    assert en_scores["accuracy"] == pytest.approx(1 / 3)
    assert en_scores["invalid_outputs"] == 1

    cross = cross_lingual_consistency(en, ko)
    assert cross["paired_cases"] == 3
    assert cross["both_correct"] == 1
    assert cross["ko_only_correct"] == 1
    assert cross["both_wrong"] == 1
    assert cross["en_only_correct"] == 0
    assert cross["consistency"] == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# refusal detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("I'm sorry, I cannot summarize this article.", True),
        ("죄송하지만 요약할 수 없습니다.", True),
        ("Rail passenger numbers rose 12% last year.", False),
        ("", False),
    ],
)
def test_refusal_detection(text, expected):
    assert is_refusal(text) is expected


def test_long_output_mentioning_sorry_is_not_a_refusal():
    text = "I'm sorry to report that " + "the ministry confirmed the figures again. " * 20
    assert is_refusal(text) is False


# --------------------------------------------------------------------------- #
# cross-check against a reference implementation
# --------------------------------------------------------------------------- #
def test_macro_f1_matches_sklearn():
    """The hand-written macro-F1 must agree with scikit-learn on present labels."""
    sklearn_metrics = pytest.importorskip("sklearn.metrics")

    labels = ["a", "b", "c", "d"]
    golds = ["a", "b", "c", "a", "b", "c", "a", "b"]
    preds = ["a", "b", "b", "c", "b", "c", "a", "a"]
    rows = [
        {"sample_id": str(i), "semantic_id": str(i), "status": "SUCCESS", "gold": g, "predicted": p}
        for i, (g, p) in enumerate(zip(golds, preds))
    ]

    ours = classification_scores(rows, labels)["macro_f1"]
    theirs = sklearn_metrics.f1_score(golds, preds, labels=["a", "b", "c"], average="macro", zero_division=0)
    assert ours == pytest.approx(theirs)


def test_hallucination_macro_f1_matches_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")

    rows = [
        {"status": "SUCCESS", "gold_label": "SUPPORTED", "output_text": "SUPPORTED"},
        {"status": "SUCCESS", "gold_label": "SUPPORTED", "output_text": "HALLUCINATED"},
        {"status": "SUCCESS", "gold_label": "HALLUCINATED", "output_text": "HALLUCINATED"},
        {"status": "SUCCESS", "gold_label": "HALLUCINATED", "output_text": "HALLUCINATED"},
    ]
    scores = hallucination_scores(rows)
    golds = [r["gold_label"] for r in rows]
    preds = [parse_hallucination_label(r["output_text"]) for r in rows]
    expected = sklearn_metrics.f1_score(
        golds, preds, labels=["SUPPORTED", "HALLUCINATED"], average="macro", zero_division=0
    )
    assert scores["macro_f1"] == pytest.approx(expected)
