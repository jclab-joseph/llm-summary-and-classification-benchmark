"""Laya: a non-generative decision model behind the local engine contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llmbench.benchmarks.classification import LOCAL_MODE, build_classification_tasks
from llmbench.core.errors import BenchmarkError
from llmbench.datasets.manifest import read_manifest
from llmbench.local.download import local_model_path, needs_single_file
from llmbench.local.laya_engine import MAX_DOCUMENTED_OPTIONS, LayaEngine
from llmbench.prompts.registry import classification_state_prompt

SLUG = "local/laya-multilingual"
LABELS = ["alarm_set", "audio_volume_mute", "weather_query"]


class StubAgent:
    def __init__(self, choice: str = "weather_query") -> None:
        self.choice = choice
        self.calls: list[tuple[Any, Any]] = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return {
            "answers": {"intent": {"choice": self.choice, "confidence": 0.7, "action": "act"}},
            "usage": {"tokens": 123},
        }


def make_engine(agent: StubAgent | None = None) -> LayaEngine:
    engine = LayaEngine.__new__(LayaEngine)
    engine.repo = "convaiinnovations/laya-multilingual"
    engine.runtime = {}
    engine.question_id = "intent"
    engine._warned_options = False
    engine._agent = agent or StubAgent()
    engine._laya_version = "0.3.4"
    return engine


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def test_checkpoint_is_loaded_by_the_engine_not_downloaded_as_a_file(prepared):
    """A transformers checkpoint is a repository, not one GGUF file."""
    model = prepared.require_model(SLUG)
    assert model.source.filename is None
    assert needs_single_file(model) is False


def test_single_file_engines_are_unaffected(prepared):
    gguf = prepared.require_model("local/qwen3.5-2b-q8-gguf")
    assert needs_single_file(gguf) is True
    assert gguf.source.filename in local_model_path(prepared, gguf).name


def test_prompt_style_is_part_of_the_cache_key(prepared):
    """A state question and a chat prompt are different requests."""
    model = prepared.require_model(SLUG)
    manifest = read_manifest(prepared.manifest_dir / prepared.benchmark.benchmarks.classification.manifest)

    before = [t.cache_key for t in build_classification_tasks(prepared, model, manifest).tasks]
    model.prompt_style = "chat"
    assert [t.cache_key for t in build_classification_tasks(prepared, model, manifest).tasks] != before


def test_state_prompt_carries_the_utterance_without_the_label_list(prepared):
    """The options travel as the question's criteria, not inside the prompt."""
    model = prepared.require_model(SLUG)
    manifest = read_manifest(prepared.manifest_dir / prepared.benchmark.benchmarks.classification.manifest)
    task = build_classification_tasks(prepared, model, manifest).tasks[0]

    assert task.meta["output_mode"] == LOCAL_MODE
    assert task.candidates
    assert "alarm_set" not in task.prompt.user
    assert task.prompt.user  # the bare utterance
    for label in task.candidates:
        assert label not in task.prompt.system


def test_state_prompt_is_localized():
    assert "인텐트" in classification_state_prompt("ko", "조용히 해").system
    assert "intent" in classification_state_prompt("en", "mute").system.lower()


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
def test_options_are_passed_as_the_questions_criteria():
    agent = StubAgent()
    engine = make_engine(agent)

    result = engine.classify(system="Which intent?", user="what's the weather", candidates=LABELS)

    state, questions = agent.calls[0]
    assert state == {"utterance": "what's the weather"}
    question = questions["intent"]
    assert question["type"] == "choice"
    assert question["instructions"] == "Which intent?"
    assert set(question["criteria"]) == set(LABELS)
    assert result.label == "weather_query"
    # Nothing is generated, so there are no completion tokens to report.
    assert result.completion_tokens == 0
    assert result.raw["probability"] == 0.7


def test_an_answer_outside_the_label_set_is_an_error():
    engine = make_engine(StubAgent(choice="not_an_intent"))
    with pytest.raises(BenchmarkError, match="outside the label set"):
        engine.classify(system="q", user="u", candidates=LABELS)


def test_empty_candidate_set_is_rejected():
    engine = make_engine()
    with pytest.raises(BenchmarkError, match="at least one candidate"):
        engine.classify(system="q", user="u", candidates=[])


def test_predict_failure_is_surfaced():
    class Boom:
        def predict(self, state, questions):
            raise RuntimeError("model exploded")

    engine = make_engine()
    engine._agent = Boom()
    with pytest.raises(BenchmarkError, match="model exploded"):
        engine.classify(system="q", user="u", candidates=LABELS)


def test_exceeding_the_documented_option_ceiling_warns_once(caplog):
    """60 intents is three times the documented ceiling; the score has to say so."""
    engine = make_engine()
    many = [f"intent_{i}" for i in range(MAX_DOCUMENTED_OPTIONS + 1)]
    engine._agent = StubAgent(choice=many[0])

    with caplog.at_level("WARNING"):
        engine.classify(system="q", user="u", candidates=many)
        engine.classify(system="q", user="u", candidates=many)

    warnings = [r for r in caplog.records if "documents a ceiling" in r.message]
    assert len(warnings) == 1, "warn once, not once per utterance"


def test_no_warning_within_the_documented_ceiling(caplog):
    engine = make_engine()
    with caplog.at_level("WARNING"):
        engine.classify(system="q", user="u", candidates=LABELS)
    assert not [r for r in caplog.records if "documents a ceiling" in r.message]


def test_describe_records_the_ceiling():
    described = make_engine().describe()
    assert described["engine"] == "laya"
    assert described["documented_option_ceiling"] == MAX_DOCUMENTED_OPTIONS
    assert "non-autoregressive" in described["decoding"]
