"""Classification under OpenRouter structured output (`classification_mode: json_schema`)."""

from __future__ import annotations

import json

import pytest

from llmbench.benchmarks.classification import (
    build_classification_tasks,
    score_classification,
    uses_structured_output,
)
from llmbench.datasets.manifest import read_manifest
from llmbench.metrics.classification import parse_batch_json_answer
from llmbench.prompts.registry import classification_json_schema, classification_prompt
from llmbench.reporting.build import generate_reports
from llmbench.reporting.leaderboard import baseline_summaries, classification_mode_of
from llmbench.reporting.summary import summary_filename, write_model_summary
from llmbench.reporting.tables import classification_mode_rows
from llmbench.runner import BenchmarkRunner

MODEL = "google/gemini-2.5-flash-lite"
LABELS = ["alarm_set", "audio_volume_mute", "iot_hue_lightchange", "weather_query"]


def manifest_of(cfg):
    return read_manifest(cfg.manifest_dir / cfg.benchmark.benchmarks.classification.manifest)


# --------------------------------------------------------------------------- #
# schema and prompt
# --------------------------------------------------------------------------- #
def test_schema_constrains_answers_to_the_label_space():
    schema = classification_json_schema(LABELS, 20)["json_schema"]

    assert schema["strict"] is True
    answers = schema["schema"]["properties"]["answers"]
    assert answers["items"]["enum"] == LABELS
    assert schema["schema"]["additionalProperties"] is False
    # The array length is NOT pinned in the schema: a 60-value enum with a fixed
    # array length exceeds Gemini's constrained-decoding state limit. The parser
    # enforces the length instead.
    assert "minItems" not in answers and "maxItems" not in answers


def test_structured_prompt_asks_for_names_not_numbers():
    text = classification_prompt("en", ["wake me up"], LABELS)
    structured = classification_prompt("en", ["wake me up"], LABELS, structured=True)

    assert structured.template_version != text.template_version
    assert "1: alarm_set" in text.user          # numbered label list
    assert "- alarm_set" in structured.user     # plain names
    assert '"answers"' in structured.user
    for prompt in (text, structured):
        assert "wake me up" in prompt.user


def test_korean_structured_prompt_is_korean():
    prompt = classification_prompt("ko", ["깨워줘"], LABELS, structured=True)
    assert "인텐트" in prompt.user and "출력하라" in prompt.user


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def test_parses_a_well_formed_reply():
    reply = json.dumps({"answers": ["weather_query", "alarm_set"]})
    assert parse_batch_json_answer(reply, 2, LABELS) == {1: 3, 2: 0}


def test_tolerates_code_fences():
    reply = "```json\n" + json.dumps({"answers": ["alarm_set", "alarm_set"]}) + "\n```"
    assert parse_batch_json_answer(reply, 2, LABELS) == {1: 0, 2: 0}


def test_length_mismatch_invalidates_the_whole_batch():
    """Position is the only link to an utterance; a gap makes every later answer a guess."""
    short = json.dumps({"answers": ["alarm_set"]})
    long = json.dumps({"answers": ["alarm_set"] * 3})
    assert parse_batch_json_answer(short, 2, LABELS) == {1: None, 2: None}
    assert parse_batch_json_answer(long, 2, LABELS) == {1: None, 2: None}


def test_unknown_label_is_invalid_not_guessed():
    reply = json.dumps({"answers": ["alarm_set", "not_a_real_intent"]})
    assert parse_batch_json_answer(reply, 2, LABELS) == {1: 0, 2: None}


@pytest.mark.parametrize("reply", ["", "nope", "{}", '{"answers": "alarm_set"}', "[1, 2]"])
def test_unusable_replies_are_invalid(reply):
    assert parse_batch_json_answer(reply, 2, LABELS) == {1: None, 2: None}


# --------------------------------------------------------------------------- #
# task construction
# --------------------------------------------------------------------------- #
def test_mode_is_off_by_default(prepared):
    assert uses_structured_output(prepared) is False
    tasks = build_classification_tasks(prepared, prepared.require_model(MODEL), manifest_of(prepared))
    assert all(task.response_format is None for task in tasks.tasks)


def test_structured_mode_sends_response_format(prepared):
    prepared.benchmark.structured_output.classification_mode = "json_schema"
    tasks = build_classification_tasks(
        prepared, prepared.require_model(MODEL), manifest_of(prepared)
    ).tasks

    assert tasks
    for task in tasks:
        schema = task.response_format["json_schema"]["schema"]["properties"]["answers"]
        assert schema["items"]["enum"]
        assert task.meta["structured_output"] is True
        assert task.key_material["extra"]["response_format_hash"]


def test_the_two_modes_never_share_a_cache_key(prepared):
    """Both conditions coexist in one cache instead of overwriting each other."""
    model = prepared.require_model(MODEL)
    manifest = manifest_of(prepared)

    text_keys = {t.cache_key for t in build_classification_tasks(prepared, model, manifest).tasks}
    prepared.benchmark.structured_output.classification_mode = "json_schema"
    json_tasks = build_classification_tasks(prepared, model, manifest).tasks
    json_keys = {t.cache_key for t in json_tasks}

    assert text_keys and json_keys
    assert text_keys.isdisjoint(json_keys)
    assert json_tasks[0].key_material["structured_output_schema_version"] == "so-1:json_schema"


def test_other_benchmarks_are_unaffected_by_the_mode(prepared):
    """Flipping the classification mode must not re-bill summarization."""
    from llmbench.benchmarks.summarization import build_summarization_tasks

    model = prepared.require_model(MODEL)
    summ = read_manifest(prepared.manifest_dir / prepared.benchmark.benchmarks.summarization.manifest)
    before = [t.cache_key for t in build_summarization_tasks(prepared, model, summ).tasks]

    prepared.benchmark.structured_output.classification_mode = "json_schema"
    after = [t.cache_key for t in build_summarization_tasks(prepared, model, summ).tasks]
    assert before == after


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
async def test_structured_run_scores_and_is_cached(prepared, store, client_factory, mock_openrouter):
    prepared.benchmark.structured_output.classification_mode = "json_schema"
    runner = BenchmarkRunner(prepared, store, client_factory=lambda: client_factory())

    outcome = await runner.run(prepared.require_model(MODEL), benchmarks=["classification"])

    assert outcome.status == "COMPLETED"
    assert outcome.metrics["classification"]["output_mode"] == "json_schema"
    assert outcome.metrics["classification"]["per_language"]["en"]["invalid_output_rate"] == 0.0
    assert all(r["response_format"]["json_schema"]["strict"] for r in mock_openrouter.requests)

    mock_openrouter.reset_counters()
    await runner.run(prepared.require_model(MODEL), benchmarks=["classification"])
    assert mock_openrouter.chat_calls == 0


async def test_preflight_blocks_a_model_without_structured_output(
    prepared, store, client_factory, mock_openrouter
):
    prepared.benchmark.structured_output.classification_mode = "json_schema"
    model = prepared.require_model(MODEL)
    model.parameters.response_format = True
    # Pretend OpenRouter stops advertising structured outputs for this model.
    from mock_openrouter import MOCK_SUPPORTED_PARAMETERS

    original = MOCK_SUPPORTED_PARAMETERS[MODEL]
    MOCK_SUPPORTED_PARAMETERS[MODEL] = [p for p in original if "structured" not in p and p != "response_format"]
    try:
        runner = BenchmarkRunner(prepared, store, client_factory=lambda: client_factory())
        outcome = await runner.run(model, benchmarks=["classification"])
    finally:
        MOCK_SUPPORTED_PARAMETERS[MODEL] = original

    assert outcome.status == "BLOCKED"
    assert mock_openrouter.chat_calls == 0
    # A strict schema needs both capabilities, so both are reported.
    assert {p["parameter"] for p in outcome.preflight["blocking"]} == {
        "response_format",
        "structured_outputs",
    }


async def test_response_format_is_not_checked_in_text_mode(prepared, store, client_factory):
    """Text mode never sends response_format, so missing support is not blocking."""
    from mock_openrouter import MOCK_SUPPORTED_PARAMETERS

    model = prepared.require_model(MODEL)
    original = MOCK_SUPPORTED_PARAMETERS[MODEL]
    MOCK_SUPPORTED_PARAMETERS[MODEL] = [p for p in original if "structured" not in p and p != "response_format"]
    try:
        runner = BenchmarkRunner(prepared, store, client_factory=lambda: client_factory())
        outcome = await runner.run(model, benchmarks=["classification"])
    finally:
        MOCK_SUPPORTED_PARAMETERS[MODEL] = original

    assert outcome.status == "COMPLETED"


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
async def test_both_conditions_survive_in_the_reports(prepared, store, client_factory):
    model = prepared.require_model(MODEL)
    runner = BenchmarkRunner(prepared, store, client_factory=lambda: client_factory())

    text_outcome = await runner.run(model, benchmarks=["classification"])
    text_path = write_model_summary(prepared, text_outcome)

    prepared.benchmark.structured_output.classification_mode = "json_schema"
    json_outcome = await runner.run(model, benchmarks=["classification"])
    json_path = write_model_summary(prepared, json_outcome)

    # The structured run must not overwrite the baseline numbers.
    assert text_path.name == "summary.json"
    assert json_path.name == "summary.classification-json_schema.json"
    assert text_path.exists() and json_path.exists()

    bundle = generate_reports(prepared)
    assert len(bundle.all_summaries) == 2
    assert len(bundle.summaries) == 1, "the leaderboard shows one row per model"
    assert classification_mode_of(bundle.summaries[0]) == "text"

    document = bundle.paths["result_md"].read_text(encoding="utf-8")
    assert "## 분류: 자유 텍스트 vs 구조화 출력" in document
    assert "Δ Acc" in document


def test_mode_comparison_computes_deltas():
    def summary(mode: str, accuracy: float, invalid: float) -> dict:
        return {
            "variant": {"classification_mode": mode},
            "model": {"model_id": "m/x", "display_name": "X"},
            "metrics": {
                "classification": {
                    "per_language": {
                        "en": {"accuracy": accuracy, "macro_f1": accuracy, "invalid_output_rate": invalid}
                    }
                }
            },
        }

    rows = classification_mode_rows([summary("text", 0.51, 0.34), summary("json_schema", 0.78, 0.0)])
    assert len(rows) == 1
    row = rows[0]
    assert row["accuracy_delta"] == pytest.approx(0.27)
    assert row["invalid_rate_delta"] == pytest.approx(-0.34)


def test_no_comparison_row_without_both_conditions():
    one = {
        "variant": {"classification_mode": "text"},
        "model": {"model_id": "m/x"},
        "metrics": {"classification": {"per_language": {"en": {"accuracy": 0.5}}}},
    }
    assert classification_mode_rows([one]) == []


def test_baseline_selection_prefers_text():
    text = {"variant": {"classification_mode": "text"}, "model": {"model_id": "m/x"}}
    structured = {"variant": {"classification_mode": "json_schema"}, "model": {"model_id": "m/x"}}

    assert baseline_summaries([structured, text]) == [text]
    assert baseline_summaries([text, structured]) == [text]
    # With only the structured run available, that is the baseline.
    assert baseline_summaries([structured]) == [structured]


def test_summary_filename_is_stable_for_the_baseline():
    assert summary_filename({}) == "summary.json"
    assert summary_filename({"classification_mode": "text"}) == "summary.json"
    assert summary_filename({"classification_mode": "json_schema"}) == "summary.classification-json_schema.json"


def test_response_schema_is_part_of_the_cache_key(prepared, monkeypatch):
    """A changed response schema is a changed request, so it must miss the cache."""
    from llmbench.prompts import registry

    prepared.benchmark.structured_output.classification_mode = "json_schema"
    model = prepared.require_model(MODEL)
    manifest = manifest_of(prepared)
    before = [t.cache_key for t in build_classification_tasks(prepared, model, manifest).tasks]

    original = registry.classification_json_schema

    def altered(label_space, count):
        schema = original(label_space, count)
        schema["json_schema"]["schema"]["properties"]["answers"]["minItems"] = count
        return schema

    monkeypatch.setattr("llmbench.benchmarks.classification.classification_json_schema", altered)
    after = [t.cache_key for t in build_classification_tasks(prepared, model, manifest).tasks]
    assert before != after


def test_text_mode_keys_carry_no_schema_hash(prepared):
    tasks = build_classification_tasks(prepared, prepared.require_model(MODEL), manifest_of(prepared)).tasks
    assert all(task.key_material["extra"] == {} for task in tasks)


def test_structured_outputs_is_a_separate_capability_from_response_format(prepared):
    """An endpoint can accept `response_format` and still reject a strict schema."""
    model = prepared.require_model(MODEL)
    assert model.supports_json_schema() is True

    model.parameters.structured_outputs = False
    assert model.supports("response_format") is True
    assert model.supports_json_schema() is False

    # OpenRouter advertises the two names separately, and so do we.
    live = ["response_format", "temperature", "top_p", "max_tokens", "reasoning", "seed", "stop"]
    problems = {p["parameter"]: p for p in model.capability_mismatches(live)}
    assert "structured_outputs" not in problems, "config already says it is unsupported"
    assert "response_format" not in problems


def test_qwen37_flash_is_marked_without_strict_schema_support(prepared):
    """Regression: its only endpoint 404s on a strict schema (17 wasted requests)."""
    model = prepared.require_model("qwen/qwen3.7-flash")
    assert model.supports("response_format") is False or model.supports("structured_outputs") is False
    assert model.supports_json_schema() is False


async def test_a_model_without_strict_schema_support_cannot_run_the_condition(
    prepared, store, client_factory, mock_openrouter
):
    """It must be blocked, not quietly run unconstrained and filed as constrained."""
    prepared.benchmark.structured_output.classification_mode = "json_schema"
    model = prepared.require_model(MODEL)
    model.parameters.structured_outputs = False

    runner = BenchmarkRunner(prepared, store, client_factory=lambda: client_factory())
    outcome = await runner.run(model, benchmarks=["classification"])

    assert outcome.status == "BLOCKED"
    assert mock_openrouter.chat_calls == 0
    assert any("cannot serve a strict json_schema" in note for note in outcome.notes)


async def test_the_same_model_still_runs_in_text_mode(prepared, store, client_factory):
    model = prepared.require_model(MODEL)
    model.parameters.structured_outputs = False

    runner = BenchmarkRunner(prepared, store, client_factory=lambda: client_factory())
    outcome = await runner.run(model, benchmarks=["classification"])
    assert outcome.status == "COMPLETED"
