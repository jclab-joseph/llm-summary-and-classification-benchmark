"""A Decisions API model run end to end against the mock OpenRouter transport."""

from __future__ import annotations

import pytest

from llmbench.openrouter.client import OpenRouterClient
from llmbench.runner import BenchmarkRunner

MODEL = "microsoft/microsoft-decision-1"


@pytest.fixture
def runner(prepared, store, client_factory):
    return BenchmarkRunner(prepared, store, client_factory=lambda: client_factory())


async def test_decision_model_runs_the_closed_label_benchmarks(prepared, runner, mock_openrouter):
    model = prepared.require_model(MODEL)

    outcome = await runner.run(model)

    assert outcome.status == "COMPLETED"
    # It writes no text, so there is nothing to summarize with.
    assert "summarization" not in outcome.plan.groups
    assert "summarization" in outcome.plan.skipped
    assert mock_openrouter.chat_calls == 0
    assert mock_openrouter.decision_calls == len(outcome.plan.tasks) > 0
    # The price came from the live listing, which only has it with every modality.
    assert outcome.cost["pricing_snapshot"]["source"] == "openrouter"
    assert outcome.cost["total_openrouter_cost"] == pytest.approx(mock_openrouter.total_cost)

    classification = outcome.metrics["classification"]
    assert classification["output_mode"] == "decisions"
    assert classification["batch_size"] == 1
    for scores in classification["per_language"].values():
        assert scores["invalid_output_rate"] == 0.0
    hallucination = outcome.metrics["hallucination"]["per_language"]["en"]
    assert hallucination["cases"] > 0
    assert hallucination["invalid_output_rate"] == 0.0

    # One utterance per request, asked as a choice over the whole label space.
    labels = set(outcome.plan.groups["classification"].tasks[0].candidates)
    utterances = [r for r in mock_openrouter.requests if "utterance" in r["state"]]
    assert len(utterances) == classification["overall"]["cases"]
    assert all(set(r["questions"]["answer"]["criteria"]) == labels for r in utterances)
    verdicts = [r for r in mock_openrouter.requests if "candidate_summary" in r["state"]]
    assert all(
        set(r["questions"]["answer"]["criteria"]) == {"SUPPORTED", "HALLUCINATED"} for r in verdicts
    )

    mock_openrouter.reset_counters()
    again = await runner.run(model)
    assert mock_openrouter.decision_calls == 0
    assert again.fresh_cost_usd == 0.0


def test_a_decisions_reply_is_read_as_the_chosen_option():
    client = OpenRouterClient(api_key="k")
    raw = {
        "id": "gen-dec-1",
        "model": "microsoft/microsoft-decision-1-20261009",
        "provider": "Azure",
        "answers": {
            "answer": {
                "type": "choice",
                "choice": "SUPPORTED",
                "probabilities": {"SUPPORTED": 0.56, "HALLUCINATED": 0.44},
            }
        },
        "usage": {"input_tokens": 609, "output_tokens": 1, "cost": 2.5578e-05},
    }
    response = client.normalize_response(raw, requested_model="microsoft/microsoft-decision-1")
    assert response.ok
    assert response.text == "SUPPORTED"
    assert response.finish_reason == "decision"
    assert response.usage.prompt_tokens == 609
    assert response.usage.completion_tokens == 1
    assert response.usage.cost == pytest.approx(2.5578e-05)


def test_decisions_url_sits_beside_v1():
    assert OpenRouterClient(api_key="k").decisions_url == "https://openrouter.ai/api/alpha/decisions"


def test_chat_models_keep_their_cache_keys(prepared):
    """Adding the decision path must not touch the key of any existing chat task."""
    from llmbench.benchmarks.hallucination import build_hallucination_tasks
    from llmbench.datasets.prepare import load_manifests

    manifests = load_manifests(prepared)
    model = prepared.require_model("google/gemini-2.5-flash-lite")
    group = build_hallucination_tasks(prepared, model, {"en": manifests.get("hallucination:en")})
    for task in group.tasks:
        assert task.decision is None
        assert "decision_hash" not in task.key_material.get("extra", {})
