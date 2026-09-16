"""CLI smoke tests (no network, no inference)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmbench.cli import app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    shutil.copytree(PROJECT_ROOT / "config", tmp_path / "config")
    monkeypatch.setenv("LLMBENCH_PROJECT_ROOT", str(tmp_path))
    # Rich truncates table cells to the terminal width; widen it so the assertions
    # below look at content rather than at ellipses.
    monkeypatch.setenv("COLUMNS", "240")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("AIHUB_FACTUALITY_DATA_PATH", raising=False)
    return tmp_path


def test_help_lists_every_documented_command(isolated):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("prepare", "models", "pricing", "estimate", "run", "report", "cache"):
        assert command in result.stdout


def test_models_lists_openrouter_slugs_and_prices(isolated):
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "openai/gpt-5.6-luna" in result.stdout
    assert "google/gemini-2.5-flash-lite" in result.stdout
    assert "mistralai/ministral-3b-2512" in result.stdout
    assert "openrouter" in result.stdout


def test_version(isolated):
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "llm-benchmark" in result.stdout


def test_run_without_api_key_is_refused(isolated):
    result = runner.invoke(app, ["run", "--model", "google/gemini-2.5-flash-lite"])
    assert result.exit_code == 2
    assert "OPENROUTER_API_KEY" in result.stdout


def test_run_rejects_unknown_model(isolated, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    result = runner.invoke(app, ["run", "--model", "not/a-real-model"])
    assert result.exit_code == 2
    assert "unknown model" in result.stdout


def test_run_rejects_unknown_judge_profile(isolated, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    result = runner.invoke(app, ["run", "--model", "openai/gpt-5.6-luna", "--judge", "medium"])
    assert result.exit_code == 2
    assert "--judge must be one of" in result.stdout


def test_cache_stats_on_empty_database(isolated):
    result = runner.invoke(app, ["cache", "stats"])
    assert result.exit_code == 0
    assert "Inference results: 0" in result.stdout


def test_report_without_results_still_writes_files(isolated):
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 0
    assert (isolated / "results" / "leaderboard.csv").is_file()
    assert (isolated / "results" / "leaderboard.md").is_file()
    assert (isolated / "results" / "report.html").is_file()
