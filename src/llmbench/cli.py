"""`benchmark` command-line interface."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from llmbench.budget import BudgetGuard
from llmbench.config.loader import AppConfig, load_config
from llmbench.config.schema import ModelConfig
from llmbench.core.errors import BenchmarkError
from llmbench.datasets.prepare import prepare_all
from llmbench.db.store import Store
from llmbench.judge.runner import JudgeOutcome, JudgeRunner
from llmbench.logging_setup import configure_logging, get_logger, prune_logs
from llmbench.openrouter.client import OpenRouterClient
from llmbench.pricing import PricingService
from llmbench.reporting.build import generate_reports
from llmbench.reporting.summary import write_model_summary
from llmbench.runner import ALL_BENCHMARKS, BenchmarkRunner, RunOutcome
from llmbench.version import BENCHMARK_VERSION, __version__

console = Console()
log = get_logger("cli")

# Set by the top-level callback so every command shares one logging setup.
_LOG_STATE: dict[str, Any] = {"verbosity": 0, "log_file": None, "path": None}

app = typer.Typer(
    name="benchmark",
    help=(
        "Reproducible summarization / hallucination / classification benchmark for "
        "low-cost LLMs. Every model is called through OpenRouter."
    ),
    no_args_is_help=True,
    add_completion=False,
)
cache_app = typer.Typer(name="cache", help="Inspect and maintain the persistent cache.", no_args_is_help=True)
app.add_typer(cache_app)


@app.callback()
def _main(
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Console log level: -v for INFO, -vv for DEBUG."
    ),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", help="Write the run log here instead of data/logs/."
    ),
) -> None:
    """Common options. Every run also writes a DEBUG log file under data/logs/."""
    _LOG_STATE["verbosity"] = verbose
    _LOG_STATE["log_file"] = log_file


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _load(config_dir: str | None = None, *, logging: bool = True) -> AppConfig:
    try:
        cfg = load_config(config_dir)
    except BenchmarkError as exc:
        console.print(f"[red]configuration error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    if logging:
        _LOG_STATE["path"] = configure_logging(
            log_dir=cfg.log_dir,
            log_file=_LOG_STATE["log_file"],
            verbosity=_LOG_STATE["verbosity"],
        )
        prune_logs(cfg.log_dir, cfg.benchmark.logging.keep_last)
    return cfg


def _log_hint() -> str:
    path = _LOG_STATE.get("path")
    return f"Full log: {path}" if path else "Logging to a file is disabled."


def _store(cfg: AppConfig) -> Store:
    cfg.ensure_dirs()
    return Store(cfg.cache_db_path)


def _load_dotenv(cfg: AppConfig) -> None:
    env_path = cfg.root / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _require_api_key() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        console.print(
            "[red]OPENROUTER_API_KEY is not set.[/red] "
            "Copy .env.example to .env and put your OpenRouter key in it."
        )
        raise typer.Exit(code=2)


def _money(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"${value:,.{digits}f}"


def _num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def _selected_models(cfg: AppConfig, model: str | None, run_all: bool) -> list[ModelConfig]:
    if run_all:
        models = cfg.models.enabled_models()
        if not models:
            console.print("[red]no enabled models in config/models.yaml[/red]")
            raise typer.Exit(code=2)
        return models
    if not model:
        console.print("[red]specify --model <openrouter-slug> or --all[/red]")
        raise typer.Exit(code=2)
    try:
        return [cfg.require_model(model)]
    except BenchmarkError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #
@app.command()
def prepare(
    force: bool = typer.Option(False, "--force", help="Re-sample and overwrite existing manifests."),
    only: Optional[list[str]] = typer.Option(
        None, "--only", help="Limit to specific benchmarks (summarization/hallucination/classification/judge)."
    ),
    config_dir: Optional[str] = typer.Option(None, "--config-dir", help="Alternate config/ directory."),
) -> None:
    """Build the frozen benchmark manifests (downloads the public datasets once)."""
    cfg = _load(config_dir)
    console.print(
        Panel.fit(
            "Frozen manifests are sampled exactly once.\n"
            "Existing manifests are reused unless you pass --force.",
            title="benchmark prepare",
        )
    )
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        results = prepare_all(cfg, force=force, only=only, progress=progress)

    # Mirror the manifests into the database so `samples` is queryable offline.
    try:
        from llmbench.datasets.prepare import load_manifests
        from llmbench.db.registry import register_manifests

        store = _store(cfg)
        register_manifests(cfg, store, load_manifests(cfg))
        store.close()
    except BenchmarkError as exc:
        console.print(f"[yellow]could not register manifests in the database: {exc}[/yellow]")

    table = Table(title="Manifests", show_lines=False)
    table.add_column("Benchmark")
    table.add_column("Status")
    table.add_column("Cases", justify="right")
    table.add_column("Manifest hash")
    table.add_column("Dataset revision")
    exit_code = 0
    for result in results:
        counts = result.counts or {}
        cases = counts.get("total", 0)
        colour = {
            "CREATED": "green",
            "EXISTS": "cyan",
            "SKIPPED": "yellow",
            "FAILED": "red",
        }.get(result.status, "white")
        table.add_row(
            result.name,
            f"[{colour}]{result.status}[/{colour}]",
            f"{cases:,}" if cases else "—",
            result.manifest_hash[:12] or "—",
            (result.dataset_revision or "—")[:12],
        )
        if result.status == "FAILED":
            exit_code = 1
    console.print(table)

    for result in results:
        if result.detail:
            colour = "red" if result.status == "FAILED" else "yellow"
            console.print(f"[{colour}]{result.name}:[/{colour}] {result.detail}")

    if any(r.name == "hallucination:ko" and r.status == "SKIPPED" for r in results):
        console.print(
            "\n[yellow]Korean hallucination benchmark is SKIPPED.[/yellow] "
            "Prepare the AI-Hub data yourself and set AIHUB_FACTUALITY_DATA_PATH; "
            "the rest of the benchmark still runs."
        )
    raise typer.Exit(code=exit_code)


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
@app.command()
def models(
    config_dir: Optional[str] = typer.Option(None, "--config-dir"),
    show_judges: bool = typer.Option(True, "--judges/--no-judges", help="Also list judge models."),
    check: bool = typer.Option(
        False,
        "--check",
        help="Verify the configured parameters against OpenRouter's live metadata.",
    ),
) -> None:
    """List configured OpenRouter model slugs and their configured prices."""
    cfg = _load(config_dir)
    if check:
        _check_capabilities(cfg, include_judges=show_judges)
        return
    constraints = cfg.pricing.constraints

    table = Table(title=f"Configured models (API provider: openrouter)")
    table.add_column("Model slug")
    table.add_column("Display name")
    table.add_column("Vendor")
    table.add_column("Enabled", justify="center")
    table.add_column("Input $/M", justify="right")
    table.add_column("Output $/M", justify="right")
    table.add_column("Max out", justify="right")
    table.add_column("Reasoning", justify="center")
    table.add_column("Routing")
    table.add_column("Within limits", justify="center")

    def add(entry: ModelConfig) -> None:
        price = cfg.pricing.get(entry.model_id)
        within = cfg.pricing.within_constraints(entry.model_id)
        within_text = "—" if within is None else ("[green]yes[/green]" if within else "[red]no[/red]")
        routing = entry.routing
        routing_text = routing.mode
        if routing.provider_order:
            routing_text += f" ({', '.join(routing.provider_order)})"
        if not routing.allow_fallbacks:
            routing_text += " no-fallback"
        table.add_row(
            entry.model_id,
            entry.display_name,
            entry.vendor,
            "yes" if entry.enabled else "no",
            _num(price.input_per_million, 4),
            _num(price.output_per_million, 4),
            str(entry.max_output_tokens),
            "on" if entry.reasoning.enabled else "off",
            routing_text,
            within_text,
        )

    for entry in cfg.models.models:
        add(entry)
    if show_judges:
        for entry in cfg.models.judges:
            add(entry)

    console.print(table)
    console.print(
        f"Price limits for the default benchmark: input ≤ ${constraints.max_input_per_million}/M, "
        f"output ≤ ${constraints.max_output_per_million}/M"
    )
    console.print("Prices come from config/pricing.yaml; run `benchmark pricing` to compare with OpenRouter.")


def _check_capabilities(cfg: AppConfig, *, include_judges: bool = True) -> None:
    """Compare `parameters:` in models.yaml against OpenRouter's supported list.

    This is the check that would have caught 1,300 identical HTTP 404s up front.
    """
    _load_dotenv(cfg)
    entries = list(cfg.models.models) + (list(cfg.models.judges) if include_judges else [])
    model_ids = [m.model_id for m in entries]

    async def _fetch() -> dict[str, Any]:
        async with OpenRouterClient() as client:
            return await client.get_model_metadata(model_ids)

    try:
        metadata = asyncio.run(_fetch())
    except Exception as exc:  # network/API problems must not look like a config error
        console.print(f"[red]could not reach OpenRouter:[/red] {type(exc).__name__}: {exc}")
        raise typer.Exit(code=2) from exc

    table = Table(title="Configured parameters vs OpenRouter metadata")
    table.add_column("Model slug")
    table.add_column("Result")
    table.add_column("Detail", overflow="fold")

    problems = 0
    for entry in entries:
        live = metadata.get(entry.model_id)
        if live is None:
            table.add_row(entry.model_id, "[red]NOT LISTED[/red]", "OpenRouter does not list this slug")
            problems += 1
            continue
        mismatches = entry.capability_mismatches(live.get("supported_parameters") or [])
        errors = [m for m in mismatches if m["severity"] == "error"]
        warnings = [m for m in mismatches if m["severity"] == "warning"]
        if errors:
            problems += 1
            table.add_row(
                entry.model_id,
                "[red]MISMATCH[/red]",
                "; ".join(m["hint"] for m in errors),
            )
        elif warnings:
            table.add_row(
                entry.model_id,
                "[yellow]OUTDATED[/yellow]",
                "; ".join(m["hint"] for m in warnings),
            )
        else:
            table.add_row(entry.model_id, "[green]OK[/green]", "")
    console.print(table)

    if problems:
        console.print(
            f"\n[red]{problems} model(s) would send a parameter OpenRouter does not support.[/red]\n"
            "With routing.require_parameters=true every request returns HTTP 404 "
            "'No endpoints found that can handle the requested parameters'.\n"
            "Set the listed flags in config/models.yaml, then re-run this check."
        )
        raise typer.Exit(code=1)
    console.print("\n[green]All configured parameters are supported.[/green]")


# --------------------------------------------------------------------------- #
# pricing
# --------------------------------------------------------------------------- #
@app.command()
def pricing(
    config_dir: Optional[str] = typer.Option(None, "--config-dir"),
    refresh: bool = typer.Option(True, "--refresh/--no-refresh", help="Query OpenRouter for live pricing."),
    save: bool = typer.Option(True, "--save/--no-save", help="Store a pricing snapshot in the database."),
) -> None:
    """Fetch current OpenRouter pricing metadata and compare it with the local config."""
    cfg = _load(config_dir)
    _load_dotenv(cfg)
    store = _store(cfg)
    service = PricingService(cfg, store)
    model_ids = [m.model_id for m in cfg.models.models] + [m.model_id for m in cfg.models.judges]

    async def _run() -> dict[str, Any]:
        if not refresh:
            return {mid: service.resolve(mid) for mid in model_ids}
        async with OpenRouterClient() as client:
            return await service.refresh(client, model_ids, persist=save)

    live = asyncio.run(_run())
    if service.last_refresh_error:
        console.print(
            f"[yellow]live pricing lookup failed ({service.last_refresh_error}); "
            "falling back to config/pricing.yaml[/yellow]"
        )
    rows = service.compare(live)

    table = Table(title="OpenRouter pricing (USD per 1M tokens)")
    table.add_column("Model slug")
    table.add_column("Config in", justify="right")
    table.add_column("Live in", justify="right")
    table.add_column("Δ in", justify="right")
    table.add_column("Config out", justify="right")
    table.add_column("Live out", justify="right")
    table.add_column("Δ out", justify="right")
    table.add_column("Cached in", justify="right")
    table.add_column("Source")
    table.add_column("Within limits", justify="center")

    for row in rows:
        def delta_text(value: float | None) -> str:
            if value is None:
                return "—"
            colour = "green" if value < 0 else ("red" if value > 0 else "white")
            return f"[{colour}]{value:+.4f}[/{colour}]"

        within = row["within_constraints"]
        table.add_row(
            row["model_id"],
            _num(row["config_input"], 4),
            _num(row["live_input"], 4),
            delta_text(row["input_delta"]),
            _num(row["config_output"], 4),
            _num(row["live_output"], 4),
            delta_text(row["output_delta"]),
            _num(row["live_cached_input"], 4),
            row["source"],
            "—" if within is None else ("[green]yes[/green]" if within else "[red]no[/red]"),
        )
    console.print(table)
    console.print(
        "[dim]Pricing is used for estimates only. The billed amount is OpenRouter's "
        "usage.cost on each response.[/dim]"
    )
    store.close()


# --------------------------------------------------------------------------- #
# estimate
# --------------------------------------------------------------------------- #
def _print_plan(cfg: AppConfig, outcome: RunOutcome, budget: float) -> None:
    plan = outcome.plan
    estimate = plan.estimate

    console.print()
    console.print(f"[bold]Model:[/bold]\n{plan.model.model_id}\n")
    console.print("[bold]API Provider:[/bold]\nOpenRouter\n")

    pending = Table(show_header=True, title="Pending (requests / cases)")
    pending.add_column("Benchmark")
    pending.add_column("Language")
    pending.add_column("Pending requests", justify="right")
    pending.add_column("Total requests", justify="right")
    pending.add_column("Pending cases", justify="right")
    pending.add_column("Total cases", justify="right")
    for name, stats in sorted(estimate.per_benchmark.items()):
        for language, lang_stats in sorted(stats["per_language"].items()):
            pending.add_row(
                name,
                language,
                f"{lang_stats['pending_tasks']:,}",
                f"{lang_stats['total_tasks']:,}",
                f"{lang_stats['pending_cases']:,}",
                f"{lang_stats['total_cases']:,}",
            )
    console.print(pending)

    for key, reason in sorted(plan.skipped.items()):
        console.print(f"[yellow]{key}: SKIPPED[/yellow] — {reason}")

    console.print(f"\n[bold]Cached:[/bold]\n{estimate.cached_cases:,} cases ({estimate.cached_tasks:,} requests)")
    console.print(f"\n[bold]Estimated new input tokens:[/bold]\n{estimate.est_input_tokens:,}")
    console.print(f"[bold]Estimated new output tokens:[/bold]\n{estimate.est_output_tokens:,}")

    pricing_snapshot = outcome.cost.get("pricing_snapshot", {})
    console.print(
        f"\n[bold]Pricing used for the estimate:[/bold] "
        f"in {_money(pricing_snapshot.get('input_per_million'))}/M, "
        f"out {_money(pricing_snapshot.get('output_per_million'))}/M "
        f"(source: {pricing_snapshot.get('source')})"
    )
    console.print("\n[bold]Estimated fresh OpenRouter cost:[/bold]")
    console.print(f"Input:  {_money(estimate.est_input_cost)}")
    console.print(f"Output: {_money(estimate.est_output_cost)}")
    console.print(f"Total:  {_money(estimate.est_total_cost)}")
    console.print(f"\n[bold]Budget:[/bold]\n{_money(budget, 2)}")

    status = outcome.cost.get("budget_status", "SAFE TO RUN")
    colour = "green" if status == "SAFE TO RUN" else "red"
    console.print(f"\n[bold]Status:[/bold]\n[{colour}]{status}[/{colour}]")
    if outcome.cost.get("budget_reason") and status != "SAFE TO RUN":
        console.print(f"[yellow]{outcome.cost['budget_reason']}[/yellow]")


@app.command()
def estimate(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="OpenRouter model slug."),
    all_models: bool = typer.Option(False, "--all", help="Estimate every enabled model."),
    benchmark: Optional[list[str]] = typer.Option(None, "--benchmark", "-b", help="Limit to benchmarks."),
    language: Optional[list[str]] = typer.Option(None, "--language", "-l", help="Limit to languages (en/ko)."),
    budget_usd: float = typer.Option(None, "--budget-usd", help="Override the per-model budget."),
    refresh_pricing: bool = typer.Option(True, "--refresh-pricing/--no-refresh-pricing"),
    config_dir: Optional[str] = typer.Option(None, "--config-dir"),
) -> None:
    """Estimate the cost of the *pending* (uncached) work for a model."""
    cfg = _load(config_dir)
    _load_dotenv(cfg)
    store = _store(cfg)
    runner = BenchmarkRunner(cfg, store)
    budget = budget_usd if budget_usd is not None else cfg.benchmark.budget.default_usd

    for entry in _selected_models(cfg, model, all_models):
        outcome = asyncio.run(
            runner.run(
                entry,
                benchmarks=benchmark,
                languages=language,
                budget_usd=budget,
                dry_run=True,
                refresh_pricing=refresh_pricing,
            )
        )
        _print_plan(cfg, outcome, budget)
    store.close()


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def _print_run_result(cfg: AppConfig, outcome: RunOutcome) -> None:
    plan = outcome.plan
    cost = outcome.cost or {}

    console.print()
    console.print(f"[bold]Model:[/bold] {plan.model.model_id}")
    console.print()
    console.print(f"Total cases: {plan.estimate.total_cases:,}")
    console.print(f"Cache hits: {plan.estimate.cached_cases:,}")
    console.print(f"Cache misses: {plan.estimate.total_cases - plan.estimate.cached_cases:,}")
    console.print(f"OpenRouter API calls: {outcome.api_calls:,}")
    console.print()
    console.print(f"Fresh OpenRouter cost:     {_money(cost.get('fresh_cost_this_run'), 6)}")
    console.print(f"Stored benchmark cost:     {_money(cost.get('stored_benchmark_cost'), 6)}")
    console.print(f"Estimated cost w/o cache:  {_money(cost.get('estimated_cost_without_cache'), 6)}")
    console.print(f"Estimated cache savings:   {_money(cost.get('estimated_cache_savings'), 6)}")

    summary = outcome.failure_summary or {}
    if outcome.failures:
        console.print()
        failure_table = Table(title="Non-success cases")
        failure_table.add_column("Status")
        failure_table.add_column("Count", justify="right")
        for status, count in sorted(outcome.failures.items()):
            failure_table.add_row(status, f"{count:,}")
        console.print(failure_table)

        # The count alone is useless; show what OpenRouter actually said.
        top_errors = summary.get("top_errors") or []
        if top_errors:
            error_table = Table(title="Error messages")
            error_table.add_column("Count", justify="right")
            error_table.add_column("Message", overflow="fold")
            for entry in top_errors:
                error_table.add_row(f"{entry['count']:,}", entry["message"])
            console.print(error_table)

        first = summary.get("first_failure") or {}
        if first.get("http_status"):
            console.print(
                f"[dim]First failure: {first.get('status')} HTTP {first['http_status']} "
                f"on sample {first.get('sample_id')}[/dim]"
            )
        console.print(f"[dim]{_log_hint()}[/dim]")
        console.print("[yellow]Re-run with --retry-failed to retry these cases.[/yellow]")

    if outcome.aborted:
        console.print()
        console.print(
            Panel.fit(
                "\n".join(outcome.notes) or "aborted",
                title="[red]RUN ABORTED[/red]",
                border_style="red",
            )
        )
        console.print(
            f"Skipped without being attempted: {outcome.skipped_by_failure_guard:,} requests "
            "(nothing stored, so a re-run retries them normally)."
        )

    metrics = outcome.metrics or {}
    if "summarization" in metrics:
        table = Table(title="Summarization (XL-Sum)")
        table.add_column("Language")
        for column in ("ROUGE-Lsum", "chrF++", "BERTScore", "Surface facts*", "Compression", "Empty", "Refusal"):
            table.add_column(column, justify="right")
        for language, block in sorted(metrics["summarization"]["per_language"].items()):
            table.add_row(
                language,
                _num(block.get("rougeLsum_f")),
                _num(block.get("chrf"), 2),
                _num(block.get("bertscore_f1")),
                _num(block.get("surface_fact_support_precision")),
                _num(block.get("compression_ratio")),
                _num(block.get("empty_output_rate")),
                _num(block.get("refusal_rate")),
            )
        console.print(table)
        console.print("[dim]* surface fact support is a diagnostic, not a hallucination metric[/dim]")

    if "hallucination" in metrics:
        table = Table(title="Hallucination detection")
        table.add_column("Language")
        for column in ("Accuracy", "Macro-F1", "Halu P", "Halu R", "Halu F1", "Supported R", "Invalid"):
            table.add_column(column, justify="right")
        for language, block in sorted(metrics["hallucination"]["per_language"].items()):
            if block.get("status") == "SKIPPED":
                table.add_row(language, "SKIPPED", "—", "—", "—", "—", "—", "—")
                continue
            table.add_row(
                language,
                _num(block.get("accuracy")),
                _num(block.get("macro_f1")),
                _num(block.get("hallucination_precision")),
                _num(block.get("hallucination_recall")),
                _num(block.get("hallucination_f1")),
                _num(block.get("supported_recall")),
                _num(block.get("invalid_output_rate")),
            )
        console.print(table)

    if "classification" in metrics:
        block = metrics["classification"]
        table = Table(title="Classification (MASSIVE)")
        table.add_column("Scope")
        table.add_column("Accuracy", justify="right")
        table.add_column("Macro-F1", justify="right")
        table.add_column("Invalid", justify="right")
        for language, stats in sorted(block["per_language"].items()):
            table.add_row(language, _num(stats.get("accuracy")), _num(stats.get("macro_f1")), _num(stats.get("invalid_output_rate")))
        overall = block.get("overall") or {}
        if overall:
            table.add_row("overall", _num(overall.get("accuracy")), _num(overall.get("macro_f1")), _num(overall.get("invalid_output_rate")))
        console.print(table)

        cross = block.get("cross_lingual") or {}
        if cross.get("paired_cases"):
            cross_table = Table(title="EN/KO cross-lingual consistency")
            cross_table.add_column("Metric")
            cross_table.add_column("Rate", justify="right")
            cross_table.add_column("Count", justify="right")
            cross_table.add_row("Consistency", _num(cross.get("consistency")), f"{cross['paired_cases']:,}")
            cross_table.add_row("Both correct", _num(cross.get("both_correct_rate")), f"{cross.get('both_correct', 0):,}")
            cross_table.add_row("EN only correct", _num(cross.get("en_only_correct_rate")), f"{cross.get('en_only_correct', 0):,}")
            cross_table.add_row("KO only correct", _num(cross.get("ko_only_correct_rate")), f"{cross.get('ko_only_correct', 0):,}")
            cross_table.add_row("Both wrong", _num(cross.get("both_wrong_rate")), f"{cross.get('both_wrong', 0):,}")
            console.print(cross_table)

    usage = outcome.usage or {}
    usage_table = Table(title="Actual OpenRouter usage (all stored cases for this model)")
    usage_table.add_column("Field")
    usage_table.add_column("Value", justify="right")
    for label, key in (
        ("Prompt Tokens", "prompt_tokens"),
        ("Cached Tokens", "cached_tokens"),
        ("Cache Write Tokens", "cache_write_tokens"),
        ("Completion Tokens", "completion_tokens"),
        ("Reasoning Tokens", "reasoning_tokens"),
        ("Total Tokens", "total_tokens"),
    ):
        usage_table.add_row(label, f"{int(usage.get(key, 0)):,}")
    console.print(usage_table)

    cost_table = Table(title="Actual cost (OpenRouter usage.cost)")
    cost_table.add_column("Item")
    cost_table.add_column("USD", justify="right")
    for label, key in (
        ("Summarization cost", "summarization_cost"),
        ("Hallucination cost", "hallucination_cost"),
        ("Classification cost", "classification_cost"),
        ("Retry/error cost", "retry_error_cost"),
        ("Judge cost", "judge_cost"),
        ("Total OpenRouter cost", "total_openrouter_cost"),
    ):
        cost_table.add_row(label, _money(cost.get(key), 6))
    console.print(cost_table)

    snapshot = cost.get("pricing_snapshot", {})
    console.print(
        f"[dim]Pricing snapshot: in ${snapshot.get('input_per_million')}/M, "
        f"cached in ${snapshot.get('cached_input_per_million')}/M, "
        f"out ${snapshot.get('output_per_million')}/M, retrieved {snapshot.get('retrieved_at')} "
        f"({snapshot.get('source')})[/dim]"
    )
    reconciliation = cost.get("reconciliation") or {}
    if reconciliation and not reconciliation.get("matches_within_tolerance", True):
        console.print(
            f"[yellow]Pricing-table recomputation "
            f"{_money(reconciliation['recomputed_from_pricing_table'], 6)} differs from billed "
            f"{_money(reconciliation['actual_usage_cost'], 6)}. usage.cost is authoritative.[/yellow]"
        )

    judge = outcome.judge or {}
    if judge.get("enabled"):
        for warning in judge.get("warnings", []):
            console.print(f"[yellow]{warning}[/yellow]")
        scores = judge.get("scores", {}) or {}
        judge_table = Table(title=f"LLM-as-a-Judge ({judge.get('judge_model')}) — advisory only")
        judge_table.add_column("Dimension")
        judge_table.add_column("Mean (1-5)", justify="right")
        for dimension in ("factual_consistency", "key_information_coverage", "conciseness", "overall_quality"):
            judge_table.add_row(dimension, _num(scores.get(dimension)))
        console.print(judge_table)
        console.print(
            f"Judge cases: {scores.get('judged_cases', 0):,} · "
            f"Judge cost (separate from the ${cfg.benchmark.budget.default_usd:.2f} benchmark budget): "
            f"{_money(judge.get('judge_cost_usd'), 6)}"
        )


async def _run_judge(
    cfg: AppConfig,
    store: Store,
    runner: BenchmarkRunner,
    outcome: RunOutcome,
    *,
    profile: str,
    samples: int | None,
    languages: list[str] | None,
) -> JudgeOutcome:
    manifest = runner.manifests().get("judge")
    if manifest is None:
        result = JudgeOutcome(enabled=True, profile=profile, status="NO_MANIFEST")
        result.warnings.append("judge manifest missing; run `benchmark prepare`")
        return result
    judge_runner = JudgeRunner(cfg, store)
    results = runner.collect_results(outcome.plan)
    async with OpenRouterClient(
        timeout=cfg.benchmark.concurrency.request_timeout_seconds,
        max_attempts=cfg.benchmark.retry.max_attempts,
    ) as client:
        return await judge_runner.run(
            profile=profile,
            target_model=outcome.plan.model,
            manifest=manifest,
            summarization_results=results,
            client=client,
            languages=languages,
            samples=samples,
            run_id=outcome.run_id,
        )


@app.command()
def run(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="OpenRouter model slug."),
    all_models: bool = typer.Option(False, "--all", help="Run every enabled model."),
    benchmark: Optional[list[str]] = typer.Option(
        None, "--benchmark", "-b", help=f"Limit to benchmarks: {', '.join(ALL_BENCHMARKS)}."
    ),
    language: Optional[list[str]] = typer.Option(None, "--language", "-l", help="Limit to languages (en/ko)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan only; never calls the inference API."),
    budget_usd: float = typer.Option(None, "--budget-usd", help="Per-model budget in USD."),
    allow_over_budget: bool = typer.Option(False, "--allow-over-budget", help="Run even if the estimate exceeds the budget."),
    retry_failed: bool = typer.Option(False, "--retry-failed", help="Also retry cases with a stored non-SUCCESS result."),
    force: bool = typer.Option(False, "--force", help="Ignore cached successes and re-run everything."),
    judge: str = typer.Option("none", "--judge", help="LLM-as-a-Judge profile: none | cheap | strong."),
    judge_samples: Optional[int] = typer.Option(None, "--judge-samples", help="Number of judge cases."),
    no_metric_cache: bool = typer.Option(False, "--no-metric-cache", help="Recompute metrics, ignoring the metric cache."),
    refresh_pricing: bool = typer.Option(True, "--refresh-pricing/--no-refresh-pricing"),
    skip_preflight: bool = typer.Option(
        False, "--skip-preflight", help="Run even if the model rejects a parameter we send."
    ),
    report: bool = typer.Option(
        True, "--report/--no-report", help="Regenerate RESULT.md, the workbook and the leaderboard."
    ),
    config_dir: Optional[str] = typer.Option(None, "--config-dir"),
) -> None:
    """Run the benchmark for one model (or all of them)."""
    cfg = _load(config_dir)
    _load_dotenv(cfg)
    if not dry_run:
        _require_api_key()
    if judge not in {"none", "cheap", "strong"}:
        console.print("[red]--judge must be one of: none, cheap, strong[/red]")
        raise typer.Exit(code=2)

    store = _store(cfg)
    budget = budget_usd if budget_usd is not None else cfg.benchmark.budget.default_usd
    exit_code = 0

    for entry in _selected_models(cfg, model, all_models):
        console.rule(f"[bold]{entry.model_id}")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("OpenRouter requests", total=1)

            def on_progress(event: str, payload: dict[str, Any]) -> None:
                if event == "task":
                    progress.update(
                        task_id,
                        total=payload["total"],
                        completed=payload["completed"],
                        description=f"{payload['benchmark']} · fresh {_money(payload['fresh_cost'], 6)}",
                    )

            runner = BenchmarkRunner(cfg, store, on_progress=on_progress)
            outcome = asyncio.run(
                runner.run(
                    entry,
                    benchmarks=benchmark,
                    languages=language,
                    budget_usd=budget,
                    allow_over_budget=allow_over_budget,
                    dry_run=dry_run,
                    retry_failed=retry_failed,
                    force=force,
                    refresh_pricing=refresh_pricing,
                    skip_preflight=skip_preflight,
                    use_metric_cache=not no_metric_cache,
                )
            )

        if dry_run:
            _print_plan(cfg, outcome, budget)
            console.print("\n[dim]Dry run: no OpenRouter inference request was made.[/dim]")
            continue

        if outcome.status == "BLOCKED":
            console.print(
                Panel.fit(
                    "\n\n".join(outcome.notes) or "blocked",
                    title="[red]BLOCKED - nothing was sent to OpenRouter[/red]",
                    border_style="red",
                )
            )
            console.print(f"[dim]{_log_hint()}[/dim]")
            exit_code = 1
            continue

        if judge != "none":
            judge_outcome = asyncio.run(
                _run_judge(
                    cfg,
                    store,
                    runner,
                    outcome,
                    profile=judge,
                    samples=judge_samples,
                    languages=language,
                )
            )
            outcome.judge = judge_outcome.to_dict()
            outcome.cost["judge_cost"] = store.judge_cost(entry.model_id)
            outcome.cost["total_openrouter_cost"] = (
                outcome.cost.get("stored_benchmark_cost", 0.0) + outcome.cost["judge_cost"]
            )

        _print_run_result(cfg, outcome)
        outcome.log_path = str(_LOG_STATE.get("path") or "")
        path = write_model_summary(cfg, outcome)
        console.print(f"\n[green]Wrote[/green] {path.relative_to(cfg.root) if path.is_relative_to(cfg.root) else path}")
        if _LOG_STATE.get("path"):
            console.print(f"[dim]{_log_hint()}[/dim]")
        if outcome.status in {"BUDGET_STOPPED", "ABORTED"}:
            exit_code = 1

    store.close()

    if report and not dry_run:
        bundle = generate_reports(cfg)
        console.rule("[bold]reports")
        for name in ("result_md", "excel", "markdown", "csv", "json", "html"):
            path = bundle.paths.get(name)
            if path is None:
                continue
            shown = path.relative_to(cfg.root) if path.is_relative_to(cfg.root) else path
            console.print(f"[green]{name}[/green]: {shown}")

    if exit_code:
        raise typer.Exit(code=exit_code)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
@app.command()
def report(
    config_dir: Optional[str] = typer.Option(None, "--config-dir"),
) -> None:
    """Build results/leaderboard.{csv,json,md} and results/report.html."""
    cfg = _load(config_dir)
    bundle = generate_reports(cfg)
    rows = bundle.rows
    if not bundle.summaries:
        console.print("[yellow]No model summaries found. Run `benchmark run --all` first.[/yellow]")

    table = Table(title="Leaderboard")
    table.add_column("Model")
    for column in ("In $/M", "Out $/M", "Sum EN", "Sum KO", "Halu EN F1", "Halu KO F1", "Cls EN", "Cls KO", "EN-KO", "Cost"):
        table.add_column(column, justify="right")
    for row in rows:
        table.add_row(
            row["model"],
            _num(row["input_per_million"]),
            _num(row["output_per_million"]),
            _num(row["summary_en"]),
            _num(row["summary_ko"]),
            _num(row["halu_en_f1"]),
            _num(row["halu_ko_f1"]),
            _num(row["cls_en"]),
            _num(row["cls_ko"]),
            _num(row["en_ko_consistency"]),
            _money(row["openrouter_cost"], 6),
        )
    console.print(table)
    for name in ("result_md", "excel", "markdown", "csv", "json", "html"):
        path = bundle.paths.get(name)
        if path is None:
            continue
        shown = path.relative_to(cfg.root) if path.is_relative_to(cfg.root) else path
        console.print(f"[green]{name}[/green]: {shown}")


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
@cache_app.command("stats")
def cache_stats(
    config_dir: Optional[str] = typer.Option(None, "--config-dir"),
) -> None:
    """Show what is in the persistent cache and what it cost to build."""
    cfg = _load(config_dir)
    store = _store(cfg)
    stats = store.global_stats()

    console.print(
        Panel.fit(
            f"Database: {cfg.cache_db_path}\n"
            f"Inference results: {stats['inference_results']:,} "
            f"({stats['successful_results']:,} SUCCESS)\n"
            f"HTTP attempts: {stats['inference_attempts']:,}\n"
            f"Metric results: {stats['metric_results']:,}\n"
            f"Judge results: {stats['judge_results']:,}\n"
            f"Total billed (usage.cost): {_money(stats['total_usage_cost'], 6)}",
            title="cache stats",
        )
    )

    table = Table(title="Per model / benchmark")
    table.add_column("Model")
    table.add_column("Benchmark")
    table.add_column("Status")
    table.add_column("Cases", justify="right")
    table.add_column("usage.cost", justify="right")
    for row in sorted(stats["per_model"], key=lambda r: (r["model_id"], r["benchmark"], r["status"])):
        table.add_row(
            row["model_id"],
            row["benchmark"],
            row["status"],
            f"{row['count']:,}",
            _money(row["usage_cost"], 6),
        )
    console.print(table)
    store.close()


@cache_app.command("clear-failed")
def cache_clear_failed(
    model: str = typer.Option(..., "--model", "-m", help="OpenRouter model slug."),
    benchmark: Optional[str] = typer.Option(None, "--benchmark", "-b"),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
    config_dir: Optional[str] = typer.Option(None, "--config-dir"),
) -> None:
    """Delete stored non-SUCCESS results so they are attempted again.

    Successful results are never touched by this command.
    """
    cfg = _load(config_dir)
    store = _store(cfg)
    keys = store.failed_keys(model, benchmark)
    if not keys:
        console.print("[green]nothing to clear[/green]")
        store.close()
        return
    if not yes:
        confirm = typer.confirm(f"Delete {len(keys):,} non-SUCCESS results for {model}?")
        if not confirm:
            store.close()
            raise typer.Abort()
    deleted = store.delete_results(keys)
    console.print(f"[green]deleted[/green] {deleted:,} non-SUCCESS results")
    store.close()


# --------------------------------------------------------------------------- #
@app.command()
def version() -> None:
    """Print the benchmark version."""
    console.print(f"llm-benchmark {__version__} (benchmark version {BENCHMARK_VERSION})")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
