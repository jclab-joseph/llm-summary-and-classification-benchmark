#!/usr/bin/env python3
"""Full-scale end-to-end demo against a mock OpenRouter transport.

Runs the *real* benchmark -- real frozen manifests, real client, real cache,
real metrics, real reports -- with the HTTP layer replaced by an in-process
mock. Nothing is sent to OpenRouter and nothing is billed.

    uv run python scripts/mock_end_to_end.py [--model google/gemini-2.5-flash-lite]

It verifies the three properties that are easiest to get wrong:

1. the second identical run issues exactly zero inference requests;
2. the reported cost equals the sum of the mock's `usage.cost` values;
3. cache savings equal what the cache avoided re-paying.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from llmbench.config.loader import load_config  # noqa: E402
from llmbench.db.store import Store  # noqa: E402
from llmbench.openrouter.client import OpenRouterClient  # noqa: E402
from llmbench.reporting.build import generate_reports  # noqa: E402
from llmbench.reporting.summary import write_model_summary  # noqa: E402
from llmbench.runner import BenchmarkRunner  # noqa: E402
from mock_openrouter import MockOpenRouter  # noqa: E402


def money(value: float) -> str:
    return f"${value:,.6f}"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemini-2.5-flash-lite")
    parser.add_argument("--keep", action="store_true", help="Keep the temporary workspace.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Copy the generated reports into this directory (marked as mock output).",
    )
    args = parser.parse_args()

    manifests = PROJECT_ROOT / "data" / "manifests"
    if not any(manifests.glob("*.jsonl")):
        print("No frozen manifests found. Run `benchmark prepare` first.", file=sys.stderr)
        return 2

    workspace = Path(tempfile.mkdtemp(prefix="llmbench-mock-e2e-"))
    try:
        shutil.copytree(PROJECT_ROOT / "config", workspace / "config")
        shutil.copytree(manifests, workspace / "data" / "manifests")
        cfg = load_config(workspace / "config", root=workspace)
        cfg.ensure_dirs()

        mock = MockOpenRouter()
        store = Store(cfg.cache_db_path)
        runner = BenchmarkRunner(
            cfg,
            store,
            client_factory=lambda: OpenRouterClient(api_key="mock", transport=mock.transport()),
        )
        model = cfg.require_model(args.model)

        print(f"workspace: {workspace}")
        print(f"model:     {model.model_id}\n")

        # ---------------------------------------------------------------- #
        print("=== dry run ===")
        dry = await runner.run(model, dry_run=True)
        assert mock.chat_calls == 0, "a dry run must not call the inference API"
        print(f"pending requests: {dry.plan.estimate.pending_tasks:,}")
        print(f"pending cases:    {dry.plan.estimate.pending_cases:,}")
        print(f"estimated cost:   {money(dry.cost['estimated_fresh_cost'])}")
        print(f"budget status:    {dry.cost['budget_status']}")
        print(f"inference calls:  {mock.chat_calls}\n")

        # ---------------------------------------------------------------- #
        print("=== first run ===")
        first = await runner.run(model, allow_over_budget=True)
        print(f"status:               {first.status}")
        print(f"OpenRouter API calls: {first.api_calls:,}")
        print(f"cache hits/misses:    {first.cache_hits:,} / {first.cache_misses:,}")
        print(f"fresh cost this run:  {money(first.cost['fresh_cost_this_run'])}")
        print(f"stored benchmark cost:{money(first.cost['stored_benchmark_cost'])}")
        print(f"mock billed total:    {money(mock.total_cost)}")
        print(f"prompt tokens:        {first.usage['prompt_tokens']:,}")
        print(f"completion tokens:    {first.usage['completion_tokens']:,}\n")

        billed = mock.total_cost
        assert abs(first.cost["stored_benchmark_cost"] - billed) < 1e-9, "cost must equal usage.cost"
        assert abs(sum(first.cost["by_benchmark"].values()) - billed) < 1e-9

        for name, block in first.metrics.items():
            if name.startswith("_"):
                continue
            print(f"[{name}]")
            for language, stats in sorted(block.get("per_language", {}).items()):
                keys = [k for k in ("rougeLsum_f", "chrf", "accuracy", "macro_f1", "hallucination_f1") if k in stats]
                rendered = ", ".join(f"{k}={stats[k]:.4f}" for k in keys if isinstance(stats.get(k), float))
                print(f"  {language}: {rendered or stats.get('status', '')}")
            cross = block.get("cross_lingual") or {}
            if cross.get("paired_cases"):
                print(f"  cross-lingual consistency: {cross['consistency']:.4f} over {cross['paired_cases']:,} pairs")
        print()

        # ---------------------------------------------------------------- #
        print("=== second run (identical) ===")
        mock.reset_counters()
        second = await runner.run(model, allow_over_budget=True)
        print(f"OpenRouter inference calls: {mock.chat_calls}")
        print(f"fresh cost this run:        {money(second.cost['fresh_cost_this_run'])}")
        print(f"stored benchmark cost:      {money(second.cost['stored_benchmark_cost'])}")
        print(f"estimated cost w/o cache:   {money(second.cost['estimated_cost_without_cache'])}")
        print(f"estimated cache savings:    {money(second.cost['estimated_cache_savings'])}\n")

        assert mock.chat_calls == 0, "second run must issue zero inference calls"
        assert second.cost["fresh_cost_this_run"] == 0.0
        assert abs(second.cost["estimated_cache_savings"] - billed) < 1e-9
        assert (
            second.metrics["summarization"]["per_language"]["en"]["rougeLsum_f"]
            == first.metrics["summarization"]["per_language"]["en"]["rougeLsum_f"]
        )

        # ---------------------------------------------------------------- #
        print("=== reports ===")
        summary_path = write_model_summary(cfg, second)
        bundle = generate_reports(cfg)
        for label, path in {"summary": summary_path, **bundle.paths}.items():
            print(f"{label:>10}: {path} ({path.stat().st_size:,} bytes)")

        assert abs(bundle.rows[0]["openrouter_cost"] - billed) < 1e-9
        assert bundle.paths["result_md"].is_file()
        assert bundle.paths["excel"].is_file()

        if args.out:
            out = args.out
            if out.exists():
                shutil.rmtree(out)
            shutil.copytree(cfg.results_dir, out)
            # RESULT.md lives at the workspace root, not under results/.
            shutil.copy2(bundle.paths["result_md"], out / "RESULT.md")
            (out / "README.txt").write_text(
                "These files were produced by scripts/mock_end_to_end.py against an\n"
                "in-process mock of the OpenRouter API. No model was called and nothing\n"
                "was billed. The scores here describe the mock's canned behaviour, NOT\n"
                f"the behaviour of {model.model_id}.\n",
                encoding="utf-8",
            )
            print(f"\ncopied sample reports to {out}")

        print("\nAll end-to-end assertions passed.")
        return 0
    finally:
        if args.keep:
            print(f"\nworkspace kept at {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
