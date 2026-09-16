"""Capture of everything needed to reproduce a benchmark run."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from llmbench.version import BENCHMARK_VERSION, __version__

__all__ = ["git_commit", "package_versions", "environment_snapshot", "utc_now_iso"]

_TRACKED_PACKAGES = (
    "httpx",
    "pydantic",
    "sqlalchemy",
    "pandas",
    "numpy",
    "scikit-learn",
    "sacrebleu",
    "rouge-score",
    "bert-score",
    "torch",
    "transformers",
    "typer",
    "rich",
    "tenacity",
    "pyarrow",
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_commit(repo_root: Path | None = None) -> str | None:
    """Best-effort git commit of the benchmark code itself."""
    root = repo_root or Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_dirty(repo_root: Path | None = None) -> bool | None:
    root = repo_root or Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def environment_snapshot() -> dict[str, Any]:
    """Everything recorded alongside a run for reproducibility (section 34)."""
    return {
        "benchmark_package_version": __version__,
        "benchmark_version": BENCHMARK_VERSION,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "captured_at": utc_now_iso(),
    }
