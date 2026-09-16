"""Logging configuration.

Two sinks with different jobs:

* **console** -- quiet by default (warnings and above), so a normal run shows the
  progress bar and the result tables, not a wall of text. `-v` turns on INFO,
  `-vv` DEBUG.
* **file** -- always DEBUG, always on. When 1,300 requests fail you want the
  exact error, the model, the sample id and the HTTP status afterwards, without
  having reproduced the run with a flag set.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from rich.logging import RichHandler

__all__ = ["configure_logging", "get_logger", "default_log_path", "prune_logs", "LOGGER_NAME"]

LOGGER_NAME = "llmbench"
_FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
_configured_path: Path | None = None


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")


def default_log_path(log_dir: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return log_dir / f"benchmark-{stamp}.log"


def configure_logging(
    *,
    log_dir: Path | None = None,
    log_file: Path | None = None,
    verbosity: int = 0,
    console: bool = True,
) -> Path | None:
    """Install the console and file handlers. Returns the log file path, if any."""
    global _configured_path

    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(logging.DEBUG)
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    if console:
        console_level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
        rich_handler = RichHandler(
            level=console_level,
            show_time=False,
            show_path=False,
            rich_tracebacks=True,
            markup=False,
        )
        rich_handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(rich_handler)

    path: Path | None = None
    if log_file is not None:
        path = Path(log_file)
    elif log_dir is not None:
        path = default_log_path(Path(log_dir))

    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
            root.addHandler(file_handler)
        except OSError:
            # A broken log destination must never take the benchmark down.
            path = None

    _configured_path = path
    return path


def configured_log_path() -> Path | None:
    return _configured_path


def prune_logs(log_dir: Path, keep_last: int) -> int:
    """Delete all but the newest ``keep_last`` log files. Returns how many went."""
    if keep_last <= 0 or not log_dir.is_dir():
        return 0
    files = sorted(log_dir.glob("benchmark-*.log"), key=lambda p: p.name, reverse=True)
    removed = 0
    for path in files[keep_last:]:
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed
