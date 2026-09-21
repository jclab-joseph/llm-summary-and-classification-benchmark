"""Weight resolution for local models."""

from __future__ import annotations

from pathlib import Path

from llmbench.config.loader import AppConfig
from llmbench.config.schema import LocalModelConfig
from llmbench.datasets.hf import download_hf_file
from llmbench.logging_setup import get_logger

__all__ = ["local_model_path", "ensure_local_model", "needs_single_file"]

log = get_logger("local.download")


def local_model_path(cfg: AppConfig, model: LocalModelConfig) -> Path:
    """Where the weights live. The revision is in the filename, like datasets."""
    safe_repo = model.source.repo.replace("/", "__")
    return cfg.local_model_dir / f"{safe_repo}@{model.source.revision[:12]}__{model.source.filename}"


def needs_single_file(model: LocalModelConfig) -> bool:
    """Whether the benchmark fetches one file, or the engine fetches the repo."""
    return bool(model.source.filename)


def ensure_local_model(cfg: AppConfig, model: LocalModelConfig, *, progress=None) -> Path:
    """Download the weights once, pinned to the configured revision.

    Engines that load a whole repository (a transformers checkpoint rather than a
    single GGUF) fetch it themselves; for those the repo id is handed through in
    place of a path.
    """
    if not needs_single_file(model):
        return Path(model.source.repo)

    path = local_model_path(cfg, model)
    if path.is_file() and path.stat().st_size > 0:
        return path
    log.info("downloading %s/%s", model.source.repo, model.source.filename)
    return download_hf_file(
        model.source.repo,
        model.source.revision,
        model.source.filename,
        cfg.local_model_dir,
        repo_type="models",
        progress=progress,
    )
