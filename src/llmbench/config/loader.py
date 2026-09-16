"""Loading and merging of the YAML configuration tree."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

from llmbench.config.schema import (
    BenchmarkConfig,
    JudgeConfig,
    ModelConfig,
    ModelsConfig,
    PricingConfig,
)
from llmbench.core.errors import ConfigError

__all__ = ["AppConfig", "load_config", "resolve_config_dir", "project_root"]


def project_root() -> Path:
    """Repository root (the directory that owns `config/` and `data/`)."""
    env = os.environ.get("LLMBENCH_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def resolve_config_dir(explicit: str | Path | None = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    elif os.environ.get("LLMBENCH_CONFIG_DIR"):
        path = Path(os.environ["LLMBENCH_CONFIG_DIR"]).expanduser().resolve()
    else:
        path = project_root() / "config"
    if not path.is_dir():
        raise ConfigError(f"config directory not found: {path}")
    return path


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing configuration file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into a copy of ``base`` (dicts merged, scalars replaced)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _expand_models(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply `defaults:` to every model/judge entry."""
    defaults = raw.get("defaults") or {}
    out: dict[str, Any] = {"version": raw.get("version", 1)}
    for key, role_default in (("models", "target"), ("judges", None)):
        entries = raw.get(key) or []
        expanded = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ConfigError(f"{key} entries must be mappings, got {type(entry).__name__}")
            merged = _deep_merge(defaults, entry)
            if role_default and "role" not in merged:
                merged["role"] = role_default
            expanded.append(merged)
        out[key] = expanded
    return out


@dataclass(frozen=True)
class AppConfig:
    """Everything loaded from `config/`, plus resolved absolute paths."""

    config_dir: Path
    root: Path
    benchmark: BenchmarkConfig
    models: ModelsConfig
    pricing: PricingConfig
    judge: JudgeConfig

    # --- resolved paths --------------------------------------------------- #
    @cached_property
    def manifest_dir(self) -> Path:
        return self._resolve(self.benchmark.paths.manifests)

    @cached_property
    def cache_db_path(self) -> Path:
        return self._resolve(self.benchmark.paths.cache_db)

    @cached_property
    def raw_data_dir(self) -> Path:
        return self._resolve(self.benchmark.paths.raw_data)

    @cached_property
    def results_dir(self) -> Path:
        return self._resolve(self.benchmark.paths.results)

    @cached_property
    def log_dir(self) -> Path:
        return self._resolve(self.benchmark.logging.directory)

    @cached_property
    def result_markdown_path(self) -> Path:
        return self._resolve(self.benchmark.paths.result_markdown)

    @cached_property
    def result_excel_path(self) -> Path:
        return self._resolve(self.benchmark.paths.result_excel)

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.root / path).resolve()

    # --- lookups ---------------------------------------------------------- #
    def require_model(self, model_id: str) -> ModelConfig:
        model = self.models.by_id(model_id)
        if model is None:
            known = ", ".join(m.model_id for m in self.models.models)
            raise ConfigError(f"unknown model '{model_id}'. Configured models: {known}")
        return model

    def judge_model(self, profile: str) -> ModelConfig:
        prof = self.judge.judge.profiles.get(profile)
        if prof is None:
            raise ConfigError(
                f"unknown judge profile '{profile}'. Available: {sorted(self.judge.judge.profiles)}"
            )
        model = self.models.by_id(prof.model_id)
        if model is None:
            raise ConfigError(
                f"judge profile '{profile}' points at '{prof.model_id}', which is not in models.yaml"
            )
        return model

    def ensure_dirs(self) -> None:
        for path in (
            self.manifest_dir,
            self.cache_db_path.parent,
            self.raw_data_dir,
            self.results_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_config(config_dir: str | Path | None = None, *, root: str | Path | None = None) -> AppConfig:
    cfg_dir = resolve_config_dir(config_dir)
    base_root = Path(root).expanduser().resolve() if root else project_root()

    benchmark = BenchmarkConfig.model_validate(_read_yaml(cfg_dir / "benchmark.yaml"))
    models = ModelsConfig.model_validate(_expand_models(_read_yaml(cfg_dir / "models.yaml")))
    pricing = PricingConfig.model_validate(_read_yaml(cfg_dir / "pricing.yaml"))
    judge = JudgeConfig.model_validate(_read_yaml(cfg_dir / "judge.yaml"))

    return AppConfig(
        config_dir=cfg_dir,
        root=base_root,
        benchmark=benchmark,
        models=models,
        pricing=pricing,
        judge=judge,
    )
