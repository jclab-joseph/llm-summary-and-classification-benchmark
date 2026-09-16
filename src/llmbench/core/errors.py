"""Project-level exception types."""

from __future__ import annotations


class BenchmarkError(Exception):
    """Base class for every error this project raises on purpose."""


class ConfigError(BenchmarkError):
    pass


class ManifestError(BenchmarkError):
    pass


class DatasetError(BenchmarkError):
    pass


class BudgetExceeded(BenchmarkError):
    """Raised when the projected or actual spend crosses the configured budget."""


class OpenRouterError(BenchmarkError):
    """Any non-retryable failure returned by OpenRouter."""

    def __init__(self, message: str, *, status_code: int | None = None, body: object = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class RateLimited(OpenRouterError):
    """429 / upstream rate limit. Retryable."""


class DryRunViolation(BenchmarkError):
    """Raised if anything tries to hit the inference API during a dry run."""
