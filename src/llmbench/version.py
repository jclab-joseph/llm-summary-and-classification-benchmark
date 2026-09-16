"""Single source of truth for the package / benchmark version."""

from __future__ import annotations

__version__ = "1.0.0"

# The benchmark version is recorded on every inference cache key. Bumping it
# invalidates *all* cached inference on purpose; do that only when the benchmark
# semantics change in a way the finer-grained keys cannot express.
BENCHMARK_VERSION = "1.0.0"
