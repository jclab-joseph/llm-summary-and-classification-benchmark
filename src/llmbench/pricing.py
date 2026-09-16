"""Pricing resolution: estimates before the run, `usage.cost` after it.

Two clearly separated notions:

* **Estimated price** -- from OpenRouter's /models metadata when reachable,
  otherwise from `config/pricing.yaml`. Used only for budget estimation and for
  validating what we were billed.
* **Actual cost** -- `usage.cost` on each OpenRouter response. Authoritative.
  When the two disagree, the reports show `usage.cost`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable

from llmbench.config.loader import AppConfig
from llmbench.core.reproducibility import utc_now_iso
from llmbench.db.store import Store
from llmbench.openrouter.client import OpenRouterClient
from llmbench.openrouter.types import ModelPricing

__all__ = ["ResolvedPricing", "PricingService", "estimate_cost_usd"]


@dataclass(slots=True)
class ResolvedPricing:
    model_id: str
    input_per_million: float | None
    output_per_million: float | None
    cached_input_per_million: float | None
    retrieved_at: str
    source: str  # openrouter | snapshot | config | unknown

    @property
    def complete(self) -> bool:
        return self.input_per_million is not None and self.output_per_million is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "input_per_million": self.input_per_million,
            "output_per_million": self.output_per_million,
            "cached_input_per_million": self.cached_input_per_million,
            "retrieved_at": self.retrieved_at,
            "source": self.source,
        }


def estimate_cost_usd(
    pricing: ResolvedPricing,
    *,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> dict[str, float]:
    """Pricing-table cost estimate. Never used as the reported actual cost."""
    inp = pricing.input_per_million or 0.0
    out = pricing.output_per_million or 0.0
    cached_rate = pricing.cached_input_per_million
    fresh_input = max(0, input_tokens - cached_input_tokens)
    input_cost = fresh_input / 1_000_000 * inp
    if cached_input_tokens:
        input_cost += cached_input_tokens / 1_000_000 * (cached_rate if cached_rate is not None else inp)
    output_cost = output_tokens / 1_000_000 * out
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


class PricingService:
    def __init__(self, cfg: AppConfig, store: Store | None = None) -> None:
        self.cfg = cfg
        self.store = store
        # Set when a live refresh failed and config values were used instead.
        self.last_refresh_error: str | None = None
        # Raw /models metadata from the most recent refresh: pricing *and* the
        # supported-parameter list the pre-run capability check needs.
        self.last_metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    def from_config(self, model_id: str) -> ResolvedPricing:
        entry = self.cfg.pricing.get(model_id)
        return ResolvedPricing(
            model_id=model_id,
            input_per_million=entry.input_per_million,
            output_per_million=entry.output_per_million,
            cached_input_per_million=entry.cached_input_per_million,
            retrieved_at="config",
            source="config" if entry.complete else "unknown",
        )

    def from_snapshot(self, model_id: str) -> ResolvedPricing | None:
        if self.store is None:
            return None
        row = self.store.latest_pricing(model_id)
        if row is None:
            return None
        return ResolvedPricing(
            model_id=model_id,
            input_per_million=row.input_per_million,
            output_per_million=row.output_per_million,
            cached_input_per_million=row.cached_input_per_million,
            retrieved_at=row.retrieved_at.isoformat() if row.retrieved_at else "",
            source="snapshot",
        )

    def resolve(self, model_id: str, *, prefer_snapshot: bool = True) -> ResolvedPricing:
        """Snapshot (from a previous `pricing`/`run`) first, config as fallback."""
        if prefer_snapshot:
            snapshot = self.from_snapshot(model_id)
            if snapshot is not None and snapshot.complete:
                return snapshot
        config = self.from_config(model_id)
        if config.complete:
            return config
        snapshot = self.from_snapshot(model_id)
        if snapshot is not None:
            return snapshot
        return config

    # ------------------------------------------------------------------ #
    async def refresh(
        self,
        client: OpenRouterClient,
        model_ids: Iterable[str],
        *,
        persist: bool = True,
    ) -> dict[str, ResolvedPricing]:
        """Fetch live pricing metadata; falls back silently to config on failure.

        Allowed during a dry run: this is metadata, not inference.
        """
        ids = list(model_ids)
        self.last_refresh_error = None
        self.last_metadata = {}
        try:
            metadata = await client.get_model_metadata(ids)
        except Exception as exc:
            # Falling back to config is the documented behaviour, but a silent
            # fallback would hide a broken endpoint, so the reason is recorded.
            self.last_refresh_error = f"{type(exc).__name__}: {exc}"
            return {mid: self.resolve(mid) for mid in ids}

        self.last_metadata = metadata
        live: dict[str, ModelPricing] = {k: v["pricing"] for k, v in metadata.items()}

        out: dict[str, ResolvedPricing] = {}
        for model_id in ids:
            entry = live.get(model_id)
            if entry is None:
                out[model_id] = self.resolve(model_id)
                continue
            resolved = ResolvedPricing(
                model_id=model_id,
                input_per_million=entry.input_per_million,
                output_per_million=entry.output_per_million,
                cached_input_per_million=entry.cached_input_per_million,
                retrieved_at=entry.retrieved_at,
                source="openrouter",
            )
            out[model_id] = resolved
            if persist and self.store is not None:
                self.store.save_pricing_snapshot(
                    {
                        "model_id": model_id,
                        "input_per_million": entry.input_per_million,
                        "output_per_million": entry.output_per_million,
                        "cached_input_per_million": entry.cached_input_per_million,
                        "retrieved_at": datetime.now(UTC),
                        "source": "openrouter",
                        "raw_json": entry.raw,
                    }
                )
        return out

    # ------------------------------------------------------------------ #
    def compare(self, live: dict[str, ResolvedPricing]) -> list[dict[str, Any]]:
        """Diff live OpenRouter pricing against `config/pricing.yaml`."""
        rows = []
        for model_id, resolved in sorted(live.items()):
            configured = self.from_config(model_id)
            def delta(a: float | None, b: float | None) -> float | None:
                if a is None or b is None:
                    return None
                return a - b

            rows.append(
                {
                    "model_id": model_id,
                    "config_input": configured.input_per_million,
                    "live_input": resolved.input_per_million,
                    "input_delta": delta(resolved.input_per_million, configured.input_per_million),
                    "config_output": configured.output_per_million,
                    "live_output": resolved.output_per_million,
                    "output_delta": delta(resolved.output_per_million, configured.output_per_million),
                    "live_cached_input": resolved.cached_input_per_million,
                    "source": resolved.source,
                    "retrieved_at": resolved.retrieved_at or utc_now_iso(),
                    "within_constraints": (
                        None
                        if not resolved.complete
                        else (
                            resolved.input_per_million <= self.cfg.pricing.constraints.max_input_per_million
                            and resolved.output_per_million
                            <= self.cfg.pricing.constraints.max_output_per_million
                        )
                    ),
                }
            )
        return rows

    def reconcile(
        self,
        pricing: ResolvedPricing,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        actual_cost: float,
    ) -> dict[str, Any]:
        """Compare the pricing-table recomputation against the billed `usage.cost`."""
        estimated = estimate_cost_usd(
            pricing,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cached_input_tokens=cached_tokens,
        )["total_cost"]
        tolerance = self.cfg.benchmark.cost
        difference = actual_cost - estimated
        relative = abs(difference) / actual_cost if actual_cost else (0.0 if not difference else 1.0)
        matches = (
            abs(difference) <= tolerance.reconciliation_tolerance_usd
            or relative <= tolerance.reconciliation_relative_tolerance
        )
        return {
            "recomputed_from_pricing_table": estimated,
            "actual_usage_cost": actual_cost,
            "difference": difference,
            "relative_difference": relative,
            "matches_within_tolerance": matches,
            "authoritative": "openrouter usage.cost",
        }
