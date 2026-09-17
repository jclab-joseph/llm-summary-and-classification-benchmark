"""Local (non-OpenRouter) inference backends.

This is a deliberately separate track. OpenRouter remains the only *API* provider
in the project -- `OpenRouterClient` is still the only thing that talks to a model
API -- and a local engine is a different execution backend, not a second vendor
integration. Local results are never billed, so their cost columns are empty
rather than zero, and the reports mark them so nobody reads "no cost" as "cheaper
than the hosted models".
"""

from llmbench.local.engine import LocalEngine, LocalResult, build_engine
from llmbench.local.client import LocalEngineClient

__all__ = ["LocalEngine", "LocalResult", "build_engine", "LocalEngineClient"]
