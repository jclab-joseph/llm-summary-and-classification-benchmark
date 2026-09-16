"""Canonical JSON + SHA-256 helpers.

Every cache key in this project is `sha256(canonical_json(payload))`. Canonical
JSON here means: sorted keys, no insignificant whitespace, UTF-8 preserved
(``ensure_ascii=False``), and a deterministic rendering of floats so that
``1.0`` and ``1`` never hash differently.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from typing import Any

__all__ = [
    "canonical_json",
    "canonical_obj",
    "sha256_hex",
    "hash_obj",
    "hash_text",
    "normalize_text",
    "short_hash",
]


def _canonical_scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"non-finite float is not canonicalizable: {value!r}")
        if value == int(value) and abs(value) < 2**53:
            return int(value)
        # 12 significant digits is well beyond anything we key on and removes
        # float repr drift between platforms.
        return float(f"{value:.12g}")
    raise TypeError(f"cannot canonicalize {type(value).__name__}: {value!r}")


def canonical_obj(obj: Any) -> Any:
    """Recursively normalize ``obj`` into JSON-native, order-stable primitives."""
    if isinstance(obj, dict):
        return {str(k): canonical_obj(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [canonical_obj(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [canonical_obj(v) for v in sorted(obj, key=repr)]
    return _canonical_scalar(obj)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON rendering used as hash input."""
    return json.dumps(
        canonical_obj(obj),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any) -> str:
    """SHA-256 of the canonical JSON form of ``obj``."""
    return sha256_hex(canonical_json(obj))


def normalize_text(text: str) -> str:
    """NFC + whitespace normalization, applied before hashing any dataset text."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(lines).strip()


def hash_text(text: str) -> str:
    """Stable content hash of a piece of dataset text."""
    return sha256_hex(normalize_text(text))


def short_hash(value: str, length: int = 12) -> str:
    return value[:length]
