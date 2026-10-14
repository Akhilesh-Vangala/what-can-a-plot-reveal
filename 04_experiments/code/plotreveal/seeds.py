"""Deterministic seeds: every model, plot and lineup gets a seed derived from its identity (gap 2.7)."""
import hashlib

import numpy as np


def seed_for(*parts) -> int:
    """Stable 32-bit seed from any identifying parts, e.g. seed_for("main", "healthy", 17)."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def rng_for(*parts) -> np.random.Generator:
    return np.random.default_rng(seed_for(*parts))
