"""Uniform negative sampling without silently returning known positives."""
from collections.abc import Sequence
from random import Random


def sample_negative(rng: Random, catalog: Sequence[int], positives: set[int]) -> int:
    for _ in range(20):
        candidate = rng.choice(catalog)
        if candidate not in positives:
            return candidate
    eligible = [item for item in catalog if item not in positives]
    if not eligible:
        raise ValueError("No negative item exists for this user in the catalog.")
    return rng.choice(eligible)
