"""Small score helpers for aggregation modules."""

from __future__ import annotations

import math


def clamp_score(value: float) -> float:
    """Clamp finite numeric values to the inclusive range [0, 1]."""

    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, float(value)))
