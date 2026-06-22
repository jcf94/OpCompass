"""Memory hierarchy modelling — multi-tier data movement analysis.

This module estimates the minimum time required to move data through
a multi-level memory hierarchy, accounting for data reuse and cache
hit rates where possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opcompass.models import MemoryHierarchy


def estimate_hierarchy_time(
    byte_count: int,
    hierarchy: MemoryHierarchy,
    l2_hit_rate: float = 0.0,
    smem_hit_rate: float = 0.0,
) -> dict[str, float]:
    """Break down data-movement time across memory tiers.

    Args:
        byte_count: Total bytes that must move through the slowest tier.
        hierarchy: The memory hierarchy definition.
        l2_hit_rate: Fraction of L2 requests that hit (0–1).
        smem_hit_rate: Fraction of shared-memory requests that hit (0–1).

    Returns:
        Dict mapping tier name → minimum transfer time in seconds.
    """
    times: dict[str, float] = {}
    remaining = byte_count

    for tier in hierarchy.tiers:
        if remaining <= 0:
            break

        # How much data actually traverses this tier?
        # For the slowest tier (HBM) we assume all data crosses.
        # Faster tiers only see the miss traffic from the tier above.
        times[tier.name] = tier.transfer_time(remaining)

        # Crude model: the next tier only sees a fraction of the data
        # (approximation — real reuse patterns are operator-specific).
        if tier.name.lower() == "hbm" or tier.name.lower().endswith("hbm"):
            remaining *= (1.0 - l2_hit_rate)
        elif "l2" in tier.name.lower():
            remaining *= (1.0 - smem_hit_rate)

    return times


def min_memory_time(
    byte_count: int,
    hierarchy: MemoryHierarchy,
) -> float:
    """Return the single-tier bottleneck time for a given byte count.

    This is the simplest useful estimate: time = bytes / slowest_bw.
    For a full multi-tier breakdown use :func:`estimate_hierarchy_time`.
    """
    if not hierarchy.tiers:
        return 0.0
    # Slowest tier = first in the list
    return hierarchy.tiers[0].transfer_time(byte_count)
