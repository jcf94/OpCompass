"""Compute model — peak throughput and utilization estimation.

Models the theoretical minimum compute time given hardware peak FLOPS
and an optional utilization factor (accounting for occupancy, instruction
mix, pipeline bubbles, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opcompass.models import DataType
    from opcompass.hardware.base import Hardware


def peak_compute_time(
    flops: int,
    hardware: Hardware,
    dtype: DataType,
    utilization: float = 1.0,
) -> float:
    """Compute theoretical minimum time for *flops* operations.

    Args:
        flops: Total floating-point operations.
        hardware: The target hardware.
        dtype: Numerical data type (determines which tensor-core / SIMD rate applies).
        utilization: Efficiency factor (0–1).  1.0 = peak, no overheads.

    Returns:
        Time in seconds.
    """
    if flops <= 0:
        return 0.0

    peak = hardware.get_peak_flops(dtype)
    if peak <= 0:
        return float("inf")

    return flops / (peak * utilization)


def effective_peak_flops(
    hardware: Hardware,
    dtype: DataType,
    utilization: float = 1.0,
) -> float:
    """Return the effective peak FLOPS after applying *utilization*."""
    return hardware.get_peak_flops(dtype) * utilization
