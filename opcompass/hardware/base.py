"""Abstract base class for all hardware targets."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opcompass.models import (
        ComputeUnit,
        DataType,
        MemoryHierarchy,
    )


class Hardware(ABC):
    """Abstract hardware target.  Each concrete hardware lives in its own
    file under ``hardware/``.

    Subclasses must set the following class-level attributes:

        name: str           — Short id, e.g. "a100"
        vendor: str         — "NVIDIA", "AMD", ...
        description: str    — Human-readable summary

    And must override:

        memory: MemoryHierarchy
        compute_unit: ComputeUnit
    """

    name: str = ""
    vendor: str = ""
    description: str = ""

    # Subclasses override these with actual objects
    memory: MemoryHierarchy
    compute_unit: ComputeUnit

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_peak_flops(self, dtype: DataType) -> float:
        """Peak FLOPS for *dtype* on the full chip."""
        return self.compute_unit.peak_flops.get(dtype, 0.0)

    def get_bandwidth(self, tier_name: str) -> float:
        """Bandwidth in bytes/sec for the named memory tier."""
        for t in self.memory.tiers:
            if t.name.lower() == tier_name.lower():
                return t.bandwidth_bytes_per_sec
        return 0.0

    @property
    def hbm_bandwidth(self) -> float:
        """Convenience: bandwidth of the first (slowest) memory tier."""
        if self.memory.tiers:
            return self.memory.tiers[0].bandwidth_bytes_per_sec
        return 0.0

    @property
    def clock_ghz(self) -> float:
        """Clock frequency in GHz."""
        return self.compute_unit.clock_mhz / 1000.0

    @property
    def num_compute_units(self) -> int:
        """Number of compute units (SMs / CUs)."""
        return self.compute_unit.count
