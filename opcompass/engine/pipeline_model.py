"""Pipeline model — stage-level latency and throughput analysis.

Used by the ``pipeline`` analysis mode when an operator provides a
:meth:`~opcompass.operators.base.Operator.get_ops_breakdown` decomposition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opcompass.models import PipelineStage, SubOp
    from opcompass.hardware.base import Hardware


def analyze_pipeline(
    sub_ops: list[SubOp],
    hardware: Hardware,
    grid_size: int = 1,
) -> dict[str, float]:
    """Estimate per-stage time via simplified pipeline scheduling.

    Args:
        sub_ops: Ordered sub-operations from the operator.
        hardware: Target hardware (provides pipeline stages).
        grid_size: Number of thread-block invocations (for wave quantization).

    Returns:
        Dict mapping stage name → estimated time in seconds.
    """
    stages: list[PipelineStage] = hardware.compute_unit.pipeline

    if not stages:
        # Fall back to a flat assignment
        return _fallback_pipeline(sub_ops, hardware)

    # Build a map from stage name → PipelineStage for quick lookup
    stage_map: dict[str, PipelineStage] = {s.name: s for s in stages}

    times: dict[str, float] = {}
    clock_s = 1.0 / (hardware.compute_unit.clock_mhz * 1e6) if hardware.compute_unit.clock_mhz > 0 else 0.0

    for sub in sub_ops:
        # Try to match sub-op to a pipeline stage
        matched = _match_stage(sub, stage_map)
        if matched is None:
            continue

        # Time = latency + (work / throughput)   [in cycles]
        cycles = matched.latency_cycles
        if matched.throughput_per_cycle > 0:
            # Determine work units based on stage type
            work = _work_for_stage(sub, matched)
            cycles += work / matched.throughput_per_cycle

        # Wave quantization: how many waves of blocks?
        # Each SM / CU can process multiple blocks concurrently
        waves = max(1, grid_size / hardware.num_compute_units)
        total_cycles = cycles * waves

        times[matched.name] = total_cycles * clock_s

    return times


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_stage(sub: SubOp, stage_map: dict[str, PipelineStage]) -> PipelineStage | None:
    """Heuristic: match a SubOp name to a PipelineStage."""
    name_lower = sub.name.lower()
    for key, stage in stage_map.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return stage
    # Fallback: return the first stage whose name shares a word
    for key, stage in stage_map.items():
        if any(word in name_lower for word in key.lower().split("_")):
            return stage
    return None


def _work_for_stage(sub: SubOp, stage: PipelineStage) -> float:
    """Determine the amount of 'work units' for a stage given a sub-op."""
    sn = stage.name.lower()
    if "read" in sn or "load" in sn or "write" in sn or "store" in sn:
        return float(sub.read_bytes + sub.write_bytes)
    if "compute" in sn or "mma" in sn or "alu" in sn:
        return float(sub.flops)
    return float(sub.flops + sub.read_bytes)


def _fallback_pipeline(
    sub_ops: list[SubOp], hardware: Hardware
) -> dict[str, float]:
    """When no pipeline stages are defined, estimate from sub-ops directly."""
    times: dict[str, float] = {}
    peak = hardware.get_peak_flops(hardware.compute_unit.peak_flops.__iter__().__next__()[0])  # first available

    for sub in sub_ops:
        mem_time = 0.0
        total_bytes = sub.read_bytes + sub.write_bytes
        if total_bytes > 0:
            mem_time = total_bytes / hardware.hbm_bandwidth

        comp_time = 0.0
        if sub.flops > 0 and peak > 0:
            comp_time = sub.flops / peak

        times[sub.name] = max(mem_time, comp_time)  # assume overlap within sub-op

    return times
