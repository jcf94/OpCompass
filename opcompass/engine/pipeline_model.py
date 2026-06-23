"""Pipeline model — DAG-based cycle-level scheduling for block-level analysis.

Produces a ``PipelineSchedule`` that maps each SubOp to a start/end cycle
on the timeline, respecting DAG dependencies and hardware overlap rules
(async copy, sparsity, can_overlap_with_compute).

This replaces the old ``analyze_pipeline()`` heuristic matching with
explicit SubOp→PipelineStage mapping and proper overlap modeling.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opcompass.models import PipelineConfig, PipelineSchedule, SubOp, TilingInfo
    from opcompass.hardware.base import Hardware


def schedule_pipeline(
    sub_ops: list[SubOp],
    hardware: Hardware,
    pipeline_config: PipelineConfig,
    tiling: TilingInfo,
    **dims,
) -> PipelineSchedule:
    """Schedule SubOps on a cycle-level timeline respecting DAG and overlap.

    Models the CUTLASS software-pipelined K-loop with three phases:
    - **Prologue** (k=0): all sub-ops sequential, no overlap possible
    - **Steady state** (k≥1): async_copy_load can overlap with previous MMA
    - **Epilogue**: final shared_store_C + global_write_C after last MMA

    Args:
        sub_ops: Operator decomposition (recurring + epilogue SubOps).
        hardware: Target hardware (provides pipeline stages, overlap info).
        pipeline_config: Feature toggles (async copy, sparsity).
        tiling: Block sizes and grid layout info.
        **dims: Problem dimensions (M, N, K for grid size).

    Returns:
        A ``PipelineSchedule`` with per-sub-op cycle-level placement.
    """
    from opcompass.models import PipelineSchedule

    cu = hardware.compute_unit
    overlap_tiers = hardware.memory.can_overlap_with_compute

    # ── Build stage map ────────────────────────────────────────────────
    stage_map: dict[str, object] = {s.name: s for s in cu.pipeline}
    dtype = dims.get("dtype")

    # ── Separate recurring vs epilogue sub-ops ─────────────────────────
    recurring = [s for s in sub_ops if s.is_recurring]
    epilogue = [s for s in sub_ops if not s.is_recurring]

    K = dims.get("K", 0)
    bK = tiling.block_k
    num_k_iterations = max(1, math.ceil(K / bK)) if K > 0 else 1

    # ── Compute per-sub-op duration in cycles ──────────────────────────
    # Two variants are needed for proper pipeline modeling:
    #   - _full_duration: latency + throughput work. Used for the prologue
    #     (pipeline fill) and epilogue, where latency is a real cost since
    #     nothing is overlapped.
    #   - _throughput_duration: throughput work only (no latency). Used for
    #     steady-state iterations, where latency is hidden by the pipeline.
    #     Adding latency per-iteration was the main source of error: with
    #     async_copy_load latency=300 and 128 K-iterations, the old code
    #     charged 300*2*128 = 76,800 cycles of latency alone.
    def _work_units(sub: SubOp, stage) -> float:
        """Compute the work units (bytes or FMA) for a sub-op on a stage."""
        stage_name_lower = stage.name.lower()
        if any(kw in stage_name_lower for kw in ("read", "load", "write", "store", "copy")):
            return sub.read_bytes + sub.write_bytes
        elif any(kw in stage_name_lower for kw in ("mma", "alu", "compute")):
            # MMA throughput is in FMA/cycle; flops counts multiply+add as 2 ops
            # So work = flops / 2 to get FMA count
            return sub.flops / 2 if sub.flops > 0 else 0
        return sub.read_bytes + sub.write_bytes + sub.flops

    def _stage_throughput(sub: SubOp, stage) -> float:
        """Return effective per-SM throughput for this sub-op and dtype."""
        throughput = stage.throughput_per_cycle
        stage_name = stage.name.lower()

        if any(kw in stage_name for kw in ("global", "async_copy")):
            if hardware.hbm_bandwidth > 0 and cu.clock_mhz > 0 and cu.count > 0:
                hbm_bytes_per_cycle_per_sm = (
                    hardware.hbm_bandwidth / (cu.clock_mhz * 1e6) / cu.count
                )
                throughput = min(throughput, hbm_bytes_per_cycle_per_sm)

        if any(kw in stage_name for kw in ("mma", "alu", "compute")):
            peak = hardware.get_peak_flops(dtype) if dtype is not None else 0.0
            if peak > 0 and cu.clock_mhz > 0 and cu.count > 0:
                throughput = peak / (2 * cu.clock_mhz * 1e6 * cu.count)

        if pipeline_config.sparsity_2_4_enabled and "mma" in stage_name:
            throughput *= 2

        return throughput

    def _throughput_duration(sub: SubOp) -> int:
        """Throughput-only duration (no latency) — for steady-state iterations."""
        stage = stage_map.get(sub.pipeline_stage)
        if stage is None:
            total_work = sub.read_bytes + sub.write_bytes
            if total_work > 0:
                return math.ceil(total_work / (hardware.hbm_bandwidth / (cu.clock_mhz * 1e6)))
            if sub.flops > 0:
                peak = hardware.get_peak_flops(dims.get("dtype"))
                return math.ceil(sub.flops / (peak / (cu.clock_mhz * 1e6)))
            return 0

        throughput = _stage_throughput(sub, stage)
        work = _work_units(sub, stage)
        if throughput <= 0 or work <= 0:
            return 0
        return math.ceil(work / throughput)

    def _full_duration(sub: SubOp) -> int:
        """Full duration: latency + throughput — for prologue/epilogue."""
        stage = stage_map.get(sub.pipeline_stage)
        if stage is None:
            return _throughput_duration(sub)
        return stage.latency_cycles + _throughput_duration(sub)

    # ── Determine if overlap is possible ───────────────────────────────
    # Overlap is possible when:
    # 1. async_copy_enabled is True (uses async_copy_load stage)
    # 2. The HBM tier is in can_overlap_with_compute
    # 3. The sub-op's pipeline_stage maps to a memory tier in overlap_tiers
    async_can_overlap = (
        pipeline_config.async_copy_enabled
        and bool(overlap_tiers)
        and any("async_copy_load" in s.name or "global_read" in s.name for s in cu.pipeline)
    )

    # ── Re-schedule using the SOL iteration formula ────────────────────
    # This produces a cleaner, more accurate timeline.
    scheduled = _reschedule_solid(
        recurring, epilogue, num_k_iterations,
        _full_duration, _throughput_duration,
        async_can_overlap, cu, tiling, dims,
    )

    # ── Compute totals ─────────────────────────────────────────────────
    if not scheduled:
        total_cycles = 0
    else:
        total_cycles = max(s.end_cycle for s in scheduled)

    M = dims.get("M", 0)
    N = dims.get("N", 0)
    grid_size = max(1, math.ceil(M / tiling.block_m)) * max(1, math.ceil(N / tiling.block_n))

    threads_per_block = max(1, tiling.num_warps_per_block * cu.threads_per_warp)
    blocks_by_threads = (
        cu.max_threads_per_unit // threads_per_block
        if cu.max_threads_per_unit > 0 else cu.max_thread_blocks_per_unit
    )
    blocks_by_warps = (
        cu.max_concurrent_warps // tiling.num_warps_per_block
        if cu.max_concurrent_warps > 0 and tiling.num_warps_per_block > 0
        else cu.max_thread_blocks_per_unit
    )
    blocks_by_smem = (
        (cu.shared_memory_max_kb * 1024) // tiling.shared_memory_per_block
        if cu.shared_memory_max_kb > 0 and tiling.shared_memory_per_block > 0
        else cu.max_thread_blocks_per_unit
    )
    resident_blocks_per_sm = max(
        1,
        min(
            x for x in (
                cu.max_thread_blocks_per_unit,
                blocks_by_threads,
                blocks_by_warps,
                blocks_by_smem,
            )
            if x and x > 0
        ),
    )
    wave_count = max(1, math.ceil(grid_size / (cu.count * resident_blocks_per_sm)))

    clock_s = 1.0 / (cu.clock_mhz * 1e6) if cu.clock_mhz > 0 else 0.0
    total_time_s = total_cycles * clock_s * wave_count

    # Identify bottleneck stage
    stage_times: dict[str, int] = {}
    for s in scheduled:
        stage_times[s.pipeline_stage] = stage_times.get(s.pipeline_stage, 0) + s.duration_cycles
    bottleneck = max(stage_times, key=stage_times.get) if stage_times else ""

    # Compute per-iteration, prologue, and epilogue cycles
    # Use full durations (with latency) for prologue/epilogue since these
    # phases have no overlap to hide latency behind. Use throughput-only
    # durations for steady-state since the pipeline hides latency there.
    load_subs = [s for s in recurring if "async_copy_load" in s.pipeline_stage or "global_read" in s.pipeline_stage]
    shared_load_subs = [s for s in recurring if "shared_load" in s.pipeline_stage]
    mma_subs = [s for s in recurring if "mma" in s.pipeline_stage or "fma_alu" in s.pipeline_stage]

    total_load_full = sum(_full_duration(s) for s in load_subs)
    total_shared_full = sum(_full_duration(s) for s in shared_load_subs)
    total_mma_full = sum(_full_duration(s) for s in mma_subs)

    total_load_tp = sum(_throughput_duration(s) for s in load_subs)
    total_shared_tp = sum(_throughput_duration(s) for s in shared_load_subs)
    total_mma_tp = sum(_throughput_duration(s) for s in mma_subs)

    # Prologue: all sequential (no overlap), latency included
    prologue_cycles = total_load_full + total_shared_full + total_mma_full

    # Steady state per iteration:
    # With async copy overlap, load[k] overlaps with mma[k-1]. shared_load[k]
    # is serial after load[k] (data dependency) but overlaps with mma[k-1].
    # The per-iteration advance is:
    #   max(load_tp + shared_tp, mma_tp)
    # — the load+shared chain vs. the mma throughput, whichever is longer.
    # Without async overlap, everything is sequential (same as prologue).
    if async_can_overlap:
        per_iteration_cycles = max(total_load_tp + total_shared_tp, total_mma_tp)
    else:
        per_iteration_cycles = total_load_full + total_shared_full + total_mma_full

    # Epilogue: sequential, latency included
    epilogue_cycles = sum(_full_duration(s) for s in epilogue)

    return PipelineSchedule(
        sub_ops=scheduled,
        total_cycles_per_block=total_cycles,
        total_time_s=total_time_s,
        wave_count=wave_count,
        grid_size=grid_size,
        num_k_iterations=num_k_iterations,
        bottleneck_stage=bottleneck,
        per_iteration_cycles=per_iteration_cycles,
        prologue_cycles=prologue_cycles,
        epilogue_cycles=epilogue_cycles,
    )


def _reschedule_solid(
    recurring, epilogue, num_k_iterations,
    full_duration_fn, throughput_duration_fn,
    async_can_overlap, cu, tiling, dims,
):
    """Re-schedule sub-ops using the SOL iteration formula.

    Produces a timeline with three phases:
    - Prologue (k=0): sequential, full durations (latency + throughput)
    - Steady state (k=1..N-1): overlap where applicable, throughput-only
      durations (latency hidden by the pipeline)
    - Epilogue: sequential, full durations

    Steady-state overlap model (async copy enabled):
      load[k] overlaps with mma[k-1] (starts at mma[k-1].start).
      shared_load[k] is serial after load[k] (data dependency) but overlaps
      with mma[k-1] (different execution units on Ampere — tensor cores vs.
      load/store units).
      mma[k] starts after both shared_load[k] and mma[k-1] complete.

      Per-iteration advance = max(load_tp + shared_tp, mma_tp)

    Without async copy, every iteration is sequential (no overlap):
      Per-iteration = load_full + shared_full + mma_full
    """
    from opcompass.models import ScheduledSubOp

    # Compute per-sub-op durations (full = with latency, tp = throughput only)
    full_durations = {s.name: full_duration_fn(s) for s in recurring}
    tp_durations = {s.name: throughput_duration_fn(s) for s in recurring}
    epilogue_durations = {s.name: full_duration_fn(s) for s in epilogue}

    # Group recurring sub-ops by pipeline_stage category
    load_subs = [s for s in recurring if "async_copy_load" in s.pipeline_stage or "global_read" in s.pipeline_stage]
    shared_load_subs = [s for s in recurring if "shared_load" in s.pipeline_stage]
    mma_subs = [s for s in recurring if "mma" in s.pipeline_stage or "fma_alu" in s.pipeline_stage]

    # ── Build scheduled sub-ops with proper cycle placement ────────────
    scheduled = []
    cycle = 0

    def _place(sub, start, dur, iteration):
        scheduled.append(ScheduledSubOp(
            name=f"{sub.name}_k{iteration}" if iteration >= 0 else sub.name,
            pipeline_stage=sub.pipeline_stage,
            start_cycle=start, end_cycle=start + dur,
            duration_cycles=dur,
            work_units=sub.read_bytes + sub.write_bytes + sub.flops,
            iteration=iteration,
        ))

    # ── Prologue (k=0): all sequential, full durations ─────────────────
    for s in load_subs:
        dur = full_durations[s.name]
        _place(s, cycle, dur, 0)
        cycle += dur
    for s in shared_load_subs:
        dur = full_durations[s.name]
        _place(s, cycle, dur, 0)
        cycle += dur
    for s in mma_subs:
        dur = full_durations[s.name]
        _place(s, cycle, dur, 0)
        cycle += dur

    prologue_end = cycle

    # ── Steady state (k=1..N-1) ────────────────────────────────────────
    if async_can_overlap and num_k_iterations > 1:
        # Track mma[k-1] position for overlap. In the prologue, mma[0] was
        # placed at the end (after load[0] + shared_load[0]).
        mma_full_dur = sum(full_durations[s.name] for s in mma_subs)
        mma_start_prev = prologue_end - mma_full_dur
        mma_end_prev = prologue_end

        for k in range(1, num_k_iterations):
            # load[k] starts at mma[k-1].start — overlaps with mma[k-1].
            # Use throughput-only durations (latency hidden by pipeline).
            load_start = mma_start_prev
            sub_cycle = load_start
            for s in load_subs:
                dur = tp_durations[s.name]
                _place(s, sub_cycle, dur, k)
                sub_cycle += dur
            load_end = sub_cycle

            # shared_load[k] starts after load[k] (data dependency).
            # It can overlap with mma[k-1] (different execution units),
            # so we don't wait for mma[k-1] to finish.
            shared_load_start = load_end
            sub_cycle = shared_load_start
            for s in shared_load_subs:
                dur = tp_durations[s.name]
                _place(s, sub_cycle, dur, k)
                sub_cycle += dur
            shared_load_end = sub_cycle

            # mma[k] starts after both shared_load[k] and mma[k-1] complete.
            mma_start = max(shared_load_end, mma_end_prev)
            sub_cycle = mma_start
            for s in mma_subs:
                dur = tp_durations[s.name]
                _place(s, sub_cycle, dur, k)
                sub_cycle += dur
            mma_end = sub_cycle

            mma_start_prev = mma_start
            mma_end_prev = mma_end

        cycle = mma_end_prev
    elif num_k_iterations > 1:
        # Without async: all sequential per iteration, full durations
        for k in range(1, num_k_iterations):
            for group in (load_subs, shared_load_subs, mma_subs):
                for s in group:
                    dur = full_durations[s.name]
                    _place(s, cycle, dur, k)
                    cycle += dur

    # ── Epilogue: sequential, full durations ───────────────────────────
    for s in epilogue:
        dur = epilogue_durations[s.name]
        _place(s, cycle, dur, -1)
        cycle += dur

    return scheduled
