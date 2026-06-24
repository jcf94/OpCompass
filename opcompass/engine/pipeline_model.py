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
    def _logical_work_units(sub: SubOp, stage) -> float:
        """Compute the work units (bytes or FMA) for a sub-op on a stage."""
        stage_name_lower = stage.name.lower()
        if any(kw in stage_name_lower for kw in ("read", "load", "write", "store", "copy")):
            return sub.read_bytes + sub.write_bytes
        elif any(kw in stage_name_lower for kw in ("mma", "alu", "compute")):
            # MMA throughput is in FMA/cycle; flops counts multiply+add as 2 ops
            # So work = flops / 2 to get FMA count
            return sub.flops / 2 if sub.flops > 0 else 0
        return sub.read_bytes + sub.write_bytes + sub.flops

    def _effective_hbm_work_units(sub: SubOp) -> float:
        """Return HBM bytes for global/async stages, defaulting to logical bytes."""
        read_bytes = (
            sub.effective_hbm_read_bytes
            if sub.effective_hbm_read_bytes is not None
            else sub.read_bytes
        )
        write_bytes = (
            sub.effective_hbm_write_bytes
            if sub.effective_hbm_write_bytes is not None
            else sub.write_bytes
        )
        return read_bytes + write_bytes

    def _stage_throughput(sub: SubOp, stage) -> float:
        """Return effective per-SM throughput for this sub-op and dtype."""
        throughput = stage.throughput_per_cycle
        stage_name = stage.name.lower()

        if any(kw in stage_name for kw in ("mma", "alu", "compute")):
            peak = hardware.get_peak_flops(dtype) if dtype is not None else 0.0
            if peak > 0 and cu.clock_mhz > 0 and cu.count > 0:
                throughput = peak / (2 * cu.clock_mhz * 1e6 * cu.count)

        if pipeline_config.sparsity_2_4_enabled and "mma" in stage_name:
            throughput *= 2

        return throughput

    def _hbm_bytes_per_cycle_per_sm() -> float:
        if hardware.hbm_bandwidth <= 0 or cu.clock_mhz <= 0 or cu.count <= 0:
            return 0.0
        return hardware.hbm_bandwidth / (cu.clock_mhz * 1e6) / cu.count

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
        logical_work = _logical_work_units(sub, stage)
        if throughput <= 0 or logical_work <= 0:
            return 0
        local_cycles = math.ceil(logical_work / throughput)

        stage_name = stage.name.lower()
        if any(kw in stage_name for kw in ("global", "async_copy")):
            hbm_throughput = _hbm_bytes_per_cycle_per_sm()
            hbm_work = _effective_hbm_work_units(sub)
            if hbm_throughput > 0 and hbm_work > 0:
                return max(local_cycles, math.ceil(hbm_work / hbm_throughput))

        return local_cycles

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
    register_file_regs = (cu.register_file_kb * 1024) // 4 if cu.register_file_kb > 0 else 0
    blocks_by_register_file = (
        register_file_regs // tiling.registers_per_block
        if register_file_regs > 0 and tiling.registers_per_block > 0
        else cu.max_thread_blocks_per_unit
    )
    blocks_by_register_block_limit = (
        cu.max_registers_per_block // tiling.registers_per_block
        if cu.max_registers_per_block > 0 and tiling.registers_per_block > 0
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
                blocks_by_register_file,
                blocks_by_register_block_limit,
            )
            if x and x > 0
        ),
    )
    blocks_per_sm = max(1, math.ceil(grid_size / cu.count)) if cu.count > 0 else grid_size
    wave_count = max(1, math.ceil(grid_size / (cu.count * resident_blocks_per_sm)))

    clock_s = 1.0 / (cu.clock_mhz * 1e6) if cu.clock_mhz > 0 else 0.0

    # Identify bottleneck stage
    stage_times: dict[str, int] = {}
    for s in scheduled:
        stage_times[s.pipeline_stage] = stage_times.get(s.pipeline_stage, 0) + s.duration_cycles
    bottleneck = max(stage_times, key=stage_times.get) if stage_times else ""

    # ``total_cycles`` is the critical-path time for one CTA using a full SM
    # pipeline. Resident CTAs improve latency hiding, but they do not multiply
    # the SM's MMA, async-copy, or load/store throughput. Estimate chip time by
    # charging each SM for the CTA stage work it must actually issue.
    resource_cycles = max(stage_times.values(), default=0) * blocks_per_sm
    total_time_s = max(total_cycles, resource_cycles) * clock_s

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

    prefetch_distance = max(0, tiling.stage_count - 1) if async_can_overlap else 0

    # Prologue: async kernels fill the software-pipeline window before the
    # first MMA. A 2-stage kernel prefetches only k0, matching the historical
    # double-buffered behavior; deeper kernels prefetch more K-slices.
    if async_can_overlap and prefetch_distance > 0:
        prologue_prefetches = min(prefetch_distance, num_k_iterations)
        prologue_cycles = (
            prologue_prefetches * (total_load_full + total_shared_full)
            + total_mma_full
        )
    else:
        prologue_cycles = total_load_full + total_shared_full + total_mma_full

    # Steady state per iteration:
    # With async copy overlap, load[k + prefetch_distance] overlaps with
    # mma[k]. shared_load is serial after its load (data dependency) but
    # overlaps with earlier MMA work.
    # The per-iteration advance is:
    #   max(load_tp + shared_tp, mma_tp)
    # — the load+shared chain vs. the mma throughput, whichever is longer.
    # Without async overlap, everything is sequential (same as prologue).
    if async_can_overlap and prefetch_distance > 0:
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
    - Prologue: sequential prefetch-window fill plus first MMA, full durations
    - Steady state (k=1..N-1): overlap where applicable, throughput-only
      durations (latency hidden by the pipeline)
    - Epilogue: sequential, full durations

    Steady-state overlap model (async copy enabled):
      load[k + prefetch_distance] overlaps with mma[k], where
      prefetch_distance = stage_count - 1.
      shared_load is serial after load (data dependency) but can overlap with
      MMA for an earlier K-slice (different execution units on Ampere —
      tensor cores vs. load/store units).
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

    def _place_prefetch(iteration, start, duration_by_name):
        sub_cycle = start
        for s in load_subs:
            dur = duration_by_name[s.name]
            _place(s, sub_cycle, dur, iteration)
            sub_cycle += dur
        for s in shared_load_subs:
            dur = duration_by_name[s.name]
            _place(s, sub_cycle, dur, iteration)
            sub_cycle += dur
        return sub_cycle

    def _place_mma(iteration, start, duration_by_name):
        sub_cycle = start
        for s in mma_subs:
            dur = duration_by_name[s.name]
            _place(s, sub_cycle, dur, iteration)
            sub_cycle += dur
        return sub_cycle

    prefetch_distance = max(0, tiling.stage_count - 1) if async_can_overlap else 0

    # ── Main loop ──────────────────────────────────────────────────────
    if async_can_overlap and prefetch_distance > 0:
        prefetched: set[int] = set()
        ready_cycle: dict[int, int] = {}

        # Prologue: fill the software-pipeline window with full latency.
        initial_prefetches = min(prefetch_distance, num_k_iterations)
        for k in range(initial_prefetches):
            cycle = _place_prefetch(k, cycle, full_durations)
            ready_cycle[k] = cycle
            prefetched.add(k)

        mma_available = cycle
        prefetch_available = cycle

        for k in range(num_k_iterations):
            if k not in prefetched:
                prefetch_available = _place_prefetch(k, prefetch_available, full_durations)
                ready_cycle[k] = prefetch_available
                prefetched.add(k)

            mma_start = max(mma_available, ready_cycle.get(k, 0))

            future_k = k + prefetch_distance
            if future_k < num_k_iterations and future_k not in prefetched:
                prefetch_start = max(mma_start, prefetch_available)
                prefetch_available = _place_prefetch(future_k, prefetch_start, tp_durations)
                ready_cycle[future_k] = prefetch_available
                prefetched.add(future_k)

            mma_durations = full_durations if k == 0 else tp_durations
            mma_available = _place_mma(k, mma_start, mma_durations)

        cycle = mma_available
    else:
        # Without async, or with only one software stage, every iteration is
        # sequential and pays full stage latency.
        for k in range(num_k_iterations):
            cycle = _place_prefetch(k, cycle, full_durations)
            cycle = _place_mma(k, cycle, full_durations)

    # ── Epilogue: sequential, full durations ───────────────────────────
    for s in epilogue:
        dur = epilogue_durations[s.name]
        _place(s, cycle, dur, -1)
        cycle += dur

    return scheduled
