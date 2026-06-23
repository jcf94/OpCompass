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
    from opcompass.models import PipelineSchedule, ScheduledSubOp

    cu = hardware.compute_unit
    overlap_tiers = hardware.memory.can_overlap_with_compute

    # ── Build stage map ────────────────────────────────────────────────
    stage_map: dict[str, object] = {s.name: s for s in cu.pipeline}

    # ── Separate recurring vs epilogue sub-ops ─────────────────────────
    recurring = [s for s in sub_ops if s.is_recurring]
    epilogue = [s for s in sub_ops if not s.is_recurring]

    K = dims.get("K", 0)
    bK = tiling.block_k
    num_k_iterations = max(1, math.ceil(K / bK)) if K > 0 else 1

    # ── Compute per-sub-op duration in cycles ──────────────────────────
    def _duration(sub: SubOp) -> int:
        """Compute duration in cycles for a single sub-op invocation."""
        stage = stage_map.get(sub.pipeline_stage)
        if stage is None:
            # Fallback: use HBM bandwidth and peak FLOPS
            total_work = sub.read_bytes + sub.write_bytes
            if total_work > 0:
                return math.ceil(total_work / (hardware.hbm_bandwidth / (cu.clock_mhz * 1e6)))
            if sub.flops > 0:
                peak = hardware.get_peak_flops(dims.get("dtype"))
                return math.ceil(sub.flops / (peak / (cu.clock_mhz * 1e6)))
            return 0

        # Determine work units based on stage type
        stage_name_lower = stage.name.lower()
        if any(kw in stage_name_lower for kw in ("read", "load", "write", "store", "copy")):
            work = sub.read_bytes + sub.write_bytes
        elif any(kw in stage_name_lower for kw in ("mma", "alu", "compute")):
            # MMA throughput is in FMA/cycle; flops counts multiply+add as 2 ops
            # So work = flops / 2 to get FMA count
            work = sub.flops / 2 if sub.flops > 0 else 0
        else:
            work = sub.read_bytes + sub.write_bytes + sub.flops

        # Apply sparsity modifier
        throughput = stage.throughput_per_cycle
        if pipeline_config.sparsity_2_4_enabled and "mma" in stage_name_lower:
            throughput *= 2

        if throughput <= 0 or work <= 0:
            return stage.latency_cycles

        return stage.latency_cycles + math.ceil(work / throughput)

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

    # ── Schedule recurring sub-ops across K iterations ─────────────────
    # We track the end_cycle of each sub-op by name for the current
    # iteration so that depends_on can reference them.
    scheduled: list[ScheduledSubOp] = []

    # Per-iteration tracking: name → end_cycle for the *previous* iteration
    prev_iter_ends: dict[str, int] = {}

    # Per-iteration tracking: name → end_cycle for the *current* iteration
    curr_iter_ends: dict[str, int] = {}

    # Track the mma end cycle for overlap calculations
    prev_mma_end = 0

    prologue_cycles = 0
    steady_per_iteration = 0

    for k in range(num_k_iterations):
        curr_iter_ends = {}
        iter_start = scheduled[-1].end_cycle if scheduled else 0

        for sub in recurring:
            dur = _duration(sub)

            # Compute start_cycle based on dependencies and overlap
            dep_end = 0
            for dep_name in sub.depends_on:
                # Intra-iteration dependency (same k)
                if dep_name in curr_iter_ends:
                    dep_end = max(dep_end, curr_iter_ends[dep_name])
                # Cross-iteration dependency for mma: mma[k] depends on
                # mma[k-1] completing (implicit — we model this via overlap)
                # This is handled by the overlap logic below

            # Overlap modeling for load stages
            start = dep_end

            is_load_stage = (
                "async_copy_load" in sub.pipeline_stage
                or "global_read" in sub.pipeline_stage
            )

            if is_load_stage and k > 0 and async_can_overlap:
                # async_copy_load[k] can overlap with mma[k-1]
                # BUT: buffer conflict — async needs shared mem buffer freed
                # by shared_load[k-1], so it starts at shared_load[k-1].end
                # In practice, the buffer is freed after shared_load completes,
                # and async_copy can start concurrently with MMA of the same
                # iteration once the buffer is free.
                #
                # Simplified model: load starts at shared_load[k-1].end_cycle
                # which happens inside the previous mma's execution window.
                # For the SOL model, we use the overlap formula:
                #   per-iteration = max(load_duration, mma_duration) + shared_load_duration
                # This is captured at the iteration level below.
                pass  # overlap handled at iteration level

            elif is_load_stage and k > 0 and not async_can_overlap:
                # Sequential: load starts after previous mma ends
                start = max(dep_end, prev_mma_end)

            # Track end cycle
            end = start + dur
            curr_iter_ends[sub.name] = end

            # Track mma end for overlap
            if sub.pipeline_stage == "mma":
                prev_mma_end = end

            scheduled.append(ScheduledSubOp(
                name=f"{sub.name}_k{k}",
                pipeline_stage=sub.pipeline_stage,
                start_cycle=start,
                end_cycle=end,
                duration_cycles=dur,
                work_units=sub.read_bytes + sub.write_bytes + sub.flops,
                iteration=k,
            ))

        # ── Iteration-level overlap scheduling ─────────────────────────
        # Instead of scheduling sub-ops individually with complex overlap
        # rules, we use the well-known SOL iteration formula:
        #
        # With async copy overlap:
        #   iteration_time = max(load_dur, mma_dur) + shared_load_dur
        # Without overlap:
        #   iteration_time = load_dur + shared_load_dur + mma_dur
        #
        # We re-schedule to reflect this formula for steady-state iterations.

    # ── Re-schedule using the SOL iteration formula ────────────────────
    # This produces a cleaner, more accurate timeline.
    scheduled = _reschedule_solid(
        recurring, epilogue, num_k_iterations, _duration,
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
    wave_count = max(1, math.ceil(grid_size / cu.count))

    clock_s = 1.0 / (cu.clock_mhz * 1e6) if cu.clock_mhz > 0 else 0.0
    total_time_s = total_cycles * clock_s * wave_count

    # Identify bottleneck stage
    stage_times: dict[str, int] = {}
    for s in scheduled:
        stage_times[s.pipeline_stage] = stage_times.get(s.pipeline_stage, 0) + s.duration_cycles
    bottleneck = max(stage_times, key=stage_times.get) if stage_times else ""

    # Compute per-iteration, prologue, and epilogue cycles
    recurring_durations = {s.name: _duration(s) for s in recurring}
    load_names = [s.name for s in recurring if "async_copy_load" in s.pipeline_stage or "global_read" in s.pipeline_stage]
    shared_load_names = [s.name for s in recurring if "shared_load" in s.pipeline_stage]
    mma_names = [s.name for s in recurring if "mma" in s.pipeline_stage]

    total_load_dur = sum(recurring_durations[n] for n in load_names)
    total_shared_load_dur = sum(recurring_durations[n] for n in shared_load_names)
    total_mma_dur = sum(recurring_durations[n] for n in mma_names)

    # Prologue: all sequential (no overlap)
    prologue_cycles = total_load_dur + total_shared_load_dur + total_mma_dur

    # Steady state per iteration
    if async_can_overlap:
        per_iteration_cycles = max(total_load_dur, total_mma_dur) + total_shared_load_dur
    else:
        per_iteration_cycles = total_load_dur + total_shared_load_dur + total_mma_dur

    # Epilogue
    epilogue_dur = sum(_duration(s) for s in epilogue)
    epilogue_cycles = epilogue_dur

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
    recurring, epilogue, num_k_iterations, duration_fn,
    async_can_overlap, cu, tiling, dims,
):
    """Re-schedule sub-ops using the SOL iteration formula.

    This produces a clear, accurate timeline that reflects:
    - Prologue: sequential execution (iteration 0)
    - Steady state: overlap where applicable (iterations 1..N-1)
    - Epilogue: sequential final stages

    The iteration-level formula:
    - With async overlap: iter_time = max(load_dur, mma_dur) + shared_load_dur
    - Without overlap:    iter_time = load_dur + shared_load_dur + mma_dur
    """
    from opcompass.models import ScheduledSubOp

    # Compute per-sub-op durations
    durations = {s.name: duration_fn(s) for s in recurring}
    epilogue_durations = {s.name: duration_fn(s) for s in epilogue}

    # Group recurring sub-ops by pipeline_stage category
    load_subs = [s for s in recurring if "async_copy_load" in s.pipeline_stage or "global_read" in s.pipeline_stage]
    shared_load_subs = [s for s in recurring if "shared_load" in s.pipeline_stage]
    mma_subs = [s for s in recurring if "mma" in s.pipeline_stage or "fma_alu" in s.pipeline_stage]

    total_load_dur = sum(durations[s.name] for s in load_subs)
    total_shared_load_dur = sum(durations[s.name] for s in shared_load_subs)
    total_mma_dur = sum(durations[s.name] for s in mma_subs)
    total_epilogue_dur = sum(epilogue_durations[s.name] for s in epilogue)

    # ── Build scheduled sub-ops with proper cycle placement ────────────
    scheduled = []
    cycle = 0

    # ── Prologue (k=0): all sequential ─────────────────────────────────
    # Load stages
    for s in load_subs:
        dur = durations[s.name]
        scheduled.append(ScheduledSubOp(
            name=f"{s.name}_k0", pipeline_stage=s.pipeline_stage,
            start_cycle=cycle, end_cycle=cycle + dur,
            duration_cycles=dur,
            work_units=s.read_bytes + s.write_bytes + s.flops,
            iteration=0,
        ))
        cycle += dur

    # Shared load stages
    for s in shared_load_subs:
        dur = durations[s.name]
        scheduled.append(ScheduledSubOp(
            name=f"{s.name}_k0", pipeline_stage=s.pipeline_stage,
            start_cycle=cycle, end_cycle=cycle + dur,
            duration_cycles=dur,
            work_units=s.read_bytes + s.write_bytes + s.flops,
            iteration=0,
        ))
        cycle += dur

    # MMA stage
    for s in mma_subs:
        dur = durations[s.name]
        scheduled.append(ScheduledSubOp(
            name=f"{s.name}_k0", pipeline_stage=s.pipeline_stage,
            start_cycle=cycle, end_cycle=cycle + dur,
            duration_cycles=dur,
            work_units=s.read_bytes + s.write_bytes + s.flops,
            iteration=0,
        ))
        cycle += dur

    prologue_end = cycle

    # ── Steady state (k=1..N-1): overlap where applicable ──────────────
    if async_can_overlap and num_k_iterations > 1:
        # With async copy overlap:
        #   load[k] starts concurrently with mma[k-1] (overlapping)
        #   shared_load[k] starts after load[k] completes
        #   mma[k] starts after shared_load[k] completes AND mma[k-1] completes
        #
        # Per-iteration formula:
        #   iter_time = max(load_dur, mma_dur) + shared_load_dur
        #
        # We schedule by placing load[k] alongside mma[k-1]:
        #   load[k].start = mma[k-1].start  (overlap with MMA)
        #   shared_load[k].start = max(load[k].end, mma[k-1].end)
        #   mma[k].start = shared_load[k].end

        mma_start_prev = prologue_end - total_mma_dur  # mma[k-1] started here
        mma_end_prev = prologue_end

        for k in range(1, num_k_iterations):
            # Load starts overlapping with previous MMA
            load_start = mma_start_prev
            load_end = load_start + total_load_dur

            # Place individual load sub-ops
            sub_cycle = load_start
            for s in load_subs:
                dur = durations[s.name]
                scheduled.append(ScheduledSubOp(
                    name=f"{s.name}_k{k}", pipeline_stage=s.pipeline_stage,
                    start_cycle=sub_cycle, end_cycle=sub_cycle + dur,
                    duration_cycles=dur,
                    work_units=s.read_bytes + s.write_bytes + s.flops,
                    iteration=k,
                ))
                sub_cycle += dur

            # Shared load starts after both load[k] and mma[k-1] complete
            shared_load_start = max(load_end, mma_end_prev)
            shared_load_end = shared_load_start + total_shared_load_dur

            sub_cycle = shared_load_start
            for s in shared_load_subs:
                dur = durations[s.name]
                scheduled.append(ScheduledSubOp(
                    name=f"{s.name}_k{k}", pipeline_stage=s.pipeline_stage,
                    start_cycle=sub_cycle, end_cycle=sub_cycle + dur,
                    duration_cycles=dur,
                    work_units=s.read_bytes + s.write_bytes + s.flops,
                    iteration=k,
                ))
                sub_cycle += dur

            # MMA starts after shared_load completes
            mma_start = shared_load_end
            mma_end = mma_start + total_mma_dur

            for s in mma_subs:
                dur = durations[s.name]
                scheduled.append(ScheduledSubOp(
                    name=f"{s.name}_k{k}", pipeline_stage=s.pipeline_stage,
                    start_cycle=mma_start, end_cycle=mma_end,
                    duration_cycles=dur,
                    work_units=s.read_bytes + s.write_bytes + s.flops,
                    iteration=k,
                ))

            # Update for next iteration
            mma_start_prev = mma_start
            mma_end_prev = mma_end

        cycle = mma_end_prev
    elif num_k_iterations > 1:
        # Without overlap: all sequential per iteration
        for k in range(1, num_k_iterations):
            for s in load_subs:
                dur = durations[s.name]
                scheduled.append(ScheduledSubOp(
                    name=f"{s.name}_k{k}", pipeline_stage=s.pipeline_stage,
                    start_cycle=cycle, end_cycle=cycle + dur,
                    duration_cycles=dur,
                    work_units=s.read_bytes + s.write_bytes + s.flops,
                    iteration=k,
                ))
                cycle += dur

            for s in shared_load_subs:
                dur = durations[s.name]
                scheduled.append(ScheduledSubOp(
                    name=f"{s.name}_k{k}", pipeline_stage=s.pipeline_stage,
                    start_cycle=cycle, end_cycle=cycle + dur,
                    duration_cycles=dur,
                    work_units=s.read_bytes + s.write_bytes + s.flops,
                    iteration=k,
                ))
                cycle += dur

            for s in mma_subs:
                dur = durations[s.name]
                scheduled.append(ScheduledSubOp(
                    name=f"{s.name}_k{k}", pipeline_stage=s.pipeline_stage,
                    start_cycle=cycle, end_cycle=cycle + dur,
                    duration_cycles=dur,
                    work_units=s.read_bytes + s.write_bytes + s.flops,
                    iteration=k,
                ))
                cycle += dur

    # ── Epilogue: sequential final stages ──────────────────────────────
    for s in epilogue:
        dur = epilogue_durations[s.name]
        scheduled.append(ScheduledSubOp(
            name=s.name, pipeline_stage=s.pipeline_stage,
            start_cycle=cycle, end_cycle=cycle + dur,
            duration_cycles=dur,
            work_units=s.read_bytes + s.write_bytes + s.flops,
            iteration=-1,  # epilogue, not part of K-loop
        ))
        cycle += dur

    return scheduled
