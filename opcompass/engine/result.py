"""Formatting helpers for AnalysisResult."""

from __future__ import annotations

from typing import TYPE_CHECKING

import csv
import io
import json

if TYPE_CHECKING:
    from opcompass.models import AnalysisResult


MAX_TRACE_SUB_OPS = 5000


def format_result(
    result: AnalysisResult,
    fmt: str = "table",
    include_trace: bool = False,
    trace_limit: int = 1000,
) -> str:
    """Format an AnalysisResult as a human-readable string.

    Args:
        result: The analysis result to format.
        fmt: One of ``"table"``, ``"json"``, ``"csv"``.

    Returns:
        Formatted string.
    """
    if fmt == "json":
        return json.dumps(
            _result_to_dict(result, include_trace=include_trace, trace_limit=trace_limit),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    if fmt == "csv":
        return _format_csv(result)

    return _format_table(result)


# ---------------------------------------------------------------------------
# Internal formatters
# ---------------------------------------------------------------------------

def _result_to_dict(
    result: AnalysisResult,
    include_trace: bool = False,
    trace_limit: int = 1000,
) -> dict:
    if trace_limit < 1 or trace_limit > MAX_TRACE_SUB_OPS:
        raise ValueError(f"trace_limit must be between 1 and {MAX_TRACE_SUB_OPS}")
    d = {
        "operator": result.operator,
        "hardware": result.hardware,
        "shapes": result.shapes,
        "dtype": result.dtype.value,
        "mode": result.mode.value,
        "requested_mode": result.requested_mode.value,
        "executed_mode": result.executed_mode.value,
        "estimate_kind": result.estimate_kind.value,
        "support_level": result.support_level.value,
        "schema_version": result.schema_version,
        "model_id": result.model_id,
        "implementation_version": result.implementation_version,
        "implementation_revision": result.implementation_revision,
        "hardware_spec_version": result.hardware_spec_version,
        "evidence": {
            "coverage": result.evidence.coverage,
            "sources": list(result.evidence.sources),
        },
        "uncertainty": {
            "status": result.uncertainty.status,
            "reason": result.uncertainty.reason,
            "lower_time_us": (
                result.uncertainty.lower_time_s * 1e6
                if result.uncertainty.lower_time_s is not None else None
            ),
            "upper_time_us": (
                result.uncertainty.upper_time_s * 1e6
                if result.uncertainty.upper_time_s is not None else None
            ),
        },
        "fallback": (
            {
                "from_mode": result.fallback.from_mode.value,
                "to_mode": result.fallback.to_mode.value,
                "reason_code": result.fallback.reason_code,
                "message": result.fallback.message,
            }
            if result.fallback else None
        ),
        "assumptions": result.assumptions,
        "warnings": result.warnings,
        "missing_effects": result.missing_effects,
        "total_flops": result.total_flops,
        "total_read_bytes": result.total_read_bytes,
        "total_write_bytes": result.total_write_bytes,
        "memory_read_time_us": result.memory_read_time_s * 1e6,
        "compute_time_us": result.compute_time_s * 1e6,
        "memory_write_time_us": result.memory_write_time_s * 1e6,
        "sol_time_us": result.sol_time_s * 1e6,
        "sol_tflops": result.sol_tflops,
        "bottleneck": result.bottleneck,
        "stage_breakdown": result.stage_breakdown,
        "roofline_data": result.roofline_data,
    }
    if result.pipeline_memory_breakdown:
        d["pipeline_memory_breakdown"] = result.pipeline_memory_breakdown
    if result.pipeline_candidates:
        d["pipeline_candidates"] = [
            {
                "name": c.name,
                "block_m": c.block_m,
                "block_n": c.block_n,
                "block_k": c.block_k,
                "warp_count": c.warp_count,
                "stage_count": c.stage_count,
                "copy_path": c.copy_path,
                "mma_path": c.mma_path,
                "scheduling": c.scheduling,
                "cta_order": c.cta_order,
                "rejection_reason": c.rejection_reason,
                "selected": i == 0 and not c.rejection_reason,
            }
            for i, c in enumerate(result.pipeline_candidates)
        ]
    if result.pipeline_ir_schedule is not None:
        ir = result.pipeline_ir_schedule
        d["pipeline_ir_schedule"] = {
            "total_cycles": ir.total_cycles,
            "loop_iterations": ir.loop_iterations,
            "resource_busy_cycles": ir.resource_busy_cycles,
            "trace": {
                "included": False,
                "total_nodes": len(ir.entries),
                "returned_nodes": 0,
            },
        }
    if result.pipeline_legacy_comparison:
        d["pipeline_legacy_comparison"] = result.pipeline_legacy_comparison

    # Pipeline-specific fields
    if result.pipeline_schedule is not None:
        ps = result.pipeline_schedule
        d["pipeline_schedule"] = {
            "total_cycles_per_block": ps.total_cycles_per_block,
            "total_time_s": ps.total_time_s,
            "total_time_us": ps.total_time_s * 1e6,
            "wave_count": ps.wave_count,
            "grid_size": ps.grid_size,
            "num_k_iterations": ps.num_k_iterations,
            "bottleneck_stage": ps.bottleneck_stage,
            "per_iteration_cycles": ps.per_iteration_cycles,
            "prologue_cycles": ps.prologue_cycles,
            "epilogue_cycles": ps.epilogue_cycles,
            "trace": {
                "included": include_trace,
                "total_sub_ops": len(ps.sub_ops),
                "returned_sub_ops": min(len(ps.sub_ops), trace_limit) if include_trace else 0,
                "complete": include_trace and len(ps.sub_ops) <= trace_limit,
                "limit": trace_limit,
            },
        }
        if include_trace:
            d["pipeline_schedule"]["sub_ops"] = [
                {
                    "name": sop.name,
                    "pipeline_stage": sop.pipeline_stage,
                    "start_cycle": sop.start_cycle,
                    "end_cycle": sop.end_cycle,
                    "duration_cycles": sop.duration_cycles,
                    "work_units": sop.work_units,
                    "iteration": sop.iteration,
                }
                for sop in ps.sub_ops[:trace_limit]
            ]

    if result.tiling_info is not None:
        ti = result.tiling_info
        d["tiling_info"] = {
            "block_m": ti.block_m,
            "block_n": ti.block_n,
            "block_k": ti.block_k,
            "shared_memory_per_block": ti.shared_memory_per_block,
            "num_warps_per_block": ti.num_warps_per_block,
            "stage_count": ti.stage_count,
            "registers_per_thread": ti.registers_per_thread,
            "registers_per_block": ti.registers_per_block,
            "candidate_name": ti.candidate_name,
        }

    if result.pipeline_config is not None:
        pc = result.pipeline_config
        d["pipeline_config"] = {
            "async_copy_enabled": pc.async_copy_enabled,
            "sparsity_2_4_enabled": pc.sparsity_2_4_enabled,
            "block_m": pc.block_m,
            "block_n": pc.block_n,
            "block_k": pc.block_k,
            "stage_count": pc.stage_count,
            "warp_count": pc.warp_count,
        }

    # Solar-specific fields
    if result.solar_data is not None:
        sd = result.solar_data
        d["solar_data"] = {
            "num_layers": sd.num_layers,
            "total_macs": sd.total_macs,
            "arch_name": sd.arch_name,
            "arch_freq_ghz": sd.arch_freq_ghz,
            "unfused": {
                "runtime_ms": sd.unfused_runtime_ms,
                "bottleneck": sd.unfused_bottleneck,
                "arithmetic_intensity": sd.unfused_arithmetic_intensity,
                "memory_bytes": sd.unfused_memory_bytes,
                "compute_cycles": sd.unfused_compute_cycles,
            },
            "fused": {
                "runtime_ms": sd.fused_runtime_ms,
                "bottleneck": sd.fused_bottleneck,
                "arithmetic_intensity": sd.fused_arithmetic_intensity,
                "memory_bytes": sd.fused_memory_bytes,
            },
            "fused_prefetched": {
                "runtime_ms": sd.fused_prefetched_runtime_ms,
                "bottleneck": sd.fused_prefetched_bottleneck,
                "arithmetic_intensity": sd.fused_prefetched_arithmetic_intensity,
                "memory_bytes": sd.fused_prefetched_memory_bytes,
            },
            "memory_breakdown": {
                "weight_bytes": sd.weight_bytes,
                "model_io_bytes": sd.model_io_bytes,
                "intermediate_bytes": sd.intermediate_bytes,
            },
            "speedup": {
                "fused_vs_unfused": sd.fused_speedup,
                "fused_prefetched_vs_unfused": sd.fused_prefetched_speedup,
            },
        }

    return d


def _format_table(result: AnalysisResult) -> str:
    ops = f"{result.total_flops / 1e9:.2f} GFLOPs"
    read = f"{result.total_read_bytes / 1e9:.2f} GB"
    write = f"{result.total_write_bytes / 1e9:.2f} GB"
    sol_us = result.sol_time_s * 1e6

    # In pipeline mode, the Read/Compute/Write figures are non-additive:
    # pipeline stages overlap, and resident CTAs share the same SM pipeline
    # throughput rather than multiplying it.
    # For non-pipeline modes the figures follow max() or sum() semantics
    # depending on hardware.can_overlap_with_compute.
    is_pipeline = result.pipeline_schedule is not None

    lines = [
        "═" * 65,
        f"  OpCompass SOL Analysis",
        "─" * 65,
        f"  Operator   : {result.operator}",
        f"  Hardware   : {result.hardware}",
        f"  Shapes     : {result.shapes}",
        f"  Dtype      : {result.dtype.value}",
        f"  Mode       : {result.requested_mode.value} → {result.executed_mode.value}",
        f"  Estimate   : {result.estimate_kind.value} ({result.support_level.value})",
        f"  Model      : {result.model_id} / schema {result.schema_version}",
        f"  Build      : {result.implementation_version} @ {result.implementation_revision[:12]}",
        f"  HW spec    : {result.hardware_spec_version}",
        "─" * 65,
        f"  Total FLOPs : {ops:>18s}",
        f"  Read bytes  : {read:>18s}",
        f"  Write bytes : {write:>18s}",
        "─" * 65,
        f"  Memory Read  time : {result.memory_read_time_s * 1e6:8.1f} µs",
        f"  Compute      time : {result.compute_time_s * 1e6:8.1f} µs",
        f"  Memory Write time : {result.memory_write_time_s * 1e6:8.1f} µs",
        "─" * 65,
        f"  ★ SOL time   : {sol_us:8.1f} µs  ({result.sol_tflops:.1f} TFLOPS)",
        f"  ★ Bottleneck : {result.bottleneck}",
    ]
    if result.fallback is not None:
        lines.append(f"  ⚠ Fallback : {result.fallback.message}")
    if is_pipeline:
        lines.append("  Note: Read/Compute/Write above are non-additive because pipeline stages overlap")
    lines.append("═" * 65)

    # Add pipeline-specific info
    if result.pipeline_schedule is not None:
        ps = result.pipeline_schedule
        # Aggregate per-stage cycle counts from scheduled sub-ops
        stage_cycles: dict[str, int] = {}
        for sop in ps.sub_ops:
            stage = sop.pipeline_stage
            stage_cycles[stage] = stage_cycles.get(stage, 0) + sop.duration_cycles

        lines += [
            "",
            "═" * 65,
            "  Pipeline Analysis",
            "─" * 65,
            f"  {'Phase':<20} {'Cycles':>10}  {'Time':>10}",
            f"  {'─'*20} {'─'*10}  {'─'*10}",
            f"  {'Prologue':<20} {ps.prologue_cycles:>10,}  {ps.prologue_cycles / result.compute_unit_clock_hz * 1e6:>9.3f} µs",
            f"  {'Steady state ×' + str(ps.num_k_iterations - 1) if ps.num_k_iterations > 1 else 'Steady state':<20} {ps.per_iteration_cycles * max(0, ps.num_k_iterations - 1):>10,}  {ps.per_iteration_cycles * max(0, ps.num_k_iterations - 1) / result.compute_unit_clock_hz * 1e6:>9.3f} µs",
            f"  {'Epilogue':<20} {ps.epilogue_cycles:>10,}  {ps.epilogue_cycles / result.compute_unit_clock_hz * 1e6:>9.3f} µs",
            "─" * 65,
            f"  Total cycles/block : {ps.total_cycles_per_block:,}",
            f"  Grid size          : {ps.grid_size} blocks",
            f"  Wave count         : {ps.wave_count}  (ceil(grid / resident CTAs))",
            f"  K iterations       : {ps.num_k_iterations}  (ceil(K / block_K))",
            f"  Bottleneck stage   : {ps.bottleneck_stage}",
        ]
        memory = result.pipeline_memory_breakdown
        if memory:
            lines += [
                "",
                "  Pipeline Memory:",
                f"    Effective HBM read     : {memory.get('effective_hbm_read_bytes', 0) / 1e9:.3f} GB",
                f"    CTA logical read       : {memory.get('logical_cta_read_bytes', 0) / 1e9:.3f} GB",
                f"    Unique tensor read     : {memory.get('unique_tensor_read_bytes', 0) / 1e9:.3f} GB",
                f"    L2 reuse factor        : {memory.get('l2_reuse_factor', 1):.2f}x",
            ]
        lines += [
            "",
            f"  Stage Cycle Breakdown:",
        ]

        # Show per-stage cycle counts sorted by magnitude
        for stage, cycles in sorted(stage_cycles.items(), key=lambda x: -x[1]):
            pct = cycles / max(ps.total_cycles_per_block, 1) * 100
            lines.append(f"    {stage:<25} {cycles:>10,}  ({pct:5.1f}%)")

        if result.tiling_info is not None:
            ti = result.tiling_info
            lines += [
                "",
                f"  Candidate         : {ti.candidate_name or 'default'}",
                f"  Tiling (bM×bN×bK) : {ti.block_m}×{ti.block_n}×{ti.block_k}",
                f"  Shared mem/block  : {ti.shared_memory_per_block:,} bytes  ({ti.shared_memory_per_block / 1024:.0f} KB)",
                f"  Warps/block       : {ti.num_warps_per_block}",
                f"  Stage count       : {ti.stage_count}",
            ]
            if ti.registers_per_thread:
                lines += [
                    f"  Registers/thread  : {ti.registers_per_thread}",
                    f"  Registers/block   : {ti.registers_per_block:,}",
                ]
        if result.pipeline_config is not None:
            pc = result.pipeline_config
            lines += [
                f"  Async copy        : {'ON' if pc.async_copy_enabled else 'OFF'}",
                f"  2:4 Sparsity      : {'ON' if pc.sparsity_2_4_enabled else 'OFF'}",
            ]
        lines += ["═" * 65]

    # Add solar-specific info
    if result.solar_data is not None:
        sd = result.solar_data
        lines += [
            "",
            "═" * 65,
            f"  SOLAR Analysis  (arch: {sd.arch_name} @ {sd.arch_freq_ghz} GHz)",
            "─" * 65,
            f"  Workload: {sd.num_layers} layers, {sd.total_macs:,} MACs, {sd.total_flops:,} FLOPs",
            "─" * 65,
            f"  {'Model':<24} {'Runtime':>8} {'Bottleneck':>14} {'AI (FLOP/B)':>13}",
            f"  {'─'*24} {'─'*8} {'─'*14} {'─'*13}",
            f"  {'Unfused':<24} {sd.unfused_runtime_ms:>7.3f} ms {sd.unfused_bottleneck:>14} {sd.unfused_arithmetic_intensity:>13.1f}",
            f"  {'Fused':<24} {sd.fused_runtime_ms:>7.3f} ms {sd.fused_bottleneck:>14} {sd.fused_arithmetic_intensity:>13.1f}",
            f"  {'Fused+Prefetched ★':<24} {sd.fused_prefetched_runtime_ms:>7.3f} ms {sd.fused_prefetched_bottleneck:>14} {sd.fused_prefetched_arithmetic_intensity:>13.1f}",
            "─" * 65,
            "  Memory Breakdown:",
            f"    Weights      : {sd.weight_bytes / 1e9:.3f} GB",
            f"    Model I/O    : {sd.model_io_bytes / 1e9:.3f} GB",
            f"    Intermediate : {sd.intermediate_bytes / 1e9:.3f} GB",
            "─" * 65,
            f"  Speedup: Fused={sd.fused_speedup:.2f}×  Fused+Prefetched={sd.fused_prefetched_speedup:.2f}×",
            "═" * 65,
        ]

    return "\n".join(lines)


def _format_csv(result: AnalysisResult) -> str:
    d = _result_to_dict(result)
    row = {
        "schema_version": d["schema_version"],
        "operator": d["operator"],
        "hardware": d["hardware"],
        "shapes_json": json.dumps(d["shapes"], separators=(",", ":")),
        "dtype": d["dtype"],
        "requested_mode": d["requested_mode"],
        "executed_mode": d["executed_mode"],
        "estimate_kind": d["estimate_kind"],
        "support_level": d["support_level"],
        "model_id": d["model_id"],
        "fallback_reason_code": d["fallback"]["reason_code"] if d["fallback"] else "",
        "total_flops": d["total_flops"],
        "total_read_bytes": d["total_read_bytes"],
        "total_write_bytes": d["total_write_bytes"],
        "sol_time_us": d["sol_time_us"],
        "sol_tflops": d["sol_tflops"],
        "bottleneck": d["bottleneck"],
    }
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    return stream.getvalue().rstrip("\n")
