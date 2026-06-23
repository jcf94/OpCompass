"""Formatting helpers for AnalysisResult."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opcompass.models import AnalysisResult


def format_result(result: AnalysisResult, fmt: str = "table") -> str:
    """Format an AnalysisResult as a human-readable string.

    Args:
        result: The analysis result to format.
        fmt: One of ``"table"``, ``"json"``, ``"csv"``.

    Returns:
        Formatted string.
    """
    if fmt == "json":
        import json
        return json.dumps(_result_to_dict(result), indent=2, ensure_ascii=False)

    if fmt == "csv":
        return _format_csv(result)

    return _format_table(result)


# ---------------------------------------------------------------------------
# Internal formatters
# ---------------------------------------------------------------------------

def _result_to_dict(result: AnalysisResult) -> dict:
    d = {
        "operator": result.operator,
        "hardware": result.hardware,
        "shapes": result.shapes,
        "dtype": result.dtype.value,
        "mode": result.mode.value,
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

    # Pipeline-specific fields
    if result.pipeline_schedule is not None:
        ps = result.pipeline_schedule
        d["pipeline_schedule"] = {
            "sub_ops": [
                {
                    "name": sop.name,
                    "pipeline_stage": sop.pipeline_stage,
                    "start_cycle": sop.start_cycle,
                    "end_cycle": sop.end_cycle,
                    "duration_cycles": sop.duration_cycles,
                    "work_units": sop.work_units,
                    "iteration": sop.iteration,
                }
                for sop in ps.sub_ops
            ],
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
        }

    if result.tiling_info is not None:
        ti = result.tiling_info
        d["tiling_info"] = {
            "block_m": ti.block_m,
            "block_n": ti.block_n,
            "block_k": ti.block_k,
            "shared_memory_per_block": ti.shared_memory_per_block,
            "num_warps_per_block": ti.num_warps_per_block,
        }

    if result.pipeline_config is not None:
        pc = result.pipeline_config
        d["pipeline_config"] = {
            "async_copy_enabled": pc.async_copy_enabled,
            "sparsity_2_4_enabled": pc.sparsity_2_4_enabled,
            "block_m": pc.block_m,
            "block_n": pc.block_n,
            "block_k": pc.block_k,
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
        f"  Mode       : {result.mode.value}",
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
            f"  {'Prologue':<20} {ps.prologue_cycles:>10,}  {ps.prologue_cycles * 1e3 / result.compute_unit_clock_hz if hasattr(result, 'compute_unit_clock_hz') else 0:>9.1f} µs",
            f"  {'Steady state ×' + str(ps.num_k_iterations - 1) if ps.num_k_iterations > 1 else 'Steady state':<20} {ps.per_iteration_cycles * max(0, ps.num_k_iterations - 1):>10,}",
            f"  {'Epilogue':<20} {ps.epilogue_cycles:>10,}",
            "─" * 65,
            f"  Total cycles/block : {ps.total_cycles_per_block:,}",
            f"  Grid size          : {ps.grid_size} blocks",
            f"  Wave count         : {ps.wave_count}  (ceil(grid / resident CTAs))",
            f"  K iterations       : {ps.num_k_iterations}  (ceil(K / block_K))",
            f"  Bottleneck stage   : {ps.bottleneck_stage}",
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
                f"  Tiling (bM×bN×bK) : {ti.block_m}×{ti.block_n}×{ti.block_k}",
                f"  Shared mem/block  : {ti.shared_memory_per_block:,} bytes  ({ti.shared_memory_per_block / 1024:.0f} KB)",
                f"  Warps/block       : {ti.num_warps_per_block}",
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
    header = ",".join(d.keys())
    values = ",".join(str(v) for v in d.values())
    return f"{header}\n{values}"
