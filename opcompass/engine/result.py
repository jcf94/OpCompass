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
        }

    return d


def _format_table(result: AnalysisResult) -> str:
    ops = f"{result.total_flops / 1e9:.2f} GFLOPs"
    read = f"{result.total_read_bytes / 1e9:.2f} GB"
    write = f"{result.total_write_bytes / 1e9:.2f} GB"
    sol_us = result.sol_time_s * 1e6

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
        "═" * 65,
    ]

    # Add pipeline-specific info
    if result.pipeline_schedule is not None:
        ps = result.pipeline_schedule
        lines += [
            "─" * 65,
            "  Pipeline Details:",
            f"  K iterations      : {ps.num_k_iterations}",
            f"  Grid size         : {ps.grid_size} blocks",
            f"  Wave count        : {ps.wave_count}",
            f"  Prologue cycles   : {ps.prologue_cycles}",
            f"  Per-iter cycles   : {ps.per_iteration_cycles}",
            f"  Epilogue cycles   : {ps.epilogue_cycles}",
            f"  Total cycles/block: {ps.total_cycles_per_block}",
            f"  Bottleneck stage  : {ps.bottleneck_stage}",
        ]
        if result.tiling_info is not None:
            ti = result.tiling_info
            lines += [
                f"  Tiling (bM×bN×bK) : {ti.block_m}×{ti.block_n}×{ti.block_k}",
                f"  Shared mem/block  : {ti.shared_memory_per_block} bytes",
                f"  Warps/block       : {ti.num_warps_per_block}",
            ]
        if result.pipeline_config is not None:
            pc = result.pipeline_config
            lines += [
                f"  Async copy        : {'ON' if pc.async_copy_enabled else 'OFF'}",
                f"  2:4 Sparsity      : {'ON' if pc.sparsity_2_4_enabled else 'OFF'}",
            ]
        lines += ["═" * 65]

    return "\n".join(lines)


def _format_csv(result: AnalysisResult) -> str:
    d = _result_to_dict(result)
    header = ",".join(d.keys())
    values = ",".join(str(v) for v in d.values())
    return f"{header}\n{values}"
