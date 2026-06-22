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
    return {
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
    }


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
    return "\n".join(lines)


def _format_csv(result: AnalysisResult) -> str:
    d = _result_to_dict(result)
    header = ",".join(d.keys())
    values = ",".join(str(v) for v in d.values())
    return f"{header}\n{values}"
