"""FastAPI server for OpCompass — serves the web UI and REST API.

Start with::

    uvicorn opcompass.server:app --reload
"""

# NOTE: This file uses Python 3.8-compatible typing (no 'from __future__ import annotations')
# because FastAPI/Pydantic needs to evaluate type annotations at runtime.

import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from opcompass.registry import (
    discover_hardware,
    discover_operators,
    get_hardware,
    get_operator,
)
from opcompass.models import AnalysisMode, DataType, PipelineConfig
from opcompass.engine.analyzer import Analyzer
from opcompass.engine.result import _result_to_dict

app = FastAPI(
    title="OpCompass API",
    description="SOL theoretical peak performance estimator for GPU operators",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/operators")
def api_list_operators() -> List[Dict[str, Any]]:
    """Return all registered operators with their metadata."""
    ops = discover_operators()
    result: List[Dict[str, Any]] = []
    for name, cls in sorted(ops.items()):
        inst = cls()
        result.append({
            "name": name,
            "description": inst.description,
            "param_dims": inst.param_dims,
        })
    return result


def _sm_version_sort_key(entry: Dict[str, Any]) -> float:
    """Extract SM version as a float for sorting (e.g. '9.0' → 9.0)."""
    try:
        return float(entry.get("sm_version", "0"))
    except (ValueError, TypeError):
        return 0.0


@app.get("/api/hardware")
def api_list_hardware() -> List[Dict[str, Any]]:
    """Return all registered hardware targets with key specs.

    Results are sorted by SM version descending (newest architecture first).
    """
    hw = discover_hardware()
    result: List[Dict[str, Any]] = []
    for name, cls in hw.items():
        inst = cls()
        result.append({
            "name": name,
            "vendor": inst.vendor,
            "description": inst.description,
            "architecture": getattr(inst, "architecture", ""),
            "sm_version": getattr(inst, "sm_version", ""),
            "num_sms": inst.num_compute_units,
            "clock_mhz": inst.compute_unit.clock_mhz,
            "hbm_bandwidth_gb_s": inst.hbm_bandwidth / 1e9,
            "peak_flops": {
                dt.value: flops
                for dt, flops in inst.compute_unit.peak_flops.items()
            },
        })

    # Sort by SM version descending
    result.sort(key=lambda hw: _sm_version_sort_key(hw), reverse=True)
    return result


@app.get("/api/hardware/overview")
def api_hardware_overview() -> List[Dict[str, Any]]:
    """Return all hardware targets with full specs for side-by-side comparison.

    Each entry includes architecture metadata, memory hierarchy, compute
    unit specs, and peak performance across all supported dtypes.  Results
    are sorted by SM version descending.

    NOTE: This route must be defined BEFORE ``/api/hardware/{{name}}`` so
    that the literal path ``overview`` takes precedence over the name
    parameter.
    """
    hw = discover_hardware()
    result: List[Dict[str, Any]] = []
    for name, cls in hw.items():
        inst = cls()
        cu = inst.compute_unit

        entry: Dict[str, Any] = {
            # ── Identity ─────────────────────────────────────────
            "name": inst.name,
            "vendor": inst.vendor,
            "description": inst.description,
            "architecture": getattr(inst, "architecture", ""),
            "sm_version": getattr(inst, "sm_version", ""),

            # ── Compute unit ─────────────────────────────────────
            "cu_name": cu.name,
            "cu_count": cu.count,
            "clock_mhz": cu.clock_mhz,

            # ── Memory ───────────────────────────────────────────
            "memory_tiers": [
                {
                    "name": t.name,
                    "capacity_gb": t.capacity_bytes / 1e9,
                    "bandwidth_gb_s": t.bandwidth_bytes_per_sec / 1e9,
                }
                for t in inst.memory.tiers
            ],
            "hbm_bandwidth_gb_s": inst.hbm_bandwidth / 1e9,

            # ── Peak FLOPs ───────────────────────────────────────
            "peak_flops": {
                dt.value: flops
                for dt, flops in cu.peak_flops.items()
            },

            # ── Per-unit resources ───────────────────────────────
            "register_file_kb": cu.register_file_kb,
            "shared_memory_max_kb": cu.shared_memory_max_kb,
            "l1_shared_combined_kb": cu.l1_shared_combined_kb,
            "warp_schedulers_per_unit": cu.warp_schedulers_per_unit,
            "tensor_cores_per_unit": cu.tensor_cores_per_unit,
            "fp32_cores_per_unit": cu.fp32_cores_per_unit,
            "fp64_cores_per_unit": cu.fp64_cores_per_unit,
            "int32_cores_per_unit": cu.int32_cores_per_unit,
            "ldst_units": cu.ldst_units,
            "sfu_units": cu.sfu_units,

            # ── Occupancy ────────────────────────────────────────
            "threads_per_warp": cu.threads_per_warp,
            "max_concurrent_warps": cu.max_concurrent_warps,
            "max_threads_per_unit": cu.max_threads_per_unit,
            "max_thread_blocks_per_unit": cu.max_thread_blocks_per_unit,
            "max_registers_per_thread": cu.max_registers_per_thread,
            "max_registers_per_block": cu.max_registers_per_block,
            "can_concurrent_fp32_int32": cu.can_concurrent_fp32_int32,

            # ── Pipeline (stage names) ───────────────────────────
            "pipeline_stages": [s.name for s in cu.pipeline],
        }
        result.append(entry)

    result.sort(key=lambda hw: _sm_version_sort_key(hw), reverse=True)
    return result


@app.get("/api/hardware/{name}")
def api_get_hardware(name: str) -> Dict[str, Any]:
    """Return detailed info for a single hardware target.

    Includes memory hierarchy, compute-unit architecture (pipeline
    stages, SM resources, occupancy limits, concurrent-execution
    capabilities), and peak performance numbers.
    """
    try:
        cls = get_hardware(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Hardware '{name}' not found")
    inst = cls()
    cu = inst.compute_unit

    # Serialize pipeline stages
    pipeline_stages: List[Dict[str, Any]] = []
    for s in cu.pipeline:
        pipeline_stages.append({
            "name": s.name,
            "latency_cycles": s.latency_cycles,
            "throughput_per_cycle": s.throughput_per_cycle,
            "description": s.description,
        })

    return {
        "name": inst.name,
        "vendor": inst.vendor,
        "description": inst.description,
        "architecture": getattr(inst, "architecture", ""),
        "sm_version": getattr(inst, "sm_version", ""),
        # ── Memory hierarchy ─────────────────────────────────
        "memory_tiers": [
            {
                "name": t.name,
                "capacity_gb": t.capacity_bytes / 1e9,
                "bandwidth_gb_s": t.bandwidth_bytes_per_sec / 1e9,
                "capacity_bytes": t.capacity_bytes,
                "bandwidth_bytes_per_sec": t.bandwidth_bytes_per_sec,
            }
            for t in inst.memory.tiers
        ],
        "can_overlap_with_compute": list(inst.memory.can_overlap_with_compute),
        "hbm_bandwidth_gb_s": inst.hbm_bandwidth / 1e9,
        # ── Compute unit ─────────────────────────────────────
        "compute_unit": {
            "name": cu.name,
            "count": cu.count,
            "clock_mhz": cu.clock_mhz,
            "peak_flops": {
                dt.value: flops
                for dt, flops in cu.peak_flops.items()
            },
            "max_concurrent_warps": cu.max_concurrent_warps,
            # Per-unit memory resources
            "register_file_kb": cu.register_file_kb,
            "shared_memory_max_kb": cu.shared_memory_max_kb,
            "l1_shared_combined_kb": cu.l1_shared_combined_kb,
            # Per-unit execution resources
            "warp_schedulers_per_unit": cu.warp_schedulers_per_unit,
            "tensor_cores_per_unit": cu.tensor_cores_per_unit,
            "fp32_cores_per_unit": cu.fp32_cores_per_unit,
            "fp64_cores_per_unit": cu.fp64_cores_per_unit,
            "int32_cores_per_unit": cu.int32_cores_per_unit,
            "ldst_units": cu.ldst_units,
            "sfu_units": cu.sfu_units,
            # Threading / occupancy
            "threads_per_warp": cu.threads_per_warp,
            "max_threads_per_unit": cu.max_threads_per_unit,
            "max_thread_blocks_per_unit": cu.max_thread_blocks_per_unit,
            "max_registers_per_thread": cu.max_registers_per_thread,
            "max_registers_per_block": cu.max_registers_per_block,
            # Parallel execution
            "can_concurrent_fp32_int32": cu.can_concurrent_fp32_int32,
            # Pipeline stages
            "pipeline": pipeline_stages,
        },
    }


@app.post("/api/analyze")
def api_analyze(body: Dict[str, Any]) -> Dict[str, Any]:
    """Run a SOL analysis.

    Expected body::

        {
            "operator": "matmul",
            "hardware": "a100",
            "dtype": "fp16",
            "mode": "hierarchy",
            "dims": {"M": 4096, "N": 4096, "K": 4096}
        }
    """
    operator_name = body.get("operator")
    hardware_name = body.get("hardware")
    dtype_str = body.get("dtype", "fp16")
    mode_str = body.get("mode", "hierarchy")
    dims = body.get("dims", {})
    pipeline_config_dict = body.get("pipeline_config", None)

    if not operator_name:
        raise HTTPException(status_code=400, detail="Missing 'operator'")
    if not hardware_name:
        raise HTTPException(status_code=400, detail="Missing 'hardware'")

    try:
        op_cls = get_operator(operator_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Operator '{operator_name}' not found")

    try:
        hw_cls = get_hardware(hardware_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Hardware '{hardware_name}' not found")

    try:
        dtype = DataType(dtype_str.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown dtype '{dtype_str}'")

    try:
        mode = AnalysisMode(mode_str.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown mode '{mode_str}'")

    # Parse pipeline_config for pipeline mode
    pipeline_config = None
    if pipeline_config_dict and mode == AnalysisMode.PIPELINE:
        pipeline_config = PipelineConfig(
            async_copy_enabled=pipeline_config_dict.get("async_copy_enabled", True),
            sparsity_2_4_enabled=pipeline_config_dict.get("sparsity_2_4_enabled", False),
        )

    op = op_cls()
    hw = hw_cls()

    analyzer = Analyzer()
    result = analyzer.analyze(op, hw, dtype, mode=mode, pipeline_config=pipeline_config, **dims)

    return _result_to_dict(result)


# ---------------------------------------------------------------------------
# Static files (web UI)
# ---------------------------------------------------------------------------

WEB_DIR = os.path.join(os.path.dirname(__file__), "..", "web")

if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def serve_index():
    """Serve the main web UI."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"message": "Web UI not found. Run from project root."}
