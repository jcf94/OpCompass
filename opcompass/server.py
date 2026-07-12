"""FastAPI server for OpCompass — serves the web UI and REST API.

Start with::

    uvicorn opcompass.server:app --reload
"""

# NOTE: This file uses Python 3.8-compatible typing (no 'from __future__ import annotations')
# because FastAPI/Pydantic needs to evaluate type annotations at runtime.

import os
import sys
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from opcompass.api_models import AnalyzeRequest, AnalyzeResponse, ErrorResponse
from opcompass import __version__

from opcompass.registry import (
    discover_hardware,
    discover_operators,
    get_hardware,
    get_operator,
)
from opcompass.models import (
    AnalysisMode, BackendUnavailableError, DataType, InfeasibleCandidateError,
    NonFiniteResultError, OperatorValidationError, PipelineConfig,
    UnsupportedAnalysisError, UnsupportedDataTypeError,
)
from opcompass.engine.analyzer import Analyzer
from opcompass.engine.result import _result_to_dict

app = FastAPI(
    title="OpCompass API",
    description="SOL theoretical peak performance estimator for GPU operators",
    version=__version__,
)


@app.exception_handler(RequestValidationError)
async def api_request_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a stable code around FastAPI/Pydantic validation details."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "invalid_api_request",
                "issues": exc.errors(),
            }
        },
    )


@app.exception_handler(Exception)
async def api_internal_error(request: Request, exc: Exception) -> JSONResponse:
    """Prevent uncaught failures from leaking implementation details."""
    return JSONResponse(
        status_code=500,
        content={"detail": {
            "code": "internal_error",
            "message": "An unexpected internal error occurred.",
        }},
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
        spec = inst.spec
        result.append({
            "name": name,
            "description": inst.description,
            "param_dims": inst.param_dims,
            "capabilities": inst.mode_capabilities(),
            "parameter_spec": [
                {
                    "name": parameter.name,
                    "aliases": list(parameter.aliases),
                    "type": parameter.value_type.__name__,
                    "required": parameter.required,
                    "default": parameter.default,
                    "minimum": parameter.minimum,
                    "multiple_of": parameter.multiple_of,
                    "kind": parameter.kind.value,
                    "description": parameter.description,
                }
                for parameter in spec.parameters
            ],
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
            "spec_version": inst.spec_version,
            "provenance_status": inst.provenance_status,
            "provenance": inst.provenance(),
            "architecture": getattr(inst, "architecture", ""),
            "sm_version": getattr(inst, "sm_version", ""),
            "spec_version": inst.spec_version,
            "provenance_status": inst.provenance_status,
            "provenance": inst.provenance(),
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
                    "capacity_bytes": t.capacity_bytes,
                    "bandwidth_gb_s": t.bandwidth_bytes_per_sec / 1e9,
                    "bandwidth_bytes_per_sec": t.bandwidth_bytes_per_sec,
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
        "spec_version": inst.spec_version,
        "provenance_status": inst.provenance_status,
        "provenance": inst.provenance(),
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


@app.get("/api/tile-constraints")
def api_tile_constraints(operator: str, hardware: str, dtype: str = "fp16") -> Dict[str, Any]:
    """Return pipeline tile granularity for an operator/hardware/dtype."""
    try:
        op_cls = get_operator(operator)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Operator '{operator}' not found")

    try:
        hw_cls = get_hardware(hardware)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Hardware '{hardware}' not found")

    try:
        resolved_dtype = DataType(dtype.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown dtype '{dtype}'")

    return op_cls().get_tile_constraints(hw_cls(), resolved_dtype)


@app.post(
    "/api/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def api_analyze(body: AnalyzeRequest) -> Dict[str, Any]:
    """Run a SOL analysis.

    Expected body::

        {
            "operator": "matmul",
            "hardware": "a100",
            "dtype": "fp16",
            "mode": "hierarchy_roofline",
            "dims": {"M": 4096, "N": 4096, "K": 4096}
        }
    """
    # Keep direct Python callers useful while FastAPI itself supplies a model.
    if isinstance(body, dict):
        try:
            body = AnalyzeRequest.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail={
                "code": "invalid_api_request",
                "issues": exc.errors(include_url=False),
            })

    operator_name = body.operator
    hardware_name = body.hardware
    dims = body.dims

    try:
        op_cls = get_operator(operator_name)
    except KeyError:
        raise HTTPException(status_code=404, detail={
            "code": "unknown_operator",
            "operator": operator_name,
            "message": f"Operator '{operator_name}' not found",
        })

    try:
        hw_cls = get_hardware(hardware_name)
    except KeyError:
        raise HTTPException(status_code=404, detail={
            "code": "unknown_hardware",
            "hardware": hardware_name,
            "message": f"Hardware '{hardware_name}' not found",
        })

    dtype = body.dtype
    mode = body.mode

    # Parse pipeline_config for pipeline mode
    pipeline_config = None
    if body.pipeline_config and mode == AnalysisMode.PIPELINE:
        pipeline_config_dict = body.pipeline_config
        pipeline_config = PipelineConfig(
            **pipeline_config_dict.model_dump()
        )

    op = op_cls()
    hw = hw_cls()

    analyzer = Analyzer()
    try:
        result = analyzer.analyze(
            op, hw, dtype, mode=mode, pipeline_config=pipeline_config,
            strict=body.strict, **dims
        )
    except OperatorValidationError as exc:
        raise HTTPException(status_code=422, detail={
            "code": exc.code,
            "operator": exc.operator,
            "issues": exc.issues,
        })
    except UnsupportedAnalysisError as exc:
        raise HTTPException(status_code=422, detail={
            "code": exc.code,
            "operator": exc.operator,
            "requested_mode": exc.mode.value,
            "message": exc.message,
        })
    except NonFiniteResultError as exc:
        raise HTTPException(status_code=500, detail={
            "code": exc.code,
            "field": exc.field_path,
            "message": str(exc),
        })
    except BackendUnavailableError as exc:
        raise HTTPException(status_code=503, detail={
            "code": exc.code,
            "backend": exc.backend,
            "missing_dependencies": exc.missing_dependencies,
            "message": str(exc),
        })
    except InfeasibleCandidateError as exc:
        raise HTTPException(status_code=422, detail={
            "code": exc.code,
            "message": exc.message,
        })
    except UnsupportedDataTypeError as exc:
        raise HTTPException(status_code=422, detail={
            "code": exc.code,
            "hardware": exc.hardware,
            "dtype": exc.dtype.value,
            "supported_dtypes": [item.value for item in exc.supported],
            "message": str(exc),
        })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={
            "code": "invalid_analysis_request",
            "message": str(exc),
        })

    return _result_to_dict(
        result, include_trace=body.include_trace, trace_limit=body.trace_limit
    )


# ---------------------------------------------------------------------------
# Static files (web UI)
# ---------------------------------------------------------------------------

_SOURCE_WEB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
_INSTALLED_WEB_DIR = os.path.join(sys.prefix, "opcompass", "web")
WEB_DIR = next(
    (path for path in (_SOURCE_WEB_DIR, _INSTALLED_WEB_DIR) if os.path.isdir(path)),
    _SOURCE_WEB_DIR,
)

if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def serve_index():
    """Serve the main web UI."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"message": "Web UI not found. Run from project root."}
