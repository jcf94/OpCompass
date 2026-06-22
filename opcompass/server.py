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
from opcompass.models import AnalysisMode, DataType
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


@app.get("/api/hardware")
def api_list_hardware() -> List[Dict[str, Any]]:
    """Return all registered hardware targets with key specs."""
    hw = discover_hardware()
    result: List[Dict[str, Any]] = []
    for name, cls in sorted(hw.items()):
        inst = cls()
        result.append({
            "name": name,
            "vendor": inst.vendor,
            "description": inst.description,
            "num_sms": inst.num_compute_units,
            "clock_mhz": inst.compute_unit.clock_mhz,
            "hbm_bandwidth_gb_s": inst.hbm_bandwidth / 1e9,
            "peak_flops": {
                dt.value: flops
                for dt, flops in inst.compute_unit.peak_flops.items()
            },
        })
    return result


@app.get("/api/hardware/{name}")
def api_get_hardware(name: str) -> Dict[str, Any]:
    """Return detailed info for a single hardware target."""
    try:
        cls = get_hardware(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Hardware '{name}' not found")
    inst = cls()
    return {
        "name": inst.name,
        "vendor": inst.vendor,
        "description": inst.description,
        "memory_tiers": [
            {
                "name": t.name,
                "capacity_gb": t.capacity_bytes / 1e9,
                "bandwidth_gb_s": t.bandwidth_bytes_per_sec / 1e9,
            }
            for t in inst.memory.tiers
        ],
        "compute_unit": {
            "name": inst.compute_unit.name,
            "count": inst.compute_unit.count,
            "clock_mhz": inst.compute_unit.clock_mhz,
            "peak_flops": {
                dt.value: flops
                for dt, flops in inst.compute_unit.peak_flops.items()
            },
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

    op = op_cls()
    hw = hw_cls()

    analyzer = Analyzer()
    result = analyzer.analyze(op, hw, dtype, mode=mode, **dims)

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
