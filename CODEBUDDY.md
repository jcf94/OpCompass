# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

## Commands

```bash
# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# CLI usage
compass list operators
compass list hardware
compass analyze matmul --hardware a100 --dtype fp16 --M 4096 --N 4096 --K 4096
compass sweep matmul --hardware a100,h100 --M 1024,2048,4096 --K 1024,2048,4096

# Web UI
uvicorn opcompass.server:app --reload

# Tests (pytest)
pytest
pytest tests/test_hardware/
pytest tests/test_operators/test_matmul.py -v
```

## Architecture

OpCompass is a SOL (Speed of Light) theoretical peak performance estimator for GPU operators. Given an operator, input shapes, and a hardware target, it estimates the theoretical lower-bound execution time.

### Auto-discovery registry

`opcompass/registry.py` uses `pkgutil.walk_packages` to scan `operators/` and `hardware/` for classes inheriting from `Operator` / `Hardware`. Adding a new operator or hardware target requires only dropping a `.py` file into the right directory — no manual registration. Each class must have a `name` class attribute (used as the lookup key). Intermediate base classes with empty `name` are skipped by the registry (e.g., `NvidiaAmpere`).

### SOL analysis flow

1. **Operator** provides `compute_flops(**dims)` → int and `compute_io_bytes(dtype, **dims)` → (read_bytes, write_bytes)
2. **Hardware** provides `peak_flops` dict, `MemoryHierarchy` tiers, and optionally `PipelineStage[]`
3. `Analyzer.analyze()` in `engine/analyzer.py` orchestrates three timing estimates:
   - `_estimate_memory_time` — bytes / bandwidth through the slowest memory tier
   - `_estimate_compute_time` — FLOPs / peak with a utilization factor
   - Synthesis: if `can_overlap_with_compute` is set, `max(compute, read, write)`; otherwise sum.

Three analysis modes (`AnalysisMode` enum):
- `simple` — roofline using HBM bandwidth only
- `hierarchy` — multi-tier memory model (default; first/slowest tier for primary estimate)
- `pipeline` — stage-level scheduling via `_match_stage` heuristics in `engine/pipeline_model.py`. Requires operators to provide `SubOp` breakdowns via `get_ops_breakdown()`.

### Data model relationships (`opcompass/models.py`)

- `ComputeUnit` holds chip-wide peak FLOPS, per-SM resource counts (register file KB, shared memory max, core counts, warp schedulers), and `PipelineStage[]`.
- `MemoryHierarchy` holds `MemoryTier[]` (chip-level: HBM → L2) and `can_overlap_with_compute` (set of tier names that allow async copy).
- `PipelineStage` has `latency_cycles` + `throughput_per_cycle`. The pipeline model categorizes stages by name substring: `read`/`load`/`write`/`store` → memory work units; `mma`/`alu` → compute work units.
- `SubOp` provides fine-grained decomposition for pipeline analysis with `flops`, `read_bytes`, `write_bytes`, and `depends_on` (dependency graph).

### Extending the system

- **New operator**: subclass `Operator` (in `operators/base.py`). Override `compute_flops`, `compute_io_bytes`. Optionally override `get_ops_breakdown` for pipeline mode, `get_tiling_strategy` for tiling suggestions.
- **New hardware**: subclass `Hardware` (in `hardware/base.py`). Set `memory: MemoryHierarchy` and `compute_unit: ComputeUnit`. For detailed pipeline modeling, populate `ComputeUnit.pipeline` with all memory/compute stages and fill in the optional SM resource fields.

For architecture-family pattern (multiple SKUs sharing the same microarchitecture): create an intermediate base class with empty `name` (so registry skips it), common SM-level parameters, and a `@classmethod` factory for `ComputeUnit`. Concrete SKUs subclass it and only supply chip-specific values (SM count, clock, peak FLOPs, memory config). See `NvidiaAmpere` / `NvidiaA100` in `nvidia_ampere.py` for the reference implementation.

### Web frontend

Two pages (tab navigation in `index.html`):
- **Analyze** — config panel + SOL results with Chart.js charts (breakdown bar, roofline scatter)
- **Hardware** — SVG pipeline diagram (`js/hardware.js`) rendered from `/api/hardware/{name}` data, SM specs grid, memory hierarchy stack

Backend (`server.py`) is FastAPI. Static files served from `web/` at `/static/`. The hardware detail endpoint returns all `ComputeUnit` fields including pipeline stages and per-SM resource counts. `server.py` does NOT use `from __future__ import annotations` because FastAPI/Pydantic needs runtime type evaluation.
