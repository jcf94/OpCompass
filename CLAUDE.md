# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable)
pip install -e ".[dev]"

# CLI
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

### Auto-discovery registry

`opcompass/registry.py` uses `pkgutil.walk_packages` to scan `operators/` and `hardware/` for classes inheriting from `Operator` / `Hardware`. Adding a new operator or hardware target requires only dropping a `.py` file into the right directory — no manual registration. Each class must have a `name` class attribute (used as the lookup key).

### SOL analysis flow

1. **Operator** provides `compute_flops(**dims)` → int and `compute_io_bytes(dtype, **dims)` → (read_bytes, write_bytes)
2. **Hardware** provides `peak_flops` dict, `MemoryHierarchy` tiers, and optionally `PipelineStage[]`
3. `Analyzer.analyze()` in `engine/analyzer.py` orchestrates three timing estimates:
   - `_estimate_memory_time` — bytes / bandwidth through the slowest memory tier
   - `_estimate_compute_time` — FLOPs / peak with a utilization factor
   - Synthesis: if `can_overlap_with_compute` is set, `max(compute, read, write)`; otherwise sum.

Three analysis modes (`AnalysisMode` enum):
- `hierarchy_roofline` — roofline using multi-tier memory model (first/slowest tier for the primary estimate)
- `pipeline` — stage-level scheduling via `_match_stage` heuristics (see `engine/pipeline_model.py`). Requires operators to provide `SubOp` breakdowns via `get_ops_breakdown()`.
- `solar` — uses the vendored `3rdparty/SOLAR` toolkit to extract a PyTorch computation graph (via torchview), convert it to einsum notation, and run hardware-independent analysis + roofline perf prediction. Requires `torch`, `torchview`, `pyyaml`. Each operator must implement `get_solar_model_source(dtype, **dims) → str` to generate a SOLAR-compatible model file. See `engine/solar_analyzer.py`.

### Data model relationships (`opcompass/models.py`)

- `ComputeUnit` holds chip-wide peak FLOPS, per-SM resource counts (register file KB, shared memory max, core counts, warp schedulers), and `PipelineStage[]`.
- `MemoryHierarchy` holds `MemoryTier[]` (chip-level: HBM → L2) and `can_overlap_with_compute` (set of tier names that allow async copy).
- `PipelineStage` has `latency_cycles` + `throughput_per_cycle` (bytes/clock for memory stages, FMA ops/clock for compute stages). The pipeline model categorizes stages by name substring: `read`/`load`/`write`/`store` → memory work units; `mma`/`alu` → compute work units.
- `SubOp` provides a fine-grained decomposition for pipeline analysis with `flops`, `read_bytes`, `write_bytes`, and `depends_on` (dependency graph).

### Web frontend

Two pages (tab navigation in `index.html`):
- **Analyze** — config panel + SOL results with Chart.js charts (breakdown bar, roofline scatter)
- **Hardware** — SVG pipeline diagram (`js/hardware.js`) rendered from `/api/hardware/{name}` data, SM specs grid, memory hierarchy stack

The backend (`server.py`) is FastAPI. Static files are served from `web/` at `/static/`. The hardware detail endpoint returns all `ComputeUnit` fields including pipeline stages and per-SM resource counts.

### Class inheritance for extensions

- New operator: subclass `Operator` (in `operators/base.py`). Override `compute_flops`, `compute_io_bytes`. Optionally override `get_ops_breakdown` for pipeline mode, `get_tiling_strategy` for tiling suggestions, `get_solar_model_source` for solar mode (generates a SOLAR-compatible PyTorch model file).
- New hardware: subclass `Hardware` (in `hardware/base.py`). Set `memory: MemoryHierarchy` and `compute_unit: ComputeUnit`. For detailed pipeline modeling, populate `ComputeUnit.pipeline` with all memory/compute stages and fill in the optional SM resource fields (`register_file_kb`, `shared_memory_max_kb`, `tensor_cores_per_unit`, `can_concurrent_fp32_int32`, etc.). For solar mode support, add a corresponding arch config YAML in `opcompass/configs/solar_arch/`.

### SOLAR arch configs

`opcompass/configs/solar_arch/` holds GPU architecture definitions in SOLAR's YAML format (DRAM/SRAM capacity and bandwidth per cycle, frequency, MAC throughput per dtype). The mapping from OpCompass hardware names to config files is in `engine/solar_analyzer.py` (`HARDWARE_TO_SOLAR_ARCH`).
