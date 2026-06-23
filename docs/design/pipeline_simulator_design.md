# Pipeline Simulator Design Analysis

## 1. Overview

The pipeline simulator models **GPU kernel performance at the CTA (thread block) level** using a cycle-accurate, three-phase software pipelining approach. It targets the CUTLASS-style tiled GEMM execution pattern, where a large K dimension is split into slices, and memory loads for slice `k` are overlapped with computation (MMA) for slice `k-1`.

The primary entry point is `schedule_pipeline()` in `opcompass/engine/pipeline_model.py`, which takes a decomposed set of sub-operations and places them on a timeline with explicit prologue/steady-state/epilogue phases.

---

## 2. System Architecture

### 2.1 Component Diagram

```
┌──────────────────────────┐    ┌────────────────────────┐
│     Operator (matmul)    │    │    Hardware (GPU)      │
│  · get_ops_breakdown()   │    │  · ComputeUnit         │
│  · get_tiling_strategy() │    │    · pipeline[]        │
│  · get_tile_constraints()│    │    · peak_flops        │
└──────────┬───────────────┘    │    · SM resource limits│
           │                    │  · MemoryHierarchy     │
           │ SubOp[]            │    · tiers[]           │
           │                    │    · can_overlap_with..│
           ▼                    └─────────┬──────────────┘
┌─────────────────────────┐               │
│   PipelineConfig        │               │
│  · async_copy_enabled   │               │
│  · sparsity_2_4_enabled │               │
│  · block_m/n/k (opt)    │               │
└──────────┬──────────────┘               │
           │                              │
           ▼                              ▼
┌──────────────────────────────────────────────────────┐
│              schedule_pipeline()                     │
│         (pipeline_model.py:236-260)                  │
│                                                      │
│  1. Separate recurring vs epilogue sub-ops           │
│  2. Compute durations (full vs throughput-only)      │
│  3. _reschedule_solid(): 3-phase placement           │
│     · Prologue  (k=0):          sequential, full dur │
│     · Steady State (k=1..N-1):  overlapped, tp-only  │
│     · Epilogue:                 sequential, full dur │
│  4. Compute aggregate metrics (grid, wave, total)    │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│              PipelineSchedule                        │
│  · sub_ops: ScheduledSubOp[] (cycle positions)       │
│  · total_cycles_per_block, total_time_s              │
│  · wave_count, grid_size, num_k_iterations           │
│  · bottleneck_stage, prologue/steady/epilogue cycles │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│           Analyzer._analyze_pipeline()               │
│         (analyzer.py:115-243)                        │
│                                                      │
│  · Derives stage_breakdown (read/compute/write)      │
│  · Computes SOL time, SOL TFLOPS                     │
│  · Assembles AnalysisResult with pipeline metadata   │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
                  Web Frontend
        (Gantt chart, stats, tiling, guidance)
```

### 2.2 Key Data Structures

**SubOp** (`models.py:127`): A micro-operation in the kernel with explicit FLOPS, byte counts, pipeline stage mapping, DAG dependencies, and a recurring flag.

```python
@dataclass
class SubOp:
    name: str                   # e.g. "async_copy_load_A"
    flops: int                  # total FLOPs
    read_bytes: int             # bytes read from memory
    write_bytes: int            # bytes written to memory
    depends_on: list[str]       # names of dependent sub-ops (DAG)
    pipeline_stage: str         # maps to PipelineStage.name
    is_recurring: bool          # True = in K-loop; False = epilogue only
```

**PipelineStage** (`models.py:62`): A hardware pipeline stage with fixed latency and throughput.

```python
@dataclass
class PipelineStage:
    name: str                   # e.g. "async_copy_load", "mma"
    latency_cycles: int         # fixed pipeline depth latency
    throughput_per_cycle: float # work units per cycle per SM
    description: str
```

**ScheduledSubOp** (`models.py:140`): A sub-op placed at specific cycle positions on the timeline.

```python
@dataclass
class ScheduledSubOp:
    name: str                   # iteration-suffixed, e.g. "mma_k3"
    pipeline_stage: str         # mapped stage name
    start_cycle: int
    end_cycle: int
    duration_cycles: int
    work_units: int             # bytes (memory) or FMA ops (compute)
    iteration: int              # K-slice index; -1 for epilogue
```

**PipelineSchedule** (`models.py:153`): The complete timeline plus aggregate metrics.

```python
@dataclass
class PipelineSchedule:
    sub_ops: list[ScheduledSubOp]
    total_cycles_per_block: int
    total_time_s: float
    wave_count: int
    grid_size: int
    num_k_iterations: int
    bottleneck_stage: str
    per_iteration_cycles: int
    prologue_cycles: int
    epilogue_cycles: int
```

### 2.3 Stage Categorization

The model categorizes pipeline stages by **substring matching on stage name** (lowercased). This is a heuristic that appears in several places:

| Category | Name keywords | Work unit | Examples |
|----------|--------------|-----------|----------|
| Memory | `read`, `load`, `write`, `store`, `copy` | bytes | `async_copy_load`, `global_read`, `shared_load`, `shared_store`, `global_write` |
| Compute | `mma`, `alu`, `compute` | FMA count (flops/2) | `mma`, `fma_alu` |
| Fallback | anything else | bytes + flops | (none currently) |

This categorization is used for:
1. **Work unit calculation** (`_work_units`): What constitutes one unit of work?
2. **Throughput caps** (`_stage_throughput`): Memory stages capped by HBM bandwidth; compute stages capped by peak FLOPS; MMA stages doubled for sparsity.
3. **Scheduling order** (`_reschedule_solid`): Groups into load_subs → shared_load_subs → mma_subs.
4. **Stage breakdown** (`_analyze_pipeline`): Consolidated into read/compute/write buckets.

---

## 3. Algorithm Design

### 3.1 Three-Phase Software Pipelining

The core algorithm is in `_reschedule_solid()` (pipeline_model.py:264-376). It models a CUTLASS-style main loop where iteration `k` performs:

```
load[k] → shared_load[k] → mma[k]   (conceptually)
```

But with **software pipelining**, the loads for iteration `k` are issued while MMA for iteration `k-1` is still executing:

```
         ┌──────────────────────────────────────────────┐
Prologue │ load[0]──shared_load[0]──mma[0]              │  (k=0, sequential)
         ├──────────────────────────────────────────────┤
Steady   │        load[1]──shared_load[1]──mma[1]       │  (k=1, overlaps mma[0])
State    │               load[2]──shared_load[2]──mma[2]│  (k=2, overlaps mma[1])
         │                      ...                     │
         ├──────────────────────────────────────────────┤
Epilogue │                              store_C─write_C │  (after last mma)
         └──────────────────────────────────────────────┘
```

**Prologue** (k=0): All sub-ops run sequentially with **full duration** (latency + throughput). No overlap is possible because there's no preceding MMA to overlap with.

**Steady State** (k=1..N-1): With async copy enabled:
- `load[k]` starts at the same cycle as `mma[k-1].start` (overlaps with previous compute).
- `shared_load[k]` starts after `load[k]` finishes (data dependency: shared memory must be populated).
- `mma[k]` starts after **both** `shared_load[k]` and `mma[k-1]` complete.
- Uses **throughput-only duration** (latency is hidden by the pipeline).
- Per-iteration advance: `max(load_tp + shared_tp, mma_tp)` — whichever chain is slower.

Without async copy: each iteration is fully sequential (same as prologue).

**Epilogue**: All non-recurring sub-ops (store + write) run sequentially with full duration after the last MMA.

### 3.2 Duration Calculation

Two duration variants for each sub-op:

| Variant | Formula | When Used |
|---------|---------|-----------|
| Throughput-only | `ceil(work_units / stage_throughput)` | Steady state |
| Full | `stage.latency_cycles + throughput_duration` | Prologue, Epilogue |

This separation is critical: an async copy might have 300 cycles of latency, but that latency is paid once during the prologue and hidden during the steady state. Without this distinction, the model would overcount latency by `300 × (num_k_iterations - 1)` cycles — the main source of error in earlier versions.

### 3.3 Throughput Calculation

Throughput is computed per sub-op, per stage by `_stage_throughput()`:

1. **Base**: `stage.throughput_per_cycle` from hardware model.
2. **HBM cap**: For stages named `global` or `async_copy`: `min(throughput, hbm_bw_per_cycle_per_sm)`.
3. **Compute cap**: For stages named `mma`/`alu`/`compute`: `peak_flops / (2 × clock × SM_count)` = FMA/cycle/SM.
4. **Sparsity boost**: For MMA stages when `sparsity_2_4_enabled`: `throughput × 2`.

### 3.4 Occupancy and Wave Count

After the CTA-level timeline is built, the model estimates chip-level performance:

```
grid_size = ceil(M / block_m) × ceil(N / block_n)
```

**Resident blocks per SM** = minimum of four limits:
1. `max_thread_blocks_per_unit` (hardware: 32 for NVIDIA)
2. `max_threads_per_unit / threads_per_block` (thread limit)
3. `max_concurrent_warps / num_warps_per_block` (warp limit)
4. `shared_memory_max_kb × 1024 / shared_memory_per_block` (SMEM limit)

**Wave count**: `ceil(grid_size / (SM_count × resident_blocks_per_sm))`

Multiple waves mean blocks must wait for resources, extending total time.

**Total time**: `max(total_cycles, max(stage_times) × blocks_per_sm) × clock_period`

This accounts for both the CTA-level critical path and per-SM resource contention.

### 3.5 Sub-Op Decomposition (Matmul)

The matmul operator's `get_ops_breakdown()` generates 6-8 sub-ops depending on architecture:

| Sub-Op | Pipeline Stage | Recurring | Purpose |
|--------|---------------|-----------|---------|
| async_copy_load_A | async_copy_load | Yes | Load A tile from HBM to shared memory |
| async_copy_load_B | async_copy_load | Yes | Load B tile from HBM to shared memory |
| shared_load_A | shared_load | Yes | Load A from shared memory to registers |
| shared_load_B | shared_load | Yes | Load B from shared memory to registers |
| mma | mma | Yes | Tensor core matrix multiply-accumulate |
| tmem_load_C | tmem_load | No (Blackwell) | Load accumulator from TMEM to registers |
| shared_store_C | shared_store | No | Store result from registers to shared memory |
| async_copy_store_C | async_copy_store | No (Hopper+) | TMA store from shared memory to HBM |
| global_write_C | global_write | No (Ampere) | Write result from shared memory to HBM |

Work calculations:
- **Load A**: `block_M × block_K × dtype_bytes` bytes
- **Load B**: `block_K × block_N × dtype_bytes` bytes
- **MMA**: `2 × block_M × block_N × block_K` FLOPs (M×N×K FMAs, each = 2 FLOPs)
- **Store/Write**: `block_M × block_N × dtype_bytes` bytes

### 3.6 Tiling Strategy

`get_tiling_strategy()` in matmul.py selects architecture-aware defaults:

| Architecture | FP16/BF16 | TF32 | FP32 |
|-------------|-----------|------|------|
| Ampere (A100) | 128×128×32 | 128×128×16 | 64×64×16 |
| Hopper (H100) | 128×128×64 | — | — |
| Blackwell (B200) | 256×128×64 | 128×128×64 | 128×128×32 |
| Fallback | 64×64×32 | — | — |

Shared memory validation: computes `2 × (bM×bK + bK×bN) × bytesize + bM×bN × bytesize` (double-buffered A/B tiles + C accumulator). If the user provides tile overrides that exceed the SMEM limit, an error is raised. For auto-tiling, `block_K` is iteratively halved until the SMEM budget fits.

User overrides are validated against Tensor Core instruction constraints (e.g., FP16 MMA requires M multiple of 16, N multiple of 8, K multiple of 16).

---

## 4. Key Design Decisions

### 4.1 Name-based stage matching (string heuristics over explicit types)

Stages are classified by substring matching on `stage.name.lower()` rather than by an explicit enum or type field. This is simple and flexible (new stages can be added to hardware without updating the scheduler), but fragile — a stage named `compute_read` would match both memory and compute keywords.

### 4.2 Implicit dependency ordering (stage category grouping over DAG traversal)

The scheduler orders sub-ops by their stage category (load → shared_load → mma) rather than by traversing the `depends_on` DAG. The dependency graph exists in the data model but is not used by the scheduler. This works because CUTLASS matmul sub-ops form a simple linear chain per iteration, but would fail for operators with non-linear dependency structures.

### 4.3 Latency/throughput decomposition

The split between full duration (latency + throughput, for prologue/epilogue) and throughput-only duration (for steady state) is the model's most important assumption. It correctly captures that pipeline depth latency is paid once during fill/drain and hidden during steady state.

### 4.4 Single-CTA perspective with occupancy scaling

The model simulates one CTA's timeline and then scales to chip level via occupancy and wave count. It does not model inter-CTA interference (shared cache contention, DRAM bank conflicts between CTAs, wave scheduling jitter).

### 4.5 Double-buffered shared memory assumption

Shared memory sizing assumes exactly two buffers for A and B tiles (ping-pong). This is hardcoded in the tiling strategy and not configurable.

### 4.6 Fixed 4-warps-per-block assumption

`num_warps_per_block` is hardcoded to 4 in `get_tiling_strategy()`. Real CUTLASS kernels can use different warp counts (e.g., 8 warps for larger tiles on Hopper).

---

## 5. Shortcomings and Limitations

### 5.1 Only One Operator Supported

Pipeline analysis is fully implemented **only for matmul**. All other 6 operators (convolution, elementwise, flash_attention, layernorm, reduction, etc.) fall back to a simple non-pipeline roofline model because they don't implement `get_ops_breakdown()`.

**Impact**: The pipeline model cannot analyze most real workloads. Convolution and flash attention are natural next candidates.

### 5.2 Implicit Dependency Ordering (DAG Not Used)

The `SubOp.depends_on` field is declared but **never read by the scheduler**. The scheduler relies entirely on name-based stage categorization to order sub-ops (load → shared_load → mma). This works for matmul's linear chain but:

- Would fail for operators with branching dependencies (e.g., multiple independent compute streams).
- Would fail for operators where the dependency structure doesn't match the stage category order.
- Cannot model fused operators where sub-ops from different logical kernels share stages.

**Mitigation**: The dependency graph should be traversed to determine the actual scheduling order, with stage categories used only for throughput/latency lookup.

### 5.3 Fixed Three-Group Stage Ordering in Steady State

`_reschedule_solid()` hardcodes three groups: `load_subs` → `shared_load_subs` → `mma_subs`. This assumes a specific CUTLASS matmul pipeline structure. Other operators may have different pipeline structures:
- Flash attention has a different pattern (Q·K^T → softmax → ×V).
- Convolution may have im2col or Winograd transform stages.
- Reduction has a tree-based pattern with multiple levels.

**Mitigation**: The three-group ordering should be derived from the dependency DAG or from operator-provided metadata, not hardcoded.

### 5.4 No Multi-CTA Interaction Modeling

The model simulates one CTA and scales linearly by wave count. It does not model:
- **L2 cache contention** between CTAs on the same SM or across SMs.
- **DRAM bank conflicts** when multiple CTAs access different HBM partitions.
- **Wave scheduling jitter** — the assumption that all waves take the same time is optimistic.
- **Tail effect** — the last wave may have fewer CTAs than `resident_blocks_per_sm`, leaving SMs underutilized.

**Impact**: Performance estimates are optimistic, especially for memory-bound kernels where cache and DRAM contention dominate.

### 5.5 No Register Pressure Modeling

The model tracks shared memory limits but **not register file limits**. The `ComputeUnit` has `register_file_kb` and `core_counts` fields, but these are not used in occupancy calculations. Register spilling can significantly degrade performance (adding extra memory round-trips), but the model assumes all data fits in registers.

**Impact**: Overly optimistic for register-heavy configurations (large tiles, many warps).

### 5.6 Hardcoded Warp Count

`num_warps_per_block = 4` is fixed. Real CUTLASS kernels use 4-8 warps depending on tile size and architecture. More warps increase occupancy but also increase register pressure.

### 5.7 Name-Based Stage Matching is Fragile

Stage classification relies on substring matching (`"mma" in name`, `"read" in name`). This is error-prone:
- A stage named `mmap` (memory-mapped) would be misclassified as compute.
- Adding a new stage type requires updating all the substring checks scattered across the codebase.
- No compile-time or startup validation that stage names are correctly classified.

**Mitigation**: Use an explicit enum or type tag on `PipelineStage` instead of name heuristics.

### 5.8 No Support for Persistent Kernels / Thread Blocks

The model assumes a traditional grid launch where CTAs retire after completion. It cannot model persistent thread blocks (where CTAs cooperatively process multiple tiles) or techniques like CUTLASS 3.x's persistent kernel design.

### 5.9 Simplified Async Copy Model

The async copy model assumes that loads can start at exactly the same cycle as the previous MMA's start. In reality:
- There is a small scheduling overhead between issuing loads and the previous MMA.
- TMA (Hopper+) has its own hardware pipeline with independent latency characteristics.
- The model doesn't distinguish between `cp.async` (Ampere) and TMA (Hopper/Blackwell) in the scheduling logic — only the throughput values differ.

### 5.10 No Support for Mixed Precision / FP8

The dtype parameter selects one data type at a time. Real kernels (e.g., FP8 input with FP16 accumulation, or INT4 weights with FP16 activations) use mixed precision. The model cannot represent multiple data types within a single kernel.

### 5.11 No Validation Against Real Hardware Measurements

There is no test suite that validates pipeline model predictions against actual GPU measurements (e.g., Nsight Compute profiles). All latency and throughput values are theoretical estimates derived from hardware spec sheets, not calibrated against real kernel performance.

### 5.12 Single-Kernel Focus (No Fusion)

The pipeline model analyzes one operator at a time. Unlike the SOLAR analyzer, it cannot model operator fusion (e.g., matmul + bias + ReLU as a single fused kernel). This limits its applicability to production workloads where fusion is common.

### 5.13 No Consideration of Power / Thermal Throttling

The model uses peak clock speeds and assumes ideal thermal conditions. Real GPUs throttle under sustained load, reducing clock speeds by 5-15%.

### 5.14 Frontend: Limited Interactivity

The Gantt chart is the main visualization but could be improved:
- No tooltip on hover showing sub-op details (work units, duration).
- Zoom/pan is canvas-based and doesn't support touch well.
- No side-by-side comparison of two pipeline schedules (e.g., with/without sparsity).
- The "Performance Guidance" text is hardcoded heuristics, not derived from the actual schedule.

### 5.15 Sub-Op Decomposition is Hand-Written

Each operator's `get_ops_breakdown()` must be manually coded with full knowledge of the target GPU architecture and the CUTLASS implementation pattern. This is labor-intensive and error-prone. An ideal system would derive the decomposition from a higher-level description (e.g., a tensor expression + tiling annotations).

---

## 6. Summary Table

| Aspect | Current State | Limitation |
|--------|--------------|------------|
| Operator coverage | 1/7 (matmul only) | Cannot analyze other ops with pipeline model |
| Dependency tracking | `depends_on` field unused | Hardcoded stage order, fails for non-linear graphs |
| Occupancy model | SMEM + threads + warps + blocks | Missing register pressure |
| Hardware validation | None | No calibration against real measurements |
| Async copy modeling | Binary toggle + throughput cap | Simplified, doesn't distinguish cp.async vs TMA behavior |
| Multi-CTA | Linear wave scaling | No cache contention, DRAM bank conflicts, tail effects |
| Operator fusion | Not supported | Single-operator only (unlike SOLAR) |
| Stage classification | Substring heuristics | Fragile, no type system |
| Warp count | Hardcoded to 4 | Not configurable |
| Precision | Single dtype per analysis | No mixed precision support |
