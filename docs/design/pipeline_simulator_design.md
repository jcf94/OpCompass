# Pipeline Simulator Design Analysis

## 1. Overview

The pipeline simulator models **GPU kernel performance at the CTA (thread block) level** using a cycle-accurate, three-phase software pipelining approach. It targets the CUTLASS-style tiled GEMM execution pattern, where a large K dimension is split into slices, and memory loads for slice `k` are overlapped with computation (MMA) for slice `k-1`.

The primary entry point is `schedule_pipeline()` in `opcompass/engine/pipeline_model.py`, which takes a decomposed set of sub-operations and places them on a timeline with explicit prologue/steady-state/epilogue phases.

The design goal is not to emulate every scoreboard event or instruction issue slot. The goal is to estimate the **best theoretically reachable implementation** for a given `op + shape + hardware` while preserving the hardware facts that materially affect performance:

- CTA tile shape, instruction granularity, and K-loop structure.
- On-SM pipeline resources: async copy / TMA, shared-memory load/store, tensor cores or CUDA cores, epilogue writeback.
- Latency hiding by software pipelining and resident CTAs.
- Resource sharing across CTAs: more resident CTAs hide latency but do not multiply one SM's tensor-core, load/store, or async-copy throughput.
- Effective HBM traffic after L2 reuse, separated from logical CTA traffic.
- Occupancy limits from threads, warps, shared memory, and eventually registers.

This makes pipeline mode different from a roofline model. Roofline estimates ideal lower bounds from unique tensor IO and peak compute. Pipeline mode should approximate the best real kernel family that could run the operation, including local movement, stage overlap, tile reuse, and epilogue costs.

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
│              pipeline_model.py                       │
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
│              analyzer.py                             │
│                                                      │
│  · Derives stage_breakdown (read/compute/write)      │
│  · Derives pipeline_memory_breakdown                 │
│  · Computes SOL time, SOL TFLOPS                     │
│  · Assembles AnalysisResult with pipeline metadata   │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
                  Web Frontend
        (Gantt chart, stats, tiling, guidance)
```

### 2.2 Key Data Structures

**SubOp** (`models.py:127`): A micro-operation in the kernel with explicit FLOPS, byte counts, optional effective HBM byte counts, pipeline stage mapping, DAG dependencies, and a recurring flag.

```python
@dataclass
class SubOp:
    name: str                   # e.g. "async_copy_load_A"
    flops: int                  # total FLOPs
    read_bytes: int             # logical CTA bytes read locally
    write_bytes: int            # logical CTA bytes written locally
    effective_hbm_read_bytes: float | None   # HBM bytes after L2 reuse
    effective_hbm_write_bytes: float | None  # HBM bytes after L2 reuse
    depends_on: list[str]       # names of dependent sub-ops (DAG)
    pipeline_stage: str         # maps to PipelineStage.name
    is_recurring: bool          # True = in K-loop; False = epilogue only
```

`read_bytes` and `write_bytes` are intentionally **logical CTA traffic**: bytes copied into shared memory, loaded from shared memory, or moved through epilogue local stages. Effective HBM fields are optional and default to logical bytes. Matmul uses them to model A/B panel reuse through L2; non-matmul operators keep logical-equals-effective behavior until they define operator-specific cache reuse.

**PipelineStage** (`models.py:62`): A hardware pipeline stage with fixed latency and throughput.

```python
@dataclass
class PipelineStage:
    name: str                   # e.g. "async_copy_load", "mma"
    latency_cycles: int         # fixed pipeline depth latency
    throughput_per_cycle: float # work units per cycle per SM
    description: str
```

**PipelineConfig**: Pipeline mode exposes feature toggles and optional user-selected implementation constraints.

```python
@dataclass
class PipelineConfig:
    async_copy_enabled: bool = True
    sparsity_2_4_enabled: bool = False
    block_m: int | None = None
    block_n: int | None = None
    block_k: int | None = None
    stage_count: int | None = None
    warp_count: int | None = None
```

When any tile, stage, or warp override is provided, the analyzer treats it as a forced candidate and either schedules that candidate or raises a validation error.

**PipelineKernelCandidate**: A concrete implementation strategy considered by pipeline mode.

```python
@dataclass
class PipelineKernelCandidate:
    name: str
    block_m: int
    block_n: int
    block_k: int
    warp_count: int
    stage_count: int
    copy_path: str          # global_load, cp_async, tma
    mma_path: str           # mma, wgmma, umma, cuda_fma
    scheduling: str         # standard, warp_specialized, persistent
    cta_order: str          # row_major, column_major, swizzled
    rejection_reason: str
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
1. **Logical work unit calculation** (`_logical_work_units`): What local pipeline resource is occupied?
2. **Effective HBM work calculation** (`_effective_hbm_work_units`): How much HBM source/sink traffic remains after cache reuse?
3. **Throughput selection** (`_stage_throughput`): Local stage throughput comes from the hardware stage; compute stages are capped by peak FLOPS; MMA stages are doubled for sparsity.
4. **Scheduling order** (`_reschedule_solid`): Groups into load_subs → shared_load_subs → mma_subs.
5. **Stage breakdown** (`_analyze_pipeline`): Consolidated into read/compute/write buckets.

---

## 3. Algorithm Design

### 3.1 Three-Phase Software Pipelining

The core algorithm is in `_reschedule_solid()` (pipeline_model.py:264-376). It models a CUTLASS-style main loop where iteration `k` performs:

```
load[k] → shared_load[k] → mma[k]   (conceptually)
```

But with **software pipelining**, future loads are issued while an earlier MMA is still executing. The prefetch distance is derived from the selected software pipeline depth:

```
prefetch_distance = stage_count - 1
load[k + prefetch_distance] overlaps mma[k]
```

For the common 2-stage case, this reduces to loading iteration `k+1` during `mma[k]`:

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

**Prologue**: The scheduler fills the prefetch window for the first `min(stage_count - 1, num_k_iterations)` K-slices with **full duration** (latency + throughput), then executes `mma[0]` with full duration. A 2-stage kernel prefetches only `k=0`; a 3-stage kernel prefetches `k=0` and `k=1` before the first MMA.

**Steady State** (k=1..N-1): With async copy enabled:
- `load[k + prefetch_distance]` starts at the same cycle as `mma[k].start` when the copy engine is available.
- `shared_load[k]` starts after `load[k]` finishes (data dependency: shared memory must be populated).
- `mma[k]` starts after **both** `shared_load[k]` and `mma[k-1]` complete.
- Uses **throughput-only duration** (latency is hidden by the pipeline).
- Per-iteration advance: `max(load_tp + shared_tp, mma_tp)` — whichever chain is slower.

Without async copy, or with `stage_count=1`, each iteration is fully sequential (same as prologue).

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
2. **Compute cap**: For stages named `mma`/`alu`/`compute`: `peak_flops / (2 × clock × SM_count)` = FMA/cycle/SM.
3. **Sparsity boost**: For MMA stages when `sparsity_2_4_enabled`: `throughput × 2`.
4. **Global/async memory duration**: For stages named `global` or `async_copy`, duration is the maximum of:
   - Local stage time: `logical CTA bytes / stage throughput`.
   - HBM source time: `effective HBM bytes / per-SM HBM bandwidth`.

This split is important. A `cp.async` or TMA engine may need to issue all logical CTA tile bytes into shared memory, but not all of those bytes necessarily come from HBM. If another CTA already pulled the panel into L2, the copy engine still moves the bytes locally while HBM sees only the effective post-reuse traffic.

### 3.4 Logical vs Effective Memory Traffic

Pipeline mode now reports separate memory quantities:

| Field | Meaning | Main use |
|-------|---------|----------|
| `logical_cta_read_bytes` | All bytes read by CTA-local stages, including global/async copies and shared-memory loads | Local pipeline pressure |
| `logical_hbm_read_bytes` | Bytes requested by global/async stages before cache reuse | CTA tiling pressure on global path |
| `effective_hbm_read_bytes` | Bytes sourced from HBM after modeled L2 reuse | HBM bandwidth pressure |
| `unique_tensor_read_bytes` | Operator-level unique input tensor bytes | Roofline/SOLAR comparison |
| `l2_reuse_factor` | `logical_hbm_read_bytes / effective_hbm_read_bytes` | Cache-reuse diagnostic |

For matmul, A and B panels are reused across CTAs in the same K slice:

```
grid_m = ceil(M / block_m)
grid_n = ceil(N / block_n)
unique_A_per_k_slice = M × block_k × dtype_bytes
unique_B_per_k_slice = N × block_k × dtype_bytes
logical_A_per_k_slice = grid_m × grid_n × block_m × block_k × dtype_bytes
logical_B_per_k_slice = grid_m × grid_n × block_k × block_n × dtype_bytes
```

If `unique_A_per_k_slice + unique_B_per_k_slice` fits in L2, effective HBM read approaches unique A+B bytes. If it exceeds L2 capacity, the model smoothly degrades toward logical CTA global traffic. This is intentionally a first-order reuse model: it captures the dominant A/B panel reuse without pretending to model exact CTA launch order, L2 set conflicts, or residency hints.

### 3.5 Occupancy and Wave Count

After the CTA-level timeline is built, the model estimates chip-level performance:

```
grid_size = ceil(M / block_m) × ceil(N / block_n)
```

**Resident blocks per SM** = minimum of six limits:
1. `max_thread_blocks_per_unit` (hardware: 32 for NVIDIA)
2. `max_threads_per_unit / threads_per_block` (thread limit)
3. `max_concurrent_warps / num_warps_per_block` (warp limit)
4. `shared_memory_max_kb × 1024 / shared_memory_per_block` (SMEM limit)
5. `register_file_capacity / registers_per_block` (register file limit)
6. `max_registers_per_block / registers_per_block` (per-block register allocation limit)

**Wave count**: `ceil(grid_size / (SM_count × resident_blocks_per_sm))`

Multiple waves mean blocks must wait for resources, extending total time.

**Total time**: `max(total_cycles, max(stage_times) × blocks_per_sm) × clock_period`

This accounts for both the CTA-level critical path and per-SM resource contention.

### 3.6 Sub-Op Decomposition (Matmul)

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
- **Load A**: `block_M × block_K × dtype_bytes` logical bytes, plus effective HBM bytes after L2 reuse.
- **Load B**: `block_K × block_N × dtype_bytes` logical bytes, plus effective HBM bytes after L2 reuse.
- **MMA**: `2 × block_M × block_N × block_K` FLOPs (M×N×K FMAs, each = 2 FLOPs)
- **Store/Write**: `block_M × block_N × dtype_bytes` bytes

### 3.7 Tiling Strategy

`get_tiling_strategy()` in matmul.py returns a feasible architecture-aware candidate. In pipeline analysis, `Analyzer._analyze_pipeline()` evaluates a small candidate set and selects the feasible candidate with the lowest scheduled time.

| Architecture | FP16/BF16 | TF32 | FP32 |
|-------------|-----------|------|------|
| Ampere (A100) | 128×128×32 | 128×128×16 | 64×64×16 |
| Hopper (H100) | 128×128×64 | — | — |
| Blackwell (B200) | 256×128×64 | 128×128×64 | 128×128×32 |
| Fallback | 64×64×32 | — | — |

Candidate generation covers tile shape, warp count, stage count, copy path, MMA path, scheduling label, and CTA order. Shared memory validation now computes:

```
stage_count × (bM×bK + bK×bN) × bytesize + bM×bN × bytesize
```

The first term is the staged A/B mainloop buffering, and the second term is the epilogue C staging buffer. Register pressure is estimated per thread/block and checked against the hardware register file and per-block allocation limits. If the user provides tile, stage, or warp overrides, the analyzer evaluates only that forced candidate and raises an error if it violates instruction granularity, shared memory, or register constraints.

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

### 4.5 Configurable shared-memory stage count

Shared memory sizing and scheduling are stage-count aware. Pipeline candidates can use different A/B mainloop buffering depths, and `PipelineConfig.stage_count` can force a specific depth.

The scheduler uses `prefetch_distance = stage_count - 1` for async paths. This makes stage count affect prologue fill length, load/MMA overlap distance, total CTA timeline, shared-memory usage, and occupancy. The model still does not include barrier costs or producer/consumer warp specialization for deeper pipelines.

### 4.6 Candidate-selected warp count

`num_warps_per_block` is selected per candidate and can be forced with `PipelineConfig.warp_count`. The v1 candidate set evaluates 4- and 8-warp variants on modern NVIDIA architectures. It does not yet model producer/consumer warp-group specialization beyond candidate metadata.

### 4.7 Best-Implementation Bias

Pipeline mode is intended to estimate a strong implementation, not an arbitrary implementation. Defaults should therefore follow the best common kernel family for the architecture:

- Ampere FP16/BF16 matmul: `cp.async` multistage or double-buffered tensor-core GEMM.
- Hopper matmul: WGMMA + TMA where shapes are large enough to benefit.
- Blackwell matmul: WGMMA/UMMA-style tensor core path with TMA and TMEM-aware epilogue.

When multiple valid implementations exist, the model should eventually evaluate candidates and choose the fastest feasible one rather than relying on a single hardcoded default.

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

### 5.4 Limited Multi-CTA Interaction Modeling

The model simulates one CTA and scales linearly by wave count. It does not model:
- **Exact L2 cache contention** between CTAs on the same SM or across SMs. Matmul has a first-order A/B panel reuse model, but not exact set residency or scheduling order.
- **DRAM bank conflicts** when multiple CTAs access different HBM partitions.
- **Wave scheduling jitter** — the assumption that all waves take the same time is optimistic.
- **Tail effect** — the last wave may have fewer CTAs than `resident_blocks_per_sm`, leaving SMs underutilized.

**Impact**: Performance estimates can be optimistic for memory-bound kernels where cache and DRAM contention dominate, and pessimistic when real kernels use persistent scheduling or swizzled CTA order to improve L2 locality.

### 5.5 Coarse Register Pressure Modeling

The model now estimates registers per thread/block and uses `register_file_kb` plus `max_registers_per_block` in occupancy and candidate rejection. This prevents obviously infeasible large-tile or high-warp candidates.

**Remaining limitation**: The estimate is coarse. It does not model compiler allocation, register reuse, accumulator layout, spilling, or warp-specialized producer/consumer register partitioning.

### 5.6 Limited Warp Count Search

The model can evaluate and force 4- or 8-warp candidates for modern NVIDIA architectures. It does not yet search the broader CUTLASS design space, model warp-group roles, or derive warp count from instruction-level scheduling.

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

### 5.16 Simplified Configurable Pipeline Stage Count

The current model supports configurable `stage_count` for candidate generation, user override, shared-memory sizing, occupancy, and async prefetch distance. Real kernels may use 2, 3, 4, or more stages:

- More stages hide global/TMA latency and smooth producer-consumer bubbles.
- More stages consume shared memory: `stage_count × (A_tile + B_tile)`.
- More stages can reduce resident CTAs, which may hurt occupancy and tail behavior.

The scheduler now models the first-order timing effect with `prefetch_distance = stage_count - 1`. The remaining limitation is that stage depth is still represented as symmetric CTA-level buffering. It does not yet model producer warp-groups, consumer warp-groups, mbarrier costs, or architecture-specific wait-group limits.

### 5.17 No Warp Specialization or Producer/Consumer Scheduling

The scheduler treats each CTA as a single stream of load → shared_load → MMA work. Hopper and Blackwell kernels often use warp specialization:

- Producer warp-groups issue TMA loads and barriers.
- Consumer warp-groups issue WGMMA.
- Epilogue warp-groups may overlap stores with late compute.

This affects occupancy, barrier costs, register allocation, and overlap windows. The current model approximates the outcome with stage overlap, but it cannot distinguish a symmetric warp layout from a producer/consumer layout.

### 5.18 Simplified Shared Memory Model

Shared-memory traffic is modeled as aggregate bytes divided by per-stage throughput. It does not model:

- Bank conflicts and swizzle layouts.
- Shared-memory multicast/broadcast behavior.
- Operand layout transformations between shared memory and tensor-core fragments.
- Barrier and memory-fence costs (`mbarrier`, `cp.async.wait_group`, `__syncthreads`).

This is acceptable for first-order throughput, but it can miss real bottlenecks in poorly aligned or nonstandard tile shapes.

### 5.19 Simplified Tensor Core Issue Model

MMA duration uses peak FMA/cycle per SM. It does not model:

- Instruction tile shape count and issue cadence directly.
- WGMMA async issue groups and wait operations.
- Tensor-core pipeline bubbles from dependency chains.
- Accumulator register pressure and fragment movement.

For large, well-tuned GEMMs this is close to the intended best-case. For small K, skinny matrices, or unusual tile shapes, instruction-level effects may dominate.

### 5.20 Simplified Epilogue Model

Epilogue is represented as shared store plus global or async/TMA store. Real epilogues may include:

- Bias, activation, residual add, scaling, quantization, or type conversion.
- Vectorized stores and write-combining behavior.
- Split-K reductions or atomic accumulation.
- Overlap between epilogue of one tile and mainloop of another tile in persistent kernels.

For plain `C = A × B`, the current model is enough to keep output writes visible. For fused or quantized matmul, epilogue needs its own decomposed compute and memory stages.

### 5.21 Limited Candidate Search

The model now evaluates a small architecture-aware matmul candidate set and selects the feasible candidate with the lowest scheduled time. It can reject candidates for instruction granularity, shared-memory overflow, and register pressure. The current search covers:

- Tile shapes: `block_m`, `block_n`, `block_k`.
- Warp count.
- Pipeline stage count.
- Async copy path labels: scalar global load, `cp.async`, TMA.
- MMA path labels: CUDA FMA, MMA, WGMMA, UMMA.
- Scheduling and CTA-order metadata.

**Remaining limitation**: The candidate set is intentionally small and matmul-specific. It does not yet search persistent scheduling, detailed warp specialization, CTA swizzle effects on L2 reuse, or a calibrated CUTLASS-scale tile catalog.

---

## 6. Summary Table

| Aspect | Current State | Limitation |
|--------|--------------|------------|
| Operator coverage | 1/7 (matmul only) | Cannot analyze other ops with pipeline model |
| Dependency tracking | `depends_on` field unused | Hardcoded stage order, fails for non-linear graphs |
| Occupancy model | SMEM + registers + threads + warps + blocks | Register model is coarse |
| Hardware validation | None | No calibration against real measurements |
| Async copy modeling | Binary toggle + local/HBM max duration | Simplified, doesn't distinguish cp.async vs TMA scheduling behavior |
| Memory traffic | Logical CTA and effective HBM split | L2 reuse is first-order, not schedule/cache-set accurate |
| Multi-CTA | Linear wave/resource scaling | Limited cache contention, DRAM bank conflicts, tail effects |
| Operator fusion | Not supported | Single-operator only (unlike SOLAR) |
| Stage classification | Substring heuristics | Fragile, no type system |
| Warp count | Candidate-selected or user-forced | Limited search, no warp specialization |
| Precision | Single dtype per analysis | No mixed precision support |
| Pipeline stages | Configurable stage count affects feasibility, occupancy, and prefetch distance | No barrier/wait-group or warp-specialized stage model |
| Candidate search | Small matmul candidate set with rejection reasons | Not a full CUTLASS-scale search |

---

## 7. V1 Plan Completion Record

The pipeline simulator v1 improvement plan has been implemented for matmul pipeline mode. The completed work is:

- Added `PipelineKernelCandidate` to represent concrete implementation strategies, including tile shape, warp count, stage count, copy path, MMA path, scheduling label, CTA order, and rejection reason.
- Extended `PipelineConfig` with `stage_count` and `warp_count`; user-provided tile, stage, or warp overrides now force a single candidate and fail clearly when infeasible.
- Extended `TilingInfo` with selected candidate name, stage count, register estimates, and the selected candidate object.
- Added architecture-aware matmul candidate generation for Ampere, Hopper, Blackwell, and fallback paths, with a small candidate set covering tile shapes, 4/8 warp variants, and 2/3 or 3/4 stage variants where appropriate.
- Updated pipeline analysis to schedule every feasible matmul candidate and select the one with the lowest `total_time_s`.
- Added candidate rejection for instruction granularity, shared-memory overflow, register-file capacity, per-block register limits, and per-thread register limits.
- Changed shared-memory sizing from fixed double-buffering to `stage_count × (A_tile + B_tile) + epilogue_smem`.
- Added coarse register-pressure estimation and included register limits in resident-CTA occupancy.
- Made async scheduler stage-depth aware with `prefetch_distance = stage_count - 1`; deeper stage counts now change prologue fill length and K-slice load/MMA overlap.
- Exposed `--stage-count` and `--warp-count` through the CLI, server API, and web UI.
- Updated JSON/table/UI output to report selected candidate, stage count, warps/block, shared memory, and register estimates.
- Added tests for candidate selection, forced overrides, stage/warp CLI options, register-pressure rejection, stage-depth prefetch distance, and updated pipeline expectations.

The v1 implementation intentionally does not yet expand beyond matmul, replace name-based stage classification, add explicit barrier/wait costs, or model warp-specialized producer/consumer groups.

---

## 8. Recommended Improvements

The following items are ordered by how much they improve the model's ability to represent realistic best-case hardware performance.

Near-term TODO items:

1. Replace fixed stage grouping with dependency-graph scheduling while preserving the current matmul fast path.
2. Split memory timing into copy-engine, L2, and HBM resources instead of only `max(local copy, HBM)`.
3. Add barrier/wait-group costs for `cp.async`, TMA, and WGMMA paths.
4. Add a confidence/known-limitations field to pipeline results for UI and CLI reporting.
5. Start the next operator template with convolution-as-implicit-GEMM before tackling flash attention.

### 8.1 Expand the Kernel Candidate Model

`PipelineKernelCandidate` v1 is implemented for matmul. The next step is to expand the candidate catalog and make more candidate fields operational:

- Add a broader architecture-specific tile catalog.
- Make `scheduling`, `cta_order`, and copy/MMA path labels affect timing and memory reuse instead of serving mostly as metadata.
- Preserve rejected candidates with enough diagnostics for UI comparison and tuning guidance.

### 8.2 Refine Stage-Depth Scheduling

`stage_count` is now part of `PipelineConfig`, tiling metadata, shared-memory sizing, candidate feasibility, occupancy, and async prefetch distance. The implemented baseline is:

```
load[k + prefetch_distance] overlaps mma[k]
prefetch_distance = stage_count - 1
```

The steady-state advance remains bounded by the slowest resource chain, while prologue fill length changes with stage depth. Remaining refinements:

- Add `cp.async.wait_group`, TMA mbarrier, and WGMMA wait/commit costs.
- Distinguish copy-engine queue depth from software stage count.
- Model producer/consumer warp specialization so Hopper/Blackwell stage depth changes register allocation and barrier behavior.

### 8.3 Refine Register-Based Occupancy

Pipeline mode now estimates registers per thread/block and includes them in resident CTA calculation:

```
blocks_by_registers = register_file_capacity / registers_per_block
```

The estimate should be refined:

- Accumulator fragments scale with `block_m × block_n`.
- Operand fragments scale with `block_m × block_k` and `block_k × block_n` per warp.
- More stages and warp specialization may add metadata/barrier registers.

Even a coarse register model is valuable because it prevents overly optimistic large-tile candidates.

### 8.4 Separate Copy Engine, L2, and HBM More Explicitly

The current global/async duration already uses `max(local copy time, HBM source time)`. The next step is to expose three memory resources:

- Copy engine or LD/ST issue bandwidth per SM.
- L2 bandwidth and capacity.
- HBM bandwidth per SM/GPC/partition.

Then memory duration can become:

```
max(copy_engine_time, l2_time, hbm_time)
```

This matters when effective HBM traffic is small but L2 bandwidth or copy engine bandwidth still limits the kernel.

### 8.5 Improve L2 Reuse Model

The current matmul L2 model should evolve from capacity-only fit ratio to a scheduling-aware model:

- CTA traversal order: row-major, column-major, grouped/swizzled.
- Reuse distance for A panels across N tiles and B panels across M tiles.
- L2 capacity per GPC/slice when known, not only whole-chip L2.
- Effective capacity discount for associativity, other traffic, and output writes.

A practical intermediate model is to add an `l2_residency_efficiency` factor by architecture and CTA order, calibrated against known GEMM behavior.

### 8.6 Model Warp Specialization for Hopper/Blackwell

For TMA/WGMMA kernels, represent producer and consumer warp-groups separately:

- Producer group issues TMA loads and barriers.
- Consumer group issues WGMMA and waits on barriers.
- Optional epilogue group drains stores.

This should affect `num_warps_per_block`, stage overlap, register allocation, and barrier overhead. It will also make Hopper/Blackwell defaults more realistic than treating them as faster Ampere-style kernels.

### 8.7 Add Barrier and Synchronization Costs

Add explicit barrier sub-ops or per-iteration overheads:

- `cp.async.commit_group` / `wait_group`.
- `mbarrier.arrive` / `mbarrier.wait`.
- `__syncthreads` for older paths.
- WGMMA group commit/wait.

These costs are small for large K but visible for small tiles, skinny GEMMs, and short-K workloads.

### 8.8 Add Tail and Small-Shape Efficiency Factors

Best kernels are still less efficient when shapes underfill CTAs or produce partial waves. Add penalties or exact accounting for:

- Partial M/N edge tiles.
- K not divisible by instruction K.
- Last wave SM underutilization.
- Very small grids where occupancy cannot hide latency.

The current model already validates tile granularity, but it does not distinguish full tiles from edge tiles in the schedule.

### 8.9 Extend Beyond Matmul with Operator-Specific Pipeline Templates

Next operator templates should be prioritized by performance relevance and clear hardware structure:

1. Convolution lowered to implicit GEMM or direct convolution.
2. Flash attention with QK, softmax, and PV stages.
3. Reductions with multi-level shared-memory and global reduction stages.
4. Fused matmul epilogues such as bias, activation, residual, and quantization.

Each operator should define both logical local traffic and effective HBM traffic. If no reuse model exists, defaulting effective to logical is acceptable but should be surfaced as a confidence limitation.

### 8.10 Add Calibration and Confidence Reporting

The model should expose a confidence level or known-missing-features list with each result:

- High: large dense GEMM, aligned shapes, supported architecture/dtype.
- Medium: GEMM with edge tiles or unusual aspect ratio.
- Low: fallback operator, missing register model, missing mixed precision, unknown copy path.

Calibration should compare representative predictions against vendor libraries or Nsight Compute counters:

- Runtime and achieved TFLOPS.
- DRAM bytes and L2 hit rate.
- Tensor-core utilization.
- Shared-memory throughput.
- Occupancy and active warps.

This keeps the theoretical model honest without requiring full cycle-accurate hardware simulation.
