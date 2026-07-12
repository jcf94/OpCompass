# OpCompass Gap Analysis and Implementation Roadmap

Status: proposed implementation baseline

Last reviewed: 2026-07-11

Scope: first-party code, tests, documentation, packaging, API, and Web UI. The
vendored `3rdparty/SOLAR/` tree is treated as an integration dependency, not as
first-party implementation progress.

## 1. Executive decision

OpCompass is a promising matmul-first research prototype, but it is not yet a
trustworthy general GPU operator Speed-of-Light estimator. Its strongest asset
is the end-to-end vertical slice: operators and hardware are discoverable; the
CLI, API, and browser UI work; matmul has candidate selection, occupancy checks,
and a visible CTA schedule; and SOLAR is integrated as a separate mode. The
central weakness is that the current output looks more general and more precise
than the underlying model actually is.

The next iterations should not begin by adding more operator names or a learned
model. They should first establish an honest, stable modeling contract and a
measurable validation loop. The recommended sequence is:

1. Make inputs, execution mode, output semantics, limitations, and model
   identity explicit.
2. Freeze the current behavior with first-party API, CLI, serialization, and
   numerical golden tests.
3. Replace string-driven matmul scheduling with an explicit dependency and
   resource model.
4. Add memory-path and synchronization semantics, then validate matmul against
   measurements.
5. Prove generality with reduction and normalization before undertaking
   convolution and attention.
6. Add a result database and calibration system before making accuracy claims.
7. Add learned ranking or residual correction only when the measurement corpus
   and holdout methodology are credible.

The immediate release should therefore be a contract-and-correctness release,
not a feature-count release.

## 2. Target product and non-goals

### 2.1 Product promise

For a fully specified operator workload and hardware target, OpCompass should
produce a versioned estimate containing:

- a theoretical lower bound;
- an implementation-family estimate when a supported kernel model exists;
- the selected or constrained implementation strategy;
- a bottleneck explanation tied to explicit hardware resources;
- assumptions, unsupported effects, warnings, and confidence;
- optional comparison with measured and calibrated results;
- enough identifiers to reproduce the estimate later.

The answer must distinguish three quantities that are currently easy to
conflate:

- **bound**: an optimistic physical lower bound such as a roofline result;
- **modeled estimate**: runtime predicted for a defined kernel family and
  schedule;
- **calibrated prediction**: a modeled estimate adjusted using versioned
  measurements.

“Speed of Light” should mean the best achievable runtime inside a stated model
scope, not an assertion that every unmodeled GPU effect is negligible.

### 2.2 Primary users

- Kernel engineers comparing tile, pipeline, and architecture choices.
- Framework/compiler engineers screening candidate schedules before tuning.
- Hardware analysts comparing operator behavior across GPU generations.
- Researchers building analytical or learned operator cost models.

Each audience needs machine-readable output. The browser UI is an explanatory
client of the API, not the source of model semantics.

### 2.3 Explicit non-goals for the next several releases

- Instruction-accurate GPU simulation.
- Correctness or code generation for arbitrary user kernels.
- End-to-end neural-network runtime prediction including framework overhead.
- Replacing Nsight Compute, CUTLASS Profiler, or vendor documentation.
- A full polyhedral compiler or mandatory MLIR/ISL dependency.
- Learned estimates before sufficient measured data and leakage-safe splits
  exist.

These may become adjacent projects, but allowing them into the critical path
would prevent OpCompass from becoming a reliable operator estimator.

## 3. Repository baseline verified on 2026-07-11

### 3.1 Components

| Area | Current implementation | Assessment |
|---|---|---|
| Shared schema | Dataclasses and enums in `models.py` | Useful but underspecified |
| Registry | Import-time package scanning | Convenient, weak failure reporting |
| Analysis | Roofline, matmul pipeline, SOLAR bridge | Three modes with inconsistent guarantees |
| Operators | 6 discoverable operator classes | Mostly formula-only |
| Hardware | 15 NVIDIA targets across Fermi–Blackwell/Thor | Broad catalog, limited provenance |
| Interfaces | Click CLI, FastAPI, vanilla JS UI | Functional vertical slice |
| Tests | 102 first-party tests | Concentrated on matmul pipeline |
| Packaging | `setup.py` plus minimal `pyproject.toml` | Installable, not production-grade |
| CI/quality | No checked-in CI, lint, type, or API schema gates | Missing |

### 3.2 Analysis modes as actually implemented

| Requested mode | Actual behavior | Key limitation |
|---|---|---|
| `hierarchy_roofline` | Computes FLOPs and unique tensor IO, times each against one HBM bandwidth and peak compute | Despite its name, it is not a real hierarchy model |
| `pipeline` + matmul | Searches a small hard-coded kernel-family candidate set and schedules grouped CTA stages | Scheduler is not dependency-driven and is not cycle-accurate |
| `pipeline` + other operator | Silently returns roofline-like output while retaining `mode=pipeline` | Consumers cannot tell fallback from real pipeline support |
| `solar` | Generates/imports a temporary torch model and invokes vendored SOLAR where supported | Optional dependency and subprocess/integration semantics are not a stable contract |

### 3.3 Operator maturity

| Operator | FLOP/IO formula | Pipeline model | SOLAR model source | Validation maturity |
|---|---:|---:|---:|---|
| Matmul | Yes | Yes, prototype | Yes | Substantial unit coverage, no measured accuracy suite |
| Convolution | Simplified same/stride-1 formula | No | No | Formula largely untested |
| Elementwise | Generic unary formula | No | No | Semantics too ambiguous for a stable operator |
| Flash attention | Simplified dense forward formula | No | No | Omits important variants and detailed IO behavior |
| LayerNorm | Approximate formula | No | No | No algorithm/precision variants |
| Reduction | Approximate formula | No | No | No reduction strategy or tail model |

The registry count of six is therefore not six equally supported operators. A
capability matrix must replace binary “available” language.

### 3.4 Hardware maturity

The registry currently discovers `gf100`, `gk110`, `gm204`, `p100`, `v100`,
`rtx6000`, `a100`, `h100`, `h100_pcie`, `b200`, `b300`, `gb200`, `gb300`,
`jetson-t4000`, and `jetson-t5000`.

The classes contain useful values for memory tiers, peak throughput, SM
resources, occupancy limits, and named stages. However:

- values have no structured per-field source, units metadata, confidence, or
  verification date;
- marketing peak, dense peak, sparse peak, boost-clock peak, and sustained
  assumptions are not represented as separate concepts;
- several stage latency and throughput values are modeling assumptions but are
  presented alongside vendor specifications without distinction;
- Python targets and SOLAR YAML cover different target sets and have no
  consistency contract;
- GB200/GB300 naming risks conflating a GPU, a Grace-Blackwell superchip, and
  node-level configuration;
- MIG, clock policy, power state, ECC, memory SKU, and interconnect are absent;
- older generations increase catalog breadth but currently contribute little
  validation depth.

### 3.5 Test facts

Verified commands:

```text
pytest tests -q
102 passed in 10.33s

pytest --collect-only -q
390 tests collected

pytest tests --cov=opcompass --cov-report=term-missing -q
TOTAL: 77%
```

The default 390-test count includes vendored SOLAR tests and must not be used as
a first-party quality metric. Important first-party coverage observations:

- `server.py`: 0%;
- `compute_model.py`: 0%;
- `memory_model.py`: 0%;
- non-matmul operator modules: approximately 42%–62%;
- CLI: 58%; result formatting: 60%;
- pipeline scheduler: 90%, but tests mostly preserve the current algorithm
  rather than establish measured correctness.

No browser test, API contract test, hardware provenance test, packaging test,
or benchmark accuracy test exists.

### 3.6 Reproduced correctness and contract failures

1. `compass analyze matmul --hardware a100 --dtype fp16 --M 4096` succeeds with
   zero runtime and infinite TFLOPS because missing `N` and `K` default to zero.
2. `compass analyze reduction ... --mode pipeline` returns JSON labeled
   `"mode": "pipeline"` without a `pipeline_schedule`; the actual algorithm is
   roofline fallback and no fallback reason is emitted.
3. The pipeline implementation and design text call the scheduler “DAG-based”,
   but `depends_on` is not the scheduling source of truth. Stages are grouped by
   string patterns such as `async_copy_load`, `shared_load`, `mma`, and
   `fma_alu`.
4. JSON includes every scheduled K-iteration sub-op by default, causing output
   size to scale with problem K and making otherwise small API responses large.
5. CSV serialization joins arbitrary nested Python values and does not perform
   standards-compliant CSV quoting or define a stable flat schema.
6. Table formatting references an undeclared `compute_unit_clock_hz` attribute
   via `hasattr`; pipeline phase time therefore prints a misleading zero rather
   than using a defined result field.
7. `DataType.byte_size` is annotated as `int` but returns `0.5` for FP4/INT4.

These are release-blocking trust issues because they can return plausible-
looking but semantically false results.

## 4. What is already worth preserving

- Small, approachable Python modules and dataclass-based data flow.
- Automatic discovery for first-party built-ins.
- Separation between operator formulas, hardware descriptions, and analysis.
- Matmul candidate rejection for tile granularity, shared memory, registers,
  and occupancy-related constraints.
- Explicit prologue, steady-state, and epilogue concepts.
- Architecture-aware candidate vocabulary (`cp_async`, TMA, WGMMA, UMMA,
  TMEM, warp-specialized scheduling), even where semantics are still coarse.
- Logical CTA traffic versus estimated effective HBM traffic.
- CLI/API/Web integration and visual schedule inspection.
- SOLAR retained as an independently identified analysis backend.

The roadmap should evolve these pieces rather than replace the project with a
large framework.

## 5. Detailed gap analysis

### 5.1 Input and workload schema — critical

Current operator methods accept permissive `**dims` dictionaries, use zero
defaults, ignore unknown keys, and mix structural dimensions with algorithm
choices such as `ops_per_element`. This prevents reliable validation,
canonicalization, caching, comparison, and dataset construction.

Required design:

- Define an `OperatorSpec` contract with typed parameter definitions:
  canonical name, aliases, kind, type, required/default status, minimum,
  divisibility, semantic description, and whether it affects tensor shape or
  implementation choice.
- Validate required, positive, and mutually constrained dimensions before any
  formula executes.
- Reject unknown keys by default; provide an explicit compatibility switch if
  loose parsing is needed.
- Canonicalize shape keys and order so `M/N/K`, API JSON, cache keys, and
  measurement records are stable.
- Separate tensor dtypes into input, weight, accumulator, and output roles.
- Add layout, transpose, batching, stride/padding/dilation/groups, causal mask,
  and algorithm variants only where the relevant operator schema declares them.
- Represent symbolic or ranged dimensions separately from a concrete analysis
  request; V0.x runtime analysis should require concrete positive values.

### 5.2 Result and diagnostic contract — critical

`AnalysisResult` currently reports a requested mode and point result but lacks
the information needed to judge it.

Add:

- `requested_mode`, `executed_mode`, and `fallback` object;
- `estimate_kind`: `theoretical_bound`, `analytical_model`, `calibrated`, or
  `learned`;
- `model_id`, semantic schema version, implementation Git revision, and
  hardware-spec version;
- support level: `unsupported`, `formula`, `pipeline`, `validated`;
- assumptions, warnings, missing effects, and rejected alternatives;
- confidence split into evidence coverage and estimated uncertainty; do not use
  an unexplained single percentage;
- lower/point/upper estimate when uncertainty is defensible;
- exact units in field names or a documented unit schema;
- compact schedule summary by default and optional trace expansion;
- stable error codes for invalid input, unsupported mode, unavailable optional
  backend, infeasible candidate, and internal failure.

Fallback must never be silent. Strict mode should reject unsupported analysis;
permissive mode may fall back but must identify the executed model.

### 5.3 Analytical roofline — high

The current “hierarchy roofline” uses `hardware.hbm_bandwidth` for transfer
time. `compute_model.py` and `memory_model.py` exist but are not the single path
used by `Analyzer`, creating duplicated semantics and dead abstractions.

Required work:

- Rename the existing behavior to a precise `roofline` bound or implement the
  promised hierarchy semantics.
- Define unique bytes, algorithmic bytes, CTA logical bytes, and effective bytes
  separately.
- Produce per-tier traffic and lower bounds for HBM, L2, L1/shared, register
  file, and interconnect where applicable.
- Define whether read and write share a link and whether bidirectional peak is
  aggregate or per direction.
- Represent overlap as a resource graph, not a non-empty set interpreted as
  “all read, compute, and write fully overlap”.
- Handle launch overhead and small-workload utilization outside the pure bound,
  and label those as modeled overhead rather than roofline physics.
- Consolidate or remove `compute_model.py` and `memory_model.py` so there is one
  tested implementation per formula.

### 5.4 Pipeline IR and scheduler — critical

The current model is an analytical three-phase formula plus a generated
timeline, not a generic DAG scheduler and not cycle-accurate. Fixed grouping,
stage-name inference, sequential handling inside groups, and per-stage
throughput approximations prevent valid modeling of independent engines,
barriers, producer/consumer warp groups, and arbitrary operators.

Introduce a minimal scheduler IR:

```text
KernelTemplate
  loops: iteration domains and tails
  nodes: work, latency, eligible resources, occupancy duration
  edges: data, control, barrier, loop-carried, queue-capacity
  resources: capacity, issue rate, latency, sharing scope
  buffers: address space, size, lifetime, stage/ring slot
  launch: grid mapping, CTA order, cluster shape, residency
```

Key rules:

- Work unit must be explicit (`bytes`, `flops`, `instructions`, `transactions`,
  `cycles`, `items`); never inferred from a name.
- Resource identity and sharing scope must be explicit: per warp, warp group,
  CTA, SM, GPC/cluster, or device.
- Dependencies must drive earliest-start time.
- Queue depth, barrier arrival/wait, pipeline commit/wait, and buffer reuse must
  be representable.
- Edge and partial tiles must use valid-work and padded/transaction work
  separately.
- Full traces should be generated from a compact repeated schedule and capped or
  paginated.
- “Cycle-accurate” should not appear in user-facing claims until instruction
  issue and latency behavior is validated; use “cycle-based analytical
  schedule” meanwhile.

Migration should keep the old scheduler as `legacy_matmul_v1` behind an
internal comparison flag until the new scheduler has golden and measured
parity reports.

### 5.5 Kernel candidate space and occupancy — high

Candidate generation is a short hard-coded list selected by architecture and
dtype. It omits instruction shapes, warp tile mapping, split-K/stream-K,
persistent schedules, CTA clusters, swizzle details, alignment, vectorization,
epilogue variants, and realistic register allocation granularity.

Required work:

- Split candidate description from candidate enumeration and scoring.
- Record instruction shape, warp/warp-group tile, CTA tile, cluster tile,
  stages, vector width, copy path, reduction scheme, epilogue, and launch order.
- Use architecture capability predicates rather than string comparisons on
  `architecture`.
- Model occupancy rounding: registers per allocation unit, shared-memory
  allocation granularity, warp/block limits, and architecture-specific opt-in
  shared memory limits.
- Distinguish feasible, modeled, and measured-supported candidates.
- Add deterministic pruning reasons and candidate-count limits.
- Optimize estimated device runtime, not only a single CTA, including waves,
  tail utilization, and small-grid underfill.

### 5.6 Memory system and data movement — critical

The current L2 reuse estimate interpolates between unique and logical traffic
based on whether a working set fits in aggregate L2. It does not model CTA
order, cache slices/sets, concurrent working sets, eviction, transaction size,
coalescing, multicast, or partition camping.

Required staged improvement:

1. Explicitly separate HBM fabric, L2, copy/TMA engine, shared-memory ports,
   register file/TMEM, and store path resources.
2. Add access descriptors: tensor, affine region, bytes requested, transaction
   granularity, vector width, stride, reuse group, and address space.
3. Add analytical reuse policies keyed by CTA order and tile shape; report the
   policy and sensitivity range rather than a false exact byte count.
4. Model bank conflicts, shared-memory multicast, and RF/TMEM pressure only
   after benchmark data can calibrate them.
5. Keep cache simulation optional; a transparent analytical approximation is
   preferable to an opaque but unvalidated detailed simulator.

### 5.7 Architecture semantics — critical for modern GPUs

Ampere `cp.async`, Hopper TMA/WGMMA, and Blackwell TMEM/UMMA are currently
mostly stage names and throughput differences. A trustworthy modern model
needs:

- Ampere async-copy group issue/commit/wait, alignment, queue depth, shared
  buffer lifetime, and synchronization;
- Hopper producer/consumer warp groups, TMA descriptors and transactions,
  mbarrier lifecycle, WGMMA issue/commit/wait, register redistribution, and
  optional CTA clusters/distributed shared memory;
- Blackwell UMMA/TMEM allocation, accumulator residence and movement,
  load/store paths, TMA store, and updated cluster/resource constraints;
- architecture capability records declaring which constructs are supported;
- microbenchmarks for each resource parameter instead of undocumented constants.

These features should be introduced as reusable resources and nodes, not as
more scheduler conditionals.

### 5.8 Operator semantics and roadmap — high

Before adding pipeline templates, formula semantics must be corrected and
versioned.

- **Elementwise**: split unary, binary, activation, transcendental, and fused
  expression forms; account for input count, vectorization, SFU cost, and
  broadcasting.
- **Reduction**: model rows and reduction length, warp/block/multi-CTA
  algorithms, partial output, atomics or second pass, synchronization, and
  non-divisible tails.
- **LayerNorm/RMSNorm**: distinguish algorithms, affine parameters, input and
  output precision, reduction passes, online statistics, vector loads, and
  fusion.
- **Convolution**: represent NCHW/NHWC, stride, padding, dilation, groups,
  output-shape validation, and direct/implicit-GEMM algorithm family. Reuse the
  GEMM template only after im2col-equivalent traffic and iterator overhead are
  explicit.
- **Flash attention**: distinguish forward/backward, causal/non-causal,
  MHA/GQA/MQA, Q/K/V lengths, head dimensions, dropout, KV cache, softmax
  precision, online reduction state, and recomputation. “Read Q/K/V once” is a
  useful algorithmic lower bound, not generally the actual HBM traffic.
- **Matmul**: add batch/grouped forms, transpose/layout, leading dimensions,
  mixed precision, fused epilogues, split-K/stream-K/persistent variants, and
  edge tiles.

Recommended proof-of-generality order is reduction, LayerNorm/RMSNorm,
convolution, then flash attention. Reduction is deliberately ahead of
convolution: it forces dependency, synchronization, tail, and multi-pass
semantics that cannot be hidden by lowering everything to GEMM.

### 5.9 Hardware data governance — critical

Create a versioned hardware-spec schema with two layers:

- **published facts**: SKU identity, SM count, memory capacity/bandwidth, clocks,
  resource limits, supported instructions;
- **model parameters**: effective issue rates, latencies, sustained-efficiency
  priors, cache behavior, barrier costs.

Every value needs units, scope, source URL/document and page where possible,
source type, assumption note, confidence, and last verification date. Calibration
overlays must never overwrite the raw source facts.

Add invariant tests:

- unique names and valid semantic versions;
- positive capacities, bandwidths, clocks, and unit counts;
- peak throughput consistency within documented tolerance;
- stage resources referenced by a kernel exist on the target;
- dtype and feature capability consistency;
- Python/SOLAR mapping consistency for overlapping targets;
- source presence for every user-visible specification.

### 5.10 SOLAR integration — medium

SOLAR is valuable as a graph-level comparison mode but should not be blended
with the first-party operator pipeline model.

- Define an optional `solar` extra with supported Python/PyTorch versions.
- Detect backend availability explicitly and return a typed unavailable error.
- Isolate temporary generated sources and cache artifacts outside the repo.
- Version the adapter against the vendored commit.
- Test source generation without requiring GPU or torch execution.
- Maintain a small integration suite separate from first-party default tests.
- Explain semantic differences: SOLAR graph fusion/prefetch scenarios are not
  interchangeable with OpCompass CTA scheduling.
- Avoid editing vendored SOLAR unless an isolated upstreamable fix is required.

### 5.11 Registry and plugin model — medium

Dynamic scanning currently swallows `ImportError`, so a broken built-in module
can silently disappear. Duplicate names overwrite earlier discoveries.

- Cache discovery deterministically.
- Detect duplicate names and report origin modules.
- Separate “optional dependency unavailable” from “module is broken”.
- Fail loudly for first-party import failures.
- Use Python entry points for third-party plugins; keep package scanning for
  built-ins.
- Return a capability manifest for each operator/hardware/backend combination.

### 5.12 API, CLI, and serialization — critical

- Introduce Pydantic request/response models and OpenAPI examples.
- Version stable routes under `/api/v1`; retain old routes temporarily with
  deprecation headers.
- Reject booleans/floats/strings where positive integer dimensions are required.
- Bound shape sizes and trace expansion to prevent accidental CPU/memory abuse.
- Add request timeout/cancellation strategy for SOLAR and future sweeps.
- Return typed 4xx errors; prevent uncaught `TypeError`, `KeyError`, and import
  failures from becoming opaque 500 responses.
- Add `--strict`, `--summary/--trace`, and a capability command to the CLI.
- Replace ad-hoc CSV joining with the standard CSV library and a documented flat
  export schema.
- Stream or write large sweeps incrementally; add stable ordering and failure
  policy per case.
- Define NaN/infinity JSON policy. Invalid workloads must never serialize
  `Infinity` as a successful estimate.

### 5.13 Web UI — medium

The UI is useful but tightly coupled to large response objects and has no
automated tests.

- Drive controls from operator and capability schemas rather than generic
  dimension fields alone.
- Clearly badge bound/modeled/calibrated, requested/executed mode, support
  level, confidence, and warnings.
- Do not show pipeline-only panels after fallback.
- Request summarized schedules first; fetch trace windows on demand.
- Add URL-serializable analysis state and reproducible share/export actions.
- Add comparison views across hardware, candidate, and model version.
- Escape or construct DOM safely for API-provided text rather than relying on
  broad `innerHTML` composition.
- Add browser smoke tests for load, validation error, successful roofline,
  supported pipeline, fallback/unsupported behavior, and API failure.
- Meet basic accessibility requirements: labels, keyboard navigation, table
  semantics, non-color bottleneck indicators, and chart alternatives.

### 5.14 Packaging, compatibility, and operations — high

The project declares Python `>=3.8` while using modern typing syntax and a very
small dependency declaration. Python 3.8 compatibility happens to work in the
current environment but must be continuously tested or the floor should move.

- Consolidate project metadata in `pyproject.toml`; remove duplicated legacy
  metadata from `setup.py` after a compatibility window.
- Define extras such as `server`, `solar`, `calibration`, `learned`, and `dev`.
- Include Web assets, YAML specs, and other runtime data explicitly in wheel and
  sdist tests.
- Add a package version API and embed it in results.
- Add license, changelog, contribution guide, security policy, and release
  process.
- Establish supported Python and dependency matrices.
- Ensure caches and generated SOLAR handlers are ignored or stored in a defined
  user cache directory.
- Add structured logging and request correlation for the server; avoid logging
  full potentially sensitive workload payloads by default.

### 5.15 Testing and quality system — critical

Configure first-party tests as the default and create separate markers/jobs:

- `unit`: formulas, validation, resources, scheduler;
- `contract`: serialization, CLI, API/OpenAPI;
- `golden`: versioned compact outputs for representative cases;
- `property`: monotonicity and invariants across generated valid shapes;
- `integration`: SOLAR adapter and packaging;
- `browser`: critical UI flows;
- `accuracy`: measured benchmark corpus, allowed to run separately.

Important properties include:

- runtime and byte counts are finite and non-negative;
- increasing work cannot reduce a pure lower bound without an explainable
  utilization transition;
- disabling overlap cannot improve predicted runtime;
- reducing peak throughput or bandwidth cannot improve the corresponding bound;
- every scheduled node respects dependency and resource capacity constraints;
- total valid work matches the operator formula including edge tiles;
- serialization round-trips without loss of units or model identity.

Use coverage as a guardrail, not the accuracy metric. Initial targets: 85%
overall first-party, 90% for schemas/analyzer/scheduler/serialization, and 100%
branch coverage for input validation and fallback decisions.

## 6. Target architecture

```text
Typed AnalysisRequest
        |
        v
Operator schema + canonical Workload
        |
        +-------------------------+
        |                         |
        v                         v
Algorithmic accounting      Capability resolver
(FLOPs/tensors/accesses)     (support + strict fallback)
        |                         |
        +-------------+-----------+
                      v
             Analysis backend
       +--------------+----------------+
       |              |                |
   Roofline       Pipeline IR        SOLAR adapter
       |              |                |
       |        Candidate generator    |
       |              |                |
       |        Resource scheduler     |
       +--------------+----------------+
                      v
             Normalized Result IR
       (identity, diagnostics, summary, trace ref)
                      |
          +-----------+------------+
          |                        |
      Calibration              Persistence
          |                        |
          +-----------+------------+
                      v
              CLI / API v1 / Web
```

The normalized result is the stability boundary. Backends may evolve, but all
must state what they executed and what their estimate means.

Suggested package boundaries:

```text
opcompass/
  schema/          request, workload, result, diagnostics, versioning
  accounting/      operator-independent tensor/access/FLOP primitives
  operators/       schemas and algorithm templates
  hardware/        versioned published specs and capability adapters
  backends/
    roofline/
    pipeline/      IR, resources, scheduler, trace compression
    solar/
  calibration/     measurements, importers, overlays, metrics
  api/             versioned Pydantic routes
  cli.py
```

Do not perform this move as a single repository-wide rewrite. Introduce the new
boundaries while preserving compatibility imports, then delete legacy paths
after parity gates pass.

## 7. Versioned implementation plan

Version labels are sequencing aids; release numbers may be adjusted. Every
version must update schema/model versions and release notes when numerical
semantics change.

### V0.2 — Trustworthy contract and baseline

Implementation handoff and remaining release work:
[`v0_2_session_handoff.md`](v0_2_session_handoff.md).

Goal: make every successful result valid, finite, reproducible, and honest
about which model ran.

Work packages:

1. Add typed operator parameter specs and centralized validation.
2. Add requested/executed mode, estimate kind, support level, fallback,
   diagnostics, assumptions, warnings, model identity, and schema version.
3. Add compact schedule summaries; make full trace opt-in with a size limit.
4. Introduce Pydantic API models and stable error codes.
5. Fix JSON infinity policy, standards-compliant CSV, and pipeline table phase
   timing.
6. Configure pytest for first-party tests and separate vendored/SOLAR jobs.
7. Add API tests, serialization round trips, CLI error tests, and compact golden
   fixtures for A100/H100/B200 matmul plus fallback cases.
8. Correct README and pipeline design claims (“cycle-based analytical”, not
   “cycle-accurate DAG scheduler”).
9. Move package metadata toward `pyproject.toml` and add wheel/sdist smoke tests.

Exit criteria:

- Missing/zero/negative/unknown dimensions return a typed error.
- A pipeline request either returns a real pipeline result or an explicit
  fallback/unsupported status; strict mode never falls back.
- Every successful numeric field is finite and has documented units.
- Default JSON size is bounded independently of K iterations.
- API/OpenAPI, CLI, and serialization contract tests pass on supported Python
  versions.
- First-party and vendored test counts are reported separately.

### V0.3 — Explicit pipeline IR and generic resource scheduler

Implementation status (2026-07-12): the generic IR, validation, deterministic
resource scheduler, compact trace windows, synthetic fixtures, and matmul IR
emission are implemented. Pipeline results expose a labeled legacy comparison
while the established phase/epilogue timeline remains available for numerical
compatibility; see [`../releases/v0.3.md`](../releases/v0.3.md) for the precise
boundary and known limitations.

Goal: remove matmul name matching and make dependencies/resources operational.

Work packages:

1. Define resource, work, buffer, node, edge, loop, launch, and compact schedule
   schemas.
2. Implement validation for missing nodes, dependency cycles, invalid units,
   resource capacity, and buffer lifetime/ring reuse.
3. Implement deterministic list scheduling with dependency earliest-start,
   resource calendars, queue capacities, and loop-carried edges.
4. Model prologue/steady/epilogue as loop expansion or analytical repetition,
   not hard-coded load/shared/MMA groups.
5. Port matmul and retain a legacy comparison report.
6. Add synthetic scheduler fixtures plus property tests for dependency and
   resource invariants.

Exit criteria:

- Scheduler contains no stage-name category matching.
- Changing an edge or resource changes the schedule as expected.
- Compact schedule can reconstruct requested trace windows.
- Matmul legacy/new differences are documented per golden case; unexplained
  regressions block release.
- At least one synthetic non-matmul DAG schedules without special-case code.

### V0.4 — Memory paths, synchronization, and modern GPU semantics

Implementation status (2026-07-12): typed memory paths and transactions,
first-order reuse policies, explicit Ampere/Hopper/Blackwell synchronization
graphs, and launch wave/underfill/tail policies are implemented. See
[`../releases/v0.4.md`](../releases/v0.4.md) for scope and limitations.

Goal: make resource bottlenecks and architecture differences explanatory.

Work packages:

1. Split HBM, L2, copy/TMA, shared, RF/TMEM, compute, synchronization, and store
   resources.
2. Add transaction/access descriptors and first-order CTA-order reuse policies.
3. Add cp.async group, TMA/mbarrier, WGMMA commit/wait, UMMA/TMEM, and
   syncthreads abstractions.
4. Add tail tiles, small-grid underfill, occupancy allocation granularity, and
   launch-overhead policy.
5. Build targeted microbenchmark specifications and fixture import format even
   if GPU collection occurs outside this repository.

Exit criteria:

- Results can distinguish HBM-, L2-, copy-engine-, shared-memory-, compute-,
  synchronization-, and epilogue-bound cases.
- Ampere/Hopper/Blackwell paths use different explicit resource graphs.
- Small K, skinny matrices, edge tiles, and insufficient-grid cases have golden
  coverage.
- Every non-published latency/throughput parameter is marked as assumption or
  calibrated value.

### V0.5 — Hardware provenance and matmul validation

Goal: establish the evidence loop before broad operator expansion.

Work packages:

1. Introduce versioned hardware facts and model-parameter overlays.
2. Populate primary-source provenance for A100, H100 SXM/PCIe, and B200 first;
   mark other targets provisional until audited.
3. Define the measurement record schema and repository layout.
4. Add CUTLASS Profiler and generic CSV/JSON importers; add optional Nsight
   Compute counter importer.
5. Record runtime distributions, environment, clocks, kernel identity,
   candidate metadata, and model version.
6. Build accuracy reports by architecture, dtype, shape regime, and bottleneck.
7. Fit transparent calibration overlays only after raw error is reported.

Exit criteria:

- A measurement round-trips without code changes and remains linked to raw
  source plus model version.
- Hardware fields shown in API/UI expose provenance.
- Accuracy evaluation uses shape-family and hardware holdouts where possible.
- Reports include median APE, p90 APE, log-RMSE, bias, and candidate ranking
  top-k recall; no single average hides failure regimes.
- Initial accuracy targets are declared from baseline data, not invented in
  advance. Release notes report achieved values and known outliers.

### V0.6 — Reduction and normalization pipelines

Goal: prove the scheduler handles non-GEMM synchronization and multi-phase
algorithms.

Work packages:

1. Replace generic reduction with explicit row/reduction shape and algorithm
   variants.
2. Model warp, block, and multi-CTA/two-pass reductions including tails and
   partial outputs.
3. Add LayerNorm and RMSNorm accounting plus one-pass/two-pass/online variants.
4. Model vector loads, shared reductions, barriers, SFU/reciprocal-sqrt cost,
   affine parameters, and output conversion.
5. Add measured fixtures and operator-specific accuracy reports.

Exit criteria:

- No Matmul-specific scheduler path is used.
- Formula, schedule, API, CLI, fallback, and measured-accuracy tests exist for
  both families.
- Multi-pass memory traffic and synchronization are visible in explanations.

### V0.7 — Convolution and richer matmul families

Goal: cover important tensor-core workloads without pretending all are plain
GEMM.

Work packages:

1. Complete convolution schema and validated output-shape calculation.
2. Implement an implicit-GEMM template with iterator/addressing overhead and
   real tensor traffic.
3. Add batched/grouped GEMM, transpose/layout, mixed precision, fused epilogues,
   and edge-aware tiles.
4. Add split-K, stream-K, and persistent candidates where evidence supports
   them.
5. Extend measurement import and accuracy reporting by algorithm family.

Exit criteria:

- NCHW/NHWC, stride, padding, dilation, groups, and non-divisible shapes are
  validated.
- Mixed precision has separate input/accumulator/output roles.
- Algorithm choice and extra traffic are visible; convolution is not reported
  as zero-cost GEMM lowering.

### V0.8 — Attention and fused LLM kernels

Goal: model online reductions and multi-phase fused kernels.

Work packages:

1. Define forward attention variants and supported scope precisely.
2. Represent QK, online max/sum update, probability/value accumulation,
   rescaling, and output phases in the generic IR.
3. Add causal masking, unequal query/KV lengths, MHA/GQA/MQA, and KV-cache read
   behavior in staged increments.
4. Add attention-specific tiling, residency, shared/register pressure, and
   measured fixtures.
5. Add backward only as a later explicitly scoped sub-release.

Exit criteria:

- Phase-level compute and memory accounting reconciles with the operator total.
- The model states whether it is an algorithmic lower bound or an implementation
  estimate.
- Accuracy is reported by sequence/head-dimension regime and causal variant.

### V0.9 — Calibration productization and comparison workflows

Goal: make validated estimates easy to consume and reproduce.

Work packages:

1. Versioned `/api/v1`, capability discovery, pagination/trace endpoints, and
   durable request/result identifiers.
2. CLI batch manifests, incremental exports, compare, explain, and reproduce
   commands.
3. Web comparison across hardware/candidates/model versions with confidence and
   provenance panels.
4. Documentation: modeling semantics, limitations, operator authoring,
   hardware authoring, benchmark collection, calibration, API, and releases.
5. Release automation, compatibility matrix, security and support policy.

Exit criteria:

- A result can be reproduced from its canonical request and version metadata.
- Bound, modeled, measured, and calibrated values are never visually conflated.
- Package artifacts contain all runtime assets and pass clean-environment smoke
  tests.

### V1.0 candidate — learned residual/ranking model, only if data gates pass

Goal: complement, not replace, the interpretable model.

Entry gates:

- Enough measured coverage across at least multiple architectures, operator
  families, and shape regimes;
- stable feature and measurement schemas;
- leakage-safe hardware/shape/kernel-family splits;
- analytical baseline and calibration metrics already published.

Start with gradient-boosted/tabular baselines for candidate ranking and
log-runtime residual correction. Only then evaluate MLP, set, or graph models.
Output must include artifact/dataset version, interval or conformal coverage,
OOD score, and disagreement with the analytical estimate. Reject or downgrade
learned output outside its support rather than extrapolating silently.

## 8. Prioritized backlog

### P0 — trust and correctness

- Concrete workload schema and strict validation.
- Requested versus executed mode and explicit fallback.
- Finite-number and units policy.
- Model/schema/hardware version identity.
- Compact schedule output and trace limit.
- Pydantic API contract and API tests.
- CSV/table serialization fixes.
- First-party pytest configuration and separated SOLAR suite.
- Correct misleading scheduler and hierarchy claims in documentation.

### P1 — modeling foundation

- Explicit work/resource/dependency/buffer/loop IR.
- Generic deterministic resource scheduler.
- HBM/L2/copy/shared/RF-TMEM/compute/sync/store resources.
- Tail, occupancy granularity, underfill, and overhead policies.
- Matmul migration with legacy comparison.
- Hardware provenance schema and primary target audit.

### P2 — evidence and generality

- Measurement database and importers.
- Matmul accuracy reports and calibration overlay.
- Reduction and LayerNorm/RMSNorm pipelines.
- Property tests and accuracy gates.

### P3 — coverage

- Convolution semantics and implicit GEMM.
- Rich matmul families, fusion, and mixed precision.
- Attention variants and online-reduction pipeline.
- Modern architecture cluster and advanced scheduling features.

### P4 — product and research extensions

- Versioned comparison/reproduction workflows.
- Third-party operator/hardware entry-point plugins.
- Learned residual and candidate-ranking models after data gates.
- Optional polyhedral/MLIR-derived representation and features.

## 9. Recommended next sprint, decomposed into reviewable PRs

Do not combine these into one rewrite.

### PR 1 — Test boundary and reproduced regressions

- Set first-party pytest paths and markers.
- Add tests that currently fail for missing/zero/negative dimensions, unknown
  dimensions, silent pipeline fallback, JSON infinity, and phase timing.
- Add baseline API tests using FastAPI's test client.
- Record first-party coverage separately from vendored coverage.

### PR 2 — Workload validation and capability contract

- Add operator parameter definitions and a canonical workload validator.
- Expose per-operator support by mode and hardware capability.
- Add strict/permissive fallback policy.
- Migrate all six operators without changing valid formula results.

### PR 3 — Result V1 and output hygiene

- Add result identity, executed mode, estimate kind, diagnostics, and units.
- Add compact schedule summary and opt-in/capped trace.
- Fix JSON, CSV, and table formatting.
- Add serialization golden tests and compatibility mapping for old fields.

### PR 4 — Typed API and UI honesty

- Add Pydantic request/response/error models and OpenAPI examples.
- Update UI to show support level, fallback, warnings, and estimate kind.
- Hide pipeline visualization when no pipeline executed.
- Add a minimal browser smoke test.

### PR 5 — Packaging and documentation baseline

- Consolidate metadata and package runtime assets.
- Add wheel/sdist clean-install smoke tests.
- Add CI for supported Python versions, unit/contract tests, coverage, lint, and
  type checks.
- Update README and `pipeline_simulator_design.md` to match implementation.

Sprint completion means the current model has become harder to misuse. It does
not require the new scheduler yet.

## 10. Measurement and acceptance framework

### 10.1 Numerical change policy

Every numerical change must be classified:

- bug fix;
- new modeled effect;
- hardware-data correction;
- calibration change;
- algorithm/candidate-space change.

The PR must provide before/after compact results for a small canonical matrix:
at least A100/H100/B200, FP32/FP16 where supported, square/skinny/small-K/tail
shapes, and async on/off where meaningful. A changed number without a changed
model/version and explanation is a regression.

### 10.2 Accuracy evaluation

Use runtime distributions, not a single timing. Preserve warmup, repetitions,
quantiles, clock/power state, software versions, and kernel identity. Compare:

- raw roofline bound;
- uncalibrated pipeline estimate;
- calibrated estimate;
- best measured kernel and, separately, the matched modeled candidate.

Report median and p90 absolute percentage error, log-RMSE, signed bias, interval
coverage, and candidate ranking recall. Slice results by architecture, dtype,
operator, shape regime, and predicted bottleneck.

### 10.3 Definition of done for an operator/backend pair

- Typed semantics and invalid-input tests.
- Formula reconciliation for FLOPs, reads, writes, and intermediate traffic.
- Explicit support/capability entry.
- Resource schedule invariants and tail behavior.
- CLI/API/serialization coverage.
- Stated assumptions and unsupported variants.
- Measured fixtures across representative regimes or support level limited to
  `formula`/`experimental`.
- Versioned accuracy report before the pair is called `validated`.

## 11. Principal risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Detailed scheduler creates false precision | Users trust exact-looking cycles | Use evidence levels, uncertainty, and “cycle-based analytical” wording |
| New IR becomes a compiler project | Roadmap stalls | Model only constructs needed by the next operator and benchmark |
| Hardware constants become unreviewable | Results cannot be defended | Separate published facts, assumptions, and calibration overlays |
| Legacy parity preserves known bugs | New model remains wrong | Classify changes and require measured/property evidence, not blind parity |
| Operator breadth outruns validation | Six names imply six mature models | Publish support matrix and gate `validated` status |
| Calibration leaks test data | Accuracy looks unrealistically good | Version datasets and split by shape/kernel/hardware families |
| Learned model extrapolates | Confident wrong predictions | OOD detection, conformal coverage, and analytical fallback |
| SOLAR dependency destabilizes base install | Core workflows break | Optional extra and isolated integration test job |
| API/trace accepts unbounded work | Server resource exhaustion | Request limits, trace pagination, cancellation, and timeout policy |
| Vendored tests distort quality metrics | Progress is overstated | Separate first-party, integration, and vendored reports |

## 12. Long-term completion criteria

OpCompass can be considered a mature 1.x estimator when:

- every request is validated and every result identifies the executed model;
- roofline, pipeline, measured, calibrated, and learned quantities have distinct
  semantics;
- dependencies and explicit resources drive pipeline schedules;
- memory paths and synchronization are first-class modeled effects;
- A100, H100, and B200 hardware data have auditable provenance and calibrated
  parameters;
- matmul, reduction/normalization, convolution, and forward attention each have
  documented support levels and accuracy reports;
- unsupported variants fail or fall back visibly;
- CLI, API, and Web consume the same versioned result schema;
- package, API, browser, property, integration, and accuracy tests run in
  separated CI jobs;
- numerical changes are reproducible and tied to model/data versions;
- learned estimates, if present, demonstrate improvement on leakage-safe
  holdouts and expose uncertainty/OOD status.

Until these conditions hold, OpCompass should describe itself as an
experimental analytical estimator with a detailed matmul prototype, not as a
general validated GPU performance predictor.
