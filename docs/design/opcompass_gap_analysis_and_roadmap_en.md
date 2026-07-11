# OpCompass Repository Gap Analysis and Roadmap

Date: 2026-07-09

This document is the English roadmap for OpCompass. It summarizes the current
repository state, the gap to the target vision, and the recommended iteration
plan. The Chinese version is:

- [opcompass_gap_analysis_and_roadmap_zh.md](opcompass_gap_analysis_and_roadmap_zh.md)

Related external work is collected in:

- [docs/reference/related_work_and_data_driven_perf_modeling.md](../reference/related_work_and_data_driven_perf_modeling.md)

## 1. Target Vision

OpCompass should be a GPU operator Speed-of-Light estimator. It should answer:

- What is the best-case runtime for an operator, shape, dtype, and target GPU?
- Is the bottleneck compute, HBM, L2, shared memory, copy engine, tensor cores,
  CUDA cores, synchronization, or epilogue writeback?
- How do tile shape, software pipeline stage count, warp count, async copy, TMA,
  WGMMA, TMEM, sparsity, and CTA order affect the result?
- How do Ampere, Hopper, Blackwell, and future targets differ in the modeled
  pipeline?
- What is the confidence of the estimate, and which effects are missing?

The current repository is a useful matmul-first prototype. The main gap is no
longer basic functionality. The remaining work is model generality, hardware
accuracy, measured validation, learned prediction, and product-quality output.

## 2. Current Snapshot

### 2.1 Package Structure

Current main components:

- `opcompass/models.py`: shared dataclasses and enums.
- `opcompass/registry.py`: dynamic operator and hardware discovery.
- `opcompass/cli.py`: Click CLI.
- `opcompass/server.py`: FastAPI API and static Web UI server.
- `opcompass/engine/analyzer.py`: main analysis dispatcher.
- `opcompass/engine/pipeline_model.py`: CTA-level pipeline scheduler.
- `opcompass/engine/solar_analyzer.py`: bridge to vendored SOLAR.
- `opcompass/operators/`: operator definitions.
- `opcompass/hardware/`: hardware target definitions.
- `opcompass/configs/solar_arch/`: SOLAR architecture YAML files.
- `web/`: static Web UI.
- `tests/`: first-party pytest tests.
- `3rdparty/SOLAR/`: vendored external code.

### 2.2 Analysis Modes

Current modes:

- `hierarchy_roofline`: roofline-style lower bound from FLOPs, bytes, and
  hardware peaks.
- `pipeline`: CTA-level pipeline scheduling, currently detailed for matmul.
- `solar`: SOLAR graph-level analysis when optional dependencies and operator
  model source generation are available.

### 2.3 Operator Coverage

Discovered operators:

- `matmul`
- `convolution`
- `flash_attention`
- `layernorm`
- `elementwise`
- `reduction`

Pipeline-level coverage is effectively 1/6:

- `matmul` has roofline, pipeline, and SOLAR source generation.
- Other operators currently have FLOP/IO formulas and mostly fall back to
  roofline behavior in pipeline mode.

### 2.4 Hardware Coverage

Discovered hardware targets:

- Fermi: `gf100`
- Kepler: `gk110`
- Maxwell: `gm204`
- Pascal: `p100`
- Volta: `v100`
- Turing: `rtx6000`
- Ampere: `a100`
- Hopper: `h100`, `h100_pcie`
- Blackwell: `b200`, `b300`, `gb200`, `gb300`
- Jetson Thor: `jetson-t5000`, `jetson-t4000`

The hardware model already includes memory tiers, peak FLOPS, SM resources,
occupancy limits, and pipeline stages. The missing pieces are calibration,
structured provenance, and more precise modeling of copy/L2/HBM, barriers,
TMA/WGMMA/TMEM, and warp specialization.

### 2.5 Verified Baseline

Commands run on this snapshot:

```bash
pytest
compass list operators
compass list hardware
compass analyze matmul --hardware a100 --dtype fp16 --M 4096 --N 4096 --K 4096
compass analyze matmul --hardware h100 --dtype fp16 --mode pipeline --M 4096 --N 4096 --K 4096 --format json
```

Observed results:

- `pytest`: `390 passed in 37.03s`.
- The default pytest run currently includes vendored `3rdparty/SOLAR/tests`.
- `compass list operators` discovers 6 operators.
- `compass list hardware` discovers 15 hardware targets.
- A100 FP16 4096^3 hierarchy roofline reports 440.5 us, 312.0 TFLOPS, and
  bottleneck `compute`.
- H100 FP16 4096^3 pipeline JSON succeeds and selects
  `hopper_128x256x64_w8_s3`, but the JSON output is too large because it emits
  the full schedule by default.

## 3. Main Gaps

Current high-level gaps:

1. Pipeline scheduling is not yet a true DAG/resource scheduler.
2. Pipeline stage semantics are inferred from string names.
3. Memory timing does not explicitly split copy engine, L2, and HBM resources.
4. Hopper/Blackwell TMA, WGMMA, TMEM, and warp specialization are coarse
   approximations.
5. Non-matmul operators do not have pipeline templates.
6. Hardware parameters lack structured provenance and calibration data.
7. CLI/API/Web output is not yet product-quality for large schedules and
   confidence reporting.
8. Tests lack API/UI, non-matmul, golden summary, calibration, lint, and type
   gates.
9. There is no operator performance result database for validating model
   accuracy.
10. There is no pure data-driven learned estimator trained from measured
   results.

## 4. Detailed Analysis

### 4.1 Data Model

Strengths:

- Dataclasses are simple and serializable.
- `PipelineConfig`, `PipelineKernelCandidate`, `SubOp`, `TilingInfo`, and
  `PipelineSchedule` are the right conceptual building blocks.
- Hardware models already expose enough resource metadata for basic occupancy
  and candidate rejection.

Gaps:

- `DataType.byte_size` is typed as `int` but returns `0.5` for FP4/INT4.
- `PipelineStage` lacks explicit kind, resource, and work-unit fields.
- Stage classification relies on name matching such as `"mma" in stage.name`.
- `SubOp.depends_on` is a simple name list and cannot express edge type,
  loop-carried dependencies, or barrier dependencies.
- `AnalysisResult` lacks diagnostics such as confidence, assumptions,
  fallback reason, model version, and missing features.

Recommended changes:

- Add explicit stage/resource enums:
  - `WorkUnit`: `bytes`, `flops`, `fma`, `cycles`, `items`
  - `StageKind`: `global_memory`, `l2`, `copy_engine`, `shared_memory`,
    `register_file`, `tensor_core`, `cuda_core`, `sync`, `tmem`
  - `OverlapClass`: `blocking`, `async_copy`, `compute`, `store`, `sync`
- Extend `PipelineStage` with `kind`, `work_unit`, `resource_name`,
  `can_overlap`, `queue_depth`, and `source`.
- Add `AnalysisDiagnostics` or equivalent fields:
  - `model_version`
  - `confidence_level`
  - `warnings`
  - `fallback_reason`
  - `assumptions`
  - `missing_features`

### 4.2 Analyzer and Roofline

Gaps:

- `compute_model.py` and `memory_model.py` are not consistently used by
  `Analyzer`.
- `hierarchy_roofline` mostly uses the slowest memory tier, so the hierarchy
  semantics are underdeveloped.
- Missing dims often become zero silently.
- Overlap is a coarse hardware-level flag, not a resource-level schedule.

Recommended changes:

- Add structured input validation for required dimensions and positive values.
- Add `memory_hierarchy_breakdown`.
- Make roofline output clearly distinguish theoretical bound from modeled and
  calibrated estimates.

### 4.3 Pipeline Scheduler

Strengths:

- Captures prologue, steady state, and epilogue.
- Separates full latency cost from throughput-only steady-state cost.
- Stage count affects prefetch distance, shared memory, and occupancy.
- Candidate search evaluates multiple feasible matmul candidates.

Gaps:

- The file claims DAG-based scheduling, but the implementation still groups
  recurring sub-ops into load, shared-load, MMA/FMA, and epilogue groups.
- `SubOp.depends_on` is not the scheduling source of truth.
- Resource occupancy is per stage name rather than per independent hardware
  resource.
- No explicit copy-engine/L2/HBM split.
- No explicit cp.async/TMA/WGMMA/mbarrier/syncthreads costs.
- Hopper/Blackwell warp specialization is metadata, not a real producer and
  consumer warp-group schedule.
- Full pipeline JSON output is too large.

Recommended changes:

- Introduce a generic DAG/resource scheduler.
- Instantiate per-iteration nodes from sub-op templates.
- Track resource availability per explicit resource.
- Make latency policy explicit for prologue, steady state, and epilogue.
- Add schedule summary and repeated-pattern compression.

### 4.4 Operators

Pipeline implementation priority:

1. Convolution as implicit GEMM.
2. Reduction.
3. LayerNorm/RMSNorm.
4. Flash attention.
5. Fused matmul epilogues.
6. Mixed precision variants.

Rationale:

- Convolution can reuse matmul infrastructure.
- Reduction and layernorm validate that the scheduler is no longer matmul-only.
- Flash attention is high value but should wait until generic resource
  scheduling and memory modeling are stronger.

### 4.5 Hardware Models

Recommended changes:

- Add hardware field provenance:
  - field
  - value
  - source
  - assumption
  - confidence
  - last verified date
- Add consistency tests between Python hardware definitions and SOLAR YAML.
- Keep raw specs separate from empirical calibration overlays.

### 4.6 External Work and Polyhedral Relevance

External references are summarized in:

- [related_work_and_data_driven_perf_modeling.md](../reference/related_work_and_data_driven_perf_modeling.md)

Main takeaways:

- Roofline and Nsight Compute validate the bound-plus-measured-counter
  direction.
- CUTLASS Profiler, Triton, and DeepBench motivate an operator-level measured
  performance database.
- AutoTVM, Ansor, MetaSchedule, Learned TPU Cost Model, TpuGraphs, nn-Meter,
  Habitat, Ithemal, and related systems validate a data-driven estimator path.
- Polyhedral systems such as Pluto, Polly, MLIR Affine, ISL, Tensor
  Comprehensions, and Tiramisu are relevant mainly as representation and
  legality-checking inspiration.

Polyhedral relevance to OpCompass:

- Useful for representing iteration domains, access maps, dependence relations,
  tiling, fusion, and loop ordering.
- Useful for feature extraction for learned models: tile volume, reuse distance,
  parallel dimensions, reduction dimensions, memory footprint, and dependence
  distance.
- Not sufficient as a full GPU performance model because GPU-specific behavior
  still requires resource timing, occupancy, TMA/WGMMA/TMEM modeling, and
  synchronization costs.

Recommended path:

- Use polyhedral concepts in V0.3's scheduler IR.
- Keep ISL/MLIR integration optional.
- Use polyhedral-derived features in the learned estimator.
- Do not make full polyhedral code generation a near-term requirement.

## 5. Recommended Roadmap

### V0.2: Modeling Contract and Output Hygiene

Goal:

Make the current matmul model honest, typed enough, and easier to consume
without changing the core numerical behavior.

Scope:

- Add diagnostics, confidence, warnings, and fallback reason.
- Add compact schedule output.
- Configure pytest to run first-party tests by default.
- Add Pydantic API models.
- Add input validation.
- Add explicit `PipelineStage` type/resource fields with backward-compatible
  inference.

Acceptance criteria:

- First-party tests pass.
- `pytest` no longer collects `3rdparty/SOLAR/tests` by default.
- Full schedule is opt-in.
- Pipeline fallback is visible in CLI/API/Web.
- Representative matmul numbers do not change unexpectedly.

Suggested PRs:

- Pytest scope and API tests.
- Diagnostics and serialization.
- Schedule output controls.
- Stage typing migration.

### V0.3: Generic DAG/Resource Scheduler

Goal:

Replace fixed matmul stage grouping with a true dependency/resource scheduler.

Scope:

- Define `PipelineNode`, `PipelineEdge`, `ResourceUsage`, and `LoopSpec`.
- Add edge types: data, resource, barrier, loop-carried.
- Convert matmul sub-ops to nodes.
- Implement list scheduling over dependencies and resources.
- Preserve prologue/steady/epilogue latency policy.
- Add a non-matmul toy DAG test.

Acceptance criteria:

- Scheduler no longer hardcodes `load_subs`, `shared_load_subs`, `mma_subs`.
- `SubOp.depends_on` participates in scheduling.
- Matmul results stay within an agreed tolerance unless a documented bug is
  fixed.

### V0.4: Memory Resources and Synchronization Costs

Goal:

Make Ampere, Hopper, and Blackwell architectural differences operational.

Scope:

- Split copy engine, L2, and HBM timing.
- Add cp.async, TMA, WGMMA, mbarrier, and syncthreads costs.
- Add architecture-specific copy/MMA paths.
- Improve L2 reuse with CTA order and reuse-distance approximations.

Acceptance criteria:

- Results can distinguish copy-engine-bound, L2-bound, HBM-bound,
  tensor-core-bound, and sync-bound cases.
- Small-K and skinny GEMM cases show visible barrier/tail effects.

### V0.5: First Non-Matmul Pipeline Operators

Goal:

Prove that the scheduler is general enough for non-matmul operators.

Priority:

1. Convolution via implicit GEMM.
2. Reduction.
3. LayerNorm/RMSNorm.

Acceptance criteria:

- Each operator has FLOP/IO tests, pipeline decomposition tests, CLI/API tests,
  and confidence/limitation output.
- Implementations do not rely on matmul-only stage ordering.

### V0.6: Flash Attention, Fused Epilogues, and Mixed Precision

Goal:

Cover high-value LLM-style fused kernels.

Scope:

- Flash attention phases: QK, online softmax, PV, output write.
- Fused matmul epilogues: bias, activation, residual, scaling, conversion,
  quantization/dequantization.
- Mixed precision: input dtype, accumulator dtype, output dtype.

Acceptance criteria:

- Flash attention reports phase-level breakdown.
- CLI/API can select fused epilogues.
- A single `dtype` no longer has to represent all precision roles.

### V0.7: Performance Result Database and Calibration

Goal:

Move from theoretical estimates to validated and calibrated estimates.

Scope:

- Performance result database schema.
- Benchmark harness.
- CUTLASS Profiler importer.
- Triton benchmark provider.
- Nsight Compute counter importer.
- Calibration overlay.
- Confidence scoring.

Minimum measured-result schema:

- operator
- hardware
- dtype/mixed precision
- shape
- provider
- kernel name
- tile/schedule metadata
- software stack
- runtime distribution
- profiler counters
- OpCompass prediction version
- prediction error

Acceptance criteria:

- Measurement data can be imported without changing code.
- Each measurement can be linked to an OpCompass model version.
- CLI/API can show raw theoretical, modeled, and calibrated estimates.
- Parser/overlay tests work without a GPU.

### V0.8: Pure Data-Driven Learned Estimator and Ensemble

Goal:

Add a learned estimator that predicts runtime from measured data and complements
the analytical/pipeline/SOLAR modes.

Scope:

- Dataset loader from `perf_results`.
- Feature schema and artifact versioning.
- Tabular baseline models:
  - log-linear regression
  - random forest
  - histogram gradient boosting
- Structured model path:
  - MLP over engineered features
  - DeepSets over sub-ops
  - GNN over schedule DAG
- Uncertainty and out-of-distribution detection.
- `mode=learned`.
- Future `mode=ensemble`.

Features:

- operator family
- shape scalars
- dtype/layout
- hardware specs
- provider/kernel family
- tile/stage/warp/candidate metadata
- roofline prediction
- pipeline summary
- polyhedral-derived features

Metrics:

- MAPE
- median absolute percentage error
- p90 absolute percentage error
- log-RMSE
- candidate ranking accuracy
- top-k candidate recall

Acceptance criteria:

- A small fixture dataset can train/evaluate without a GPU.
- CLI/API support `mode=learned`.
- Output includes runtime, interval, dataset version, model version, OOD score,
  and disagreement with pipeline.
- Candidate ranking metrics are reported.

### V0.9: Productization and Documentation

Goal:

Make OpCompass easier to install, integrate, validate, and use.

Scope:

- Packaging extras: base, dev, solar, learned.
- Versioned API under `/api/v1`.
- Stable response schemas.
- Hardware provenance reporting.
- Web comparison workflows.
- Documentation:
  - user guide
  - operator authoring guide
  - hardware target guide
  - modeling limitations guide
  - calibration guide
  - performance result database guide
  - learned estimator guide
- CI:
  - unit tests
  - API tests
  - optional SOLAR tests
  - lint/type checks

Acceptance criteria:

- API response schemas are versioned.
- Web UI supports side-by-side comparison.
- Docs clearly state what each mode can and cannot claim.

## 6. Prioritized Backlog

### P0

- First-party-only pytest default.
- Diagnostics/confidence/fallback fields.
- Compact schedule JSON.
- API analyze request/response schema.
- API route tests.
- Explicit `PipelineStage.kind`, `work_unit`, and `resource_name`.
- Documentation correction for the current non-DAG scheduler.

### P1

- Generic DAG/resource scheduler.
- Polyhedral-inspired loop/domain/dependence/schedule IR.
- Copy engine/L2/HBM split.
- Barrier/wait costs.
- Roofline memory hierarchy breakdown.
- Hardware/SOLAR YAML consistency tests.

### P2

- Convolution implicit-GEMM pipeline.
- Reduction pipeline.
- LayerNorm/RMSNorm pipeline.
- Fused matmul epilogue.
- Mixed precision.

### P3

- Flash attention pipeline.
- Causal attention.
- Grouped-query/multi-query attention.
- Batched/grouped GEMM.
- Persistent/split-K/stream-K.

### P4

- Performance result database.
- Benchmark harness.
- Nsight ingestion.
- Calibration overlay.
- Data-driven learned estimator.
- Ensemble prediction.
- Web comparison workflow.
- Versioned API.

## 7. Recommended Next Sprint

The next sprint should implement V0.2 rather than adding a new operator.

Suggested issue list:

1. Configure pytest with `testpaths = ["tests"]`.
2. Add API tests.
3. Add diagnostics fields.
4. Surface pipeline fallback reason.
5. Add compact schedule output and opt-in full schedule.
6. Add Pydantic analyze request/response models.
7. Add `PipelineStage` type/resource fields with compatibility inference.
8. Update README and the pipeline design doc.

Expected result:

- Faster and more focused tests.
- Smaller and clearer API output.
- Visible confidence and fallback information.
- A cleaner contract for the scheduler rewrite.

## 8. Risks

Scheduler rewrite changes matmul numbers.

- Mitigation: add golden summary tests first and compare old/new scheduler
  outputs during migration.

Hardware constants become a debate sink.

- Mitigation: separate raw specs, assumptions, provenance, and calibration
  overlays.

Flash attention is implemented before the scheduler is ready.

- Mitigation: implement convolution/reduction/layernorm first.

Learned estimator overfits early data.

- Mitigation: require shape/hardware/provider holdout tests and OOD reporting.

Polyhedral scope becomes too large.

- Mitigation: use polyhedral concepts for IR/features first; keep full
  polyhedral codegen optional and out of the critical path.

## 9. Definition of Done for the Long-Term Roadmap

OpCompass is mature when:

- Pipeline mode supports matmul, convolution, reduction/layernorm, and flash
  attention.
- Scheduler uses explicit dependencies and resources.
- Memory model distinguishes copy engine, L2, and HBM.
- Hopper/Blackwell model TMA, WGMMA, TMEM, and barriers explicitly enough to
  explain major bottlenecks.
- Results include confidence and known limitations.
- Hardware specs include provenance.
- Measured result database supports validation and calibration.
- Learned estimator supports runtime prediction with uncertainty.
- CLI/API/Web can compare analytical, pipeline, calibrated, and learned
  estimates.
- Tests cover units, API behavior, golden summaries, optional SOLAR,
  calibration parsing, and learned-estimator fixtures.

