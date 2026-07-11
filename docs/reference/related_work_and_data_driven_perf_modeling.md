# Related Work and Reference Notes for OpCompass

Date: 2026-07-09

This document collects external work relevant to OpCompass: analytical
performance bounds, GPU profiler workflows, kernel benchmark tools, tensor
compiler cost models, measured performance datasets, and data-driven runtime
prediction. The goal is not to copy any one system, but to identify what
OpCompass should borrow, avoid, or integrate with.

## 1. Summary

OpCompass currently combines roofline-style estimation, a matmul-first pipeline
model, and optional SOLAR graph analysis. Related work suggests two important
extensions:

1. Build an operator performance result database. This database should store
   measured runtimes, profiler counters, kernel metadata, hardware metadata,
   and OpCompass predictions so analytical models can be validated and
   calibrated.
2. Add a purely data-driven estimation path. A learned model can predict
   operator runtime from shape, dtype, hardware, implementation, and profiler
   features. It should complement, not replace, analytical models.

Recommended end state:

- `hierarchy_roofline`: fast theoretical lower bound.
- `pipeline`: interpretable resource/pipeline model.
- `solar`: graph/fusion-aware estimate through SOLAR.
- `calibrated_pipeline`: pipeline model plus empirical calibration overlay.
- `learned`: data-driven latency predictor trained from measured runs.
- `ensemble`: combined result with confidence and disagreement reporting.

## 2. Analytical and Roofline Models

### 2.1 Roofline

Source:

- [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
- [ACM DOI page](https://dl.acm.org/doi/10.1145/1498765.1498785)

What it does:

- Relates attainable performance to operational intensity and hardware ceilings.
- Gives an upper bound and bottleneck intuition rather than a detailed kernel
  schedule.

Relevance to OpCompass:

- OpCompass `hierarchy_roofline` is directly aligned with this family.
- The current model should make the roofline contract explicit: it is a bound,
  not a calibrated runtime prediction.
- Hierarchical rooflines motivate exposing HBM, L2, shared memory, and compute
  ceilings separately.

Implementation implications:

- Keep roofline mode simple and stable.
- Add per-tier operational intensity and per-tier ceilings.
- Report whether a value is a lower bound, modeled estimate, or calibrated
  estimate.

### 2.2 NVIDIA Nsight Compute Speed Of Light and Roofline

Sources:

- [NVIDIA Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- [NVIDIA Nsight Compute product page](https://developer.nvidia.com/nsight-compute)
- [NVIDIA Developer Blog: Nsight Compute Roofline Analysis](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/)

What it does:

- Provides measured kernel-level GPU utilization, Speed Of Light sections,
  roofline charts, and detailed counters.
- Supports CLI/UI workflows and post-processing.

Relevance to OpCompass:

- Nsight Compute should be the primary source of validation counters.
- Useful counters include runtime, DRAM bytes, L2 hit rate, SM throughput,
  tensor-core utilization, shared-memory throughput, occupancy, achieved FLOPS,
  and stall reasons.

Implementation implications:

- Add an import path for Nsight Compute CSV or JSON-like exports.
- Store raw counters and derived metrics in the performance result database.
- Compare OpCompass bottleneck predictions with Nsight Compute SOL bottlenecks.

## 3. Kernel Libraries, Profilers, and Benchmark Sources

### 3.1 NVIDIA CUTLASS and CUTLASS Profiler

Sources:

- [CUTLASS overview](https://docs.nvidia.com/cutlass/latest/overview.html)
- [CUTLASS Profiler documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html)
- [CUTLASS performance profiling wiki](https://github.com/NVIDIA/cutlass/wiki/Performance-Profiling)

What it does:

- Provides CUDA C++ template abstractions for GEMM and related computations.
- Includes a profiler that can execute GEMM, sparse GEMM, Conv2d, and Conv3d
  kernels from the CUTLASS instance library.

Relevance to OpCompass:

- CUTLASS is the best practical reference for matmul/convolution candidate
  catalogs, tile names, stage counts, warp specialization, TMA, WGMMA, and
  epilogue variants.
- CUTLASS profiler output can seed OpCompass's measured result database.

Implementation implications:

- Add a `cutlass_profiler` ingestion script.
- Store provider, operation kind, kernel name, tile description, dtype,
  accumulator type, runtime, FLOPS, bandwidth, and verification status.
- Use CUTLASS candidate names to calibrate OpCompass candidate search.

### 3.2 Triton

Sources:

- [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton programming model introduction](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)
- [Triton benchmarking API: `triton.testing.do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)
- [Triton paper](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

What it does:

- Expresses tiled GPU kernels in a block/program-level language.
- Provides autotuning and benchmarking utilities.

Relevance to OpCompass:

- Triton kernels can provide an accessible source of measured operator results,
  especially for matmul, reductions, layernorm, and attention-like kernels.
- The matmul tutorial highlights program reordering for L2 locality, which maps
  directly to OpCompass's planned CTA-order/L2 reuse model.

Implementation implications:

- Add a `triton_bench` provider in the performance database.
- Capture meta-parameters such as block sizes, number of warps, number of
  stages, and program order.
- Use Triton kernels to generate data for learned models when CUTLASS does not
  cover an operator variant.

### 3.3 DeepBench

Sources:

- [DeepBench GitHub repository](https://github.com/baidu-research/deepbench)
- [DeepBench project page](https://svail.github.io/DeepBench/)

What it does:

- Benchmarks low-level deep learning operations across hardware and libraries.
- Covers the idea that the same operation family can be compute-bound,
  bandwidth-bound, or occupancy-bound depending on shape and implementation.

Relevance to OpCompass:

- DeepBench is a useful historical model for an operator-level performance
  corpus.
- Its scope is closer to OpCompass than full-model benchmarks because it focuses
  on primitive operations.

Implementation implications:

- Design OpCompass performance results around operator instances, not only full
  neural networks.
- Preserve library/provider identity because performance is implementation
  dependent.

### 3.4 MLPerf

Sources:

- [MLPerf Inference: Datacenter](https://mlcommons.org/benchmarks/inference-datacenter/)
- [MLPerf Inference results repositories](https://github.com/mlcommons)

What it does:

- Standardizes full-system inference benchmarking across models, scenarios, and
  submitters.

Relevance to OpCompass:

- MLPerf is not operator-level enough for direct pipeline model validation.
- It is still useful for future end-to-end sanity checks and hardware metadata.

Implementation implications:

- Do not use MLPerf as the first calibration source for operator models.
- Use it later to validate aggregation from operator estimates to model-level
  estimates.

## 4. Tensor Compiler Cost Models and Auto-Schedulers

### 4.0 Polyhedral Compilation

Sources:

- [A Survey of General-purpose Polyhedral Compilers](https://dl.acm.org/doi/10.1145/3674735)
- [Pluto compiler project](https://pluto-compiler.sourceforge.net/)
- [Pluto GitHub repository](https://github.com/bondhugula/pluto)
- [MLIR Affine dialect documentation](https://mlir.llvm.org/docs/Dialects/Affine/)
- [Integer Set Library manual](https://libisl.sourceforge.io/user.html)
- [ISL manual PDF](https://libisl.sourceforge.io/manual.pdf)
- [Polly LLVM project](https://polly.llvm.org/)
- [Tensor Comprehensions paper](https://arxiv.org/pdf/1802.04730)
- [Tensor Comprehensions announcement](https://research.facebook.com/blog/2018/2/announcing-tensor-comprehensions/)
- [Tiramisu paper](https://commit.csail.mit.edu/papers/2018/tiramisu_paper.pdf)
- [Tiramisu compiler project](https://tiramisu-compiler.org/)

What it does:

- Represents static-control loop nests with affine iteration domains,
  dependence relations, access functions, and schedules.
- Enables legality checks and transformations such as loop interchange, fusion,
  skewing, tiling, parallelization, and locality optimization.
- Tools such as Pluto and Polly focus on general loop optimization; MLIR Affine
  brings polyhedral-style abstractions into a modern compiler IR; Tensor
  Comprehensions and Tiramisu show polyhedral ideas applied to tensor and deep
  learning kernels.

Relevance to OpCompass:

- Polyhedral methods are most useful as a schedule representation and legality
  layer, not as a complete GPU performance estimator.
- OpCompass's planned generic scheduler needs a precise way to represent:
  - iteration domains
  - affine shape-dependent bounds
  - producer/consumer dependencies
  - tiling and loop ordering
  - fusion legality
  - data reuse distance features
- These concepts map well to future OpCompass objects such as `PipelineNode`,
  `PipelineEdge`, `LoopSpec`, and `PipelineKernelCandidate`.

What OpCompass should borrow:

- A schedule IR based on domains, access maps, and schedule maps.
- Explicit dependence legality checks before accepting a transformed candidate.
- Structured tiling/fusion/interchange metadata instead of free-form strings.
- Feature extraction from affine domains for learned models:
  - tile volume
  - reuse distance
  - parallel dimensions
  - reduction dimensions
  - memory footprint per tile
  - dependence distance vectors

What OpCompass should not over-adopt initially:

- Full automatic polyhedral code generation.
- Full ISL dependency as a mandatory runtime dependency.
- Attempting to express all GPU-specific behavior, such as WGMMA wait groups,
  TMA barriers, occupancy, and warp specialization, purely as affine schedules.

Recommended integration path:

1. Use polyhedral concepts in OpCompass's internal schema without requiring ISL.
2. Add an optional `polyhedral` extra later, possibly using `islpy` or MLIR
   exports, for advanced legality/reuse analysis.
3. Use affine features as inputs to the data-driven estimator.
4. Keep GPU resource timing in OpCompass's pipeline/resource model.

Design implication:

Polyhedral compilation should influence OpCompass V0.3 and V0.8:

- V0.3 generic scheduler should use explicit loop/domain/dependence objects.
- V0.8 learned estimator should include polyhedral-derived schedule features.
- A future code-generation path may use MLIR Affine/Triton/TIR, but the near
  term objective is performance estimation, not automatic kernel generation.

### 4.1 AutoTVM

Sources:

- [Learning to Optimize Tensor Programs](https://papers.nips.cc/paper/2018/file/8b5700012be65c9da25f49408d959ca0-Paper.pdf)
- [arXiv page](https://arxiv.org/abs/1805.08166)

What it does:

- Uses a learned statistical cost model to guide search over tensor operator
  implementations.
- Focuses on ranking candidate schedules during tuning rather than producing a
  human-readable analytical bottleneck explanation.

Relevance to OpCompass:

- Validates the idea of a learned operator cost model.
- Transfer across workloads is important for limiting measurement cost.

Implementation implications:

- OpCompass's learned model should support both regression and ranking:
  - regression for runtime estimate
  - ranking for candidate selection
- Store candidate-level features, not just operator-level features.

### 4.2 Ansor / TVM Auto-Scheduler

Sources:

- [Ansor paper](https://www.usenix.org/system/files/osdi20-zheng.pdf)
- [arXiv page](https://arxiv.org/abs/2006.06762)
- [TVM auto-scheduler introduction](https://tvm.apache.org/2021/03/03/intro-auto-scheduler)

What it does:

- Samples tensor programs from a hierarchical search space.
- Uses evolutionary search and a learned cost model.
- Measures promising programs on real hardware and feeds measurements back into
  the search.

Relevance to OpCompass:

- Strong template for a measurement loop:
  generate candidates, predict, run top candidates, update model.
- Shows that a learned model and real hardware measurement should be connected
  iteratively.

Implementation implications:

- Add an active-learning workflow:
  - sample shape/candidate space
  - predict uncertainty
  - measure high-value points
  - retrain
- Track dataset splits by hardware and operator family.

### 4.3 TVM MetaSchedule

Sources:

- [Tensor Program Optimization with Probabilistic Programs](https://proceedings.neurips.cc/paper_files/paper/2022/file/e894eafae43e68b4c8dfdacf742bcbf3-Paper-Conference.pdf)
- [OpenReview entry](https://openreview.net/forum?id=nyCr6-0hinG)

What it does:

- Represents search-space construction with probabilistic programs.
- Uses an end-to-end learning-driven framework for tensor program optimization.

Relevance to OpCompass:

- Reinforces the need for a structured candidate representation.
- Suggests that OpCompass should not hardcode only a tiny matmul catalog forever.

Implementation implications:

- Treat `PipelineKernelCandidate` as a feature-rich schedule/config object.
- Keep analytical constraints and learned ranking separate.

### 4.4 Hidet

Sources:

- [Hidet paper PDF](https://arxiv.org/pdf/2210.09603)
- [Amazon Science publication page](https://www.amazon.science/publications/hidet-task-mapping-programming-paradigm-for-deep-learning-tensor-programs)

What it does:

- Embeds scheduling into tensor programs with task mappings.
- Explicitly targets optimizations such as double buffering.

Relevance to OpCompass:

- Shows why high-performance tensor kernels need more than declarative loop
  schedules.
- Relevant to OpCompass's plan to model producer/consumer warp groups and
  software pipeline stages.

Implementation implications:

- Add schedule structure that can express mapping and ordering, not just tile
  sizes.

### 4.5 Rammer

Sources:

- [USENIX OSDI page](https://www.usenix.org/conference/osdi20/presentation/ma)
- [Rammer paper PDF](https://web.eecs.umich.edu/~mosharaf/Readings/Rammer.pdf)

What it does:

- Uses static spatial-temporal scheduling for DNN execution.
- Focuses on graph-level scheduling and hardware utilization.

Relevance to OpCompass:

- Less directly operator-level, but useful for future fusion and multi-kernel
  scheduling.

Implementation implications:

- Keep OpCompass operator-level first, but avoid designing APIs that prevent
  graph-level/fusion extensions later.

## 5. Learned Runtime and Performance Prediction

### 5.1 Learned TPU Cost Model and Learned TPU Performance Model

Sources:

- [Learned TPU Cost Model for XLA Tensor Programs](https://research.google/pubs/learned-tpu-cost-model-for-xla-tensor-programs/)
- [Paper PDF](https://mlforsystems.org/assets/papers/neurips2019/learned_tpu_kaufman_2019.pdf)
- [A Learned Performance Model for the Tensor Processing Unit](https://research.google/pubs/a-learned-performance-model-for-the-tensor-processing-unit/)
- [Paper PDF](https://proceedings.mlsys.org/paper_files/paper/2021/file/6bcfac823d40046dca25ef6d6d59cc3f-Paper.pdf)

What it does:

- Trains learned models over tensor computation programs/subgraphs to predict
  runtime or cost.
- Demonstrates that learned models can compete with highly engineered analytical
  compiler cost models.

Relevance to OpCompass:

- Strong evidence for adding a data-driven estimator.
- Shows that graph/subgraph structure can matter, not only scalar shape
  features.

Implementation implications:

- Start with operator-level tabular models, but keep the schema compatible with
  graph features and schedule DAG features.
- Use analytical-model outputs as input features to the learned model.

### 5.2 TpuGraphs Dataset

Sources:

- [TpuGraphs arXiv HTML](https://arxiv.org/html/2308.13490v3)
- [OpenXLA discussion of TpuGraphs dataset](https://groups.google.com/a/openxla.org/g/openxla-discuss/c/8MfJkxCXX9U)

What it does:

- Provides a performance prediction dataset of tensor programs represented as
  computational graphs running on TPUs.

Relevance to OpCompass:

- Shows what a public performance dataset can look like.
- Useful as a conceptual template for schema design.

Implementation implications:

- Store graph/schedule structure and compiler configuration, not only runtime.
- Include train/validation/test splits that prevent shape leakage.

### 5.3 Ithemal and Neural Code Comprehension

Sources:

- [Ithemal paper page](https://proceedings.mlr.press/v97/mendis19a.html)
- [Ithemal arXiv page](https://arxiv.org/abs/1808.07412)
- [Neural Code Comprehension NeurIPS page](https://proceedings.neurips.cc/paper/2018/hash/17c3433fecc21b57000debdf7ad5c930-Abstract.html)
- [Neural Code Comprehension arXiv page](https://arxiv.org/abs/1806.07336)

What it does:

- Learns performance-related properties from code or instruction sequences.
- Ithemal predicts basic-block throughput from assembly-like inputs.
- Neural Code Comprehension learns code representations from IR and applies
  them to performance-related tasks.

Relevance to OpCompass:

- Useful if OpCompass later ingests generated CUDA/PTX/SASS/Triton IR.
- Not the right first model for OpCompass because current operators are better
  represented by structured shape/schedule/hardware features.

Implementation implications:

- Keep a future path for code/IR features.
- Do not start with raw code embeddings; first build a clean tabular/structured
  dataset.

### 5.4 nn-Meter

Sources:

- [nn-Meter GitHub repository](https://github.com/microsoft/nn-Meter)
- [nn-Meter paper PDF](https://air.tsinghua.edu.cn/pdf/nn-Meter-Towards-Accurate-Latency-Prediction-of-Deep-Learning-Model-Inference-on-Diverse-Edge-Devices.pdf)
- [ACM DOI page](https://dl.acm.org/doi/10.1145/3458864.3467882)

What it does:

- Predicts DNN inference latency by decomposing models into kernel-level units
  and using kernel-level prediction.

Relevance to OpCompass:

- Very aligned with OpCompass's operator-level philosophy.
- Provides a precedent for using kernel-level predictors as building blocks for
  model-level latency.

Implementation implications:

- Build OpCompass result DB around kernel/operator instances.
- Later aggregate operator predictions into graph/model predictions.

### 5.5 Habitat

Sources:

- [USENIX ATC page](https://www.usenix.org/conference/atc21/presentation/yu)
- [Habitat GitHub repository](https://github.com/geoffxy/habitat)

What it does:

- Predicts DNN training iteration time on a target GPU using runtime
  measurements on a source GPU and GPU scaling features.

Relevance to OpCompass:

- Useful for cross-hardware transfer and adaptation.
- Suggests a practical path for predicting on expensive or unavailable target
  GPUs from cheaper source measurements.

Implementation implications:

- Add transfer-learning experiments:
  - train on A100/H100
  - adapt to B200/Jetson with limited measurements
- Include hardware feature ratios as learned-model inputs.

### 5.6 Daydream and Paleo

Sources:

- [Daydream USENIX ATC page](https://www.usenix.org/conference/atc20/presentation/zhu-hongyu)
- [Daydream paper PDF](https://www.usenix.org/system/files/atc20-zhu-hongyu.pdf)
- [Paleo OpenReview page](https://openreview.net/forum?id=SyVVJ85lg)
- [Paleo GitHub repository](https://github.com/TalwalkarLab/paleo)

What they do:

- Daydream estimates DNN optimization effects using traces and graph
  transformations.
- Paleo is an analytical performance model for deep neural network systems.

Relevance to OpCompass:

- More model/system-level than operator-level.
- Useful references for future end-to-end model aggregation and optimization
  what-if analysis.

Implementation implications:

- Keep operator result DB compatible with future graph-level aggregation.
- Add trace IDs and graph node IDs when measurements come from full workloads.

## 6. Proposed OpCompass Performance Result Database

### 6.1 Purpose

The performance result database should support:

- Validating analytical estimates.
- Calibrating hardware/resource factors.
- Training data-driven estimators.
- Comparing providers such as cuBLAS, CUTLASS, Triton, PyTorch, and custom
  kernels.
- Tracking regressions across OpCompass model versions.

### 6.2 Data Granularity

Use one row per measured operator/kernel run, with optional repeated samples.

Recommended hierarchy:

- `hardware`: machine/GPU identity and raw specs.
- `software_stack`: driver, CUDA, library, compiler, framework versions.
- `operator_case`: operator family, shape, dtype, layout, attributes.
- `implementation`: provider, kernel name, tile/schedule metadata.
- `measurement_run`: timestamp, warmup, repeat count, runtime distribution.
- `profiler_counters`: optional Nsight/CUPTI counters.
- `opcompass_prediction`: predictions from each model version/mode.

### 6.3 Minimum Schema

Required fields:

- `case_id`
- `operator`
- `hardware_name`
- `gpu_name_raw`
- `dtype`
- `shape_json`
- `layout`
- `provider`
- `kernel_name`
- `runtime_us_mean`
- `runtime_us_p50`
- `runtime_us_p90`
- `runtime_us_std`
- `warmup_iters`
- `measure_iters`
- `software_stack_json`
- `timestamp`

Recommended fields:

- `runtime_us_min`
- `runtime_us_max`
- `achieved_tflops`
- `dram_bytes`
- `l2_read_bytes`
- `l2_write_bytes`
- `l2_hit_rate`
- `sm_efficiency`
- `tensor_core_utilization`
- `shared_load_bytes`
- `shared_store_bytes`
- `occupancy`
- `active_warps`
- `registers_per_thread`
- `shared_memory_per_block`
- `block_m`
- `block_n`
- `block_k`
- `stage_count`
- `warp_count`
- `cta_order`
- `opcompass_mode`
- `opcompass_model_version`
- `predicted_runtime_us`
- `prediction_error_pct`

### 6.4 Storage Format

Start simple:

- `docs/reference/` for design notes only.
- `benchmarks/results/` or `data/perf_results/` for local measured files later.
- Use Parquet or SQLite for structured datasets.
- Use JSONL for append-only raw imports.
- Keep large measured datasets out of git unless explicitly approved.

Recommended initial layout:

```text
data/
  perf_results/
    raw/
      cutlass_profiler/
      triton_bench/
      nsight_compute/
    processed/
      op_results.parquet
      op_results.sqlite
    schemas/
      op_result.schema.json
      profiler_counter.schema.json
```

### 6.5 Measurement Providers

Priority order:

1. CUTLASS profiler for GEMM, sparse GEMM, Conv2d, Conv3d.
2. cuBLAS/cuDNN microbenchmarks for vendor-library baselines.
3. Triton benchmarks for custom/flexible operators.
4. PyTorch eager/compiled benchmarks for framework-level baselines.
5. Nsight Compute imports for counter-rich calibration samples.

## 7. Proposed Pure Data-Driven Estimator

### 7.1 Goal

Add a learned estimator that predicts operator runtime directly from data. It
should answer:

- What is the expected runtime for this operator/hardware/shape/provider?
- How uncertain is the prediction?
- Which existing measured cases are nearest neighbors?
- Does the learned prediction disagree with analytical/pipeline estimates?

### 7.2 Model Variants

Phase 1: tabular baseline.

- Models:
  - linear/log-linear regression
  - random forest
  - gradient boosted trees, such as XGBoost/LightGBM if allowed later
  - scikit-learn histogram gradient boosting
- Features:
  - operator family
  - shape scalars
  - dtype
  - layout
  - hardware specs
  - provider/kernel family
  - analytical roofline outputs
  - pipeline summary outputs
- Target:
  - `log(runtime_us)`

Phase 2: structured schedule model.

- Models:
  - MLP over engineered features
  - DeepSets over sub-ops
  - graph neural network over pipeline DAG
  - transformer over schedule tokens if the dataset becomes large
- Features:
  - sub-op graph
  - resource usage per node
  - candidate schedule metadata
  - hardware resource embeddings

Phase 3: transfer and uncertainty.

- Techniques:
  - hardware embeddings
  - multi-task learning by operator/hardware
  - fine-tuning on new hardware
  - conformal prediction or quantile regression
  - ensemble disagreement

### 7.3 Training and Evaluation

Recommended splits:

- Random split for basic sanity.
- Shape holdout split to test interpolation/extrapolation.
- Hardware holdout split to test transfer.
- Operator holdout split only after there are enough operator families.
- Provider holdout split to test implementation generalization.

Metrics:

- MAPE
- median absolute percentage error
- p90 absolute percentage error
- RMSE on log runtime
- pairwise ranking accuracy for candidate selection
- top-k candidate recall
- calibration error for uncertainty intervals

### 7.4 Integration With Existing Modes

Add a new mode:

```text
learned
```

And a combined mode later:

```text
ensemble
```

Result output should include:

- `learned_runtime_us`
- `learned_error_interval`
- `nearest_training_cases`
- `training_dataset_version`
- `model_artifact_version`
- `feature_schema_version`
- `out_of_distribution_score`
- `disagreement_with_pipeline_pct`

### 7.5 Why This Complements Analytical Modeling

Analytical models are interpretable and can extrapolate from hardware facts, but
they miss hidden implementation details. Learned models can absorb empirical
effects from real runs, but they need coverage and can fail out of distribution.

The best OpCompass design is hybrid:

- Analytical roofline gives a fast bound.
- Pipeline gives interpretable resource reasoning.
- Measured results validate and calibrate.
- Learned model predicts empirical runtime where enough data exists.
- Ensemble reports disagreement and confidence.

## 8. Roadmap Impact

The existing OpCompass roadmap should be extended as follows:

- V0.7 should become "Performance Result Database and Calibration".
- V0.8 should become "Learned Estimator and Ensemble Prediction".
- Productization can move to V0.9 if needed.

Recommended immediate additions to the next roadmap revision:

- Add `data/perf_results` schema design.
- Add benchmark provider abstraction.
- Add CUTLASS profiler importer.
- Add Triton benchmark runner.
- Add Nsight Compute counter importer.
- Add first tabular learned estimator baseline.
- Add model registry fields for learned estimator artifacts.
- Add validation dashboards comparing roofline, pipeline, calibrated, and
  learned predictions.
