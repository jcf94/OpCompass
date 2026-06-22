"""NVIDIA Ampere architecture hardware definitions.

Based on the NVIDIA A100 Tensor Core GPU Architecture whitepaper (V1.0).
Covers Compute Capability 8.0 GPUs: A100, A40, A10, and other Ampere SKUs.

Architecture overview
---------------------
The Ampere GA10x GPU family is built on TSMC 7 nm N7 (GA100) and Samsung
8 nm 8N (GA102/GA104).  The GA100 GPU has 54.2 billion transistors on an
826 mm² die.

SM (Streaming Multiprocessor) architecture
------------------------------------------
Each SM contains:

**Execution resources**
- 4 warp schedulers  (each dispatches 1 instruction / clock)
- 64 FP32 (CUDA) cores  →  64 FP32 FMA / clock / SM
- 32 FP64 cores          →  32 FP64 FMA / clock / SM  (GA100 only; GA102/GA104
  have reduced FP64)
- 64 INT32 cores         →  64 INT32 ops / clock / SM (separate datapath)
- 4 third-generation Tensor Cores  →  up to 1024 FP16 FMA / clock / SM
- 4 load/store units (LSU)
- 4 special function units (SFU)

**On-chip memory**
- Register file: 256 KB (65536 × 32-bit registers) per SM
- L1 data cache + shared memory: 192 KB combined, configurable:
    - Shared memory up to 164 KB (vs 96 KB on V100)
    - Remaining capacity serves as L1 data cache
- Constant cache, texture cache (read-only)

**Threading / occupancy**
- 32 threads per warp
- Max 64 warps / 2048 threads resident per SM
- Max 32 thread blocks per SM
- Max 255 registers per thread, 65536 registers per block
- Max thread block size: 1024 threads

**Concurrent / parallel execution**
- FP32 and INT32 operations issue simultaneously (separate datapaths)
- Tensor Core MMA can overlap with:
    - Asynchronous copy (global → shared memory, bypasses L1 + RF)
    - Independent FP32/INT32 operations on other warps
    - Shared memory load/store on other warps
- Async copy engine: loads data directly from global memory (HBM / L2)
  into shared memory *without* going through L1 cache or the register
  file.  This saves register file bandwidth and storage, and allows
  computation to proceed during the transfer.
- Async barriers: hardware-accelerated in shared memory; separate
  "arrive" and "wait" phases enable producer-consumer pipelines.
- Warp-level reduction: single-step hardware reduce (ADD, MIN, MAX,
  AND, OR, XOR) — replaces the 5-step SHFL sequence on prior archs.
- Multi-Instance GPU (MIG): up to 7 isolated GPU instances (GA100 only),
  each with dedicated memory bandwidth, L2 slices, and SMs.

Tensor Core details (3rd generation)
------------------------------------
- 4 Tensor Cores per SM
- Each TC executes 256 FP16/FP32 mixed-precision FMA per clock
  → 1024 dense FP16 FMA / clock / SM  (2× Volta SM throughput)
- Supported formats with per-SM throughput:
    FP16   : 1024 FMA/clk  →  2048 FP16 ops/clk
    BF16   : 1024 FMA/clk  →  same as FP16
    TF32   :  512 FMA/clk  →  1024 TF32 ops/clk
    FP64   :   64 FMA/clk  →   128 FP64 ops/clk  (GA100 only)
    INT8   : 2048 OPS/clk  →  2048 INT8 ops/clk
    INT4   : 4096 OPS/clk  →  4096 INT4 ops/clk
    Binary : 8192 OPS/clk  →  8192 Binary ops/clk
- 2:4 fine-grained structured sparsity doubles effective throughput
  for FP16, BF16, TF32, INT8, INT4, and Binary.
- Matrix shape per TC per clock: 8×4×8 (FP16) vs V100's 4×4×4.
- Data sharing across all 32 threads in a warp (vs 8 on V100),
  reducing register file accesses by 2.9× vs V100.

Key Ampere improvements over Volta (V100)
-----------------------------------------
- 2.5× FP16 Tensor TFLOPS  (312 vs 125 on A100)
- 10× TF32 vs V100 FP32     (156 vs 15.7 on A100)
- 2.5× FP64 Tensor TFLOPS  (19.5 vs 7.8 on A100)
- 6.7× larger L2 cache      (40 MB vs 6 MB on A100)
- 1.7× larger shared memory (164 KB vs 96 KB)
- Async copy (global → shared, bypassing L1 + RF)
- Async barriers in shared memory
- Warp-level reduction (single-step)
- Fine-grained structured sparsity (2:4)
- Multi-Instance GPU (MIG)
- L2 cache residency controls
"""

from __future__ import annotations

from opcompass.hardware.base import Hardware
from opcompass.models import (
    ComputeUnit,
    DataType,
    MemoryHierarchy,
    MemoryTier,
    PipelineStage,
)


class NvidiaAmpere(Hardware):
    """Base class for NVIDIA Ampere architecture GPUs (Compute Capability 8.0).

    Provides the common SM microarchitecture parameters, pipeline stages,
    and resource counts shared across all Ampere-based GPUs (A100, A40,
    A10, etc.).  This class is NOT registered as a standalone hardware
    target — only its concrete subclasses are.

    Subclasses need to provide:

    * ``name`` — short id, e.g. ``"a100"``
    * ``description`` — human-readable summary
    * ``memory`` — :class:`MemoryHierarchy` with SKU-specific tiers
    * ``compute_unit`` — :class:`ComputeUnit` (use :meth:`_make_compute_unit`
      with the SKU's SM count, clock, and peak FLOPs)

    Architecture-level constants available to subclasses:

    * ``register_file_kb``, ``shared_memory_max_kb``, ``l1_shared_combined_kb``
    * ``warp_schedulers_per_unit``, ``tensor_cores_per_unit``,
      ``fp32_cores_per_unit``, ``fp64_cores_per_unit``, ``int32_cores_per_unit``
    * ``ldst_units``, ``sfu_units``
    * ``max_concurrent_warps``, ``max_threads_per_unit``,
      ``max_thread_blocks_per_unit``, ``max_registers_per_thread``,
      ``max_registers_per_block``
    * ``can_concurrent_fp32_int32``, ``threads_per_warp``
    """

    # NOTE: name is intentionally left as "" so the auto-discovery registry
    # skips this intermediate base class (it checks for a non-empty name).
    vendor = "NVIDIA"

    # ── Architecture identity ────────────────────────────────────────
    architecture = "Ampere"
    sm_version = "8.0"          # Compute Capability

    # ── Per-SM memory resources (common across all Ampere GPUs) ──────
    register_file_kb = 256       # 65536 × 32-bit registers
    shared_memory_max_kb = 164   # Configurable up to 164 KB
    l1_shared_combined_kb = 192  # L1 + shared memory pool

    # ── Per-SM execution resources ───────────────────────────────────
    warp_schedulers_per_unit = 4
    tensor_cores_per_unit = 4    # 3rd-gen, 256 FP16 FMA/clk each
    fp32_cores_per_unit = 64     # Also used for FP16 non-Tensor
    fp64_cores_per_unit = 32     # GA100; GA102/GA104 may differ
    int32_cores_per_unit = 64    # Separate datapath from FP32
    ldst_units = 4               # Load/store units
    sfu_units = 4                # Special function units

    # ── Threading / occupancy limits ─────────────────────────────────
    max_concurrent_warps = 64
    max_threads_per_unit = 2048   # 64 warps × 32 threads
    max_thread_blocks_per_unit = 32
    max_registers_per_thread = 255
    max_registers_per_block = 65536

    # ── Parallel / concurrent execution capabilities ─────────────────
    can_concurrent_fp32_int32 = True   # Separate FP32 + INT32 datapaths
    threads_per_warp = 32

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    @classmethod
    def _ampere_pipeline(cls) -> list[PipelineStage]:
        """Return the common Ampere SM pipeline stages.

        Pipeline order: global_read → async_copy_load → shared_load →
        mma → fma_alu → shared_store → global_write.

        Notes for subclasses
        --------------------
        * The ``async_copy_load`` stage is Ampere-specific (bypasses L1
          and the register file).  If a consumer-grade Ampere SKU does
          not expose async copy, override or omit this stage.
        * ``throughput_per_cycle`` values are per-SM.  The pipeline model
          scales them by SM count automatically.
        """
        return [
            # ── Memory: HBM → L2 → L1 → registers ──────────────────
            PipelineStage(
                name="global_read",
                latency_cycles=300,
                throughput_per_cycle=64,        # bytes/cycle per SM (L1 → RF)
                description="HBM → L2 → L1 → registers (traditional load path)",
            ),

            # ── Memory: HBM/L2 → shared memory (async copy) ────────
            PipelineStage(
                name="async_copy_load",
                latency_cycles=300,
                throughput_per_cycle=64,        # bytes/cycle per SM (shared mem write BW)
                description=(
                    "HBM/L2 → shared memory (async copy; "
                    "bypasses L1 and register file)"
                ),
            ),

            # ── Memory: shared memory → registers ──────────────────
            PipelineStage(
                name="shared_load",
                latency_cycles=20,
                throughput_per_cycle=128,       # bytes/cycle per SM (32 banks × 4 B)
                description="Shared memory → registers (per-warp load)",
            ),

            # ── Compute: Tensor Core matrix multiply-accumulate ─────
            PipelineStage(
                name="mma",
                latency_cycles=8,
                # 4 TCs × 256 FP16 FMA/TC/clk = 1024 FMA/clk/SM
                # Other precisions: TF32=512, FP64=64 FMA/clk/SM
                throughput_per_cycle=1024,
                description=(
                    "Matrix multiply-accumulate (3rd-gen Tensor Core). "
                    "FP16/BF16: 1024 FMA/clk/SM. "
                    "TF32: 512, FP64: 64 FMA/clk/SM. "
                    "2× with 2:4 structured sparsity."
                ),
            ),

            # ── Compute: CUDA core FMA (FP32 / FP64) ───────────────
            PipelineStage(
                name="fma_alu",
                latency_cycles=4,
                # 64 FP32 cores × 1 FMA/clk = 64 FMA/clk/SM
                throughput_per_cycle=64,
                description=(
                    "FP32/FP64 fused multiply-add on CUDA cores. "
                    "Runs simultaneously with INT32 operations."
                ),
            ),

            # ── Memory: registers → shared memory ──────────────────
            PipelineStage(
                name="shared_store",
                latency_cycles=20,
                throughput_per_cycle=128,       # bytes/cycle per SM
                description="Registers → shared memory (per-warp store)",
            ),

            # ── Memory: registers → HBM (via L2) ───────────────────
            PipelineStage(
                name="global_write",
                latency_cycles=300,
                throughput_per_cycle=64,        # bytes/cycle per SM (L2 write BW)
                description="Registers → L2 → HBM (write-back path)",
            ),
        ]

    @classmethod
    def _make_compute_unit(
        cls,
        count: int,
        clock_mhz: float,
        peak_flops: dict[DataType, float],
        **overrides,
    ) -> ComputeUnit:
        """Create a :class:`ComputeUnit` pre-filled with Ampere SM defaults.

        Parameters
        ----------
        count:
            Number of SMs on the full chip (e.g. 108 for A100).
        clock_mhz:
            Boost / typical clock frequency in MHz.
        peak_flops:
            Per-dtype peak FLOPS/OPS on the full chip.
        **overrides:
            Any additional keyword arguments are forwarded to the
            :class:`ComputeUnit` constructor, allowing SKU-specific
            overrides of the Ampere defaults (e.g. reduced FP64 cores
            on GA102/GA104 consumer SKUs).
        """
        kwargs: dict = dict(
            name="SM",
            count=count,
            clock_mhz=clock_mhz,
            peak_flops=peak_flops,
            pipeline=cls._ampere_pipeline(),

            # ── Occupancy ──────────────────────────────────────────
            max_concurrent_warps=cls.max_concurrent_warps,

            # ── Per-SM memory resources ────────────────────────────
            register_file_kb=cls.register_file_kb,
            shared_memory_max_kb=cls.shared_memory_max_kb,
            l1_shared_combined_kb=cls.l1_shared_combined_kb,

            # ── Per-SM execution resources ─────────────────────────
            warp_schedulers_per_unit=cls.warp_schedulers_per_unit,
            tensor_cores_per_unit=cls.tensor_cores_per_unit,
            fp32_cores_per_unit=cls.fp32_cores_per_unit,
            fp64_cores_per_unit=cls.fp64_cores_per_unit,
            int32_cores_per_unit=cls.int32_cores_per_unit,
            ldst_units=cls.ldst_units,
            sfu_units=cls.sfu_units,

            # ── Threading limits ───────────────────────────────────
            max_threads_per_unit=cls.max_threads_per_unit,
            max_thread_blocks_per_unit=cls.max_thread_blocks_per_unit,
            max_registers_per_thread=cls.max_registers_per_thread,
            max_registers_per_block=cls.max_registers_per_block,

            # ── Parallel execution capabilities ────────────────────
            can_concurrent_fp32_int32=cls.can_concurrent_fp32_int32,
            threads_per_warp=cls.threads_per_warp,
        )
        kwargs.update(overrides)
        return ComputeUnit(**kwargs)


class NvidiaA100(NvidiaAmpere):
    """NVIDIA A100 80 GB (SXM4) — GA100 GPU, Ampere architecture.

    Full-chip peak (108 SMs @ 1410 MHz):
        FP16 Tensor Core:  312 TFLOPS (624 w/ sparsity)
        TF32 Tensor Core:  156 TFLOPS (312 w/ sparsity)
        FP64 Tensor Core:   19.5 TFLOPS
        FP32 CUDA Core:     19.5 TFLOPS
        INT8 Tensor Core:  624 TOPS  (1248 w/ sparsity)
    """

    name = "a100"
    description = "NVIDIA A100 80GB SXM4 — Ampere, 108 SM, 312 TFLOPS FP16"

    # ── Memory hierarchy ──────────────────────────────────────────────

    memory = MemoryHierarchy(
        tiers=[
            MemoryTier(
                name="HBM2e",
                capacity_bytes=80 * 1024**3,          # 80 GB
                bandwidth_bytes_per_sec=2.0e12,       # ~2.0 TB/s
            ),
            MemoryTier(
                name="L2",
                capacity_bytes=40 * 1024**2,          # 40 MB
                # Peak read BW = 5120 B/clk × 1410 MHz = 7.22 TB/s.
                # Effective BW ≈ 5.0 TB/s after protocol overheads.
                bandwidth_bytes_per_sec=5.0e12,
            ),
        ],
        # Async copy engine and DMA engines can move data between
        # HBM2e and SMs while the SMs are computing.
        can_overlap_with_compute={"HBM2e"},
    )

    # ── Compute unit ──────────────────────────────────────────────────

    compute_unit = NvidiaAmpere._make_compute_unit(
        count=108,
        clock_mhz=1410,

        # Peak FLOPs / OPS per data type (full chip).
        #
        # Uses the *maximum achievable* throughput for each dtype:
        #   - FP16, BF16, TF32, INT8 → Tensor Core peak
        #   - FP32, FP64 → CUDA-core peak (Tensor Cores accelerate
        #     these only via TF32 / FP64 MMA instructions)
        #
        # CUDA-core (non-Tensor) peaks for reference:
        #   FP64:  9.7 TFLOPS    FP32: 19.5 TFLOPS
        #   FP16: 78   TFLOPS    BF16: 39   TFLOPS    INT32: 19.5 TOPS
        peak_flops={
            DataType.FP64:  9.7e12,     # CUDA core
            DataType.FP32: 19.5e12,     # CUDA core
            DataType.TF32: 156e12,      # Tensor Core: 512 FMA/clk/SM × 108 × 1410 MHz × 2
            DataType.FP16: 312e12,      # Tensor Core: 1024 FMA/clk/SM × 108 × 1410 MHz × 2
            DataType.BF16: 312e12,      # Tensor Core: same rate as FP16
            DataType.INT8: 624e12,      # Tensor Core: 2048 ops/clk/SM × 108 × 1410 MHz
        },
    )
