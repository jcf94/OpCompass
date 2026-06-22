"""NVIDIA Volta GV100 architecture hardware definitions.

Based on the NVIDIA Tesla V100 Architecture Whitepaper (WP-08608-001_v1.1).
Covers Compute Capability 7.0 GPUs: Tesla V100 (GV100).

Architecture overview
---------------------
The GV100 GPU is fabricated on TSMC 12 nm FFN, with 21.1 billion
transistors on an 815 mm² die.  Volta is a landmark architecture that
introduced Tensor Cores, independent FP32+INT32 datapaths, and the
first unified L1/shared memory pool.

SM architecture
---------------
The GV100 SM is partitioned into 4 processing blocks, each containing:

**Execution resources per processing block**
- 1 warp scheduler + 1 dispatch unit
- 16 FP32 (CUDA) cores
- 8 FP64 cores
- 16 INT32 cores (NEW: separate datapath, concurrent with FP32)
- 2 Tensor Cores (1st gen — NEW)
- 1 L0 instruction cache

**Per-SM totals**
- 4 warp schedulers
- 64 FP32 (CUDA) cores  →  64 FP32 FMA / clock / SM
- 32 FP64 cores          →  32 FP64 FMA / clock / SM
- 64 INT32 cores         →  separate datapath, concurrent with FP32
- 8 Tensor Cores (1st gen) → 512 FMA / clock / SM (FP16)
- 4 texture units

**On-chip memory**
- Register file: 256 KB (65536 × 32-bit registers) per SM
- L1 data cache + shared memory: 128 KB combined pool per SM
    - Shared memory configurable up to 96 KB (up from 64 KB in Pascal)
    - Remaining capacity serves as L1 data cache

**Threading / occupancy** (CC 7.0)
- 32 threads per warp
- Max 64 warps / 2048 threads resident per SM
- Max 32 thread blocks per SM
- Max 255 registers per thread
- Max thread block size: 1024 threads

**Tensor Core details (1st generation)**
- 8 Tensor Cores per SM
- Each TC executes 64 FP16 FMA / clock → 512 FMA / clock / SM
- Mixed-precision: FP16 input, FP32 accumulate
- FP16 dense: 125 TFLOPS (V100, 80 SM)
- INT8 inference: ~62 TOPS (not explicitly stated in whitepaper)

Key Volta innovations (over Pascal)
-----------------------------------
- Tensor Cores (1st gen): 125 TFLOPS FP16 (12× Pascal P100 FP32)
- Independent INT32 datapath (concurrent with FP32)
- Unified L1/shared memory: 128 KB configurable pool
- Independent thread scheduling (per-thread PC, SIMT model enhancement)
- FP32 FMA issue latency reduced to 4 cycles (from 6 in Pascal)
- HBM2: 900 GB/s on V100 (vs 720 GB/s on P100)
- 6 MB L2 cache (vs 4 MB in GP100)
- NVLink 2.0: 300 GB/s total (6 links × 25 GB/s/dir × 2)
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


class NvidiaVolta(Hardware):
    """Base class for NVIDIA Volta GV100 architecture GPUs (CC 7.0).

    Provides the common SM microarchitecture parameters, pipeline stages,
    and resource counts shared across Volta-based GPUs.
    This class is NOT registered as a standalone hardware target.

    Subclasses need to provide:
    * ``name`` — short id, e.g. ``"v100"``
    * ``description`` — human-readable summary
    * ``memory`` — :class:`MemoryHierarchy` with SKU-specific tiers
    * ``compute_unit`` — via :meth:`_make_compute_unit`
    """

    vendor = "NVIDIA"

    # ── Architecture identity ────────────────────────────────────────
    architecture = "Volta"
    sm_version = "7.0"          # Compute Capability

    # ── Per-SM memory resources ──────────────────────────────────────
    register_file_kb = 256       # 65536 × 32-bit registers
    shared_memory_max_kb = 96    # Configurable up to 96 KB
    l1_shared_combined_kb = 128  # L1 + shared memory unified pool

    # ── Per-SM execution resources ───────────────────────────────────
    warp_schedulers_per_unit = 4
    tensor_cores_per_unit = 8    # 1st-gen, 64 FP16 FMA/clk each
    fp32_cores_per_unit = 64
    fp64_cores_per_unit = 32     # 1:2 ratio
    int32_cores_per_unit = 64    # NEW: Separate datapath from FP32
    ldst_units = 16              # Estimated
    sfu_units = 4                # Estimated

    # ── Threading / occupancy limits (CC 7.0) ───────────────────────
    max_concurrent_warps = 64
    max_threads_per_unit = 2048
    max_thread_blocks_per_unit = 32
    max_registers_per_thread = 255
    max_registers_per_block = 65536

    # ── Parallel / concurrent execution capabilities ─────────────────
    can_concurrent_fp32_int32 = True   # Volta introduced separate INT32 datapath!
    threads_per_warp = 32

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    @classmethod
    def _volta_pipeline(cls) -> list[PipelineStage]:
        """Return the common Volta GV100 SM pipeline stages.

        Pipeline: global_read → shared_load → mma → fma_alu →
        shared_store → global_write.

        Volta introduces the first Tensor Core (mma) stage.  The mma
        stage can overlap with memory operations on other warps.
        """
        return [
            PipelineStage(
                name="global_read",
                latency_cycles=300,
                throughput_per_cycle=64,        # bytes/cycle per SM (L1 → RF bandwidth)
                description="HBM2 → L2 → L1/Shared → registers (load path)",
            ),
            PipelineStage(
                name="shared_load",
                latency_cycles=20,
                throughput_per_cycle=128,       # bytes/cycle per SM
                description="Shared memory → registers (per-warp load)",
            ),
            PipelineStage(
                name="mma",
                latency_cycles=8,
                # 8 TCs × 64 FP16 FMA/TC/clk = 512 FMA/clk/SM
                throughput_per_cycle=512,
                description=(
                    "Matrix multiply-accumulate (1st-gen Tensor Core).  "
                    "FP16: 512 FMA/clk/SM.  "
                    "8 Tensor Cores per SM, each 64 FMA/clk."
                ),
            ),
            PipelineStage(
                name="fma_alu",
                latency_cycles=4,               # Reduced from Pascal's 6
                throughput_per_cycle=64,        # FMA ops/cycle per SM (64 FP32 cores)
                description=(
                    "FP32/FP64 fused multiply-add on CUDA cores.  "
                    "Runs simultaneously with INT32 operations "
                    "(separate datapaths, new in Volta)."
                ),
            ),
            PipelineStage(
                name="shared_store",
                latency_cycles=20,
                throughput_per_cycle=128,       # bytes/cycle per SM
                description="Registers → shared memory (per-warp store)",
            ),
            PipelineStage(
                name="global_write",
                latency_cycles=300,
                throughput_per_cycle=64,        # bytes/cycle per SM
                description="Registers → L2 → HBM2 (write-back path)",
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
        """Create a :class:`ComputeUnit` pre-filled with Volta SM defaults."""
        kwargs: dict = dict(
            name="SM",
            count=count,
            clock_mhz=clock_mhz,
            peak_flops=peak_flops,
            pipeline=cls._volta_pipeline(),

            max_concurrent_warps=cls.max_concurrent_warps,

            register_file_kb=cls.register_file_kb,
            shared_memory_max_kb=cls.shared_memory_max_kb,
            l1_shared_combined_kb=cls.l1_shared_combined_kb,

            warp_schedulers_per_unit=cls.warp_schedulers_per_unit,
            tensor_cores_per_unit=cls.tensor_cores_per_unit,
            fp32_cores_per_unit=cls.fp32_cores_per_unit,
            fp64_cores_per_unit=cls.fp64_cores_per_unit,
            int32_cores_per_unit=cls.int32_cores_per_unit,
            ldst_units=cls.ldst_units,
            sfu_units=cls.sfu_units,

            max_threads_per_unit=cls.max_threads_per_unit,
            max_thread_blocks_per_unit=cls.max_thread_blocks_per_unit,
            max_registers_per_thread=cls.max_registers_per_thread,
            max_registers_per_block=cls.max_registers_per_block,

            can_concurrent_fp32_int32=cls.can_concurrent_fp32_int32,
            threads_per_warp=cls.threads_per_warp,
        )
        kwargs.update(overrides)
        return ComputeUnit(**kwargs)


class NvidiaV100(NvidiaVolta):
    """NVIDIA Tesla V100 (SXM2) — GV100 GPU, Volta architecture.

    Full-chip peak (80 SMs @ 1530 MHz GPU Boost):
        FP64 CUDA Core:   7.8 TFLOPS
        FP32 CUDA Core:  15.7 TFLOPS
        FP16 Tensor Core: 125 TFLOPS

    The full GV100 die has 84 SMs (6 GPCs × 7 TPCs).
    Tesla V100 ships with 80 SMs enabled.  TDP: 300 W.

    V100 was later offered in a 32 GB HBM2 variant as well.
    """

    name = "v100"
    description = "NVIDIA Tesla V100 — Volta, 80 SM, 125 TFLOPS FP16 Tensor"

    # ── Memory hierarchy ──────────────────────────────────────────────

    memory = MemoryHierarchy(
        tiers=[
            MemoryTier(
                name="HBM2",
                capacity_bytes=16 * 1024**3,          # 16 GB (4 stacks; 32 GB variant later)
                bandwidth_bytes_per_sec=900e9,        # 900 GB/s (4096-bit interface)
            ),
            MemoryTier(
                name="L2",
                capacity_bytes=6 * 1024**2,           # 6 MB L2 (6144 KB)
                bandwidth_bytes_per_sec=3.0e12,       # Approximate
            ),
        ],
        can_overlap_with_compute={"HBM2"},
    )

    # ── Compute unit ──────────────────────────────────────────────────

    compute_unit = NvidiaVolta._make_compute_unit(
        count=80,
        clock_mhz=1530,             # Tesla V100 GPU Boost clock

        peak_flops={
            # 80 SMs × 32 FP64 cores × 2 FMA × 1.53 GHz = 7834 GFLOPS
            DataType.FP64: 7834e9,
            # 80 SMs × 64 FP32 cores × 2 FMA × 1.53 GHz = 15667 GFLOPS
            DataType.FP32: 15667e9,
            # 80 SMs × 8 TCs × 64 FMA/TC × 2 ops/FMA × 1.53 GHz = 125,338 GFLOPS
            DataType.FP16: 125.3e12,
            # TF32 not supported (introduced in Ampere)
        },
    )
