"""NVIDIA Turing TU102 architecture hardware definitions.

Based on the NVIDIA Turing Architecture Whitepaper (WP-09183-001_v01).
Covers Compute Capability 7.5 GPUs: Quadro RTX 6000/8000, RTX 2080 Ti.

Architecture overview
---------------------
The TU102 GPU is fabricated on TSMC 12 nm FFN, with 18.6 billion
transistors on a 754 mm² die.  Turing introduces RT Cores for real-time
ray tracing, 2nd-gen Tensor Cores with INT8/INT4 support, and unified
L1/shared memory with configurable split.

SM architecture
---------------
The Turing SM is partitioned into 4 processing blocks, each containing:

**Execution resources per processing block**
- 1 warp scheduler + 1 dispatch unit
- 16 FP32 (CUDA) cores
- 16 INT32 cores (separate datapath, concurrent with FP32)
- 2 Tensor Cores (2nd gen)
- 1 L0 instruction cache

**Per-SM totals**
- 4 warp schedulers
- 64 FP32 (CUDA) cores  →  64 FP32 FMA / clock / SM
- 64 INT32 cores         →  separate datapath, concurrent with FP32
- 8 Tensor Cores (2nd gen) → 512 FMA / clock / SM (FP16)
- 2 FP64 cores           →  severely reduced (1:32 ratio)
- 4 texture units
- 1 RT Core (ray tracing)

**On-chip memory**
- Register file: 256 KB (65536 × 32-bit registers) per SM
- L1 data cache + shared memory: 96 KB unified pool per SM
    - Configurable: 32 KB shared / 64 KB L1, or 64 KB shared / 32 KB L1
    - Graphics default: 64 KB "graphics shader RAM" (shared)

**Threading / occupancy** (CC 7.5)
- 32 threads per warp
- Max 32 warps / 1024 threads resident per SM (reduced from Volta's 64/2048)
- Max 16 thread blocks per SM
- Max 255 registers per thread
- Max thread block size: 1024 threads

**Tensor Core details (2nd generation)**
- 8 Tensor Cores per SM (same count as Volta, improved throughput)
- Per TC per clock: 64 FP16 FMA (same as Volta for FP16)
- New in Turing: INT8 (2× FP16 rate), INT4 (4× FP16 rate)
- FP16 accumulate speed equal to FP32 accumulate (Volta throttled FP32 accumulate)
- BF16 NOT supported

Key Turing innovations (over Volta)
-----------------------------------
- RT Cores for real-time ray tracing (10 GigaRays/sec on RTX 2080 Ti)
- 2nd-gen Tensor Cores with INT8 and INT4 support
- Unified L1/shared memory with configurable split (96 KB unified pool)
- GDDR6 memory: up to 672 GB/s on 384-bit bus
- Mesh Shading, Variable Rate Shading, Texture-Space Shading
- Memory bandwidth compression (delta color compression)
- Independent INT32 datapath (36% effective shader throughput uplift)
- 6 MB L2 cache on TU102 (double Pascal GP102)
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


class NvidiaTuring(Hardware):
    """Base class for NVIDIA Turing TU102 architecture GPUs (CC 7.5).

    Provides the common SM microarchitecture parameters, pipeline stages,
    and resource counts shared across Turing GPUs (RTX 2080 Ti, Quadro
    RTX 6000/8000, Tesla T4).
    This class is NOT registered as a standalone hardware target.

    Key differences from Volta: reduced max threads/SM (1024 vs 2048),
    reduced max warps/SM (32 vs 64), reduced FP64 (2/SM vs 32/SM),
    unified L1/shared (96 KB vs 128 KB), 2nd-gen Tensor Cores with
    INT8/INT4.
    """

    vendor = "NVIDIA"

    # ── Architecture identity ────────────────────────────────────────
    architecture = "Turing"
    sm_version = "7.5"          # Compute Capability

    # ── Per-SM memory resources ──────────────────────────────────────
    register_file_kb = 256       # 65536 × 32-bit registers
    shared_memory_max_kb = 64    # Max configurable as shared (64 KB shared / 32 KB L1)
    l1_shared_combined_kb = 96   # L1 + shared memory unified pool

    # ── Per-SM execution resources ───────────────────────────────────
    warp_schedulers_per_unit = 4
    tensor_cores_per_unit = 8    # 2nd-gen, 64 FP16 FMA/clk each, +INT8/INT4
    fp32_cores_per_unit = 64
    fp64_cores_per_unit = 2      # Severely reduced (1:32 ratio)
    int32_cores_per_unit = 64    # Separate datapath from FP32
    ldst_units = 16              # Estimated (one per partition)
    sfu_units = 4                # Estimated (one per partition)

    # ── Threading / occupancy limits (CC 7.5) ── NOTE: reduced vs Volta
    max_concurrent_warps = 32    # Reduced from Volta's 64
    max_threads_per_unit = 1024  # Reduced from Volta's 2048
    max_thread_blocks_per_unit = 16
    max_registers_per_thread = 255
    max_registers_per_block = 65536

    # ── Parallel / concurrent execution capabilities ─────────────────
    can_concurrent_fp32_int32 = True   # Separate INT32 datapath (same as Volta)
    threads_per_warp = 32

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    @classmethod
    def _turing_pipeline(cls) -> list[PipelineStage]:
        """Return the common Turing SM pipeline stages.

        Pipeline: global_read → shared_load → mma → fma_alu →
        shared_store → global_write.

        Turing does not have an async copy engine (that came with Ampere).
        The unified L1/shared memory is 96 KB per SM, configurable split.
        """
        return [
            PipelineStage(
                name="global_read",
                latency_cycles=300,
                throughput_per_cycle=64,        # bytes/cycle per SM
                description="GDDR6 → L2 → L1/Shared → registers (load path)",
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
                # INT8: 2× FP16, INT4: 4× FP16
                throughput_per_cycle=512,
                description=(
                    "Matrix multiply-accumulate (2nd-gen Tensor Core).  "
                    "FP16: 512 FMA/clk/SM.  "
                    "INT8: 1024 ops/clk/SM.  INT4: 2048 ops/clk/SM."
                ),
            ),
            PipelineStage(
                name="fma_alu",
                latency_cycles=4,
                throughput_per_cycle=64,        # FMA ops/cycle per SM (64 FP32 cores)
                description=(
                    "FP32/FP64 fused multiply-add on CUDA cores.  "
                    "Runs simultaneously with INT32 operations.  "
                    "FP64 severely reduced (2 DP units/SM)."
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
                description="Registers → L2 → GDDR6 (write-back path)",
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
        """Create a :class:`ComputeUnit` pre-filled with Turing SM defaults."""
        kwargs: dict = dict(
            name="SM",
            count=count,
            clock_mhz=clock_mhz,
            peak_flops=peak_flops,
            pipeline=cls._turing_pipeline(),

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


class NvidiaRTX6000(NvidiaTuring):
    """NVIDIA Quadro RTX 6000 — TU102 GPU, Turing architecture.

    Full-chip (72 SMs @ 1770 MHz GPU Boost):
        FP64 CUDA Core:       0.51 TFLOPS
        FP32 CUDA Core:      16.3 TFLOPS
        FP16 CUDA Core:      32.6 TFLOPS  (2× FP32 via packed FP16)
        FP16 Tensor Core:   130.5 TFLOPS
        INT8 Tensor Core:   261.0 TOPS
        INT4 Tensor Core:   522.0 TOPS

    The full TU102 die has 72 SMs (6 GPCs × 6 TPCs × 2 SMs/TPC).
    Quadro RTX 6000 ships with all 72 SMs enabled and 24 GB GDDR6.
    RTX 2080 Ti ships with 68 SMs (4 disabled) and 11 GB GDDR6.

    Quadro RTX 8000 is the same chip with 48 GB GDDR6 (dual-sided).
    """

    name = "rtx6000"
    description = "NVIDIA Quadro RTX 6000 — Turing, 72 SM, 130.5 TFLOPS FP16 Tensor"

    # ── Memory hierarchy ──────────────────────────────────────────────

    memory = MemoryHierarchy(
        tiers=[
            MemoryTier(
                name="GDDR6",
                capacity_bytes=24 * 1024**3,          # 24 GB (Quadro RTX 6000)
                bandwidth_bytes_per_sec=672e9,        # 672 GB/s (384-bit @ 14 Gbps)
            ),
            MemoryTier(
                name="L2",
                capacity_bytes=6 * 1024**2,           # 6 MB L2 (6144 KB)
                bandwidth_bytes_per_sec=3.0e12,       # Approximate
            ),
        ],
        can_overlap_with_compute={"GDDR6"},
    )

    # ── Compute unit ──────────────────────────────────────────────────

    compute_unit = NvidiaTuring._make_compute_unit(
        count=72,
        clock_mhz=1770,             # Quadro RTX 6000 GPU Boost clock

        peak_flops={
            # 72 SMs × 2 FP64 cores × 2 FMA × 1.77 GHz = 509.8 GFLOPS (1:32 ratio)
            DataType.FP64: 509.8e9,
            # 72 SMs × 64 FP32 cores × 2 FMA × 1.77 GHz = 16312 GFLOPS
            DataType.FP32: 16312e9,
            # Tensor Core FP16: 72 SMs × 8 TCs × 64 FMA/TC × 2 ops/FMA × 1.77 GHz = 130.5 TFLOPS
            # (CUDA-core FP16 via packed instruction is 32.6 TFLOPS — not used here;
            #  we follow the convention of reporting Tensor Core peak for FP16/BF16/INT8)
            DataType.FP16: 130.5e12,
            # INT8 Tensor Core: 2× FP16 TC rate
            DataType.INT8: 261.0e12,
            # TF32 not supported (introduced in Ampere)
            # BF16 not supported
        },
    )
