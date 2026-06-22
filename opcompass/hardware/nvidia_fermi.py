"""NVIDIA Fermi architecture hardware definitions.

Based on the NVIDIA Fermi Compute Architecture Whitepaper (V1.1).
Covers Compute Capability 2.0 GPUs: GF100 (GTX 480, Tesla C2050/C2070).

Architecture overview
---------------------
The GF100 GPU is fabricated on TSMC 40 nm, with 3.0 billion transistors.
It represents the first complete GPU computing architecture with full
IEEE 754-2008 FP64 support, ECC memory, and a unified L2 cache.

SM architecture
---------------
Each SM contains:

**Execution resources**
- 2 warp schedulers (dual-issue: 1 instruction from each warp per clock)
- 2 instruction dispatch units
- 32 FP32 (CUDA) cores  → 32 FP32 FMA / clock / SM
- 16 FP64 cores         → 16 FP64 FMA / clock / SM (half-rate)
- 16 load/store units
- 4 special function units (transcendentals, etc.)
- **No separate INT32 datapath** — integer ops execute on CUDA cores
- **No Tensor Cores**

**On-chip memory**
- Register file: 128 KB (32768 × 32-bit registers) per SM
- 64 KB on-chip SRAM configurable as:
    - 48 KB shared memory + 16 KB L1 cache
    - 16 KB shared memory + 48 KB L1 cache

**Threading / occupancy**
- 32 threads per warp
- Max 48 warps / 1536 threads resident per SM
- Max 8 thread blocks per SM
- Max 63 registers per thread
- Max thread block size: 1024 threads

**Concurrent execution**
- Dual warp issue: two warps execute concurrently each clock
- FP64 instructions cannot dual-issue with any other operation
- FP32 and INT32 share the same datapath (NOT concurrent)

Key Fermi innovations (first generation of modern GPU compute)
--------------------------------------------------------------
- First GPU with full IEEE 754-2008 FP64 support
- First GPU with ECC memory protection
- Unified L2 cache (768 KB) shared across all SMs
- C++ support (virtual functions, recursion, new/delete)
- Configurable L1/shared memory split
- Dual warp scheduler for improved utilization
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


class NvidiaFermi(Hardware):
    """Base class for NVIDIA Fermi architecture GPUs (Compute Capability 2.0).

    Provides the common SM microarchitecture parameters, pipeline stages,
    and resource counts shared across all Fermi GPUs (GF100, GF104, etc.).
    This class is NOT registered as a standalone hardware target — only
    its concrete subclasses are.
    """

    vendor = "NVIDIA"

    # ── Architecture identity ────────────────────────────────────────
    architecture = "Fermi"
    sm_version = "2.0"          # Compute Capability

    # ── Per-SM memory resources ──────────────────────────────────────
    register_file_kb = 128       # 32768 × 32-bit registers
    shared_memory_max_kb = 48    # Max configurable (alt: 16 KB with 48 KB L1)
    l1_max_kb = 48               # Max configurable (alt: 16 KB with 48 KB shared)
    l1_shared_combined_kb = 64   # Total on-chip SRAM pool

    # ── Per-SM execution resources ───────────────────────────────────
    warp_schedulers_per_unit = 2
    tensor_cores_per_unit = 0    # No Tensor Cores (introduced in Volta)
    fp32_cores_per_unit = 32
    fp64_cores_per_unit = 16     # Half-rate FP64
    int32_cores_per_unit = 0     # Shared datapath with FP32 (not separate)
    ldst_units = 16
    sfu_units = 4

    # ── Threading / occupancy limits (CC 2.0) ───────────────────────
    max_concurrent_warps = 48
    max_threads_per_unit = 1536   # 48 warps × 32 threads
    max_thread_blocks_per_unit = 8
    max_registers_per_thread = 63
    max_registers_per_block = 32768  # Entire register file per SM

    # ── Parallel / concurrent execution capabilities ─────────────────
    can_concurrent_fp32_int32 = False  # Shared datapath
    threads_per_warp = 32

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    @classmethod
    def _fermi_pipeline(cls) -> list[PipelineStage]:
        """Return the common Fermi SM pipeline stages.

        Simple pipeline: global_read → fma_alu → global_write.
        No async copy, no tensor cores, no separate shared memory stage
        (shared memory is accessed through the same LD/ST path as L1).
        """
        return [
            PipelineStage(
                name="global_read",
                latency_cycles=400,
                throughput_per_cycle=16,        # bytes/cycle per SM (16 LD/ST units × 1 B?)
                description="GDDR5 → L2 → L1/Shared → registers (load path)",
            ),
            PipelineStage(
                name="fma_alu",
                latency_cycles=22,              # Fermi FP32 pipeline depth
                throughput_per_cycle=32,        # FMA ops/cycle per SM (32 FP32 cores)
                description="FP32/FP64 fused multiply-add on CUDA cores.",
            ),
            PipelineStage(
                name="global_write",
                latency_cycles=400,
                throughput_per_cycle=16,        # bytes/cycle per SM
                description="Registers → L2 → GDDR5 (write-back path)",
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
        """Create a :class:`ComputeUnit` pre-filled with Fermi SM defaults."""
        kwargs: dict = dict(
            name="SM",
            count=count,
            clock_mhz=clock_mhz,
            peak_flops=peak_flops,
            pipeline=cls._fermi_pipeline(),

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


class NvidiaGF100(NvidiaFermi):
    """NVIDIA Tesla C2050/C2070 — GF100 GPU, Fermi architecture.

    Full-chip (16 SMs @ 575 MHz — Tesla C2050 reference clock):
        FP32 CUDA Core:  589 GFLOPS
        FP64 CUDA Core:  294 GFLOPS

    The GF100 full die has 16 SMs.  Consumer SKUs like GTX 480 ship with
    15 SMs enabled and a higher 700 MHz clock, yielding ~672 GFLOPS FP32.
    """

    name = "gf100"
    description = "NVIDIA GF100 (Tesla C2050) — Fermi, 16 SM, 589 GFLOPS FP32"

    # ── Memory hierarchy ──────────────────────────────────────────────

    memory = MemoryHierarchy(
        tiers=[
            MemoryTier(
                name="GDDR5",
                capacity_bytes=3 * 1024**3,           # 3 GB (Tesla C2050)
                bandwidth_bytes_per_sec=144e9,        # 144 GB/s (384-bit @ 3 Gbps)
            ),
            MemoryTier(
                name="L2",
                capacity_bytes=768 * 1024,            # 768 KB unified L2
                bandwidth_bytes_per_sec=300e9,        # Approximate
            ),
        ],
        # Fermi does not have async copy engines
        can_overlap_with_compute=set(),
    )

    # ── Compute unit ──────────────────────────────────────────────────

    compute_unit = NvidiaFermi._make_compute_unit(
        count=16,
        clock_mhz=575,              # Tesla C2050 shader clock

        peak_flops={
            # 16 SMs × 16 FP64 cores/SM × 2 FMA × 0.575 GHz = 294.4 GFLOPS
            DataType.FP64: 294.4e9,
            # 16 SMs × 32 FP32 cores/SM × 2 FMA × 0.575 GHz = 588.8 GFLOPS
            DataType.FP32: 588.8e9,
        },
    )
