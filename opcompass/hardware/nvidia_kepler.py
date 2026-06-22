"""NVIDIA Kepler GK110 architecture hardware definitions.

Based on the NVIDIA Kepler GK110 Architecture Whitepaper.
Covers Compute Capability 3.5 GPUs: Tesla K20/K20X/K40, GTX 780 Ti.

Architecture overview
---------------------
The GK110 GPU is fabricated on TSMC 28 nm, with 7.1 billion transistors.
It was built "first and foremost for Tesla" HPC products.  The SM was
renamed to SMX ("Streaming Multiprocessor eXtreme") to reflect the major
increase in per-SM resources.

SMX architecture
----------------
Each SMX contains:

**Execution resources**
- 4 warp schedulers, each with 2 dispatch units (8 total dispatch units)
- 192 FP32 (CUDA) cores  →  192 FP32 FMA / clock / SMX (6× Fermi SM)
- 64 FP64 cores          →   64 FP64 FMA / clock / SMX (1:3 ratio to FP32)
- 32 special function units (8× Fermi GF110)
- 32 load/store units
- **No separate INT32 datapath** — integer ops execute on CUDA cores
- **No Tensor Cores**

**On-chip memory**
- Register file: 256 KB (65536 × 32-bit registers) per SMX
- 64 KB on-chip SRAM configurable as:
    - 48 KB shared + 16 KB L1
    - 32 KB shared + 32 KB L1
    - 16 KB shared + 48 KB L1
- 48 KB read-only data cache (separate from L1/shared)

**Threading / occupancy**
- 32 threads per warp
- Max 64 warps / 2048 threads resident per SMX
- Max 16 thread blocks per SMX
- Max 255 registers per thread
- Max thread block size: 1024 threads

**Concurrent execution**
- 4 warps can be issued concurrently (up from 2 in Fermi)
- FP64 instructions can be paired with other instructions (unlike Fermi)
- GPU Boost: dynamic clock adjustment based on power/thermal headroom
- FP32 and INT32 share the same datapath (NOT concurrent)

Key Kepler GK110 innovations
----------------------------
- Dynamic Parallelism: GPU can launch child kernels autonomously
- Hyper-Q: 32 simultaneous MPI streams (vs 1 in Fermi)
- GPUDirect RDMA: direct GPU-to-GPU data transfers across nodes
- 1.5 MB L2 cache (2× Fermi's 768 KB)
- Up to 255 registers per thread (vs 63 in Fermi)
- Shader clock unified with core clock (no more 2× shader clock)
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


class NvidiaKepler(Hardware):
    """Base class for NVIDIA Kepler GK110 architecture GPUs (CC 3.5).

    Provides the common SMX microarchitecture parameters shared across
    GK110-based GPUs (Tesla K20/K20X/K40, GTX 780 Ti).
    This class is NOT registered as a standalone hardware target.
    """

    vendor = "NVIDIA"

    # ── Architecture identity ────────────────────────────────────────
    architecture = "Kepler"
    sm_version = "3.5"          # Compute Capability (GK110 specifically)

    # ── Per-SMX memory resources ─────────────────────────────────────
    register_file_kb = 256       # 65536 × 32-bit registers
    shared_memory_max_kb = 48    # Max configurable in 48/16 split
    l1_shared_combined_kb = 64   # Total on-chip SRAM pool

    # ── Per-SMX execution resources ──────────────────────────────────
    warp_schedulers_per_unit = 4
    tensor_cores_per_unit = 0    # No Tensor Cores
    fp32_cores_per_unit = 192    # 6× Fermi's 32
    fp64_cores_per_unit = 64     # 1:3 ratio to FP32 (GK110 specific)
    int32_cores_per_unit = 0     # Shared datapath with FP32
    ldst_units = 32
    sfu_units = 32               # 8× Fermi GF110

    # ── Threading / occupancy limits (CC 3.5) ───────────────────────
    max_concurrent_warps = 64
    max_threads_per_unit = 2048   # 64 warps × 32 threads
    max_thread_blocks_per_unit = 16
    max_registers_per_thread = 255
    max_registers_per_block = 65536

    # ── Parallel / concurrent execution capabilities ─────────────────
    can_concurrent_fp32_int32 = False  # Shared datapath
    threads_per_warp = 32

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    @classmethod
    def _kepler_pipeline(cls) -> list[PipelineStage]:
        """Return the common Kepler GK110 pipeline stages.

        Pipeline: global_read → fma_alu → global_write.
        Kepler doubled shared memory bandwidth to 256 B/clock per SMX
        compared to Fermi.
        """
        return [
            PipelineStage(
                name="global_read",
                latency_cycles=400,
                throughput_per_cycle=32,        # bytes/cycle per SMX (32 LD/ST units)
                description="GDDR5 → L2 → L1/Shared → registers (load path)",
            ),
            PipelineStage(
                name="fma_alu",
                latency_cycles=6,               # Kepler FP32 pipeline depth (improved from Fermi's 22)
                throughput_per_cycle=192,       # FMA ops/cycle per SMX (192 FP32 cores)
                description="FP32/FP64 fused multiply-add on CUDA cores.",
            ),
            PipelineStage(
                name="global_write",
                latency_cycles=400,
                throughput_per_cycle=32,        # bytes/cycle per SMX
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
        """Create a :class:`ComputeUnit` pre-filled with Kepler GK110 SMX defaults."""
        kwargs: dict = dict(
            name="SMX",
            count=count,
            clock_mhz=clock_mhz,
            peak_flops=peak_flops,
            pipeline=cls._kepler_pipeline(),

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


class NvidiaGK110(NvidiaKepler):
    """NVIDIA Tesla K40 — GK110 GPU, Kepler architecture.

    Full-chip (15 SMX units @ 876 MHz GPU Boost):
        FP32 CUDA Core:  5.05 TFLOPS
        FP64 CUDA Core:  1.68 TFLOPS

    The full GK110 die has 15 SMX units.  Tesla K40 ships with all 15
    enabled at 876 MHz boost.  Earlier SKUs like Tesla K20X ship with
    14 SMX at lower clocks.
    """

    name = "gk110"
    description = "NVIDIA GK110 (Tesla K40) — Kepler, 15 SMX, 5.05 TFLOPS FP32"

    # ── Memory hierarchy ──────────────────────────────────────────────

    memory = MemoryHierarchy(
        tiers=[
            MemoryTier(
                name="GDDR5",
                capacity_bytes=12 * 1024**3,          # 12 GB (Tesla K40)
                bandwidth_bytes_per_sec=288e9,        # 288 GB/s (384-bit @ 6 Gbps)
            ),
            MemoryTier(
                name="L2",
                capacity_bytes=1536 * 1024,           # 1.5 MB L2 (2× Fermi)
                bandwidth_bytes_per_sec=500e9,        # Approximate
            ),
        ],
        can_overlap_with_compute=set(),
    )

    # ── Compute unit ──────────────────────────────────────────────────

    compute_unit = NvidiaKepler._make_compute_unit(
        count=15,
        clock_mhz=876,              # Tesla K40 GPU Boost clock

        peak_flops={
            # 15 SMX × 64 FP64 cores × 2 FMA × 0.876 GHz = 1682 GFLOPS
            DataType.FP64: 1682e9,
            # 15 SMX × 192 FP32 cores × 2 FMA × 0.876 GHz = 5046 GFLOPS
            DataType.FP32: 5046e9,
        },
    )
