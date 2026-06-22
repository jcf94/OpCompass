"""NVIDIA Pascal GP100 architecture hardware definitions.

Based on the NVIDIA Tesla P100 Architecture Whitepaper (WP-08019-001_v01.1).
Covers Compute Capability 6.0 GPUs: Tesla P100 (GP100).

Architecture overview
---------------------
The GP100 GPU is fabricated on TSMC 16 nm FinFET, with 15.3 billion
transistors on a 610 mm² die.  Pascal is the first GPU to use HBM2
stacked memory and the first to support NVLink interconnects.

SM architecture (Pascal SM)
----------------------------
Each SM is partitioned into 2 processing blocks, each containing:

**Execution resources per processing block**
- 1 warp scheduler (dispatch 2 instructions/clock)
- 32 FP32 (CUDA) cores
- 16 FP64 cores  (1:2 ratio — Pascal maintains strong FP64)
- 8 texture units

**Per-SM totals**
- 2 warp schedulers (reduced from 4 in Maxwell, but each dispatches 2/clock)
- 64 FP32 (CUDA) cores  →  64 FP32 FMA / clock / SM (half Maxwell's count but higher clock)
- 32 FP64 cores          →  32 FP64 FMA / clock / SM (1:2 FP32:FP64 ratio)
- 4 texture units
- **No separate INT32 datapath** — shared with FP32
- **No Tensor Cores** (introduced in Volta)

**On-chip memory**
- Register file: 256 KB (65536 × 32-bit registers) per SM
- 64 KB dedicated shared memory (no longer configurable split)
- L1 cache: separate coalescing/texture cache

**Threading / occupancy** (CC 6.0)
- 32 threads per warp
- Max 64 warps / 2048 threads resident per SM
- Max 32 thread blocks per SM
- Max 255 registers per thread
- Max thread block size: 1024 threads

**FP16 support**
- FP16 throughput is 2× FP32 via paired-operation instructions (two FP16
  ops per instruction issue slot).  This is CUDA-core FP16, not Tensor Core.

Key Pascal innovations
----------------------
- HBM2 stacked memory: 720 GB/s via 4096-bit interface (3× Maxwell GM200)
- NVLink: 160 GB/s bidirectional GPU-to-GPU interconnect (4 links)
- First GPU on 16 nm FinFET process (significant power efficiency gain)
- Unified memory with 49-bit virtual addressing (512 TB)
- Page Migration Engine for seamless CPU-GPU data movement
- FP16 2× throughput for inference workloads
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


class NvidiaPascal(Hardware):
    """Base class for NVIDIA Pascal GP100 architecture GPUs (CC 6.0).

    Provides the common SM microarchitecture parameters shared across
    GP100-based GPUs (Tesla P100, P100 PCIe).
    This class is NOT registered as a standalone hardware target.
    """

    vendor = "NVIDIA"

    # ── Architecture identity ────────────────────────────────────────
    architecture = "Pascal"
    sm_version = "6.0"          # Compute Capability (GP100)

    # ── Per-SM memory resources ──────────────────────────────────────
    register_file_kb = 256       # 65536 × 32-bit registers
    shared_memory_max_kb = 64    # Dedicated shared memory (not configurable)
    l1_shared_combined_kb = 64   # Separate from L1 texture cache

    # ── Per-SM execution resources ───────────────────────────────────
    warp_schedulers_per_unit = 2
    tensor_cores_per_unit = 0    # No Tensor Cores (introduced in Volta)
    fp32_cores_per_unit = 64     # Fewer than Maxwell's 128 but clocked higher
    fp64_cores_per_unit = 32     # Strong FP64 (1:2 ratio)
    int32_cores_per_unit = 0     # Shared datapath with FP32
    ldst_units = 16              # 8 per processing block
    sfu_units = 8                # 4 per processing block

    # ── Threading / occupancy limits (CC 6.0) ───────────────────────
    max_concurrent_warps = 64
    max_threads_per_unit = 2048   # 64 warps × 32 threads
    max_thread_blocks_per_unit = 32
    max_registers_per_thread = 255
    max_registers_per_block = 65536

    # ── Parallel / concurrent execution capabilities ─────────────────
    can_concurrent_fp32_int32 = False  # Shared datapath (separate INT32 starts in Volta)
    threads_per_warp = 32

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    @classmethod
    def _pascal_pipeline(cls) -> list[PipelineStage]:
        """Return the common Pascal GP100 SM pipeline stages.

        Pipeline: global_read → shared_load → fma_alu → shared_store → global_write.
        Pascal adds a dedicated shared memory path (64 KB, always available).
        No async copy engine yet.
        """
        return [
            PipelineStage(
                name="global_read",
                latency_cycles=350,
                throughput_per_cycle=16,        # bytes/cycle per SM (estimated)
                description="HBM2 → L2 → L1/Shared → registers (load path)",
            ),
            PipelineStage(
                name="shared_load",
                latency_cycles=20,
                throughput_per_cycle=128,       # bytes/cycle per SM
                description="Shared memory → registers (per-warp load)",
            ),
            PipelineStage(
                name="fma_alu",
                latency_cycles=6,               # Pascal FP32 pipeline depth
                throughput_per_cycle=64,        # FMA ops/cycle per SM (64 FP32 cores)
                description="FP32/FP64 fused multiply-add on CUDA cores. FP16 packed: 2× throughput.",
            ),
            PipelineStage(
                name="shared_store",
                latency_cycles=20,
                throughput_per_cycle=128,       # bytes/cycle per SM
                description="Registers → shared memory (per-warp store)",
            ),
            PipelineStage(
                name="global_write",
                latency_cycles=350,
                throughput_per_cycle=16,        # bytes/cycle per SM (estimated)
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
        """Create a :class:`ComputeUnit` pre-filled with Pascal SM defaults."""
        kwargs: dict = dict(
            name="SM",
            count=count,
            clock_mhz=clock_mhz,
            peak_flops=peak_flops,
            pipeline=cls._pascal_pipeline(),

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


class NvidiaP100(NvidiaPascal):
    """NVIDIA Tesla P100 (SXM2) — GP100 GPU, Pascal architecture.

    Full-chip peak (56 SMs @ 1480 MHz GPU Boost):
        FP64 CUDA Core:  5.3 TFLOPS
        FP32 CUDA Core: 10.6 TFLOPS
        FP16 CUDA Core: 21.2 TFLOPS (2× FP32 via packed FP16 instruction)

    The full GP100 die has 60 SMs (6 GPCs).  Tesla P100 ships with 56 SMs
    enabled.  TDP: 300 W.
    """

    name = "p100"
    description = "NVIDIA Tesla P100 — Pascal, 56 SM, 10.6 TFLOPS FP32"

    # ── Memory hierarchy ──────────────────────────────────────────────

    memory = MemoryHierarchy(
        tiers=[
            MemoryTier(
                name="HBM2",
                capacity_bytes=16 * 1024**3,          # 16 GB (4 stacks × 4 GB)
                bandwidth_bytes_per_sec=720e9,        # 720 GB/s (4096-bit interface)
            ),
            MemoryTier(
                name="L2",
                capacity_bytes=4 * 1024**2,           # 4 MB L2 (4096 KB)
                bandwidth_bytes_per_sec=2.0e12,       # Approximate
            ),
        ],
        can_overlap_with_compute={"HBM2"},
    )

    # ── Compute unit ──────────────────────────────────────────────────

    compute_unit = NvidiaPascal._make_compute_unit(
        count=56,
        clock_mhz=1480,             # Tesla P100 GPU Boost clock

        peak_flops={
            # 56 SMs × 32 FP64 cores × 2 FMA × 1.48 GHz = 5304 GFLOPS
            DataType.FP64: 5304e9,
            # 56 SMs × 64 FP32 cores × 2 FMA × 1.48 GHz = 10609 GFLOPS
            DataType.FP32: 10609e9,
            # FP16: 2× FP32 via packed instructions = 21.2 TFLOPS
            DataType.FP16: 21218e9,
        },
    )
