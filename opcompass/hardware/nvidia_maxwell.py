"""NVIDIA Maxwell GM204 architecture hardware definitions.

Based on the NVIDIA Maxwell GM204 Architecture Whitepaper.
Covers Compute Capability 5.2 GPUs: GTX 980, GTX 970.

Architecture overview
---------------------
The GM204 GPU is fabricated on TSMC 28 nm, with 5.2 billion transistors
on a 398 mm² die.  Maxwell is a power-efficiency focused architecture —
the GTX 980 delivers nearly 2× performance/Watt over Kepler GK110.

The SM was renamed to SMM ("Streaming Multiprocessor Maxwell") and
partitioned into 4 processing blocks of 32 CUDA cores each, aligned
with the warp size for simpler scheduling.

SMM architecture
----------------
Each SMM contains (4 processing blocks, each with):

**Execution resources per processing block**
- 1 warp scheduler (dispatch 2 instructions/warp/cycle)
- 32 FP32 (CUDA) cores
- 1 FP64 core (only 4 DP units per SMM total — 1:32 ratio)
- 8 texture units

**Per-SMM totals**
- 4 warp schedulers
- 128 FP32 (CUDA) cores  →  128 FP32 FMA / clock / SMM (fewer than Kepler's 192, but more efficient)
- 4 FP64 cores            →  4 FP64 FMA / clock / SMM (1/32 ratio, severely reduced)
- 8 texture units
- **No separate INT32 datapath** — shared with FP32
- **No Tensor Cores**

**On-chip memory**
- Register file: 256 KB (65536 × 32-bit registers) per SMM
- 96 KB dedicated shared memory (NOT configurable with L1 — separate)
- L1 cache unified with texture cache (size not publicly stated)

**Threading / occupancy** (CC 5.2)
- 32 threads per warp
- Max 64 warps / 2048 threads resident per SMM
- Max 32 thread blocks per SMM
- Max 255 registers per thread
- Max thread block size: 1024 threads

Key Maxwell innovations
-----------------------
- 2× performance/Watt over Kepler GK110
- Dedicated shared memory (96 KB, no longer configurable with L1)
- L1 unified with texture cache for better graphics caching
- 2 MB L2 cache (vs 1.5 MB in GK110)
- Reduced FP64 rate (1:32 vs Kepler GK110's 1:3) — datacenter FP64 moved to GK210
- Improved warp scheduling: 4 independent 32-core partitions
- Tile-based immediate-mode rasterizer for reduced memory bandwidth
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


class NvidiaMaxwell(Hardware):
    """Base class for NVIDIA Maxwell architecture GPUs (CC 5.2).

    Provides the common SMM microarchitecture parameters shared across
    GM204-based GPUs (GTX 980, GTX 970) and GM200 variants.
    This class is NOT registered as a standalone hardware target.
    """

    vendor = "NVIDIA"

    # ── Architecture identity ────────────────────────────────────────
    architecture = "Maxwell"
    sm_version = "5.2"          # Compute Capability (GM204)

    # ── Per-SMM memory resources ─────────────────────────────────────
    register_file_kb = 256       # 65536 × 32-bit registers
    shared_memory_max_kb = 96    # Dedicated (NOT configurable with L1)
    l1_shared_combined_kb = 96   # Separate from L1; L1 is unified with texture cache

    # ── Per-SMM execution resources ──────────────────────────────────
    warp_schedulers_per_unit = 4
    tensor_cores_per_unit = 0    # No Tensor Cores
    fp32_cores_per_unit = 128    # Reduced from Kepler's 192 but more efficient
    fp64_cores_per_unit = 4      # Severely reduced (1:32 ratio)
    int32_cores_per_unit = 0     # Shared datapath with FP32
    ldst_units = 16              # Estimated
    sfu_units = 4                # Estimated (1 per processing block)

    # ── Threading / occupancy limits (CC 5.2) ───────────────────────
    max_concurrent_warps = 64
    max_threads_per_unit = 2048
    max_thread_blocks_per_unit = 32
    max_registers_per_thread = 255
    max_registers_per_block = 65536

    # ── Parallel / concurrent execution capabilities ─────────────────
    can_concurrent_fp32_int32 = False  # Shared datapath
    threads_per_warp = 32

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    @classmethod
    def _maxwell_pipeline(cls) -> list[PipelineStage]:
        """Return the common Maxwell SMM pipeline stages.

        Pipeline: global_read → fma_alu → global_write.
        Maxwell's dedicated shared memory provides consistent 96 KB
        without configuration trade-off, but no async copy engine.
        """
        return [
            PipelineStage(
                name="global_read",
                latency_cycles=350,
                throughput_per_cycle=16,        # bytes/cycle per SMM (estimated)
                description="GDDR5 → L2 → L1/Shared → registers (load path)",
            ),
            PipelineStage(
                name="fma_alu",
                latency_cycles=6,
                throughput_per_cycle=128,       # FMA ops/cycle per SMM (128 FP32 cores)
                description="FP32/FP64 fused multiply-add on CUDA cores.",
            ),
            PipelineStage(
                name="global_write",
                latency_cycles=350,
                throughput_per_cycle=16,        # bytes/cycle per SMM (estimated)
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
        """Create a :class:`ComputeUnit` pre-filled with Maxwell SMM defaults."""
        kwargs: dict = dict(
            name="SMM",
            count=count,
            clock_mhz=clock_mhz,
            peak_flops=peak_flops,
            pipeline=cls._maxwell_pipeline(),

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


class NvidiaGM204(NvidiaMaxwell):
    """NVIDIA GeForce GTX 980 — GM204 GPU, Maxwell architecture.

    Full-chip (16 SMMs @ 1216 MHz GPU Boost):
        FP32 CUDA Core:  4.98 TFLOPS

    The full GM204 die has 16 SMMs in 4 GPCs (4 SMMs/GPC).
    GTX 980 ships with all 16 SMMs enabled.  GTX 970 has 13 SMMs.
    """

    name = "gm204"
    description = "NVIDIA GM204 (GTX 980) — Maxwell, 16 SMM, 4.98 TFLOPS FP32"

    # ── Memory hierarchy ──────────────────────────────────────────────

    memory = MemoryHierarchy(
        tiers=[
            MemoryTier(
                name="GDDR5",
                capacity_bytes=4 * 1024**3,           # 4 GB (GTX 980)
                bandwidth_bytes_per_sec=224e9,        # 224 GB/s (256-bit @ 7 Gbps)
            ),
            MemoryTier(
                name="L2",
                capacity_bytes=2 * 1024**2,           # 2 MB L2
                bandwidth_bytes_per_sec=400e9,        # Approximate
            ),
        ],
        can_overlap_with_compute=set(),
    )

    # ── Compute unit ──────────────────────────────────────────────────

    compute_unit = NvidiaMaxwell._make_compute_unit(
        count=16,
        clock_mhz=1216,             # GTX 980 GPU Boost clock

        peak_flops={
            # 16 SMMs × 4 FP64 cores × 2 FMA × 1.216 GHz = 155.6 GFLOPS (1:32 ratio)
            DataType.FP64: 155.6e9,
            # 16 SMMs × 128 FP32 cores × 2 FMA × 1.216 GHz = 4979 GFLOPS
            DataType.FP32: 4979e9,
        },
    )
