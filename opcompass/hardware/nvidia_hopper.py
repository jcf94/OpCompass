"""NVIDIA Hopper architecture hardware definitions.

Based on the NVIDIA H100 Tensor Core GPU Architecture whitepaper (V1.04).
Covers Compute Capability 9.0 GPUs: H100 SXM5, H100 PCIe.

Architecture overview
---------------------
The GH100 GPU is fabricated using TSMC's 4N process customized for NVIDIA,
with 80 billion transistors on an 814 mm² die.

Full GH100 die (unshipped):
    8 GPCs, 72 TPCs, 144 SMs, 18432 FP32 cores, 576 Tensor Cores
    60 MB L2 cache, 6 HBM3/HBM2e stacks, 12 x 512-bit Memory Controllers

H100 SXM5 (shipped):
    8 GPCs, 66 TPCs, 132 SMs, 16896 FP32 cores, 528 Tensor Cores
    50 MB L2 cache, 80 GB HBM3, 5 stacks, 10 x 512-bit Memory Controllers
    TDP: 700 W

H100 PCIe (shipped):
    7-8 GPCs, 57 TPCs, 114 SMs, 14592 FP32 cores, 456 Tensor Cores
    50 MB L2 cache, 80 GB HBM2e, 5 stacks, 10 x 512-bit Memory Controllers
    TDP: 350 W

SM (Streaming Multiprocessor) architecture
------------------------------------------
The H100 SM is organized into 4 partition units. Each partition has:

    **Execution resources per partition**
    - 1 warp scheduler  (4 total per SM)
    - 32 FP32 CUDA cores  →  128 per SM  (2× A100)
    - 16 FP64 CUDA cores  →   64 per SM  (2× A100)
    - 16 INT32 cores      →   64 per SM  (same as A100, separate datapath)
    - 1 fourth-generation Tensor Core  →  4 per SM  (2× MMA throughput per TC clock-for-clock)
    - 4 load/store units  →  16 per SM  (4× A100)
    - 1 special function unit  →   4 per SM

    **On-chip memory per SM**
    - Register file: 256 KB (65536 × 32-bit registers)
    - L1 data cache + shared memory: 256 KB combined pool (1.33× A100)
      Configurable up to 228 KB shared memory

H100 has dual GPU Boost clocks depending on instruction type.
SXM5: 1830 MHz (TC) / 1980 MHz (non-TC).  PCIe: ~1620 MHz (TC) / ~1755 MHz (non-TC).

Tensor Core details (4th generation)
-------------------------------------
- 4 Tensor Cores per SM (organization same as A100)
- Each TC delivers 2× the MMA throughput per clock compared to A100 (clock-for-clock)
- 2× INT8 throughput vs FP16 (half the data size on the same datapath)
- Per-SM MMA throughput per clock:
    FP8   : 4096 FMA/clk  → 8192 FP8 ops/clk  (new in Hopper)
    FP16  : 2048 FMA/clk  → 4096 FP16 ops/clk
    BF16  : 2048 FMA/clk  → 4096 BF16 ops/clk
    TF32  : 1024 FMA/clk  → 2048 TF32 ops/clk
    FP64  :  128 FMA/clk  →  256 FP64 ops/clk
    INT8  : 8192 OPS/clk  → 8192 INT8 ops/clk
- 2:4 fine-grained structured sparsity doubles effective throughput for all formats

Key Hopper improvements over Ampere (A100)
-------------------------------------------
- 3.2× FP16/BF16 Tensor TFLOPS  (989.4 vs 312 on A100 SXM)
- 3.4× FP32 TFLOPS              (66.9 vs 19.5)
- 3.5× FP64 TFLOPS              (33.5 vs 9.7)
- 3.2× TF32 Tensor TFLOPS       (494.7 vs 156)
- 1.25× L2 cache                (50 MB vs 40 MB)
- 1.33× combined L1+shared mem  (256 KB vs 192 KB)
- 1.39× max shared mem          (228 KB vs 164 KB)
- 2× HBM bandwidth              (3.35 TB/s vs 2.0 TB/s for SXM5)
- 4× LD/ST units per SM         (16 vs 4)
- FP8 Tensor Cores (new in Hopper)
- Tensor Memory Accelerator (TMA) — new async copy engine with hardware address generation
- DPX instructions for dynamic programming (Smith-Waterman, Floyd-Warshall)
- Thread Block Clusters — new hierarchy level between Thread Block and Grid
- Distributed Shared Memory — cross-SM shared memory access
- Asynchronous Transaction Barrier — transaction-counting split barrier
- Transformer Engine — automatic FP8/FP16 management for Transformer models
- Thread Block Cluster size: up to 16 Thread Blocks per Cluster
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


class NvidiaHopper(Hardware):
    """Base class for NVIDIA Hopper architecture GPUs (Compute Capability 9.0).

    Provides the common SM microarchitecture parameters, pipeline stages,
    and resource counts shared across all Hopper-based GPUs (H100 SXM5,
    H100 PCIe, etc.).  This class is NOT registered as a standalone hardware
    target — only its concrete subclasses are.

    Subclasses need to provide:

    * ``name`` — short id, e.g. ``"h100"``
    * ``description`` — human-readable summary
    * ``memory`` — :class:`MemoryHierarchy` with SKU-specific tiers
    * ``compute_unit`` — :class:`ComputeUnit` (use :meth:`_make_compute_unit`
      with the SKU's SM count, clock, and peak FLOPs)

    Architecture-level constants available to subclasses:

    * ``register_file_kb``, ``shared_memory_max_kb``, ``l1_shared_combined_kb``
    * ``warp_schedulers_per_unit``, ``tensor_cores_per_unit``,
      ``fp32_cores_per_unit``, ``fp64_cores_per_unit``, ``int32_cores_per_unit``,
      ``ldst_units``, ``sfu_units``
    * ``max_concurrent_warps``, ``max_threads_per_unit``,
      ``max_thread_blocks_per_unit``, ``max_registers_per_thread``,
      ``max_registers_per_block``
    * ``can_concurrent_fp32_int32``, ``threads_per_warp``
    """

    # NOTE: name is intentionally left as "" so the auto-discovery registry
    # skips this intermediate base class (it checks for a non-empty name).
    vendor = "NVIDIA"

    # ── Architecture identity ────────────────────────────────────────
    architecture = "Hopper"
    sm_version = "9.0"          # Compute Capability

    # ── Per-SM memory resources (common across all Hopper GPUs) ──────
    register_file_kb = 256       # 65536 × 32-bit registers
    shared_memory_max_kb = 228   # Configurable up to 228 KB
    l1_shared_combined_kb = 256  # L1 + shared memory pool (1.33× A100)

    # ── Per-SM execution resources ───────────────────────────────────
    warp_schedulers_per_unit = 4
    tensor_cores_per_unit = 4    # 4th-gen, 512 FP16 FMA/clk each
    fp32_cores_per_unit = 128    # 2× A100 (also used for non-Tensor FP16)
    fp64_cores_per_unit = 64     # 2× A100
    int32_cores_per_unit = 64    # Separate datapath from FP32 (same as A100)
    ldst_units = 16              # 4× A100 (4 per partition × 4 partitions)
    sfu_units = 4                # Special function units (same as A100)

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
    def _hopper_pipeline(cls) -> list[PipelineStage]:
        """Return the common Hopper SM pipeline stages.

        Pipeline order: global_read → async_copy_load (TMA) → shared_load →
        mma → fma_alu → shared_store → async_copy_store (TMA) → global_write.

        New in Hopper: the Tensor Memory Accelerator (TMA) handles async
        memory copies with hardware address generation, replacing the
        simpler Ampere async copy engine.  TMA supports 1D-5D tensor copies,
        frees CUDA threads from address calculations, and can transfer up
        to the full shared memory capacity in a single operation.

        Hopper has a single unified TMA engine shared between loads and
        stores.  The epilogue store can use TMA (async_copy_store at
        128 B/clk/SM) which is 2× faster than the traditional global_write
        path (64 B/clk/SM).  On Blackwell this is further improved with
        dedicated Load/Store TMA engines at 256 B/clk/SM each.

        ``throughput_per_cycle`` values are per-SM.  The pipeline model
        scales them by SM count automatically.
        """
        return [
            # ── Memory: HBM → L2 → L1 → registers ──────────────────
            PipelineStage(
                name="global_read",
                latency_cycles=280,
                throughput_per_cycle=64,        # bytes/cycle per SM (L1 → RF)
                description="HBM → L2 → L1 → registers (traditional load path)",
            ),

            # ── Memory: HBM/L2 → shared memory (TMA async copy) ────
            PipelineStage(
                name="async_copy_load",
                latency_cycles=280,
                throughput_per_cycle=128,       # bytes/cycle per SM (TMA BW)
                description=(
                    "HBM/L2 → shared memory via Tensor Memory Accelerator "
                    "(TMA).  Hardware address generation for 1D-5D tensors.  "
                    "Single-thread launch, frees other threads for compute."
                ),
            ),

            # ── Memory: shared memory → registers ──────────────────
            PipelineStage(
                name="shared_load",
                latency_cycles=18,
                throughput_per_cycle=256,       # bytes/cycle per SM (wider than A100's 128)
                description="Shared memory → registers (per-warp load, 32 banks × 4 B)",
            ),

            # ── Compute: Tensor Core matrix multiply-accumulate ─────
            PipelineStage(
                name="mma",
                latency_cycles=6,
                # 4 TCs × 512 FP16 FMA/TC/clk = 2048 FMA/clk/SM
                # Other precisions: FP8=4096, TF32=1024, FP64=128 FMA/clk/SM
                # INT8: 8192 ops/clk/SM
                throughput_per_cycle=2048,
                description=(
                    "Matrix multiply-accumulate (4th-gen Tensor Core).  "
                    "FP16/BF16: 2048 FMA/clk/SM.  "
                    "FP8: 4096, TF32: 1024, FP64: 128 FMA/clk/SM.  "
                    "INT8: 8192 ops/clk/SM.  "
                    "2× with 2:4 structured sparsity."
                ),
            ),

            # ── Compute: CUDA core FMA (FP32 / FP64) ───────────────
            PipelineStage(
                name="fma_alu",
                latency_cycles=4,
                # 128 FP32 cores × 1 FMA/clk = 128 FMA/clk/SM
                throughput_per_cycle=128,
                description=(
                    "FP32/FP64 fused multiply-add on CUDA cores.  "
                    "Runs simultaneously with INT32 operations."
                ),
            ),

            # ── Memory: registers → shared memory ──────────────────
            PipelineStage(
                name="shared_store",
                latency_cycles=18,
                throughput_per_cycle=256,       # bytes/cycle per SM (wider than A100's 128)
                description="Registers → shared memory (per-warp store, 32 banks × 4 B)",
            ),

            # ── Memory: shared memory → HBM via TMA store ─────────
            PipelineStage(
                name="async_copy_store",
                latency_cycles=280,
                throughput_per_cycle=128,       # bytes/cycle per SM (unified TMA engine)
                description=(
                    "Shared memory → HBM via TMA store (cp.async.bulk).  "
                    "Hopper has a single unified TMA engine shared between "
                    "load and store — 128 B/clk/SM.  During the epilogue "
                    "there is no load contention, so the full TMA bandwidth "
                    "is available.  2× the traditional global_write path."
                ),
            ),

            # ── Memory: registers → HBM (via L2) ───────────────────
            PipelineStage(
                name="global_write",
                latency_cycles=280,
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
        """Create a :class:`ComputeUnit` pre-filled with Hopper SM defaults.

        Parameters
        ----------
        count:
            Number of SMs on the full chip (e.g. 132 for H100 SXM5).
        clock_mhz:
            Boost / typical clock frequency in MHz.
        peak_flops:
            Per-dtype peak FLOPS/OPS on the full chip.
        **overrides:
            Any additional keyword arguments are forwarded to the
            :class:`ComputeUnit` constructor, allowing SKU-specific
            overrides of the Hopper defaults.
        """
        kwargs: dict = dict(
            name="SM",
            count=count,
            clock_mhz=clock_mhz,
            peak_flops=peak_flops,
            pipeline=cls._hopper_pipeline(),

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


class NvidiaH100SXM5(NvidiaHopper):
    """NVIDIA H100 80 GB (SXM5) — GH100 GPU, Hopper architecture.

    The flagship SXM5 form factor with NVLink and HBM3.  Highest
    bandwidth and SM count of the H100 family.

    Full-chip peak (132 SMs, dual GPU Boost clocks):
        ===========================  ================  ================
        Precision                     Peak (dense)      Clock
        ===========================  ================  ================
        FP8   Tensor Core            1978.9 TFLOPS     @ 1830 MHz
        FP16  Tensor Core             989.4 TFLOPS     @ 1830 MHz
        BF16  Tensor Core             989.4 TFLOPS     @ 1830 MHz
        TF32  Tensor Core             494.7 TFLOPS     @ 1830 MHz
        INT8  Tensor Core            1978.9 TOPS       @ 1830 MHz
        FP64  Tensor Core              66.9 TFLOPS     @ 1980 MHz
        FP32  CUDA Core                66.9 TFLOPS     @ 1980 MHz
        FP64  CUDA Core                33.5 TFLOPS     @ 1980 MHz
        ===========================  ================  ================

    Memory: 80 GB HBM3, 3.35 TB/s
    NVLink: 900 GB/s bidirectional (7 links)
    TDP: 700 W

    Sparse (2:4 structured sparsity) doubles all Tensor Core peaks.
    """

    name = "h100"
    description = "NVIDIA H100 80GB SXM5 — Hopper, 132 SM, 989 TFLOPS FP16"

    # ── Memory hierarchy ──────────────────────────────────────────────

    memory = MemoryHierarchy(
        tiers=[
            MemoryTier(
                name="HBM3",
                capacity_bytes=80 * 1024**3,          # 80 GB (5 stacks × 16 GB)
                bandwidth_bytes_per_sec=3.352e12,     # 3352 GB/sec (5120-bit @ 2619 MHz DDR)
            ),
            MemoryTier(
                name="L2",
                capacity_bytes=50 * 1024**2,          # 50 MB
                bandwidth_bytes_per_sec=7.5e12,       # Approximate
            ),
        ],
        # TMA engine and DMA engines can move data between HBM3 and
        # SMs while the SMs are computing.
        can_overlap_with_compute={"HBM3"},
    )

    # ── Compute unit ──────────────────────────────────────────────────

    # Use the FP32/FP64 boost clock (1980 MHz) as the canonical clock.
    # The lower Tensor Core clock (1830 MHz) is implicitly reflected in
    # the peak_flops values below.
    compute_unit = NvidiaHopper._make_compute_unit(
        count=132,
        clock_mhz=1980,

        peak_flops={
            # CUDA core @ 1980 MHz:
            #   FP64: 132 × 64  × 2 FMA × 1.98 GHz =  33.5 TFLOPS
            #   FP32: 132 × 128 × 2 FMA × 1.98 GHz =  66.9 TFLOPS
            DataType.FP64:  33.5e12,
            DataType.FP32:  66.9e12,
            # Tensor Core @ 1830 MHz:
            #   TF32: 132 × 1024 × 2 × 1.83 GHz = 494.7 TFLOPS
            #   FP16: 132 × 2048 × 2 × 1.83 GHz = 989.4 TFLOPS
            #   FP8:  132 × 4096 × 2 × 1.83 GHz = 1978.9 TFLOPS
            #   INT8: 132 × 8192 ×     1.83 GHz = 1978.9 TOPS
            DataType.TF32: 494.7e12,
            DataType.FP16: 989.4e12,
            DataType.BF16: 989.4e12,
            DataType.FP8: 1978.9e12,
            DataType.INT8: 1978.9e12,
        },
    )


class NvidiaH100PCIe(NvidiaHopper):
    """NVIDIA H100 80 GB (PCIe) — GH100 GPU, Hopper architecture.

    The dual-slot PCIe form factor for standard server compatibility.
    Uses HBM2e instead of HBM3 and fewer SMs with lower clocks, but
    fits within a 350 W TDP envelope.

    Full-chip peak (114 SMs, dual GPU Boost clocks):
        ===========================  ================  ================
        Precision                     Peak (dense)      Clock
        ===========================  ================  ================
        FP8   Tensor Core            1513.1 TFLOPS     @ 1620 MHz
        FP16  Tensor Core             756.5 TFLOPS     @ 1620 MHz
        BF16  Tensor Core             756.5 TFLOPS     @ 1620 MHz
        TF32  Tensor Core             378.3 TFLOPS     @ 1620 MHz
        INT8  Tensor Core            1513.1 TOPS       @ 1620 MHz
        FP64  Tensor Core              51.2 TFLOPS     @ 1755 MHz
        FP32  CUDA Core                51.2 TFLOPS     @ 1755 MHz
        FP64  CUDA Core                25.6 TFLOPS     @ 1755 MHz
        ===========================  ================  ================

    Memory: 80 GB HBM2e, ~2.0 TB/s
    Interconnect: PCIe Gen 5 x16 (128 GB/s bidirectional)
    TDP: 350 W

    Sparse (2:4 structured sparsity) doubles all Tensor Core peaks.
    """

    name = "h100_pcie"
    description = "NVIDIA H100 80GB PCIe — Hopper, 114 SM, 756 TFLOPS FP16"

    # ── Memory hierarchy ──────────────────────────────────────────────

    memory = MemoryHierarchy(
        tiers=[
            MemoryTier(
                name="HBM2e",
                capacity_bytes=80 * 1024**3,          # 80 GB (5 stacks × 16 GB HBM2e)
                bandwidth_bytes_per_sec=2.0e12,       # ~2.0 TB/s (same as A100 HBM2e)
            ),
            MemoryTier(
                name="L2",
                capacity_bytes=50 * 1024**2,          # 50 MB
                bandwidth_bytes_per_sec=7.5e12,       # Approximate
            ),
        ],
        can_overlap_with_compute={"HBM2e"},
    )

    # ── Compute unit ──────────────────────────────────────────────────

    # Use the FP32/FP64 boost clock (1755 MHz) as the canonical clock.
    # The lower Tensor Core clock (1620 MHz) is implicitly reflected in
    # the peak_flops values below.
    compute_unit = NvidiaHopper._make_compute_unit(
        count=114,
        clock_mhz=1755,

        peak_flops={
            # CUDA core @ 1755 MHz:
            #   FP64: 114 × 64  × 2 FMA × 1.755 GHz =  25.6 TFLOPS
            #   FP32: 114 × 128 × 2 FMA × 1.755 GHz =  51.2 TFLOPS
            DataType.FP64:  25.6e12,
            DataType.FP32:  51.2e12,
            # Tensor Core @ 1620 MHz:
            #   TF32: 114 × 1024 × 2 × 1.62 GHz = 378.3 TFLOPS
            #   FP16: 114 × 2048 × 2 × 1.62 GHz = 756.5 TFLOPS
            #   FP8:  114 × 4096 × 2 × 1.62 GHz = 1513.1 TFLOPS
            #   INT8: 114 × 8192 ×     1.62 GHz = 1513.1 TOPS
            DataType.TF32: 378.3e12,
            DataType.FP16: 756.5e12,
            DataType.BF16: 756.5e12,
            DataType.FP8: 1513.1e12,
            DataType.INT8: 1513.1e12,
        },
    )
