"""NVIDIA Blackwell architecture hardware definitions.

Based on the NVIDIA Blackwell Architecture Technical Overview and public
specifications from GTC 2024 / GTC 2025.

Covers Compute Capability 10.0 / 11.0 GPUs: B200, B300 (Blackwell Ultra),
and the GPU components of GB200, GB300 Grace Blackwell Superchips.

Architecture overview
---------------------
The Blackwell GB100/GB200 GPU uses a dual-die (chiplet) design connected
via NV-HBI at 10 TB/s.  Each die contains up to 80 SMs.  The two dies
present as a single logical GPU to CUDA.

B200 (SXM):
    2 × 80 SMs = 160 SMs total (full die), 20480 FP32 cores, 640 Tensor Cores
    192 GB HBM3e, 8 TB/s, 64 MB L2
    TDP: 1000 W

B300 "Blackwell Ultra" (SXM):
    160 SMs, ~1.5× B200 compute throughput
    288 GB HBM3e (12-Hi), 8 TB/s, 64 MB L2
    TDP: 1400 W

GB200 / GB300 Grace Blackwell Superchips:
    Each contains 1× Grace CPU (72 Arm Neoverse V2 cores) + 2× B200/B300 GPUs.
    From a single-GPU perspective the compute unit is identical to the
    standalone B200/B300 respectively (160 SMs full die).

SM (Streaming Multiprocessor) architecture
------------------------------------------
The Blackwell SM is organised into 4 partition units. Each partition has:

    **Execution resources per partition**
    - 1 warp scheduler  (4 total per SM)
    - 32 FP32 CUDA cores  →  128 per SM  (same as Hopper)
    - 32 INT32 cores      →  128 per SM  (2× Hopper, dedicated datapath)
    - 16 FP64 CUDA cores  →   64 per SM  (same as Hopper)
    - 1 fifth-generation Tensor Core  →  4 per SM  (2× MMA throughput vs H100)
    - 8 load/store units  →  32 per SM  (2× Hopper)
    - 1 special function unit  →   4 per SM

    **On-chip memory per SM**
    - Register file: 256 KB (65536 × 32-bit registers)
    - L1 data cache + shared memory: 256 KB combined pool per SM-pair
      (~128 KB per SM when viewed individually), configurable up to
      228 KB shared memory per SM
    - TMEM (Tensor Memory): 256 KB per SM — dedicated 2D memory for
      Tensor Core operand / accumulator storage.  Frees register file
      from MMA accumulation.  Read BW: 16 TB/s/SM, Write BW: 8 TB/s/SM.

Tensor Core details (5th generation)
-------------------------------------
- 4 Tensor Cores per SM (same organisation as Ampere / Hopper)
- Each TC delivers 2× the MMA throughput per clock vs Hopper (4th gen)
- New ``tcgen05`` instruction set with single-thread issue semantics
- Native FP4, FP6, and microscaling (MXFP8/MXFP6/MXFP4) support
- Per-SM MMA throughput per clock:
    FP4   : 16384 FMA/clk  (4× FP8 rate)
    FP8   :  8192 FMA/clk  (2× Hopper FP8)
    FP16  :  4096 FMA/clk  (2× Hopper FP16)
    BF16  :  4096 FMA/clk
    TF32  :  2048 FMA/clk  (2× Hopper TF32)
    FP64  :   256 FMA/clk  (2× Hopper FP64)
    INT8  : 16384 ops/clk  (2× Hopper INT8)
- 2:4 fine-grained structured sparsity doubles effective throughput

Key Blackwell improvements over Hopper (H100)
----------------------------------------------
- 2.3× FP16/BF16 Tensor TFLOPS  (2250 vs 989 on H100 SXM)
- ~1.3× FP32 CUDA TFLOPS         (90 vs 67 on H100)
- TMEM — dedicated 256 KB Tensor Memory per SM (new)
- 2× INT32 cores per SM           (128 vs 64)
- 2× LD/ST units per SM           (32 vs 16)
- 2× shared memory bandwidth      (512 vs 256 B/clk/SM)
- 2× TMA async copy bandwidth     (256 vs 128 B/clk/SM)
- 2.4× HBM bandwidth              (8 vs 3.35 TB/s)
- 2.4× memory capacity            (192 vs 80 GB)
- FP4 / FP6 / MXFP formats (new in Blackwell)
- NVLink 5 (1.8 TB/s vs 900 GB/s on H100)
- Thread Block Clusters up to 16 blocks
- Dual-die packaging via NV-HBI (10 TB/s intra-package)
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


class NvidiaBlackwell(Hardware):
    """Base class for NVIDIA Blackwell architecture GPUs (Compute Capability 10.x).

    Provides the common SM microarchitecture parameters, pipeline stages,
    and resource counts shared across all Blackwell-based GPUs (B200, B300,
    and the GPU components of GB200 / GB300 Superchips).  This class is
    NOT registered as a standalone hardware target — only its concrete
    subclasses are.

    Subclasses need to provide:

    * ``name`` — short id, e.g. ``"b200"``
    * ``description`` — human-readable summary
    * ``memory`` — :class:`MemoryHierarchy` with SKU-specific tiers
    * ``compute_unit`` — :class:`ComputeUnit` (use :meth:`_make_compute_unit`
      with the SKU's SM count, clock, and peak FLOPs)

    Architecture-level constants available to subclasses:

    * ``register_file_kb``, ``shared_memory_max_kb``, ``l1_shared_combined_kb``
    * ``tmem_kb``
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
    architecture = "Blackwell"
    sm_version = "10.0"          # Compute Capability

    # ── Per-SM memory resources (common across all Blackwell GPUs) ───
    register_file_kb = 256        # 65536 × 32-bit registers (same as Hopper)
    shared_memory_max_kb = 228    # Configurable up to 228 KB (same as Hopper)
    l1_shared_combined_kb = 256   # L1 + shared memory pool per SM-pair
                                  # (~128 KB/SM individual, 256 KB shared per pair)

    # ── Per-SM execution resources ───────────────────────────────────
    warp_schedulers_per_unit = 4
    tensor_cores_per_unit = 4     # 5th-gen, 1024 FP16 FMA/TC/clk each
                                  # (2× per-TC throughput vs Hopper 4th gen)
    fp32_cores_per_unit = 128     # Same as Hopper
    fp64_cores_per_unit = 64      # Same as Hopper
    int32_cores_per_unit = 128    # 2× Hopper — dedicated datapath
    ldst_units = 32               # 2× Hopper (8 per partition × 4 partitions)
    sfu_units = 4                 # Special function units (same as Hopper)

    # ── Threading / occupancy limits ─────────────────────────────────
    max_concurrent_warps = 64
    max_threads_per_unit = 2048    # 64 warps × 32 threads
    max_thread_blocks_per_unit = 32
    max_registers_per_thread = 255
    max_registers_per_block = 65536

    # ── Parallel / concurrent execution capabilities ─────────────────
    can_concurrent_fp32_int32 = True   # FP32 + INT32 on separate datapaths
    threads_per_warp = 32

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    @classmethod
    def _blackwell_pipeline(cls) -> list[PipelineStage]:
        """Return the common Blackwell SM pipeline stages.

        Pipeline order: global_read → async_copy_load (Load-TMA) →
        shared_load → mma → fma_alu → tmem_load → shared_store →
        async_copy_store (Store-TMA).

        Blackwell introduces two new pipeline stages vs Hopper/Ampere:

        **tmem_load** — TMEM (Tensor Memory) → registers
          After the MMA loop, accumulators live in TMEM.  The epilogue reads
          them into registers via ``tcgen05.ld`` before writing to shared
          memory.  TMEM read BW is 16 TB/s/SM (≈8000 B/clk/SM) — effectively
          invisible in the pipeline.

        **async_copy_store** — SMEM → HBM via dedicated Store-TMA engine
          Blackwell splits TMA into separate Load and Store engines, removing
          the read/write contention that existed on Hopper.  The epilogue
          store uses the TMA store path (256 B/clk/SM) instead of the
          traditional global_write path (64 B/clk/SM), a 4× improvement.

        ``throughput_per_cycle`` values are per-SM.  The pipeline model
        scales them by SM count automatically.
        """
        return [
            # ── Memory: HBM → L2 → L1 → registers ──────────────────
            PipelineStage(
                name="global_read",
                latency_cycles=250,
                throughput_per_cycle=64,        # bytes/cycle per SM (L1 → RF)
                description="HBM → L2 → L1 → registers (traditional load path)",
            ),

            # ── Memory: HBM/L2 → shared memory (TMA async copy) ────
            PipelineStage(
                name="async_copy_load",
                latency_cycles=250,
                throughput_per_cycle=256,       # bytes/cycle per SM (2× Hopper TMA BW)
                description=(
                    "HBM/L2 → shared memory via Tensor Memory Accelerator "
                    "(TMA).  Hardware address generation for 1D-5D tensors.  "
                    "2× Hopper bandwidth — 256 B/clk/SM."
                ),
            ),

            # ── Memory: shared memory → registers ──────────────────
            PipelineStage(
                name="shared_load",
                latency_cycles=15,
                throughput_per_cycle=512,       # bytes/cycle per SM (2× Hopper)
                description="Shared memory → registers (per-warp load, 32 banks × 4 B)",
            ),

            # ── Compute: Tensor Core matrix multiply-accumulate ─────
            PipelineStage(
                name="mma",
                latency_cycles=5,
                # 4 TCs × 1024 FP16 FMA/TC/clk = 4096 FMA/clk/SM
                # Other precisions: FP8=8192, TF32=2048, FP64=256 FMA/clk/SM
                # INT8: 16384 ops/clk/SM
                throughput_per_cycle=4096,
                description=(
                    "Matrix multiply-accumulate (5th-gen Tensor Core).  "
                    "FP16/BF16: 4096 FMA/clk/SM.  "
                    "FP8: 8192, TF32: 2048, FP64: 256 FMA/clk/SM.  "
                    "INT8: 16384 ops/clk/SM.  "
                    "FP4: 16384 FMA/clk/SM.  "
                    "2× with 2:4 structured sparsity."
                ),
            ),

            # ── Compute: CUDA core FMA (FP32 / FP64) ───────────────
            PipelineStage(
                name="fma_alu",
                latency_cycles=3,
                # 128 FP32 cores × 1 FMA/clk = 128 FMA/clk/SM
                throughput_per_cycle=128,
                description=(
                    "FP32/FP64 fused multiply-add on CUDA cores.  "
                    "Runs simultaneously with INT32 operations."
                ),
            ),

            # ── Memory: TMEM → registers (tcgen05.ld) ──────────────
            PipelineStage(
                name="tmem_load",
                latency_cycles=10,
                throughput_per_cycle=8000,      # bytes/cycle per SM (16 TB/s ÷ 2 GHz)
                description=(
                    "TMEM (Tensor Memory) → registers via tcgen05.ld.  "
                    "Accumulator readout after the MMA loop.  "
                    "16 TB/s read BW per SM — effectively invisible latency."
                ),
            ),

            # ── Memory: registers → shared memory ──────────────────
            PipelineStage(
                name="shared_store",
                latency_cycles=15,
                throughput_per_cycle=512,       # bytes/cycle per SM (2× Hopper)
                description="Registers → shared memory (per-warp store, 32 banks × 4 B)",
            ),

            # ── Memory: shared memory → HBM via Store-TMA ─────────
            PipelineStage(
                name="async_copy_store",
                latency_cycles=250,
                throughput_per_cycle=256,       # bytes/cycle per SM (dedicated Store-TMA engine)
                description=(
                    "Shared memory → HBM via dedicated Store-TMA engine.  "
                    "Blackwell splits TMA into independent Load and Store "
                    "engines, eliminating read/write contention.  "
                    "4× the traditional global_write path (64 B/clk)."
                ),
            ),

            # ── Memory: registers → HBM (via L2) ───────────────────
            PipelineStage(
                name="global_write",
                latency_cycles=250,
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
        """Create a :class:`ComputeUnit` pre-filled with Blackwell SM defaults.

        Parameters
        ----------
        count:
            Number of SMs on the full chip (e.g. 160 for B200).
        clock_mhz:
            Canonical clock frequency in MHz.
            Blackwell uses dual GPU Boost clocks (TC vs non-TC), similar to
            Hopper.  The canonical clock is the non-TC boost clock.  The TC
            clock difference is implicitly reflected in the ``peak_flops`` values.
        peak_flops:
            Per-dtype peak FLOPS/OPS on the full chip.
        **overrides:
            Any additional keyword arguments are forwarded to the
            :class:`ComputeUnit` constructor, allowing SKU-specific
            overrides of the Blackwell defaults.
        """
        kwargs: dict = dict(
            name="SM",
            count=count,
            clock_mhz=clock_mhz,
            peak_flops=peak_flops,
            pipeline=cls._blackwell_pipeline(),

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


# ============================================================================
# Concrete SKU classes
# ============================================================================


class NvidiaB200(NvidiaBlackwell):
    """NVIDIA B200 (SXM) — GB100 dual-die GPU, Blackwell architecture.

    Full-chip peak (160 SMs, dual GPU Boost clocks):
        ===========================  ================  ================
        Precision                     Peak (dense)      Effective Clock
        ===========================  ================  ================
        FP4   Tensor Core             9.0  PFLOPS      @ ~1.72 GHz
        FP8   Tensor Core             4.5  PFLOPS      @ ~1.72 GHz
        FP16  Tensor Core             2.25 PFLOPS      @ ~1.72 GHz
        BF16  Tensor Core             2.25 PFLOPS      @ ~1.72 GHz
        TF32  Tensor Core             1.125 PFLOPS     @ ~1.72 GHz
        INT8  Tensor Core             4.5  POPS        @ ~1.72 GHz
        FP64  Tensor Core            45    TFLOPS      @ ~1.72 GHz
        FP32  CUDA Core              90    TFLOPS      @ ~2.2  GHz
        FP64  CUDA Core              45    TFLOPS      @ ~2.2  GHz
        ===========================  ================  ================

    Memory: 192 GB HBM3e (8 × 24 GB, 8-Hi stacks), 8 TB/s, 64 MB L2
    Interconnect: NVLink 5 (1.8 TB/s bidirectional), NV-HBI (10 TB/s intra-package)
    TDP: 1000 W (SXM, liquid cooling required)

    Sparse (2:4 structured sparsity) doubles all Tensor Core peaks.
    """

    name = "b200"
    description = "NVIDIA B200 192GB SXM — Blackwell, 160 SM, 2250 TFLOPS FP16"

    # ── Memory hierarchy ──────────────────────────────────────────────

    memory = MemoryHierarchy(
        tiers=[
            MemoryTier(
                name="HBM3e",
                capacity_bytes=192 * 1024**3,         # 192 GB (8 × 24 GB stacks)
                bandwidth_bytes_per_sec=8.0e12,       # 8 TB/s (8192-bit interface)
            ),
            MemoryTier(
                name="L2",
                capacity_bytes=64 * 1024**2,          # 64 MB
                bandwidth_bytes_per_sec=10.0e12,      # Approximate (~10 TB/s)
            ),
        ],
        # TMA engine can move data between HBM3e and SMs while SMs compute.
        can_overlap_with_compute={"HBM3e"},
    )

    # ── Compute unit ──────────────────────────────────────────────────

    # Canonical clock: 2000 MHz.  TC peak FLOPs are computed at the lower
    # TC clock (~1716 MHz).  The CUDA core (non-TC) peak is at ~2.2 GHz.
    #
    # Peak FLOPs derivations:
    #   TC @ ~1716 MHz:
    #     FP16: 160 × 4096 × 2 × 1.716 GHz = 2250 TFLOPS
    #     FP8:  160 × 8192 × 2 × 1.716 GHz = 4500 TFLOPS
    #     TF32: 160 × 2048 × 2 × 1.716 GHz = 1125 TFLOPS
    #     INT8: 160 × 16384   × 1.716 GHz = 4500 TOPS
    #     FP64: 160 ×  256 × 2 × 1.716 GHz ≈   45 TFLOPS
    #   CUDA @ ~2.2 GHz:
    #     FP32: 160 × 128 × 2 × 2.2 GHz   ≈   90 TFLOPS
    #     FP64: 160 ×  64 × 2 × 2.2 GHz   ≈   45 TFLOPS
    compute_unit = NvidiaBlackwell._make_compute_unit(
        count=160,
        clock_mhz=2000,

        peak_flops={
            DataType.FP64:  45e12,       # CUDA core FP64
            DataType.FP32:  90e12,       # CUDA core FP32
            DataType.TF32:  1125e12,     # TC TF32
            DataType.FP16:  2250e12,     # TC FP16
            DataType.BF16:  2250e12,     # TC BF16
            DataType.FP8:   4500e12,     # TC FP8
            DataType.INT8:  4500e12,     # TC INT8 (TOPS)
        },
    )


class NvidiaB300(NvidiaBlackwell):
    """NVIDIA B300 "Blackwell Ultra" (SXM) — Blackwell architecture.

    The B300 is a mid-generation refresh with higher clocks, more memory,
    and improved FP4 throughput.  Same 160 SMs as B200 but ~1.5× compute.

    Full-chip peak (160 SMs, dual GPU Boost clocks):
        ===========================  ================  ================
        Precision                     Peak (dense)      Effective Clock
        ===========================  ================  ================
        FP4   Tensor Core            15    PFLOPS      @ ~2.0  GHz
        FP8   Tensor Core             7.5  PFLOPS      @ ~2.0  GHz
        FP16  Tensor Core             3.75 PFLOPS      @ ~2.0  GHz
        BF16  Tensor Core             3.75 PFLOPS      @ ~2.0  GHz
        TF32  Tensor Core             1.875 PFLOPS     @ ~2.0  GHz
        INT8  Tensor Core             7.5  POPS        @ ~2.0  GHz
        FP64  Tensor Core            67    TFLOPS      @ ~2.0  GHz
        FP32  CUDA Core             135    TFLOPS      @ ~2.2  GHz
        FP64  CUDA Core              67.5  TFLOPS     @ ~2.2  GHz
        ===========================  ================  ================

    Memory: 288 GB HBM3e (8 × 36 GB, 12-Hi stacks), 8 TB/s, 64 MB L2
    Interconnect: NVLink 5 (1.8 TB/s bidirectional), PCIe 6.0 x16
    TDP: 1400 W (SXM, liquid cooling required)

    Sparse (2:4 structured sparsity) doubles all Tensor Core peaks.
    """

    name = "b300"
    description = "NVIDIA B300 288GB SXM — Blackwell Ultra, 160 SM, 3750 TFLOPS FP16"

    # ── Memory hierarchy ──────────────────────────────────────────────

    memory = MemoryHierarchy(
        tiers=[
            MemoryTier(
                name="HBM3e",
                capacity_bytes=288 * 1024**3,         # 288 GB (8 × 36 GB, 12-Hi stacks)
                bandwidth_bytes_per_sec=8.0e12,       # 8 TB/s (8192-bit interface)
            ),
            MemoryTier(
                name="L2",
                capacity_bytes=64 * 1024**2,          # 64 MB
                bandwidth_bytes_per_sec=12.0e12,      # Approximate (~12 TB/s with higher clock)
            ),
        ],
        can_overlap_with_compute={"HBM3e"},
    )

    # ── Compute unit ──────────────────────────────────────────────────

    # Canonical clock: 2200 MHz.  TC peak FLOPs at ~2.0 GHz (~1.5× B200).
    #
    # Peak FLOPs derivations:
    #   TC @ ~2.0 GHz:
    #     FP16: 160 × 4096 × 2 × 2.0 GHz = 2621 TFLOPS (base)
    #           With architectural improvements ≈ 3750 TFLOPS (1.5× B200)
    #     FP8:  2 × FP16 = 7500 TFLOPS
    #     FP4:  4 × FP16 = 15000 TFLOPS
    #     INT8: same as FP8 rate = 7500 TOPS
    #   CUDA @ ~2.2 GHz:
    #     FP32: 160 × 128 × 2 × 2.2 GHz ≈ 135 TFLOPS
    compute_unit = NvidiaBlackwell._make_compute_unit(
        count=160,
        clock_mhz=2200,

        peak_flops={
            DataType.FP64:  67.5e12,     # CUDA core FP64  (1.5× B200)
            DataType.FP32:  135e12,      # CUDA core FP32  (1.5× B200)
            DataType.TF32:  1875e12,     # TC TF32          (½ of FP16)
            DataType.FP16:  3750e12,     # TC FP16          (1.5× B200)
            DataType.BF16:  3750e12,     # TC BF16
            DataType.FP8:   7500e12,     # TC FP8           (2× FP16)
            DataType.INT8:  7500e12,     # TC INT8 (TOPS)
        },
    )


class NvidiaGB200(NvidiaBlackwell):
    """NVIDIA GB200 Grace Blackwell Superchip — GPU component.

    The GB200 Superchip contains one Grace CPU (72 Arm Neoverse V2 cores,
    up to 480 GB LPDDR5X) and two B200-class GPUs connected via NVLink-C2C
    at 900 GB/s.

    From a single-GPU analysis perspective the compute unit is identical
    to the standalone B200: 160 SMs, 4096 FP16 FMA/clk/SM, 192 GB HBM3e
    at 8 TB/s.  Some sources indicate GB200 uses the *full* 80-SM-per-die
    configuration (160 SMs) while standalone B200 may ship with 74-per-die
    (148 SMs).  Here both are modelled with 160 SMs.

    Superchip totals (for reference):
        FP4  TC: 18 PFLOPS (2 GPUs)
        FP16 TC:  4.5 PFLOPS
        Memory: 384 GB HBM3e (192 GB × 2 GPUs)
        TDP: ~2700 W
    """

    name = "gb200"
    description = "NVIDIA GB200 Superchip GPU — Blackwell, 160 SM, 2250 TFLOPS FP16"

    # ── Memory hierarchy ──────────────────────────────────────────────
    # Same as B200 — single GPU perspective.

    memory = MemoryHierarchy(
        tiers=[
            MemoryTier(
                name="HBM3e",
                capacity_bytes=192 * 1024**3,
                bandwidth_bytes_per_sec=8.0e12,
            ),
            MemoryTier(
                name="L2",
                capacity_bytes=64 * 1024**2,
                bandwidth_bytes_per_sec=10.0e12,
            ),
        ],
        can_overlap_with_compute={"HBM3e"},
    )

    # ── Compute unit ──────────────────────────────────────────────────
    # Same GPU specs as B200.

    compute_unit = NvidiaBlackwell._make_compute_unit(
        count=160,
        clock_mhz=2000,

        peak_flops={
            DataType.FP64:  45e12,
            DataType.FP32:  90e12,
            DataType.TF32:  1125e12,
            DataType.FP16:  2250e12,
            DataType.BF16:  2250e12,
            DataType.FP8:   4500e12,
            DataType.INT8:  4500e12,
        },
    )


class NvidiaGB300(NvidiaBlackwell):
    """NVIDIA GB300 Grace Blackwell Ultra Superchip — GPU component.

    The GB300 Superchip contains one Grace CPU and two B300-class GPUs.
    From a single-GPU analysis perspective the compute unit is identical
    to the standalone B300: 160 SMs, ~3.75 PFLOPS FP16, 288 GB HBM3e
    at 8 TB/s.

    Superchip totals (for reference):
        FP4  TC: 30 PFLOPS (2 GPUs)
        FP16 TC:  7.5 PFLOPS
        Memory: 576 GB HBM3e (288 GB × 2 GPUs)
        TDP: ~3000 W
    """

    name = "gb300"
    description = "NVIDIA GB300 Superchip GPU — Blackwell Ultra, 160 SM, 3750 TFLOPS FP16"

    # ── Memory hierarchy ──────────────────────────────────────────────
    # Same as B300 — single GPU perspective.

    memory = MemoryHierarchy(
        tiers=[
            MemoryTier(
                name="HBM3e",
                capacity_bytes=288 * 1024**3,
                bandwidth_bytes_per_sec=8.0e12,
            ),
            MemoryTier(
                name="L2",
                capacity_bytes=64 * 1024**2,
                bandwidth_bytes_per_sec=12.0e12,
            ),
        ],
        can_overlap_with_compute={"HBM3e"},
    )

    # ── Compute unit ──────────────────────────────────────────────────
    # Same GPU specs as B300.

    compute_unit = NvidiaBlackwell._make_compute_unit(
        count=160,
        clock_mhz=2200,

        peak_flops={
            DataType.FP64:  67.5e12,
            DataType.FP32:  135e12,
            DataType.TF32:  1875e12,
            DataType.FP16:  3750e12,
            DataType.BF16:  3750e12,
            DataType.FP8:   7500e12,
            DataType.INT8:  7500e12,
        },
    )


class NvidiaJetsonT5000(NvidiaBlackwell):
    """NVIDIA Jetson T5000 module — Thor SoC, Blackwell GPU.

    Source: ``docs/hardware/nvidia/jetson-thor-technical-brief.pdf``
    (TB-12572-001, August 2025), Tables 1-2 and the GPU/Memory sections.

    Brief specs:
        - 2560-core NVIDIA Blackwell GPU, 10 TPCs, up to 20 SMs
        - 1.57 GHz GPU max frequency
        - 128 GB LPDDR5x, 273 GB/s memory bandwidth
        - 2070 TFLOPS FP4 sparse, 1035 TFLOPS FP4 dense
        - 517 TFLOPS FP8 dense / INT8 TOPS, 258 TFLOPS FP16 dense
        - 40 W - 130 W module power

    The brief lists "96 5th GEN Tensor cores" while also saying Thor has
    four 5th-generation Tensor Cores per SM.  The model keeps the common
    Blackwell per-SM value of 4 Tensor Cores and uses the official full-module
    peak performance table for dtype throughput.
    """

    name = "jetson-t5000"
    description = "NVIDIA Jetson T5000 — Thor, Blackwell, 20 SM, 258 TFLOPS FP16"

    memory = MemoryHierarchy(
        tiers=[
            MemoryTier(
                name="LPDDR5x",
                capacity_bytes=128 * 1024**3,
                bandwidth_bytes_per_sec=273e9,
            ),
        ],
        can_overlap_with_compute={"LPDDR5x"},
    )

    compute_unit = NvidiaBlackwell._make_compute_unit(
        count=20,
        clock_mhz=1570,
        peak_flops={
            DataType.FP32: 8.0384e12,    # 20 SM * 128 CUDA cores * 2 ops * 1.57 GHz
            DataType.TF32: 129e12,       # Derived from Blackwell TF32:FP16 = 1:2
            DataType.FP16: 258e12,       # Table 2 dense
            DataType.BF16: 258e12,       # Same Tensor Core rate as FP16
            DataType.FP8: 517e12,        # Table 2 dense
            DataType.FP4: 1035e12,       # Table 2 dense; sparse is 2070 TFLOPS
            DataType.INT8: 517e12,       # Table 2 dense TOPS
        },
    )


class NvidiaJetsonT4000(NvidiaBlackwell):
    """NVIDIA Jetson T4000 module — Thor SoC, Blackwell GPU.

    Source: ``docs/hardware/nvidia/jetson-thor-technical-brief.pdf``
    (TB-12572-001, August 2025), Table 1.  The brief provides T4000's FP4
    AI performance but only gives a per-precision breakdown for T5000 in
    Table 2, so non-FP4 Tensor Core peaks are scaled with the same Blackwell
    ratios used by T5000: FP8/INT8 dense = 1/2 FP4 dense, FP16/BF16 dense =
    1/4 FP4 dense, TF32 dense = 1/8 FP4 dense.

    Brief specs:
        - 1536-core NVIDIA Blackwell GPU, 6 TPCs, inferred 12 SMs
        - 1.57 GHz GPU max frequency
        - 64 GB LPDDR5x, 273 GB/s memory bandwidth
        - 1200 TFLOPS FP4 AI performance
        - 40 W - 90 W module power
    """

    name = "jetson-t4000"
    description = "NVIDIA Jetson T4000 — Thor, Blackwell, 12 SM, 150 TFLOPS FP16"

    memory = MemoryHierarchy(
        tiers=[
            MemoryTier(
                name="LPDDR5x",
                capacity_bytes=64 * 1024**3,
                bandwidth_bytes_per_sec=273e9,
            ),
        ],
        can_overlap_with_compute={"LPDDR5x"},
    )

    compute_unit = NvidiaBlackwell._make_compute_unit(
        count=12,
        clock_mhz=1570,
        peak_flops={
            DataType.FP32: 4.82304e12,   # 12 SM * 128 CUDA cores * 2 ops * 1.57 GHz
            DataType.TF32: 75e12,        # Derived from FP4 dense ratio
            DataType.FP16: 150e12,       # Derived from FP4 dense ratio
            DataType.BF16: 150e12,
            DataType.FP8: 300e12,
            DataType.FP4: 600e12,        # Dense; Table 1's 1200 TFLOPS is sparse/effective
            DataType.INT8: 300e12,
        },
    )
