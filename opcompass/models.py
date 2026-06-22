"""Core data models shared across OpCompass."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DataType(str, Enum):
    """Supported numerical data types."""

    FP64 = "fp64"
    FP32 = "fp32"
    TF32 = "tf32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    FP8 = "fp8"
    INT4 = "int4"

    @property
    def byte_size(self) -> int:
        """Return the size in bytes for one element."""
        _sizes = {
            DataType.FP64: 8,
            DataType.FP32: 4,
            DataType.TF32: 4,
            DataType.FP16: 2,
            DataType.BF16: 2,
            DataType.INT8: 1,
            DataType.FP8: 1,
            DataType.INT4: 0.5,
        }
        return _sizes[self]


class AnalysisMode(str, Enum):
    """Analysis precision mode."""

    SIMPLE = "simple"       # Pure roofline: max(FLOPs/peak, bytes/bandwidth)
    HIERARCHY = "hierarchy"  # Multi-level memory hierarchy
    PIPELINE = "pipeline"   # Pipeline stage-level modeling


@dataclass
class MemoryTier:
    """A single level in the memory hierarchy."""

    name: str                       # "HBM", "L2", "SharedMem", "RegisterFile"
    capacity_bytes: int
    bandwidth_bytes_per_sec: float  # Theoretical peak bandwidth

    def transfer_time(self, bytes_count: int) -> float:
        """Return minimum seconds to move `bytes_count` through this tier."""
        if self.bandwidth_bytes_per_sec <= 0:
            return 0.0
        return bytes_count / self.bandwidth_bytes_per_sec


@dataclass
class PipelineStage:
    """One stage in a compute or memory pipeline."""

    name: str                       # "global_read", "shared_load", "mma", "writeback"
    latency_cycles: int             # Fixed latency per invocation
    throughput_per_cycle: float     # Work units per cycle (bytes for mem, ops for compute)
    description: str = ""


@dataclass
class MemoryHierarchy:
    """Full memory hierarchy specification."""

    tiers: list[MemoryTier]                    # From slowest to fastest
    can_overlap_with_compute: set[str] = field(default_factory=set)  # Tier names that allow async copy


@dataclass
class ComputeUnit:
    """Description of one type of compute unit (e.g., SM, CU)."""

    name: str                       # "SM", "CU"
    count: int                      # Number of units on the full chip
    clock_mhz: float
    peak_flops: dict[DataType, float] = field(default_factory=dict)
    pipeline: list[PipelineStage] = field(default_factory=list)
    max_concurrent_warps: int = 0

    # ── Per-unit memory resources ──────────────────────────────────
    register_file_kb: int = 0               # Register file size per compute unit
    shared_memory_max_kb: int = 0           # Max configurable shared memory per unit
    l1_shared_combined_kb: int = 0          # Combined L1 + shared memory capacity per unit

    # ── Per-unit execution resources ───────────────────────────────
    warp_schedulers_per_unit: int = 0       # Number of warp schedulers
    tensor_cores_per_unit: int = 0          # Tensor Cores per compute unit
    fp32_cores_per_unit: int = 0            # FP32 (CUDA) cores per unit
    fp64_cores_per_unit: int = 0            # FP64 cores per unit
    int32_cores_per_unit: int = 0           # INT32 cores per unit
    ldst_units: int = 0                     # Load/Store units
    sfu_units: int = 0                      # Special Function Units

    # ── Threading / occupancy limits ───────────────────────────────
    max_threads_per_unit: int = 0           # Max resident threads
    max_thread_blocks_per_unit: int = 0     # Max resident thread blocks
    max_registers_per_thread: int = 0       # Register file granularity
    max_registers_per_block: int = 0        # Max registers allocatable per block

    # ── Parallel / concurrent execution capabilities ───────────────
    can_concurrent_fp32_int32: bool = False  # Simultaneous FP32 + INT32 issue
    threads_per_warp: int = 32               # Typically 32 for NVIDIA GPUs


@dataclass
class SubOp:
    """A sub-operation within an operator, for pipeline-level analysis."""

    name: str                       # e.g., "load_A_tile", "mma", "store_C"
    flops: int = 0                  # FLOPs in this sub-op
    read_bytes: int = 0             # Bytes read from memory hierarchy
    write_bytes: int = 0            # Bytes written
    depends_on: list[str] = field(default_factory=list)  # Names of sub-ops this depends on


@dataclass
class TilingInfo:
    """Recommended tiling / blocking strategy for an operator on specific hardware."""

    block_m: int
    block_n: int
    block_k: int
    shared_memory_per_block: int = 0
    num_warps_per_block: int = 0


@dataclass
class AnalysisResult:
    """Output of a SOL analysis."""

    operator: str
    hardware: str
    shapes: dict[str, int] = field(default_factory=dict)
    dtype: DataType = DataType.FP16
    mode: AnalysisMode = AnalysisMode.HIERARCHY

    # ——— fundamental quantities ———
    total_flops: int = 0
    total_read_bytes: int = 0
    total_write_bytes: int = 0

    # ——— per-phase theoretical lower bounds (seconds) ———
    memory_read_time_s: float = 0.0
    compute_time_s: float = 0.0
    memory_write_time_s: float = 0.0

    # ——— synthesis ———
    bottleneck: str = ""            # "compute" | "memory_read" | "memory_write"
    sol_time_s: float = 0.0         # Overall theoretical lower bound
    sol_tflops: float = 0.0         # Effective TFLOPS at SOL

    # ——— detailed breakdown ———
    stage_breakdown: dict[str, float] = field(default_factory=dict)
    roofline_data: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a one-line summary string."""
        return (
            f"[{self.operator}] shape={self.shapes} dtype={self.dtype.value} "
            f"on {self.hardware} → SOL={self.sol_time_s*1e6:.1f} µs "
            f"({self.sol_tflops:.1f} TFLOPS) bottleneck={self.bottleneck}"
        )
