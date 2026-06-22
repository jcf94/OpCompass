from __future__ import annotations
"""NVIDIA A100 (80 GB, SXM4) hardware definition.

Key specs:
    - 108 SMs, 1410 MHz base clock
    - 80 GB HBM2e, ~2.0 TB/s bandwidth
    - 40 MB L2 cache
    - Peak: 312 TFLOPS FP16/BF16 (with sparsity: 624)
    - Peak: 19.5 TFLOPS FP32
    - Peak: 156 TFLOPS TF32
    - 64 warps / SM max occupancy
"""

from opcompass.hardware.base import Hardware
from opcompass.models import (
    ComputeUnit,
    DataType,
    MemoryHierarchy,
    MemoryTier,
    PipelineStage,
)


class NvidiaA100(Hardware):
    """NVIDIA A100 80 GB (SXM4)."""

    name = "a100"
    vendor = "NVIDIA"
    description = "NVIDIA A100 80GB SXM4 — Ampere, 108 SM, 312 TFLOPS FP16"

    memory = MemoryHierarchy(
        tiers=[
            MemoryTier(
                name="HBM2e",
                capacity_bytes=80 * 1024**3,
                bandwidth_bytes_per_sec=2.0e12,   # ~2.0 TB/s
            ),
            MemoryTier(
                name="L2",
                capacity_bytes=40 * 1024**2,       # 40 MB
                bandwidth_bytes_per_sec=5.0e12,    # ~5 TB/s (approx)
            ),
        ],
        can_overlap_with_compute={"HBM2e"},        # Async copy engine
    )

    compute_unit = ComputeUnit(
        name="SM",
        count=108,
        clock_mhz=1410,
        peak_flops={
            DataType.FP64: 9.7e12,
            DataType.FP32: 19.5e12,
            DataType.TF32: 156e12,
            DataType.FP16: 312e12,
            DataType.BF16: 312e12,
            DataType.INT8: 624e12,
        },
        pipeline=[
            PipelineStage(
                name="global_read",
                latency_cycles=300,
                throughput_per_cycle=64,            # bytes/cycle per SM (approx)
                description="HBM → registers (via L2)",
            ),
            PipelineStage(
                name="shared_load",
                latency_cycles=20,
                throughput_per_cycle=128,           # bytes/cycle per SM
                description="Shared memory → registers",
            ),
            PipelineStage(
                name="mma",
                latency_cycles=8,
                throughput_per_cycle=256,           # FMA ops/cycle per SM (FP16 tensor core)
                description="Matrix multiply-accumulate (tensor core)",
            ),
            PipelineStage(
                name="writeback",
                latency_cycles=300,
                throughput_per_cycle=64,
                description="Registers → HBM (via L2)",
            ),
        ],
        max_concurrent_warps=64,
    )
