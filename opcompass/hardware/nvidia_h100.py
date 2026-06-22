from __future__ import annotations
"""NVIDIA H100 (80 GB, SXM5) hardware definition.

Key specs:
    - 132 SMs, 1980 MHz base clock
    - 80 GB HBM3, ~3.35 TB/s bandwidth
    - 50 MB L2 cache
    - Peak: 989 TFLOPS FP16/BF16 (with sparsity: 1979)
    - Peak: 67 TFLOPS FP32
    - Peak: 495 TFLOPS TF32
    - Peak: 1979 TOPS INT8
    - FP8 support (via Transformer Engine): 1979 TFLOPS
"""

from opcompass.hardware.base import Hardware
from opcompass.models import (
    ComputeUnit,
    DataType,
    MemoryHierarchy,
    MemoryTier,
    PipelineStage,
)


class NvidiaH100(Hardware):
    """NVIDIA H100 80 GB (SXM5)."""

    name = "h100"
    vendor = "NVIDIA"
    description = "NVIDIA H100 80GB SXM5 — Hopper, 132 SM, 989 TFLOPS FP16"

    memory = MemoryHierarchy(
        tiers=[
            MemoryTier(
                name="HBM3",
                capacity_bytes=80 * 1024**3,
                bandwidth_bytes_per_sec=3.35e12,   # ~3.35 TB/s
            ),
            MemoryTier(
                name="L2",
                capacity_bytes=50 * 1024**2,        # 50 MB
                bandwidth_bytes_per_sec=7.5e12,     # ~7.5 TB/s (approx)
            ),
        ],
        can_overlap_with_compute={"HBM3"},
    )

    compute_unit = ComputeUnit(
        name="SM",
        count=132,
        clock_mhz=1980,
        peak_flops={
            DataType.FP64: 34e12,
            DataType.FP32: 67e12,
            DataType.TF32: 495e12,
            DataType.FP16: 989e12,
            DataType.BF16: 989e12,
            DataType.FP8: 1979e12,
            DataType.INT8: 1979e12,
        },
        pipeline=[
            PipelineStage(
                name="global_read",
                latency_cycles=280,
                throughput_per_cycle=64,
                description="HBM → registers (via L2)",
            ),
            PipelineStage(
                name="shared_load",
                latency_cycles=18,
                throughput_per_cycle=128,
                description="Shared memory → registers",
            ),
            PipelineStage(
                name="mma",
                latency_cycles=6,
                throughput_per_cycle=512,           # FP16 tensor core throughput per SM
                description="Matrix multiply-accumulate (tensor core, Hopper)",
            ),
            PipelineStage(
                name="writeback",
                latency_cycles=280,
                throughput_per_cycle=64,
                description="Registers → HBM (via L2)",
            ),
        ],
        max_concurrent_warps=64,
    )
