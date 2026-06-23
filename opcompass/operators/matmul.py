from __future__ import annotations
"""Matrix multiplication: C = A × B  with shapes (M, K) × (K, N) → (M, N)."""

import math

from opcompass.models import DataType, SubOp, TilingInfo
from opcompass.operators.base import Operator


class Matmul(Operator):
    """Standard dense matrix multiplication.

    Shapes:
        A: (M, K)
        B: (K, N)
        C: (M, N)

    FLOPs = 2 * M * N * K  (one multiply-add = 2 ops)
    """

    name = "matmul"
    description = "Dense matrix multiplication C[M,N] = A[M,K] × B[K,N]"

    @property
    def param_dims(self) -> dict[str, str]:
        return {
            "M": "Rows of A / C",
            "N": "Cols of B / C",
            "K": "Inner dimension (cols of A, rows of B)",
        }

    def compute_torch(self, inputs: List["torch.Tensor"], **kwargs) -> List["torch.Tensor"]:
        import torch

        A, B = inputs
        return [torch.matmul(A, B)]

    def compute_flops(self, M: int = 0, N: int = 0, K: int = 0, **kwargs) -> int:
        return 2 * M * N * K

    def compute_io_bytes(
        self, dtype: DataType, M: int = 0, N: int = 0, K: int = 0, **kwargs
    ) -> tuple[int, int]:
        bs = dtype.byte_size
        read_bytes = (M * K + K * N) * bs
        write_bytes = M * N * bs
        return read_bytes, write_bytes

    # ------------------------------------------------------------------
    # Pipeline-level decomposition
    # ------------------------------------------------------------------

    def get_tiling_strategy(self, hardware, dtype=None, **dims):
        """Return a CUTLASS-style tiling strategy for the given hardware.

        Uses architecture-aware default tile sizes, then validates
        shared memory constraints and shrinks block_K if needed.
        """
        if dtype is None:
            dtype = DataType.FP16

        cu = hardware.compute_unit
        arch = getattr(hardware, "architecture", "").lower()

        # CUTLASS default tiling per architecture
        if arch == "ampere":
            if dtype in (DataType.FP16, DataType.BF16):
                bM, bN, bK = 128, 128, 32
            elif dtype == DataType.TF32:
                bM, bN, bK = 128, 128, 16
            elif dtype == DataType.FP32:
                bM, bN, bK = 64, 64, 16
            else:
                bM, bN, bK = 128, 128, 32
        elif arch == "hopper":
            bM, bN, bK = 128, 128, 64
        else:
            bM, bN, bK = 64, 64, 32

        bs = dtype.byte_size
        # Double-buffered shared memory: 2 × (A_tile + B_tile) + C_stage
        smem = 2 * (bM * bK + bK * bN) * bs + bM * bN * bs

        # Check against hardware shared memory limit
        smem_limit = cu.shared_memory_max_kb * 1024
        if smem > smem_limit:
            while bK > 8 and smem > smem_limit:
                bK //= 2
                smem = 2 * (bM * bK + bK * bN) * bs + bM * bN * bs

        num_warps = 4  # standard warp-group size for CUTLASS

        return TilingInfo(
            block_m=bM,
            block_n=bN,
            block_k=bK,
            shared_memory_per_block=smem,
            num_warps_per_block=num_warps,
        )

    def get_ops_breakdown(self, dtype=None, hardware=None, pipeline_config=None, **dims):
        """Decompose matmul into CUTLASS-style sub-ops per K-slice iteration.

        A single thread-block processes a (block_M, block_N) output tile.
        For each K-slice of width block_K, the pipeline stages are:

        - async_copy_load_A/B: load tiles from HBM to shared memory (bypassing L1)
        - shared_load_A/B: move tiles from shared memory to registers
        - mma: matrix multiply-accumulate on Tensor Cores
        - shared_store_C: store partial C to shared memory (epilogue only)
        - global_write_C: write final C back to HBM (epilogue only)

        When async_copy_enabled=False, async_copy_load is replaced by
        global_read (lower throughput, same bytes).
        """
        if dtype is None or hardware is None:
            return []

        M = dims.get("M", 0)
        N = dims.get("N", 0)
        K = dims.get("K", 0)
        bs = dtype.byte_size

        tiling = self.get_tiling_strategy(hardware, dtype, **dims)
        if tiling is None:
            return []

        bM = tiling.block_m
        bN = tiling.block_n
        bK = tiling.block_k

        async_on = True
        if pipeline_config is not None:
            async_on = pipeline_config.async_copy_enabled

        # Determine load stage name based on async copy toggle
        load_stage = "async_copy_load" if async_on else "global_read"
        load_a_name = "async_copy_load_A" if async_on else "global_read_A"
        load_b_name = "async_copy_load_B" if async_on else "global_read_B"

        # Per-iteration recurring sub-ops
        sub_ops = [
            SubOp(
                name=load_a_name,
                pipeline_stage=load_stage,
                read_bytes=bM * bK * bs,
                depends_on=[],
                is_recurring=True,
            ),
            SubOp(
                name=load_b_name,
                pipeline_stage=load_stage,
                read_bytes=bK * bN * bs,
                depends_on=[],
                is_recurring=True,
            ),
            SubOp(
                name="shared_load_A",
                pipeline_stage="shared_load",
                read_bytes=bM * bK * bs,
                depends_on=[load_a_name],
                is_recurring=True,
            ),
            SubOp(
                name="shared_load_B",
                pipeline_stage="shared_load",
                read_bytes=bK * bN * bs,
                depends_on=[load_b_name],
                is_recurring=True,
            ),
            SubOp(
                name="mma",
                pipeline_stage="mma",
                flops=2 * bM * bN * bK,
                depends_on=["shared_load_A", "shared_load_B"],
                is_recurring=True,
            ),
        ]

        # Epilogue sub-ops (one-shot)
        sub_ops += [
            SubOp(
                name="shared_store_C",
                pipeline_stage="shared_store",
                write_bytes=bM * bN * bs,
                depends_on=["mma"],
                is_recurring=False,
            ),
            SubOp(
                name="global_write_C",
                pipeline_stage="global_write",
                write_bytes=bM * bN * bs,
                depends_on=["shared_store_C"],
                is_recurring=False,
            ),
        ]

        return sub_ops
