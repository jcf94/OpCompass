from __future__ import annotations
"""Matrix multiplication: C = A × B  with shapes (M, K) × (K, N) → (M, N)."""

from opcompass.models import DataType
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

    def compute_flops(self, M: int = 0, N: int = 0, K: int = 0, **kwargs) -> int:
        return 2 * M * N * K

    def compute_io_bytes(
        self, dtype: DataType, M: int = 0, N: int = 0, K: int = 0, **kwargs
    ) -> tuple[int, int]:
        bs = dtype.byte_size
        read_bytes = (M * K + K * N) * bs
        write_bytes = M * N * bs
        return read_bytes, write_bytes
