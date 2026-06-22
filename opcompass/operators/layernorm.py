from __future__ import annotations
"""LayerNorm / RMSNorm: normalisation over the last dimension.

Input:  (B, ..., D)
Output: (B, ..., D)

For LayerNorm:
    mean = reduce_sum(x) / D                     → D ops
    var  = reduce_sum((x - mean)^2) / D          → 2D ops
    y    = (x - mean) / sqrt(var + ε) * γ + β    → 3D ops
    Total ≈ 6 * B * ... * D FLOPs
"""

from opcompass.models import DataType
from opcompass.operators.base import Operator


class LayerNorm(Operator):
    """Layer Normalisation over the last dimension.

    Shapes:
        Input:  (N, D)  — N = batch * sequence (all leading dims flattened)
        Output: (N, D)

    FLOPs ≈ 6 * N * D  (excluding ε and sqrt as negligible)
    """

    name = "layernorm"
    description = "LayerNorm / RMSNorm over the last dimension"

    @property
    def param_dims(self) -> dict[str, str]:
        return {
            "N": "Total elements in all leading dimensions (batch * seq * …)",
            "D": "Hidden dimension (last axis)",
        }

    def compute_flops(self, N: int = 0, D: int = 0, **kwargs) -> int:
        return 6 * N * D

    def compute_io_bytes(
        self, dtype: DataType, N: int = 0, D: int = 0, **kwargs
    ) -> tuple[int, int]:
        bs = dtype.byte_size
        read_bytes = N * D * bs
        write_bytes = N * D * bs
        return read_bytes, write_bytes
