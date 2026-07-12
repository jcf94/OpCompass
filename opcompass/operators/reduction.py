from __future__ import annotations
"""Reduction operations (sum, max, mean) along a dimension.

Input:  (..., D) with N total elements
Output: (..., 1) or (...,) depending on keepdim

FLOPs ≈ N  (one binary-op reduction per element)
"""

from opcompass.models import (
    DataType, OperatorParameterSpec, OperatorSpec, OperatorValidationError,
)
from opcompass.operators.base import Operator


class Reduction(Operator):
    """Sum / max / mean reduction along one axis.

    Shapes:
        Input:  (N,) — N total elements
        Output: (N // D,) — reduced

    FLOPs ≈ N  (tree-reduction ≈ N-1 ≈ N for large N)
    """

    name = "reduction"
    description = "Reduction (sum, max, mean) along an axis"

    @property
    def param_dims(self) -> dict[str, str]:
        return {
            "N": "Total input elements",
            "D": "Reduction dimension size (output elements = N / D)",
        }

    @property
    def spec(self) -> OperatorSpec:
        return OperatorSpec(self.name, (
            OperatorParameterSpec("N", "Total input element count"),
            OperatorParameterSpec("D", "Reduced dimension; must divide N"),
        ))

    def compute_flops(self, N: int = 0, **kwargs) -> int:
        return N  # ~N-1 binary ops for tree reduction

    def validate_dimensions(self, dimensions):
        canonical = super().validate_dimensions(dimensions)
        if canonical["N"] % canonical["D"] != 0:
            raise OperatorValidationError(self.name, [{
                "parameter": "D",
                "reason": "constraint",
                "message": "parameter 'D' must divide 'N' exactly",
            }])
        return canonical

    def compute_io_bytes(
        self, dtype: DataType, N: int = 0, D: int = 0, **kwargs
    ) -> tuple[int, int]:
        bs = dtype.byte_size
        read_bytes = N * bs
        out_elements = N // D if D > 0 else 1
        write_bytes = out_elements * bs
        return read_bytes, write_bytes
