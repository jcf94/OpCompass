from __future__ import annotations
"""Element-wise operations (add, multiply, activation functions, etc.).

Input: (..., N)  — tensor of any shape with N total elements
Output: (..., N) — same shape

FLOPs = 1 * N  (one operation per element: multiply, add, exp, tanh, etc.)
Some element-wise ops like gelu/swish count as a few ops per element.
"""

from opcompass.models import DataType
from opcompass.operators.base import Operator


class Elementwise(Operator):
    """Generic element-wise unary or binary operation.

    Shape:
        Input:  (N,)  — N total elements (can be any-dimensional)
        Output: (N,)  — same shape

    FLOPs = ops_per_element * N
    """

    name = "elementwise"
    description = "Element-wise operations (add, mul, gelu, relu, …)"

    @property
    def param_dims(self) -> dict[str, str]:
        return {
            "N": "Total number of elements",
            "ops_per_element": "FLOPs per element (default 1 for simple ops, ~6 for gelu)",
        }

    def compute_flops(
        self, N: int = 0, ops_per_element: int = 1, **kwargs
    ) -> int:
        return ops_per_element * N

    def compute_io_bytes(
        self, dtype: DataType, N: int = 0, **kwargs
    ) -> tuple[int, int]:
        bs = dtype.byte_size
        read_bytes = N * bs
        write_bytes = N * bs
        return read_bytes, write_bytes
