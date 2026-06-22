from __future__ import annotations
"""2-D convolution: commonly used in CNNs.

Input:  (N, C_in, H, W)
Kernel: (C_out, C_in, K_h, K_w)
Output: (N, C_out, H_out, W_out)

FLOPs (per convention) = 2 * N * C_out * H_out * W_out * C_in * K_h * K_w
"""

from opcompass.models import DataType
from opcompass.operators.base import Operator


class Convolution(Operator):
    """2-D convolution (direct / implicit gemm).

    Shapes:
        Input:  (N, C_in, H, W)
        Kernel: (C_out, C_in, K_h, K_w)
        Output: (N, C_out, H_out, W_out)

    Assumes ``padding = same`` / ``stride = 1`` for simplicity
    (H_out = H, W_out = W).  This can be overridden via explicit dimensions.
    """

    name = "convolution"
    description = "2-D convolution: direct or implicit-GEMM form"

    @property
    def param_dims(self) -> dict[str, str]:
        return {
            "N": "Batch size",
            "C_in": "Input channels",
            "C_out": "Output channels",
            "H": "Input height",
            "W": "Input width",
            "K_h": "Kernel height",
            "K_w": "Kernel width",
            "H_out": "Output height (default = H)",
            "W_out": "Output width (default = W)",
        }

    def compute_flops(
        self,
        N: int = 0,
        C_in: int = 0,
        C_out: int = 0,
        H: int = 0,
        W: int = 0,
        K_h: int = 0,
        K_w: int = 0,
        H_out: int | None = None,
        W_out: int | None = None,
        **kwargs,
    ) -> int:
        if H_out is None:
            H_out = H
        if W_out is None:
            W_out = W
        return 2 * N * C_out * H_out * W_out * C_in * K_h * K_w

    def compute_io_bytes(
        self,
        dtype: DataType,
        N: int = 0,
        C_in: int = 0,
        C_out: int = 0,
        H: int = 0,
        W: int = 0,
        K_h: int = 0,
        K_w: int = 0,
        H_out: int | None = None,
        W_out: int | None = None,
        **kwargs,
    ) -> tuple[int, int]:
        if H_out is None:
            H_out = H
        if W_out is None:
            W_out = W
        bs = dtype.byte_size
        read_bytes = (N * C_in * H * W + C_out * C_in * K_h * K_w) * bs
        write_bytes = N * C_out * H_out * W_out * bs
        return read_bytes, write_bytes
