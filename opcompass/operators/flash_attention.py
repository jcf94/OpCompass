from __future__ import annotations
"""Flash Attention: fused scaled dot-product attention.

Standard attention:
    S = Q @ K^T                    (batch, heads, seq, seq)
    P = softmax(S * scale)         element-wise
    O = P @ V                      (batch, heads, seq, head_dim)

With FlashAttention the I/O is greatly reduced because the full S/P
matrices are never materialised in HBM — they are computed in tiles
and kept in on-chip SRAM.

The FLOPs count is the same as standard attention; the I/O model
reflects the tiled, fused approach (reads Q, K, V once each; writes O once).
"""

from opcompass.models import DataType
from opcompass.operators.base import Operator


class FlashAttention(Operator):
    """Flash Attention (fused, tiled).

    Shapes:
        Q: (B, H, S, D)
        K: (B, H, S, D)
        V: (B, H, S, D)
        O: (B, H, S, D)

    FLOPs = 4 * B * H * S^2 * D
      (2 * S * D * S for QK^T, plus 2 * S * S * D for PV, each counted as 2 ops per madd)
    """

    name = "flash_attention"
    description = "Flash Attention — fused scaled dot-product attention (tiled, I/O optimal)"

    @property
    def param_dims(self) -> dict[str, str]:
        return {
            "B": "Batch size",
            "H": "Number of attention heads",
            "S": "Sequence length",
            "D": "Head dimension (d_k = d_v)",
        }

    def compute_flops(
        self, B: int = 0, H: int = 0, S: int = 0, D: int = 0, **kwargs
    ) -> int:
        # S @ K^T: 2 * B * H * S * D * S = 2 * B * H * S^2 * D
        # P @ V:   2 * B * H * S * S * D = 2 * B * H * S^2 * D
        return 4 * B * H * S * S * D

    def compute_io_bytes(
        self, dtype: DataType, B: int = 0, H: int = 0, S: int = 0, D: int = 0, **kwargs
    ) -> tuple[int, int]:
        bs = dtype.byte_size
        # FlashAttention: Q, K, V each read once; O written once.
        # The intermediate S (B*H*S*S) never goes to HBM.
        read_bytes = 3 * B * H * S * D * bs
        write_bytes = B * H * S * D * bs
        return read_bytes, write_bytes
