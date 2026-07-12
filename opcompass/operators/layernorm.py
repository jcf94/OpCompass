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

from opcompass.models import (
    DataType, OperatorParameterSpec, OperatorSpec, OperatorValidationError,
    ParameterKind, SubOp, TilingInfo,
)
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

    @property
    def spec(self) -> OperatorSpec:
        return OperatorSpec(self.name, (
            OperatorParameterSpec("N", "Product of all leading dimensions"),
            OperatorParameterSpec("D", "Normalized hidden dimension"),
            OperatorParameterSpec("variant", "layernorm or rmsnorm", value_type=str,
                                  required=False, default="layernorm", minimum=None,
                                  kind=ParameterKind.IMPLEMENTATION),
            OperatorParameterSpec("algorithm", "one_pass, two_pass, or online",
                                  value_type=str, required=False, default="auto",
                                  minimum=None, kind=ParameterKind.IMPLEMENTATION),
        ))

    def validate_dimensions(self, dimensions):
        canonical = super().validate_dimensions(dimensions)
        if canonical["variant"] not in {"layernorm", "rmsnorm"}:
            raise OperatorValidationError(self.name, [{"parameter": "variant", "reason": "choice", "message": "variant must be layernorm or rmsnorm"}])
        if canonical["algorithm"] not in {"auto", "one_pass", "two_pass", "online"}:
            raise OperatorValidationError(self.name, [{"parameter": "algorithm", "reason": "choice", "message": "algorithm must be auto, one_pass, two_pass, or online"}])
        return canonical

    def compute_flops(self, N: int = 0, D: int = 0, variant: str = "layernorm", **kwargs) -> int:
        return (5 if variant == "rmsnorm" else 6) * N * D

    def compute_io_bytes(
        self, dtype: DataType, N: int = 0, D: int = 0, **kwargs
    ) -> tuple[int, int]:
        bs = dtype.byte_size
        # Input plus gamma, and beta for LayerNorm.
        read_bytes = N * D * bs + D * bs * (1 if kwargs.get("variant") == "rmsnorm" else 2)
        write_bytes = N * D * bs
        return read_bytes, write_bytes

    def select_algorithm(self, D, algorithm="auto"):
        return ("one_pass" if D <= 1024 else "online" if D <= 8192 else "two_pass") if algorithm == "auto" else algorithm

    def get_tiling_strategy(self, hardware, dtype=None, pipeline_config=None, **dims):
        algorithm = self.select_algorithm(dims["D"], dims.get("algorithm", "auto"))
        return TilingInfo(1, 1, min(dims["D"], 1024), num_warps_per_block=8,
                          stage_count=2, candidate_name=algorithm)

    def get_ops_breakdown(self, dtype=None, hardware=None, pipeline_config=None, **dims):
        dtype = dtype or DataType.FP16
        N, D = dims["N"], dims["D"]
        variant = dims.get("variant", "layernorm")
        algorithm = self.select_algorithm(D, dims.get("algorithm", "auto"))
        row_bytes = D * dtype.byte_size
        ops = [
            SubOp("vector_load", read_bytes=row_bytes, pipeline_stage="global_read", is_recurring=True),
            SubOp("shared_reduce", read_bytes=row_bytes, depends_on=["vector_load"], pipeline_stage="shared_load", is_recurring=True),
            SubOp("statistics", flops=(2 if variant == "rmsnorm" else 3) * D,
                  depends_on=["shared_reduce"], pipeline_stage="fma_alu", is_recurring=True),
            SubOp("barrier", flops=1, depends_on=["statistics"], pipeline_stage="fma_alu", is_recurring=True),
            SubOp("rsqrt", flops=8, depends_on=["barrier"], pipeline_stage="fma_alu", is_recurring=True),
            SubOp("affine_convert", flops=3 * D, depends_on=["rsqrt"], pipeline_stage="fma_alu", is_recurring=True),
        ]
        if algorithm == "two_pass":
            ops += [
                SubOp("partial_statistics", write_bytes=N * dtype.byte_size * 2,
                      depends_on=["statistics"], pipeline_stage="global_write"),
                SubOp("reload_statistics", read_bytes=N * dtype.byte_size * 2,
                      depends_on=["partial_statistics"], pipeline_stage="global_read"),
            ]
        ops.append(SubOp("output_store", write_bytes=N * D * dtype.byte_size,
                         depends_on=[ops[-1].name], pipeline_stage="global_write"))
        return ops

    def get_pipeline_program(self, hardware, dtype, pipeline_config=None, **dims):
        from opcompass.engine.pipeline_ir import Launch, ResourceKind, program_from_sub_ops
        ops = self.get_ops_breakdown(dtype, hardware, pipeline_config, **dims)
        return program_from_sub_ops(
            ops,
            {"global_read": ResourceKind.HBM, "shared_load": ResourceKind.SHARED,
             "fma_alu": ResourceKind.COMPUTE, "global_write": ResourceKind.STORE},
            launch=Launch(grid_size=dims["N"], compute_units=hardware.compute_unit.count),
        )
