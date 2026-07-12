from __future__ import annotations
"""Reduction operations (sum, max, mean) along a dimension.

Input:  (..., D) with N total elements
Output: (..., 1) or (...,) depending on keepdim

FLOPs ≈ N  (one binary-op reduction per element)
"""

from opcompass.models import (
    DataType, OperatorParameterSpec, OperatorSpec, OperatorValidationError,
    ParameterKind, PipelineConfig, SubOp, TilingInfo,
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
            OperatorParameterSpec(
                "algorithm", "warp, block, or two_pass reduction strategy",
                value_type=str, required=False, default="auto", minimum=None,
                kind=ParameterKind.IMPLEMENTATION,
            ),
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
        if canonical["algorithm"] not in {"auto", "warp", "block", "two_pass"}:
            raise OperatorValidationError(self.name, [{
                "parameter": "algorithm", "reason": "choice",
                "message": "parameter 'algorithm' must be auto, warp, block, or two_pass",
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

    def select_algorithm(self, D: int, algorithm: str = "auto") -> str:
        if algorithm != "auto":
            return algorithm
        return "warp" if D <= 32 else "block" if D <= 4096 else "two_pass"

    def get_tiling_strategy(self, hardware, dtype=None, pipeline_config=None, **dims):
        algorithm = self.select_algorithm(dims["D"], dims.get("algorithm", "auto"))
        threads = 32 if algorithm == "warp" else 256
        return TilingInfo(1, 1, min(dims["D"], 1024), num_warps_per_block=threads // 32,
                          stage_count=2, candidate_name=algorithm)

    def get_ops_breakdown(self, dtype=None, hardware=None, pipeline_config=None, **dims):
        dtype = dtype or DataType.FP16
        rows, D = dims["N"] // dims["D"], dims["D"]
        algorithm = self.select_algorithm(D, dims.get("algorithm", "auto"))
        byte_count = D * dtype.byte_size
        ops = [
            SubOp("load_row", read_bytes=byte_count, pipeline_stage="global_read", is_recurring=True),
            SubOp("reduce_row", flops=max(1, D - 1), depends_on=["load_row"],
                  pipeline_stage="fma_alu", is_recurring=True),
        ]
        if algorithm in {"block", "two_pass"}:
            ops.insert(1, SubOp("shared_reduce", read_bytes=byte_count,
                               depends_on=["load_row"], pipeline_stage="shared_load",
                               is_recurring=True))
            ops[-1].depends_on = ["shared_reduce"]
        if algorithm == "two_pass":
            partials = (D + 1023) // 1024
            ops += [
                SubOp("write_partials", write_bytes=partials * dtype.byte_size,
                      depends_on=["reduce_row"], pipeline_stage="global_write"),
                SubOp("read_partials", read_bytes=partials * dtype.byte_size,
                      depends_on=["write_partials"], pipeline_stage="global_read"),
            ]
        ops.append(SubOp("write_output", write_bytes=rows * dtype.byte_size,
                         depends_on=[ops[-1].name], pipeline_stage="global_write"))
        return ops

    def get_pipeline_program(self, hardware, dtype, pipeline_config=None, **dims):
        from opcompass.engine.pipeline_ir import Launch, ResourceKind, program_from_sub_ops
        ops = self.get_ops_breakdown(dtype, hardware, pipeline_config, **dims)
        rows = dims["N"] // dims["D"]
        return program_from_sub_ops(
            ops,
            {"global_read": ResourceKind.HBM, "shared_load": ResourceKind.SHARED,
             "fma_alu": ResourceKind.COMPUTE, "global_write": ResourceKind.STORE},
            launch=Launch(grid_size=rows, compute_units=hardware.compute_unit.count),
        )
