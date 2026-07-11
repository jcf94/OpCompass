"""Abstract base class for all operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from opcompass.models import (
    OperatorParameterSpec,
    OperatorSpec,
    OperatorValidationError,
)

if TYPE_CHECKING:
    from opcompass.models import DataType, SubOp, TilingInfo


class Operator(ABC):
    """Abstract operator that every plug-in operator must subclass.

    Each operator lives in its own file under ``operators/``.
    """

    name: str = ""          # Unique short id, e.g. "matmul"
    description: str = ""   # Human-readable one-liner

    @property
    def param_dims(self) -> dict[str, str]:
        """Named dimension parameters that the operator expects.

        Example for matmul:
            {"M": "batch_rows", "N": "output_cols", "K": "inner_dim"}
        """
        return {}

    @property
    def spec(self) -> OperatorSpec:
        """Machine-readable parameter contract.

        Operators can override this to declare defaults, aliases, divisibility,
        or implementation parameters. The compatibility ``param_dims`` mapping
        remains the source for simple third-party operators.
        """
        return OperatorSpec(
            name=self.name,
            parameters=tuple(
                OperatorParameterSpec(name=name, description=description)
                for name, description in self.param_dims.items()
            ),
        )

    def validate_dimensions(self, dimensions: dict[str, Any]) -> dict[str, Any]:
        """Validate and canonicalize a concrete analysis request."""
        specs = self.spec.parameters
        by_name = {parameter.name: parameter for parameter in specs}
        aliases = {
            alias: parameter.name
            for parameter in specs
            for alias in parameter.aliases
        }
        canonical: dict[str, Any] = {}
        issues: list[dict[str, str]] = []

        for supplied_name, value in dimensions.items():
            name = aliases.get(supplied_name, supplied_name)
            if name not in by_name:
                issues.append({
                    "parameter": supplied_name,
                    "reason": "unknown",
                    "message": f"unknown parameter '{supplied_name}'",
                })
                continue
            if name in canonical:
                issues.append({
                    "parameter": name,
                    "reason": "duplicate",
                    "message": f"parameter '{name}' was supplied more than once",
                })
                continue
            canonical[name] = value

        for parameter in specs:
            if parameter.name not in canonical:
                if parameter.required:
                    issues.append({
                        "parameter": parameter.name,
                        "reason": "missing",
                        "message": f"missing required parameter '{parameter.name}'",
                    })
                elif parameter.default is not None:
                    canonical[parameter.name] = parameter.default
                continue

            value = canonical[parameter.name]
            if parameter.value_type is int and (not isinstance(value, int) or isinstance(value, bool)):
                issues.append({
                    "parameter": parameter.name,
                    "reason": "type",
                    "message": f"parameter '{parameter.name}' must be an integer",
                })
                continue
            if parameter.minimum is not None and value < parameter.minimum:
                issues.append({
                    "parameter": parameter.name,
                    "reason": "minimum",
                    "message": f"parameter '{parameter.name}' must be >= {parameter.minimum}",
                })
            if parameter.multiple_of and value % parameter.multiple_of != 0:
                issues.append({
                    "parameter": parameter.name,
                    "reason": "multiple_of",
                    "message": f"parameter '{parameter.name}' must be a multiple of {parameter.multiple_of}",
                })

        if issues:
            raise OperatorValidationError(self.name, issues)
        # Rebuild in declaration order so result shapes and future cache keys
        # are independent of request-object ordering.
        return {
            parameter.name: canonical[parameter.name]
            for parameter in specs
            if parameter.name in canonical
        }

    def compute_torch(self, inputs: list["torch.Tensor"], **kwargs) -> list["torch.Tensor"]:
        """Compute the operator using PyTorch (optional).

        Override this in subclasses to enable PyTorch-based validation.

        Args:
            inputs: List of input tensors.
            **kwargs: Additional keyword arguments (e.g., dimensions).

        Returns:
            List of output tensors.
        """
        raise NotImplementedError(f"{self.name}: compute_torch not implemented")

    @abstractmethod
    def compute_flops(self, **dims: int) -> int:
        """Return total floating-point operations for the given concrete dims."""
        ...

    @abstractmethod
    def compute_io_bytes(
        self, dtype: DataType, **dims: int
    ) -> tuple[int, int]:
        """Return (read_bytes, write_bytes) for the given concrete dims."""
        ...

    # ------------------------------------------------------------------
    # Solar mode support — generate a SOLAR-compatible model file
    # ------------------------------------------------------------------

    def get_solar_model_source(self, dtype, **dims: int) -> str:
        """Generate Python source for a SOLAR-compatible model file.

        The returned string must be a complete Python module containing:

        - A ``Model(torch.nn.Module)`` class whose ``forward()`` implements
          the operator computation.
        - A ``get_inputs()`` function that returns the input tensors (as a
          list or tuple) with the correct shapes and dtypes.

        Override this in subclasses to enable solar analysis mode.

        Args:
            dtype: Data type for the computation.
            **dims: Problem dimensions (e.g., M, N, K for matmul).

        Returns:
            Python source code as a string.
        """
        raise NotImplementedError(
            f"{self.name}: get_solar_model_source not implemented. "
            f"Solar mode requires this method to generate a SOLAR-compatible model."
        )

    # ------------------------------------------------------------------
    # Optional hooks — override these for finer-grained analysis
    # ------------------------------------------------------------------

    def get_ops_breakdown(self, dtype=None, hardware=None, pipeline_config=None, **dims: int) -> list[SubOp]:
        """Decompose this operator into a sequence of sub-operations.

        Used by the *pipeline* analysis mode.  Default returns an
        empty list, which means the engine falls back to a simpler model.

        Args:
            dtype: Data type for the computation.
            hardware: Target hardware (provides pipeline stages).
            pipeline_config: Feature toggles (async copy, sparsity, etc.).
            **dims: Problem dimensions (e.g., M, N, K for matmul).
        """
        return []

    def get_tiling_strategy(
        self, hardware: "Hardware", dtype=None, pipeline_config=None, **dims: int
    ) -> TilingInfo | None:
        """Suggest a tiling / blocking strategy for the given hardware.

        Default returns None (engine uses a naïve strategy).

        Args:
            hardware: Target hardware (provides SM resources for constraint checks).
            dtype: Data type for the computation.
            pipeline_config: Pipeline feature toggles and optional tile overrides.
            **dims: Problem dimensions.
        """
        return None

    def get_tile_constraints(self, hardware=None, dtype=None) -> dict:
        """Return tile alignment constraints for user-selected pipeline blocks.

        Operators can override this to expose instruction-level tile
        granularity, e.g. Tensor Core MMA shapes. The returned dictionary
        should contain ``block_m``, ``block_n``, and ``block_k`` entries.
        """
        return {
            "block_m": {"multiple_of": 1, "min": 1},
            "block_n": {"multiple_of": 1, "min": 1},
            "block_k": {"multiple_of": 1, "min": 1},
        }
