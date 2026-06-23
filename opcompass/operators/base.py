"""Abstract base class for all operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

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
        self, hardware: "Hardware", dtype=None, **dims: int
    ) -> TilingInfo | None:
        """Suggest a tiling / blocking strategy for the given hardware.

        Default returns None (engine uses a naïve strategy).

        Args:
            hardware: Target hardware (provides SM resources for constraint checks).
            dtype: Data type for the computation.
            **dims: Problem dimensions.
        """
        return None
