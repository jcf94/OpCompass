"""Main SOL analyzer — orchestrates memory, compute, and pipeline models."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opcompass.models import AnalysisMode, AnalysisResult, DataType
    from opcompass.operators.base import Operator
    from opcompass.hardware.base import Hardware


class Analyzer:
    """Entry point for SOL analysis.

    Usage::

        analyzer = Analyzer()
        result = analyzer.analyze(matmul_op, a100_hw, dtype=DataType.FP16, M=4096, N=4096, K=4096)
    """

    def analyze(
        self,
        operator: Operator,
        hardware: Hardware,
        dtype: DataType,
        mode: AnalysisMode | None = None,
        **dims: int,
    ) -> AnalysisResult:
        """Run SOL analysis and return an ``AnalysisResult``."""
        from opcompass.models import AnalysisMode, AnalysisResult

        if mode is None:
            mode = AnalysisMode.HIERARCHY

        # 1. Fundamental quantities — always from the operator
        total_flops = operator.compute_flops(**dims)
        read_bytes, write_bytes = operator.compute_io_bytes(dtype, **dims)

        # 2. Per-phase times
        mem_read_time = self._estimate_memory_time(
            read_bytes, hardware, mode
        )
        compute_time = self._estimate_compute_time(
            total_flops, hardware, dtype, mode
        )
        mem_write_time = self._estimate_memory_time(
            write_bytes, hardware, mode
        )

        # 3. Synthesis — factor in overlap
        overlap = hardware.memory.can_overlap_with_compute
        if overlap:
            sol_time = max(compute_time, mem_read_time, mem_write_time)
        else:
            sol_time = mem_read_time + compute_time + mem_write_time

        # 4. Identify bottleneck
        times = {
            "memory_read": mem_read_time,
            "compute": compute_time,
            "memory_write": mem_write_time,
        }
        bottleneck = max(times, key=times.get)  # type: ignore[arg-type]

        sol_tflops = (total_flops / sol_time / 1e12) if sol_time > 0 else float("inf")

        # 5. Assemble result
        result = AnalysisResult(
            operator=operator.name,
            hardware=hardware.name,
            shapes=dims,
            dtype=dtype,
            mode=mode,
            total_flops=total_flops,
            total_read_bytes=read_bytes,
            total_write_bytes=write_bytes,
            memory_read_time_s=mem_read_time,
            compute_time_s=compute_time,
            memory_write_time_s=mem_write_time,
            bottleneck=bottleneck,
            sol_time_s=sol_time,
            sol_tflops=sol_tflops,
            stage_breakdown=self._build_stage_breakdown(
                read_bytes, write_bytes, total_flops, hardware, dtype, mode
            ),
            roofline_data=self._build_roofline_data(
                total_flops, read_bytes + write_bytes, hardware, dtype
            ),
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_memory_time(
        self, byte_count: int, hardware: Hardware, mode, tier_index: int = 0
    ) -> float:
        """Estimate the minimum time to move *byte_count* through memory."""
        if byte_count <= 0:
            return 0.0

        from opcompass.models import AnalysisMode

        if mode == AnalysisMode.SIMPLE:
            # Use only HBM bandwidth
            bw = hardware.hbm_bandwidth
            return byte_count / bw if bw > 0 else 0.0

        # HIERARCHY / PIPELINE: use the first (slowest) tier for now;
        # a more sophisticated model would model reuse at each tier.
        tiers = hardware.memory.tiers
        if tiers:
            return tiers[tier_index].transfer_time(byte_count)
        return 0.0

    def _estimate_compute_time(
        self, flops: int, hardware: Hardware, dtype, mode
    ) -> float:
        """Estimate the minimum compute time."""
        if flops <= 0:
            return 0.0

        peak = hardware.get_peak_flops(dtype)
        if peak <= 0:
            return float("inf")

        # A simple utilization factor — can be refined per-hardware
        utilization = 1.0
        return flops / (peak * utilization)

    def _build_stage_breakdown(
        self, read_bytes, write_bytes, flops, hardware, dtype, mode
    ) -> dict[str, float]:
        """Build a per-stage time breakdown for display."""
        return {
            "read": self._estimate_memory_time(read_bytes, hardware, mode),
            "compute": self._estimate_compute_time(flops, hardware, dtype, mode),
            "write": self._estimate_memory_time(write_bytes, hardware, mode),
        }

    def _build_roofline_data(
        self, flops: int, io_bytes: int, hardware: Hardware, dtype
    ) -> dict:
        """Build data needed by a roofline plot."""
        peak_flops = hardware.get_peak_flops(dtype)
        peak_bw = hardware.hbm_bandwidth

        operational_intensity = (flops / io_bytes) if io_bytes > 0 else float("inf")
        achievable_flops = min(peak_flops, operational_intensity * peak_bw)

        return {
            "operational_intensity": operational_intensity,
            "peak_flops": peak_flops,
            "peak_bandwidth": peak_bw,
            "achievable_flops": achievable_flops,
        }
