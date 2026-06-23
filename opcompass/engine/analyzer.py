"""Main SOL analyzer — orchestrates memory, compute, and pipeline models."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opcompass.models import AnalysisMode, AnalysisResult, DataType, PipelineConfig
    from opcompass.operators.base import Operator
    from opcompass.hardware.base import Hardware


class Analyzer:
    """Entry point for SOL analysis.

    Usage::

        analyzer = Analyzer()
        result = analyzer.analyze(matmul_op, a100_hw, dtype=DataType.FP16, M=4096, N=4096, K=4096)

    For pipeline mode with feature toggles::

        config = PipelineConfig(async_copy_enabled=True, sparsity_2_4_enabled=False)
        result = analyzer.analyze(matmul_op, a100_hw, dtype=DataType.FP16,
                                  mode=AnalysisMode.PIPELINE, pipeline_config=config,
                                  M=4096, N=4096, K=4096)
    """

    def analyze(
        self,
        operator: Operator,
        hardware: Hardware,
        dtype: DataType,
        mode: AnalysisMode | None = None,
        pipeline_config: PipelineConfig | None = None,
        **dims: int,
    ) -> AnalysisResult:
        """Run SOL analysis and return an ``AnalysisResult``."""
        from opcompass.models import AnalysisMode, AnalysisResult, PipelineConfig

        if mode is None:
            mode = AnalysisMode.HIERARCHY_ROOFLINE

        # ── Solar mode: use SOLAR pytorch graph pipeline ──────────────
        if mode == AnalysisMode.SOLAR:
            return self._analyze_solar(operator, hardware, dtype, dims)

        # ── Pipeline mode: use the new DAG-based scheduler ────────────
        if mode == AnalysisMode.PIPELINE:
            return self._analyze_pipeline(operator, hardware, dtype, pipeline_config, dims)

        # ── HIERARCHY_ROOFLINE mode ────────────────────────────────────
        # 1. Fundamental quantities — always from the operator
        total_flops = operator.compute_flops(**dims)
        read_bytes, write_bytes = operator.compute_io_bytes(dtype, **dims)

        # 2. Per-phase times
        mem_read_time = self._estimate_memory_time(
            read_bytes, hardware
        )
        compute_time = self._estimate_compute_time(
            total_flops, hardware, dtype
        )
        mem_write_time = self._estimate_memory_time(
            write_bytes, hardware
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
                read_bytes, write_bytes, total_flops, hardware, dtype
            ),
            roofline_data=self._build_roofline_data(
                total_flops, read_bytes + write_bytes, hardware, dtype
            ),
        )
        return result

    # ------------------------------------------------------------------
    # Pipeline mode
    # ------------------------------------------------------------------

    def _analyze_pipeline(
        self, operator, hardware, dtype, pipeline_config, dims
    ) -> AnalysisResult:
        """Run pipeline analysis using DAG-based scheduling."""
        from opcompass.models import AnalysisMode, AnalysisResult, PipelineConfig
        from opcompass.engine.pipeline_model import schedule_pipeline

        # Default config if not provided
        if pipeline_config is None:
            pipeline_config = PipelineConfig()

        # Get tiling strategy
        tiling = operator.get_tiling_strategy(hardware, dtype, **dims)

        # Get sub-op decomposition
        sub_ops = operator.get_ops_breakdown(dtype, hardware, pipeline_config, **dims)

        # Fallback to roofline if operator doesn't support pipeline
        if not sub_ops or not tiling:
            total_flops = operator.compute_flops(**dims)
            read_bytes, write_bytes = operator.compute_io_bytes(dtype, **dims)
            mem_read_time = self._estimate_memory_time(read_bytes, hardware)
            compute_time = self._estimate_compute_time(total_flops, hardware, dtype)
            mem_write_time = self._estimate_memory_time(write_bytes, hardware)
            overlap = hardware.memory.can_overlap_with_compute
            sol_time = max(compute_time, mem_read_time, mem_write_time) if overlap else (mem_read_time + compute_time + mem_write_time)
            times = {"memory_read": mem_read_time, "compute": compute_time, "memory_write": mem_write_time}
            bottleneck = max(times, key=times.get)

            return AnalysisResult(
                operator=operator.name, hardware=hardware.name,
                shapes=dims, dtype=dtype, mode=AnalysisMode.PIPELINE,
                total_flops=total_flops,
                total_read_bytes=read_bytes, total_write_bytes=write_bytes,
                memory_read_time_s=mem_read_time,
                compute_time_s=compute_time,
                memory_write_time_s=mem_write_time,
                bottleneck=bottleneck, sol_time_s=sol_time,
                sol_tflops=(total_flops / sol_time / 1e12) if sol_time > 0 else float("inf"),
                stage_breakdown={"read": mem_read_time, "compute": compute_time, "write": mem_write_time},
            )

        # Schedule pipeline
        schedule = schedule_pipeline(sub_ops, hardware, pipeline_config, tiling, **dims)

        # Derive ALL metrics from pipeline schedule for consistency
        total_flops = operator.compute_flops(**dims)
        read_bytes, write_bytes = operator.compute_io_bytes(dtype, **dims)
        clock_s = 1.0 / (hardware.compute_unit.clock_mhz * 1e6)
        wave_count = schedule.wave_count

        # Per-stage busy time. These are "stage occupied" times — the sum of
        # each sub-op's duration within one block. They are NOT additive to
        # SOL because pipeline stages overlap (e.g. mma runs concurrently
        # with async_copy_load of the next iteration). We scale by wave_count
        # so the breakdown is on the same scale as SOL_time (which already
        # includes wave_count via schedule.total_time_s).
        stage_breakdown = {}
        for sop in schedule.sub_ops:
            stage = sop.pipeline_stage
            stage_breakdown[stage] = stage_breakdown.get(stage, 0) + sop.duration_cycles * clock_s * wave_count

        # Consolidate pipeline stages → read / compute / write
        _read_stages = {"async_copy_load", "global_read", "shared_load"}
        _compute_stages = {"mma", "fma_alu"}
        _write_stages = {"shared_store", "global_write"}

        mem_read_time = sum(
            t for stage, t in stage_breakdown.items()
            if stage in _read_stages
        )
        compute_time = sum(
            t for stage, t in stage_breakdown.items()
            if stage in _compute_stages
        )
        mem_write_time = sum(
            t for stage, t in stage_breakdown.items()
            if stage in _write_stages
        )

        sol_time = schedule.total_time_s
        sol_tflops = (total_flops / sol_time / 1e12) if sol_time > 0 else float("inf")

        # Roofline data from pipeline model rather than simple model
        peak_flops = hardware.get_peak_flops(dtype)
        peak_bw = hardware.hbm_bandwidth
        total_io = read_bytes + write_bytes
        oi = (total_flops / total_io) if total_io > 0 else float("inf")
        achievable = min(peak_flops, oi * peak_bw)

        return AnalysisResult(
            operator=operator.name,
            hardware=hardware.name,
            shapes=dims,
            dtype=dtype,
            mode=AnalysisMode.PIPELINE,
            total_flops=total_flops,
            total_read_bytes=read_bytes,
            total_write_bytes=write_bytes,
            memory_read_time_s=mem_read_time,
            compute_time_s=compute_time,
            memory_write_time_s=mem_write_time,
            bottleneck=schedule.bottleneck_stage,
            sol_time_s=sol_time,
            sol_tflops=sol_tflops,
            stage_breakdown={
                "read": mem_read_time,
                "compute": compute_time,
                "write": mem_write_time,
            },
            roofline_data={
                "operational_intensity": oi,
                "peak_flops": peak_flops,
                "peak_bandwidth": peak_bw,
                "achievable_flops": achievable,
            },
            pipeline_schedule=schedule,
            pipeline_config=pipeline_config,
            tiling_info=tiling,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_memory_time(
        self, byte_count: int, hardware: Hardware, tier_index: int = 0
    ) -> float:
        """Estimate the minimum time to move *byte_count* through memory."""
        if byte_count <= 0:
            return 0.0

        tiers = hardware.memory.tiers
        if tiers:
            return tiers[tier_index].transfer_time(byte_count)
        return 0.0

    def _estimate_compute_time(
        self, flops: int, hardware: Hardware, dtype
    ) -> float:
        """Estimate the minimum compute time."""
        if flops <= 0:
            return 0.0

        peak = hardware.get_peak_flops(dtype)
        if peak <= 0:
            return float("inf")

        utilization = 1.0
        return flops / (peak * utilization)

    def _build_stage_breakdown(
        self, read_bytes, write_bytes, flops, hardware, dtype
    ) -> dict[str, float]:
        """Build a per-stage time breakdown for display."""
        return {
            "read": self._estimate_memory_time(read_bytes, hardware),
            "compute": self._estimate_compute_time(flops, hardware, dtype),
            "write": self._estimate_memory_time(write_bytes, hardware),
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

    # ------------------------------------------------------------------
    # Solar mode
    # ------------------------------------------------------------------

    def _analyze_solar(
        self, operator, hardware, dtype, dims: dict
    ) -> AnalysisResult:
        """Run SOLAR pytorch graph analysis."""
        from opcompass.engine.solar_analyzer import SolarAnalyzer

        solar = SolarAnalyzer()
        return solar.analyze(operator, hardware, dtype, **dims)
