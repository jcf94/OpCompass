"""Solar analysis engine — bridges OpCompass operators to SOLAR's pytorch graph pipeline.

Uses ``3rdparty/SOLAR`` to extract a torch computation graph from an operator's
``compute_torch`` implementation, then runs SOLAR's full pipeline:

  1. PyTorch graph extraction (torchview)
  2. Einsum conversion
  3. Hardware-independent analysis (MACs, memory elements)
  4. Roofline performance prediction against GPU arch configs

The result is a richer SOL estimate that accounts for fused vs unfused
memory, intermediate tensors, and per-op bottlenecks.

Dependencies: torch, torchview, pyyaml (required by SOLAR).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from opcompass.models import AnalysisResult, DataType, SolarAnalysisData
    from opcompass.operators.base import Operator
    from opcompass.hardware.base import Hardware


# ---------------------------------------------------------------------------
# Path to SOLAR (vendored under 3rdparty)
# ---------------------------------------------------------------------------
_SOLAR_ROOT = Path(__file__).resolve().parents[2] / "3rdparty" / "SOLAR"
_SOLAR_ARCH_DIR = Path(__file__).resolve().parents[1] / "configs" / "solar_arch"


# ---------------------------------------------------------------------------
# Dependency checks (lazy — only when SolarAnalyzer is actually used)
# ---------------------------------------------------------------------------

def _check_solar_dependencies():
    """Raise a helpful error if SOLAR dependencies are missing."""
    missing = []
    try:
        import yaml  # noqa: F401
    except ImportError:
        missing.append("pyyaml")
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import torchview  # noqa: F401
    except ImportError:
        missing.append("torchview")

    if missing:
        raise ImportError(
            f"Solar analysis mode requires additional dependencies: {', '.join(missing)}. "
            f"Install them with:\n"
            f"  pip install {' '.join(missing)}\n"
            f"Or follow SOLAR's install guide: 3rdparty/SOLAR/install.sh"
        )


# ---------------------------------------------------------------------------
# Path to SOLAR (vendored under 3rdparty)
# ---------------------------------------------------------------------------
_SOLAR_ROOT = Path(__file__).resolve().parents[2] / "3rdparty" / "SOLAR"
_SOLAR_ARCH_DIR = Path(__file__).resolve().parents[1] / "configs" / "solar_arch"


# ---------------------------------------------------------------------------
# Mapping from OpCompass hardware → SOLAR arch config
# ---------------------------------------------------------------------------
HARDWARE_TO_SOLAR_ARCH: Dict[str, str] = {
    "a100": str(_SOLAR_ARCH_DIR / "A100.yaml"),
    "h100": str(_SOLAR_ARCH_DIR / "H100.yaml"),
    "h100_pcie": str(_SOLAR_ARCH_DIR / "H100_PCIe.yaml"),
}

# Add the built-in SOLAR arch configs for hardware we don't have custom configs for
_SOLAR_BUILTIN = _SOLAR_ROOT / "configs" / "arch"
for _name in ["A6000", "B200", "Jetson_Thor_T5000"]:
    _cfg = _SOLAR_BUILTIN / f"{_name}.yaml"
    if _cfg.exists():
        HARDWARE_TO_SOLAR_ARCH[_name.lower()] = str(_cfg)

# H100_PCIe also available from SOLAR builtin
if (_SOLAR_BUILTIN / "H100_PCIe.yaml").exists():
    HARDWARE_TO_SOLAR_ARCH.setdefault("h100_pcie", str(_SOLAR_BUILTIN / "H100_PCIe.yaml"))


class SolarAnalyzer:
    """Run SOLAR's pytorch graph analysis on an OpCompass operator.

    Usage::

        analyzer = SolarAnalyzer()
        result = analyzer.analyze(matmul_op, a100_hw, dtype=DataType.FP16,
                                  M=4096, N=4096, K=4096)
    """

    def analyze(
        self,
        operator: Operator,
        hardware: Hardware,
        dtype: DataType,
        **dims: int,
    ) -> AnalysisResult:
        """Run solar analysis and return an ``AnalysisResult``.

        Args:
            operator: The operator instance to analyze.
            hardware: Target hardware (used to select SOLAR arch config).
            dtype: Numerical data type.
            **dims: Problem dimensions (e.g. M=4096, N=4096, K=4096).

        Returns:
            AnalysisResult with solar_data populated.
        """
        from opcompass.models import AnalysisMode, AnalysisResult, SolarAnalysisData

        # Check dependencies before attempting SOLAR import
        _check_solar_dependencies()

        # Resolve arch config path
        arch_path = self._resolve_arch(hardware.name)

        # Generate model source and run SOLAR pipeline
        model_source = operator.get_solar_model_source(dtype, **dims)

        with tempfile.TemporaryDirectory(prefix="opcompass_solar_") as tmpdir:
            tmp = Path(tmpdir)

            # Write the model file
            model_file = tmp / "model.py"
            model_file.write_text(model_source)

            # Stage directories
            graph_dir = tmp / "graph"
            einsum_dir = tmp / "einsum"
            analysis_dir = tmp / "analysis"
            perf_dir = tmp / "perf"

            # ── Stage 1: PyTorch graph extraction ──────────────────────
            solar_data = self._run_solar_pipeline(
                model_file, graph_dir, einsum_dir, analysis_dir, perf_dir,
                arch_path, dtype,
            )

        # Compute fundamental quantities (from operator, for consistency)
        total_flops = operator.compute_flops(**dims)
        read_bytes, write_bytes = operator.compute_io_bytes(dtype, **dims)

        # Map SOLAR's fused_prefetched result (best case) to our SOL time
        sol_time_s = solar_data.fused_prefetched_runtime_ms / 1000.0 if solar_data.fused_prefetched_runtime_ms > 0 else 0.0
        sol_tflops = (total_flops / sol_time_s / 1e12) if sol_time_s > 0 else float("inf")

        # Traditional phase times using our own models
        mem_read_time = self._estimate_memory_time(read_bytes, hardware)
        compute_time = self._estimate_compute_time(total_flops, hardware, dtype)
        mem_write_time = self._estimate_memory_time(write_bytes, hardware)

        return AnalysisResult(
            operator=operator.name,
            hardware=hardware.name,
            shapes=dims,
            dtype=dtype,
            mode=AnalysisMode.SOLAR,
            total_flops=total_flops,
            total_read_bytes=read_bytes,
            total_write_bytes=write_bytes,
            memory_read_time_s=mem_read_time,
            compute_time_s=compute_time,
            memory_write_time_s=mem_write_time,
            bottleneck=solar_data.fused_prefetched_bottleneck,
            sol_time_s=sol_time_s,
            sol_tflops=sol_tflops,
            stage_breakdown={
                "read": mem_read_time,
                "compute": compute_time,
                "write": mem_write_time,
            },
            roofline_data=self._build_roofline_data(
                total_flops, read_bytes + write_bytes, hardware, dtype
            ),
            solar_data=solar_data,
        )

    # ------------------------------------------------------------------
    # Internal: SOLAR pipeline
    # ------------------------------------------------------------------

    def _run_solar_pipeline(
        self,
        model_file: Path,
        graph_dir: Path,
        einsum_dir: Path,
        analysis_dir: Path,
        perf_dir: Path,
        arch_path: str,
        dtype: DataType,
    ) -> SolarAnalysisData:
        """Execute SOLAR's 4-stage pipeline and return parsed data."""
        from opcompass.models import SolarAnalysisData

        # Ensure SOLAR is on sys.path
        solar_root_str = str(_SOLAR_ROOT)
        if solar_root_str not in sys.path:
            sys.path.insert(0, solar_root_str)

        # Stage 1: Graph extraction
        from solar.graph import PyTorchProcessor
        from solar.common.types import ProcessingConfig

        proc_config = ProcessingConfig(debug=False, safe_mode=False, force_rerun=True)
        processor = PyTorchProcessor(config=proc_config)
        ok = processor.process_model_file(str(model_file), str(graph_dir))
        if not ok:
            raise RuntimeError(
                f"SOLAR Stage 1 (graph extraction) failed for {model_file}"
            )

        pytorch_graph = graph_dir / "pytorch_graph.yaml"
        if not pytorch_graph.exists():
            raise RuntimeError(
                f"SOLAR Stage 1 did not produce pytorch_graph.yaml in {graph_dir}"
            )

        # Stage 2: Einsum conversion
        from solar.einsum import PyTorchToEinsum

        converter = PyTorchToEinsum(debug=False)
        einsum_result = converter.convert(
            str(pytorch_graph), str(einsum_dir), enable_rename=False
        )
        if einsum_result is None:
            raise RuntimeError(
                f"SOLAR Stage 2 (einsum conversion) failed for {pytorch_graph}"
            )

        einsum_graph = einsum_dir / "einsum_graph.yaml"
        if not einsum_graph.exists():
            raise RuntimeError(
                f"SOLAR Stage 2 did not produce einsum graph in {einsum_dir}"
            )

        # Stage 3: Hardware-independent analysis
        from solar.analysis import EinsumGraphAnalyzer

        solar_precision = _dtype_to_solar_precision(dtype)
        graph_analyzer = EinsumGraphAnalyzer(debug=False)
        analysis = graph_analyzer.analyze_graph(
            str(einsum_graph), str(analysis_dir), precision=solar_precision
        )
        if analysis is None:
            raise RuntimeError(
                f"SOLAR Stage 3 (analysis) failed for {einsum_graph}"
            )

        analysis_yaml = analysis_dir / "analysis.yaml"
        if not analysis_yaml.exists():
            raise RuntimeError(
                f"SOLAR Stage 3 did not produce analysis.yaml in {analysis_dir}"
            )

        # Stage 4: Performance prediction
        from solar.perf import EinsumGraphPerfModel

        perf_model = EinsumGraphPerfModel(debug=False)
        perf = perf_model.predict(
            str(analysis_yaml), str(perf_dir),
            arch_config=arch_path, precision=solar_precision,
        )
        if perf is None:
            raise RuntimeError(
                f"SOLAR Stage 4 (perf prediction) failed for {analysis_yaml}"
            )

        # ── Map SOLAR perf output to SolarAnalysisData ─────────────────
        unfused = perf.get("unfused", {})
        fused = perf.get("fused", {})
        fused_prefetched = perf.get("fused_prefetched", {})
        workload = perf.get("workload", {})
        memory_breakdown = perf.get("memory_breakdown", {})
        speedup = perf.get("speedup", {})
        arch = perf.get("arch", {})

        return SolarAnalysisData(
            unfused_runtime_ms=float(unfused.get("runtime_ms", 0)),
            unfused_bottleneck=str(unfused.get("bottleneck", "")),
            unfused_arithmetic_intensity=float(unfused.get("arithmetic_intensity", 0)),
            unfused_memory_bytes=int(unfused.get("memory_bytes", 0)),
            unfused_compute_cycles=int(unfused.get("compute_cycles", 0)),
            fused_runtime_ms=float(fused.get("runtime_ms", 0)),
            fused_bottleneck=str(fused.get("bottleneck", "")),
            fused_arithmetic_intensity=float(fused.get("arithmetic_intensity", 0)),
            fused_memory_bytes=int(fused.get("memory_bytes", 0)),
            fused_prefetched_runtime_ms=float(fused_prefetched.get("runtime_ms", 0)),
            fused_prefetched_bottleneck=str(fused_prefetched.get("bottleneck", "")),
            fused_prefetched_arithmetic_intensity=float(fused_prefetched.get("arithmetic_intensity", 0)),
            fused_prefetched_memory_bytes=int(fused_prefetched.get("memory_bytes", 0)),
            total_macs=int(workload.get("total_macs", 0)),
            total_flops=int(workload.get("total_flops", 0)),
            num_layers=0,  # filled below if available
            weight_bytes=int(memory_breakdown.get("weight_bytes", 0)),
            model_io_bytes=int(memory_breakdown.get("model_io_bytes", 0)),
            intermediate_bytes=int(memory_breakdown.get("intermediate_bytes", 0)),
            fused_speedup=float(speedup.get("fused_vs_unfused", 1.0)),
            fused_prefetched_speedup=float(speedup.get("fused_prefetched_vs_unfused", 1.0)),
            arch_name=str(arch.get("name", "")),
            arch_freq_ghz=float(arch.get("freq_GHz", 1.0)),
            arch_mac_per_cycle=float(arch.get("MAC_per_cycle", 1.0)),
            arch_dram_bw_per_cycle=float(arch.get("DRAM_byte_per_cycle", 1.0)),
            mac_per_cycle_key=str(arch.get("mac_per_cycle_key", "")),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_arch(self, hardware_name: str) -> str:
        """Map an OpCompass hardware name to a SOLAR arch config path."""
        name_lower = hardware_name.lower()
        if name_lower in HARDWARE_TO_SOLAR_ARCH:
            return HARDWARE_TO_SOLAR_ARCH[name_lower]

        # Try as a direct path to a file
        direct_path = _SOLAR_ARCH_DIR / f"{name_lower}.yaml"
        if direct_path.exists():
            return str(direct_path)

        # Try SOLAR builtin
        builtin = _SOLAR_BUILTIN / f"{name_lower}.yaml"
        if builtin.exists():
            return str(builtin)

        raise ValueError(
            f"No SOLAR arch config found for hardware '{hardware_name}'. "
            f"Available: {sorted(HARDWARE_TO_SOLAR_ARCH.keys())}. "
            f"Place a YAML config in {_SOLAR_ARCH_DIR}."
        )

    def _estimate_memory_time(self, byte_count: int, hardware: Hardware) -> float:
        if byte_count <= 0:
            return 0.0
        tiers = hardware.memory.tiers
        if tiers:
            return tiers[0].transfer_time(byte_count)
        return 0.0

    def _estimate_compute_time(self, flops: int, hardware: Hardware, dtype) -> float:
        if flops <= 0:
            return 0.0
        peak = hardware.get_peak_flops(dtype)
        if peak <= 0:
            return float("inf")
        return flops / peak

    def _build_roofline_data(
        self, flops: int, io_bytes: int, hardware: Hardware, dtype
    ) -> dict:
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dtype_to_solar_precision(dtype: DataType) -> str:
    """Map OpCompass DataType to a SOLAR precision string."""
    from opcompass.models import DataType

    _map = {
        DataType.FP64: "fp64",
        DataType.FP32: "fp32",
        DataType.TF32: "tf32",
        DataType.FP16: "fp16",
        DataType.BF16: "bf16",
        DataType.INT8: "int8",
        DataType.FP8: "fp8",
        DataType.INT4: "int4",
    }
    return _map.get(dtype, "fp16")  # type: ignore[arg-type]
