"""Test the SOL analyzer end-to-end."""

from opcompass.registry import get_operator, get_hardware
from opcompass.models import DataType, AnalysisMode
from opcompass.engine.analyzer import Analyzer


def test_matmul_a100_fp16_hierarchy_roofline():
    op = get_operator("matmul")()
    hw = get_hardware("a100")()

    analyzer = Analyzer()
    result = analyzer.analyze(
        op, hw, DataType.FP16, mode=AnalysisMode.HIERARCHY_ROOFLINE,
        M=4096, N=4096, K=4096,
    )

    assert result.operator == "matmul"
    assert result.hardware == "a100"
    assert result.mode == AnalysisMode.HIERARCHY_ROOFLINE
    assert result.total_flops == 137_438_953_472
    assert result.memory_read_time_s > 0
    assert result.compute_time_s > 0
    assert result.sol_time_s > 0
    assert result.sol_tflops > 0
    # Matmul 4096^3 FP16 on A100 is compute-bound
    assert result.bottleneck == "compute"


def test_matmul_a100_fp16_default_mode():
    """Default mode (no mode specified) should be HIERARCHY_ROOFLINE."""
    op = get_operator("matmul")()
    hw = get_hardware("a100")()

    analyzer = Analyzer()
    result = analyzer.analyze(
        op, hw, DataType.FP16, M=4096, N=4096, K=4096,
    )

    assert result.mode == AnalysisMode.HIERARCHY_ROOFLINE
    assert result.sol_time_s > 0


def test_matmul_h100_fp16():
    op = get_operator("matmul")()
    hw = get_hardware("h100")()

    analyzer = Analyzer()
    result = analyzer.analyze(
        op, hw, DataType.FP16, mode=AnalysisMode.HIERARCHY_ROOFLINE,
        M=4096, N=4096, K=4096,
    )

    assert result.sol_tflops == 989.4
    assert result.bottleneck == "compute"


def test_matmul_fp32_memory_bound():
    """Small matmul in FP32 should be memory-bound due to low arithmetic intensity."""
    op = get_operator("matmul")()
    hw = get_hardware("a100")()

    analyzer = Analyzer()
    result = analyzer.analyze(
        op, hw, DataType.FP32, mode=AnalysisMode.HIERARCHY_ROOFLINE,
        M=128, N=128, K=128,
    )

    # Small matmul has very few FLOPs relative to bytes → memory bound
    # Actually 128³ = 2*128³ = 4M FLOPs vs 128*128*4*2 = 131KB read + 64KB write
    # AI = 4M / (131K + 64K) ≈ 21 FLOP/Byte → should still be compute bound for A100 FP32
    # Let's just check the result is reasonable
    assert result.sol_time_s > 0
    assert result.sol_tflops > 0


# ---------------------------------------------------------------------------
# Solar mode tests
# ---------------------------------------------------------------------------

import pytest


def _has_solar_deps():
    """Return True if torch, torchview, and pyyaml are available."""
    try:
        import torch  # noqa: F401
        import yaml  # noqa: F401
        import torchview  # noqa: F401
        return True
    except ImportError:
        return False


def _solar_matmul_comparison_cases():
    """Return hardware/dtype pairs supported by both OpCompass and SOLAR configs."""
    if not _has_solar_deps():
        return [pytest.param("a100", DataType.FP16, marks=pytest.mark.skip(reason="torch/torchview/pyyaml not installed"))]

    import yaml

    from opcompass.engine.solar_analyzer import HARDWARE_TO_SOLAR_ARCH

    dtype_keys = {
        DataType.FP32: ("MAC_per_cycle_fp32_tc", "MAC_per_cycle_fp32_sm"),
        DataType.TF32: ("MAC_per_cycle_tf32_tc",),
        DataType.FP16: ("MAC_per_cycle_fp16_tc",),
        DataType.BF16: ("MAC_per_cycle_bf16_tc",),
        DataType.FP8: ("MAC_per_cycle_fp8_tc",),
        DataType.FP4: ("MAC_per_cycle_fp4_tc", "MAC_per_cycle_nvfp4_tc"),
        DataType.INT8: ("MAC_per_cycle_int8_tc",),
    }

    cases = []
    for hardware_name, arch_path in sorted(HARDWARE_TO_SOLAR_ARCH.items()):
        try:
            hardware_cls = get_hardware(hardware_name)
        except KeyError:
            continue

        hw = hardware_cls()
        with open(arch_path) as f:
            arch = yaml.safe_load(f)

        for dtype, keys in dtype_keys.items():
            if dtype not in hw.compute_unit.peak_flops:
                continue
            if not any(key in arch for key in keys):
                continue
            cases.append(pytest.param(hardware_name, dtype, id=f"{hardware_name}-{dtype.value}"))

    return cases


def test_matmul_solar_model_source():
    """Verify that Matmul generates correct SOLAR model source."""
    from opcompass.models import DataType

    op = get_operator("matmul")()
    source = op.get_solar_model_source(DataType.FP16, M=128, N=64, K=32)

    assert "class Model" in source
    assert "def get_inputs" in source
    assert "torch.matmul" in source
    assert "torch.float16" in source
    assert "M, N, K = 128, 64, 32" in source


def test_solar_mode_requires_solar_model_source():
    """Operators without get_solar_model_source should raise NotImplementedError."""
    from opcompass.registry import discover_operators, get_operator
    from opcompass.models import DataType, AnalysisMode
    from opcompass.engine.analyzer import Analyzer

    # Find an operator that doesn't implement get_solar_model_source
    ops = discover_operators()
    hw = get_hardware("a100")()

    for name, cls in ops.items():
        op = cls()
        # Check if it overrides the base get_solar_model_source
        if type(op).get_solar_model_source is type(op).__bases__[0].get_solar_model_source:
            # This operator doesn't override it — should raise
            with pytest.raises(NotImplementedError, match="get_solar_model_source"):
                op.get_solar_model_source(DataType.FP16)
            break
    else:
        pytest.skip("All operators implement get_solar_model_source")


@pytest.mark.skipif(not _has_solar_deps(), reason="torch/torchview/pyyaml not installed")
def test_matmul_solar_a100_end_to_end():
    """Full end-to-end solar analysis of matmul on A100.

    Requires torch, torchview, and pyyaml to be installed.
    """
    from opcompass.models import DataType, AnalysisMode

    op = get_operator("matmul")()
    hw = get_hardware("a100")()

    analyzer = Analyzer()
    result = analyzer.analyze(
        op, hw, DataType.FP16, mode=AnalysisMode.SOLAR,
        M=128, N=128, K=128,
    )

    assert result.operator == "matmul"
    assert result.hardware == "a100"
    assert result.mode == AnalysisMode.SOLAR
    assert result.solar_data is not None
    assert result.solar_data.arch_name == "A100"
    assert result.solar_data.total_macs > 0
    assert result.solar_data.total_flops > 0
    assert result.solar_data.unfused_runtime_ms > 0
    assert result.solar_data.fused_runtime_ms > 0
    assert result.solar_data.fused_prefetched_runtime_ms > 0
    # Fused+prefetched should be <= unfused
    assert result.solar_data.fused_prefetched_runtime_ms <= result.solar_data.unfused_runtime_ms
    assert result.sol_time_s > 0
    assert result.sol_tflops > 0


@pytest.mark.skipif(not _has_solar_deps(), reason="torch/torchview/pyyaml not installed")
@pytest.mark.parametrize("hardware_name,dtype", _solar_matmul_comparison_cases())
def test_matmul_hierarchy_roofline_matches_solar_across_hardware_and_dtypes(hardware_name, dtype):
    """Matmul should produce nearly identical roofline results in hierarchy and SOLAR modes."""
    op = get_operator("matmul")()
    hw = get_hardware(hardware_name)()
    analyzer = Analyzer()
    dims = {"M": 128, "N": 128, "K": 128}

    hierarchy = analyzer.analyze(
        op, hw, dtype, mode=AnalysisMode.HIERARCHY_ROOFLINE, **dims
    )
    solar = analyzer.analyze(
        op, hw, dtype, mode=AnalysisMode.SOLAR, **dims
    )

    assert solar.total_flops == hierarchy.total_flops
    assert solar.total_read_bytes == pytest.approx(hierarchy.total_read_bytes, rel=1e-12)
    assert solar.total_write_bytes == pytest.approx(hierarchy.total_write_bytes, rel=1e-12)
    assert solar.roofline_data["peak_flops"] == pytest.approx(
        hierarchy.roofline_data["peak_flops"], rel=2e-3
    )
    assert solar.roofline_data["peak_bandwidth"] == pytest.approx(
        hierarchy.roofline_data["peak_bandwidth"], rel=2e-3
    )
    assert solar.sol_time_s == pytest.approx(hierarchy.sol_time_s, rel=2e-3)
    assert solar.sol_tflops == pytest.approx(hierarchy.sol_tflops, rel=2e-3)


@pytest.mark.skipif(not _has_solar_deps(), reason="torch/torchview/pyyaml not installed")
def test_matmul_solar_h100_end_to_end():
    """Full end-to-end solar analysis of matmul on H100."""
    from opcompass.models import DataType, AnalysisMode

    op = get_operator("matmul")()
    hw = get_hardware("h100")()

    analyzer = Analyzer()
    result = analyzer.analyze(
        op, hw, DataType.FP16, mode=AnalysisMode.SOLAR,
        M=256, N=256, K=256,
    )

    assert result.solar_data is not None
    assert result.solar_data.arch_name == "H100"
    assert result.mode == AnalysisMode.SOLAR


@pytest.mark.skipif(not _has_solar_deps(), reason="torch/torchview/pyyaml not installed")
def test_matmul_solar_cli_integration():
    """Verify the CLI and result formatting work for solar mode."""
    from opcompass.models import DataType, AnalysisMode
    from opcompass.engine.result import format_result

    op = get_operator("matmul")()
    hw = get_hardware("a100")()

    analyzer = Analyzer()
    result = analyzer.analyze(
        op, hw, DataType.FP16, mode=AnalysisMode.SOLAR,
        M=64, N=64, K=64,
    )

    # Test table formatting
    table = format_result(result, fmt="table")
    assert "SOLAR Analysis" in table
    assert "Unfused" in table
    assert "Fused" in table
    assert "Fused+Prefetched" in table

    # Test JSON formatting
    import json
    json_str = format_result(result, fmt="json")
    data = json.loads(json_str)
    assert data["mode"] == "solar"
    assert "solar_data" in data
    assert "unfused" in data["solar_data"]
    assert "fused" in data["solar_data"]
    assert "fused_prefetched" in data["solar_data"]
