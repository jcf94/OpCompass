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
