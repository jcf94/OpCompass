"""Compact numerical and semantic baselines for the v0.2 contract."""

import json
from pathlib import Path

import pytest

from opcompass.engine.analyzer import Analyzer
from opcompass.engine.result import _result_to_dict
from opcompass.models import AnalysisMode, DataType
from opcompass.registry import get_hardware, get_operator


GOLDEN = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "v0_2_compact_golden.json").read_text()
)


@pytest.mark.parametrize("hardware_name", ["a100", "h100", "b200"])
def test_matmul_pipeline_compact_golden(hardware_name):
    result = Analyzer().analyze(
        get_operator("matmul")(), get_hardware(hardware_name)(), DataType.FP16,
        mode=AnalysisMode.PIPELINE, M=256, N=256, K=256,
    )
    serialized = _result_to_dict(result)
    expected = GOLDEN["matmul_pipeline"][hardware_name]
    tiling = serialized["tiling_info"]
    schedule = serialized["pipeline_schedule"]

    assert serialized["requested_mode"] == "pipeline"
    assert serialized["executed_mode"] == "pipeline"
    assert serialized["estimate_kind"] == "analytical_model"
    assert serialized["support_level"] == "pipeline"
    assert serialized["model_id"] == "legacy_matmul_v1"
    assert serialized["implementation_version"] == "0.4.0.dev0"
    assert serialized["implementation_revision"]
    assert serialized["hardware_spec_version"] == "legacy-v1"
    assert serialized["evidence"]["coverage"] == "analytical_model"
    assert serialized["uncertainty"]["status"] == "unquantified"
    assert serialized["total_flops"] == 33_554_432
    assert serialized["total_read_bytes"] == 262_144
    assert serialized["total_write_bytes"] == 131_072
    assert serialized["sol_time_us"] == pytest.approx(expected["sol_time_us"])
    assert serialized["sol_tflops"] == pytest.approx(expected["sol_tflops"])
    assert serialized["bottleneck"] == expected["bottleneck"]
    assert tiling["candidate_name"] == expected["candidate_name"]
    assert [tiling["block_m"], tiling["block_n"], tiling["block_k"]] == expected["tile"]
    for field in (
        "total_cycles_per_block", "num_k_iterations", "prologue_cycles",
        "per_iteration_cycles", "epilogue_cycles",
    ):
        assert schedule[field] == expected[field]
    assert "sub_ops" not in schedule


def test_pipeline_fallback_compact_golden():
    result = Analyzer().analyze(
        get_operator("reduction")(), get_hardware("a100")(), DataType.FP16,
        mode=AnalysisMode.PIPELINE, N=4096, D=256,
    )
    serialized = _result_to_dict(result)
    expected = GOLDEN["pipeline_fallback"]

    for field in (
        "operator", "hardware", "shapes", "requested_mode", "executed_mode",
        "estimate_kind", "support_level", "model_id", "total_flops",
        "total_read_bytes", "total_write_bytes", "bottleneck",
    ):
        assert serialized[field] == expected[field]
    assert serialized["fallback"]["reason_code"] == expected["fallback_reason_code"]
    assert serialized["sol_time_us"] == pytest.approx(expected["sol_time_us"])
    assert serialized["sol_tflops"] == pytest.approx(expected["sol_tflops"])
