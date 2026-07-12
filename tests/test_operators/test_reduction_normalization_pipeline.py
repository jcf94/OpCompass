import pytest

from opcompass.engine.analyzer import Analyzer
from opcompass.engine.pipeline_ir import ResourceKind, schedule
from opcompass.models import AnalysisMode, DataType, OperatorValidationError
from opcompass.registry import get_hardware, get_operator


def test_reduction_selects_warp_block_and_two_pass_strategies():
    op = get_operator("reduction")()
    assert op.select_algorithm(32) == "warp"
    assert op.select_algorithm(256) == "block"
    assert op.select_algorithm(8192) == "two_pass"
    two_pass = op.get_ops_breakdown(DataType.FP16, None, None, N=8192, D=8192,
                                    algorithm="two_pass")
    assert {item.name for item in two_pass} >= {"write_partials", "read_partials"}


def test_reduction_pipeline_is_real_and_strict_mode_does_not_fallback():
    result = Analyzer().analyze(
        get_operator("reduction")(), get_hardware("a100")(), DataType.FP16,
        mode=AnalysisMode.PIPELINE, strict=True, N=4096, D=256,
    )
    assert result.executed_mode == AnalysisMode.PIPELINE
    assert result.fallback is None
    assert result.tiling_info.candidate_name == "block"
    assert result.pipeline_ir_schedule is not None


def test_reduction_algorithm_validation():
    with pytest.raises(OperatorValidationError, match="auto, warp, block"):
        get_operator("reduction")().validate_dimensions(
            {"N": 1024, "D": 32, "algorithm": "atomic"}
        )


@pytest.mark.parametrize("variant,flops", [("layernorm", 6 * 8 * 128),
                                             ("rmsnorm", 5 * 8 * 128)])
def test_normalization_variants_have_explicit_accounting(variant, flops):
    op = get_operator("layernorm")()
    assert op.compute_flops(N=8, D=128, variant=variant) == flops
    read_bytes, write_bytes = op.compute_io_bytes(DataType.FP16, N=8, D=128,
                                                   variant=variant)
    assert read_bytes > 8 * 128 * 2
    assert write_bytes == 8 * 128 * 2


def test_layernorm_two_pass_exposes_extra_traffic_and_generic_resources():
    op = get_operator("layernorm")()
    hw = get_hardware("h100")()
    ops = op.get_ops_breakdown(DataType.FP16, hw, None, N=16, D=16384,
                               variant="layernorm", algorithm="two_pass")
    assert {item.name for item in ops} >= {"barrier", "rsqrt", "affine_convert",
                                           "partial_statistics", "reload_statistics"}
    program = op.get_pipeline_program(hw, DataType.FP16, N=16, D=16384,
                                      variant="layernorm", algorithm="two_pass")
    assert {resource.kind for resource in program.resources} >= {
        ResourceKind.HBM, ResourceKind.SHARED, ResourceKind.COMPUTE, ResourceKind.STORE,
    }
    result = schedule(program)
    by_name = {entry.node: entry for entry in result.entries}
    assert by_name["rsqrt"].start_cycle >= by_name["barrier"].end_cycle
    assert by_name["reload_statistics"].start_cycle >= by_name["partial_statistics"].end_cycle


def test_layernorm_pipeline_api_contract_executes_pipeline():
    result = Analyzer().analyze(
        get_operator("layernorm")(), get_hardware("a100")(), DataType.FP16,
        mode=AnalysisMode.PIPELINE, N=32, D=768,
    )
    assert result.executed_mode == AnalysisMode.PIPELINE
    assert result.tiling_info.candidate_name == "one_pass"
    assert result.pipeline_ir_schedule.total_cycles > 0
