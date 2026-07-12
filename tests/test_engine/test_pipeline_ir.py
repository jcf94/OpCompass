"""V0.3 explicit pipeline IR and generic scheduler contract tests."""

import pytest

from opcompass.engine.pipeline_ir import (
    Buffer, BufferAccess, Edge, Loop, Node, PipelineIRValidationError,
    PipelineProgram, Resource, ResourceDemand, Work, schedule, validate_program,
)
from opcompass.engine.analyzer import Analyzer
from opcompass.models import AnalysisMode, DataType, PipelineConfig
from opcompass.registry import get_hardware, get_operator


def demand(name, units=1):
    return (ResourceDemand(name, units),)


def synthetic_program(capacity=1):
    return PipelineProgram(
        resources=(Resource("copy", capacity), Resource("compute")),
        nodes=(
            Node("left", 4, Work(64, "bytes"), demand("copy")),
            Node("right", 4, Work(64, "bytes"), demand("copy")),
            Node("join", 3, Work(32, "fma"), demand("compute")),
        ),
        edges=(Edge("left", "join"), Edge("right", "join")),
    )


def test_synthetic_non_matmul_dag_obeys_edges_and_resources():
    result = schedule(synthetic_program())
    entries = {entry.node: entry for entry in result.entries}
    assert entries["left"].end_cycle <= entries["right"].start_cycle
    assert entries["join"].start_cycle >= max(
        entries["left"].end_cycle, entries["right"].end_cycle
    )
    assert result.total_cycles == 11


def test_resource_capacity_and_edge_changes_are_operational():
    serial = schedule(synthetic_program(capacity=1))
    parallel = schedule(synthetic_program(capacity=2))
    assert parallel.total_cycles < serial.total_cycles

    program = synthetic_program(capacity=2)
    delayed = PipelineProgram(
        program.resources, program.nodes,
        (Edge("left", "join", latency_cycles=7), Edge("right", "join")),
    )
    assert schedule(delayed).total_cycles == parallel.total_cycles + 7


def test_loop_carried_edge_and_initiation_interval():
    program = PipelineProgram(
        resources=(Resource("alu"),),
        nodes=(Node("update", 3, Work(1, "operations"), demand("alu")),),
        edges=(Edge("update", "update", iteration_distance=1),),
        loop=Loop(iterations=4, initiation_interval=1),
    )
    result = schedule(program)
    assert [entry.start_cycle for entry in result.entries] == [0, 3, 6, 9]


@pytest.mark.parametrize("case", ["missing", "cycle", "units", "capacity", "ring"])
def test_validation_failures_are_typed(case):
    resource = Resource("r")
    node = Node("a", 1, Work(1, "operations"), demand("r"))
    if case == "missing":
        program = PipelineProgram((resource,), (node,), (Edge("a", "gone"),))
    elif case == "cycle":
        other = Node("b", 1, Work(1, "operations"), demand("r"))
        program = PipelineProgram((resource,), (node, other), (Edge("a", "b"), Edge("b", "a")))
    elif case == "units":
        program = PipelineProgram((resource,), (Node("a", 1, Work(1, "widgets"), demand("r")),))
    elif case == "capacity":
        program = PipelineProgram((resource,), (Node("a", 1, Work(1, "operations"), demand("r", 2)),))
    else:
        buffer = Buffer("ring", 16, slots=2)
        writer = Node("a", 1, Work(1, "bytes"), demand("r"), (BufferAccess("ring", "write"),))
        reader = Node("b", 1, Work(1, "bytes"), demand("r"), (BufferAccess("ring", "read"),))
        program = PipelineProgram((resource,), (writer, reader), (Edge("a", "b", iteration_distance=2),), (buffer,))
    with pytest.raises(PipelineIRValidationError):
        validate_program(program)


def test_compact_schedule_reconstructs_trace_windows():
    result = schedule(synthetic_program())
    assert [entry.node for entry in result.trace_window(4, 8)] == ["right"]
    assert result.trace_window(result.total_cycles, result.total_cycles + 1) == ()


def test_scheduler_is_deterministic_and_never_overcommits():
    for capacity in (1, 2, 3):
        program = synthetic_program(capacity)
        first = schedule(program)
        assert first == schedule(program)
        for cycle in range(first.total_cycles):
            used = sum(
                demand.units
                for entry in first.entries if entry.start_cycle <= cycle < entry.end_cycle
                for demand in entry.resources if demand.resource == "copy"
            )
            assert used <= capacity


def test_matmul_analysis_exposes_ir_and_legacy_comparison():
    result = Analyzer().analyze(
        get_operator("matmul")(), get_hardware("a100")(), DataType.FP16,
        mode=AnalysisMode.PIPELINE,
        pipeline_config=PipelineConfig(
            block_m=128, block_n=128, block_k=32, stage_count=2, warp_count=4
        ),
        M=256, N=256, K=128,
    )
    assert result.model_id == "legacy_matmul_v1"
    assert result.pipeline_ir_schedule.loop_iterations == 4
    assert result.pipeline_ir_schedule.total_cycles > 0
    assert result.pipeline_legacy_comparison["legacy_cycles_per_block"] > 0
    assert result.pipeline_legacy_comparison["ir_cycles_per_block"] > 0
