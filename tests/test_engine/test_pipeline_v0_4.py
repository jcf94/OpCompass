"""V0.4 memory-path, synchronization, and launch-policy tests."""

import pytest

from opcompass.engine.pipeline_ir import (
    Launch, MemoryAccess, Node, PipelineIRValidationError, PipelineProgram,
    Resource, ResourceDemand, ResourceKind, SyncPrimitive, Work, schedule,
)
from opcompass.models import DataType, PipelineConfig
from opcompass.registry import get_hardware, get_operator


def matmul_program(hardware, **dims):
    return get_operator("matmul")().get_pipeline_program(
        get_hardware(hardware)(), DataType.FP16,
        PipelineConfig(block_m=128, block_n=128, block_k=32,
                       stage_count=2, warp_count=4),
        **dims,
    )


@pytest.mark.parametrize("hardware,expected", [
    ("a100", {SyncPrimitive.CP_ASYNC_COMMIT, SyncPrimitive.CP_ASYNC_WAIT,
              SyncPrimitive.SYNCTHREADS}),
    ("h100", {SyncPrimitive.MBARRIER, SyncPrimitive.WGMMA_COMMIT,
              SyncPrimitive.WGMMA_WAIT}),
    ("b200", {SyncPrimitive.MBARRIER, SyncPrimitive.UMMA_COMMIT,
              SyncPrimitive.UMMA_WAIT}),
])
def test_architectures_emit_distinct_explicit_synchronization(hardware, expected):
    program = matmul_program(hardware, M=256, N=256, K=128)
    assert {node.synchronization for node in program.nodes if node.synchronization} == expected
    assert any(resource.kind == ResourceKind.SYNCHRONIZATION for resource in program.resources)


def test_memory_accesses_declare_paths_transactions_and_reuse():
    program = matmul_program("a100", M=256, N=256, K=128)
    accesses = [node.memory_access for node in program.nodes if node.memory_access]
    assert accesses
    assert any(access.path[:2] == (ResourceKind.HBM, ResourceKind.L2) for access in accesses)
    assert all(access.transactions * access.transaction_bytes >= access.bytes for access in accesses)
    assert {access.reuse_policy for access in accesses} >= {"cta_order", "resident"}


def test_tail_and_small_grid_underfill_are_visible():
    program = matmul_program("a100", M=129, N=65, K=33)
    result = schedule(program)
    assert result.tail_fraction == pytest.approx((129 * 65) / (256 * 128))
    assert result.underfilled is True
    assert result.wave_count == 1
    assert result.loop_iterations == 2


def test_launch_waves_and_overhead_are_operational():
    node = Node("work", 5, Work(1, "operations"), (ResourceDemand("alu"),))
    program = PipelineProgram(
        (Resource("alu", kind=ResourceKind.COMPUTE),), (node,),
        launch=Launch(grid_size=9, resident_blocks=2, compute_units=2,
                      launch_overhead_cycles=7),
    )
    result = schedule(program)
    assert result.wave_count == 3
    assert result.total_cycles == 22


def test_invalid_transaction_coverage_is_rejected():
    node = Node(
        "load", 1, Work(65, "bytes"), (ResourceDemand("hbm"),),
        memory_access=MemoryAccess((ResourceKind.HBM,), 65, 32, 2),
    )
    with pytest.raises(PipelineIRValidationError, match="do not cover"):
        schedule(PipelineProgram((Resource("hbm", kind=ResourceKind.HBM),), (node,)))
