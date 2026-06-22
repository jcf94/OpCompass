"""Test matmul operator FLOPs and I/O calculations."""

import pytest
from opcompass.operators.matmul import Matmul
from opcompass.models import DataType


def test_matmul_flops():
    op = Matmul()
    # 4096^3 matmul: 2 * 4096^3 = 2 * 68,719,476,736 = 137,438,953,472
    flops = op.compute_flops(M=4096, N=4096, K=4096)
    assert flops == 2 * 4096 * 4096 * 4096
    assert flops == 137_438_953_472


def test_matmul_io_bytes_fp16():
    op = Matmul()
    read_b, write_b = op.compute_io_bytes(DataType.FP16, M=4096, N=4096, K=4096)
    # read: (M*K + K*N) * 2 = (4096^2 + 4096^2) * 2 = 67,108,864
    # write: M*N * 2 = 4096^2 * 2 = 33,554,432
    assert read_b == 67_108_864
    assert write_b == 33_554_432


def test_matmul_small():
    op = Matmul()
    flops = op.compute_flops(M=16, N=32, K=8)
    assert flops == 2 * 16 * 32 * 8  # 8192


def test_matmul_zero_dim():
    op = Matmul()
    flops = op.compute_flops(M=0, N=1024, K=1024)
    assert flops == 0
