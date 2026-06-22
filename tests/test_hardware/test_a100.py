"""Test A100 hardware definition."""

from opcompass.hardware.nvidia_ampere import NvidiaA100
from opcompass.models import DataType


def test_a100_peak_flops():
    hw = NvidiaA100()
    assert hw.get_peak_flops(DataType.FP16) == 312e12
    assert hw.get_peak_flops(DataType.FP32) == 19.5e12
    assert hw.get_peak_flops(DataType.BF16) == 312e12


def test_a100_memory():
    hw = NvidiaA100()
    assert hw.hbm_bandwidth == 2.0e12
    assert hw.num_compute_units == 108


def test_a100_clock():
    hw = NvidiaA100()
    assert hw.compute_unit.clock_mhz == 1410
    assert hw.clock_ghz == 1.41
