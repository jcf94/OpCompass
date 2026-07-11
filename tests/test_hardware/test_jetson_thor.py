"""Test NVIDIA Jetson Thor hardware definitions."""

import pytest

from opcompass.hardware.nvidia_blackwell import NvidiaJetsonT4000, NvidiaJetsonT5000
from opcompass.models import DataType
from opcompass.registry import discover_hardware


def test_jetson_thor_targets_are_registered():
    targets = discover_hardware()
    assert targets["jetson-t5000"] is NvidiaJetsonT5000
    assert targets["jetson-t4000"] is NvidiaJetsonT4000


def test_jetson_t5000_specs():
    hw = NvidiaJetsonT5000()
    assert hw.num_compute_units == 20
    assert hw.compute_unit.clock_mhz == 1570
    assert hw.hbm_bandwidth == 273e9
    assert hw.memory.tiers[0].capacity_bytes == 128 * 1024**3
    assert hw.memory.tiers[1].name == "L2"
    assert hw.memory.tiers[1].capacity_bytes == 12 * 1024**2
    assert hw.memory.tiers[1].bandwidth_bytes_per_sec == pytest.approx(400.35e9)
    assert hw.get_peak_flops(DataType.FP4) == pytest.approx(1035e12)
    assert hw.get_peak_flops(DataType.FP8) == pytest.approx(517.5e12)
    assert hw.get_peak_flops(DataType.FP16) == pytest.approx(258.75e12)


def test_jetson_t4000_specs():
    hw = NvidiaJetsonT4000()
    assert hw.num_compute_units == 12
    assert hw.compute_unit.clock_mhz == 1530
    assert hw.hbm_bandwidth == 273e9
    assert hw.memory.tiers[0].capacity_bytes == 64 * 1024**3
    assert hw.memory.tiers[1].name == "L2"
    assert hw.memory.tiers[1].capacity_bytes == round(12 * 1024**2 * 12 / 20)
    assert hw.memory.tiers[1].bandwidth_bytes_per_sec == pytest.approx(234.09e9)
    assert hw.get_peak_flops(DataType.FP4) == pytest.approx(600e12)
    assert hw.get_peak_flops(DataType.FP8) == pytest.approx(300e12)
    assert hw.get_peak_flops(DataType.FP16) == pytest.approx(150e12)
