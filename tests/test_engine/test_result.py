"""Serialization and human-readable result contract tests."""

import csv
import io
import json
import re

import pytest

from opcompass.engine.analyzer import Analyzer
from opcompass.engine.result import format_result
from opcompass.models import AnalysisMode, DataType
from opcompass.registry import get_hardware, get_operator


def _pipeline_result():
    return Analyzer().analyze(
        get_operator("matmul")(), get_hardware("a100")(), DataType.FP16,
        mode=AnalysisMode.PIPELINE, M=256, N=256, K=256,
    )


def test_csv_has_stable_flat_schema_and_standard_quoting():
    result = _pipeline_result()
    rows = list(csv.DictReader(io.StringIO(format_result(result, fmt="csv"))))

    assert len(rows) == 1
    assert list(rows[0]) == [
        "schema_version", "operator", "hardware", "shapes_json", "dtype",
        "requested_mode", "executed_mode", "estimate_kind", "support_level",
        "model_id", "fallback_reason_code", "total_flops", "total_read_bytes",
        "total_write_bytes", "sol_time_us", "sol_tflops", "bottleneck",
    ]
    assert json.loads(rows[0]["shapes_json"]) == {"M": 256, "N": 256, "K": 256}
    assert ',"{""M"":256,""N"":256,""K"":256}",' in format_result(result, fmt="csv")


def test_json_rejects_non_finite_numbers():
    result = _pipeline_result()
    result.sol_tflops = float("nan")

    with pytest.raises(ValueError, match="JSON compliant"):
        format_result(result, fmt="json")


def test_pipeline_table_prints_real_phase_times():
    result = _pipeline_result()
    table = format_result(result, fmt="table")
    match = re.search(r"Prologue\s+[\d,]+\s+([\d.]+) µs", table)

    assert match is not None
    assert float(match.group(1)) == pytest.approx(
        result.pipeline_schedule.prologue_cycles
        / result.compute_unit_clock_hz * 1e6,
        abs=0.0005,
    )
    assert float(match.group(1)) > 0
    assert re.search(r"Epilogue\s+[\d,]+\s+[\d.]+ µs", table)
    assert "Build      : 0.4.0.dev0 @" in table
    assert "HW spec    : legacy-v1" in table
