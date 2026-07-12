import json

import pytest

from opcompass.calibration import (
    MeasurementRecord, evaluate_accuracy, fit_scale_overlay,
    import_measurements, load_measurements, save_measurements,
)
from opcompass.registry import get_hardware
from opcompass.server import api_get_hardware


def record():
    return MeasurementRecord(
        "a100-matmul-1", "matmul", "a100", "fp16",
        {"M": 256, "N": 256, "K": 128}, (7.1, 7.0, 7.2),
        "cutlass_tensorop", "0.5.0.dev0",
        environment={"cuda": "12.8", "driver": "570"},
        clocks_mhz={"sm": 1410, "memory": 1215},
        candidate={"tile": "128x128x32"}, counters={"dram_bytes": 262144},
        raw_source="raw/cutlass-a100.csv",
    )


def test_measurement_json_round_trip_preserves_source_and_identity(tmp_path):
    path = tmp_path / "records.json"
    save_measurements([record()], path)
    assert load_measurements(path) == [record()]
    assert json.loads(path.read_text())[0]["raw_source"] == "raw/cutlass-a100.csv"


def test_generic_and_cutlass_csv_import(tmp_path):
    generic = tmp_path / "generic.csv"
    generic.write_text("record_id,operator,hardware,dtype,M,N,K,runtime_us,kernel,model_version\n1,matmul,a100,fp16,128,256,64,3.5,k,model\n")
    assert import_measurements(generic)[0].median_runtime_us == 3.5

    cutlass = tmp_path / "cutlass.csv"
    cutlass.write_text("Operation,Device,A,M,N,K,Runtime\ngemm,a100,fp16,128,128,32,1.25\n")
    imported = import_measurements(cutlass, "cutlass")[0]
    assert imported.kernel == "gemm"
    assert imported.shapes == {"M": 128, "N": 128, "K": 32}


def test_accuracy_report_raw_metrics_groups_and_ranking():
    rows = [
        {"record_id": "1", "predicted_us": 8, "measured_us": 10, "hardware": "a100", "dtype": "fp16", "shape_regime": "square", "bottleneck": "compute", "predicted_rank": 1, "measured_rank": 1},
        {"record_id": "2", "predicted_us": 12, "measured_us": 10, "hardware": "h100", "dtype": "fp16", "shape_regime": "skinny", "bottleneck": "memory", "predicted_rank": 2, "measured_rank": 1},
    ]
    report = evaluate_accuracy(rows)
    assert report.median_ape == pytest.approx(.2)
    assert report.p90_ape == pytest.approx(.2)
    assert report.bias == 0
    assert report.top_k_recall == .5
    assert report.groups["hardware:a100"]["count"] == 1

    overlay = fit_scale_overlay(rows, "scale-a", "model-v1")
    assert overlay.training_record_ids == ("1", "2")
    assert overlay.apply(8) > 0


@pytest.mark.parametrize("name", ["a100", "h100", "h100_pcie", "b200"])
def test_primary_hardware_targets_expose_versioned_provenance(name):
    hardware = get_hardware(name)()
    assert hardware.spec_version != "legacy-v1"
    assert hardware.provenance_status == "audited"
    assert hardware.provenance()
    payload = api_get_hardware(name)
    assert payload["spec_version"] == hardware.spec_version
    assert payload["provenance"][0]["verified_on"]
