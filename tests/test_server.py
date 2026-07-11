"""First-party API contract tests."""

import pytest
from fastapi import HTTPException

from opcompass.server import api_analyze, api_list_operators


def test_operator_api_exposes_typed_parameter_specs():
    operators = {item["name"]: item for item in api_list_operators()}
    matmul = operators["matmul"]

    assert [item["name"] for item in matmul["parameter_spec"]] == ["M", "N", "K"]
    assert all(item["type"] == "int" for item in matmul["parameter_spec"])
    assert all(item["required"] is True for item in matmul["parameter_spec"])
    assert all(item["minimum"] == 1 for item in matmul["parameter_spec"])


def test_analyze_api_returns_stable_validation_error():
    with pytest.raises(HTTPException) as exc_info:
        api_analyze({
            "operator": "matmul",
            "hardware": "a100",
            "dtype": "fp16",
            "dims": {"M": 128, "N": 128},
        })

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail["code"] == "invalid_operator_parameters"
    assert exc_info.value.detail["issues"] == [{
        "parameter": "K",
        "reason": "missing",
        "message": "missing required parameter 'K'",
    }]


def test_analyze_api_serializes_explicit_fallback_contract():
    result = api_analyze({
        "operator": "reduction",
        "hardware": "a100",
        "dtype": "fp16",
        "mode": "pipeline",
        "dims": {"N": 4096, "D": 256},
    })

    assert result["requested_mode"] == "pipeline"
    assert result["executed_mode"] == "hierarchy_roofline"
    assert result["estimate_kind"] == "theoretical_bound"
    assert result["support_level"] == "formula"
    assert result["schema_version"] == "0.2.0"
    assert result["fallback"]["reason_code"] == "pipeline_model_unavailable"


def test_analyze_api_strict_mode_returns_stable_unsupported_error():
    with pytest.raises(HTTPException) as exc_info:
        api_analyze({
            "operator": "reduction",
            "hardware": "a100",
            "dtype": "fp16",
            "mode": "pipeline",
            "strict": True,
            "dims": {"N": 4096, "D": 256},
        })

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail["code"] == "unsupported_analysis_mode"
    assert exc_info.value.detail["requested_mode"] == "pipeline"
