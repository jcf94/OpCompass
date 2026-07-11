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


def test_analyze_api_pipeline_trace_is_opt_in_and_bounded():
    body = {
        "operator": "matmul",
        "hardware": "a100",
        "dtype": "fp16",
        "mode": "pipeline",
        "dims": {"M": 256, "N": 256, "K": 256},
    }
    compact = api_analyze(body)
    traced = api_analyze({**body, "include_trace": True, "trace_limit": 2})

    assert "sub_ops" not in compact["pipeline_schedule"]
    assert compact["pipeline_schedule"]["trace"]["included"] is False
    assert len(traced["pipeline_schedule"]["sub_ops"]) == 2
    assert traced["pipeline_schedule"]["trace"]["returned_sub_ops"] == 2


def test_analyze_api_rejects_unbounded_trace_limit():
    with pytest.raises(HTTPException) as exc_info:
        api_analyze({
            "operator": "matmul",
            "hardware": "a100",
            "mode": "pipeline",
            "include_trace": True,
            "trace_limit": 5001,
            "dims": {"M": 256, "N": 256, "K": 256},
        })

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail["code"] == "invalid_api_request"
    assert exc_info.value.detail["issues"][0]["loc"] == ("trace_limit",)


def test_openapi_exposes_typed_analyze_contract():
    from opcompass.server import app

    schema = app.openapi()
    operation = schema["paths"]["/api/analyze"]["post"]

    assert operation["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith("AnalyzeRequest")
    assert operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("AnalyzeResponse")


def test_http_api_wraps_pydantic_errors_with_stable_code():
    import asyncio
    import json

    from fastapi.exceptions import RequestValidationError
    from pydantic import ValidationError

    from opcompass.api_models import AnalyzeRequest
    from opcompass.server import api_request_validation_error

    with pytest.raises(ValidationError) as model_error:
        AnalyzeRequest.model_validate({
            "operator": "matmul",
            "hardware": "a100",
            "dims": {"M": "128", "N": 128, "K": 128},
            "unexpected": True,
        })

    exc = RequestValidationError(model_error.value.errors(include_url=False))
    response = asyncio.run(api_request_validation_error(None, exc))
    payload = json.loads(response.body)
    assert response.status_code == 422
    assert payload["detail"]["code"] == "invalid_api_request"
    locations = {tuple(issue["loc"]) for issue in payload["detail"]["issues"]}
    assert ("dims", "M") in locations
    assert ("unexpected",) in locations
