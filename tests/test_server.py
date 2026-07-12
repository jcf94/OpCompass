"""First-party API contract tests."""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from opcompass.server import api_analyze, api_list_operators, app


client = TestClient(app, raise_server_exceptions=False)


def test_operator_api_exposes_typed_parameter_specs():
    operators = {item["name"]: item for item in api_list_operators()}
    matmul = operators["matmul"]

    assert [item["name"] for item in matmul["parameter_spec"]] == ["M", "N", "K"]
    assert all(item["type"] == "int" for item in matmul["parameter_spec"])
    assert all(item["required"] is True for item in matmul["parameter_spec"])
    assert all(item["minimum"] == 1 for item in matmul["parameter_spec"])
    assert matmul["capabilities"] == {
        "hierarchy_roofline": "formula",
        "pipeline": "pipeline",
        "solar": "formula",
    }
    assert operators["reduction"]["capabilities"] == {
        "hierarchy_roofline": "formula",
        "pipeline": "unsupported",
        "solar": "unsupported",
    }


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
    assert result["schema_version"] == "0.5.0"
    assert result["implementation_version"] == "0.5.0.dev0"
    assert result["implementation_revision"]
    assert result["hardware_spec_version"] == "nvidia-a100-v1"
    assert result["evidence"] == {
        "coverage": "formula",
        "sources": ["operator_formula", "hardware_theoretical_peaks"],
    }
    assert result["uncertainty"]["status"] == "unquantified"
    assert result["uncertainty"]["lower_time_us"] is None
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


def test_analyze_api_rejects_unsupported_solar_operator_before_backend():
    with pytest.raises(HTTPException) as exc_info:
        api_analyze({
            "operator": "reduction",
            "hardware": "a100",
            "mode": "solar",
            "dims": {"N": 4096, "D": 256},
        })

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail["code"] == "unsupported_analysis_mode"
    assert exc_info.value.detail["requested_mode"] == "solar"


def test_analyze_api_returns_stable_optional_backend_error(monkeypatch):
    from opcompass.models import BackendUnavailableError
    import opcompass.engine.solar_analyzer as solar_analyzer

    def unavailable():
        raise BackendUnavailableError("solar", ["torchview"])

    monkeypatch.setattr(solar_analyzer, "_check_solar_dependencies", unavailable)
    with pytest.raises(HTTPException) as exc_info:
        api_analyze({
            "operator": "matmul",
            "hardware": "a100",
            "mode": "solar",
            "dims": {"M": 128, "N": 128, "K": 128},
        })

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {
        "code": "optional_backend_unavailable",
        "backend": "solar",
        "missing_dependencies": ["torchview"],
        "message": "solar requires additional dependencies: torchview",
    }


def test_analyze_api_returns_stable_infeasible_candidate_error():
    with pytest.raises(HTTPException) as exc_info:
        api_analyze({
            "operator": "matmul",
            "hardware": "a100",
            "mode": "pipeline",
            "dims": {"M": 128, "N": 128, "K": 128},
            "pipeline_config": {"block_m": 63, "block_n": 64, "block_k": 16},
        })

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail["code"] == "infeasible_pipeline_candidate"
    assert "multiple of 16" in exc_info.value.detail["message"]


def test_analyze_api_returns_stable_unsupported_dtype_error():
    with pytest.raises(HTTPException) as exc_info:
        api_analyze({
            "operator": "matmul",
            "hardware": "a100",
            "dtype": "fp8",
            "dims": {"M": 128, "N": 128, "K": 128},
        })

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail["code"] == "unsupported_dtype"
    assert exc_info.value.detail["hardware"] == "a100"
    assert exc_info.value.detail["dtype"] == "fp8"
    assert "fp16" in exc_info.value.detail["supported_dtypes"]


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


def test_http_analyze_success_uses_typed_json_contract():
    response = client.post("/api/analyze", json={
        "operator": "matmul",
        "hardware": "a100",
        "dtype": "fp16",
        "dims": {"M": 128, "N": 128, "K": 128},
    })

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()
    assert payload["schema_version"] == "0.5.0"
    assert payload["requested_mode"] == "hierarchy_roofline"
    assert payload["roofline_data"]["peak_flops"] > 0


def test_http_analyze_rejects_invalid_pydantic_request():
    response = client.post("/api/analyze", json={
        "operator": "matmul",
        "hardware": "a100",
        "dims": {"M": "128", "N": 128, "K": 128},
        "unexpected": True,
    })

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["detail"]["code"] == "invalid_api_request"


def test_http_analyze_exposes_fallback_and_strict_rejection():
    body = {
        "operator": "reduction",
        "hardware": "a100",
        "mode": "pipeline",
        "dims": {"N": 4096, "D": 256},
    }
    fallback = client.post("/api/analyze", json=body)
    rejected = client.post("/api/analyze", json={**body, "strict": True})

    assert fallback.status_code == 200
    assert fallback.json()["fallback"]["reason_code"] == "pipeline_model_unavailable"
    assert rejected.status_code == 422
    assert rejected.json()["detail"]["code"] == "unsupported_analysis_mode"


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({"operator": "missing", "hardware": "a100", "dims": {}}, 404, "unknown_operator"),
        ({"operator": "matmul", "hardware": "missing", "dims": {}}, 404, "unknown_hardware"),
        ({
            "operator": "matmul", "hardware": "a100", "dtype": "fp8",
            "dims": {"M": 128, "N": 128, "K": 128},
        }, 422, "unsupported_dtype"),
        ({
            "operator": "matmul", "hardware": "a100", "mode": "pipeline",
            "dims": {"M": 128, "N": 128, "K": 128},
            "pipeline_config": {"block_m": 63, "block_n": 64, "block_k": 16},
        }, 422, "infeasible_pipeline_candidate"),
    ],
)
def test_http_analyze_returns_stable_domain_errors(body, status, code):
    response = client.post("/api/analyze", json=body)

    assert response.status_code == status
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["detail"]["code"] == code


def test_http_analyze_returns_optional_backend_error(monkeypatch):
    from opcompass.models import BackendUnavailableError
    import opcompass.engine.solar_analyzer as solar_analyzer

    def unavailable():
        raise BackendUnavailableError("solar", ["torchview"])

    monkeypatch.setattr(solar_analyzer, "_check_solar_dependencies", unavailable)
    response = client.post("/api/analyze", json={
        "operator": "matmul", "hardware": "a100", "mode": "solar",
        "dims": {"M": 128, "N": 128, "K": 128},
    })

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "optional_backend_unavailable"


def test_http_analyze_trace_is_limited():
    response = client.post("/api/analyze", json={
        "operator": "matmul", "hardware": "a100", "mode": "pipeline",
        "dims": {"M": 256, "N": 256, "K": 256},
        "include_trace": True, "trace_limit": 2,
    })

    assert response.status_code == 200
    schedule = response.json()["pipeline_schedule"]
    assert len(schedule["sub_ops"]) == 2
    assert schedule["trace"]["returned_sub_ops"] == 2
    assert schedule["trace"]["complete"] is False


def test_http_analyze_wraps_uncaught_internal_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("sensitive implementation detail")

    monkeypatch.setattr("opcompass.server.Analyzer.analyze", fail)
    response = client.post("/api/analyze", json={
        "operator": "matmul", "hardware": "a100",
        "dims": {"M": 128, "N": 128, "K": 128},
    })

    assert response.status_code == 500
    assert response.json() == {"detail": {
        "code": "internal_error",
        "message": "An unexpected internal error occurred.",
    }}


def test_openapi_documents_structured_analyze_errors():
    operation = app.openapi()["paths"]["/api/analyze"]["post"]

    for status in ("400", "404", "422", "500", "503"):
        schema = operation["responses"][status]["content"]["application/json"]["schema"]
        assert schema["$ref"].endswith("ErrorResponse")
