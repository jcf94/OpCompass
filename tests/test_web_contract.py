"""Pure-JavaScript tests for Web result-contract presentation semantics."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


NODE = shutil.which("node")
SCRIPT = Path(__file__).parents[1] / "web" / "js" / "result_contract.js"


def _build(payload):
    if NODE is None:
        pytest.skip("node is not installed")
    program = (
        f"const C=require({json.dumps(str(SCRIPT))});"
        f"process.stdout.write(JSON.stringify(C.build({json.dumps(payload)})));"
    )
    completed = subprocess.run(
        [NODE, "-e", program], check=True, capture_output=True, text=True
    )
    return json.loads(completed.stdout)


def test_result_contract_distinguishes_requested_and_executed_modes():
    view = _build({
        "mode": "pipeline",
        "requested_mode": "pipeline",
        "executed_mode": "hierarchy_roofline",
        "fallback": {"message": "Pipeline unavailable; roofline executed."},
        "estimate_kind": "theoretical_bound",
        "support_level": "formula",
        "model_id": "hierarchy_roofline_v1",
        "implementation_version": "0.2.0.dev0",
        "implementation_revision": "1234567890abcdef",
        "hardware_spec_version": "legacy-v1",
        "evidence": {"coverage": "formula", "sources": ["operator_formula"]},
        "uncertainty": {"status": "unquantified", "reason": "No measurements."},
    })

    assert view["route"] == "pipeline → hierarchy_roofline"
    assert view["fallback"] is True
    assert view["status"] == "Fallback executed"
    assert view["message"] == "Pipeline unavailable; roofline executed."
    assert view["build"] == "0.2.0.dev0 @ 1234567890ab"
    assert view["evidenceSources"] == "operator_formula"


def test_result_contract_marks_matching_execution_without_fallback():
    view = _build({
        "mode": "pipeline",
        "requested_mode": "pipeline",
        "executed_mode": "pipeline",
        "estimate_kind": "analytical_model",
        "support_level": "pipeline",
    })

    assert view["route"] == "pipeline"
    assert view["fallback"] is False
    assert view["status"] == "Executed as requested"
