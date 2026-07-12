"""Packaging metadata and runtime resource smoke tests."""

from pathlib import Path

import opcompass
from opcompass.engine.solar_analyzer import HARDWARE_TO_SOLAR_ARCH
from opcompass.server import WEB_DIR, app


def test_runtime_versions_are_consistent():
    assert opcompass.__version__ == "0.3.0.dev0"
    assert app.version == opcompass.__version__


def test_implementation_revision_can_be_injected(monkeypatch):
    monkeypatch.setenv("OPCOMPASS_GIT_REVISION", "release-build-revision")
    assert opcompass.implementation_revision() == "release-build-revision"


def test_runtime_resource_paths_exist():
    assert (Path(WEB_DIR) / "index.html").is_file()
    assert (Path(WEB_DIR) / "js" / "result_contract.js").is_file()
    assert HARDWARE_TO_SOLAR_ARCH
    assert all(Path(path).is_file() for path in HARDWARE_TO_SOLAR_ARCH.values())
