# Repository Guidelines

## Project Structure & Module Organization

`opcompass/` contains the Python package. Core data models live in `models.py`, dynamic discovery in `registry.py`, the Click CLI in `cli.py`, and the FastAPI app in `server.py`. Analysis logic is under `opcompass/engine/`; pluggable operator implementations are in `opcompass/operators/`; hardware targets are in `opcompass/hardware/`; YAML architecture specs are in `opcompass/configs/solar_arch/`. The browser UI is static HTML/CSS/JS under `web/`. Tests mirror the package by area in `tests/test_engine/`, `tests/test_hardware/`, and `tests/test_operators/`. `3rdparty/SOLAR/` is vendored external code; avoid changing it unless the task specifically requires it.

## Build, Test, and Development Commands

- `pip install -e .` installs the package and `compass` console command for local development.
- `pip install -e ".[dev]"` installs pytest and coverage tooling.
- `compass list operators` and `compass list hardware` verify discovery.
- `compass analyze matmul --hardware a100 --dtype fp16 --M 4096 --N 4096 --K 4096` runs a representative CLI analysis.
- `uvicorn opcompass.server:app --reload` starts the Web UI/API at `http://127.0.0.1:8000`.
- `pytest` runs the test suite; use `pytest tests/test_engine/test_analyzer.py` for a focused run.

## Coding Style & Naming Conventions

Use Python 3.8+ and four-space indentation. Follow the existing straightforward, typed-enough style: small classes, explicit names, and dataclass-style models where appropriate. Operator modules use lowercase names such as `matmul.py`; operator classes expose a stable lowercase `name` used by discovery. Hardware modules follow `nvidia_<generation>.py` or another descriptive lowercase pattern. Tests are named `test_*.py` and test functions should describe the behavior under test.

## Testing Guidelines

The suite uses pytest. Add or update tests when changing engine behavior, operator formulas, hardware specs, registry discovery, or CLI-visible behavior. Prefer focused tests in the matching subdirectory and include numeric assertions that explain expected FLOPs, bandwidth, or timing relationships. Run `pytest` before submitting broad changes; run the nearest test file for small edits.

## Commit & Pull Request Guidelines

Recent history uses short, imperative summaries such as `Update` and `Add SOLAR as 3rd party evaluate mode`. Keep commits concise and scoped; mention the affected subsystem when useful, for example `Update Hopper bandwidth model`. Pull requests should describe the behavior change, list verification commands, link related issues, and include screenshots when the web UI changes.

## Security & Configuration Tips

Do not commit generated caches, local environment files, or large new vendor artifacts without review. Treat hardware specifications and SOLAR-derived data as source inputs: cite or document provenance when adding new targets or architecture YAML files.
