# OpCompass v0.2 Session Handoff

Status: implementation complete; release-candidate packaging pending

Branch: `dev/v0.2`

Package version: `0.2.0.dev0`

Last updated: 2026-07-12

This document is the continuation point for the next development session. The
source plan remains
[`opcompass_gap_analysis_and_roadmap.md`](opcompass_gap_analysis_and_roadmap.md);
this file records what has actually landed, what was verified, and what remains
before declaring v0.2 release-ready.

## 1. Current outcome

The main v0.2 trust contract is operational:

- Analyzer inputs are centrally validated and canonicalized.
- Requested and executed modes are distinct, and fallback is never silent.
- Strict mode rejects unsupported fallback.
- Results identify their estimate kind, support level, model, schema, build,
  implementation revision, hardware-spec version, evidence, and uncertainty.
- Successful Analyzer results are recursively checked for NaN and infinity.
- Default pipeline JSON contains a compact summary; trace expansion is opt-in
  and capped at 5,000 sub-operations.
- API requests use Pydantic models and important failures have stable codes.
- CSV has a fixed flat schema and standards-compliant quoting.
- Pipeline table phase times use the declared compute-unit clock.
- First-party and optional SOLAR tests are separately selectable.
- A100, H100, B200, and fallback compact golden fixtures freeze the baseline.
- Wheel and sdist builds include the Web UI and SOLAR architecture YAML files.
- The Web UI exposes requested-to-executed routing, fallback, model identity,
  evidence, and uncertainty instead of relying on the legacy `mode` field.
- The analyze HTTP response is a closed nested Pydantic contract, including
  mode-specific pipeline and SOLAR sections and a documented error envelope.
- Real ASGI requests cover success, fallback, strict rejection, domain errors,
  trace limits, optional-backend failures, and uncaught internal failures.
- `scripts/package_smoke.py` builds wheel and sdist, verifies a clean wheel
  install, and rebuilds a wheel from the produced sdist.

The current pipeline implementation is still `legacy_matmul_v1`: a
cycle-based analytical Matmul schedule, not a generic dependency-driven or
cycle-accurate GPU scheduler. That replacement begins in v0.3.

## 2. Landed commits

The v0.2 work in this development sequence is split into reviewable commits:

| Commit | Purpose |
|---|---|
| `86cc7aa` | Add operator parameter validation contract |
| `ebefccf` | Make analysis fallback semantics explicit |
| `4d348a3` | Bound pipeline trace serialization |
| `b14ebc5` | Harden API and result serialization contracts |
| `af494be` | Separate first-party and SOLAR test suites |
| `d556658` | Freeze v0.2 baselines and package runtime assets |
| `269bd00` | Add reproducibility and capability metadata |
| `bdcefaf` | Add stable backend and candidate errors |
| `6399d47` | Reject unsupported dtypes before analysis |
| `c62cd76` | Surface analysis contract in Web UI |

## 3. Current verification baseline

The latest complete run passed:

```text
pytest -q
160 passed in 12.39s

node --check web/js/result_contract.js
node --check web/js/app.js
python -m compileall -q opcompass tests
git diff --check
```

The test-suite boundary is:

```bash
# Fast first-party suite without optional SOLAR integration
pytest -m "not solar" -q

# Optional SOLAR integration suite
pytest -m solar -q

# All first-party tests, including SOLAR when dependencies are available
pytest -q
```

Packaging was also checked with `python scripts/package_smoke.py`:

- built `opcompass-0.2.0.dev0-py3-none-any.whl`;
- built `opcompass-0.2.0.dev0.tar.gz`;
- installed the wheel in an isolated venv;
- confirmed package/API version agreement;
- confirmed `web/index.html` and JavaScript assets were installed;
- confirmed packaged SOLAR YAML paths were readable.

## 4. Remaining v0.2 work

### P0 — release-candidate audit

#### 4.1 Complete the HTTP API contract — completed 2026-07-12

`AnalyzeResponse` now uses closed nested response models for all serialized
analysis sections. `/api/analyze` documents a common structured error envelope
and returns stable codes for unknown operators, unknown hardware, and internal
failures. v0.2 retains `/api/*`; route versioning is deferred until a written
compatibility/migration policy requires a new namespace. Other metadata routes
remain unversioned dictionaries and should be addressed with that policy.

#### 4.2 Add real HTTP integration tests — completed 2026-07-12

`httpx` is now a dev dependency. FastAPI `TestClient` verifies actual JSON and
content types for success, Pydantic rejection, fallback, strict rejection,
unsupported dtype, unavailable SOLAR, infeasible candidates, trace limiting,
unknown resources, and internal failures. The tests exposed and fixed a real
fallback response-validation mismatch for empty roofline detail.

#### 4.3 Automate clean package smoke tests — completed 2026-07-12

`scripts/package_smoke.py` now builds wheel and sdist, installs the wheel into
a clean venv, exercises the CLI and representative JSON analysis, generates
OpenAPI, checks Web/SOLAR resources, and rebuilds from the sdist. It passed
locally on Python 3.8. Wiring it into CI remains part of 4.4.

#### 4.4 Add CI and supported-Python gates — explicitly omitted

The project intentionally does not add GitHub Actions or a Python-version
matrix for v0.2. The release baseline is local Python 3.8 verification plus the
clean-install package smoke script. Packaging continues to declare `>=3.8`;
this is a source-compatibility claim, not a CI-verified version matrix.

#### 4.5 Write release documentation — completed 2026-07-12

Release notes, semantic schema compatibility policy, and known limitations are
published under `docs/releases/` and `docs/reference/`. The package remains
`0.2.0.dev0` until an explicit release packaging/versioning step is requested.

### P1 — finish or explicitly defer

#### 4.6 Make operator specs uniformly explicit — completed 2026-07-12

All six built-in operators now override `spec` explicitly. Tests enforce typed,
positive parameter declarations and canonical declaration order. Richer
layout, transpose, mixed-dtype, convolution, attention, and algorithm semantics
are documented as deferred rather than implied by the current formulas.

#### 4.7 Normalize diagnostics — explicitly deferred to v0.3

Assumptions, warnings, missing effects, fallback, candidate rejection, and
errors exist, but they do not yet share one diagnostic schema.

Adding a new unified diagnostic collection after freezing schema `0.2.0` would
be a contract change. v0.2 retains structured fallback, errors, and candidate
rejections alongside the existing convenience lists. The known-limitations
page explicitly records the unified code/severity/message/context schema as a
v0.3 task.

### Explicitly deferred to v0.3+

The following are known gaps but are not v0.2 release blockers:

- replacing `legacy_matmul_v1` with explicit Pipeline IR;
- dependency-driven resource scheduling;
- compact repeated schedule as the internal representation;
- queue, barrier, buffer-lifetime, and loop-carried dependency semantics;
- detailed memory paths and modern GPU synchronization;
- measured calibration and hardware fact provenance;
- non-Matmul pipeline models.

## 5. Exit-criteria audit

| v0.2 exit criterion | Status | Notes |
|---|---|---|
| Invalid dimensions return typed errors | Met at Analyzer/API boundary | Direct formula methods remain low-level and permissive |
| Pipeline fallback is explicit; strict never falls back | Met | Web UI now displays executed mode and fallback |
| Successful numeric result fields are finite | Met | Recursive Analyzer finalization check plus strict JSON |
| Default JSON size is independent of K trace length | Met for serialization | Internal sub-op generation still scales with K; defer to v0.3 |
| API/OpenAPI, CLI, serialization tests pass on release baseline | Met | 160 tests pass locally on Python 3.8; HTTP TestClient included; CI matrix omitted by decision |
| First-party and vendored/SOLAR counts are separate | Met | `solar` marker and `testpaths = ["tests"]` |

## 6. Recommended first task next session

Perform the explicit release packaging step when desired: choose the RC version,
update package/release-note version references together, rerun the full suite
and `scripts/package_smoke.py`, and tag the resulting revision. No v0.2 feature
implementation remains.

Suggested sequence:

```text
1. Select the RC version (for example `0.2.0rc1`).
2. Update package and release-note version references.
3. Run `pytest -q` and `python scripts/package_smoke.py`.
4. Commit and tag the release candidate.
```

Useful starting files:

- `opcompass/api_models.py`
- `opcompass/server.py`
- `opcompass/models.py`
- `opcompass/engine/result.py`
- `tests/test_server.py`
- `tests/test_packaging.py`
- `pyproject.toml`

## 7. Session caveat

The Web result-contract behavior has Node tests and static-resource checks. An
interactive screenshot-level browser QA pass was attempted on 2026-07-12, but
the browser runtime reported no available in-app browser instances. Perform
that optional visual pass when an in-app browser instance is available.
