# OpCompass v0.2 Session Handoff

Status: implementation in progress

Branch: `dev/v0.2`

Package version: `0.2.0.dev0`

Last updated: 2026-07-11

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
147 passed in 9.93s

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

Packaging was also checked manually:

- built `opcompass-0.2.0.dev0-py3-none-any.whl`;
- built `opcompass-0.2.0.dev0.tar.gz`;
- installed the wheel in an isolated venv;
- confirmed package/API version agreement;
- confirmed `web/index.html` and JavaScript assets were installed;
- confirmed packaged SOLAR YAML paths were readable.

## 4. Remaining v0.2 work

### P0 — required before release candidate

#### 4.1 Complete the HTTP API contract

`AnalyzeRequest` is strict, but `AnalyzeResponse` still leaves mode-specific
sections partially open through `Dict[str, Any]` and allowed extra fields.
Other routes also return unversioned `Dict` payloads.

Next actions:

1. Define nested Pydantic response models for fallback, evidence, uncertainty,
   roofline, compact schedule, trace metadata, tiling, candidates, pipeline
   memory, and SOLAR data.
2. Add structured error response models and document them in OpenAPI.
3. Give unknown operator, unknown hardware, and uncaught internal failures
   stable error codes; do not return a mixture of typed objects and plain
   strings.
4. Decide whether v0.2 retains `/api/*` or introduces `/api/v1/*`; avoid
   duplicating routes unless a compatibility policy is written down.

#### 4.2 Add real HTTP integration tests

Current API tests call route functions directly and inspect `app.openapi()`.
They do not execute ASGI HTTP requests because `httpx` is not in the dev
dependencies.

Next actions:

1. Add a compatible `httpx` version to the `dev` extra.
2. Use FastAPI `TestClient` to cover successful analysis, Pydantic rejection,
   fallback, strict unsupported mode, unsupported dtype, unavailable optional
   backend, infeasible candidate, trace limiting, and response validation.
3. Verify content types and the exact serialized error envelope.

#### 4.3 Automate clean package smoke tests

Resource packaging was manually verified and source-tree tests check resource
paths, but pytest/CI does not build and install artifacts in a clean
environment.

Next actions:

1. Add the `build` package to development/release tooling.
2. Build wheel and sdist in a temporary output directory.
3. Install the wheel into a clean venv and run:
   - `compass --version`;
   - `compass list operators`;
   - `compass list hardware`;
   - representative JSON analysis;
   - API import/OpenAPI generation;
   - Web and SOLAR YAML resource checks.
4. Repeat a build from the produced sdist.

#### 4.4 Add CI and supported-Python gates

The project currently has no checked-in CI. v0.2 has been verified locally on
Python 3.8, but the declared support range is `>=3.8`.

Next actions:

1. Add a first-party GitHub Actions matrix for supported Python versions.
2. Run `pytest -m "not solar"` as the base gate.
3. Run SOLAR integration separately so optional dependencies and vendored
   tests do not destabilize the base job.
4. Add a package build/clean-install job.
5. Choose and document the upper Python version actually supported by the
   dependency set; do not leave an unbounded claim without CI evidence.

#### 4.5 Write release documentation

Next actions:

1. Add a changelog or v0.2 release note summarizing contract changes and
   compatibility impact.
2. Document the semantic schema version and compatibility policy.
3. Publish a concise known-limitations page covering:
   - `hierarchy_roofline` is currently an HBM/peak bound, not a true hierarchy
     model;
   - detailed pipeline support is Matmul-only;
   - uncertainty is explicitly unquantified;
   - hardware facts are still `legacy-v1` without per-field provenance;
   - the scheduler builds the full internal sub-op list even though default
     serialization is compact.
4. Change `0.2.0.dev0` to an RC only after the release gates pass.

### P1 — finish or explicitly defer

#### 4.6 Make operator specs uniformly explicit

Central validation is complete, but most operators still derive a basic spec
from legacy `param_dims`. Only selected operators declare defaults or
cross-field constraints explicitly.

For v0.2, either:

- convert every current operator to an explicit `OperatorSpec`, including
  parameter kinds and meaningful constraints; or
- document that richer layout, transpose, mixed-dtype, stride, padding,
  dilation, groups, causal, and algorithm semantics are deferred to the
  operator-specific releases in the roadmap.

Do not expand v0.2 into implementing all those later operator semantics.

#### 4.7 Normalize diagnostics

Assumptions, warnings, missing effects, fallback, candidate rejection, and
errors exist, but they do not yet share one diagnostic schema.

A small v0.2-compatible improvement would define code/severity/message/context
for diagnostics while retaining the existing convenience lists. A larger
diagnostic rewrite should be deferred.

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
| API/OpenAPI, CLI, serialization tests pass on supported Python | Partial | Local Python 3.8 passes; HTTP TestClient and version matrix remain |
| First-party and vendored/SOLAR counts are separate | Met | `solar` marker and `testpaths = ["tests"]` |

## 6. Recommended first task next session

Start with real HTTP contract testing because it will expose any mismatch
between Pydantic models, FastAPI exception handling, and actual JSON responses
before CI freezes the interface.

Suggested sequence:

```text
1. Add httpx to the dev extra.
2. Add TestClient success/error tests.
3. Replace AnalyzeResponse Any/dict sections with nested models.
4. Add structured error response models to OpenAPI.
5. Run the complete suite.
6. Commit the API contract as one focused change.
7. Add package smoke CI and the Python matrix next.
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
interactive screenshot-level browser QA pass was not completed because the
configured browser-control skill file was unavailable in this environment.
Perform that pass before the release candidate if browser tooling is available.
