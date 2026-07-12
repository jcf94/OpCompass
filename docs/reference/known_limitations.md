# Known Limitations of OpCompass v0.2

OpCompass v0.2 produces theoretical bounds and a Matmul-specific analytical
schedule. Results are useful for comparing modeled constraints, not as a
promise of measured kernel runtime.

- `hierarchy_roofline` currently uses unique tensor I/O, HBM bandwidth, and
  peak compute. Despite its retained compatibility name, it is not a true
  L2/L1/shared/register hierarchy model.
- Detailed `pipeline` support is limited to Matmul. Other operators either
  return an explicit permissive roofline fallback or fail in strict mode.
- The pipeline model is `legacy_matmul_v1`: a cycle-based analytical schedule,
  not a dependency-driven or instruction-accurate GPU simulator.
- Default serialization is compact, but the scheduler still constructs the
  complete internal K-iteration sub-operation list. Very large K can therefore
  consume time and memory even when traces are omitted.
- Queue capacity, barriers, buffer lifetimes, loop-carried dependencies,
  detailed memory paths, launch overhead, and small-grid/tail utilization are
  incomplete or absent.
- Uncertainty is reported as `unquantified`; there is no measurement-backed
  confidence interval or calibrated prediction in v0.2.
- Hardware facts use `hardware_spec_version=legacy-v1`. Individual fields do
  not yet carry source, verification date, units provenance, or confidence.
- Operator schemas intentionally omit richer semantics such as layouts,
  transpose, mixed dtypes, convolution stride/padding/dilation/groups,
  attention causal masks, and algorithm variants. These belong to later
  operator-specific releases.
- Diagnostics remain split across structured fallback/errors/candidate
  rejection and the convenience `assumptions`, `warnings`, and
  `missing_effects` lists. A unified code/severity/message/context schema is
  deferred to v0.3 to avoid changing the frozen v0.2 result contract.
- SOLAR is an optional, separately identified backend whose availability and
  supported operators depend on external packages and vendored integration.

No v0.2 result should be described as validated unless it is independently
compared with measurements under a recorded software, clock, power, and kernel
configuration.
