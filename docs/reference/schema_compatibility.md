# Result Schema Compatibility

OpCompass versions its machine-readable result contract independently from the
Python package. The `schema_version` field uses `major.minor.patch` semantics.

- Patch changes may clarify documentation, add optional enum values, or fix a
  serializer bug without removing or changing the meaning of existing fields.
- Minor changes may add optional fields. Consumers must ignore unknown fields.
- Major changes may remove fields, rename fields, change units or types, or
  alter field semantics.

Within the v0.2 release line, successful analysis responses keep schema version
`0.2.0`. Numeric time fields in JSON use explicit `_us` or `_s` suffixes. The
schema version describes semantics; `implementation_version`,
`implementation_revision`, `model_id`, and `hardware_spec_version` identify the
code and inputs that produced a result.

## Compatibility fields

`mode` is retained in v0.2 as a deprecated alias for the requested mode. New
consumers must use `requested_mode` and `executed_mode`, and must inspect
`fallback`. The alias may be removed only in a future major schema revision.

The `/api/*` route namespace remains unchanged for v0.2. A future `/api/v1/*`
namespace will be introduced only with a documented migration window; v0.2
does not duplicate routes preemptively.

Pipeline trace data is optional by contract. Consumers must handle an absent
`sub_ops` field and use the always-present compact schedule plus `trace`
metadata. Error responses are separate from successful result schemas and use
the stable envelope `{"detail": {"code": "..."}}`.

## Numerical changes

A numerical change must also change the relevant `model_id`, implementation
revision, or hardware-spec version and be classified as a bug fix, modeled
effect, hardware-data correction, calibration change, or candidate-space
change. Golden fixtures freeze representative v0.2 A100, H100, B200, and
fallback results.
