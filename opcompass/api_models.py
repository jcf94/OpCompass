"""Versioned Pydantic contracts for the public HTTP API."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from opcompass.engine.result import MAX_TRACE_SUB_OPS
from opcompass.models import AnalysisMode, DataType


class ApiModel(BaseModel):
    """Closed base model used by every public API contract."""

    model_config = ConfigDict(extra="forbid")


class PipelineConfigRequest(ApiModel):
    async_copy_enabled: bool = True
    sparsity_2_4_enabled: bool = False
    block_m: Optional[int] = Field(default=None, gt=0)
    block_n: Optional[int] = Field(default=None, gt=0)
    block_k: Optional[int] = Field(default=None, gt=0)
    stage_count: Optional[int] = Field(default=None, gt=0)
    warp_count: Optional[int] = Field(default=None, gt=0)


class AnalyzeRequest(ApiModel):
    """Stable request body for ``POST /api/analyze``."""

    operator: str = Field(min_length=1)
    hardware: str = Field(min_length=1)
    dtype: DataType = DataType.FP16
    mode: AnalysisMode = AnalysisMode.HIERARCHY_ROOFLINE
    dims: Dict[str, StrictInt]
    pipeline_config: Optional[PipelineConfigRequest] = None
    strict: bool = False
    include_trace: bool = False
    trace_limit: int = Field(default=1000, ge=1, le=MAX_TRACE_SUB_OPS)


class FallbackResponse(ApiModel):
    from_mode: AnalysisMode
    to_mode: AnalysisMode
    reason_code: str
    message: str


class EvidenceResponse(ApiModel):
    coverage: str
    sources: List[str]


class UncertaintyResponse(ApiModel):
    status: str
    reason: str
    lower_time_us: Optional[float]
    upper_time_us: Optional[float]


class RooflineResponse(ApiModel):
    operational_intensity: Optional[float] = None
    peak_flops: Optional[float] = None
    peak_bandwidth: Optional[float] = None
    achievable_flops: Optional[float] = None


class TraceMetadataResponse(ApiModel):
    included: bool
    total_sub_ops: int
    returned_sub_ops: int
    complete: bool
    limit: int


class ScheduledSubOpResponse(ApiModel):
    name: str
    pipeline_stage: str
    start_cycle: int
    end_cycle: int
    duration_cycles: int
    work_units: int
    iteration: int


class PipelineScheduleResponse(ApiModel):
    total_cycles_per_block: int
    total_time_s: float
    total_time_us: float
    wave_count: int
    grid_size: int
    num_k_iterations: int
    bottleneck_stage: str
    per_iteration_cycles: int
    prologue_cycles: int
    epilogue_cycles: int
    trace: TraceMetadataResponse
    sub_ops: Optional[List[ScheduledSubOpResponse]] = None


class TilingResponse(ApiModel):
    block_m: int
    block_n: int
    block_k: int
    shared_memory_per_block: int
    num_warps_per_block: int
    stage_count: int
    registers_per_thread: int
    registers_per_block: int
    candidate_name: str


class PipelineCandidateResponse(ApiModel):
    name: str
    block_m: int
    block_n: int
    block_k: int
    warp_count: int
    stage_count: int
    copy_path: str
    mma_path: str
    scheduling: str
    cta_order: str
    rejection_reason: str
    selected: bool


class PipelineMemoryResponse(ApiModel):
    logical_cta_read_bytes: float
    logical_cta_write_bytes: float
    logical_hbm_read_bytes: float
    logical_hbm_write_bytes: float
    effective_hbm_read_bytes: float
    effective_hbm_write_bytes: float
    unique_tensor_read_bytes: float
    unique_tensor_write_bytes: float
    l2_reuse_factor: float


class SolarVariantResponse(ApiModel):
    runtime_ms: float
    bottleneck: str
    arithmetic_intensity: float
    memory_bytes: int
    compute_cycles: Optional[int] = None


class SolarMemoryResponse(ApiModel):
    weight_bytes: int
    model_io_bytes: int
    intermediate_bytes: int


class SolarSpeedupResponse(ApiModel):
    fused_vs_unfused: float
    fused_prefetched_vs_unfused: float


class SolarResponse(ApiModel):
    num_layers: int
    total_macs: int
    arch_name: str
    arch_freq_ghz: float
    unfused: SolarVariantResponse
    fused: SolarVariantResponse
    fused_prefetched: SolarVariantResponse
    memory_breakdown: SolarMemoryResponse
    speedup: SolarSpeedupResponse


class AnalyzeResponse(ApiModel):
    """Closed, versioned response contract for every analysis mode."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    operator: str
    hardware: str
    shapes: Dict[str, int]
    dtype: DataType
    mode: AnalysisMode
    requested_mode: AnalysisMode
    executed_mode: AnalysisMode
    estimate_kind: str
    support_level: str
    schema_version: str
    model_id: str
    implementation_version: str
    implementation_revision: str
    hardware_spec_version: str
    evidence: EvidenceResponse
    uncertainty: UncertaintyResponse
    fallback: Optional[FallbackResponse]
    assumptions: List[str]
    warnings: List[str]
    missing_effects: List[str]
    total_flops: int
    total_read_bytes: Union[int, float]
    total_write_bytes: Union[int, float]
    memory_read_time_us: float
    compute_time_us: float
    memory_write_time_us: float
    sol_time_us: float
    sol_tflops: float
    bottleneck: str
    stage_breakdown: Dict[str, float]
    roofline_data: RooflineResponse
    pipeline_schedule: Optional[PipelineScheduleResponse] = None
    pipeline_config: Optional[PipelineConfigRequest] = None
    tiling_info: Optional[TilingResponse] = None
    pipeline_memory_breakdown: Optional[PipelineMemoryResponse] = None
    pipeline_candidates: Optional[List[PipelineCandidateResponse]] = None
    pipeline_ir_schedule: Optional[Dict[str, object]] = None
    pipeline_legacy_comparison: Optional[Dict[str, Union[str, int, float]]] = None
    solar_data: Optional[SolarResponse] = None


class ErrorIssue(ApiModel):
    loc: Optional[List[Union[str, int]]] = None
    parameter: Optional[str] = None
    reason: Optional[str] = None
    message: Optional[str] = None
    type: Optional[str] = None
    input: Optional[object] = None
    ctx: Optional[Dict[str, object]] = None


class ErrorDetail(ApiModel):
    code: str
    message: Optional[str] = None
    operator: Optional[str] = None
    hardware: Optional[str] = None
    dtype: Optional[str] = None
    requested_mode: Optional[str] = None
    backend: Optional[str] = None
    field: Optional[str] = None
    supported_dtypes: Optional[List[str]] = None
    missing_dependencies: Optional[List[str]] = None
    issues: Optional[List[ErrorIssue]] = None


class ErrorResponse(ApiModel):
    detail: ErrorDetail
