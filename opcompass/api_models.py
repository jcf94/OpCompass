"""Versioned Pydantic contracts for the public HTTP API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from opcompass.engine.result import MAX_TRACE_SUB_OPS
from opcompass.models import AnalysisMode, DataType


class PipelineConfigRequest(BaseModel):
    """Validated pipeline feature and tile overrides."""

    model_config = ConfigDict(extra="forbid")

    async_copy_enabled: bool = True
    sparsity_2_4_enabled: bool = False
    block_m: Optional[int] = Field(default=None, gt=0)
    block_n: Optional[int] = Field(default=None, gt=0)
    block_k: Optional[int] = Field(default=None, gt=0)
    stage_count: Optional[int] = Field(default=None, gt=0)
    warp_count: Optional[int] = Field(default=None, gt=0)


class AnalyzeRequest(BaseModel):
    """Stable request body for ``POST /api/analyze``."""

    model_config = ConfigDict(extra="forbid")

    operator: str = Field(min_length=1)
    hardware: str = Field(min_length=1)
    dtype: DataType = DataType.FP16
    mode: AnalysisMode = AnalysisMode.HIERARCHY_ROOFLINE
    dims: Dict[str, StrictInt]
    pipeline_config: Optional[PipelineConfigRequest] = None
    strict: bool = False
    include_trace: bool = False
    trace_limit: int = Field(default=1000, ge=1, le=MAX_TRACE_SUB_OPS)


class AnalyzeResponse(BaseModel):
    """Core response contract; mode-specific extensions remain explicit extras."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

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
    fallback: Optional[Dict[str, Any]]
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
    roofline_data: Dict[str, float]
