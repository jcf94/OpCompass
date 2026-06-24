from __future__ import annotations
"""Matrix multiplication: C = A × B  with shapes (M, K) × (K, N) → (M, N)."""

import math
from dataclasses import replace

from opcompass.models import DataType, PipelineConfig, PipelineKernelCandidate, SubOp, TilingInfo
from opcompass.operators.base import Operator


def _dtype_to_torch_str(dtype: DataType) -> str:
    """Map OpCompass DataType to a torch dtype string for code generation."""
    _map = {
        DataType.FP64: "torch.float64",
        DataType.FP32: "torch.float32",
        DataType.TF32: "torch.float32",  # TF32 uses FP32 storage
        DataType.FP16: "torch.float16",
        DataType.BF16: "torch.bfloat16",
        DataType.INT8: "torch.int8",
        DataType.FP8: "torch.float8_e4m3fn",
    }
    return _map.get(dtype, "torch.float16")


class Matmul(Operator):
    """Standard dense matrix multiplication.

    Shapes:
        A: (M, K)
        B: (K, N)
        C: (M, N)

    FLOPs = 2 * M * N * K  (one multiply-add = 2 ops)
    """

    name = "matmul"
    description = "Dense matrix multiplication C[M,N] = A[M,K] × B[K,N]"

    @property
    def param_dims(self) -> dict[str, str]:
        return {
            "M": "Rows of A / C",
            "N": "Cols of B / C",
            "K": "Inner dimension (cols of A, rows of B)",
        }

    def compute_torch(self, inputs: List["torch.Tensor"], **kwargs) -> List["torch.Tensor"]:
        import torch

        A, B = inputs
        return [torch.matmul(A, B)]

    # ------------------------------------------------------------------
    # Solar mode support
    # ------------------------------------------------------------------

    def get_solar_model_source(self, dtype, **dims) -> str:
        """Generate a SOLAR-compatible model file for matmul C = A @ B.

        Shapes: A(M,K) × B(K,N) → C(M,N)
        """
        M = dims.get("M", 0)
        N = dims.get("N", 0)
        K = dims.get("K", 0)
        torch_dtype = _dtype_to_torch_str(dtype)

        return f'''import torch
from torch import nn

class Model(nn.Module):
    """Matmul: C[M,N] = A[M,K] × B[K,N]."""

    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return torch.matmul(A, B)


def get_inputs():
    M, N, K = {M}, {N}, {K}
    A = torch.randn(M, K, dtype={torch_dtype})
    B = torch.randn(K, N, dtype={torch_dtype})
    return [A, B]
'''

    def compute_flops(self, M: int = 0, N: int = 0, K: int = 0, **kwargs) -> int:
        return 2 * M * N * K

    def compute_io_bytes(
        self, dtype: DataType, M: int = 0, N: int = 0, K: int = 0, **kwargs
    ) -> tuple[int, int]:
        bs = dtype.byte_size
        read_bytes = (M * K + K * N) * bs
        write_bytes = M * N * bs
        return read_bytes, write_bytes

    # ------------------------------------------------------------------
    # Pipeline-level decomposition
    # ------------------------------------------------------------------

    def get_tile_constraints(self, hardware=None, dtype=None) -> dict:
        """Return CTA tile granularity derived from the compute instruction.

        The pipeline model maps the CTA tile onto hardware instructions:
        Tensor Core MMA has architecture/dtype-specific K granularity, while
        CUDA-core FMA is scalar and only requires positive integers.
        """
        if dtype is None:
            dtype = DataType.FP16

        arch = getattr(hardware, "architecture", "").lower() if hardware is not None else ""

        # CUDA-core FMA path.
        if dtype in (DataType.FP32, DataType.FP64):
            granularity = (1, 1, 1)
            instruction = "cuda_fma"
        # Tensor Core paths. M/N follow the common m16n8 instruction tile.
        elif dtype in (DataType.FP16, DataType.BF16):
            granularity = (16, 8, 16)
            instruction = "mma_m16n8k16"
        elif dtype == DataType.TF32:
            granularity = (16, 8, 8)
            instruction = "mma_m16n8k8"
        elif dtype in (DataType.INT8, DataType.FP8):
            granularity = (16, 8, 32)
            instruction = "mma_m16n8k32"
        elif dtype == DataType.INT4:
            granularity = (16, 8, 64)
            instruction = "mma_m16n8k64"
        else:
            granularity = (16, 8, 16)
            instruction = "mma_m16n8k16"

        # Older Volta WMMA FP16 is commonly exposed as m16n16k16.
        if arch == "volta" and dtype == DataType.FP16:
            granularity = (16, 16, 16)
            instruction = "wmma_m16n16k16"

        m, n, k = granularity
        return {
            "block_m": {"multiple_of": m, "min": m},
            "block_n": {"multiple_of": n, "min": n},
            "block_k": {"multiple_of": k, "min": k},
            "instruction": instruction,
        }

    def get_tiling_strategy(self, hardware, dtype=None, pipeline_config=None, **dims):
        """Return a CUTLASS-style tiling strategy for the given hardware.

        Uses architecture-aware default tile sizes, then validates
        shared memory constraints and shrinks block_K if needed.
        """
        if dtype is None:
            dtype = DataType.FP16

        candidate = self.get_pipeline_candidates(
            hardware, dtype, pipeline_config=pipeline_config, **dims
        )[0]
        return self._tiling_from_candidate(candidate, hardware, dtype)

    def get_pipeline_candidates(self, hardware, dtype=None, pipeline_config=None, **dims):
        """Return feasible matmul pipeline candidates in preference order."""
        if dtype is None:
            dtype = DataType.FP16
        if pipeline_config is None:
            pipeline_config = PipelineConfig()

        raw_candidates = self._raw_pipeline_candidates(hardware, dtype, pipeline_config)
        feasible: list[PipelineKernelCandidate] = []
        rejected: list[PipelineKernelCandidate] = []

        for candidate in raw_candidates:
            try:
                self._tiling_from_candidate(candidate, hardware, dtype)
                feasible.append(candidate)
            except ValueError as exc:
                rejected.append(replace(candidate, rejection_reason=str(exc)))

        if feasible:
            return feasible

        if self._has_forced_pipeline_shape(pipeline_config) and rejected:
            raise ValueError(rejected[0].rejection_reason)
        if rejected:
            reasons = "; ".join(c.rejection_reason for c in rejected[:3])
            raise ValueError(f"No feasible pipeline candidates for {hardware.name}: {reasons}")
        raise ValueError(f"No pipeline candidates for {hardware.name}")

    def get_rejected_pipeline_candidates(self, hardware, dtype=None, pipeline_config=None, **dims):
        """Return rejected candidates with reasons for diagnostics."""
        if dtype is None:
            dtype = DataType.FP16
        if pipeline_config is None:
            pipeline_config = PipelineConfig()

        rejected: list[PipelineKernelCandidate] = []
        for candidate in self._raw_pipeline_candidates(hardware, dtype, pipeline_config):
            try:
                self._tiling_from_candidate(candidate, hardware, dtype)
            except ValueError as exc:
                rejected.append(replace(candidate, rejection_reason=str(exc)))
        return rejected

    def _raw_pipeline_candidates(self, hardware, dtype, pipeline_config):
        arch = getattr(hardware, "architecture", "").lower()
        base_tiles = self._base_candidate_tiles(arch, dtype)
        copy_path = self._copy_path(arch, pipeline_config.async_copy_enabled)
        mma_path = self._mma_path(arch, dtype)
        scheduling = "warp_specialized" if arch in {"hopper", "blackwell"} else "standard"

        forced = self._has_forced_pipeline_shape(pipeline_config)
        if forced:
            default_m, default_n, default_k = base_tiles[0]
            tile = (
                pipeline_config.block_m or default_m,
                pipeline_config.block_n or default_n,
                pipeline_config.block_k or default_k,
            )
            stage_counts = [pipeline_config.stage_count or self._default_stage_count(arch)]
            warp_counts = [pipeline_config.warp_count or self._default_warp_count(arch, tile)]
            base_tiles = [tile]
        else:
            stage_counts = [2, 3] if arch == "ampere" else [3, 4] if arch in {"hopper", "blackwell"} else [2]
            warp_counts = [4, 8] if arch in {"ampere", "hopper", "blackwell"} else [4]

        candidates: list[PipelineKernelCandidate] = []
        seen: set[tuple[int, int, int, int, int]] = set()
        for bM, bN, bK in base_tiles:
            for stage_count in stage_counts:
                for warp_count in warp_counts:
                    key = (bM, bN, bK, warp_count, stage_count)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(PipelineKernelCandidate(
                        name=f"{arch or 'generic'}_{bM}x{bN}x{bK}_w{warp_count}_s{stage_count}",
                        block_m=bM,
                        block_n=bN,
                        block_k=bK,
                        warp_count=warp_count,
                        stage_count=stage_count,
                        copy_path=copy_path,
                        mma_path=mma_path,
                        scheduling=scheduling,
                        cta_order="swizzled" if arch in {"hopper", "blackwell"} else "row_major",
                    ))
        return candidates

    @staticmethod
    def _has_forced_pipeline_shape(pipeline_config) -> bool:
        return bool(
            pipeline_config
            and (
                pipeline_config.block_m
                or pipeline_config.block_n
                or pipeline_config.block_k
                or pipeline_config.stage_count
                or pipeline_config.warp_count
            )
        )

    @staticmethod
    def _base_candidate_tiles(arch: str, dtype) -> list[tuple[int, int, int]]:
        if arch == "ampere":
            if dtype in (DataType.FP16, DataType.BF16):
                return [(128, 128, 32), (128, 64, 32), (64, 128, 32)]
            if dtype == DataType.TF32:
                return [(128, 128, 16), (128, 64, 16), (64, 128, 16)]
            if dtype == DataType.FP32:
                return [(64, 64, 16), (128, 64, 16), (64, 128, 16)]
            return [(128, 128, 32), (128, 64, 32), (64, 128, 32)]
        if arch == "hopper":
            if dtype == DataType.FP32:
                return [(128, 128, 32), (64, 128, 32), (128, 64, 32)]
            return [(128, 128, 64), (128, 256, 64), (256, 128, 64)]
        if arch == "blackwell":
            if dtype in (DataType.FP16, DataType.BF16, DataType.FP8):
                return [(256, 128, 64), (128, 128, 64), (256, 256, 64)]
            if dtype == DataType.TF32:
                return [(128, 128, 64), (128, 256, 64), (256, 128, 64)]
            if dtype == DataType.FP32:
                return [(128, 128, 32), (128, 64, 32), (64, 128, 32)]
            return [(256, 128, 64), (128, 128, 64), (256, 256, 64)]
        return [(64, 64, 32)]

    @staticmethod
    def _copy_path(arch: str, async_enabled: bool) -> str:
        if not async_enabled:
            return "global_load"
        if arch in {"hopper", "blackwell"}:
            return "tma"
        return "cp_async"

    @staticmethod
    def _mma_path(arch: str, dtype) -> str:
        if dtype in (DataType.FP32, DataType.FP64):
            return "cuda_fma"
        if arch == "blackwell":
            return "umma"
        if arch == "hopper":
            return "wgmma"
        return "mma"

    @staticmethod
    def _default_stage_count(arch: str) -> int:
        return 3 if arch in {"hopper", "blackwell"} else 2

    @staticmethod
    def _default_warp_count(arch: str, tile: tuple[int, int, int]) -> int:
        bM, bN, _ = tile
        return 8 if arch in {"hopper", "blackwell"} and bM * bN >= 32768 else 4

    def _tiling_from_candidate(self, candidate, hardware, dtype) -> TilingInfo:
        constraints = self.get_tile_constraints(hardware, dtype)
        for name, value in (
            ("block_m", candidate.block_m),
            ("block_n", candidate.block_n),
            ("block_k", candidate.block_k),
        ):
            self._apply_tile_override(name, value, value, constraints)
        if candidate.stage_count <= 0:
            raise ValueError("stage_count must be a positive integer")
        if candidate.warp_count <= 0:
            raise ValueError("warp_count must be a positive integer")

        bs = dtype.byte_size
        smem = self._shared_memory_bytes(
            candidate.block_m, candidate.block_n, candidate.block_k, bs, candidate.stage_count
        )
        cu = hardware.compute_unit
        smem_limit = cu.shared_memory_max_kb * 1024
        if smem_limit > 0 and smem > smem_limit:
            raise ValueError(
                f"Requested tile {candidate.block_m}x{candidate.block_n}x{candidate.block_k} "
                f"with {candidate.stage_count} stages uses {smem / 1024:.1f} KB shared memory, "
                f"exceeding {hardware.name} limit of {cu.shared_memory_max_kb} KB per {cu.name}."
            )

        regs_per_thread = self._estimate_registers_per_thread(candidate)
        threads_per_block = candidate.warp_count * cu.threads_per_warp
        regs_per_block = regs_per_thread * threads_per_block
        register_file_regs = (cu.register_file_kb * 1024) // 4 if cu.register_file_kb > 0 else 0
        if cu.max_registers_per_thread > 0 and regs_per_thread > cu.max_registers_per_thread:
            raise ValueError(
                f"Candidate {candidate.name} estimates {regs_per_thread} registers/thread, "
                f"exceeding {hardware.name} limit of {cu.max_registers_per_thread}."
            )
        if cu.max_registers_per_block > 0 and regs_per_block > cu.max_registers_per_block:
            raise ValueError(
                f"Candidate {candidate.name} estimates {regs_per_block} registers/block, "
                f"exceeding {hardware.name} limit of {cu.max_registers_per_block}."
            )
        if register_file_regs > 0 and regs_per_block > register_file_regs:
            raise ValueError(
                f"Candidate {candidate.name} estimates {regs_per_block} registers/block, "
                f"exceeding {hardware.name} register file capacity of {register_file_regs} registers."
            )

        return TilingInfo(
            block_m=candidate.block_m,
            block_n=candidate.block_n,
            block_k=candidate.block_k,
            shared_memory_per_block=smem,
            num_warps_per_block=candidate.warp_count,
            stage_count=candidate.stage_count,
            registers_per_thread=regs_per_thread,
            registers_per_block=regs_per_block,
            candidate_name=candidate.name,
            candidate=candidate,
        )

    @staticmethod
    def _shared_memory_bytes(block_m: int, block_n: int, block_k: int, bytes_per_element: float, stage_count: int) -> int:
        mainloop = stage_count * (block_m * block_k + block_k * block_n) * bytes_per_element
        epilogue = block_m * block_n * bytes_per_element
        return int(math.ceil(mainloop + epilogue))

    @staticmethod
    def _estimate_registers_per_thread(candidate) -> int:
        threads = max(1, candidate.warp_count * 32)
        accumulator_regs = math.ceil(candidate.block_m * candidate.block_n / threads)
        operand_regs = math.ceil(candidate.block_k * (candidate.block_m + candidate.block_n) / (threads * 8))
        stage_regs = 2 * max(1, candidate.stage_count - 1)
        return int(24 + accumulator_regs + operand_regs + stage_regs)

    @staticmethod
    def _apply_tile_override(name: str, default: int, value, constraints: dict) -> int:
        if value in (None, ""):
            return default
        try:
            tile = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a positive integer")
        if tile <= 0:
            raise ValueError(f"{name} must be a positive integer")
        rule = constraints.get(name, {})
        multiple = int(rule.get("multiple_of", 1))
        minimum = int(rule.get("min", multiple))
        if tile < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        if multiple > 1 and tile % multiple != 0:
            raise ValueError(
                f"{name}={tile} is invalid for {constraints.get('instruction', 'this instruction')}; "
                f"it must be a multiple of {multiple}"
            )
        return tile

    @staticmethod
    def _l2_capacity_bytes(hardware) -> int:
        for tier in getattr(hardware.memory, "tiers", []):
            if tier.name.lower() == "l2":
                return int(tier.capacity_bytes)
        return 0

    def _matmul_l2_reuse_bytes(self, hardware, dtype, tiling, M: int, N: int) -> tuple[float, float]:
        """Return average per-CTA HBM bytes for A/B loads in one K-slice."""
        bs = dtype.byte_size
        bM = tiling.block_m
        bN = tiling.block_n
        bK = tiling.block_k
        grid_m = max(1, math.ceil(M / bM))
        grid_n = max(1, math.ceil(N / bN))
        grid_size = grid_m * grid_n

        logical_a = grid_size * bM * bK * bs
        logical_b = grid_size * bK * bN * bs
        unique_a = M * bK * bs
        unique_b = N * bK * bs

        l2_capacity = self._l2_capacity_bytes(hardware)
        working_set = unique_a + unique_b
        fit_ratio = min(1.0, l2_capacity / working_set) if working_set > 0 and l2_capacity > 0 else 0.0

        effective_a = unique_a * fit_ratio + logical_a * (1.0 - fit_ratio)
        effective_b = unique_b * fit_ratio + logical_b * (1.0 - fit_ratio)
        return effective_a / grid_size, effective_b / grid_size

    def get_ops_breakdown(self, dtype=None, hardware=None, pipeline_config=None, **dims):
        """Decompose matmul into CUTLASS-style sub-ops per K-slice iteration.

        A single thread-block processes a (block_M, block_N) output tile.
        For each K-slice of width block_K, the pipeline stages are:

        - async_copy_load_A/B: load tiles from HBM to shared memory (bypassing L1)
        - shared_load_A/B: move tiles from shared memory to registers
        - mma: matrix multiply-accumulate on Tensor Cores
        - shared_store_C: store partial C to shared memory (epilogue only)
        - global_write_C: write final C back to HBM (epilogue only)

        When async_copy_enabled=False, async_copy_load is replaced by
        global_read (lower throughput, same bytes).
        """
        if dtype is None or hardware is None:
            return []

        M = dims.get("M", 0)
        N = dims.get("N", 0)
        K = dims.get("K", 0)
        bs = dtype.byte_size

        tiling = self.get_tiling_strategy(
            hardware, dtype, pipeline_config=pipeline_config, **dims
        )
        if tiling is None:
            return []

        bM = tiling.block_m
        bN = tiling.block_n
        bK = tiling.block_k

        async_on = True
        if pipeline_config is not None:
            async_on = pipeline_config.async_copy_enabled

        # Determine load stage name based on async copy toggle
        load_stage = "async_copy_load" if async_on else "global_read"
        load_a_name = "async_copy_load_A" if async_on else "global_read_A"
        load_b_name = "async_copy_load_B" if async_on else "global_read_B"

        compute_stage = "fma_alu" if dtype in (DataType.FP32, DataType.FP64) else "mma"
        effective_hbm_a, effective_hbm_b = self._matmul_l2_reuse_bytes(
            hardware, dtype, tiling, M, N
        )

        # Per-iteration recurring sub-ops
        sub_ops = [
            SubOp(
                name=load_a_name,
                pipeline_stage=load_stage,
                read_bytes=bM * bK * bs,
                effective_hbm_read_bytes=effective_hbm_a,
                depends_on=[],
                is_recurring=True,
            ),
            SubOp(
                name=load_b_name,
                pipeline_stage=load_stage,
                read_bytes=bK * bN * bs,
                effective_hbm_read_bytes=effective_hbm_b,
                depends_on=[],
                is_recurring=True,
            ),
            SubOp(
                name="shared_load_A",
                pipeline_stage="shared_load",
                read_bytes=bM * bK * bs,
                depends_on=[load_a_name],
                is_recurring=True,
            ),
            SubOp(
                name="shared_load_B",
                pipeline_stage="shared_load",
                read_bytes=bK * bN * bs,
                depends_on=[load_b_name],
                is_recurring=True,
            ),
            SubOp(
                name=compute_stage,
                pipeline_stage=compute_stage,
                flops=2 * bM * bN * bK,
                depends_on=["shared_load_A", "shared_load_B"],
                is_recurring=True,
            ),
        ]

        # Epilogue sub-ops (one-shot)
        # Architecture-specific epilogue paths:
        #
        # Blackwell (TMA + TMEM):
        #   TMEM → RF (tmem_load) → SMEM (shared_store) → HBM (async_copy_store)
        #   Async MMA writes accumulators to TMEM, freeing register file.
        #   Dedicated Store-TMA engine at 256 B/clk/SM.
        #
        # Hopper (TMA, no TMEM):
        #   RF → SMEM (shared_store) → HBM (async_copy_store)
        #   Accumulator in registers (wgmma).  Unified TMA engine shared
        #   between load and store at 128 B/clk/SM — 2× global_write.
        #
        # Ampere / older (async copy, no TMA):
        #   RF → SMEM (shared_store) → HBM (global_write)
        #   No TMA — epilogue uses traditional L2 write path at 64 B/clk/SM.
        arch = getattr(hardware, "architecture", "").lower()
        if async_on and arch == "blackwell":
            # TMEM accumulator readout + TMA store
            sub_ops += [
                SubOp(
                    name="tmem_load_C",
                    pipeline_stage="tmem_load",
                    read_bytes=bM * bN * bs,
                    depends_on=["mma"],
                    is_recurring=False,
                ),
                SubOp(
                    name="shared_store_C",
                    pipeline_stage="shared_store",
                    write_bytes=bM * bN * bs,
                    depends_on=["tmem_load_C"],
                    is_recurring=False,
                ),
                SubOp(
                    name="async_copy_store_C",
                    pipeline_stage="async_copy_store",
                    write_bytes=bM * bN * bs,
                    effective_hbm_write_bytes=bM * bN * bs,
                    depends_on=["shared_store_C"],
                    is_recurring=False,
                ),
            ]
        elif async_on and arch == "hopper":
            # TMA store (no TMEM — accumulator in registers after wgmma)
            sub_ops += [
                SubOp(
                    name="shared_store_C",
                    pipeline_stage="shared_store",
                    write_bytes=bM * bN * bs,
                    depends_on=["mma"],
                    is_recurring=False,
                ),
                SubOp(
                    name="async_copy_store_C",
                    pipeline_stage="async_copy_store",
                    write_bytes=bM * bN * bs,
                    effective_hbm_write_bytes=bM * bN * bs,
                    depends_on=["shared_store_C"],
                    is_recurring=False,
                ),
            ]
        else:
            sub_ops += [
                SubOp(
                    name="shared_store_C",
                    pipeline_stage="shared_store",
                    write_bytes=bM * bN * bs,
                    depends_on=["mma"],
                    is_recurring=False,
                ),
                SubOp(
                    name="global_write_C",
                    pipeline_stage="global_write",
                    write_bytes=bM * bN * bs,
                    effective_hbm_write_bytes=bM * bN * bs,
                    depends_on=["shared_store_C"],
                    is_recurring=False,
                ),
            ]

        return sub_ops
