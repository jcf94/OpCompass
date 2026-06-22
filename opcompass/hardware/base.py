"""Abstract base class for all hardware targets.

To add a new hardware target, create a file under ``hardware/`` (e.g.
``nvidia_hopper.py``) with a class that subclasses :class:`Hardware` and sets
the attributes described below.  The auto-discovery registry in
``opcompass/registry.py`` will pick it up automatically — no manual
registration is needed.

The reference implementation is :class:`NvidiaAmpere` /
:class:`NvidiaA100` in ``opcompass/hardware/nvidia_ampere.py``.

Required class-level attributes
-------------------------------
``name: str``
    Short internal id used as the lookup key (e.g. ``"a100"``, ``"h100"``).
    Must be unique across all hardware targets.

``vendor: str``
    Manufacturer / vendor name, e.g. ``"NVIDIA"``, ``"AMD"``, ``"Intel"``.

``description: str``
    Human-readable one-line summary shown in CLI and UI listings.
    Include key specs: chip codename, SM/CU count, peak TFLOPS, memory size.

Memory hierarchy (``memory``)
-----------------------------
Set ``memory`` to a :class:`MemoryHierarchy` instance with a ``tiers`` list
of :class:`MemoryTier` entries ordered from *slowest/capacious* (e.g. HBM) to
*fastest/smallest* (e.g. L2).  Each tier needs:

``name: str``
    Tier label — ``"HBM2e"``, ``"L2"``, etc.

``capacity_bytes: int``
    Total capacity in bytes.  Use ``80 * 1024**3`` for 80 GB, etc.

``bandwidth_bytes_per_sec: float``
    Theoretical peak bandwidth in bytes/sec.  Use scientific notation, e.g.
    ``2.0e12`` for 2 TB/s.

``can_overlap_with_compute: set[str]``
    A set of tier names whose data transfers can proceed asynchronously while
    compute units are executing (e.g. DMA engines, async copy).  For NVIDIA
    GPUs this is typically ``{"HBM2e"}`` or ``{"HBM3"}``.  Tiers not listed
    here are assumed to block compute.

Example::

    memory = MemoryHierarchy(
        tiers=[
            MemoryTier(
                name="HBM2e",
                capacity_bytes=80 * 1024**3,
                bandwidth_bytes_per_sec=2.0e12,
            ),
            MemoryTier(
                name="L2",
                capacity_bytes=40 * 1024**2,
                bandwidth_bytes_per_sec=5.0e12,
            ),
        ],
        can_overlap_with_compute={"HBM2e"},
    )

Compute unit (``compute_unit``)
-------------------------------
Set ``compute_unit`` to a :class:`ComputeUnit` instance.  This is the
workhorse of the hardware model.  Every field has a sensible default (0 or
empty), so you only need to fill in what your target supports.

Core fields
~~~~~~~~~~~

``name: str``
    Name of one compute unit, e.g. ``"SM"`` (NVIDIA) or ``"CU"`` (AMD).

``count: int``
    Number of compute units on the full chip (e.g. 108 for A100).

``clock_mhz: float``
    Boost / typical clock frequency in MHz (e.g. ``1410``).

Peak performance
~~~~~~~~~~~~~~~~

``peak_flops: dict[DataType, float]``
    Dictionary mapping each supported :class:`DataType` to its peak
    throughput in FLOPS (or OPS for integer types) on the **full chip**.
    Common keys: ``DataType.FP64``, ``DataType.FP32``, ``DataType.TF32``,
    ``DataType.FP16``, ``DataType.BF16``, ``DataType.INT8``.

    For NVIDIA GPUs: use Tensor Core throughput for FP16/BF16/TF32/INT8 (the
    dominant path in practice) and CUDA-core throughput for FP32/FP64 (since
    tensor cores accelerate those only via TF32/FP64 MMA instructions).

    The formula for per-dtype peak is::

        ops_per_clk_per_unit × count × clock_hz

    Where ``ops_per_clk_per_unit`` is the maximum FMA/ops per clock per SM,
    and the ×2 factor for FMA (2 ops per fused multiply-add) is already
    included.

Pipeline stages
~~~~~~~~~~~~~~~

``pipeline: list[PipelineStage]``
    Ordered list of stages that data flows through on the compute unit.  Each
    :class:`PipelineStage` has:

    ``name: str``
        Stage identifier.  Memory stages should contain ``"read"``,
        ``"load"``, ``"write"``, or ``"store"`` in the name — the pipeline
        model uses these substrings to categorise work units as bytes.
        Compute stages should contain ``"mma"`` or ``"alu"`` — categorised as
        FMA/ALU operations.

    ``latency_cycles: int``
        Fixed latency in clock cycles for one invocation (pipeline depth,
        first-element delay).

    ``throughput_per_cycle: float``
        Peak throughput in *work units per clock per compute unit*.  Work
        units are:

        - **bytes** for memory stages (e.g. 64 B/clk/SM for L1 bandwidth)
        - **FMA operations** for compute stages (e.g. 1024 FMA/clk/SM for
          A100 FP16 Tensor Core)

    ``description: str``
        Human-readable explanation of the stage.

    A typical pipeline looks like::

        global_read → shared_load → mma → shared_store → global_write

    With optional async copy stages (e.g. ``async_copy_load`` for Ampere+).
    See :class:`NvidiaAmpere` / :class:`NvidiaA100` for a complete example
    with detailed comments.

Per-unit memory resources
~~~~~~~~~~~~~~~~~~~~~~~~~

``register_file_kb: int``
    Register file size per compute unit in KB (e.g. 256 for A100 SM).

``shared_memory_max_kb: int``
    Maximum configurable shared memory per unit in KB (e.g. 164 for A100).
    Some architectures allow trading L1 cache for shared memory.

``l1_shared_combined_kb: int``
    Total combined L1 + shared memory pool per unit in KB.  Set this when
    L1 and shared memory share a single SRAM pool (common on NVIDIA GPUs).

Per-unit execution resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``warp_schedulers_per_unit: int``
    Number of warp schedulers / wavefront issuers per compute unit.

``tensor_cores_per_unit: int``
    Number of tensor/matrix cores per compute unit (e.g. 4 for A100).

``fp32_cores_per_unit: int``
    Number of FP32 (CUDA) cores per unit (e.g. 64 for A100).

``fp64_cores_per_unit: int``
    Number of FP64 cores per unit (often half of FP32, e.g. 32 for A100).

``int32_cores_per_unit: int``
    Number of INT32 cores per unit.  On modern NVIDIA GPUs these are
    dedicated datapaths that can issue concurrently with FP32.

``ldst_units: int``
    Number of load/store units (LSUs) per compute unit.

``sfu_units: int``
    Number of special function units (transcendentals, etc.).

Threading / occupancy limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``max_concurrent_warps: int``
    Maximum resident warps/wavefronts per compute unit (e.g. 64 for A100).

``max_threads_per_unit: int``
    Maximum resident threads per compute unit (e.g. 2048 for A100).

``max_thread_blocks_per_unit: int``
    Maximum resident thread blocks / workgroups per compute unit
    (e.g. 32 for A100).

``max_registers_per_thread: int``
    Maximum registers allocatable per thread (e.g. 255 for A100).

``max_registers_per_block: int``
    Maximum registers allocatable per thread block (e.g. 65536 for A100).
    This limits occupancy when combined with ``register_file_kb`` — if a
    block uses many registers, fewer blocks can be resident.

Parallel / concurrent execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``can_concurrent_fp32_int32: bool``
    Whether FP32 and INT32 instructions can issue simultaneously on the
    same compute unit (separate datapaths).  True for Volta+ NVIDIA GPUs.

``threads_per_warp: int``
    Threads per warp / wavefront.  Typically 32 for NVIDIA, 64 for AMD.

Architecture-family pattern
----------------------------
When multiple GPU SKUs share the same microarchitecture (e.g. A100, A40,
A10 are all Ampere), it is recommended to introduce an **intermediate base
class** that captures the common SM-level parameters.  This avoids
duplication and makes it trivial to add new SKUs later.

The pattern used in ``nvidia_ampere.py`` is::

    class NvidiaAmpere(Hardware):
        \"\"\"Common Ampere SM microarchitecture — NOT registered as a target.\"\"\"

        # Leave ``name`` empty so the registry skips this class
        vendor = \"NVIDIA\"
        architecture = \"Ampere\"
        sm_version = \"8.0\"

        # Per-SM resources, execution units, threading limits, and
        # capabilities that are identical across all Ampere SKUs …
        register_file_kb = 256
        tensor_cores_per_unit = 4
        # … etc.

        @classmethod
        def _make_compute_unit(cls, count, clock_mhz, peak_flops, **overrides):
            \"\"\"Factory that returns a ComputeUnit pre-filled with the
            Ampere SM defaults.  Subclasses call this with their SKU's
            SM count, clock, and peak FLOPs.\"\"\"
            return ComputeUnit(
                name=\"SM\",
                count=count,
                clock_mhz=clock_mhz,
                peak_flops=peak_flops,
                pipeline=cls._pipeline(),
                register_file_kb=cls.register_file_kb,
                # … all common fields …
                **overrides,
            )

    class NvidiaA100(NvidiaAmpere):
        \"\"\"A100 SKU — only the chip-specific parts.\"\"\"
        name = \"a100\"
        description = \"NVIDIA A100 80GB SXM4 — Ampere, 108 SM, 312 TFLOPS FP16\"

        memory = MemoryHierarchy(
            tiers=[
                MemoryTier(name=\"HBM2e\", capacity_bytes=80*1024**3,
                           bandwidth_bytes_per_sec=2.0e12),
                MemoryTier(name=\"L2\", capacity_bytes=40*1024**2,
                           bandwidth_bytes_per_sec=5.0e12),
            ],
            can_overlap_with_compute={\"HBM2e\"},
        )

        compute_unit = NvidiaAmpere._make_compute_unit(
            count=108,
            clock_mhz=1410,
            peak_flops={DataType.FP16: 312e12, DataType.FP32: 19.5e12, …},
        )

Key points about the intermediate base class:

* Leave ``name`` as the empty string (inherited from :class:`Hardware`).
  The auto-discovery registry skips classes with ``name == ""``, so the
  base class is never exposed as a selectable target.
* ``vendor`` can be set once on the base class — all subclasses inherit it.
* Store architecture metadata (``architecture``, ``sm_version``) on the
  base class so tooling can query it.
* Provide a ``@classmethod`` factory for :class:`ComputeUnit` that fills in
  the common per-SM defaults.  Subclasses then only pass the SKU-specific
  ``count``, ``clock_mhz``, and ``peak_flops``.  Use ``**overrides`` to
  handle rare deviations (e.g. reduced FP64 cores on GA102).
* The common pipeline stages should also live on the base class via a
  ``@classmethod`` (e.g. ``_pipeline()`` or ``_ampere_pipeline()``) so
  every SKU in the family shares the same stage definitions.
* Per-SM resource constants and threading limits are plain class attributes
  so subclasses can reference or override them individually if needed.

When the new SKU deviates substantially — different memory type (HBM vs
GDDR), different pipeline, different SM resource counts — simply override
the relevant class attribute on the SKU subclass, or pass the override
through ``**overrides`` to the factory.

Quick-start checklist
---------------------
When adding a new hardware target, work through this list:

**If this is the first GPU in a new architecture family:**

1. Create a new file under ``hardware/`` (e.g. ``nvidia_hopper.py``).
2. Add an intermediate base class (e.g. ``NvidiaHopper(Hardware)``) with:
   - ``vendor``, ``architecture``, ``sm_version``
   - Per-SM resource constants and threading limits
   - A ``@classmethod`` that returns the common pipeline stages
   - A ``@classmethod`` factory for ``ComputeUnit``
   - ``name`` left empty (so the registry skips it)
3. Add one or more concrete SKU subclasses, each specifying:
   - ``name``, ``description``
   - ``memory`` (capacity, bandwidth, tiers vary by SKU)
   - ``compute_unit`` via the factory (SM count, clock, peak FLOPs)

**If adding a new SKU to an existing architecture family:**

1. Subclass the existing base class (e.g. ``class NvidiaA40(NvidiaAmpere)``).
2. Set ``name``, ``description``.
3. Set ``memory`` with the SKU's memory config.
4. Call the base class's ``_make_compute_unit`` with the SKU's SM count,
   clock, and peak FLOPs.  Pass any deviations via ``**overrides``.

**If adding a one-off GPU with no family (standalone):**

1. Subclass :class:`Hardware` directly.
2. Set ``name``, ``vendor``, ``description``.
3. ``memory`` → :class:`MemoryHierarchy` with at least one
   :class:`MemoryTier` (HBM).  Add L2 if data is available.  Tag tiers
   that support async copy in ``can_overlap_with_compute``.
4. ``compute_unit`` → :class:`ComputeUnit` with ``name``, ``count``,
   ``clock_mhz``.
5. **Peak FLOPS**: ``peak_flops`` dict covering all supported dtypes.
6. **Pipeline** (optional but recommended): ``pipeline`` list of
   :class:`PipelineStage` from global memory reads through compute to
   write-back.
7. **SM resources**: ``register_file_kb``, ``shared_memory_max_kb``,
   ``tensor_cores_per_unit``, ``fp32_cores_per_unit``, etc.
8. **Occupancy**: ``max_concurrent_warps``, ``max_threads_per_unit``,
   ``max_thread_blocks_per_unit``, register limits.
9. **Capabilities**: ``can_concurrent_fp32_int32``, ``threads_per_warp``.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opcompass.models import (
        ComputeUnit,
        DataType,
        MemoryHierarchy,
    )


class Hardware(ABC):
    """Abstract hardware target.  Each concrete hardware lives in its own
    file under ``hardware/``.

    Subclasses must set the following class-level attributes:

        name: str           — Short id, e.g. "a100"
        vendor: str         — "NVIDIA", "AMD", ...
        description: str    — Human-readable summary

    And must override:

        memory: MemoryHierarchy
        compute_unit: ComputeUnit

    Two patterns are supported:

    * **Direct subclass** — for one-off GPUs with no family.  Set every
      field directly on the :class:`ComputeUnit`.
    * **Architecture-family** — introduce an intermediate base class
      (e.g. :class:`NvidiaAmpere`) that holds the common SM
      microarchitecture, pipeline stages, and a factory method.  Concrete
      SKUs then only supply chip-specific values (SM count, clock, peak
      FLOPs, memory config).  Leave ``name`` empty on the intermediate
      base class to prevent it from being registered.

    See the module-level docstring for a detailed guide and the
    :class:`NvidiaAmpere` / :class:`NvidiaA100` pair in
    ``opcompass/hardware/nvidia_ampere.py`` for the reference implementation.
    """

    name: str = ""
    vendor: str = ""
    description: str = ""

    # Subclasses override these with actual objects
    memory: MemoryHierarchy
    compute_unit: ComputeUnit

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_peak_flops(self, dtype: DataType) -> float:
        """Peak FLOPS for *dtype* on the full chip."""
        return self.compute_unit.peak_flops.get(dtype, 0.0)

    def get_bandwidth(self, tier_name: str) -> float:
        """Bandwidth in bytes/sec for the named memory tier."""
        for t in self.memory.tiers:
            if t.name.lower() == tier_name.lower():
                return t.bandwidth_bytes_per_sec
        return 0.0

    @property
    def hbm_bandwidth(self) -> float:
        """Convenience: bandwidth of the first (slowest) memory tier."""
        if self.memory.tiers:
            return self.memory.tiers[0].bandwidth_bytes_per_sec
        return 0.0

    @property
    def clock_ghz(self) -> float:
        """Clock frequency in GHz."""
        return self.compute_unit.clock_mhz / 1000.0

    @property
    def num_compute_units(self) -> int:
        """Number of compute units (SMs / CUs)."""
        return self.compute_unit.count
