"""Explicit pipeline intermediate representation and deterministic scheduler.

The IR deliberately knows nothing about operators or GPU stage names.  A model
declares dependencies and resource demands; the scheduler only interprets
those declarations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import heapq
from typing import Iterable


class PipelineIRValidationError(ValueError):
    """An invalid pipeline graph, resource, or buffer declaration."""

    code = "invalid_pipeline_ir"

    def __init__(self, issues: Iterable[str]):
        self.issues = tuple(issues)
        super().__init__("; ".join(self.issues))


class ResourceKind(str, Enum):
    HBM = "hbm"
    L2 = "l2"
    COPY = "copy"
    SHARED = "shared"
    REGISTER = "register"
    TMEM = "tmem"
    COMPUTE = "compute"
    SYNCHRONIZATION = "synchronization"
    STORE = "store"
    GENERIC = "generic"


class SyncPrimitive(str, Enum):
    SYNCTHREADS = "syncthreads"
    CP_ASYNC_COMMIT = "cp_async_commit"
    CP_ASYNC_WAIT = "cp_async_wait"
    MBARRIER = "mbarrier"
    WGMMA_COMMIT = "wgmma_commit"
    WGMMA_WAIT = "wgmma_wait"
    UMMA_COMMIT = "umma_commit"
    UMMA_WAIT = "umma_wait"


@dataclass(frozen=True)
class Resource:
    name: str
    capacity: int = 1
    queue_capacity: int | None = None
    kind: ResourceKind = ResourceKind.GENERIC


@dataclass(frozen=True)
class Work:
    amount: float
    unit: str


@dataclass(frozen=True)
class ResourceDemand:
    resource: str
    units: int = 1


@dataclass(frozen=True)
class Buffer:
    name: str
    size_bytes: int
    slots: int = 1


@dataclass(frozen=True)
class BufferAccess:
    buffer: str
    mode: str  # read | write
    slot_offset: int = 0


@dataclass(frozen=True)
class MemoryAccess:
    path: tuple[ResourceKind, ...]
    bytes: int
    transaction_bytes: int
    transactions: int
    reuse_policy: str = "none"


@dataclass(frozen=True)
class Node:
    name: str
    duration_cycles: int
    work: Work
    demands: tuple[ResourceDemand, ...]
    accesses: tuple[BufferAccess, ...] = ()
    memory_access: MemoryAccess | None = None
    synchronization: SyncPrimitive | None = None


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    latency_cycles: int = 0
    iteration_distance: int = 0


@dataclass(frozen=True)
class Loop:
    iterations: int = 1
    initiation_interval: int = 0


@dataclass(frozen=True)
class Launch:
    grid_size: int = 1
    resident_blocks: int = 1
    compute_units: int = 1
    launch_overhead_cycles: int = 0
    tail_fraction: float = 1.0


@dataclass(frozen=True)
class PipelineProgram:
    resources: tuple[Resource, ...]
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...] = ()
    buffers: tuple[Buffer, ...] = ()
    loop: Loop = Loop()
    launch: Launch = Launch()


@dataclass(frozen=True)
class ScheduledNode:
    node: str
    iteration: int
    start_cycle: int
    end_cycle: int
    resources: tuple[ResourceDemand, ...]


@dataclass(frozen=True)
class CompactSchedule:
    entries: tuple[ScheduledNode, ...]
    total_cycles: int
    resource_busy_cycles: dict[str, int]
    loop_iterations: int
    wave_count: int = 1
    underfilled: bool = False
    tail_fraction: float = 1.0
    launch_overhead_cycles: int = 0

    def trace_window(self, start_cycle: int, end_cycle: int) -> tuple[ScheduledNode, ...]:
        """Reconstruct the entries intersecting a half-open cycle window."""
        if start_cycle < 0 or end_cycle < start_cycle:
            raise ValueError("trace window must satisfy 0 <= start <= end")
        return tuple(
            entry for entry in self.entries
            if entry.start_cycle < end_cycle and entry.end_cycle > start_cycle
        )


def validate_program(program: PipelineProgram) -> None:
    """Validate graph references, units, capacities, cycles, and ring reuse."""
    issues: list[str] = []
    resource_map = {item.name: item for item in program.resources}
    node_map = {item.name: item for item in program.nodes}
    buffer_map = {item.name: item for item in program.buffers}

    if len(resource_map) != len(program.resources):
        issues.append("resource names must be unique")
    if len(node_map) != len(program.nodes):
        issues.append("node names must be unique")
    if len(buffer_map) != len(program.buffers):
        issues.append("buffer names must be unique")
    if program.loop.iterations < 1:
        issues.append("loop iterations must be positive")
    if program.loop.initiation_interval < 0:
        issues.append("loop initiation interval cannot be negative")
    if (program.launch.grid_size < 1 or program.launch.resident_blocks < 1
            or program.launch.compute_units < 1):
        issues.append("launch values must be positive")
    if program.launch.launch_overhead_cycles < 0:
        issues.append("launch overhead cannot be negative")
    if not 0 < program.launch.tail_fraction <= 1:
        issues.append("launch tail fraction must be in (0, 1]")

    valid_units = {"bytes", "flops", "fma", "operations", "cycles"}
    for resource in program.resources:
        if resource.capacity < 1:
            issues.append(f"resource '{resource.name}' capacity must be positive")
        if resource.queue_capacity is not None and resource.queue_capacity < 1:
            issues.append(f"resource '{resource.name}' queue capacity must be positive")
    for buffer in program.buffers:
        if buffer.size_bytes < 1 or buffer.slots < 1:
            issues.append(f"buffer '{buffer.name}' size and slots must be positive")
    for node in program.nodes:
        if node.duration_cycles < 0:
            issues.append(f"node '{node.name}' duration cannot be negative")
        if node.work.amount < 0 or node.work.unit not in valid_units:
            issues.append(f"node '{node.name}' has invalid work unit or amount")
        if not node.demands:
            issues.append(f"node '{node.name}' must demand at least one resource")
        for demand in node.demands:
            resource = resource_map.get(demand.resource)
            if resource is None:
                issues.append(f"node '{node.name}' references missing resource '{demand.resource}'")
            elif demand.units < 1 or demand.units > resource.capacity:
                issues.append(f"node '{node.name}' exceeds resource '{demand.resource}' capacity")
        for access in node.accesses:
            buffer = buffer_map.get(access.buffer)
            if buffer is None:
                issues.append(f"node '{node.name}' references missing buffer '{access.buffer}'")
            elif access.mode not in {"read", "write"}:
                issues.append(f"node '{node.name}' has invalid buffer access mode '{access.mode}'")
            elif abs(access.slot_offset) >= buffer.slots:
                issues.append(f"node '{node.name}' buffer slot offset exceeds ring size")
        access = node.memory_access
        if access is not None:
            if not access.path or access.bytes < 0 or access.transaction_bytes < 1:
                issues.append(f"node '{node.name}' has an invalid memory access")
            elif access.transactions * access.transaction_bytes < access.bytes:
                issues.append(f"node '{node.name}' memory transactions do not cover its bytes")
            if access.reuse_policy not in {"none", "cta_order", "resident", "streaming"}:
                issues.append(f"node '{node.name}' has an invalid reuse policy")

    for edge in program.edges:
        if edge.source not in node_map or edge.target not in node_map:
            issues.append(f"edge '{edge.source}->{edge.target}' references a missing node")
        if edge.latency_cycles < 0 or edge.iteration_distance < 0:
            issues.append(f"edge '{edge.source}->{edge.target}' has a negative value")

    # Only distance-zero edges participate in the per-iteration DAG.
    indegree = {name: 0 for name in node_map}
    outgoing = {name: [] for name in node_map}
    for edge in program.edges:
        if edge.iteration_distance == 0 and edge.source in node_map and edge.target in node_map:
            indegree[edge.target] += 1
            outgoing[edge.source].append(edge.target)
    ready = [name for name, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        name = ready.pop()
        visited += 1
        for target in outgoing[name]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(node_map):
        issues.append("dependency graph contains a cycle")

    # A writer must not wrap its ring before all explicitly linked readers.
    for buffer in program.buffers:
        writers = {n.name for n in program.nodes for a in n.accesses if a.buffer == buffer.name and a.mode == "write"}
        readers = {n.name for n in program.nodes for a in n.accesses if a.buffer == buffer.name and a.mode == "read"}
        for writer in writers:
            for reader in readers:
                distances = [e.iteration_distance for e in program.edges if e.source == writer and e.target == reader]
                if distances and max(distances) >= buffer.slots:
                    issues.append(f"buffer '{buffer.name}' ring is too small for declared lifetime")

    if issues:
        raise PipelineIRValidationError(issues)


def schedule(program: PipelineProgram) -> CompactSchedule:
    """Expand loops and deterministically list-schedule the explicit DAG."""
    validate_program(program)
    resources = {resource.name: resource for resource in program.resources}
    nodes = {node.name: node for node in program.nodes}
    instances = [(iteration, node.name) for iteration in range(program.loop.iterations) for node in program.nodes]
    predecessors: dict[tuple[int, str], list[tuple[tuple[int, str], int]]] = {key: [] for key in instances}
    successors: dict[tuple[int, str], list[tuple[int, str]]] = {key: [] for key in instances}
    for iteration, _ in instances:
        for edge in program.edges:
            source_iteration = iteration - edge.iteration_distance
            if source_iteration >= 0:
                source = (source_iteration, edge.source)
                target = (iteration, edge.target)
                predecessors[target].append((source, edge.latency_cycles))
                successors[source].append(target)

    remaining = {key: len(value) for key, value in predecessors.items()}
    ready: list[tuple[int, str]] = [key for key, count in remaining.items() if count == 0]
    heapq.heapify(ready)
    completed: dict[tuple[int, str], int] = {}
    calendars: dict[str, list[tuple[int, int, int]]] = {name: [] for name in resources}
    entries: list[ScheduledNode] = []

    def resources_fit(node: Node, start: int) -> bool:
        end = start + node.duration_cycles
        for demand in node.demands:
            overlapping = [
                (left, right, units) for left, right, units in calendars[demand.resource]
                if left < end and right > start
            ]
            used = sum(units for _, _, units in overlapping)
            if used + demand.units > resources[demand.resource].capacity:
                return False
            queue_capacity = resources[demand.resource].queue_capacity
            if queue_capacity is not None and len(overlapping) >= queue_capacity:
                return False
        return True

    while ready:
        iteration, name = heapq.heappop(ready)
        node = nodes[name]
        earliest = iteration * program.loop.initiation_interval
        for predecessor, latency in predecessors[(iteration, name)]:
            earliest = max(earliest, completed[predecessor] + latency)
        start = earliest
        while not resources_fit(node, start):
            boundaries = [right for demand in node.demands for left, right, _ in calendars[demand.resource] if left <= start < right]
            start = min(boundaries) if boundaries else start + 1
        end = start + node.duration_cycles
        for demand in node.demands:
            calendars[demand.resource].append((start, end, demand.units))
        entry = ScheduledNode(name, iteration, start, end, node.demands)
        entries.append(entry)
        completed[(iteration, name)] = end
        for target in successors[(iteration, name)]:
            remaining[target] -= 1
            if remaining[target] == 0:
                heapq.heappush(ready, target)

    if len(entries) != len(instances):
        raise PipelineIRValidationError(["expanded dependency graph contains a loop-carried cycle"])
    entries.sort(key=lambda item: (item.start_cycle, item.iteration, item.node))
    busy = {
        name: sum((right - left) * units for left, right, units in calendar)
        for name, calendar in calendars.items()
    }
    slots = program.launch.compute_units * program.launch.resident_blocks
    waves = (program.launch.grid_size + slots - 1) // slots
    return CompactSchedule(
        entries=tuple(entries),
        total_cycles=max((entry.end_cycle for entry in entries), default=0) * waves
        + program.launch.launch_overhead_cycles,
        resource_busy_cycles=busy,
        loop_iterations=program.loop.iterations,
        wave_count=waves,
        underfilled=program.launch.grid_size < slots,
        tail_fraction=program.launch.tail_fraction,
        launch_overhead_cycles=program.launch.launch_overhead_cycles,
    )
