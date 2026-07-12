"""Versioned, lossless measurement records and generic importers."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable


@dataclass(frozen=True)
class MeasurementRecord:
    record_id: str
    operator: str
    hardware: str
    dtype: str
    shapes: dict[str, int]
    runtimes_us: tuple[float, ...]
    kernel: str
    model_version: str
    environment: dict[str, str] = field(default_factory=dict)
    clocks_mhz: dict[str, float] = field(default_factory=dict)
    candidate: dict[str, Any] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=dict)
    raw_source: str = ""
    schema_version: str = "0.5.0"

    def __post_init__(self):
        if not self.record_id or not self.runtimes_us or any(value <= 0 for value in self.runtimes_us):
            raise ValueError("measurement id and positive runtime samples are required")
        if not self.shapes or any(value <= 0 for value in self.shapes.values()):
            raise ValueError("measurement shapes must be concrete and positive")

    @property
    def median_runtime_us(self) -> float:
        return median(self.runtimes_us)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["runtimes_us"] = list(self.runtimes_us)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MeasurementRecord":
        value = dict(value)
        value["runtimes_us"] = tuple(float(item) for item in value["runtimes_us"])
        value["shapes"] = {key: int(item) for key, item in value["shapes"].items()}
        return cls(**value)


def save_measurements(records: Iterable[MeasurementRecord], path: str | Path) -> None:
    Path(path).write_text(json.dumps([record.to_dict() for record in records], indent=2, sort_keys=True) + "\n")


def load_measurements(path: str | Path) -> list[MeasurementRecord]:
    return [MeasurementRecord.from_dict(item) for item in json.loads(Path(path).read_text())]


def import_measurements(path: str | Path, format: str = "auto") -> list[MeasurementRecord]:
    """Import canonical JSON, generic CSV, or CUTLASS Profiler CSV."""
    path = Path(path)
    selected = path.suffix.lower().lstrip(".") if format == "auto" else format.lower()
    if selected == "json":
        return load_measurements(path)
    if selected not in {"csv", "cutlass"}:
        raise ValueError(f"unsupported measurement format: {selected}")
    records = []
    with path.open(newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream), 1):
            # CUTLASS uses Runtime while the generic format uses runtime_us.
            runtime = row.get("runtime_us") or row.get("Runtime")
            shapes = {key: int(row[key]) for key in ("M", "N", "K") if row.get(key)}
            records.append(MeasurementRecord(
                record_id=row.get("record_id") or f"{path.stem}-{index}",
                operator=row.get("operator") or "matmul",
                hardware=row.get("hardware") or row.get("Device") or "unknown",
                dtype=(row.get("dtype") or row.get("A") or "unknown").lower(),
                shapes=shapes,
                runtimes_us=(float(runtime),),
                kernel=row.get("kernel") or row.get("Operation") or "unknown",
                model_version=row.get("model_version") or "unlinked",
                raw_source=str(path),
            ))
    return records
